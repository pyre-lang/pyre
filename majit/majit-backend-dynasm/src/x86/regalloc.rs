//! Port of `rpython/jit/backend/x86/regalloc.py` — the register
//! configuration, the `oplist` dispatch and the `consider_*` family.
//!
//! Methods that read/write the shared `RegAlloc` state are declared
//! as a second `impl` block on `crate::regalloc::RegAlloc<'a>`. The
//! `x86` module is `#[cfg(target_arch = "x86_64")]` gated in
//! `lib.rs`, so these methods only compile on x86_64.

use crate::j2plan::{GuardKind, IntBinKind, IntUnaryKind, LirOp, LoadKind, StoreKind};
use crate::regalloc::{
    FrameManager, Lifetime, RegAlloc, RegAllocAssembler, SAVE_ALL_REGS, SAVE_DEFAULT_REGS,
    SAVE_GCREF_REGS, compute_vars_longevity, fits_in_32bits, is_math_sqrt_call,
};
use crate::regloc::X86_64_SCRATCH_REG;
use crate::regloc::{
    EAX, EBP, EBX, ECX, EDI, EDX, ESI, R8, R9, R10, R12, R13, R14, R15, RegLoc, XMM0, XMM1, XMM2,
    XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14,
};
use crate::regloc::{Loc, X86_64_XMM_SCRATCH_REG};
use crate::x86::reghint;
use majit_ir::{Op, OpRc, OpRef, Type};
use majit_ir::{OpCode, OpTypeIndex, descr_identity};

/// x86/regalloc.py X86_64_RegisterManager.all_regs — the GPR allocation
/// pool.  Order chosen to prefer caller-save first (popped from end).
/// On Win64, `all_regs.remove(r13)` runs at class-construction time, so
/// the pool length drops 13 → 12 and `all_reg_indexes` (which is built
/// from the post-removal `all_regs`) shifts R14/R15 down by one slot.
#[cfg(not(target_os = "windows"))]
pub const ALL_CORE_REGS: &[RegLoc] = &[
    ECX, EAX, EDX, EBX, ESI, EDI, R8, R9, R10, R12, R13, R14, R15,
];
#[cfg(target_os = "windows")]
pub const ALL_CORE_REGS: &[RegLoc] = &[ECX, EAX, EDX, EBX, ESI, EDI, R8, R9, R10, R12, R14, R15];

pub fn all_core_regs() -> Vec<RegLoc> {
    ALL_CORE_REGS.to_vec()
}

/// x86/regalloc.py: caller-save GPR list (registers spilled around
/// calls per System V AMD64 / Win64 ABI).
#[cfg(not(target_os = "windows"))]
pub const SAVE_AROUND_CALL_CORE_REGS: &[RegLoc] = &[EAX, ECX, EDX, ESI, EDI, R8, R9, R10];
#[cfg(target_os = "windows")]
pub const SAVE_AROUND_CALL_CORE_REGS: &[RegLoc] = &[EAX, ECX, EDX, R8, R9, R10];

pub fn save_around_call_core_regs() -> Vec<RegLoc> {
    SAVE_AROUND_CALL_CORE_REGS.to_vec()
}

/// x86/regalloc.py X86_64_XMMRegisterManager.all_regs — XMM allocation
/// pool.  On non-Win64 xmm15 is reserved as scratch.  On Win64 PyPy
/// uses a separate `X86_64_WIN_XMMRegisterManager` (regalloc.py)
/// with only `[xmm0..xmm4]`, reserving xmm5 as scratch and leaving
/// xmm6..xmm15 callee-save untouched so the JIT prologue/epilogue
/// does not need to save them.  `save_around_call_regs = all_regs`
/// for XMMs across both ABIs.
#[cfg(not(target_os = "windows"))]
pub const ALL_FLOAT_REGS: &[RegLoc] = &[
    XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14,
];
#[cfg(target_os = "windows")]
pub const ALL_FLOAT_REGS: &[RegLoc] = &[XMM0, XMM1, XMM2, XMM3, XMM4];

pub fn all_float_regs() -> Vec<RegLoc> {
    ALL_FLOAT_REGS.to_vec()
}

/// `frame_reg` on x86_64 is RBP (callee-save), holding the JitFrame
/// pointer for the duration of the JIT-compiled procedure.
pub fn frame_reg() -> RegLoc {
    EBP
}

/// `call_result_gpr` — x86_64 AMD64 ABI return register.
pub fn call_result_gpr() -> RegLoc {
    EAX
}

/// x86/assembler.py `cond_call_register_arguments` —
/// `CallBuilder64.ARGUMENTS_GPR[:4]`. `CondCallSlowPath.generate_body`
/// remaps extra COND_CALL / COND_CALL_VALUE arguments into these
/// registers on the taken path only.
pub fn cond_call_argument_regs() -> &'static [RegLoc] {
    let regs = crate::x86::callbuilder::ARGUMENTS_GPR;
    &regs[..4]
}

/// `call_result_fpr` — x86_64 AMD64 ABI XMM return register.
pub fn call_result_fpr() -> RegLoc {
    XMM0
}

/// `core_reg_index` returns the canonical jitframe slot for `reg`.
///
/// regalloc.py `all_reg_indexes` is built from the post-Win64-removal
/// `all_regs` list: on Win64 `all_regs.remove(r13)` runs first, so
/// the per-reg index table records R14 at slot 10 and R15 at slot 11
/// (instead of 11/12 on non-Win64).  Mirror that here by looking up
/// the position in `ALL_CORE_REGS`, which is itself Win64-aware.
///
/// `_push_all_regs_to_frame`, `_pop_all_regs_from_frame`, `get_gcmap`
/// must all consume slots through this function — never via iteration
/// position inside `all_core_regs()` — so save_regs_label, the gcmap
/// bitmap, and the post-call pop stay in agreement (they happen to be
/// equal once both sides walk the same Win64-aware list, but keeping
/// the lookup central guards against a future helper computing slots
/// off a non-canonical iteration).
pub fn core_reg_index(reg: RegLoc) -> Option<usize> {
    ALL_CORE_REGS.iter().position(|candidate| *candidate == reg)
}

/// x86/regalloc.py consider_call_malloc_nursery:
///   `spill_or_move_registers_before_call([ecx, edx])`
///   `force_allocate_reg(op, selected_reg=ecx)`        → result
///   `force_allocate_reg(tmp_box, selected_reg=edx)`   → temp
/// reghint.py consider_call_malloc_nursery:
///   `longevity.fixed_register(position, ecx, op)`
///   `longevity.fixed_register(position, edx)`
pub const MALLOC_NURSERY_CLOBBER: [RegLoc; 2] = [ECX, EDX];

/// x86_64: result register after the nursery bump (ecx per
/// regalloc.py:1021).
pub const MALLOC_NURSERY_RESULT: RegLoc = ECX;

fn is_threadlocalref_get_call(op: &Op) -> bool {
    op.with_call_descr(|cd| cd.get_extra_info().oopspecindex)
        == Some(majit_ir::OopSpecIndex::ThreadlocalrefGet)
}

/// `consider_int_*` family.
///
/// x86's two-operand encoding (`OP dst, src` with dst = first
/// operand) means the result register is forced onto the lhs slot
/// via `force_result_in_reg`.  RPython parity:
/// `rpython/jit/backend/x86/regalloc.py _consider_binop_part`
/// and `:566 consider_int_add` (plus `_consider_lea` for the
/// `add reg, imm32` LEA shortcut).
impl<'a> RegAlloc<'a> {
    /// x86/regalloc.py `_consider_binop_part`. Sets up `dst =
    /// lhs` register coupling; for symmetric ops, swaps `lhs`/`rhs`
    /// when that lets us avoid a spill.
    fn _consider_binop_part(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        symm: bool,
    ) -> (Loc, Loc) {
        let mut x = lhs;
        let mut y = rhs;
        let xloc = self.loc(x, self.tp(x));
        let mut argloc = self.loc(y, self.tp(y));

        if symm && !xloc.is_reg() && argloc.is_reg() {
            let x_lives_longer = !self.longevity.contains(x)
                || self.longevity.get(x).unwrap().last_usage > self.rm.position;
            let y_dies = self
                .longevity
                .get(y)
                .map(|lt| lt.last_usage == self.rm.position)
                .unwrap_or(false);
            if x_lives_longer && y_dies {
                std::mem::swap(&mut x, &mut y);
                argloc = self.loc(y, self.tp(y));
            }
        }

        let tp = self.tp(x);
        let args = [lhs, rhs];
        let loc = self.rm.force_result_in_reg(
            dst,
            x,
            tp,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        (loc, argloc)
    }

    /// x86/regalloc.py `_consider_binop` — asymmetric binop.
    pub(crate) fn _consider_binop(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let (loc, argloc) = self._consider_binop_part(dst, lhs, rhs, false);
        self.perform(i, [loc, argloc], Some(loc), asm);
    }

    /// x86/regalloc.py `_consider_binop_symm` — symmetric binop
    /// with the swap heuristic.
    pub(crate) fn _consider_binop_symm(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let (loc, argloc) = self._consider_binop_part(dst, lhs, rhs, true);
        self.perform(i, [loc, argloc], Some(loc), asm);
    }

    /// x86/regalloc.py `_consider_lea` — emits `LEA dst, [lhs +
    /// rhs]` so the result register can differ from `lhs`. Limited
    /// to RHS constants that fit a 32-bit displacement.
    fn _consider_lea(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let loc = self.make_sure_var_in_reg(lhs, self.tp(lhs), &[], None, false);
        self.possibly_free_var(lhs, self.tp(lhs));
        let argloc = self.loc(rhs, self.tp(rhs));
        let resloc = Loc::Reg(self.force_allocate_reg(dst, Type::Int, &[], None, false));
        self.perform(i, [loc, argloc], Some(resloc), asm);
    }

    /// x86/regalloc.py `consider_int_add` — LEA shortcut when
    /// the RHS is a 32-bit-fitting immediate, otherwise symmetric
    /// 2-operand `ADD dst, src`.
    pub(crate) fn consider_int_add(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        if rhs.is_constant() {
            let val = self.const_value(rhs);
            if fits_in_32bits(val) {
                return self._consider_lea(dst, lhs, rhs, i, asm);
            }
        }
        self._consider_binop_symm(dst, lhs, rhs, i, asm);
    }

    /// x86/regalloc.py `consider_int_sub` — LEA shortcut with
    /// negated immediate (`LEA dst, [lhs - imm]`).
    pub(crate) fn consider_int_sub(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        if rhs.is_constant() {
            let val = self.const_value(rhs);
            // x86/regalloc.py `rx86.fits_in_32bits(-y.value)`. The
            // negate-then-range-check filters `i64::MIN` naturally
            // because `wrapping_neg(i64::MIN) == i64::MIN`, which sits
            // outside the i32 range, so the LEA shortcut is skipped
            // and the regular SUB path is taken.
            if fits_in_32bits(val.wrapping_neg()) {
                return self._consider_lea(dst, lhs, rhs, i, asm);
            }
        }
        self._consider_binop(dst, lhs, rhs, i, asm);
    }

    /// x86/regalloc.py `consider_int_lshift` — shift count must
    /// be in ECX (`SHL/SHR/SAR reg, CL`) unless it's an immediate.
    pub(crate) fn consider_int_lshift(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let loc2 = if rhs.is_constant() {
            self.rm.convert_to_imm(rhs, &self.constants)
        } else {
            self.make_sure_var_in_reg(rhs, Type::Int, &[], Some(ECX), false)
        };
        let args = [lhs, rhs];
        let loc1 = self.rm.force_result_in_reg(
            dst,
            lhs,
            Type::Int,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, [loc1, loc2], Some(loc1), asm);
    }

    /// x86/regalloc.py `consider_int_neg` / `consider_int_invert`
    /// — `NEG dst` and `NOT dst` overwrite their argument register.
    pub(crate) fn consider_int_neg(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let loc = self.rm.force_result_in_reg(
            dst,
            arg,
            Type::Int,
            &[],
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, [loc], Some(loc), asm);
    }

    /// x86/regalloc.py `consider_zero_array`. A constant length up to
    /// `16 * 8` bytes is unrolled by `genop_discard_zero_array`
    /// (`[base_loc, startindex_loc, imm(constbytes), imm(start_itemsize),
    /// imm(baseofs), null_loc]`). Otherwise memset() is called; upstream
    /// emits the `LEA`s and `simple_call_no_collect` from here, and the
    /// assembler emits them from `[base_loc, startindex_loc,
    /// imm(start_itemsize), imm(baseofs), dstaddr_loc, length_loc,
    /// imm(len_itemsize), bytes_loc]`.
    pub(crate) fn consider_zero_array(
        &mut self,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let baseofs = op
            .with_array_descr(|ad| ad.base_size() as i64)
            .expect("ZERO_ARRAY needs an ArrayDescr");
        let mut length_box = args[2];

        assert!(args[3].is_constant());
        let start_itemsize = self.const_value(args[3]);

        assert!(args[4].is_constant());
        let len_itemsize = self.const_value(args[4]);
        // rewrite handles the mul of a constant length box
        let constbytes = if length_box.is_constant() {
            self.const_value(length_box)
        } else {
            -1
        };
        let base_loc = self.make_sure_var_in_reg(args[0], self.tp(args[0]), args, None, false);
        let startindex_loc =
            self.make_sure_var_in_reg(args[1], self.tp(args[1]), args, None, false);
        if (0..=16 * 8).contains(&constbytes) {
            let null_loc = Loc::Reg(X86_64_XMM_SCRATCH_REG);
            self.perform_discard(
                i,
                vec![
                    base_loc,
                    startindex_loc,
                    Loc::immed(constbytes),
                    Loc::immed(start_itemsize),
                    Loc::immed(baseofs),
                    null_loc,
                ],
                asm,
            );
        } else {
            // base_loc and startindex_loc are in two regs here (or they are
            // immediates).  Compute the dstaddr_loc, which is the raw
            // address that we will pass as first argument to memset().
            // It can be in the same register as either one, but not in
            // args[2], because we're still needing the latter.
            let dstaddr_box = self.fresh_temp_var();
            self.longevity.set(
                dstaddr_box,
                Lifetime::new(self.rm.position, self.rm.position),
            );
            let dstaddr_loc = Loc::Reg(self.rm.force_allocate_reg(
                dstaddr_box,
                &[args[2]],
                None,
                false,
                &mut self.longevity,
                &mut self.fm,
            ));
            self.flush_moves(asm);
            asm.regalloc_lea_scaled_addr(
                dstaddr_loc,
                &startindex_loc,
                start_itemsize,
                &base_loc,
                baseofs,
            );
            //
            let length_loc = if constbytes >= 0 {
                Loc::immed(constbytes)
            } else {
                // load length_loc in a register different than dstaddr_loc
                let mut length_loc =
                    self.make_sure_var_in_reg(length_box, Type::Int, &[dstaddr_box], None, false);
                if len_itemsize > 1 {
                    // we need a register that is different from dstaddr_loc,
                    // but which can be identical to length_loc (as usual,
                    // only if the length_box is not used by future operations)
                    let bytes_box = self.fresh_temp_var();
                    self.longevity
                        .set(bytes_box, Lifetime::new(self.rm.position, self.rm.position));
                    let bytes_loc = Loc::Reg(self.rm.force_allocate_reg(
                        bytes_box,
                        &[dstaddr_box],
                        None,
                        false,
                        &mut self.longevity,
                        &mut self.fm,
                    ));
                    self.flush_moves(asm);
                    asm.regalloc_lea_scaled_addr(
                        bytes_loc,
                        &length_loc,
                        len_itemsize,
                        &Loc::immed(0),
                        0,
                    );
                    length_box = bytes_box;
                    length_loc = bytes_loc;
                }
                length_loc
            };
            //
            // call memset()
            self.preserve_for_zero_array_memset();
            self.perform_discard(i, vec![dstaddr_loc, length_loc], asm);
            self.rm
                .possibly_free_var(length_box, &mut self.longevity, &mut self.fm, Type::Int);
            self.rm
                .possibly_free_var(dstaddr_box, &mut self.longevity, &mut self.fm, Type::Int);
        }
    }

    /// x86/regalloc.py `consider_int_is_true` / `consider_int_is_zero`.
    /// `TEST src, src ; SETcc dst` — input does not need to be in a
    /// register, and the result is allocated via
    /// `force_allocate_reg_or_cc` so that consumers like
    /// `GUARD_TRUE/FALSE`/`COND_CALL` may receive the boolean via the
    /// CPU condition flags instead of materialising it through SETcc.
    pub(crate) fn consider_int_is_true(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let argloc = self.loc(arg, Type::Int);
        let ops_ref: &[OpRc] = self.operations;
        let resloc = self.force_allocate_reg_or_cc(dst, ops_ref, i);
        self.perform(i, [argloc], Some(resloc), asm);
    }

    /// x86/regalloc.py `consider_uint_mul_high` — emits `MUL src`,
    /// which clobbers EAX (low) and EDX (high), so EAX must be
    /// pinned to one operand and EDX to the result.
    pub(crate) fn consider_uint_mul_high(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let mut arg1 = lhs;
        let mut arg2 = rhs;
        if arg1.is_constant() {
            std::mem::swap(&mut arg1, &mut arg2);
        }
        self.make_sure_var_in_reg(arg2, Type::Int, &[], Some(EAX), false);
        // `l1 = self.loc(arg1)` before EAX and EDX are claimed. When arg1
        // lives in EDX, claiming the result there spills it with a store
        // emitted ahead of the `MUL`, so EDX still holds arg1 when `MUL`
        // reads it.
        let l1 = self.loc(arg1, Type::Int);
        self.possibly_free_var(arg2, Type::Int);
        let tmp = self.fresh_temp_var();
        self.longevity.set(
            tmp,
            crate::regalloc::Lifetime::new(self.rm.position, self.rm.position),
        );
        self.rm.force_allocate_reg(
            tmp,
            &[],
            Some(EAX),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        self.rm.force_allocate_reg(
            dst,
            &[],
            Some(EDX),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        // x86/regalloc.py `consider_uint_mul_high` asserts l1 can be `MUL`'s operand:
        // it must not be an immediate (immediate would need a scratch
        // register to materialise, which the optimiser already folded),
        // and it can only coincide with eax when arg1 IS arg2 (the
        // swap above already pinned arg2 to eax).
        assert!(
            !matches!(l1, Loc::Immed(_) | Loc::ImmedFloat(_)),
            "consider_uint_mul_high: l1 must be a register/frame, got immediate {l1:?}"
        );
        assert!(
            !matches!(l1, Loc::Reg(r) if r.value == EAX.value) || arg1 == arg2,
            "consider_uint_mul_high: l1==eax only allowed when arg1 is arg2"
        );
        self.perform(i, [l1], Some(Loc::Reg(EDX)), asm);
    }
}

/// x86/regalloc.py `RegAlloc` — the `consider_*` family and its `oplist` dispatch.
impl<'a> RegAlloc<'a> {
    /// x86/regalloc.py _prepare
    pub fn _prepare(&mut self) {
        self.longevity = compute_vars_longevity(self.inputargs, self.operations);
        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as i32;
        self.fm = FrameManager::new(base_ofs);
        self.rm = Self::make_gpr_manager();
        self.xrm = Self::make_xmm_manager();
        self.jump_target_descr = None;
        self.final_jump_args = None;
        self.final_jump_op_position = -1;
        // aarch64/assembler.py `_walk_operations` / x86/regalloc.py
        // `walk_operations` dispatch on the live ResOp. Materializing a
        // `Vec<LirOp>` here cloned every op (and every guard's fail_args
        // SmallVec) before the walk; lower one op on the stack instead.
        // Tests that inject a replacement plan still write `j2_ops`.
        self.j2_ops.clear();
        // x86/regalloc.py:191 X86RegisterHints().add_hints(longevity, inputargs, operations)
        let hints = reghint::RegisterHints::new(
            self.rm.save_around_call_regs.clone(),
            self.xrm.save_around_call_regs.clone(),
            self.rm.all_regs.clone(),
            self.xrm.all_regs.clone(),
            self.constants.clone(),
        );
        hints.add_hints(&mut self.longevity, self.inputargs, self.operations);
    }

    /// x86/regalloc.py compute_hint_frame_locations — pre-walk scan
    /// that records the closing JUMP so `consider_label` (and the
    /// same-loop branch in this routine) can feed frame/register hints
    /// back to the forward regalloc pass.
    pub(crate) fn compute_hint_frame_locations(&mut self) {
        let operations = self.operations;
        let Some(op) = operations.last() else {
            return;
        };
        if op.opcode != OpCode::Jump {
            return;
        }
        let descr_id = op.getdescr().as_ref().map(descr_identity);
        self.final_jump_args = descr_id.map(|id| {
            (
                id,
                op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect()),
            )
        });
        self.final_jump_op_position = (operations.len() - 1) as i32;
        let Some(descr) = op.getdescr() else {
            return;
        };
        // x86/regalloc.py:1274 descr._ll_loop_code != 0:
        //   target LABEL already compiled (bridge/external jump) → apply
        //   hints now. else: same-loop, wait for consider_label.
        if let Some(target) = descr.as_loop_target_descr()
            && target.ll_loop_code() != 0
        {
            self._compute_hint_locations_from_descr(target);
        }
    }

    /// x86/regalloc.py _compute_hint_locations_from_descr: hints both
    /// Frame and Reg targets.
    pub(crate) fn _compute_hint_locations_from_descr(
        &mut self,
        descr: &dyn majit_ir::descr::LoopTargetDescr,
    ) {
        use majit_ir::descr::TargetArgLoc;
        let Some((_, jump_args)) = self.final_jump_args.as_ref() else {
            return;
        };
        let target_arglocs = descr.target_arglocs();
        // x86/regalloc.py `_compute_hint_locations_from_descr`:
        // `assert len(arglocs) == jump_op.numargs()`. The assembler remap
        // is the same 1:1 walk; a length mismatch here is the JUMP vs
        // LABEL arity bug, not a hint to skip.
        assert_eq!(
            jump_args.len(),
            target_arglocs.len(),
            "JUMP args ({}) != target LABEL args ({})",
            jump_args.len(),
            target_arglocs.len(),
        );
        let position = self.final_jump_op_position;
        let mut hinted: Vec<OpRef> = Vec::new();
        for (arg, target) in jump_args.iter().zip(target_arglocs.iter()) {
            if arg.is_constant() {
                continue;
            }
            match *target {
                TargetArgLoc::Frame {
                    position: pos,
                    ebp_offset,
                    is_float,
                } => {
                    let f = crate::regloc::FrameLoc::new(pos, ebp_offset, is_float);
                    self.fm.add_frame_pos_hint(*arg, f, &mut self.longevity);
                }
                TargetArgLoc::Reg { regnum, is_xmm } => {
                    if !hinted.contains(arg) {
                        hinted.push(*arg);
                        let r = crate::regloc::RegLoc::new(regnum, is_xmm);
                        self.longevity.fixed_register(position, r, Some(*arg));
                    }
                }
                _ => {}
            }
        }
    }

    /// x86/regalloc.py make_sure_var_in_reg
    pub fn make_sure_var_in_reg(
        &mut self,
        v: OpRef,
        tp: Type,
        forbidden_vars: &[OpRef],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
    ) -> Loc {
        // x86/regalloc.py `RegAlloc.make_sure_var_in_reg`: a `ConstFloat` is
        // `FloatImmedLoc(getfloatstorage())` before `xrm.make_sure_var_in_reg`,
        // so `save_into_mem` stores the bits and never a `ConstFloatLoc`.
        // `selected_reg` / `need_lower_byte` do not apply on that path.
        if tp == Type::Float && v.is_constant() {
            return Loc::immed_float(self.const_value(v));
        }
        if tp == Type::Float {
            self.xrm.make_sure_var_in_reg(
                v,
                tp,
                forbidden_vars,
                selected_reg,
                need_lower_byte,
                &mut self.longevity,
                &mut self.fm,
                &self.constants,
                &mut self.pending_moves,
            )
        } else {
            self.rm.make_sure_var_in_reg(
                v,
                tp,
                forbidden_vars,
                selected_reg,
                need_lower_byte,
                &mut self.longevity,
                &mut self.fm,
                &self.constants,
                &mut self.pending_moves,
            )
        }
    }

    pub(crate) fn _dispatch_j2(
        &mut self,
        j2_op: &LirOp,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        match j2_op {
            LirOp::Label { args } => self.consider_label(args, op, i, asm),
            LirOp::Jump { args } => self.consider_jump(args, op, i, asm),
            LirOp::Finish { args } => self.consider_finish(args, i, asm),
            LirOp::IntBin {
                kind,
                dst,
                lhs,
                rhs,
            } => self._dispatch_j2_int_bin(*kind, *dst, *lhs, *rhs, i, asm),
            LirOp::IntUnary { kind, dst, arg } => {
                self._dispatch_j2_int_unary(*kind, *dst, *arg, i, asm)
            }
            LirOp::IntCmp { dst, lhs, rhs, .. } => self._consider_compop(*dst, *lhs, *rhs, i, asm),
            LirOp::Guard {
                kind,
                args,
                fail_args: _,
            } => self._dispatch_j2_guard(*kind, args, op, i, asm),
            LirOp::Load {
                kind,
                dst,
                base,
                offset,
                index,
                scale,
                size,
            } => self._dispatch_j2_load(
                *kind, *dst, *base, *offset, *index, *scale, *size, op, i, asm,
            ),
            LirOp::Store {
                kind,
                base,
                offset,
                index,
                scale,
                value,
                size,
            } => self._dispatch_j2_store(
                *kind, *base, *offset, *index, *scale, *value, *size, op, i, asm,
            ),
            LirOp::Call { opcode, dst, args } => {
                self._dispatch_j2_call(*opcode, *dst, args, op, i, asm)
            }
            LirOp::Opcode {
                opcode,
                dst,
                args,
                fail_args,
            } => self._dispatch_j2_opcode(*opcode, *dst, args, fail_args, op, i, asm),
        }
    }

    pub(crate) fn _dispatch_j2_int_bin(
        &mut self,
        kind: IntBinKind,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        match kind {
            IntBinKind::Add => self.consider_int_add(dst, lhs, rhs, i, asm),
            IntBinKind::Sub => self.consider_int_sub(dst, lhs, rhs, i, asm),
            // `consider_int_add_ovf = _consider_binop_symm` and
            // `consider_int_sub_ovf = _consider_binop`: LEA cannot set flags.
            IntBinKind::AddOvf => self._consider_binop_symm(dst, lhs, rhs, i, asm),
            IntBinKind::SubOvf => self._consider_binop(dst, lhs, rhs, i, asm),
            IntBinKind::Mul
            | IntBinKind::MulOvf
            | IntBinKind::And
            | IntBinKind::Or
            | IntBinKind::Xor => self._consider_binop_symm(dst, lhs, rhs, i, asm),
            IntBinKind::LShift | IntBinKind::RShift | IntBinKind::URShift => {
                self.consider_int_lshift(dst, lhs, rhs, i, asm)
            }
        }
    }

    pub(crate) fn _dispatch_j2_int_unary(
        &mut self,
        kind: IntUnaryKind,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        match kind {
            IntUnaryKind::Neg | IntUnaryKind::Invert => self.consider_int_neg(dst, arg, i, asm),
            // x86/regalloc.py `consider_int_is_true` — arg stays in
            // memory/imm (no force-into-reg), result goes through
            // `force_allocate_reg_or_cc`.
            IntUnaryKind::IsTrue | IntUnaryKind::IsZero => {
                self.consider_int_is_true(dst, arg, i, asm)
            }
        }
    }

    pub(crate) fn _dispatch_j2_guard(
        &mut self,
        kind: GuardKind,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        match kind {
            GuardKind::True | GuardKind::False | GuardKind::NonNull | GuardKind::IsNull => {
                if let Some(&arg) = args.first() {
                    self._consider_guard_cc(arg, op, i, asm);
                }
            }
            GuardKind::Value => {
                if args.len() >= 2 {
                    self.consider_guard_value(args[0], args[1], op, i, asm);
                }
            }
            GuardKind::Class | GuardKind::GcType | GuardKind::NonNullClass => {
                if args.len() >= 2 {
                    self.consider_guard_class(args[0], args[1], op, i, asm);
                }
            }
            GuardKind::Subclass => {
                if args.len() >= 2 {
                    self.consider_guard_subclass(args[0], args[1], op, i, asm);
                }
            }
            GuardKind::IsObject => {
                if let Some(&arg) = args.first() {
                    self.consider_guard_is_object(arg, op, i, asm);
                }
            }
            GuardKind::NotForced if op.opcode == OpCode::GuardNotForced2 => {
                self.consider_guard_not_forced_2(op, i, asm)
            }
            GuardKind::NoException
            | GuardKind::NoOverflow
            | GuardKind::Overflow
            | GuardKind::NotInvalidated
            | GuardKind::FutureCondition
            | GuardKind::NotForced
            | GuardKind::AlwaysFails => self.consider_guard_no_exception(op, i, asm),
            GuardKind::Exception => self.consider_guard_exception(args, op, i, asm),
            GuardKind::Other(_) => self.consider_guard_no_exception(op, i, asm),
        }
    }

    pub(crate) fn _dispatch_j2_load(
        &mut self,
        kind: LoadKind,
        dst: OpRef,
        base: OpRef,
        offset: Option<OpRef>,
        index: Option<OpRef>,
        scale: Option<OpRef>,
        size: Option<OpRef>,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        match kind {
            LoadKind::Gc | LoadKind::Raw => {
                if let Some(offset) = offset {
                    self._consider_gc_load(dst, base, offset, size, op, i, asm);
                }
            }
            LoadKind::GcIndexed => {
                if let (Some(index), Some(scale), Some(offset), Some(size)) =
                    (index, scale, offset, size)
                {
                    self._consider_gc_load_indexed(
                        dst, base, index, scale, offset, size, op, i, asm,
                    );
                }
            }
        }
    }

    pub(crate) fn _dispatch_j2_store(
        &mut self,
        kind: StoreKind,
        base: OpRef,
        offset: Option<OpRef>,
        index: Option<OpRef>,
        scale: Option<OpRef>,
        value: OpRef,
        size: Option<OpRef>,
        _op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        match kind {
            StoreKind::Gc | StoreKind::Raw => {
                if let Some(offset) = offset {
                    self.consider_gc_store(base, offset, value, size, i, asm);
                }
            }
            StoreKind::GcIndexed => {
                if let (Some(index), Some(scale), Some(offset), Some(size)) =
                    (index, scale, offset, size)
                {
                    self.consider_gc_store_indexed(base, index, value, scale, offset, size, i, asm);
                }
            }
        }
    }

    pub(crate) fn _dispatch_j2_call(
        &mut self,
        opcode: OpCode,
        dst: Option<OpRef>,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        match opcode {
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
            | OpCode::CallMayForceN => {
                // x86/regalloc.py `_consider_real_call`: OS_MATH_SQRT is
                // `_consider_math_sqrt` and OS_THREADLOCALREF_GET is
                // `_consider_threadlocalref_get`, not real calls.
                if is_math_sqrt_call(op) {
                    let dst = dst.unwrap_or(op.pos().get());
                    let src = args.get(1).copied().unwrap_or(dst);
                    self._consider_math_sqrt(dst, src, i, asm);
                } else if is_threadlocalref_get_call(op) {
                    self._consider_threadlocalref_get(
                        dst.unwrap_or(op.pos().get()),
                        args,
                        op,
                        i,
                        asm,
                    );
                } else {
                    self._consider_call(dst, args, op, i, asm, opcode.is_call_may_force(), 1);
                }
            }
            OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN => {
                self._consider_call(dst, args, op, i, asm, true, 2);
            }
            OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                self._consider_call_assembler(dst, args, op, i, asm);
            }
            OpCode::CondCallN => self.consider_cond_call(args, i, asm),
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                self.consider_cond_call_value_i(dst, args, op, i, asm);
            }
            OpCode::CallMallocNursery | OpCode::CallMallocNurseryHeaderless => {
                self.consider_call_malloc_nursery(dst, args, i, asm)
            }
            OpCode::CallMallocNurseryVarsizeFrame => {
                self.consider_call_malloc_nursery_varsize_frame(dst, args, i, asm);
            }
            OpCode::CallMallocNurseryVarsize | OpCode::CallMallocNurseryVarsizeHeaderless => {
                self.consider_call_malloc_nursery_varsize(dst, args, i, asm);
            }
            OpCode::CheckMemoryError => self.consider_check_memory_error(args, i, asm),
            OpCode::New
            | OpCode::NewWithVtable
            | OpCode::NewArray
            | OpCode::NewArrayClear
            | OpCode::Newstr
            | OpCode::Newunicode => {
                self.consider_raw_call_like(dst, args, op, i, asm, SAVE_DEFAULT_REGS);
            }
            _ => self.not_implemented_op(opcode),
        }
    }

    pub(crate) fn _dispatch_j2_opcode(
        &mut self,
        opcode: OpCode,
        dst: Option<OpRef>,
        args: &[OpRef],
        _fail_args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        match opcode {
            OpCode::NurseryPtrIncrement if args.len() >= 2 => {
                self.consider_int_add(dst.unwrap_or(op.pos().get()), args[0], args[1], i, asm);
            }
            OpCode::IntForceGeZero if !args.is_empty() => {
                self.consider_int_force_ge_zero(dst.unwrap_or(op.pos().get()), args[0], i, asm);
            }
            OpCode::IntFloorDiv | OpCode::IntMod if args.len() >= 2 => {
                self._consider_binop(dst.unwrap_or(op.pos().get()), args[0], args[1], i, asm);
            }
            OpCode::UintMulHigh if args.len() >= 2 => {
                self.consider_uint_mul_high(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    args[1],
                    i,
                    asm,
                );
            }
            OpCode::IntSignext if args.len() >= 2 => {
                self.consider_int_signext(dst.unwrap_or(op.pos().get()), args[0], args[1], i, asm);
            }
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv
                if args.len() >= 2 =>
            {
                self._consider_float_op(dst.unwrap_or(op.pos().get()), args[0], args[1], i, asm);
            }
            OpCode::FloatNeg | OpCode::FloatAbs if !args.is_empty() => {
                self._consider_float_unary_op(dst.unwrap_or(op.pos().get()), args[0], i, asm);
            }
            // x86/regalloc.py `consider_cast_float_to_singlefloat`.
            OpCode::CastFloatToSinglefloat if !args.is_empty() => {
                self.consider_cast_float_to_singlefloat(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    i,
                    asm,
                );
            }
            // x86/regalloc.py `consider_cast_singlefloat_to_float`,
            // an alias of `consider_cast_int_to_float`.
            OpCode::CastSinglefloatToFloat if !args.is_empty() => {
                self.consider_cast_int_to_float(dst.unwrap_or(op.pos().get()), args[0], i, asm);
            }
            OpCode::FloatLt
            | OpCode::FloatLe
            | OpCode::FloatEq
            | OpCode::FloatNe
            | OpCode::FloatGt
            | OpCode::FloatGe
                if args.len() >= 2 =>
            {
                self._consider_float_cmp(dst.unwrap_or(op.pos().get()), args[0], args[1], i, asm);
            }
            OpCode::CastIntToFloat if !args.is_empty() => {
                self.consider_cast_int_to_float(dst.unwrap_or(op.pos().get()), args[0], i, asm);
            }
            OpCode::CastFloatToInt if !args.is_empty() => {
                self.consider_cast_float_to_int(dst.unwrap_or(op.pos().get()), args[0], i, asm);
            }
            // Same bank split as the non-j2 arm — see the citation there.
            OpCode::ConvertFloatBytesToLonglong if !args.is_empty() => {
                self.consider_cast_float_to_int(dst.unwrap_or(op.pos().get()), args[0], i, asm);
            }
            OpCode::ConvertLonglongBytesToFloat if !args.is_empty() => {
                self.consider_cast_int_to_float(dst.unwrap_or(op.pos().get()), args[0], i, asm);
            }
            OpCode::CastPtrToInt
            | OpCode::CastIntToPtr
            | OpCode::CastOpaquePtr
            | OpCode::SameAsI
            | OpCode::SameAsR
            | OpCode::SameAsF
            | OpCode::LoadFromGcTable
            | OpCode::VirtualRefR
                if !args.is_empty() =>
            {
                self._consider_same_as(dst.unwrap_or(op.pos().get()), args[0], op, i, asm);
            }
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldRawI
            | OpCode::GetfieldRawR
            | OpCode::GetfieldRawF
            | OpCode::ArraylenGc
            | OpCode::Strlen
            | OpCode::Unicodelen
                if !args.is_empty() =>
            {
                self.consider_getfield(dst.unwrap_or(op.pos().get()), args[0], op, i, asm);
            }
            OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF
            | OpCode::GetarrayitemGcPureI
            | OpCode::GetarrayitemGcPureR
            | OpCode::GetarrayitemGcPureF
            | OpCode::GetarrayitemRawI
            | OpCode::GetarrayitemRawR
            | OpCode::GetarrayitemRawF
            | OpCode::Strgetitem
            | OpCode::Unicodegetitem
            | OpCode::GetinteriorfieldGcI
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetinteriorfieldGcF
                if args.len() >= 2 =>
            {
                self.consider_getarrayitem(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    args[1],
                    op,
                    i,
                    asm,
                );
            }
            OpCode::SetfieldGc | OpCode::SetfieldRaw if args.len() >= 2 => {
                self.consider_setfield(args[0], args[1], i, asm);
            }
            OpCode::SetarrayitemGc
            | OpCode::SetarrayitemRaw
            | OpCode::SetinteriorfieldGc
            | OpCode::SetinteriorfieldRaw
                if args.len() >= 3 =>
            {
                self.consider_setarrayitem(args[0], args[1], args[2], i, asm);
            }
            OpCode::ZeroArray if args.len() >= 3 => {
                self.consider_zero_array(args, op, i, asm);
            }
            OpCode::Strsetitem | OpCode::Unicodesetitem if args.len() >= 3 => {
                self.consider_strsetitem(args, i, asm);
            }
            OpCode::Copystrcontent | OpCode::Copyunicodecontent => {
                self.consider_copystrcontent(args, i, asm);
            }
            OpCode::New
            | OpCode::NewWithVtable
            | OpCode::NewArray
            | OpCode::NewArrayClear
            | OpCode::Newstr
            | OpCode::Newunicode => {
                self.consider_raw_call_like(dst, args, op, i, asm, SAVE_DEFAULT_REGS);
            }
            OpCode::ForceToken => self.consider_force_token(dst.unwrap_or(op.pos().get()), i, asm),
            OpCode::LoadEffectiveAddress if args.len() >= 4 => {
                self.consider_load_effective_address(dst.unwrap_or(op.pos().get()), args, i, asm);
            }
            OpCode::SaveException | OpCode::SaveExcClass => {
                self.consider_save_exception(dst.unwrap_or(op.pos().get()), op, i, asm);
            }
            OpCode::RestoreException if args.len() >= 2 => {
                self.consider_restore_exception(args, i, asm);
            }
            OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                self.consider_cond_call_gc_wb(args, i, asm);
            }
            OpCode::JitDebug | OpCode::Keepalive => {}
            // `consider_enter_portal_frame` / `consider_leave_portal_frame`
            // hand the op to llsupport/assembler.py `enter_portal_frame` /
            // `leave_portal_frame`, which only feed the codemap
            // (`HAS_CODEMAP`); dynasm keeps no codemap.
            OpCode::EnterPortalFrame | OpCode::LeavePortalFrame => {}
            OpCode::IncrementDebugCounter => self.consider_increment_debug_counter(args, i, asm),
            OpCode::ForceSpill => self._consider_force_spill(args),
            _ => self.not_implemented_op(opcode),
        }
    }

    /// x86/regalloc.py `consider_increment_debug_counter`.
    pub(crate) fn consider_increment_debug_counter(
        &mut self,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let base_loc = self.loc(args[0], Type::Int);
        self.perform_discard(i, vec![base_loc], asm);
    }

    /// x86/regalloc.py `_consider_force_spill`: used only for testing.
    pub(crate) fn _consider_force_spill(&mut self, args: &[OpRef]) {
        let v = args[0];
        let tp = self.tp(v);
        self.force_spill_var(v, tp);
    }

    /// x86/regalloc.py `not_implemented_op`, the `oplist` default.
    pub(crate) fn not_implemented_op(&self, opcode: OpCode) -> ! {
        crate::regalloc::not_implemented(&format!("not implemented operation: {opcode:?}"))
    }

    pub(crate) fn _consider_compop(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let mut arglocs = [self.loc(lhs, self.tp(lhs)), self.loc(rhs, self.tp(rhs))];
        let lhs_in_reg = self.rm.reg_bindings_contains(lhs, &self.longevity);
        let rhs_in_reg = self.rm.reg_bindings_contains(rhs, &self.longevity);
        if !lhs_in_reg && !rhs_in_reg && !lhs.is_constant() && !rhs.is_constant() {
            arglocs[0] = self.make_sure_var_in_reg(lhs, Type::Int, &[], None, false);
        }
        let ops_ref: &[OpRc] = self.operations;
        let result_loc = self.force_allocate_reg_or_cc(dst, ops_ref, i);
        self.perform(i, arglocs, Some(result_loc), asm);
    }

    /// x86/regalloc.py `consider_int_force_ge_zero`.
    /// The result is forbidden from the argument's register: `genop_int_force_ge_zero`
    /// does `MOV res, 0` before `CMOVNS res, src`.
    pub(crate) fn consider_int_force_ge_zero(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let argloc = self.make_sure_var_in_reg(arg, Type::Int, &[], None, false);
        let resloc = self.force_allocate_reg(dst, Type::Int, &[arg], None, false);
        self.perform(i, [argloc], Some(Loc::Reg(resloc)), asm);
    }

    pub(crate) fn consider_int_signext(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        numbytes: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let argloc = self.loc(arg, Type::Int);
        let numbytesloc = self.loc(numbytes, Type::Int);
        let resloc = Loc::Reg(self.force_allocate_reg(dst, Type::Int, &[], None, false));
        self.perform(i, [argloc, numbytesloc], Some(resloc), asm);
    }

    pub(crate) fn _consider_guard_cc(
        &mut self,
        arg: OpRef,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        // x86 `_consider_guard_cc` passes `self.loc(arg)` into
        // `load_condition_into_cc` → `test_location`, which emits
        // `CMP [mem], 0` for a frame location.
        let loc = self.loc(arg, self.tp(arg));
        self.perform_guard(op, i, [loc], None, asm);
    }

    /// x86/regalloc.py `consider_guard_value`: arg0 in a register, arg1
    /// stays `self.loc` (`UCOMISD` accepts that location code).
    pub(crate) fn guard_value_locs(&mut self, lhs: OpRef, rhs: OpRef) -> (Loc, Loc) {
        let x = self.make_sure_var_in_reg(lhs, self.tp(lhs), &[], None, false);
        let y = self.loc(rhs, self.tp(rhs));
        (x, y)
    }

    pub(crate) fn consider_guard_value(
        &mut self,
        lhs: OpRef,
        rhs: OpRef,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let (x, y) = self.guard_value_locs(lhs, rhs);
        self.perform_guard(op, i, vec![x, y], None, asm);
    }

    pub(crate) fn consider_guard_class(
        &mut self,
        value: OpRef,
        class: OpRef,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let x = self.make_sure_var_in_reg(value, Type::Ref, &[], None, false);
        let y = self.loc(class, Type::Int);
        self.perform_guard(op, i, vec![x, y], None, asm);
    }

    pub(crate) fn consider_guard_is_object(
        &mut self,
        value: OpRef,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let x = self.make_sure_var_in_reg(value, Type::Ref, &[], None, false);
        // x86/regalloc.py:514 tmp_box = TempVar()
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        let loc_tmp = Loc::Reg(self.rm.force_allocate_reg(
            tmp,
            &[value],
            None,
            false,
            &mut self.longevity,
            &mut self.fm,
        ));
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        self.perform_guard(op, i, vec![x, loc_tmp], None, asm);
    }

    pub(crate) fn consider_guard_subclass(
        &mut self,
        value: OpRef,
        class: OpRef,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let x = self.make_sure_var_in_reg(value, Type::Ref, &[], None, false);
        let y = self.loc(class, Type::Int);
        // x86/regalloc.py:521 tmp_box = TempVar()
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        let loc_tmp = Loc::Reg(self.rm.force_allocate_reg(
            tmp,
            &[value],
            None,
            false,
            &mut self.longevity,
            &mut self.fm,
        ));
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        self.perform_guard(op, i, vec![x, y, loc_tmp], None, asm);
    }

    pub(crate) fn consider_guard_exception(
        &mut self,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let Some(&exception_class) = args.first() else {
            return self.consider_guard_no_exception(op, i, asm);
        };
        let loc = self.make_sure_var_in_reg(exception_class, Type::Ref, &[], None, false);
        // x86/regalloc.py:470 box = TempVar()
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        let guard_args: Vec<OpRef> = args.to_vec();
        let loc1 = Loc::Reg(self.rm.force_allocate_reg(
            tmp,
            &guard_args,
            None,
            false,
            &mut self.longevity,
            &mut self.fm,
        ));
        let resloc = if self.longevity.contains(op.pos().get()) {
            let mut forbidden = guard_args;
            forbidden.push(tmp);
            Some(Loc::Reg(self.rm.force_allocate_reg(
                op.pos().get(),
                &forbidden,
                None,
                false,
                &mut self.longevity,
                &mut self.fm,
            )))
        } else {
            None
        };
        self.perform_guard(op, i, vec![loc, loc1], resloc, asm);
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
    }

    pub(crate) fn consider_guard_no_exception(
        &mut self,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        self.perform_guard(op, i, vec![], None, asm);
    }

    /// x86/regalloc.py `consider_guard_not_forced_2`.
    pub(crate) fn consider_guard_not_forced_2(
        &mut self,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        // regalloc.py `before_call(op.getfailargs())`. The list is the
        // guard's own failargs; `locs_for_fail` walks them again for
        // locations, the same split as `_guard_impl` after `before_call`.
        let fail_args: Vec<OpRef> = op
            .guard_fail_args()
            .map(|fa| fa.iter().map(|a| a.to_opref()).collect())
            .unwrap_or_default();
        self.rm.before_call(
            &fail_args,
            SAVE_ALL_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &fail_args,
            SAVE_ALL_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.perform_guard(op, i, vec![], None, asm);
    }

    pub(crate) fn consider_restore_exception(
        &mut self,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let loc0 = self.make_sure_var_in_reg(args[0], Type::Ref, args, None, false);
        let loc1 = self.make_sure_var_in_reg(args[1], Type::Ref, args, None, false);
        self.perform_discard(i, vec![loc0, loc1], asm);
    }

    pub(crate) fn consider_check_memory_error(
        &mut self,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let x = args[0];
        let tp = self.tp(x);
        let loc = self.make_sure_var_in_reg(x, tp, &[], None, false);
        self.perform_discard(i, vec![loc], asm);
    }

    pub(crate) fn consider_finish(
        &mut self,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        if let Some(&arg) = args.first() {
            let tp = self.tp(arg);
            let loc = self.make_sure_var_in_reg(arg, tp, &[], None, false);
            self.perform(i, [loc], None, asm);
        } else {
            self.perform(i, [], None, asm);
        }
    }

    pub(crate) fn _consider_same_as(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let tp = op.opcode.result_type();
        // x86/regalloc.py `_consider_same_as`: `argloc = self.loc(...)`, then
        // `resloc = self.force_allocate_reg(op)`. The source stays where it
        // is; the move into `resloc` is the assembler's.
        let argloc = self.loc(arg, self.tp(arg));
        let resloc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        self.perform(i, [argloc], Some(resloc), asm);
    }

    pub(crate) fn _consider_float_op(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        // x86/regalloc.py `_consider_float_op`: `addsd/mulsd` is two-operand,
        // so the result has to land on the lhs (`force_result_in_reg`).
        let loc1 = self.xrm.loc(
            rhs,
            Type::Float,
            false,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
        );
        let args = [lhs, rhs];
        let loc0 = self.xrm.force_result_in_reg(
            dst,
            lhs,
            Type::Float,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, [loc0, loc1], Some(loc0), asm);
    }

    pub(crate) fn _consider_math_sqrt(
        &mut self,
        dst: OpRef,
        src: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let loc = self.xrm.force_result_in_reg(
            dst,
            src,
            Type::Float,
            &[],
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, [loc], Some(loc), asm);
    }

    /// x86/regalloc.py `_consider_threadlocalref_get`, the
    /// `translate_support_code` branch: `execute_token` always hands the
    /// compiled code the real `threadlocalref_addr`, which the header saves
    /// at `THREADLOCAL_OFS`. `getarg(0)` is the helper address, `getarg(1)`
    /// the offset.
    pub(crate) fn _consider_threadlocalref_get(
        &mut self,
        dst: OpRef,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let offset = self.const_value(args[1]);
        let (size, sign) = op
            .with_call_descr(|cd| (cd.result_size(), cd.is_result_signed()))
            .expect("OS_THREADLOCALREF_GET carries a CallDescr");
        let resloc = Loc::Reg(self.force_allocate_reg(dst, Type::Int, &[], None, false));
        self.perform(
            i,
            [
                Loc::immed(offset),
                Loc::immed(size as i64),
                Loc::immed(i64::from(sign)),
            ],
            Some(resloc),
            asm,
        );
    }

    pub(crate) fn _consider_float_unary_op(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let loc = self.xrm.force_result_in_reg(
            dst,
            arg,
            Type::Float,
            &[],
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, [loc], Some(loc), asm);
    }

    /// `_consider_float_cmp`: UCOMISD's first operand is a register. When
    /// neither `loc` is a `RegLoc`, force the non-constant side through
    /// `xrm.make_sure_var_in_reg`.
    pub(crate) fn float_cmp_ensure_reg(&mut self, vx: OpRef, vy: OpRef, arglocs: &mut [Loc; 2]) {
        if arglocs[0].is_reg() || arglocs[1].is_reg() {
            return;
        }
        if vx.is_constant() {
            arglocs[1] = self.make_sure_var_in_reg(vy, Type::Float, &[], None, false);
        } else {
            arglocs[0] = self.make_sure_var_in_reg(vx, Type::Float, &[], None, false);
        }
    }

    pub(crate) fn _consider_float_cmp(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let mut arglocs = [self.loc(lhs, Type::Float), self.loc(rhs, Type::Float)];
        self.float_cmp_ensure_reg(lhs, rhs, &mut arglocs);
        let ops_ref: &[OpRc] = self.operations;
        let result_loc = self.force_allocate_reg_or_cc(dst, ops_ref, i);
        self.perform(i, arglocs, Some(result_loc), asm);
    }

    pub(crate) fn consider_cast_int_to_float(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let loc0 = self.make_sure_var_in_reg(arg, Type::Int, &[], None, false);
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, Type::Float, &[], None, false));
        self.perform(i, [loc0], Some(result_loc), asm);
    }

    /// x86/regalloc.py `consider_cast_float_to_int` calls
    /// `xrm.make_sure_var_in_reg`. On x86, `RegAlloc::make_sure_var_in_reg`
    /// returns `FloatImmedLoc` for a ConstFloat before that call, and
    /// `CVTTSD2SI` / `FCVTZS` only encode a register source.
    pub(crate) fn cast_float_src_in_reg(&mut self, arg: OpRef) -> Loc {
        if arg.is_constant() {
            self.const_float_in_reg(arg, &[])
        } else {
            self.make_sure_var_in_reg(arg, Type::Float, &[], None, false)
        }
    }

    pub(crate) fn consider_cast_float_to_int(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        // execute_cast_const lets a non-finite or out-of-range ConstFloat reach here.
        let loc0 = self.cast_float_src_in_reg(arg);
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, Type::Int, &[], None, false));
        self.perform(i, [loc0], Some(result_loc), asm);
        self.xrm.free_temp_vars(&mut self.longevity, &mut self.fm);
    }

    /// x86/regalloc.py `consider_cast_float_to_singlefloat`.
    ///
    /// The arg stays in an xmm, the result is a GPR (the f32 bits in the
    /// low 32), and an xmm temp is allocated after the result. The temp
    /// may alias the arg, so the forbidden list is empty.
    pub(crate) fn consider_cast_float_to_singlefloat(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        // execute_cast_const lets a non-finite or out-of-range ConstFloat reach here.
        let loc0 = self.cast_float_src_in_reg(arg);
        let loc1 = Loc::Reg(self.force_allocate_reg(dst, Type::Int, &[], None, false));
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        let loctmp = Loc::Reg(self.xrm.force_allocate_reg(
            tmp,
            &[],
            None,
            false,
            &mut self.longevity,
            &mut self.fm,
        ));
        self.xrm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Float);
        self.perform(i, [loc0, loctmp], Some(loc1), asm);
        self.xrm.free_temp_vars(&mut self.longevity, &mut self.fm);
    }

    pub(crate) fn consider_getfield(
        &mut self,
        dst: OpRef,
        base: OpRef,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &[], None, false);
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        self.perform(i, [base_loc], Some(result_loc), asm);
    }

    pub(crate) fn consider_getarrayitem(
        &mut self,
        dst: OpRef,
        base: OpRef,
        index: OpRef,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let args = [base, index];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let index_loc = self.make_sure_var_in_reg(index, Type::Int, &args, None, false);
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        self.perform(i, [base_loc, index_loc], Some(result_loc), asm);
    }

    pub(crate) fn gc_load_nsize(&self, op: &Op, size: Option<OpRef>) -> i64 {
        if let Some(size) = size {
            return self.const_value(size);
        }
        op.getdescr()
            .and_then(|d| {
                d.as_field_descr().map(|fd| {
                    let s = fd.field_size() as i64;
                    if fd.is_field_signed() { -s } else { s }
                })
            })
            .or_else(|| {
                op.getdescr().and_then(|d| {
                    d.as_array_descr().map(|ad| {
                        let s = ad.item_size() as i64;
                        if ad.is_item_signed() { -s } else { s }
                    })
                })
            })
            .unwrap_or(8)
    }

    pub(crate) fn gc_offset_loc(&mut self, ofs: i64) -> Loc {
        // x86 `AddressLoc` keeps any displacement that `fits_in_32bits`,
        // including negatives.
        if fits_in_32bits(ofs) {
            Loc::immed(ofs)
        } else {
            self.pending_moves
                .push((Loc::immed(ofs), Loc::Reg(X86_64_SCRATCH_REG)));
            Loc::Reg(X86_64_SCRATCH_REG)
        }
    }

    pub(crate) fn _consider_gc_load(
        &mut self,
        dst: OpRef,
        base: OpRef,
        offset: OpRef,
        size: Option<OpRef>,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let args = [base, offset];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let ofs_loc = self.gc_offset_loc(self.const_value(offset));
        let nsize = self.gc_load_nsize(op, size);
        let tp = op.opcode.result_type();
        let res_loc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        self.perform_gc_load(i, base_loc, ofs_loc, res_loc, nsize, asm);
    }

    /// x86/regalloc.py _consider_gc_load_indexed parity, j2plan path.
    /// Emits the same locs ordering as the regular path so the assembler
    /// dispatch can be shared.
    pub(crate) fn _consider_gc_load_indexed(
        &mut self,
        dst: OpRef,
        base: OpRef,
        index: OpRef,
        scale: OpRef,
        offset: OpRef,
        size: OpRef,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let args = [base, index, scale, offset, size];
        // x86/regalloc.py:1175
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        // x86/regalloc.py:1176
        let ofs_loc = self.make_sure_var_in_reg(index, Type::Int, &args, None, false);
        // x86/regalloc.py:1177
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        // x86/regalloc.py:1178-1186
        let scale_value = self.const_value(scale);
        let offset_value = self.const_value(offset);
        let nsize = self.const_value(size);
        // x86/regalloc.py:1187 `size_loc = imm(abs(nsize))`
        let size_loc = Loc::immed(nsize.unsigned_abs() as i64);
        // x86/regalloc.py:1188-1191 sign_loc = imm1 if nsize < 0 else imm0
        let sign_loc = Loc::immed(if nsize < 0 { 1 } else { 0 });
        // x86/regalloc.py:1192-1193
        self.perform(
            i,
            vec![
                base_loc,
                ofs_loc,
                Loc::immed(scale_value),
                Loc::immed(offset_value),
                size_loc,
                sign_loc,
            ],
            Some(result_loc),
            asm,
        );
    }

    pub(crate) fn consider_setfield(
        &mut self,
        base: OpRef,
        value: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let args = [base, value];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let tp_val = self.tp(value);
        let val_loc = self.make_sure_var_in_reg(value, tp_val, &args, None, false);
        self.perform_discard(i, vec![base_loc, val_loc], asm);
    }

    pub(crate) fn consider_setarrayitem(
        &mut self,
        base: OpRef,
        index: OpRef,
        value: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let args = [base, index, value];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let index_loc = self.make_sure_var_in_reg(index, Type::Int, &args, None, false);
        let tp_val = self.tp(value);
        let val_loc = self.make_sure_var_in_reg(value, tp_val, &args, None, false);
        self.perform_discard(i, vec![base_loc, index_loc, val_loc], asm);
    }

    pub(crate) fn consider_gc_store(
        &mut self,
        base: OpRef,
        offset: OpRef,
        value: OpRef,
        size: Option<OpRef>,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let args = [base, offset, value];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let tp_val = self.tp(value);
        let value_loc = self.make_sure_var_in_reg(value, tp_val, &args, None, false);
        let size = size.map(|arg| self.const_value(arg)).unwrap_or(8);
        let ofs = self.const_value(offset);
        // pyre's GC header sits at `obj - HDR_SIZE`, so `gen_initialize_tid`
        // stores at a small negative offset, an immediate disp8. Wider
        // displacements, including other negatives and offsets >= 4096, stay
        // immediate via `gc_offset_loc` when `fits_in_32bits` (`AddressLoc`).
        let ofs_loc = if (-256..0).contains(&ofs) {
            Loc::immed(ofs)
        } else {
            self.gc_offset_loc(ofs)
        };
        self.perform_discard_gc_store(i, value_loc, base_loc, ofs_loc, size, asm);
    }

    /// x86/regalloc.py consider_gc_store_indexed parity.
    pub(crate) fn consider_gc_store_indexed(
        &mut self,
        base: OpRef,
        index: OpRef,
        value: OpRef,
        scale: OpRef,
        offset: OpRef,
        size: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let args = [base, index, value, scale, offset, size];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let factor = self.const_value(scale);
        let offset = self.const_value(offset);
        let size = self.const_value(size);
        assert!(
            size >= 1,
            "x86 GcStoreIndexed size must be >= 1 (got {size})"
        );
        let need_lower_byte = size == 1;
        let tp_val = self.tp(value);
        let value_loc = self.make_sure_var_in_reg(value, tp_val, &args, None, need_lower_byte);
        let ofs_loc = self.make_sure_var_in_reg(index, Type::Int, &args, None, false);
        self.perform_discard(
            i,
            vec![
                base_loc,
                ofs_loc,
                value_loc,
                Loc::immed(factor),
                Loc::immed(offset),
                Loc::immed(size),
            ],
            asm,
        );
    }

    pub(crate) fn _consider_call(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
        save_all_regs: bool,
        first_arg_index: usize,
    ) {
        let descr_arc = op.getdescr().expect("call op without CallDescr");
        let calldescr = descr_arc
            .as_call_descr()
            .expect("call op without CallDescr");
        assert_eq!(calldescr.arg_types().len(), args.len() - first_arg_index);

        let mut arglocs = Vec::with_capacity(args.len() + 3);
        arglocs.push(Loc::immed(0));
        arglocs.push(Loc::immed(calldescr.result_size() as i64));
        arglocs.push(Loc::immed(if calldescr.is_result_signed() { 1 } else { 0 }));
        let mut force_store_refs = Vec::new();
        for (arg_index, &arg) in args.iter().enumerate() {
            let tp = if arg_index >= first_arg_index {
                calldescr.arg_types()[arg_index - first_arg_index]
            } else {
                self.tp(arg)
            };
            arglocs.push(self.loc(arg, tp));
            if tp == Type::Ref && !arg.is_constant() {
                force_store_refs.push(arg);
            }
        }

        let can_collect = calldescr.get_extra_info().check_can_collect();
        let save_regs = if save_all_regs {
            SAVE_ALL_REGS
        } else if can_collect {
            SAVE_GCREF_REGS
        } else {
            SAVE_DEFAULT_REGS
        };
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.before_call(
            &force_store_refs,
            save_regs,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &[],
            save_regs,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );

        // Same gcmap snapshot as `consider_call` above. The j2 dispatch
        // handles the same Call*/CallMayForce*/CallReleaseGil* opcodes,
        // so a collecting callee must record the live Ref slots that
        // before_call just synced to the jitframe.  call_result_gpr is
        // excluded for the same reason as `consider_call`
        // (x86/callbuilder.py push_gcmap, aarch64/callbuilder.py push_gcmap).
        let gcmap = if can_collect {
            Some(self.get_gcmap(&[call_result_gpr()], false) as usize)
        } else {
            None
        };

        let result_tp = op.opcode.result_type();
        let result_loc = if result_tp != Type::Void {
            let dst = dst.unwrap_or(op.pos().get());
            let r = if result_tp == Type::Float {
                self.xrm.after_call(dst, &mut self.longevity)
            } else {
                self.rm.after_call(dst, &mut self.longevity)
            };
            Some(Loc::Reg(r))
        } else {
            None
        };
        arglocs[0] = result_loc.unwrap_or(Loc::immed(0));

        if let Some(gcmap) = gcmap {
            self.perform_with_gcmap_ptr(i, arglocs, result_loc, gcmap, asm);
        } else {
            self.perform(i, arglocs, result_loc, asm);
        }
    }

    pub(crate) fn _consider_call_assembler(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        // llsupport/regalloc.py `locs_for_call_assembler`.
        let arglocs = if args.len() == 2 {
            let vable = args[1];
            let tp = self.tp(vable);
            self.rm
                ._sync_var_to_stack(vable, tp, &mut self.longevity, &mut self.fm);
            vec![
                self.loc(args[0], self.tp(args[0])),
                Loc::Frame(self.fm.loc(vable, tp, false, &mut self.longevity)),
            ]
        } else {
            assert!(args.len() == 1);
            vec![self.loc(args[0], self.tp(args[0]))]
        };

        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.before_call(
            &[],
            SAVE_ALL_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &[],
            SAVE_ALL_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );

        let result_tp = op.opcode.result_type();
        let result_loc = if result_tp != Type::Void {
            let dst = dst.unwrap_or(op.pos().get());
            let r = if result_tp == Type::Float {
                self.xrm.after_call(dst, &mut self.longevity)
            } else {
                self.rm.after_call(dst, &mut self.longevity)
            };
            Some(Loc::Reg(r))
        } else {
            None
        };
        let gcmap = self.get_gcmap(&[call_result_gpr()], false) as usize;
        self.perform_with_gcmap_ptr(i, arglocs, result_loc, gcmap, asm);
    }

    pub(crate) fn consider_raw_call_like(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
        save_regs: u8,
    ) {
        Self::check_cond_call_value_descr_arity(op, args.len());
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.before_call(
            &[],
            save_regs,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &[],
            save_regs,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );

        let mut arglocs = Vec::new();
        for &arg in args {
            let tp = self.tp(arg);
            arglocs.push(self.loc(arg, tp));
        }

        let result_tp = op.opcode.result_type();
        let result_loc = if result_tp != Type::Void {
            let dst = dst.unwrap_or(op.pos().get());
            let r = if result_tp == Type::Float {
                self.xrm.after_call(dst, &mut self.longevity)
            } else {
                self.rm.after_call(dst, &mut self.longevity)
            };
            Some(Loc::Reg(r))
        } else {
            None
        };
        self.perform(i, arglocs, result_loc, asm);
    }

    pub(crate) fn consider_call_malloc_nursery(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.spill_or_move_registers_before_call(
            &MALLOC_NURSERY_CLOBBER,
            &[],
            SAVE_DEFAULT_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        let result_reg = self.rm.force_allocate_reg(
            dst.unwrap_or(OpRef::NONE),
            &[],
            Some(MALLOC_NURSERY_RESULT),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        // x86/regalloc.py:1025-1026: `t = TempVar()`; reserve the second
        // clobber register.
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        self.rm.force_allocate_reg(
            tmp,
            &[],
            Some(MALLOC_NURSERY_CLOBBER[1]),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        let size_val = args.first().map(|&arg| self.const_value(arg)).unwrap_or(0);
        self.perform_with_gcmap(i, [Loc::immed(size_val)], Some(Loc::Reg(result_reg)), asm);
    }

    pub(crate) fn consider_call_malloc_nursery_varsize_frame(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let Some(&size_arg) = args.first() else {
            return;
        };
        let sizeloc = self.make_sure_var_in_reg(size_arg, Type::Int, &[], None, false);
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.spill_or_move_registers_before_call(
            &MALLOC_NURSERY_CLOBBER,
            &[],
            SAVE_DEFAULT_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.possibly_free_var(size_arg, Type::Int);
        let result_reg = self.rm.force_allocate_reg(
            dst.unwrap_or(OpRef::NONE),
            &[],
            Some(MALLOC_NURSERY_RESULT),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        // x86/regalloc.py `consider_call_malloc_nursery_varsize_frame`:
        // `tmp_box = TempVar()`.
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        self.rm.force_allocate_reg(
            tmp,
            &[],
            Some(MALLOC_NURSERY_CLOBBER[1]),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        self.perform_with_gcmap(i, [sizeloc], Some(Loc::Reg(result_reg)), asm);
    }

    /// x86/regalloc.py `consider_call_malloc_nursery_varsize`.
    pub(crate) fn consider_call_malloc_nursery_varsize(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let length_box = args[2];
        // Spill only when `stays_alive(length_box)`. A length that dies here
        // is unbound with no stack sync, and `loc` would then allocate an
        // empty slot.
        if self.rm.stays_alive(length_box, &self.longevity) {
            let type_index = OpTypeIndex::from_parts(
                self.inputargs,
                self.operations,
                &self.inputarg_pos,
                &self.op_pos,
            );
            self.rm.spill_or_move_registers_before_call(
                &MALLOC_NURSERY_CLOBBER,
                &[],
                // The `malloc_slowpath_varsize` helper saves every register
                // but ecx/edx into the jitframe, so the default set is enough.
                SAVE_DEFAULT_REGS,
                &mut self.longevity,
                &mut self.fm,
                &mut self.pending_moves,
                &type_index,
            );
        }
        // The result is in ecx / x0.
        let result_reg = self.rm.force_allocate_reg(
            dst.unwrap_or(OpRef::NONE),
            &[],
            Some(MALLOC_NURSERY_RESULT),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        // edx / x1 is a temporary: the bump-pointer fast path writes the new
        // nursery_free into it, so no live Ref may stay bound there.
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        self.rm.force_allocate_reg(
            tmp,
            &[],
            Some(MALLOC_NURSERY_CLOBBER[1]),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        // `gcmap = self.get_gcmap([ecx, edx])`: allocate the gcmap *before*
        // freeing the temporary.
        let gcmap = self.get_gcmap(&MALLOC_NURSERY_CLOBBER, false) as usize;
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        // length_box always survives: it is typically also present in the
        // next operation that copies it inside the new array. Loading it
        // from the stack is fine as long as it is not ecx / edx.
        let lengthloc = self.rm.loc(
            length_box,
            Type::Int,
            false,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
        );
        self.rm
            .possibly_free_var(length_box, &mut self.longevity, &mut self.fm, Type::Int);
        let itemsize = self.const_value(args[1]);
        let kind = self.const_value(args[0]);
        self.perform_with_gcmap_ptr(
            i,
            [lengthloc, Loc::immed(itemsize), Loc::immed(kind)],
            Some(Loc::Reg(result_reg)),
            gcmap,
            asm,
        );
    }

    pub(crate) fn consider_jump(
        &mut self,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let descr_id = op.getdescr().as_ref().map(descr_identity);
        self.final_jump_args = descr_id.map(|id| (id, args.to_vec()));
        self.jump_target_descr = descr_id;
        let mut locs = Vec::new();
        for &arg in args {
            let tp = self.tp(arg);
            locs.push(self.loc_must_exist(arg, tp));
        }
        self.perform(i, locs, None, asm);
    }

    pub(crate) fn consider_label(
        &mut self,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let position = self.rm.position;
        for &arg in args {
            let tp = self.tp(arg);
            if self
                .longevity
                .get(arg)
                .is_some_and(|lt| lt.is_last_real_use_before(position))
            {
                self.force_spill_var(arg, tp);
            }
        }

        // x86/regalloc.py `consider_label`: we need to make sure that no
        // variable is stored in ebp
        let ebp = self.rm.frame_reg;
        for &arg in args {
            let tp = self.tp(arg);
            if matches!(self.loc(arg, tp), Loc::Reg(r) if r == ebp) {
                let loc2 = self.fm.loc(arg, tp, false, &mut self.longevity);
                self.rm.spill_moves.push((Loc::Reg(ebp), Loc::Frame(loc2)));
            }
        }
        self.rm.box_currently_in_frame_reg = None;

        let mut locs = Vec::new();
        for &arg in args {
            let tp = self.tp(arg);
            let loc = self.loc(arg, tp);
            debug_assert!(!matches!(loc, Loc::Reg(r) if r == self.rm.frame_reg));
            if let Loc::Reg(_) = loc {
                self.fm.mark_as_free(arg, tp, &mut self.longevity)
            }
            locs.push(loc);
        }

        if let Some(label_id) = op.getdescr().as_ref().map(descr_identity)
            && self
                .final_jump_args
                .as_ref()
                .is_some_and(|(jump_id, _)| *jump_id == label_id)
        {
            let jump_args = self.final_jump_args.as_ref().unwrap().1.clone();
            let position = self.final_jump_op_position;
            let mut hinted: Vec<OpRef> = Vec::new();
            for (iarg, target_loc) in jump_args.iter().zip(locs.iter()) {
                if iarg.is_constant() {
                    continue;
                }
                match *target_loc {
                    Loc::Frame(frame_loc) => {
                        self.fm
                            .add_frame_pos_hint(*iarg, frame_loc, &mut self.longevity);
                    }
                    Loc::Reg(r) => {
                        if !hinted.contains(iarg) {
                            hinted.push(*iarg);
                            self.longevity.fixed_register(position, r, Some(*iarg));
                        }
                    }
                    _ => {}
                }
            }
        }
        self.perform(i, locs, None, asm);
    }

    pub(crate) fn consider_force_token(
        &mut self,
        dst: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, Type::Ref, &[], None, false));
        self.perform(i, [] as [Loc; 0], Some(result_loc), asm);
    }

    pub(crate) fn consider_load_effective_address(
        &mut self,
        dst: OpRef,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let [p0, i0, baseofs, shift] = match args {
            [a, b, c, d, ..] => [*a, *b, *c, *d],
            other => panic!("LoadEffectiveAddress expects 4 args, got {other:?}"),
        };
        let ploc = self.make_sure_var_in_reg(p0, Type::Int, &[i0], None, false);
        let iloc = self.make_sure_var_in_reg(i0, Type::Int, &[p0], None, false);
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, Type::Int, &[p0, i0], None, false));
        let locs = vec![
            ploc,
            iloc,
            self.loc(baseofs, Type::Int),
            self.loc(shift, Type::Int),
        ];
        self.perform(i, locs, Some(result_loc), asm);
    }

    pub(crate) fn consider_save_exception(
        &mut self,
        dst: OpRef,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        self.perform(i, [] as [Loc; 0], Some(result_loc), asm);
    }

    pub(crate) fn consider_strsetitem(
        &mut self,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let mut locs = Vec::new();
        for &arg in args {
            let tp = self.tp(arg);
            locs.push(self.make_sure_var_in_reg(arg, tp, args, None, false));
        }
        self.perform_discard(i, locs, asm);
    }

    pub(crate) fn preserve_for_zero_array_memset(&mut self) {
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.before_call(
            &[],
            SAVE_DEFAULT_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &[],
            SAVE_DEFAULT_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
    }

    /// x86/regalloc.py `consider_cond_call` COND_CALL_VALUE arm
    /// (`consider_cond_call_value_i = consider_cond_call`, `_r` the same).
    ///
    /// The result shares args[0]'s register via `force_result_in_reg`
    /// (`forbidden_vars=args[2:]`). The callee address stays an Immed on
    /// the op. The helper returns a plain word — no `before_call` / raw-call
    /// result-in-eax rewrite. Extra-arg locations are captured before
    /// `force_result_in_reg`; `CondCallSlowPath.generate_body` remaps them
    /// into `cond_call_register_arguments` on the taken path only.
    pub(crate) fn consider_cond_call_value_i(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        Self::check_cond_call_value_descr_arity(op, args.len());
        assert!(
            (2..=6).contains(&args.len()),
            "COND_CALL_VALUE takes 0..=4 extra arguments (x86/regalloc.py consider_cond_call)"
        );
        let extra_locs: Vec<Loc> = args[2..]
            .iter()
            .map(|&arg| self.loc(arg, self.tp(arg)))
            .collect();
        let value = args[0];
        let result_v = dst.unwrap_or(op.pos().get());
        let forbidden: Vec<OpRef> = args[2..].to_vec();
        let resloc = self.rm.force_result_in_reg(
            result_v,
            value,
            self.tp(value),
            &forbidden,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        let res_reg = match resloc {
            Loc::Reg(r) => r,
            other => panic!(
                "force_result_in_reg must return a register (consider_cond_call), got {other:?}"
            ),
        };
        let gcmap = self.get_gcmap(&[res_reg], false) as usize;
        // Assembler prefix stays `[argloc, resloc]`; x86 tests `resloc`.
        let mut locs = Vec::with_capacity(extra_locs.len() + 2);
        locs.push(resloc);
        locs.push(resloc);
        locs.extend(extra_locs);
        self.perform_with_gcmap_ptr(i, locs, Some(resloc), gcmap, asm);
    }

    /// x86/regalloc.py `consider_cond_call`. CondCallN has no result.
    pub(crate) fn consider_cond_call(
        &mut self,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        // x86/regalloc.py `consider_cond_call`: extra-arg locations first
        // (`self.loc` of args[2:]), then the gcmap. The predicate and func
        // sit in the assembler prefix; extras are remapped on the taken
        // path only (`CondCallSlowPath.generate_body`).
        let mut locs = Vec::new();
        for &arg in args {
            let tp = self.tp(arg);
            locs.push(self.loc(arg, tp));
        }
        let gcmap = self.get_gcmap(&[], false) as usize;
        self.perform_with_gcmap_ptr(i, locs, None, gcmap, asm);
    }

    pub(crate) fn consider_copystrcontent(
        &mut self,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let mut locs = Vec::new();
        for &arg in args {
            let tp = self.tp(arg);
            locs.push(self.loc(arg, tp));
        }
        self.perform_discard(i, locs, asm);
    }

    pub(crate) fn consider_cond_call_gc_wb(
        &mut self,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let mut arglocs = Vec::new();
        for (idx, &arg) in args.iter().enumerate() {
            let tp = if idx == 0 { Type::Ref } else { Type::Int };
            arglocs.push(self.make_sure_var_in_reg(arg, tp, args, None, false));
        }
        self.perform_discard(i, arglocs, asm);
    }
}
