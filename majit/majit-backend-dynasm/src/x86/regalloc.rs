//! Port of `rpython/jit/backend/x86/regalloc.py` — arch-specific
//! register configuration plus the `consider_int_*` family that
//! depends on x86's 2-operand encoding (`OP dst, src` where dst
//! must be the same as the first operand).
//! Upstream splits per arch directory; pyre matches that split here.
//!
//! Methods that read/write the shared `RegAlloc` state are declared
//! as a second `impl` block on `crate::regalloc::RegAlloc<'a>`. The
//! `x86` module is `#[cfg(target_arch = "x86_64")]` gated in
//! `lib.rs`, so these methods only compile on x86_64.

use crate::regalloc::{RegAlloc, RegAllocOp, fits_in_32bits};
use crate::regloc::Loc;
use crate::regloc::{
    EAX, EBP, EBX, ECX, EDI, EDX, ESI, R8, R9, R10, R12, R13, R14, R15, RegLoc, XMM0, XMM1, XMM2,
    XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14,
};
use majit_ir::{OpRc, OpRef, Type};

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

/// `consider_int_*` family — x86-side `consider_*_j2` entries.
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
    fn _consider_binop_part_j2(
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
    pub(crate) fn consider_binop_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let (loc, argloc) = self._consider_binop_part_j2(dst, lhs, rhs, false);
        self.perform(i, [loc, argloc], Some(loc), output);
    }

    /// x86/regalloc.py `_consider_binop_symm` — symmetric binop
    /// with the swap heuristic.
    pub(crate) fn consider_binop_symm_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let (loc, argloc) = self._consider_binop_part_j2(dst, lhs, rhs, true);
        self.perform(i, [loc, argloc], Some(loc), output);
    }

    /// x86/regalloc.py `_consider_lea` — emits `LEA dst, [lhs +
    /// rhs]` so the result register can differ from `lhs`. Limited
    /// to RHS constants that fit a 32-bit displacement.
    fn _consider_lea_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let loc = self.make_sure_var_in_reg(lhs, self.tp(lhs), &[], None, false);
        self.possibly_free_var(lhs, self.tp(lhs));
        let argloc = self.loc(rhs, self.tp(rhs));
        let resloc = Loc::Reg(self.force_allocate_reg(dst, Type::Int, &[], None, false));
        self.perform(i, [loc, argloc], Some(resloc), output);
    }

    /// x86/regalloc.py `consider_int_add` — LEA shortcut when
    /// the RHS is a 32-bit-fitting immediate, otherwise symmetric
    /// 2-operand `ADD dst, src`.
    pub(crate) fn consider_int_add_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        if rhs.is_constant() {
            let val = self.const_value(rhs);
            if fits_in_32bits(val) {
                return self._consider_lea_j2(dst, lhs, rhs, i, output);
            }
        }
        self.consider_binop_symm_j2(dst, lhs, rhs, i, output);
    }

    /// x86/regalloc.py `consider_int_sub` — LEA shortcut with
    /// negated immediate (`LEA dst, [lhs - imm]`).
    pub(crate) fn consider_int_sub_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        if rhs.is_constant() {
            let val = self.const_value(rhs);
            // x86/regalloc.py `rx86.fits_in_32bits(-y.value)`. The
            // negate-then-range-check filters `i64::MIN` naturally
            // because `wrapping_neg(i64::MIN) == i64::MIN`, which sits
            // outside the i32 range, so the LEA shortcut is skipped
            // and the regular SUB path is taken.
            if fits_in_32bits(val.wrapping_neg()) {
                return self._consider_lea_j2(dst, lhs, rhs, i, output);
            }
        }
        self.consider_binop_j2(dst, lhs, rhs, i, output);
    }

    /// x86/regalloc.py `consider_int_lshift` — shift count must
    /// be in ECX (`SHL/SHR/SAR reg, CL`) unless it's an immediate.
    pub(crate) fn consider_int_lshift_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
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
        self.perform(i, [loc1, loc2], Some(loc1), output);
    }

    /// x86/regalloc.py `consider_int_add_ovf = _consider_binop_symm`.
    /// Distinct from `consider_int_add` because the LEA shortcut does
    /// not set CPU flags and cannot be used when the trace needs the
    /// overflow condition.
    pub(crate) fn consider_int_add_ovf_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.consider_binop_symm_j2(dst, lhs, rhs, i, output);
    }

    /// x86/regalloc.py `consider_int_sub_ovf = _consider_binop`.
    /// LEA cannot set flags, so SUB-via-LEA is unavailable for the
    /// overflow form.
    pub(crate) fn consider_int_sub_ovf_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.consider_binop_j2(dst, lhs, rhs, i, output);
    }

    /// x86/regalloc.py `consider_int_neg` / `consider_int_invert`
    /// — `NEG dst` and `NOT dst` overwrite their argument register.
    pub(crate) fn consider_unary_int_j2(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let args = [arg];
        let loc = self.rm.force_result_in_reg(
            dst,
            arg,
            Type::Int,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, [loc], Some(loc), output);
    }

    /// x86/regalloc.py `consider_int_is_true` / `consider_int_is_zero`.
    /// `TEST src, src ; SETcc dst` — input does not need to be in a
    /// register, and the result is allocated via
    /// `force_allocate_reg_or_cc` so that consumers like
    /// `GUARD_TRUE/FALSE`/`COND_CALL` may receive the boolean via the
    /// CPU condition flags instead of materialising it through SETcc.
    pub(crate) fn consider_int_is_true_j2(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let argloc = self.loc(arg, Type::Int);
        let ops_ref: &[OpRc] = self.operations;
        let resloc = self.force_allocate_reg_or_cc(dst, ops_ref, i);
        self.perform(i, [argloc], Some(resloc), output);
    }

    /// x86/regalloc.py `consider_uint_mul_high` — emits `MUL src`,
    /// which clobbers EAX (low) and EDX (high), so EAX must be
    /// pinned to one operand and EDX to the result.
    pub(crate) fn consider_uint_mul_high_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let mut arg1 = lhs;
        let mut arg2 = rhs;
        if arg1.is_constant() {
            std::mem::swap(&mut arg1, &mut arg2);
        }
        self.make_sure_var_in_reg(arg2, Type::Int, &[], Some(EAX), false);
        // Same-register square: arg1 is arg2, already in EAX. Freeing that
        // binding below drops the location, so keep this snapshot.
        let l1_same = self.loc(arg1, Type::Int);
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
        // MUL's high half is EDX. If arg1 already lived there, claiming the
        // result spills it, and a snapshot taken before the claim still names
        // EDX. Read the operand after the claim. A same-register square stays
        // on the EAX snapshot: that binding was freed above.
        let l1 = if arg1 == arg2 {
            l1_same
        } else {
            self.loc(arg1, Type::Int)
        };
        // x86/regalloc.py `consider_uint_mul_high` asserts l1 can be `MUL`'s operand:
        // it must not be an immediate (immediate would need a scratch
        // register to materialise, which the optimiser already folded),
        // and it can only coincide with eax when arg1 IS arg2 (the
        // swap above already pinned arg2 to eax).
        assert!(
            !matches!(l1, Loc::Immed(_) | Loc::ImmedFloat(_)),
            "consider_uint_mul_high_j2: l1 must be a register/frame, got immediate {l1:?}"
        );
        assert!(
            !matches!(l1, Loc::Reg(r) if r.value == EAX.value) || arg1 == arg2,
            "consider_uint_mul_high_j2: l1==eax only allowed when arg1 is arg2"
        );
        self.perform(i, [l1], Some(Loc::Reg(EDX)), output);
    }
}
