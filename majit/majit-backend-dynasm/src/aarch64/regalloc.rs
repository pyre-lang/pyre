//! Port of `rpython/jit/backend/aarch64/regalloc.py` — the register
//! configuration, the `regalloc_operations` dispatch and the
//! `prepare_op_*` family.
//!
//! Methods that read/write the shared `RegAlloc` state are declared
//! as a second `impl` block on `crate::regalloc::RegAlloc<'a>`, so
//! Rust's name resolution picks up the right arch flavour at link
//! time (the `aarch64` module is `#[cfg(target_arch = "aarch64")]`
//! gated in `lib.rs`).

use crate::aarch64::registers;
use crate::j2plan::{GuardKind, IntBinKind, IntUnaryKind, LirOp, LoadKind, StoreKind};
use crate::regalloc::{
    FrameManager, Lifetime, RegAlloc, RegAllocAssembler, SAVE_ALL_REGS, SAVE_DEFAULT_REGS,
    SAVE_GCREF_REGS, compute_vars_longevity, is_math_sqrt_call,
};
use crate::regloc::{Loc, RegLoc};
use majit_ir::{Op, OpRc, OpRef, Type};
use majit_ir::{OpCode, OpTypeIndex, descr_identity};

/// aarch64/regalloc.py `DEFAULT_IMM_SIZE = 4096`.
const DEFAULT_IMM_SIZE: i64 = 4096;

/// aarch64/regalloc.py `check_imm_arg`.
fn check_imm_arg(arg: i64) -> bool {
    (0..DEFAULT_IMM_SIZE).contains(&arg)
}

/// aarch64/regalloc.py `check_imm_box`. PyPy accepts only
/// `ConstInt` values in the AArch64 immediate range; non-Int
/// constants (ConstFloat/ConstPtr) and box references fall through
/// to the register form. PyPy reads `arg.getint()` directly because
/// the int is inlined on `ConstInt` (history.py `ConstInt.value`);
/// pyre's inline-Const variant `OpRef::ConstInt` carries the i64
/// on the OpRef itself (`arg.getint()` never fails).
pub(crate) fn check_imm_box(arg: OpRef) -> bool {
    match arg {
        OpRef::ConstInt(v) => (0..DEFAULT_IMM_SIZE).contains(&v),
        _ => false,
    }
}

/// aarch64/registers.py
///   `all_regs = registers[:14] + [x19, x20]`
pub fn all_core_regs() -> Vec<RegLoc> {
    registers::ALL_REGS.to_vec()
}

/// aarch64/registers.py
///   `caller_resp = argument_regs + [x8, x9, x10, x11, x12, x13]`
///
/// In RPython, `save_around_call_regs` is the AAPCS64 caller-saved
/// subset (x0..x13) — the regs whose contents must be assumed
/// clobbered after a `bl`.  Mirrored verbatim here.
pub fn save_around_call_core_regs() -> Vec<RegLoc> {
    registers::CALLER_RESP.to_vec()
}

/// aarch64/registers.py: `all_vfp_regs = vfpregisters[:8]`.  Pyre's
/// VFP allocator pool stays at the upstream cap of 8 (d0..d7).
pub fn all_float_regs() -> Vec<RegLoc> {
    registers::ALL_VFP_REGS.to_vec()
}

/// aarch64/registers.py `fp = x29`.  RPython's frame-pointer
/// register on AAPCS64.
pub fn frame_reg() -> RegLoc {
    registers::FP
}

/// aarch64/locations.py: `call_result_location` returns x0 for GPR
/// (AAPCS64 first return register).
pub fn call_result_gpr() -> RegLoc {
    RegLoc::new(0, false)
}

/// aarch64/registers.py `argument_regs`. `_prepare_op_cond_call` places
/// extra COND_CALL / COND_CALL_VALUE arguments here (`x0` …).
pub fn cond_call_argument_regs() -> &'static [RegLoc] {
    &registers::ARGUMENT_REGS
}

/// aarch64/locations.py: `call_result_location` returns d0 for VFP.
pub fn call_result_fpr() -> RegLoc {
    RegLoc::new(0, true)
}

/// `core_reg_index` returns the position of `reg` in the canonical
/// `all_core_regs` list — used by gcmap and jitframe slot tables.
///
/// aarch64 mapping (matches `all_core_regs`):
///   x0..x13 → 0..13, x19 → 14, x20 → 15.
pub fn core_reg_index(reg: RegLoc) -> Option<usize> {
    match reg.value {
        0..=13 => Some(reg.value as usize),
        19 => Some(14),
        20 => Some(15),
        _ => None,
    }
}

/// aarch64/regalloc.py:962 nursery-bump path clobbers `[r.x0, r.x1]`.
/// Exported as a per-arch pair so the shared regalloc base can spill
/// the clobbered set without branching on `cfg!`.
pub const MALLOC_NURSERY_CLOBBER: [RegLoc; 2] = [
    RegLoc {
        value: 0,
        is_xmm: false,
    },
    RegLoc {
        value: 1,
        is_xmm: false,
    },
];

/// aarch64: result register after the nursery bump (x0).  Identical
/// value to MALLOC_NURSERY_CLOBBER[0] but separated to document
/// intent.
pub const MALLOC_NURSERY_RESULT: RegLoc = RegLoc {
    value: 0,
    is_xmm: false,
};

/// `prepare_op_int_*` family.
///
/// AAPCS64's `ADD/SUB/MUL/AND/ORR/EOR/LSL/ASR/LSR Rd, Rn, Rm` accepts
/// three distinct registers, so the result is always allocated via
/// `force_allocate_reg` independently of the inputs.  RPython parity:
/// `rpython/jit/backend/aarch64/regalloc.py prepare_op_int_add`
/// and `:362 prepare_op_int_mul` (the latter is reused for
/// `and/or/xor/lshift/rshift/urshift/uint_mul_high`).
impl<'a> RegAlloc<'a> {
    /// aarch64/regalloc.py `prepare_op_int_mul`. 3-operand form:
    /// both operands in registers, result allocated separately.
    /// PyPy passes the full `boxes = op.getarglist()` as
    /// `forbidden_vars` for both `make_sure_var_in_reg` calls
    /// (regalloc.py:366-367), and after allocating the result also
    /// calls `possibly_free_var(op)` (regalloc.py:372) so the result
    /// register is reclaimable when the op is dead.
    pub(crate) fn prepare_op_int_mul(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let boxes = [lhs, rhs];
        let lhs_loc = self.make_sure_var_in_reg(lhs, Type::Int, &boxes, None, false);
        let rhs_loc = self.make_sure_var_in_reg(rhs, Type::Int, &boxes, None, false);
        self.possibly_free_var(lhs, Type::Int);
        self.possibly_free_var(rhs, Type::Int);
        let res = self.force_allocate_reg(dst, Type::Int, &[], None, false);
        self.possibly_free_var(dst, Type::Int);
        let res_loc = Loc::Reg(res);
        self.perform(i, [lhs_loc, rhs_loc], Some(res_loc), asm);
    }

    /// aarch64/regalloc.py `prepare_op_int_add` → `prepare_int_ri`:
    /// allows either operand to be an immediate that fits the AArch64
    /// `add Rd, Rn, #imm12` (or shifted) encoding.  For non-immediate
    /// or out-of-range constants, fall through to the register form.
    pub(crate) fn prepare_op_int_add(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        self.prepare_int_ri(dst, lhs, rhs, i, asm);
    }

    /// aarch64/regalloc.py `prepare_op_int_sub`. `lhs` always
    /// becomes Rn; `rhs` accepts the `sub Rd, Rn, #imm12` immediate
    /// form only when it is a `ConstInt` in `[0, 4096)`
    /// (`check_imm_box`). Other constants (ConstFloat/ConstPtr) and
    /// out-of-range ints fall through to the register form.
    pub(crate) fn prepare_op_int_sub(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let boxes = [lhs, rhs];
        // history.py — inline-Const carries its value directly.
        let imm_rhs = check_imm_box(rhs);
        let lhs_loc = self.make_sure_var_in_reg(lhs, Type::Int, &boxes, None, false);
        let rhs_loc = if imm_rhs {
            self.loc(rhs, Type::Int)
        } else {
            self.make_sure_var_in_reg(rhs, Type::Int, &boxes, None, false)
        };
        self.possibly_free_var(lhs, Type::Int);
        self.possibly_free_var(rhs, Type::Int);
        let res = self.force_allocate_reg(dst, Type::Int, &[], None, false);
        self.perform(i, [lhs_loc, rhs_loc], Some(Loc::Reg(res)), asm);
    }

    /// aarch64/regalloc.py `prepare_unary` covers `int_neg`,
    /// `int_invert`, `int_is_true`, `int_is_zero`. `neg X(d), X(s)`
    /// and `mvn X(d), X(s)` are 3-operand: result independent of
    /// source. PyPy asserts `not isinstance(a0, Const)`
    /// (regalloc.py:458) since the optimizer is expected to have
    /// folded any const-arg unary into a constant result before this
    /// point.
    pub(crate) fn prepare_unary(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        assert!(
            !arg.is_constant(),
            "prepare_unary expects a non-const arg; got constant OpRef {arg:?} (should have been folded earlier)"
        );
        let arg_loc = self.make_sure_var_in_reg(arg, Type::Int, &[], None, false);
        self.possibly_free_var(arg, Type::Int);
        let res = self.force_allocate_reg(dst, Type::Int, &[], None, false);
        self.perform(i, [arg_loc], Some(Loc::Reg(res)), asm);
    }

    /// aarch64 `int_is_true` / `int_is_zero`: shares the 3-op `prepare_unary`
    /// shape from regalloc.py:456 since `cmp Xn, #0 ; cset Wd, ne` keeps
    /// the input register live while writing a fresh destination.
    ///
    /// When the next op consumes the flags, `regalloc.py
    /// prepare_comp_unary` allocates no destination at all and
    /// `opassembler.py emit_comp_op_int_is_true` emits the `cmp` alone,
    /// returning the condition.  `force_allocate_reg_or_cc` spells that as
    /// the frame-register sentinel, which the `IntIsTrue` / `IntIsZero` emit
    /// arms recognise through `flush_cc`.
    pub(crate) fn prepare_comp_unary(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        assert!(
            !arg.is_constant(),
            "prepare_comp_unary expects a non-const arg; got constant OpRef {arg:?} (should have been folded earlier)"
        );
        let arg_loc = self.make_sure_var_in_reg(arg, Type::Int, &[], None, false);
        self.possibly_free_var(arg, Type::Int);
        let ops_ref: &[majit_ir::OpRc] = self.operations;
        let res = self.force_allocate_reg_or_cc(dst, ops_ref, i);
        self.perform(i, [arg_loc], Some(res), asm);
    }

    /// aarch64/regalloc.py `prepare_int_ri`. Either operand may
    /// take the `add Rd, Rn, #imm12` immediate form when it is a
    /// `ConstInt` in `[0, 4096)` (`check_imm_box`). Non-Int constants
    /// (ConstFloat/ConstPtr) and out-of-range ints fall through to
    /// the register form.
    fn prepare_int_ri(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let boxes = [lhs, rhs];
        // history.py — inline-Const variants carry the value on the OpRef.
        let imm_lhs = check_imm_box(lhs);
        let imm_rhs = check_imm_box(rhs);
        let (l0, l1) = if !imm_lhs && imm_rhs {
            let r0 = self.make_sure_var_in_reg(lhs, Type::Int, &boxes, None, false);
            let r1 = self.loc(rhs, Type::Int);
            (r0, r1)
        } else if imm_lhs && !imm_rhs {
            // PyPy regalloc.py:329-331 swaps so the immediate stays on
            // the Rn slot; aarch64 `add` is commutative.
            let r1 = self.loc(lhs, Type::Int);
            let r0 = self.make_sure_var_in_reg(rhs, Type::Int, &boxes, None, false);
            (r0, r1)
        } else {
            let r0 = self.make_sure_var_in_reg(lhs, Type::Int, &boxes, None, false);
            let r1 = self.make_sure_var_in_reg(rhs, Type::Int, &boxes, None, false);
            (r0, r1)
        };
        self.possibly_free_var(lhs, Type::Int);
        self.possibly_free_var(rhs, Type::Int);
        let res = self.force_allocate_reg(dst, Type::Int, &[], None, false);
        self.perform(i, [l0, l1], Some(Loc::Reg(res)), asm);
    }
}

/// aarch64/regalloc.py `Regalloc` — the `prepare_op_*` family and its
/// `regalloc_operations` dispatch.
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
        //
        // Strict parity: upstream `rpython/jit/backend/x86/reghint.py`
        // is x86-only; no backend/aarch64 counterpart exists, so the
        // pre-pass runs on x86_64 alone.  aarch64 loses
        // call-position caller-save blocking — a latent upstream bug
        // that pyre does not paper over.
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
            self._compute_hint_frame_locations_from_descr(target);
        }
    }

    /// aarch64/regalloc.py _compute_hint_frame_locations_from_descr: only
    /// Frame targets are hinted.
    pub(crate) fn _compute_hint_frame_locations_from_descr(
        &mut self,
        descr: &dyn majit_ir::descr::LoopTargetDescr,
    ) {
        use majit_ir::descr::TargetArgLoc;
        let Some((_, jump_args)) = self.final_jump_args.as_ref() else {
            return;
        };
        let target_arglocs = descr.target_arglocs();
        // aarch64/regalloc.py `_compute_hint_frame_locations_from_descr`:
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
                _ => {}
            }
        }
    }

    /// aarch64/regalloc.py make_sure_var_in_reg
    pub fn make_sure_var_in_reg(
        &mut self,
        v: OpRef,
        tp: Type,
        forbidden_vars: &[OpRef],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
    ) -> Loc {
        // aarch64/regalloc.py `Regalloc.make_sure_var_in_reg` has no
        // ConstFloat short-circuit: `ARMRegisterManager.return_constant` loads
        // `VFPRegisterManager.convert_to_imm`'s `ConstFloatLoc` into a VFP
        // scratch (`AssemblerARM64.load`). The walk installs the data block
        // on `xrm`, so that immediate is `Loc::ConstFloat`.
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
            LirOp::Label { args } => self.prepare_op_label(args, op, i, asm),
            LirOp::Jump { args } => self.prepare_op_jump(args, op, i, asm),
            LirOp::Finish { args } => self.prepare_op_finish(args, i, asm),
            LirOp::IntBin {
                kind,
                dst,
                lhs,
                rhs,
            } => self._dispatch_j2_int_bin(*kind, *dst, *lhs, *rhs, i, asm),
            LirOp::IntUnary { kind, dst, arg } => {
                self._dispatch_j2_int_unary(*kind, *dst, *arg, i, asm)
            }
            LirOp::IntCmp { dst, lhs, rhs, .. } => self.prepare_int_cmp(*dst, *lhs, *rhs, i, asm),
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
            IntBinKind::Add => self.prepare_op_int_add(dst, lhs, rhs, i, asm),
            IntBinKind::Sub => self.prepare_op_int_sub(dst, lhs, rhs, i, asm),
            // `prepare_comp_op_int_add_ovf = prepare_int_ri` and
            // `prepare_comp_op_int_sub_ovf = prepare_op_int_sub`: `adds` /
            // `subs` accept `#imm12`.
            IntBinKind::AddOvf => self.prepare_int_ri(dst, lhs, rhs, i, asm),
            IntBinKind::SubOvf => self.prepare_op_int_sub(dst, lhs, rhs, i, asm),
            IntBinKind::Mul
            | IntBinKind::MulOvf
            | IntBinKind::And
            | IntBinKind::Or
            | IntBinKind::Xor
            | IntBinKind::LShift
            | IntBinKind::RShift
            | IntBinKind::URShift => self.prepare_op_int_mul(dst, lhs, rhs, i, asm),
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
            IntUnaryKind::Neg | IntUnaryKind::Invert => self.prepare_unary(dst, arg, i, asm),
            // `prepare_op_int_is_true = prepare_unary` /
            // `prepare_comp_op_int_is_true = prepare_comp_unary`.
            IntUnaryKind::IsTrue | IntUnaryKind::IsZero => {
                self.prepare_comp_unary(dst, arg, i, asm)
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
                    self.guard_no_cc_impl(arg, op, i, asm);
                }
            }
            GuardKind::Value => {
                if args.len() >= 2 {
                    self.prepare_op_guard_value(args[0], args[1], op, i, asm);
                }
            }
            GuardKind::Class | GuardKind::GcType | GuardKind::NonNullClass => {
                if args.len() >= 2 {
                    self.prepare_op_guard_class(args[0], args[1], op, i, asm);
                }
            }
            GuardKind::Subclass => {
                if args.len() >= 2 {
                    self.prepare_op_guard_subclass(args[0], args[1], op, i, asm);
                }
            }
            GuardKind::IsObject => {
                if let Some(&arg) = args.first() {
                    self.prepare_op_guard_is_object(arg, op, i, asm);
                }
            }
            GuardKind::NotForced if op.opcode == OpCode::GuardNotForced2 => {
                self.prepare_op_guard_not_forced_2(op, i, asm)
            }
            GuardKind::NoException
            | GuardKind::NoOverflow
            | GuardKind::Overflow
            | GuardKind::NotInvalidated
            | GuardKind::FutureCondition
            | GuardKind::NotForced
            | GuardKind::AlwaysFails => self._guard_impl(op, i, asm),
            GuardKind::Exception => self.prepare_op_guard_exception(args, op, i, asm),
            GuardKind::Other(_) => self._guard_impl(op, i, asm),
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
                    self._prepare_op_gc_load(dst, base, offset, size, op, i, asm);
                }
            }
            LoadKind::GcIndexed => {
                if let (Some(index), Some(scale), Some(offset), Some(size)) =
                    (index, scale, offset, size)
                {
                    self._prepare_op_gc_load_indexed(
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
                    self.prepare_op_gc_store(base, offset, value, size, i, asm);
                }
            }
            StoreKind::GcIndexed => {
                if let (Some(index), Some(scale), Some(offset), Some(size)) =
                    (index, scale, offset, size)
                {
                    self.prepare_op_gc_store_indexed(
                        base, index, value, scale, offset, size, i, asm,
                    );
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
                    self._prepare_op_math_sqrt(dst, src, i, asm);
                } else {
                    self._prepare_call(dst, args, op, i, asm, opcode.is_call_may_force(), 1);
                }
            }
            OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN => {
                self._prepare_call(dst, args, op, i, asm, true, 2);
            }
            OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                self.prepare_op_call_assembler(dst, args, op, i, asm);
            }
            OpCode::CondCallN => self.prepare_op_cond_call(args, i, asm),
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                self.prepare_op_cond_call_value_i(dst, args, op, i, asm);
            }
            OpCode::CallMallocNursery | OpCode::CallMallocNurseryHeaderless => {
                self.prepare_op_call_malloc_nursery(dst, args, i, asm)
            }
            OpCode::CallMallocNurseryVarsizeFrame => {
                self.prepare_op_call_malloc_nursery_varsize_frame(dst, args, i, asm);
            }
            OpCode::CallMallocNurseryVarsize | OpCode::CallMallocNurseryVarsizeHeaderless => {
                self.prepare_op_call_malloc_nursery_varsize(dst, args, i, asm);
            }
            OpCode::CheckMemoryError => self.prepare_op_check_memory_error(args, i, asm),
            OpCode::New
            | OpCode::NewWithVtable
            | OpCode::NewArray
            | OpCode::NewArrayClear
            | OpCode::Newstr
            | OpCode::Newunicode => {
                self.prepare_raw_call_like(dst, args, op, i, asm, SAVE_DEFAULT_REGS);
            }
            _ => self.notimplemented(opcode),
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
                self.prepare_op_int_add(dst.unwrap_or(op.pos().get()), args[0], args[1], i, asm);
            }
            OpCode::IntForceGeZero if !args.is_empty() => {
                self.prepare_op_int_force_ge_zero(dst.unwrap_or(op.pos().get()), args[0], i, asm);
            }
            OpCode::IntFloorDiv | OpCode::IntMod if args.len() >= 2 => {
                self.prepare_op_int_mul(dst.unwrap_or(op.pos().get()), args[0], args[1], i, asm);
            }
            OpCode::UintMulHigh if args.len() >= 2 => {
                self.prepare_op_int_mul(dst.unwrap_or(op.pos().get()), args[0], args[1], i, asm);
            }
            OpCode::IntSignext if args.len() >= 2 => {
                self.prepare_op_int_signext(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    args[1],
                    i,
                    asm,
                );
            }
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv
                if args.len() >= 2 =>
            {
                self.prepare_two_regs_op(dst.unwrap_or(op.pos().get()), args[0], args[1], i, asm);
            }
            OpCode::FloatNeg | OpCode::FloatAbs if !args.is_empty() => {
                self.prepare_op_float_neg(dst.unwrap_or(op.pos().get()), args[0], i, asm);
            }
            // aarch64/regalloc.py has no `prepare_op_cast_float_to_singlefloat`
            // or `prepare_op_cast_singlefloat_to_float`: the AArch64 CPU keeps
            // model.py's `supports_singlefloats = False`, so the codewriter
            // never emits either cast and the dispatch table's
            // `notimplemented` entry raises.
            OpCode::CastFloatToSinglefloat | OpCode::CastSinglefloatToFloat => {
                panic!("{:?}: aarch64 does not support singlefloats", op.opcode)
            }
            OpCode::FloatLt
            | OpCode::FloatLe
            | OpCode::FloatEq
            | OpCode::FloatNe
            | OpCode::FloatGt
            | OpCode::FloatGe
                if args.len() >= 2 =>
            {
                self.prepare_float_op(dst.unwrap_or(op.pos().get()), args[0], args[1], i, asm);
            }
            OpCode::CastIntToFloat if !args.is_empty() => {
                self.prepare_op_cast_int_to_float(dst.unwrap_or(op.pos().get()), args[0], i, asm);
            }
            OpCode::CastFloatToInt if !args.is_empty() => {
                self.prepare_op_cast_float_to_int(dst.unwrap_or(op.pos().get()), args[0], i, asm);
            }
            // Same bank split as the non-j2 arm — see the citation there.
            OpCode::ConvertFloatBytesToLonglong if !args.is_empty() => {
                self.prepare_op_cast_float_to_int(dst.unwrap_or(op.pos().get()), args[0], i, asm);
            }
            OpCode::ConvertLonglongBytesToFloat if !args.is_empty() => {
                self.prepare_op_cast_int_to_float(dst.unwrap_or(op.pos().get()), args[0], i, asm);
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
                self._prepare_op_same_as(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    op.opcode.result_type(),
                    i,
                    asm,
                );
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
                self.prepare_op_getfield(dst.unwrap_or(op.pos().get()), args[0], op, i, asm);
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
                self.prepare_op_getarrayitem(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    args[1],
                    op,
                    i,
                    asm,
                );
            }
            OpCode::SetfieldGc | OpCode::SetfieldRaw if args.len() >= 2 => {
                self.prepare_op_setfield(args[0], args[1], i, asm);
            }
            OpCode::SetarrayitemGc
            | OpCode::SetarrayitemRaw
            | OpCode::SetinteriorfieldGc
            | OpCode::SetinteriorfieldRaw
                if args.len() >= 3 =>
            {
                self.prepare_op_setarrayitem(args[0], args[1], args[2], i, asm);
            }
            OpCode::ZeroArray if args.len() >= 3 => {
                self.prepare_op_zero_array(args, op, i, asm);
            }
            OpCode::Strsetitem | OpCode::Unicodesetitem if args.len() >= 3 => {
                self.prepare_op_strsetitem(args, i, asm);
            }
            OpCode::Copystrcontent | OpCode::Copyunicodecontent => {
                self.prepare_op_copystrcontent(args, i, asm);
            }
            OpCode::New
            | OpCode::NewWithVtable
            | OpCode::NewArray
            | OpCode::NewArrayClear
            | OpCode::Newstr
            | OpCode::Newunicode => {
                self.prepare_raw_call_like(dst, args, op, i, asm, SAVE_DEFAULT_REGS);
            }
            OpCode::ForceToken => {
                self.prepare_op_force_token(dst.unwrap_or(op.pos().get()), i, asm)
            }
            OpCode::LoadEffectiveAddress if args.len() >= 4 => {
                self.prepare_op_load_effective_address(dst.unwrap_or(op.pos().get()), args, i, asm);
            }
            OpCode::SaveException | OpCode::SaveExcClass => {
                self.prepare_op_save_exception(dst.unwrap_or(op.pos().get()), op, i, asm);
            }
            OpCode::RestoreException if args.len() >= 2 => {
                self.prepare_op_restore_exception(args, i, asm);
            }
            OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                self.prepare_op_cond_call_gc_wb(args, i, asm);
            }
            // aarch64/regalloc.py `prepare_op_jit_debug = void`,
            // `prepare_op_keepalive = void`.
            OpCode::JitDebug | OpCode::Keepalive => {}
            // `prepare_op_enter_portal_frame` / `prepare_op_leave_portal_frame`
            // hand the op to llsupport/assembler.py `enter_portal_frame` /
            // `leave_portal_frame`, which only feed the codemap
            // (`HAS_CODEMAP`); dynasm keeps no codemap.
            OpCode::EnterPortalFrame | OpCode::LeavePortalFrame => {}
            OpCode::IncrementDebugCounter => self.prepare_op_increment_debug_counter(args, i, asm),
            // aarch64/assembler.py `_walk_operations`: `FORCE_SPILL` (tests
            // only) goes straight to `regalloc.force_spill_var`.
            OpCode::ForceSpill => {
                let v = args[0];
                let tp = self.tp(v);
                self.force_spill_var(v, tp);
            }
            _ => self.notimplemented(opcode),
        }
    }

    /// aarch64/regalloc.py `prepare_op_increment_debug_counter`.
    pub(crate) fn prepare_op_increment_debug_counter(
        &mut self,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let a0 = args[0];
        let base_loc = self.make_sure_var_in_reg(a0, Type::Int, args, None, false);
        // `get_scratch_reg(INT, boxes)`.
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        let value_loc = Loc::Reg(self.rm.force_allocate_reg(
            tmp,
            args,
            None,
            false,
            &mut self.longevity,
            &mut self.fm,
        ));
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        self.perform_discard(i, vec![base_loc, value_loc], asm);
    }

    /// aarch64/regalloc.py `notimplemented`, the `operations` default.
    /// `compile_loop` catches the panic and declines the trace, as
    /// `NotImplementedError` does upstream.
    pub(crate) fn notimplemented(&self, opcode: OpCode) -> ! {
        panic!("[ARM64/regalloc] {opcode:?} not implemented")
    }

    pub(crate) fn prepare_int_cmp(
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
    pub(crate) fn prepare_op_int_force_ge_zero(
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

    pub(crate) fn prepare_op_int_signext(
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

    pub(crate) fn guard_no_cc_impl(
        &mut self,
        arg: OpRef,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        // aarch64/regalloc.py `guard_no_cc_impl`: the argument in a register,
        // then `_guard_impl`'s one walk of `getfailargs()`.
        let loc = self.make_sure_var_in_reg(arg, self.tp(arg), &[], None, false);
        self.perform_guard(op, i, [loc], None, asm);
    }

    /// aarch64/regalloc.py `prepare_op_guard_value`: arg in a register.
    /// A ConstFloat is loaded by `ARMRegisterManager.return_constant`
    /// (`FCMP_dd` has no immediate form).
    pub(crate) fn prepare_guard_value_arg(&mut self, v: OpRef, forbidden: &[OpRef]) -> Loc {
        if v.is_constant() && self.tp(v) == Type::Float {
            self.const_float_in_reg(v, forbidden)
        } else {
            self.make_sure_var_in_reg(v, self.tp(v), forbidden, None, false)
        }
    }

    /// aarch64/regalloc.py `prepare_op_guard_value`.
    pub(crate) fn guard_value_locs(&mut self, lhs: OpRef, rhs: OpRef) -> (Loc, Loc) {
        let arg = self.prepare_guard_value_arg(lhs, &[]);
        let l1 = if check_imm_box(rhs) {
            self.loc(rhs, self.tp(rhs))
        } else {
            self.prepare_guard_value_arg(rhs, &[lhs])
        };
        (arg, l1)
    }

    pub(crate) fn prepare_op_guard_value(
        &mut self,
        lhs: OpRef,
        rhs: OpRef,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let (x, y) = self.guard_value_locs(lhs, rhs);
        self.perform_guard(op, i, vec![x, y], None, asm);
        // `get_scratch_reg` temps die with this instruction.
        self.xrm.free_temp_vars(&mut self.longevity, &mut self.fm);
    }

    pub(crate) fn prepare_op_guard_class(
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

    pub(crate) fn prepare_op_guard_is_object(
        &mut self,
        value: OpRef,
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let x = self.make_sure_var_in_reg(value, Type::Ref, &[], None, false);
        self.perform_guard(op, i, [x], None, asm);
    }

    pub(crate) fn prepare_op_guard_subclass(
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

    pub(crate) fn prepare_op_guard_exception(
        &mut self,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let Some(&exception_class) = args.first() else {
            return self._guard_impl(op, i, asm);
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

    pub(crate) fn _guard_impl(&mut self, op: &Op, i: usize, asm: &mut dyn RegAllocAssembler) {
        self.perform_guard(op, i, vec![], None, asm);
    }

    /// aarch64/regalloc.py `prepare_op_guard_not_forced_2`.
    pub(crate) fn prepare_op_guard_not_forced_2(
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

    pub(crate) fn prepare_op_restore_exception(
        &mut self,
        args: &[OpRef],
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let loc0 = self.make_sure_var_in_reg(args[0], Type::Ref, args, None, false);
        let loc1 = self.make_sure_var_in_reg(args[1], Type::Ref, args, None, false);
        self.perform_discard(i, vec![loc0, loc1], asm);
    }

    pub(crate) fn prepare_op_check_memory_error(
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

    pub(crate) fn prepare_op_finish(
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

    pub(crate) fn _prepare_op_same_as(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        tp: Type,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let args = [arg];
        let argloc = if arg.is_constant() {
            self.loc(arg, self.tp(arg))
        } else {
            self.make_sure_var_in_reg(arg, tp, &args, None, false)
        };
        if !arg.is_constant() && !arg.is_none() {
            self.possibly_free_var(arg, tp);
        }
        self.rm.free_temp_vars(&mut self.longevity, &mut self.fm);
        self.xrm.free_temp_vars(&mut self.longevity, &mut self.fm);
        let resloc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        self.perform(i, [argloc], Some(resloc), asm);
    }

    pub(crate) fn prepare_two_regs_op(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        // aarch64/regalloc.py `prepare_two_regs_op`: `fadd/fmul Dd, Dn, Dm`
        // writes a third register. The first `make_sure_var_in_reg` takes no
        // forbidden vars; the second receives `op.getarglist()`.
        let boxes = [lhs, rhs];
        let loc0 = self.make_sure_var_in_reg(lhs, Type::Float, &[], None, false);
        let loc1 = self.make_sure_var_in_reg(rhs, Type::Float, &boxes, None, false);
        self.possibly_free_var(lhs, Type::Float);
        self.possibly_free_var(rhs, Type::Float);
        // aarch64/regalloc.py `Regalloc.free_temp_vars` / `prepare_two_regs_op`
        self.rm.free_temp_vars(&mut self.longevity, &mut self.fm);
        self.xrm.free_temp_vars(&mut self.longevity, &mut self.fm);
        let res = self.force_allocate_reg(dst, Type::Float, &[], None, false);
        self.perform(i, [loc0, loc1], Some(Loc::Reg(res)), asm);
    }

    pub(crate) fn _prepare_op_math_sqrt(
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

    pub(crate) fn prepare_op_float_neg(
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

    pub(crate) fn prepare_float_op(
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

    pub(crate) fn prepare_op_cast_int_to_float(
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

    pub(crate) fn prepare_op_cast_float_to_int(
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

    pub(crate) fn prepare_op_getfield(
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

    pub(crate) fn prepare_op_getarrayitem(
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
        // aarch64/regalloc.py `check_imm_arg` (`0..DEFAULT_IMM_SIZE`).
        if check_imm_arg(ofs) {
            Loc::immed(ofs)
        } else {
            self.pending_moves
                .push((Loc::immed(ofs), Loc::Reg(registers::IP1)));
            Loc::Reg(registers::IP1)
        }
    }

    pub(crate) fn _prepare_op_gc_load(
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

    /// aarch64/regalloc.py _prepare_op_gc_load_indexed parity, j2plan path.
    pub(crate) fn _prepare_op_gc_load_indexed(
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
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let index_loc = self.make_sure_var_in_reg(index, Type::Int, &args, None, false);
        let scale_value = self.const_value(scale);
        assert_eq!(
            scale_value, 1,
            "aarch64 GcLoadIndexed requires factor == 1 (got {scale_value})"
        );
        let ofs = self.const_value(offset);
        let nsize = self.const_value(size);
        let tp = op.opcode.result_type();
        let res_loc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        self.perform(
            i,
            vec![
                res_loc,
                base_loc,
                index_loc,
                Loc::immed(nsize),
                Loc::immed(ofs),
            ],
            Some(res_loc),
            asm,
        );
    }

    pub(crate) fn prepare_op_setfield(
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

    pub(crate) fn prepare_op_setarrayitem(
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

    pub(crate) fn prepare_op_gc_store(
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
        // stores at a small negative offset, within STUR's signed 9-bit
        // immediate range.
        let ofs_loc = if (-256..0).contains(&ofs) {
            Loc::immed(ofs)
        } else {
            self.gc_offset_loc(ofs)
        };
        self.perform_discard_gc_store(i, value_loc, base_loc, ofs_loc, size, asm);
    }

    /// aarch64/regalloc.py prepare_op_gc_store_indexed parity.
    pub(crate) fn prepare_op_gc_store_indexed(
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
        let tp_val = self.tp(value);
        let value_loc = self.make_sure_var_in_reg(value, tp_val, &args, None, false);
        let index_loc = self.make_sure_var_in_reg(index, Type::Int, &args, None, false);
        let scale_value = self.const_value(scale);
        assert_eq!(
            scale_value, 1,
            "aarch64 GcStoreIndexed requires factor == 1 (got {scale_value})"
        );
        let ofs = self.const_value(offset);
        let size = self.const_value(size);
        self.perform_discard(
            i,
            vec![
                value_loc,
                base_loc,
                index_loc,
                Loc::immed(size),
                Loc::immed(ofs),
            ],
            asm,
        );
    }

    pub(crate) fn _prepare_call(
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

    pub(crate) fn prepare_op_call_assembler(
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

    pub(crate) fn prepare_raw_call_like(
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

    pub(crate) fn prepare_op_call_malloc_nursery(
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
        // aarch64/regalloc.py:965-966: `t = TempInt()`; reserve the second
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

    pub(crate) fn prepare_op_call_malloc_nursery_varsize_frame(
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
        // aarch64/regalloc.py t = TempInt()
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

    /// aarch64/regalloc.py `prepare_op_call_malloc_nursery_varsize`.
    pub(crate) fn prepare_op_call_malloc_nursery_varsize(
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
                // The emitter still expects a live length in its frame slot,
                // so every binding is spilled.
                SAVE_ALL_REGS,
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

    pub(crate) fn prepare_op_jump(
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

    pub(crate) fn prepare_op_label(
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

        let mut locs = Vec::new();
        for &arg in args {
            let tp = self.tp(arg);
            let loc = self.loc(arg, tp);
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
            for (iarg, target_loc) in jump_args.iter().zip(locs.iter()) {
                if iarg.is_constant() {
                    continue;
                }
                match *target_loc {
                    Loc::Frame(frame_loc) => {
                        self.fm
                            .add_frame_pos_hint(*iarg, frame_loc, &mut self.longevity);
                    }
                    _ => {}
                }
            }
        }
        self.perform(i, locs, None, asm);
    }

    pub(crate) fn prepare_op_force_token(
        &mut self,
        dst: OpRef,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, Type::Ref, &[], None, false));
        self.perform(i, [] as [Loc; 0], Some(result_loc), asm);
    }

    pub(crate) fn prepare_op_load_effective_address(
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

    pub(crate) fn prepare_op_save_exception(
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

    pub(crate) fn prepare_op_strsetitem(
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

    /// Select the inline-store or residual-memset ZERO_ARRAY path using the
    /// same per-backend limits as upstream.  x86/regalloc.py consider_zero_array does
    /// this split in regalloc; aarch64/opassembler.py emit_op_zero_array does it while
    /// emitting, but the residual call still needs `before_call` here so only
    /// live caller-saved registers are preserved.
    pub(crate) fn zero_array_uses_memset(&self, op: &Op) -> bool {
        let size = op.arg(2).to_opref();
        if !size.is_constant() {
            return true;
        }
        let mut size = self.const_value(size);
        let scale = op.arg(4).to_opref();
        if !scale.is_constant() {
            return true;
        }
        size *= self.const_value(scale);

        // aarch64/opassembler.py:796-812 — choose the natural STR width
        // from the array item size.  A constant start permits aligned
        // runs of eight-byte STRs even for byte/halfword/word arrays.
        let item_size = op.with_array_descr(|ad| ad.item_size() as i64).unwrap_or(8);
        let item_size = if item_size & 1 != 0 {
            1
        } else if item_size & 2 != 0 {
            2
        } else if item_size & 4 != 0 {
            4
        } else {
            8
        };
        let start_is_constant = op.arg(1).to_opref().is_constant();
        let limit = if item_size < 8 && start_is_constant {
            8
        } else {
            item_size
        };
        size < 0 || size > 14 * limit
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

    pub(crate) fn prepare_op_zero_array(
        &mut self,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let mut locs = Vec::with_capacity(args.len());
        for &arg in args {
            locs.push(self.make_sure_var_in_reg(arg, self.tp(arg), args, None, false));
        }
        if self.zero_array_uses_memset(op) {
            self.preserve_for_zero_array_memset();
        }
        self.perform_discard(i, locs, asm);
    }

    /// aarch64/regalloc.py `prepare_op_cond_call_value_i`
    /// (`prepare_op_cond_call_value_r` is the same) → `_prepare_op_cond_call`.
    ///
    /// The result shares args[0]'s register via `force_result_in_reg`
    /// (`forbidden_vars=args[2:]`). The callee address stays an Immed on
    /// the op. The helper returns a plain word — no `before_call` / raw-call
    /// result-in-x0 rewrite.
    pub(crate) fn prepare_op_cond_call_value_i(
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
            "COND_CALL_VALUE takes 0..=4 extra arguments (aarch64/regalloc.py _prepare_op_cond_call)"
        );
        self._prepare_op_cond_call(dst, args, op, i, asm);
    }

    /// aarch64/regalloc.py `_prepare_op_cond_call`: extra args (from index 2)
    /// go into `argument_regs`. Arglocs are `[argloc, resloc]`.
    pub(crate) fn _prepare_op_cond_call(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        op: &Op,
        i: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        let argument_regs = cond_call_argument_regs();
        let mut args_so_far: Vec<OpRef> = Vec::new();
        for (j, &arg) in args.iter().enumerate().skip(2) {
            let reg = argument_regs[j - 2];
            self.make_sure_var_in_reg(arg, self.tp(arg), &args_so_far, Some(reg), false);
            args_so_far.push(arg);
        }
        let value = args[0];
        let argloc = self.make_sure_var_in_reg(value, self.tp(value), &args_so_far, None, false);
        let result_v = dst.unwrap_or(op.pos().get());
        let forbidden: Vec<OpRef> = args.iter().skip(2).copied().collect();
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
                "force_result_in_reg must return a register (aarch64/regalloc.py _prepare_op_cond_call), got {other:?}"
            ),
        };
        // aarch64/opassembler.py `_emit_op_cond_call`:
        // `gcmap = self._regalloc.get_gcmap([res_loc])` — exclude the result
        // register; on the miss path it is overwritten with the helper word.
        let gcmap = self.get_gcmap(&[res_reg], false) as usize;
        let locs = [argloc, resloc];
        self.perform_with_gcmap_ptr(i, locs, Some(resloc), gcmap, asm);
    }

    /// aarch64/regalloc.py `prepare_op_cond_call` and
    /// `aarch64/opassembler.py _emit_op_cond_call`:
    /// `gcmap = self._regalloc.get_gcmap([res_loc])`. CondCallN has no
    /// result. aarch64 spills every register into the jitframe on the
    /// taken path and publishes this map so `force_now` allocations
    /// forward those slots.
    pub(crate) fn prepare_op_cond_call(
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

    pub(crate) fn prepare_op_copystrcontent(
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

    pub(crate) fn prepare_op_cond_call_gc_wb(
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
