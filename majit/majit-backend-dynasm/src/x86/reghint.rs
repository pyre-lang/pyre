//! Port of `rpython/jit/backend/x86/reghint.py`.
//!
//! Register hints that tell the regalloc which registers to prefer
//! for specific opcodes.  Runs as a pre-pass before the main regalloc
//! walk (invoked from `RegAlloc::_prepare`).

use crate::regalloc::{LifetimeManager, SAVE_ALL_REGS, SAVE_DEFAULT_REGS, SAVE_GCREF_REGS};
use crate::regloc::{EAX, ECX, EDX, RegLoc};
use crate::x86::callbuilder::{ARGUMENTS_GPR, ARGUMENTS_XMM};
use indexmap::IndexMap;
use majit_ir::operand::Operand;
use majit_ir::{Op, OpCode, OpRef, Type};

/// reghint.py X86RegisterHints.
pub struct RegisterHints {
    save_around_call_regs_gpr: Vec<RegLoc>,
    all_regs_gpr: Vec<RegLoc>,
    all_regs_xmm: Vec<RegLoc>,
    /// Rust adaptation: RPython's `ConstInt.value` is embedded on the
    /// Box object; pyre's `OpRef` is a flat `u32` so constant values
    /// are looked up through this map.  Structural equivalent of
    /// `isinstance(arg, ConstInt) and arg.value`.
    constants: IndexMap<u32, i64>,
}

impl RegisterHints {
    pub fn new(
        save_around_call_regs_gpr: Vec<RegLoc>,
        _save_around_call_regs_xmm: Vec<RegLoc>,
        all_regs_gpr: Vec<RegLoc>,
        all_regs_xmm: Vec<RegLoc>,
        constants: IndexMap<u32, i64>,
    ) -> Self {
        RegisterHints {
            save_around_call_regs_gpr,
            all_regs_gpr,
            all_regs_xmm,
            constants,
        }
    }

    /// Rust equivalent of `isinstance(arg, ConstInt) and arg.value`
    /// — returns the const int value if available.
    fn get_const_int(&self, arg: OpRef) -> Option<i64> {
        if !arg.is_constant() {
            return None;
        }
        // history.py/268/314 — inline-Const variants carry value inline.
        arg.inline_const_bits()
            .or_else(|| self.constants.get(&arg.raw()).copied())
    }

    /// reghint.py `add_hints` — main entry called from
    /// `RegAlloc::_prepare`.
    pub fn add_hints(
        &self,
        longevity: &mut LifetimeManager,
        _inputargs: &[majit_ir::InputArgRc],
        operations: &[impl AsRef<Op>],
    ) {
        for (i, op) in operations.iter().enumerate() {
            let op = op.as_ref();
            let position = i as i32;
            // reghint.py:34-35 skip dead no-side-effect ops.
            if op.opcode.has_no_side_effect() && !longevity.contains(op.pos().get()) {
                continue;
            }
            self.dispatch(op.opcode, longevity, op, position);
        }
    }

    /// reghint.py `oplist` dispatch.  Implemented as a match
    /// here because Rust cannot build a class-method oplist at
    /// import time the way Python can.
    fn dispatch(&self, opcode: OpCode, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        match opcode {
            // reghint.py consider_int_neg
            OpCode::IntNeg | OpCode::IntInvert => {
                self.consider_int_neg(longevity, op, position);
            }
            // reghint.py _consider_binop_symm
            OpCode::IntMul
            | OpCode::IntAnd
            | OpCode::IntOr
            | OpCode::IntXor
            | OpCode::IntMulOvf
            | OpCode::IntAddOvf => {
                self._consider_binop_symm(longevity, op, position);
            }
            // reghint.py _consider_binop
            OpCode::IntSubOvf => {
                self._consider_binop(longevity, op, position);
            }
            // reghint.py consider_int_add + consider_nursery_ptr_increment
            OpCode::IntAdd | OpCode::NurseryPtrIncrement => {
                self.consider_int_add(longevity, op, position);
            }
            // reghint.py consider_int_sub
            OpCode::IntSub => {
                self.consider_int_sub(longevity, op, position);
            }
            // reghint.py _consider_float_op
            OpCode::FloatAdd
            | OpCode::FloatSub
            | OpCode::FloatMul
            | OpCode::FloatTrueDiv
            | OpCode::FloatNeg
            | OpCode::FloatAbs => {
                self._consider_float_op(longevity, op, position);
            }
            // reghint.py consider_int_lshift / rshift / urshift
            OpCode::IntLshift | OpCode::IntRshift | OpCode::UintRshift => {
                self.consider_int_lshift(longevity, op, position);
            }
            // reghint.py consider_uint_mul_high
            OpCode::UintMulHigh => {
                self.consider_uint_mul_high(longevity, position);
            }
            // reghint.py consider_call_malloc_nursery + varsize variants
            OpCode::CallMallocNursery
            | OpCode::CallMallocNurseryHeaderless
            | OpCode::CallMallocNurseryVarsize
            | OpCode::CallMallocNurseryVarsizeHeaderless
            | OpCode::CallMallocNurseryVarsizeFrame => {
                self.consider_call_malloc_nursery(longevity, op, position);
            }
            // reghint.py consider_call_{i,r,f,n} → _consider_real_call
            OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN => {
                self._consider_real_call(longevity, op, position);
            }
            // reghint.py:154-157 consider_call_may_force_*
            OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN => {
                self._consider_call(longevity, op, position, true, 1);
            }
            // reghint.py:162-164 consider_call_release_gil_*
            OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN => {
                self._consider_call(longevity, op, position, true, 2);
            }
            _ => {}
        }
    }

    /// reghint.py consider_int_neg / consider_int_invert.
    fn consider_int_neg(&self, longevity: &mut LifetimeManager, op: &Op, _position: i32) {
        longevity.try_use_same_register(op.arg(0).to_opref(), op.pos().get());
    }

    /// reghint.py `_consider_binop_part`.
    fn _consider_binop_part(
        &self,
        longevity: &mut LifetimeManager,
        op: &Op,
        position: i32,
        symm: bool,
    ) {
        let mut x = op.arg(0).to_opref();
        let mut y = op.arg(1).to_opref();

        // For symmetrical operations, if y won't be used after the
        // current operation finishes, but x will be, then swap the
        // role of 'x' and 'y'.
        if symm {
            if x.is_constant() {
                std::mem::swap(&mut x, &mut y);
            } else if !y.is_constant() {
                let x_last = longevity.get(x).map(|lt| lt.last_usage).unwrap_or(-1);
                let y_last = longevity.get(y).map(|lt| lt.last_usage).unwrap_or(-1);
                if x_last > position && y_last == position {
                    std::mem::swap(&mut x, &mut y);
                }
            }
        }

        if !x.is_constant() {
            longevity.try_use_same_register(x, op.pos().get());
        }
    }

    /// reghint.py `_consider_binop`.
    fn _consider_binop(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        self._consider_binop_part(longevity, op, position, false);
    }

    /// reghint.py `_consider_binop_symm`.
    fn _consider_binop_symm(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        self._consider_binop_part(longevity, op, position, true);
    }

    /// reghint.py `consider_int_add`.
    fn consider_int_add(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        let y = op.arg(1).to_opref();
        if let Some(val) = self.get_const_int(y) {
            if fits_in_32bits(val) {
                // nothing to be hinted
                return;
            }
        }
        self._consider_binop_symm(longevity, op, position);
    }

    /// reghint.py `consider_int_sub`.
    fn consider_int_sub(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        let y = op.arg(1).to_opref();
        if let Some(val) = self.get_const_int(y) {
            if fits_in_32bits(-val) {
                return;
            }
        }
        self._consider_binop(longevity, op, position);
    }

    /// reghint.py `_consider_float_op`.
    fn _consider_float_op(&self, longevity: &mut LifetimeManager, op: &Op, _position: i32) {
        let x = op.arg(0).to_opref();
        if !x.is_constant() {
            longevity.try_use_same_register(x, op.pos().get());
        }
    }

    /// reghint.py `consider_int_lshift` (shared with rshift /
    /// uint_rshift per aliases at reghint.py).
    ///
    /// Shift amount must go in ecx (x86 `shl`/`shr` only accept cl).
    /// Fix `ecx` for `y` at this position, and bias the result to
    /// share with `x` via try_use_same_register.
    fn consider_int_lshift(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        let x = op.arg(0).to_opref();
        let y = op.arg(1).to_opref();
        if !y.is_constant() {
            longevity.fixed_register(position, ECX, Some(y));
        }
        if !x.is_constant() {
            longevity.try_use_same_register(x, op.pos().get());
        }
    }

    /// reghint.py `consider_uint_mul_high`.
    ///
    /// 64x64→128 MUL on x86 returns low in rax, high in rdx.  Reserve
    /// both at this position so the result/temp aren't spilled over.
    fn consider_uint_mul_high(&self, longevity: &mut LifetimeManager, position: i32) {
        longevity.fixed_register(position, EAX, None);
        longevity.fixed_register(position, EDX, None);
    }

    /// reghint.py `consider_call_malloc_nursery` and its
    /// varsize/varsize_frame aliases — pins result to ecx and temp
    /// to edx, matching the regalloc.py:1013-1028 contract.
    fn consider_call_malloc_nursery(
        &self,
        longevity: &mut LifetimeManager,
        op: &Op,
        position: i32,
    ) {
        longevity.fixed_register(position, ECX, Some(op.pos().get()));
        longevity.fixed_register(position, EDX, None);
    }

    /// reghint.py `_consider_real_call`.
    fn _consider_real_call(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        let Some(descr) = op.getdescr() else {
            return;
        };
        let Some(calldescr) = descr.as_call_descr() else {
            return;
        };
        // reghint.py:142-144 skip oopspec helpers.
        if calldescr.get_extra_info().has_oopspec() {
            return;
        }
        self._consider_call(longevity, op, position, false, 1);
    }

    /// reghint.py `_consider_call`.
    fn _consider_call(
        &self,
        longevity: &mut LifetimeManager,
        op: &Op,
        position: i32,
        guard_not_forced: bool,
        first_arg_index: usize,
    ) {
        let Some(descr) = op.getdescr() else {
            return;
        };
        let Some(calldescr) = descr.as_call_descr() else {
            return;
        };
        let gc_level = compute_gc_level(calldescr, guard_not_forced);
        let arglist = op.getarglist();
        let args = &arglist[first_arg_index..];
        let argtypes = calldescr.arg_classes();
        self.hint(longevity, position, args, argtypes.as_bytes(), gc_level);
    }

    /// reghint.py `CallHints64.hint`.
    fn hint(
        &self,
        longevity: &mut LifetimeManager,
        position: i32,
        args: &[Operand],
        argtypes: &[u8],
        save_all_regs: u8,
    ) {
        let mut hinted_xmm: Vec<RegLoc> = Vec::new();
        let mut hinted_gpr: Vec<RegLoc> = Vec::new();
        let mut hinted_args: Vec<OpRef> = Vec::new();
        let mut next_arg_gpr = 0usize;
        let mut next_arg_xmm = 0usize;

        for (i, operand) in args.iter().enumerate() {
            let arg = operand.to_opref();
            if operand.type_() == Type::Float || argtypes.get(i) == Some(&b'S') {
                if next_arg_xmm < ARGUMENTS_XMM.len() {
                    let tgt = ARGUMENTS_XMM[next_arg_xmm];
                    if !arg.is_constant() && !hinted_args.contains(&arg) {
                        longevity.fixed_register(position, tgt, Some(arg));
                        hinted_xmm.push(tgt);
                        hinted_args.push(arg);
                    }
                    next_arg_xmm += 1;
                }
            } else if next_arg_gpr < ARGUMENTS_GPR.len() {
                let tgt = ARGUMENTS_GPR[next_arg_gpr];
                if !arg.is_constant() && !hinted_args.contains(&arg) {
                    longevity.fixed_register(position, tgt, Some(arg));
                    hinted_gpr.push(tgt);
                    hinted_args.push(arg);
                }
                next_arg_gpr += 1;
            }
        }
        self._block_non_caller_save(longevity, position, save_all_regs, &hinted_gpr, &hinted_xmm);
    }

    /// reghint.py `_block_non_caller_save`.
    fn _block_non_caller_save(
        &self,
        longevity: &mut LifetimeManager,
        position: i32,
        save_all_regs: u8,
        hinted_gpr: &[RegLoc],
        hinted_xmm: &[RegLoc],
    ) {
        // reghint.py:184-188 GPR selection.
        let gpr_regs: &[RegLoc] = if save_all_regs == SAVE_ALL_REGS {
            &self.all_regs_gpr
        } else {
            &self.save_around_call_regs_gpr
        };
        for &reg in gpr_regs {
            if !hinted_gpr.contains(&reg) {
                longevity.fixed_register(position, reg, None);
            }
        }
        // reghint.py:192-194 XMM: every x86-64 XMM reg is caller-save.
        for &reg in &self.all_regs_xmm {
            if !hinted_xmm.contains(&reg) {
                longevity.fixed_register(position, reg, None);
            }
        }
    }
}

/// x86/regalloc.py `compute_gc_level`.
fn compute_gc_level(calldescr: &dyn majit_ir::descr::CallDescr, guard_not_forced: bool) -> u8 {
    if guard_not_forced {
        return SAVE_ALL_REGS;
    }
    if !calldescr.get_extra_info().check_can_collect() {
        return SAVE_DEFAULT_REGS;
    }
    SAVE_GCREF_REGS
}

/// x86/rx86.py `fits_in_32bits`.  x86 ADD/SUB imm32 encoding accepts
/// signed 32-bit immediates.
fn fits_in_32bits(value: i64) -> bool {
    (i32::MIN as i64) <= value && value <= (i32::MAX as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regalloc::compute_vars_longevity;
    use majit_ir::effectinfo::{EffectInfo, ExtraEffect, OopSpecIndex};
    use majit_ir::forwarding::bound_operand_from_opref as rb;
    use majit_ir::{InputArg, OpRc};

    /// `CallHints64.hint` sends an argument to XMM when `arg.type == 'f'`
    /// or `argtypes[i] == 'S'`; the singlefloat argument does not take a
    /// GPR slot, so the next integer argument still gets ARGUMENTS_GPR[0].
    #[test]
    fn call_hint_singlefloat_takes_xmm() {
        let f = OpRef::input_arg_int(0);
        let i = OpRef::input_arg_int(1);
        let inputargs = vec![InputArg::new_int_rc(0), InputArg::new_int_rc(1)];
        let mut effect = EffectInfo::new(ExtraEffect::CannotRaise, OopSpecIndex::None);
        effect.can_collect = false;
        let call = Op::new(OpCode::CallN, &[rb(OpRef::const_int(0x1000)), rb(f), rb(i)]);
        call.pos().set(OpRef::void_op(0));
        call.setdescr(majit_ir::descr::make_call_descr_full_with_classes(
            0,
            "Si".to_string(),
            vec![Type::Int, Type::Int],
            Type::Void,
            'v',
            false,
            0,
            effect,
        ));
        let finish = Op::new(OpCode::Finish, &[]);
        finish.pos().set(OpRef::void_op(1));
        let ops: Vec<OpRc> = vec![OpRc::new(call), OpRc::new(finish)];
        let mut longevity = compute_vars_longevity(&inputargs, &ops);
        RegisterHints::new(vec![], vec![], vec![], vec![], IndexMap::new()).add_hints(
            &mut longevity,
            &inputargs,
            &ops,
        );
        let fixed = |v: OpRef| longevity.get(v).unwrap().find_fixed_register(0, &longevity);
        assert_eq!(fixed(f), Some(ARGUMENTS_XMM[0]));
        // The separate GPR/XMM counters are the SysV numbering.
        #[cfg(not(target_os = "windows"))]
        assert_eq!(fixed(i), Some(ARGUMENTS_GPR[0]));
    }
}
