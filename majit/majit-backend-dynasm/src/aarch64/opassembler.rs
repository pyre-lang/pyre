/// aarch64/opassembler.py parity: arch-specific operation emission.
///
/// RPython separates instruction emission from assembly control flow:
/// - `llsupport/assembler.py` → common (BaseAssembler)
/// - `aarch64/opassembler.py` → arch-specific emit_op_* (ResOpAssembler)
/// - `aarch64/assembler.py` → control flow (AssemblerARM64)
///
/// This module contains aarch64-specific emit helpers called from
/// the regalloc-aware dispatch in assembler.rs.
use dynasmrt::{DynasmApi, dynasm};

use super::assembler::AssemblerARM64;
use crate::jump::RegallocMoves;
use crate::regloc::{Loc, RegLoc};

impl<'a> AssemblerARM64<'a> {
    // ── emit_op_setfield_gc ──
    // aarch64/opassembler.py: emit_op_setfield_gc
    // x86/assembler.py: genop_discard_setfield_gc

    pub(crate) fn emit_op_setfield_regalloc(
        &mut self,
        base: &RegLoc,
        val_loc: &Loc,
        ofs: i32,
        field_size: usize,
    ) {
        match val_loc {
            Loc::Reg(v) if v.is_xmm => {
                // `get_type_flag` banks only `lltype.Float` (8 bytes) as
                // FLAG_FLOAT. A singlefloat store is int-kind (`str W`).
                debug_assert_eq!(field_size, 8, "float-kind access is 8 bytes");
                dynasm!(self.mc ; .arch aarch64 ; str D(v.value), [X(base.value), ofs as u32]);
            }
            Loc::Reg(v) => match field_size {
                1 => {
                    dynasm!(self.mc ; .arch aarch64 ; strb W(v.value), [X(base.value), ofs as u32])
                }
                2 => {
                    dynasm!(self.mc ; .arch aarch64 ; strh W(v.value), [X(base.value), ofs as u32])
                }
                4 => dynasm!(self.mc ; .arch aarch64 ; str W(v.value), [X(base.value), ofs as u32]),
                _ => dynasm!(self.mc ; .arch aarch64 ; str X(v.value), [X(base.value), ofs as u32]),
            },
            _ => {
                let scratch = RegLoc::new(16, false);
                self.regalloc_mov(val_loc, &Loc::Reg(scratch));
                match field_size {
                    1 => {
                        dynasm!(self.mc ; .arch aarch64 ; strb W(scratch.value), [X(base.value), ofs as u32])
                    }
                    2 => {
                        dynasm!(self.mc ; .arch aarch64 ; strh W(scratch.value), [X(base.value), ofs as u32])
                    }
                    4 => {
                        dynasm!(self.mc ; .arch aarch64 ; str W(scratch.value), [X(base.value), ofs as u32])
                    }
                    _ => {
                        dynasm!(self.mc ; .arch aarch64 ; str X(scratch.value), [X(base.value), ofs as u32])
                    }
                }
            }
        }
    }

    // ── emit_op_gc_load ──
    // aarch64/opassembler.py emit_op_gc_store emit_op_gc_load_i/r/f
    // x86/assembler.py:1645 genop_gc_load_i — size_loc/sign_loc

    /// `size`: byte size (1/2/4/8). Negative = signed load (for ints).
    pub(crate) fn emit_op_gcload_regalloc(
        &mut self,
        base: &RegLoc,
        ofs_loc: &Loc,
        dst: &RegLoc,
        size: i64,
    ) {
        let abs_size = size.unsigned_abs() as usize;
        let signed = size < 0;
        match ofs_loc {
            Loc::Immed(i) | Loc::ImmedFloat(i) => {
                let o = i.value as i32;
                self.emit_load_sized(base, o, None, dst, abs_size, signed);
            }
            Loc::Reg(ofs_r) => {
                self.emit_load_sized(base, 0, Some(ofs_r), dst, abs_size, signed);
            }
            _ => {}
        }
    }

    /// Emit a sized load: [base + ofs] or [base + ofs_reg].
    fn emit_load_sized(
        &mut self,
        base: &RegLoc,
        ofs: i32,
        ofs_reg: Option<&RegLoc>,
        dst: &RegLoc,
        size: usize,
        signed: bool,
    ) {
        if dst.is_xmm {
            // `get_type_flag` banks only `lltype.Float` (8 bytes). A
            // singlefloat load is int-kind, not `ldr S` + `fcvt`.
            debug_assert_eq!(size, 8, "float-kind access is 8 bytes");
            if let Some(r) = ofs_reg {
                dynasm!(self.mc ; .arch aarch64 ; ldr D(dst.value), [X(base.value), X(r.value)]);
            } else {
                dynasm!(self.mc ; .arch aarch64 ; ldr D(dst.value), [X(base.value), ofs as u32]);
            }
            return;
        }
        // Integer load with size/sign
        if let Some(r) = ofs_reg {
            match (size, signed) {
                (1, false) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldrb W(dst.value), [X(base.value), X(r.value)])
                }
                (1, true) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldrsb X(dst.value), [X(base.value), X(r.value)])
                }
                (2, false) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldrh W(dst.value), [X(base.value), X(r.value)])
                }
                (2, true) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldrsh X(dst.value), [X(base.value), X(r.value)])
                }
                (4, false) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldr W(dst.value), [X(base.value), X(r.value)])
                }
                (4, true) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldrsw X(dst.value), [X(base.value), X(r.value)])
                }
                _ => {
                    dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [X(base.value), X(r.value)])
                }
            }
        } else {
            match (size, signed) {
                (1, false) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldrb W(dst.value), [X(base.value), ofs as u32])
                }
                (1, true) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldrsb X(dst.value), [X(base.value), ofs as u32])
                }
                (2, false) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldrh W(dst.value), [X(base.value), ofs as u32])
                }
                (2, true) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldrsh X(dst.value), [X(base.value), ofs as u32])
                }
                (4, false) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldr W(dst.value), [X(base.value), ofs as u32])
                }
                (4, true) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldrsw X(dst.value), [X(base.value), ofs as u32])
                }
                _ => {
                    dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [X(base.value), ofs as u32])
                }
            }
        }
    }

    /// Acquire integer load from an address already in `x16`.
    ///
    /// `ldar X` / `ldar W` / `ldarh` / `ldarb`, then sign-extend when
    /// `signed` and the width is narrower than a word.
    pub(crate) fn emit_ldar_from_x16(&mut self, dst: &RegLoc, size: usize, signed: bool) {
        match size {
            1 => {
                dynasm!(self.mc ; .arch aarch64 ; ldarb W(dst.value), [x16]);
                if signed {
                    dynasm!(self.mc ; .arch aarch64 ; sxtb X(dst.value), W(dst.value));
                }
            }
            2 => {
                dynasm!(self.mc ; .arch aarch64 ; ldarh W(dst.value), [x16]);
                if signed {
                    dynasm!(self.mc ; .arch aarch64 ; sxth X(dst.value), W(dst.value));
                }
            }
            4 => {
                dynasm!(self.mc ; .arch aarch64 ; ldar W(dst.value), [x16]);
                if signed {
                    dynasm!(self.mc ; .arch aarch64 ; sxtw X(dst.value), W(dst.value));
                }
            }
            _ => {
                dynasm!(self.mc ; .arch aarch64 ; ldar X(dst.value), [x16]);
            }
        }
    }

    /// Form `x16 = base + ofs_loc` then [`Self::emit_ldar_from_x16`].
    pub(crate) fn emit_acquire_int_load(
        &mut self,
        base: &RegLoc,
        ofs_loc: &Loc,
        dst: &RegLoc,
        size: usize,
        signed: bool,
    ) {
        dynasm!(self.mc ; .arch aarch64 ; mov x16, X(base.value));
        match ofs_loc {
            Loc::Immed(i) | Loc::ImmedFloat(i) => {
                let o = i.value as i32;
                if o != 0 {
                    if (0..4096).contains(&o) {
                        dynasm!(self.mc ; .arch aarch64 ; add x16, x16, o as u32);
                    } else {
                        self.emit_mov_imm64(17, o as i64);
                        dynasm!(self.mc ; .arch aarch64 ; add x16, x16, x17);
                    }
                }
            }
            Loc::Reg(ofs_r) => {
                dynasm!(self.mc ; .arch aarch64 ; add x16, x16, X(ofs_r.value));
            }
            _ => {}
        }
        self.emit_ldar_from_x16(dst, size, signed);
    }

    // ── emit_op_gc_store ──
    // aarch64/opassembler.py emit_op_gc_store
    // x86/assembler.py:1746 genop_gc_store — size_loc

    /// aarch64/opassembler.py `_emit_op_gc_load_indexed` parity.
    /// Folds `ofs` into the index via ADD (with LARGE_IMM_SCRATCH = x16
    /// spillover when the immediate does not fit the ADD encoding),
    /// then emits the sized load from `[base + combined_index]`.  The
    /// upstream aarch64 regalloc asserts the IR-level `factor == 1`
    /// (see `_prepare_op_gc_load_indexed` at regalloc.py:566), so the
    /// index already carries byte-scaled units by construction.
    ///
    /// `ofs` is kept at full i64 width to mirror upstream's `ofs_loc.value`
    /// (regalloc.py:566 passes the raw Signed); `_try_use_older_box` may
    /// fold large displacements into the offset immediate, so narrowing to
    /// i32 would silently truncate those.
    pub(crate) fn emit_op_gcload_indexed_regalloc(
        &mut self,
        base: &RegLoc,
        index: &RegLoc,
        dst: &RegLoc,
        ofs: i64,
        size: i64,
    ) {
        let abs_size = size.unsigned_abs() as usize;
        let signed = size < 0;
        if ofs != 0 {
            // aarch64/opassembler.py:403-408 — combined = index + ofs
            // staged through x16 (LARGE_IMM_SCRATCH).  `check_imm_arg`
            // (regalloc.py:161) admits 0..4095 for ADD_ri; anything else
            // falls back to `mov x16, #ofs; add x16, x16, index`.
            if (0..4096).contains(&ofs) {
                // `opassembler.py:403 ADD_ri(ip0, index, ofs)` is a single
                // instruction; dynasm's `add Xd|SP, Xn|SP, #uimm` form rejects a
                // dynamic register operand, so encode the word directly
                // (`codebuilder.py ADD_ri`).
                self.emit_add_ri(16, index.value, ofs as u32);
            } else {
                self.emit_mov_imm64(16, ofs);
                dynasm!(self.mc ; .arch aarch64
                    ; add x16, x16, X(index.value));
            }
            let combined = RegLoc::new(16, false);
            self.emit_load_sized(base, 0, Some(&combined), dst, abs_size, signed);
        } else {
            self.emit_load_sized(base, 0, Some(index), dst, abs_size, signed);
        }
    }

    /// `size`: byte size (1/2/4/8).
    /// aarch64/opassembler.py emit_op_gc_store parity.
    pub(crate) fn emit_op_gcstore_regalloc(
        &mut self,
        base: &RegLoc,
        ofs_loc: &Loc,
        val: &RegLoc,
        size: usize,
    ) {
        match ofs_loc {
            Loc::Immed(i) | Loc::ImmedFloat(i) => {
                let o = i.value as i32;
                if (-256..0).contains(&o) {
                    self.emit_stur_sized(base, o, val, size);
                } else if o < 0 {
                    // Negative offset outside stur range: load into scratch x16
                    let scratch = RegLoc::new(16, false);
                    dynasm!(self.mc ; .arch aarch64
                        ; movn X(scratch.value), ((!o) as u32) & 0xFFFF
                    );
                    self.emit_store_sized(base, 0, Some(&scratch), val, size);
                } else {
                    self.emit_store_sized(base, o, None, val, size);
                }
            }
            Loc::Reg(ofs_r) => {
                self.emit_store_sized(base, 0, Some(ofs_r), val, size);
            }
            _ => {}
        }
    }

    /// Emit a sized store using STUR (unscaled signed offset, ±256 range).
    fn emit_stur_sized(&mut self, base: &RegLoc, ofs: i32, val: &RegLoc, size: usize) {
        debug_assert!((-256..256).contains(&ofs));
        if val.is_xmm {
            // `get_type_flag` banks only `lltype.Float` (8 bytes). A
            // singlefloat store is int-kind, not `fcvt` + `stur S`.
            debug_assert_eq!(size, 8, "float-kind access is 8 bytes");
            dynasm!(self.mc ; .arch aarch64 ; stur D(val.value), [X(base.value), ofs]);
            return;
        }
        match size {
            8 => dynasm!(self.mc ; .arch aarch64 ; stur X(val.value), [X(base.value), ofs]),
            4 => dynasm!(self.mc ; .arch aarch64 ; stur W(val.value), [X(base.value), ofs]),
            _ => dynasm!(self.mc ; .arch aarch64 ; stur X(val.value), [X(base.value), ofs]),
        }
    }

    /// Emit a sized store: [base + ofs] or [base + ofs_reg].
    fn emit_store_sized(
        &mut self,
        base: &RegLoc,
        ofs: i32,
        ofs_reg: Option<&RegLoc>,
        val: &RegLoc,
        size: usize,
    ) {
        if val.is_xmm {
            // `get_type_flag` banks only `lltype.Float` (8 bytes). A
            // singlefloat store is int-kind, not `fcvt` + `str S`.
            debug_assert_eq!(size, 8, "float-kind access is 8 bytes");
            if let Some(r) = ofs_reg {
                dynasm!(self.mc ; .arch aarch64 ; str D(val.value), [X(base.value), X(r.value)]);
            } else {
                dynasm!(self.mc ; .arch aarch64 ; str D(val.value), [X(base.value), ofs as u32]);
            }
            return;
        }
        if let Some(r) = ofs_reg {
            match size {
                1 => {
                    dynasm!(self.mc ; .arch aarch64 ; strb W(val.value), [X(base.value), X(r.value)])
                }
                2 => {
                    dynasm!(self.mc ; .arch aarch64 ; strh W(val.value), [X(base.value), X(r.value)])
                }
                4 => {
                    dynasm!(self.mc ; .arch aarch64 ; str W(val.value), [X(base.value), X(r.value)])
                }
                _ => {
                    dynasm!(self.mc ; .arch aarch64 ; str X(val.value), [X(base.value), X(r.value)])
                }
            }
        } else {
            match size {
                1 => {
                    dynasm!(self.mc ; .arch aarch64 ; strb W(val.value), [X(base.value), ofs as u32])
                }
                2 => {
                    dynasm!(self.mc ; .arch aarch64 ; strh W(val.value), [X(base.value), ofs as u32])
                }
                4 => {
                    dynasm!(self.mc ; .arch aarch64 ; str W(val.value), [X(base.value), ofs as u32])
                }
                _ => {
                    dynasm!(self.mc ; .arch aarch64 ; str X(val.value), [X(base.value), ofs as u32])
                }
            }
        }
    }
}
