//! x86-64 encoder for memory operands and immediates.
//!
//! Byte selection follows `rx86.py` `encode_mem_reg_plus_const`,
//! `encode_mem_reg_plus_scaled_reg_plus_const`, `encode_stack_bp`,
//! `encode_stack_sp`, `common_modes`, and `X86_64_CodeBuilder::MOV_ri`.
//! `rsp` / `rbp` bases use the stack forms; every other base uses the
//! register-plus-const form, including the `r12` SIB and `r13` forced
//! displacement cases.
use dynasmrt::DynasmApi;

/// `assembler.rs` `Assembler` — `Assembler386.mc`.
pub type Assembler = super::assembler::Assembler;

pub(crate) const BYTE_REG_FLAG: u8 = 0x20;
pub(crate) const NO_BASE_REGISTER: i16 = -1;

pub(crate) const EAX: u8 = 0;
pub(crate) const ECX: u8 = 1;
pub(crate) const EDX: u8 = 2;
pub(crate) const EBX: u8 = 3;
pub(crate) const ESP: u8 = 4;
pub(crate) const EBP: u8 = 5;
pub(crate) const ESI: u8 = 6;
pub(crate) const EDI: u8 = 7;
#[cfg(test)]
pub(crate) const R8: u8 = 8;
#[cfg(test)]
pub(crate) const R9: u8 = 9;
pub(crate) const R10: u8 = 10;
pub(crate) const R11: u8 = 11;
pub(crate) const R12: u8 = 12;
#[cfg(test)]
pub(crate) const R13: u8 = 13;
#[cfg(test)]
pub(crate) const R14: u8 = 14;
#[cfg(test)]
pub(crate) const R15: u8 = 15;

/// `rx86.py` `REX_W`.
pub(crate) const REX_W: u8 = 8;
/// `rx86.py` `REX_R`.
pub(crate) const REX_R: u8 = 4;
/// `rx86.py` `REX_X`.
pub(crate) const REX_X: u8 = 2;
/// `rx86.py` `REX_B`.
pub(crate) const REX_B: u8 = 1;

#[derive(Clone, Copy)]
enum RexKind {
    /// `rex_w`: always emit REX.W.
    W,
    /// `rex_nw`: emit REX only when a high register needs it.
    Nw,
    /// `rex_fw`: always emit REX, without REX.W.
    Fw,
}

/// `rx86.py` `single_byte`.
pub fn single_byte(value: i64) -> bool {
    (-128..128).contains(&value)
}

/// `rx86.py` `fits_in_32bits`.
pub fn fits_in_32bits(value: i64) -> bool {
    (-2147483648..=2147483647).contains(&value)
}

/// `rx86.py` `reg_number_3bits` for a 64-bit code builder (`WORD == 8`).
pub(crate) fn reg_number_3bits(reg: u8) -> u8 {
    assert!(reg < 16, "register number out of range");
    reg & 7
}

/// `rx86.py` `rex_register`. Factor 1 is the ModRM rm/base field (`REX_B`);
/// factor 8 is the ModRM reg field (`REX_R`).
fn rex_register(reg: u8, factor: u8) -> u8 {
    if reg >= 8 {
        if factor == 1 {
            REX_B
        } else if factor == 8 {
            REX_R
        } else {
            panic!("rex_register: factor {factor}");
        }
    } else {
        0
    }
}

/// `rx86.py` `rex_byte_register`. Strips `BYTE_REG_FLAG`, then `rex_register`.
fn rex_byte_register(reg: u8, factor: u8) -> u8 {
    rex_register(reg & !BYTE_REG_FLAG, factor)
}

fn push_bytes(mc: &mut Assembler, bytes: &[u8]) {
    for &byte in bytes {
        DynasmApi::push(mc, byte);
    }
}

fn writeimm8(mc: &mut Assembler, imm: i32) {
    DynasmApi::push(mc, imm as u8);
}

fn writeimm16(mc: &mut Assembler, imm: i32) {
    push_bytes(mc, &(imm as u16).to_le_bytes());
}

fn writeimm32(mc: &mut Assembler, imm: i32) {
    debug_assert!(fits_in_32bits(i64::from(imm)));
    push_bytes(mc, &imm.to_le_bytes());
}

fn writeimm64(mc: &mut Assembler, imm: i64) {
    push_bytes(mc, &imm.to_le_bytes());
}

/// `rx86.py` `encode_rex`. `w` is `REX_W` or zero.
fn encode_rex(mc: &mut Assembler, rexbyte: u8, w: u8) {
    assert!(rexbyte < 8);
    DynasmApi::push(mc, 0x40 | w | rexbyte);
}

/// `rx86.py` `encode_rex_opt`. A zero `rexbyte` emits nothing.
fn encode_rex_opt(mc: &mut Assembler, rexbyte: u8) {
    assert!(rexbyte < 8);
    if rexbyte != 0 {
        DynasmApi::push(mc, 0x40 | rexbyte);
    }
}

fn emit_prefix_rex(mc: &mut Assembler, mandatory: u8, kind: RexKind, rex: u8) {
    if mandatory != 0 {
        DynasmApi::push(mc, mandatory);
    }
    match kind {
        RexKind::W => encode_rex(mc, rex, REX_W),
        RexKind::Nw => encode_rex_opt(mc, rex),
        RexKind::Fw => encode_rex(mc, rex, 0),
    }
}

/// `rx86.py` `encode_mem_reg_plus_const`.
///
/// `reg` must not be `rsp` or `rbp` (`esp` / `ebp`). `r12` forces a SIB byte.
/// `r13` cannot use the no-displacement ModRM encoding.
pub(crate) fn encode_mem_reg_plus_const(mc: &mut Assembler, mem: (u8, i32), orbyte: u8) {
    let (reg, offset) = mem;
    assert!(reg != ESP && reg != EBP, "rsp/rbp use the stack encoders");
    let reg1 = reg_number_3bits(reg);
    let mut no_offset = offset == 0;
    let mut sib: i32 = -1;
    if reg1 == ESP {
        sib = i32::from((ESP << 3) | ESP);
    } else if reg1 == EBP {
        no_offset = false;
    }
    if no_offset {
        DynasmApi::push(mc, 0x00 | orbyte | reg1);
        if sib >= 0 {
            DynasmApi::push(mc, sib as u8);
        }
    } else if single_byte(i64::from(offset)) {
        DynasmApi::push(mc, 0x40 | orbyte | reg1);
        if sib >= 0 {
            DynasmApi::push(mc, sib as u8);
        }
        writeimm8(mc, offset);
    } else {
        DynasmApi::push(mc, 0x80 | orbyte | reg1);
        if sib >= 0 {
            DynasmApi::push(mc, sib as u8);
        }
        writeimm32(mc, offset);
    }
}

/// `rx86.py` `encode_stack_bp`. `[rbp + offset]`.
pub(crate) fn encode_stack_bp(mc: &mut Assembler, offset: i32, force_32bits: bool, orbyte: u8) {
    if !force_32bits && single_byte(i64::from(offset)) {
        DynasmApi::push(mc, 0x40 | orbyte | EBP);
        writeimm8(mc, offset);
    } else {
        DynasmApi::push(mc, 0x80 | orbyte | EBP);
        writeimm32(mc, offset);
    }
}

/// `rx86.py` `encode_stack_sp`. `[rsp + offset]`, always with a SIB byte.
pub(crate) fn encode_stack_sp(mc: &mut Assembler, offset: i32, orbyte: u8) {
    let sib = (ESP << 3) | ESP;
    if offset == 0 {
        DynasmApi::push(mc, 0x04 | orbyte);
        DynasmApi::push(mc, sib);
    } else if single_byte(i64::from(offset)) {
        DynasmApi::push(mc, 0x44 | orbyte);
        DynasmApi::push(mc, sib);
        writeimm8(mc, offset);
    } else {
        DynasmApi::push(mc, 0x84 | orbyte);
        DynasmApi::push(mc, sib);
        writeimm32(mc, offset);
    }
}

/// Dispatch on the base register: `encode_stack_sp`, `encode_stack_bp`, or
/// `encode_mem_reg_plus_const`.
fn encode_m(mc: &mut Assembler, base: u8, offset: i32, orbyte: u8) {
    if base == ESP {
        encode_stack_sp(mc, offset, orbyte);
    } else if base == EBP {
        encode_stack_bp(mc, offset, false, orbyte);
    } else {
        encode_mem_reg_plus_const(mc, (base, offset), orbyte);
    }
}

/// `rx86.py` `encode_mem_reg_plus_scaled_reg_plus_const`.
///
/// `addr` is `(base, index, scaleshift, offset)`. `scaleshift` is 0..4
/// (`*1`, `*2`, `*4`, `*8`). `base == NO_BASE_REGISTER` forces a disp32.
/// A base whose low 3 bits are `rbp` (`rbp` itself is rejected; `r13` is not)
/// cannot use the no-displacement form. The index cannot be `rsp`.
pub(crate) fn encode_mem_reg_plus_scaled_reg_plus_const(
    mc: &mut Assembler,
    addr: (i16, u8, u8, i32),
    orbyte: u8,
) {
    let (reg1, reg2, scaleshift, offset) = addr;
    assert!(reg1 != i16::from(EBP) && reg2 != ESP);
    assert!(scaleshift < 4);
    let reg2_3 = reg_number_3bits(reg2);
    if reg1 == NO_BASE_REGISTER {
        DynasmApi::push(mc, 0x04 | orbyte);
        DynasmApi::push(mc, (scaleshift << 6) | (reg2_3 << 3) | 5);
        writeimm32(mc, offset);
        return;
    }
    let reg1_3 = reg_number_3bits(reg1 as u8);
    let sib = (scaleshift << 6) | (reg2_3 << 3) | reg1_3;
    let mut no_offset = offset == 0;
    if reg1_3 == EBP {
        no_offset = false;
    }
    if no_offset {
        DynasmApi::push(mc, 0x04 | orbyte);
        DynasmApi::push(mc, sib);
    } else if single_byte(i64::from(offset)) {
        DynasmApi::push(mc, 0x44 | orbyte);
        DynasmApi::push(mc, sib);
        writeimm8(mc, offset);
    } else {
        DynasmApi::push(mc, 0x84 | orbyte);
        DynasmApi::push(mc, sib);
        writeimm32(mc, offset);
    }
}

/// Register-register ModRM: `0xC0 | orbyte | (reg << 3) | rm`.
///
/// This is the encoding `rx86.py` `encode_register` produces when the
/// following byte is `0xC0` (`MOV_rr`, `ADD_rr`, and the other `*_rr` forms).
pub(crate) fn encode_modrm_reg_reg(mc: &mut Assembler, reg: u8, rm: u8, orbyte: u8) {
    DynasmApi::push(
        mc,
        0xC0 | orbyte | (reg_number_3bits(reg) << 3) | reg_number_3bits(rm),
    );
}

/// `rx86.py` `rex_mem_reg_plus_const`. `REX_B` when the base is `r8`–`r15`.
fn rex_mem_reg_plus_const(mem: (u8, i32)) -> u8 {
    let (reg, _offset) = mem;
    if reg >= 8 { REX_B } else { 0 }
}

/// `rx86.py` `rex_mem_reg_plus_scaled_reg_plus_const`.
/// `REX_B` from the base and `REX_X` from the index. `NO_BASE_REGISTER` sets neither.
fn rex_mem_reg_plus_scaled_reg_plus_const(addr: (i16, u8, u8, i32)) -> u8 {
    let (reg1, reg2, _scaleshift, _offset) = addr;
    let mut rex = 0;
    if reg1 >= 8 {
        rex |= REX_B;
    }
    if reg2 >= 8 {
        rex |= REX_X;
    }
    rex
}

fn op_mem_rex(
    mc: &mut Assembler,
    kind: RexKind,
    mandatory: u8,
    opcode: &[u8],
    reg_rex: u8,
    reg: u8,
    base: u8,
    offset: i32,
) {
    let rex = reg_rex | rex_mem_reg_plus_const((base, offset));
    emit_prefix_rex(mc, mandatory, kind, rex);
    push_bytes(mc, opcode);
    encode_m(mc, base, offset, reg_number_3bits(reg) << 3);
}

fn op_mem(
    mc: &mut Assembler,
    kind: RexKind,
    mandatory: u8,
    opcode: &[u8],
    reg: u8,
    base: u8,
    offset: i32,
) {
    op_mem_rex(
        mc,
        kind,
        mandatory,
        opcode,
        rex_register(reg, 8),
        reg,
        base,
        offset,
    );
}

fn op_addr_rex(
    mc: &mut Assembler,
    kind: RexKind,
    mandatory: u8,
    opcode: &[u8],
    reg_rex: u8,
    reg: u8,
    addr: (i16, u8, u8, i32),
) {
    let rex = reg_rex | rex_mem_reg_plus_scaled_reg_plus_const(addr);
    emit_prefix_rex(mc, mandatory, kind, rex);
    push_bytes(mc, opcode);
    encode_mem_reg_plus_scaled_reg_plus_const(mc, addr, reg_number_3bits(reg) << 3);
}

fn op_addr(
    mc: &mut Assembler,
    kind: RexKind,
    mandatory: u8,
    opcode: &[u8],
    reg: u8,
    addr: (i16, u8, u8, i32),
) {
    op_addr_rex(mc, kind, mandatory, opcode, rex_register(reg, 8), reg, addr);
}

fn op_bp(mc: &mut Assembler, kind: RexKind, mandatory: u8, opcode: &[u8], reg: u8, offset: i32) {
    emit_prefix_rex(mc, mandatory, kind, rex_register(reg, 8));
    push_bytes(mc, opcode);
    encode_stack_bp(mc, offset, false, reg_number_3bits(reg) << 3);
}

fn op_sp(mc: &mut Assembler, kind: RexKind, mandatory: u8, opcode: &[u8], reg: u8, offset: i32) {
    emit_prefix_rex(mc, mandatory, kind, rex_register(reg, 8));
    push_bytes(mc, opcode);
    encode_stack_sp(mc, offset, reg_number_3bits(reg) << 3);
}

fn alu_ri(mc: &mut Assembler, ext: u8, reg: u8, immed: i32) {
    encode_rex(mc, rex_register(reg, 1), REX_W);
    let modrm = 0xC0 | (ext << 3) | reg_number_3bits(reg);
    if single_byte(i64::from(immed)) {
        DynasmApi::push(mc, 0x83);
        DynasmApi::push(mc, modrm);
        writeimm8(mc, immed);
    } else {
        DynasmApi::push(mc, 0x81);
        DynasmApi::push(mc, modrm);
        writeimm32(mc, immed);
    }
}

fn alu_mi(mc: &mut Assembler, ext: u8, mem: (u8, i32), immed: i32) {
    encode_rex(mc, rex_mem_reg_plus_const(mem), REX_W);
    if single_byte(i64::from(immed)) {
        DynasmApi::push(mc, 0x83);
        encode_m(mc, mem.0, mem.1, ext << 3);
        writeimm8(mc, immed);
    } else {
        DynasmApi::push(mc, 0x81);
        encode_m(mc, mem.0, mem.1, ext << 3);
        writeimm32(mc, immed);
    }
}

fn shift_ri(mc: &mut Assembler, ext: u8, reg: u8, immed: i32) {
    encode_rex(mc, rex_register(reg, 1), REX_W);
    let modrm = 0xC0 | (ext << 3) | reg_number_3bits(reg);
    if immed == 1 {
        DynasmApi::push(mc, 0xD1);
        DynasmApi::push(mc, modrm);
    } else {
        DynasmApi::push(mc, 0xC1);
        DynasmApi::push(mc, modrm);
        writeimm8(mc, immed);
    }
}

/// Reg-reg ModRM. `reg` is the ModRM.reg field, `rm` the ModRM.rm field.
/// The mandatory prefix is emitted before the REX byte (`xmminsn`).
fn op_rr(mc: &mut Assembler, kind: RexKind, mandatory: u8, opcode: &[u8], reg: u8, rm: u8) {
    let rex = rex_register(reg, 8) | rex_register(rm, 1);
    emit_prefix_rex(mc, mandatory, kind, rex);
    push_bytes(mc, opcode);
    encode_modrm_reg_reg(mc, reg, rm, 0);
}

// MOV

/// `MOV_rm` — `mov r64, [base + ofs]`.
pub fn mov_rm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::W, 0, &[0x8B], dst, mem.0, mem.1);
}

/// `MOV_mr` — `mov [base + ofs], r64`.
pub fn mov_mr(mc: &mut Assembler, mem: (u8, i32), src: u8) {
    op_mem(mc, RexKind::W, 0, &[0x89], src, mem.0, mem.1);
}

/// `MOV_ra` — `mov r64, [base + index*scale + ofs]`.
pub fn mov_ra(mc: &mut Assembler, dst: u8, addr: (i16, u8, u8, i32)) {
    op_addr(mc, RexKind::W, 0, &[0x8B], dst, addr);
}

/// `MOV_ar` — `mov [base + index*scale + ofs], r64`.
pub fn mov_ar(mc: &mut Assembler, addr: (i16, u8, u8, i32), src: u8) {
    op_addr(mc, RexKind::W, 0, &[0x89], src, addr);
}

/// `MOV_rb` — `mov r64, [rbp + ofs]`.
pub(crate) fn mov_rb(mc: &mut Assembler, dst: u8, offset: i32) {
    op_bp(mc, RexKind::W, 0, &[0x8B], dst, offset);
}

/// `MOV_br` — `mov [rbp + ofs], r64`.
pub(crate) fn mov_br(mc: &mut Assembler, offset: i32, src: u8) {
    op_bp(mc, RexKind::W, 0, &[0x89], src, offset);
}

/// `MOV_rs` — `mov r64, [rsp + ofs]`.
pub(crate) fn mov_rs(mc: &mut Assembler, dst: u8, offset: i32) {
    op_sp(mc, RexKind::W, 0, &[0x8B], dst, offset);
}

/// `MOV_sr` — `mov [rsp + ofs], r64`.
pub(crate) fn mov_sr(mc: &mut Assembler, offset: i32, src: u8) {
    op_sp(mc, RexKind::W, 0, &[0x89], src, offset);
}

/// `LEA_rs` — `lea r64, [rsp + ofs]`.
pub(crate) fn lea_rs(mc: &mut Assembler, dst: u8, offset: i32) {
    op_sp(mc, RexKind::W, 0, &[0x8D], dst, offset);
}

/// `MOV32_rm` — `mov r32, [base + ofs]`.
pub fn mov32_rm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::Nw, 0, &[0x8B], dst, mem.0, mem.1);
}

/// `MOV32_rr` — `mov r32, r32`. Zero-extends into the full register.
pub(crate) fn mov32_rr(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0, &[0x8B], dst, src);
}

/// `MOV32_mr` — `mov [base + ofs], r32`.
pub fn mov32_mr(mc: &mut Assembler, mem: (u8, i32), src: u8) {
    op_mem(mc, RexKind::Nw, 0, &[0x89], src, mem.0, mem.1);
}

/// `MOV32_ra` — `mov r32, [base + index*scale + ofs]`.
pub fn mov32_ra(mc: &mut Assembler, dst: u8, addr: (i16, u8, u8, i32)) {
    op_addr(mc, RexKind::Nw, 0, &[0x8B], dst, addr);
}

/// `MOV32_ar` — `mov [base + index*scale + ofs], r32`.
pub fn mov32_ar(mc: &mut Assembler, addr: (i16, u8, u8, i32), src: u8) {
    op_addr(mc, RexKind::Nw, 0, &[0x89], src, addr);
}

/// `MOV16_mr` — `mov [base + ofs], r16`.
pub fn mov16_mr(mc: &mut Assembler, mem: (u8, i32), src: u8) {
    op_mem(mc, RexKind::Nw, 0x66, &[0x89], src, mem.0, mem.1);
}

/// `MOV16_ar` — `mov [base + index*scale + ofs], r16`.
pub fn mov16_ar(mc: &mut Assembler, addr: (i16, u8, u8, i32), src: u8) {
    op_addr(mc, RexKind::Nw, 0x66, &[0x89], src, addr);
}

/// `MOV8_mr` — `mov [base + ofs], r8`.
pub fn mov8_mr(mc: &mut Assembler, mem: (u8, i32), src: u8) {
    let reg_rex = rex_byte_register(src, 8);
    let src = src & !BYTE_REG_FLAG;
    op_mem_rex(mc, RexKind::Fw, 0, &[0x88], reg_rex, src, mem.0, mem.1);
}

/// `MOV8_ar` — `mov [base + index*scale + ofs], r8`.
pub fn mov8_ar(mc: &mut Assembler, addr: (i16, u8, u8, i32), src: u8) {
    let reg_rex = rex_byte_register(src, 8);
    let src = src & !BYTE_REG_FLAG;
    op_addr_rex(mc, RexKind::Fw, 0, &[0x88], reg_rex, src, addr);
}

/// `MOV_mi` — `mov [base + ofs], imm32`.
pub fn mov_mi(mc: &mut Assembler, mem: (u8, i32), immed: i32) {
    encode_rex(mc, rex_mem_reg_plus_const(mem), REX_W);
    DynasmApi::push(mc, 0xC7);
    encode_m(mc, mem.0, mem.1, 0);
    writeimm32(mc, immed);
}

/// `MOV_ai` — `mov [base + index*scale + ofs], imm32`.
pub fn mov_ai(mc: &mut Assembler, addr: (i16, u8, u8, i32), immed: i32) {
    encode_rex(
        mc,
        rex_register(0, 8) | rex_mem_reg_plus_scaled_reg_plus_const(addr),
        REX_W,
    );
    DynasmApi::push(mc, 0xC7);
    encode_mem_reg_plus_scaled_reg_plus_const(mc, addr, 0);
    writeimm32(mc, immed);
}

/// `MOV_bi` — `mov [rbp + ofs], imm32`.
pub(crate) fn mov_bi(mc: &mut Assembler, offset: i32, immed: i32) {
    encode_rex(mc, 0, REX_W);
    DynasmApi::push(mc, 0xC7);
    encode_stack_bp(mc, offset, false, 0);
    writeimm32(mc, immed);
}

/// `MOV_si` — `mov qword [rsp + ofs], imm32`. `rex_w`, opcode `C7 /0`.
/// The immediate is the last four bytes (`IncreaseStackSlowPath.generate_body`).
pub(crate) fn mov_si(mc: &mut Assembler, offset: i32, immed: i32) {
    encode_rex(mc, 0, REX_W);
    DynasmApi::push(mc, 0xC7);
    encode_stack_sp(mc, offset, 0);
    writeimm32(mc, immed);
}

/// `MOV32_si` — `mov dword [rsp + ofs], imm32`. `rex_nw`, no `REX.W`.
pub(crate) fn mov32_si(mc: &mut Assembler, offset: i32, immed: i32) {
    DynasmApi::push(mc, 0xC7);
    encode_stack_sp(mc, offset, 0);
    writeimm32(mc, immed);
}

/// `MOV32_bi` — `mov dword [rbp + ofs], imm32`. `rex_nw`, no `REX.W`.
pub(crate) fn mov32_bi(mc: &mut Assembler, offset: i32, immed: i32) {
    DynasmApi::push(mc, 0xC7);
    encode_stack_bp(mc, offset, false, 0);
    writeimm32(mc, immed);
}

/// `MOV32_mi` — `mov dword [base + ofs], imm32`.
pub fn mov32_mi(mc: &mut Assembler, mem: (u8, i32), immed: i32) {
    encode_rex_opt(mc, rex_mem_reg_plus_const(mem));
    DynasmApi::push(mc, 0xC7);
    encode_m(mc, mem.0, mem.1, 0);
    writeimm32(mc, immed);
}

/// `MOV32_ai` — `mov dword [base + index*scale + ofs], imm32`.
pub fn mov32_ai(mc: &mut Assembler, addr: (i16, u8, u8, i32), immed: i32) {
    encode_rex_opt(
        mc,
        rex_register(0, 8) | rex_mem_reg_plus_scaled_reg_plus_const(addr),
    );
    DynasmApi::push(mc, 0xC7);
    encode_mem_reg_plus_scaled_reg_plus_const(mc, addr, 0);
    writeimm32(mc, immed);
}

/// `MOV16_mi` — `mov word [base + ofs], imm16`.
pub fn mov16_mi(mc: &mut Assembler, mem: (u8, i32), immed: i32) {
    DynasmApi::push(mc, 0x66);
    encode_rex_opt(mc, rex_mem_reg_plus_const(mem));
    DynasmApi::push(mc, 0xC7);
    encode_m(mc, mem.0, mem.1, 0);
    writeimm16(mc, immed);
}

/// `MOV16_ai` — `mov word [base + index*scale + ofs], imm16`.
pub fn mov16_ai(mc: &mut Assembler, addr: (i16, u8, u8, i32), immed: i32) {
    DynasmApi::push(mc, 0x66);
    encode_rex_opt(
        mc,
        rex_register(0, 8) | rex_mem_reg_plus_scaled_reg_plus_const(addr),
    );
    DynasmApi::push(mc, 0xC7);
    encode_mem_reg_plus_scaled_reg_plus_const(mc, addr, 0);
    writeimm16(mc, immed);
}

/// `MOV8_mi` — `mov byte [base + ofs], imm8`.
pub fn mov8_mi(mc: &mut Assembler, mem: (u8, i32), immed: i32) {
    encode_rex(mc, rex_mem_reg_plus_const(mem), 0);
    DynasmApi::push(mc, 0xC6);
    encode_m(mc, mem.0, mem.1, 0);
    writeimm8(mc, immed);
}

/// `MOV8_ai` — `mov byte [base + index*scale + ofs], imm8`.
pub fn mov8_ai(mc: &mut Assembler, addr: (i16, u8, u8, i32), immed: i32) {
    encode_rex(
        mc,
        rex_register(0, 8) | rex_mem_reg_plus_scaled_reg_plus_const(addr),
        0,
    );
    DynasmApi::push(mc, 0xC6);
    encode_mem_reg_plus_scaled_reg_plus_const(mc, addr, 0);
    writeimm8(mc, immed);
}

/// `MOVZX8_rm` — `movzx r64, byte [base + ofs]`.
pub fn movzx8_rm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::W, 0, &[0x0F, 0xB6], dst, mem.0, mem.1);
}

/// `MOVZX8_ra` — `movzx r64, byte [base + index*scale + ofs]`.
pub fn movzx8_ra(mc: &mut Assembler, dst: u8, addr: (i16, u8, u8, i32)) {
    op_addr(mc, RexKind::W, 0, &[0x0F, 0xB6], dst, addr);
}

/// `MOVZX16_rm` — `movzx r64, word [base + ofs]`.
pub fn movzx16_rm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::W, 0, &[0x0F, 0xB7], dst, mem.0, mem.1);
}

/// `MOVZX16_ra` — `movzx r64, word [base + index*scale + ofs]`.
pub fn movzx16_ra(mc: &mut Assembler, dst: u8, addr: (i16, u8, u8, i32)) {
    op_addr(mc, RexKind::W, 0, &[0x0F, 0xB7], dst, addr);
}

/// `MOVSX8_rm` — `movsx r64, byte [base + ofs]`.
pub fn movsx8_rm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::W, 0, &[0x0F, 0xBE], dst, mem.0, mem.1);
}

/// `MOVSX8_ra` — `movsx r64, byte [base + index*scale + ofs]`.
pub fn movsx8_ra(mc: &mut Assembler, dst: u8, addr: (i16, u8, u8, i32)) {
    op_addr(mc, RexKind::W, 0, &[0x0F, 0xBE], dst, addr);
}

/// `MOVSX16_rm` — `movsx r64, word [base + ofs]`.
pub fn movsx16_rm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::W, 0, &[0x0F, 0xBF], dst, mem.0, mem.1);
}

/// `MOVSX16_ra` — `movsx r64, word [base + index*scale + ofs]`.
pub fn movsx16_ra(mc: &mut Assembler, dst: u8, addr: (i16, u8, u8, i32)) {
    op_addr(mc, RexKind::W, 0, &[0x0F, 0xBF], dst, addr);
}

/// `MOVSX32_rm` — `movsxd r64, dword [base + ofs]`.
pub fn movsx32_rm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::W, 0, &[0x63], dst, mem.0, mem.1);
}

/// `MOVSX32_ra` — `movsxd r64, dword [base + index*scale + ofs]`.
pub fn movsx32_ra(mc: &mut Assembler, dst: u8, addr: (i16, u8, u8, i32)) {
    op_addr(mc, RexKind::W, 0, &[0x63], dst, addr);
}

/// `MOV_ri` — `mov r64, imm`.
pub(crate) fn mov_ri(mc: &mut Assembler, reg: u8, immed: i64) {
    if (0..=0xFFFF_FFFF).contains(&immed) {
        mov_riu32(mc, reg, immed as u32 as i32);
    } else if fits_in_32bits(immed) {
        mov_ri32(mc, reg, immed as i32);
    } else {
        mov_ri64(mc, reg, immed);
    }
}

/// `MOV_riu32` — `mov r32, imm32`.
pub(crate) fn mov_riu32(mc: &mut Assembler, reg: u8, immed: i32) {
    encode_rex_opt(mc, rex_register(reg, 1));
    DynasmApi::push(mc, 0xB8 | reg_number_3bits(reg));
    writeimm32(mc, immed);
}

/// `MOV_ri32` — `mov r64, imm32`.
pub(crate) fn mov_ri32(mc: &mut Assembler, reg: u8, immed: i32) {
    encode_rex(mc, rex_register(reg, 1), REX_W);
    DynasmApi::push(mc, 0xC7);
    DynasmApi::push(mc, 0xC0 | reg_number_3bits(reg));
    writeimm32(mc, immed);
}

/// `MOV_ri64` — `mov r64, imm64`.
pub(crate) fn mov_ri64(mc: &mut Assembler, reg: u8, immed: i64) {
    encode_rex(mc, rex_register(reg, 1), REX_W);
    DynasmApi::push(mc, 0xB8 | reg_number_3bits(reg));
    writeimm64(mc, immed);
}

// LEA

/// `LEA_rm` — `lea r64, [base + ofs]`.
pub fn lea_rm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::W, 0, &[0x8D], dst, mem.0, mem.1);
}

/// `LEA_ra` — `lea r64, [base + index*scale + ofs]`.
pub fn lea_ra(mc: &mut Assembler, dst: u8, addr: (i16, u8, u8, i32)) {
    op_addr(mc, RexKind::W, 0, &[0x8D], dst, addr);
}

// ALU reg, imm

/// `ADD_ri8`/`ADD_ri32` — `add r64, imm8` / `add r64, imm32`.
pub(crate) fn add_ri(mc: &mut Assembler, reg: u8, immed: i32) {
    alu_ri(mc, 0, reg, immed);
}

/// `OR_ri8`/`OR_ri32` — `or r64, imm8` / `or r64, imm32`.
pub(crate) fn or_ri(mc: &mut Assembler, reg: u8, immed: i32) {
    alu_ri(mc, 1, reg, immed);
}

/// `AND_ri8`/`AND_ri32` — `and r64, imm8` / `and r64, imm32`.
pub(crate) fn and_ri(mc: &mut Assembler, reg: u8, immed: i32) {
    alu_ri(mc, 4, reg, immed);
}

/// `SUB_ri8`/`SUB_ri32` — `sub r64, imm8` / `sub r64, imm32`.
pub(crate) fn sub_ri(mc: &mut Assembler, reg: u8, immed: i32) {
    alu_ri(mc, 5, reg, immed);
}

/// `XOR_ri8`/`XOR_ri32` — `xor r64, imm8` / `xor r64, imm32`.
pub(crate) fn xor_ri(mc: &mut Assembler, reg: u8, immed: i32) {
    alu_ri(mc, 6, reg, immed);
}

/// `CMP_ri8`/`CMP_ri32` — `cmp r64, imm8` / `cmp r64, imm32`.
pub(crate) fn cmp_ri(mc: &mut Assembler, reg: u8, immed: i32) {
    alu_ri(mc, 7, reg, immed);
}

/// `CMOVNS_rr` — `cmovns r64, r64`.
pub(crate) fn cmovns_rr(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::W, 0, &[0x0F, 0x49], dst, src);
}

/// `SHL_ri` — `shl r64, 1` / `shl r64, imm8`.
pub(crate) fn shl_ri(mc: &mut Assembler, reg: u8, immed: i32) {
    shift_ri(mc, 4, reg, immed);
}

/// `SHR_ri` — `shr r64, 1` / `shr r64, imm8`.
pub(crate) fn shr_ri(mc: &mut Assembler, reg: u8, immed: i32) {
    shift_ri(mc, 5, reg, immed);
}

/// `SAR_ri` — `sar r64, 1` / `sar r64, imm8`.
pub(crate) fn sar_ri(mc: &mut Assembler, reg: u8, immed: i32) {
    shift_ri(mc, 7, reg, immed);
}

/// `IMUL_rri8`/`IMUL_rri32` — `imul r64, r64, imm8` / `imul r64, r64, imm32`.
pub(crate) fn imul_rri(mc: &mut Assembler, dst: u8, src: u8, immed: i32) {
    encode_rex(mc, rex_register(dst, 8) | rex_register(src, 1), REX_W);
    DynasmApi::push(
        mc,
        if single_byte(i64::from(immed)) {
            0x6B
        } else {
            0x69
        },
    );
    encode_modrm_reg_reg(mc, dst, src, 0);
    if single_byte(i64::from(immed)) {
        writeimm8(mc, immed);
    } else {
        writeimm32(mc, immed);
    }
}

/// `IMUL_ri` — `imul r64, imm`.
pub(crate) fn imul_ri(mc: &mut Assembler, reg: u8, immed: i32) {
    imul_rri(mc, reg, reg, immed);
}

/// `IMUL_rri8`/`IMUL_rri32` — `imul r64, [base + ofs], imm8` / `imul r64, [base + ofs], imm32`.
pub(crate) fn imul_rmi(mc: &mut Assembler, dst: u8, mem: (u8, i32), immed: i32) {
    encode_rex(
        mc,
        rex_register(dst, 8) | rex_mem_reg_plus_const(mem),
        REX_W,
    );
    if single_byte(i64::from(immed)) {
        DynasmApi::push(mc, 0x6B);
        encode_m(mc, mem.0, mem.1, reg_number_3bits(dst) << 3);
        writeimm8(mc, immed);
    } else {
        DynasmApi::push(mc, 0x69);
        encode_m(mc, mem.0, mem.1, reg_number_3bits(dst) << 3);
        writeimm32(mc, immed);
    }
}

// ALU reg, mem

/// `ADD_rb` — `add r64, [rbp + ofs]`.
pub(crate) fn add_rb(mc: &mut Assembler, dst: u8, offset: i32) {
    op_bp(mc, RexKind::W, 0, &[0x03], dst, offset);
}

/// `OR_rb` — `or r64, [rbp + ofs]`.
pub(crate) fn or_rb(mc: &mut Assembler, dst: u8, offset: i32) {
    op_bp(mc, RexKind::W, 0, &[0x0B], dst, offset);
}

/// `AND_rb` — `and r64, [rbp + ofs]`.
pub(crate) fn and_rb(mc: &mut Assembler, dst: u8, offset: i32) {
    op_bp(mc, RexKind::W, 0, &[0x23], dst, offset);
}

/// `SUB_rm` — `sub r64, [base + ofs]`.
pub fn sub_rm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::W, 0, &[0x2B], dst, mem.0, mem.1);
}

/// `SUB_rb` — `sub r64, [rbp + ofs]`.
pub(crate) fn sub_rb(mc: &mut Assembler, dst: u8, offset: i32) {
    op_bp(mc, RexKind::W, 0, &[0x2B], dst, offset);
}

/// `XOR_rb` — `xor r64, [rbp + ofs]`.
pub(crate) fn xor_rb(mc: &mut Assembler, dst: u8, offset: i32) {
    op_bp(mc, RexKind::W, 0, &[0x33], dst, offset);
}

/// `CMP_rm` — `cmp r64, [base + ofs]`.
pub fn cmp_rm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::W, 0, &[0x3B], dst, mem.0, mem.1);
}

/// `CMP_rb` — `cmp r64, [rbp + ofs]`.
pub(crate) fn cmp_rb(mc: &mut Assembler, dst: u8, offset: i32) {
    op_bp(mc, RexKind::W, 0, &[0x3B], dst, offset);
}

/// `CMP_mr` — `cmp [base + ofs], r64`.
pub fn cmp_mr(mc: &mut Assembler, mem: (u8, i32), src: u8) {
    op_mem(mc, RexKind::W, 0, &[0x39], src, mem.0, mem.1);
}

/// `CMP_br` — `cmp [rbp + ofs], r64`.
pub(crate) fn cmp_br(mc: &mut Assembler, offset: i32, src: u8) {
    op_bp(mc, RexKind::W, 0, &[0x39], src, offset);
}

/// `CMP_mi8`/`CMP_mi32` — `cmp [base + ofs], imm8` / `cmp [base + ofs], imm32`.
pub fn cmp_mi(mc: &mut Assembler, mem: (u8, i32), immed: i32) {
    alu_mi(mc, 7, mem, immed);
}

/// `CMP32_mi` — `cmp dword [base + ofs], imm32`. Always the imm32 form
/// (`rex_nw`, opcode `0x81`), never the imm8 `0x83` form and never REX.W.
pub(crate) fn cmp32_mi(mc: &mut Assembler, mem: (u8, i32), immed: i32) {
    encode_rex_opt(mc, rex_mem_reg_plus_const(mem));
    DynasmApi::push(mc, 0x81);
    encode_m(mc, mem.0, mem.1, 7 << 3);
    writeimm32(mc, immed);
}

/// `CMP_bi8`/`CMP_bi32` — `cmp [rbp + ofs], imm8` / `cmp [rbp + ofs], imm32`.
pub(crate) fn cmp_bi(mc: &mut Assembler, offset: i32, immed: i32) {
    alu_mi(mc, 7, (EBP, offset), immed);
}

/// `SUB_mi8`/`SUB_mi32` — `sub [base + ofs], imm8` / `sub [base + ofs], imm32`.
pub fn sub_mi(mc: &mut Assembler, mem: (u8, i32), immed: i32) {
    alu_mi(mc, 5, mem, immed);
}

/// `SUB_ji8` — `sub qword [abs32], imm8`. `encode_abs` with `/5`.
pub fn sub_ji8(mc: &mut Assembler, abs_addr: i32, immed: i32) {
    DynasmApi::push(mc, 0x48);
    DynasmApi::push(mc, 0x83);
    DynasmApi::push(mc, 0x2C);
    DynasmApi::push(mc, 0x25);
    writeimm32(mc, abs_addr);
    writeimm8(mc, immed);
}

/// `TEST8_mi` — `test byte [base + ofs], imm8`.
pub fn test8_mi(mc: &mut Assembler, mem: (u8, i32), immed: i32) {
    encode_rex_opt(mc, rex_mem_reg_plus_const(mem));
    DynasmApi::push(mc, 0xF6);
    encode_m(mc, mem.0, mem.1, 0);
    writeimm8(mc, immed);
}

/// `TEST_ri` — `test r64, imm32`.
pub(crate) fn test_ri(mc: &mut Assembler, reg: u8, immed: i32) {
    encode_rex(mc, rex_register(reg, 1), REX_W);
    DynasmApi::push(mc, 0xF7);
    DynasmApi::push(mc, 0xC0 | reg_number_3bits(reg));
    writeimm32(mc, immed);
}

/// `TEST_mi` — `test qword [base + ofs], imm32`.
pub(crate) fn test_mi(mc: &mut Assembler, mem: (u8, i32), immed: i32) {
    encode_rex(mc, rex_mem_reg_plus_const(mem), REX_W);
    DynasmApi::push(mc, 0xF7);
    encode_m(mc, mem.0, mem.1, 0);
    writeimm32(mc, immed);
}

/// `TEST_ai` — `test qword [base + index*scale + ofs], imm32`.
pub(crate) fn test_ai(mc: &mut Assembler, addr: (i16, u8, u8, i32), immed: i32) {
    encode_rex(mc, rex_mem_reg_plus_scaled_reg_plus_const(addr), REX_W);
    DynasmApi::push(mc, 0xF7);
    encode_mem_reg_plus_scaled_reg_plus_const(mc, addr, 0);
    writeimm32(mc, immed);
}

/// `TEST_ji` — `test qword [abs32], imm32`.
pub(crate) fn test_ji(mc: &mut Assembler, abs_addr: i32, immed: i32) {
    encode_rex(mc, 0, REX_W);
    DynasmApi::push(mc, 0xF7);
    DynasmApi::push(mc, 0x04);
    DynasmApi::push(mc, 0x25);
    writeimm32(mc, abs_addr);
    writeimm32(mc, immed);
}

/// `TEST_bi` — `test qword [rbp + ofs], imm32`.
pub(crate) fn test_bi(mc: &mut Assembler, offset: i32, immed: i32) {
    encode_rex(mc, 0, REX_W);
    DynasmApi::push(mc, 0xF7);
    encode_stack_bp(mc, offset, false, 0);
    writeimm32(mc, immed);
}

/// `TEST_br` — `test qword [rbp + ofs], r64`.
pub(crate) fn test_br(mc: &mut Assembler, offset: i32, src: u8) {
    op_bp(mc, RexKind::W, 0, &[0x85], src, offset);
}

/// `TEST8_ji` — `test byte [abs32], imm8`. `encode_abs`: modrm `0x04`, sib `0x25`, disp32.
pub fn test8_ji(mc: &mut Assembler, abs_addr: i32, immed: i32) {
    DynasmApi::push(mc, 0xF6);
    DynasmApi::push(mc, 0x04);
    DynasmApi::push(mc, 0x25);
    writeimm32(mc, abs_addr);
    writeimm8(mc, immed);
}

/// `TEST8_ai` — `test byte [base + index*scale + ofs], imm8`.
pub(crate) fn test8_ai(mc: &mut Assembler, addr: (i16, u8, u8, i32), immed: i32) {
    encode_rex_opt(mc, rex_mem_reg_plus_scaled_reg_plus_const(addr));
    DynasmApi::push(mc, 0xF6);
    encode_mem_reg_plus_scaled_reg_plus_const(mc, addr, 0);
    writeimm8(mc, immed);
}

/// `OR8_mi` — `or byte [base + ofs], imm8`.
pub fn or8_mi(mc: &mut Assembler, mem: (u8, i32), immed: i32) {
    encode_rex_opt(mc, rex_mem_reg_plus_const(mem));
    DynasmApi::push(mc, 0x80);
    encode_m(mc, mem.0, mem.1, 1 << 3);
    writeimm8(mc, immed);
}

/// `OR8_ji` — `or byte [abs32], imm8`. `encode_abs` with `/1`.
pub fn or8_ji(mc: &mut Assembler, abs_addr: i32, immed: i32) {
    DynasmApi::push(mc, 0x80);
    DynasmApi::push(mc, 0x0C);
    DynasmApi::push(mc, 0x25);
    writeimm32(mc, abs_addr);
    writeimm8(mc, immed);
}

/// `BTS_mr` — `bts qword [base + ofs], r64`.
///
/// `rx86.py`: `BTS_mr = insn(rex_w, '\x0F\xAB', register(2,8), mem_reg_plus_const(1))`.
/// WriteBarrierSlowPath uses this with a signed bit offset in the register.
pub fn bts_mr(mc: &mut Assembler, mem: (u8, i32), src: u8) {
    op_mem(mc, RexKind::W, 0, &[0x0F, 0xAB], src, mem.0, mem.1);
}

/// `BTS_jr` — `bts qword [abs32], r64`. `encode_abs`.
pub fn bts_jr(mc: &mut Assembler, abs_addr: i32, src: u8) {
    emit_prefix_rex(mc, 0, RexKind::W, rex_register(src, 8));
    push_bytes(mc, &[0x0F, 0xAB]);
    let orbyte = reg_number_3bits(src) << 3;
    DynasmApi::push(mc, 0x04 | orbyte);
    DynasmApi::push(mc, 0x25);
    writeimm32(mc, abs_addr);
}

/// `SET_ir` — `setcc r8`. `rex_fw` forces a REX so `rsi`/`rdi`/`rbp`/`rsp`
/// address `sil`/`dil`/`bpl`/`spl`. `hw_cond` is the hardware condition
/// (`0x90 + hw_cond` is the opcode).
pub(crate) fn set_ir(mc: &mut Assembler, hw_cond: u8, byte_reg: u8) {
    let reg = byte_reg & !BYTE_REG_FLAG;
    emit_prefix_rex(mc, 0, RexKind::Fw, rex_byte_register(byte_reg, 1));
    DynasmApi::push(mc, 0x0F);
    DynasmApi::push(mc, 0x90 | hw_cond);
    DynasmApi::push(mc, 0xC0 | reg_number_3bits(reg));
}

/// `IMUL_rb` — `imul r64, [rbp + ofs]`.
pub(crate) fn imul_rb(mc: &mut Assembler, dst: u8, offset: i32) {
    op_bp(mc, RexKind::W, 0, &[0x0F, 0xAF], dst, offset);
}

/// `MUL_b` — `mul qword [rbp + ofs]`.
pub(crate) fn mul_b(mc: &mut Assembler, offset: i32) {
    encode_rex(mc, 0, REX_W);
    DynasmApi::push(mc, 0xF7);
    encode_stack_bp(mc, offset, false, 4 << 3);
}

/// `NOT_b` — `not qword [rbp + ofs]`.
pub(crate) fn not_b(mc: &mut Assembler, offset: i32) {
    encode_rex(mc, 0, REX_W);
    DynasmApi::push(mc, 0xF7);
    encode_stack_bp(mc, offset, false, 2 << 3);
}

/// `PUS1_b` — `push qword [rbp + ofs]`.
pub(crate) fn push_b(mc: &mut Assembler, offset: i32) {
    DynasmApi::push(mc, 0xFF);
    encode_stack_bp(mc, offset, false, 6 << 3);
}

/// `PUS1_m` — `push qword [base + ofs]`.
pub(crate) fn push_m(mc: &mut Assembler, mem: (u8, i32)) {
    encode_rex_opt(mc, rex_mem_reg_plus_const(mem));
    DynasmApi::push(mc, 0xFF);
    encode_m(mc, mem.0, mem.1, 6 << 3);
}

/// `PUS1_j` — `push qword [abs32]`. `encode_abs`: modrm `0x04 | 6<<3`,
/// sib `0x25`, disp32.
pub(crate) fn push_j(mc: &mut Assembler, abs_addr: i32) {
    DynasmApi::push(mc, 0xFF);
    DynasmApi::push(mc, 0x34);
    DynasmApi::push(mc, 0x25);
    writeimm32(mc, abs_addr);
}

/// `INC_m` — `inc qword [base + ofs]`.
pub(crate) fn inc_m(mc: &mut Assembler, mem: (u8, i32)) {
    encode_rex(mc, rex_mem_reg_plus_const(mem), REX_W);
    DynasmApi::push(mc, 0xFF);
    encode_m(mc, mem.0, mem.1, 0);
}

/// `INC_j` — `inc qword [abs32]`. `encode_abs`: modrm `0x04 | 0<<3`,
/// sib `0x25`, disp32.
pub(crate) fn inc_j(mc: &mut Assembler, abs_addr: i32) {
    encode_rex(mc, 0, REX_W);
    DynasmApi::push(mc, 0xFF);
    DynasmApi::push(mc, 0x04);
    DynasmApi::push(mc, 0x25);
    writeimm32(mc, abs_addr);
}

/// `CALL_b` — `call qword [rbp + ofs]`. No REX.W (`rx86.py` `CALL_b`).
pub fn call_b(mc: &mut Assembler, offset: i32) {
    DynasmApi::push(mc, 0xFF);
    encode_stack_bp(mc, offset, false, 2 << 3);
}

/// `CALL_s` — `call qword [rsp + ofs]`. No REX.W (`rx86.py` `CALL_s`).
pub fn call_s(mc: &mut Assembler, offset: i32) {
    DynasmApi::push(mc, 0xFF);
    encode_stack_sp(mc, offset, 2 << 3);
}

/// `PO1_b` — `pop qword [rbp + ofs]`.
pub(crate) fn pop_b(mc: &mut Assembler, offset: i32) {
    DynasmApi::push(mc, 0x8F);
    encode_stack_bp(mc, offset, false, 0);
}

/// `PUS1_r` — `push r64`. `rex_nw`: REX.B for `r8`–`r15`, no REX.W.
/// The opcode low bits carry the register (`0x50 + reg`).
pub(crate) fn push_r(mc: &mut Assembler, reg: u8) {
    encode_rex_opt(mc, rex_register(reg, 1));
    DynasmApi::push(mc, 0x50 | reg_number_3bits(reg));
}

/// `PUS1_i8` — `push imm8` (`0x6A`).
pub(crate) fn push_i8(mc: &mut Assembler, imm: i8) {
    DynasmApi::push(mc, 0x6A);
    DynasmApi::push(mc, imm as u8);
}

/// `PUS1_i32` — `push imm32` (`0x68`).
pub(crate) fn push_i32(mc: &mut Assembler, imm: i32) {
    DynasmApi::push(mc, 0x68);
    writeimm32(mc, imm);
}

// SSE

/// `MOVSD_xm` — `movsd xmm, [base + ofs]`.
pub fn movsd_xm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::Nw, 0xF2, &[0x0F, 0x10], dst, mem.0, mem.1);
}

/// `MOVSD_xj` — `movsd xmm, [abs]`. `encode_abs`: modrm `0x04|reg`, sib `0x25`, disp32.
pub fn movsd_xj(mc: &mut Assembler, dst: u8, abs_addr: i32) {
    emit_prefix_rex(mc, 0xF2, RexKind::Nw, rex_register(dst, 8));
    push_bytes(mc, &[0x0F, 0x10]);
    let orbyte = reg_number_3bits(dst) << 3;
    DynasmApi::push(mc, 0x04 | orbyte);
    DynasmApi::push(mc, 0x25);
    writeimm32(mc, abs_addr);
}

/// `MOVSD_mx` — `movsd [base + ofs], xmm`.
pub fn movsd_mx(mc: &mut Assembler, mem: (u8, i32), src: u8) {
    op_mem(mc, RexKind::Nw, 0xF2, &[0x0F, 0x11], src, mem.0, mem.1);
}

/// `MOVSD_xa` — `movsd xmm, [base + index*scale + ofs]`.
pub fn movsd_xa(mc: &mut Assembler, dst: u8, addr: (i16, u8, u8, i32)) {
    op_addr(mc, RexKind::Nw, 0xF2, &[0x0F, 0x10], dst, addr);
}

/// `MOVSD_ax` — `movsd [base + index*scale + ofs], xmm`.
pub fn movsd_ax(mc: &mut Assembler, addr: (i16, u8, u8, i32), src: u8) {
    op_addr(mc, RexKind::Nw, 0xF2, &[0x0F, 0x11], src, addr);
}

/// `MOVSD_xb` — `movsd xmm, [rbp + ofs]`.
pub(crate) fn movsd_xb(mc: &mut Assembler, dst: u8, offset: i32) {
    op_bp(mc, RexKind::Nw, 0xF2, &[0x0F, 0x10], dst, offset);
}

/// `MOVSD_bx` — `movsd [rbp + ofs], xmm`.
pub(crate) fn movsd_bx(mc: &mut Assembler, offset: i32, src: u8) {
    op_bp(mc, RexKind::Nw, 0xF2, &[0x0F, 0x11], src, offset);
}

/// `MOVSD_sx` — `movsd [rsp + ofs], xmm`.
pub(crate) fn movsd_sx(mc: &mut Assembler, offset: i32, src: u8) {
    op_sp(mc, RexKind::Nw, 0xF2, &[0x0F, 0x11], src, offset);
}

/// `MOVSD_xs` — `movsd xmm, [rsp + ofs]`.
pub(crate) fn movsd_xs(mc: &mut Assembler, dst: u8, offset: i32) {
    op_sp(mc, RexKind::Nw, 0xF2, &[0x0F, 0x10], dst, offset);
}

/// `MOVSS_xm` — `movss xmm, [base + ofs]`.
pub fn movss_xm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::Nw, 0xF3, &[0x0F, 0x10], dst, mem.0, mem.1);
}

/// `MOVSS_mx` — `movss [base + ofs], xmm`.
pub fn movss_mx(mc: &mut Assembler, mem: (u8, i32), src: u8) {
    op_mem(mc, RexKind::Nw, 0xF3, &[0x0F, 0x11], src, mem.0, mem.1);
}

/// `MOVSS_xa` — `movss xmm, [base + index*scale + ofs]`.
pub fn movss_xa(mc: &mut Assembler, dst: u8, addr: (i16, u8, u8, i32)) {
    op_addr(mc, RexKind::Nw, 0xF3, &[0x0F, 0x10], dst, addr);
}

/// `MOVSS_ax` — `movss [base + index*scale + ofs], xmm`.
pub fn movss_ax(mc: &mut Assembler, addr: (i16, u8, u8, i32), src: u8) {
    op_addr(mc, RexKind::Nw, 0xF3, &[0x0F, 0x11], src, addr);
}

/// `MOVUPS_mx` — `movups [base + ofs], xmm`.
pub fn movups_mx(mc: &mut Assembler, mem: (u8, i32), src: u8) {
    op_mem(mc, RexKind::Nw, 0, &[0x0F, 0x11], src, mem.0, mem.1);
}

/// `MOVUPS_ax` — `movups [base + index*scale + ofs], xmm`.
pub fn movups_ax(mc: &mut Assembler, addr: (i16, u8, u8, i32), src: u8) {
    op_addr(mc, RexKind::Nw, 0, &[0x0F, 0x11], src, addr);
}

/// `MOVQ_mx` — `movq [base + ofs], xmm`.
pub(crate) fn movq_mx(mc: &mut Assembler, mem: (u8, i32), src: u8) {
    op_mem(mc, RexKind::Nw, 0x66, &[0x0F, 0xD6], src, mem.0, mem.1);
}

/// `ADDSD_xx` — `addsd xmm, xmm`.
pub(crate) fn addsd_xx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0xF2, &[0x0F, 0x58], dst, src);
}

/// `SUBSD_xx` — `subsd xmm, xmm`.
pub(crate) fn subsd_xx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0xF2, &[0x0F, 0x5C], dst, src);
}

/// `MULSD_xx` — `mulsd xmm, xmm`.
pub(crate) fn mulsd_xx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0xF2, &[0x0F, 0x59], dst, src);
}

/// `DIVSD_xx` — `divsd xmm, xmm`.
pub(crate) fn divsd_xx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0xF2, &[0x0F, 0x5E], dst, src);
}

/// `UCOMISD_xx` — `ucomisd xmm, xmm`.
pub(crate) fn ucomisd_xx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0x66, &[0x0F, 0x2E], dst, src);
}

/// `define_modrm_modes` memory forms for `ADDSD`/`SUBSD`/`MULSD`/`DIVSD`
/// (`0xF2 0F`) and `UCOMISD` (`0x66 0F 2E`). `'b'` is `[rbp+disp]`, `'m'` is
/// `[base+disp]`, `'j'` is `encode_abs`.
fn sse_xb(mc: &mut Assembler, prefix: u8, opcode: u8, dst: u8, offset: i32) {
    op_bp(mc, RexKind::Nw, prefix, &[0x0F, opcode], dst, offset);
}

fn sse_xm(mc: &mut Assembler, prefix: u8, opcode: u8, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::Nw, prefix, &[0x0F, opcode], dst, mem.0, mem.1);
}

fn sse_xj(mc: &mut Assembler, prefix: u8, opcode: u8, dst: u8, abs_addr: i32) {
    emit_prefix_rex(mc, prefix, RexKind::Nw, rex_register(dst, 8));
    push_bytes(mc, &[0x0F, opcode]);
    let orbyte = reg_number_3bits(dst) << 3;
    DynasmApi::push(mc, 0x04 | orbyte);
    DynasmApi::push(mc, 0x25);
    writeimm32(mc, abs_addr);
}

/// `ADDSD_xb` — `addsd xmm, [rbp + ofs]`.
pub(crate) fn addsd_xb(mc: &mut Assembler, dst: u8, offset: i32) {
    sse_xb(mc, 0xF2, 0x58, dst, offset);
}

/// `ADDSD_xm` — `addsd xmm, [base + ofs]`.
pub(crate) fn addsd_xm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    sse_xm(mc, 0xF2, 0x58, dst, mem);
}

/// `ADDSD_xj` — `addsd xmm, [abs]`.
pub(crate) fn addsd_xj(mc: &mut Assembler, dst: u8, abs_addr: i32) {
    sse_xj(mc, 0xF2, 0x58, dst, abs_addr);
}

/// `SUBSD_xb` — `subsd xmm, [rbp + ofs]`.
pub(crate) fn subsd_xb(mc: &mut Assembler, dst: u8, offset: i32) {
    sse_xb(mc, 0xF2, 0x5C, dst, offset);
}

/// `SUBSD_xm` — `subsd xmm, [base + ofs]`.
pub(crate) fn subsd_xm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    sse_xm(mc, 0xF2, 0x5C, dst, mem);
}

/// `SUBSD_xj` — `subsd xmm, [abs]`.
pub(crate) fn subsd_xj(mc: &mut Assembler, dst: u8, abs_addr: i32) {
    sse_xj(mc, 0xF2, 0x5C, dst, abs_addr);
}

/// `MULSD_xb` — `mulsd xmm, [rbp + ofs]`.
pub(crate) fn mulsd_xb(mc: &mut Assembler, dst: u8, offset: i32) {
    sse_xb(mc, 0xF2, 0x59, dst, offset);
}

/// `MULSD_xm` — `mulsd xmm, [base + ofs]`.
pub(crate) fn mulsd_xm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    sse_xm(mc, 0xF2, 0x59, dst, mem);
}

/// `MULSD_xj` — `mulsd xmm, [abs]`.
pub(crate) fn mulsd_xj(mc: &mut Assembler, dst: u8, abs_addr: i32) {
    sse_xj(mc, 0xF2, 0x59, dst, abs_addr);
}

/// `DIVSD_xb` — `divsd xmm, [rbp + ofs]`.
pub(crate) fn divsd_xb(mc: &mut Assembler, dst: u8, offset: i32) {
    sse_xb(mc, 0xF2, 0x5E, dst, offset);
}

/// `DIVSD_xm` — `divsd xmm, [base + ofs]`.
pub(crate) fn divsd_xm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    sse_xm(mc, 0xF2, 0x5E, dst, mem);
}

/// `DIVSD_xj` — `divsd xmm, [abs]`.
pub(crate) fn divsd_xj(mc: &mut Assembler, dst: u8, abs_addr: i32) {
    sse_xj(mc, 0xF2, 0x5E, dst, abs_addr);
}

/// `UCOMISD_xb` — `ucomisd xmm, [rbp + ofs]`.
pub(crate) fn ucomisd_xb(mc: &mut Assembler, dst: u8, offset: i32) {
    sse_xb(mc, 0x66, 0x2E, dst, offset);
}

/// `UCOMISD_xm` — `ucomisd xmm, [base + ofs]`.
pub(crate) fn ucomisd_xm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    sse_xm(mc, 0x66, 0x2E, dst, mem);
}

/// `UCOMISD_xj` — `ucomisd xmm, [abs]`.
pub(crate) fn ucomisd_xj(mc: &mut Assembler, dst: u8, abs_addr: i32) {
    sse_xj(mc, 0x66, 0x2E, dst, abs_addr);
}

/// `SQRTSD_xx` — `sqrtsd xmm, xmm`.
pub(crate) fn sqrtsd_xx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0xF2, &[0x0F, 0x51], dst, src);
}

/// `MOVAPD_xx` — `movapd xmm, xmm`.
pub(crate) fn movapd_xx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0x66, &[0x0F, 0x28], dst, src);
}

/// `PXOR_xx` — `pxor xmm, xmm`.
pub(crate) fn pxor_xx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0x66, &[0x0F, 0xEF], dst, src);
}

/// `XORPD_xx` — `xorpd xmm, xmm`.
pub(crate) fn xorpd_xx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0x66, &[0x0F, 0x57], dst, src);
}

/// `XORPD_xm` — `xorpd xmm, [base + ofs]`.
pub(crate) fn xorpd_xm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::Nw, 0x66, &[0x0F, 0x57], dst, mem.0, mem.1);
}

/// `XORPD_xj` — `xorpd xmm, [abs]`. `encode_abs`: modrm `0x04|reg`, sib `0x25`, disp32.
pub(crate) fn xorpd_xj(mc: &mut Assembler, dst: u8, abs_addr: i32) {
    emit_pd_xj(mc, dst, abs_addr, 0x57);
}

/// `ANDPD_xx` — `andpd xmm, xmm`.
pub(crate) fn andpd_xx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0x66, &[0x0F, 0x54], dst, src);
}

/// `ANDPD_xm` — `andpd xmm, [base + ofs]`.
pub(crate) fn andpd_xm(mc: &mut Assembler, dst: u8, mem: (u8, i32)) {
    op_mem(mc, RexKind::Nw, 0x66, &[0x0F, 0x54], dst, mem.0, mem.1);
}

/// `ANDPD_xj` — `andpd xmm, [abs]`.
pub(crate) fn andpd_xj(mc: &mut Assembler, dst: u8, abs_addr: i32) {
    emit_pd_xj(mc, dst, abs_addr, 0x54);
}

/// Absolute `xorpd`/`andpd` (`*_xj`): prefix `0x66`, `rex_nw`, `0F xx`, then `encode_abs`.
fn emit_pd_xj(mc: &mut Assembler, dst: u8, abs_addr: i32, opcode: u8) {
    emit_prefix_rex(mc, 0x66, RexKind::Nw, rex_register(dst, 8));
    push_bytes(mc, &[0x0F, opcode]);
    let orbyte = reg_number_3bits(dst) << 3;
    DynasmApi::push(mc, 0x04 | orbyte);
    DynasmApi::push(mc, 0x25);
    writeimm32(mc, abs_addr);
}

/// `XORPS_xx` — `xorps xmm, xmm`.
pub(crate) fn xorps_xx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0, &[0x0F, 0x57], dst, src);
}

/// `CVTSD2SS_xx` — `cvtsd2ss xmm, xmm`.
pub(crate) fn cvtsd2ss_xx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0xF2, &[0x0F, 0x5A], dst, src);
}

/// `CVTSS2SD_xx` — `cvtss2sd xmm, xmm`.
pub(crate) fn cvtss2sd_xx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0xF3, &[0x0F, 0x5A], dst, src);
}

/// `CVTSI2SD_xr` — `cvtsi2sd xmm, r64`.
pub(crate) fn cvtsi2sd_xr(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::W, 0xF2, &[0x0F, 0x2A], dst, src);
}

/// `CVTTSD2SI_rx` — `cvttsd2si r64, xmm`.
pub(crate) fn cvttsd2si_rx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::W, 0xF2, &[0x0F, 0x2C], dst, src);
}

/// `CVTTSD2SI_rb` — `cvttsd2si r64, [rbp + ofs]`.
pub(crate) fn cvttsd2si_rb(mc: &mut Assembler, dst: u8, offset: i32) {
    op_bp(mc, RexKind::W, 0xF2, &[0x0F, 0x2C], dst, offset);
}

/// `MOVDQ_xr` — `movq xmm, r64`.
pub(crate) fn movdq_xr(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::W, 0x66, &[0x0F, 0x6E], dst, src);
}

/// `MOVDQ_rx` — `movq r64, xmm`.
pub(crate) fn movdq_rx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::W, 0x66, &[0x0F, 0x7E], src, dst);
}

/// `MOVD32_xr` — `movd xmm, r32`. `rex_nw`, unlike `MOVDQ_xr`.
pub(crate) fn movd32_xr(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0x66, &[0x0F, 0x6E], dst, src);
}

/// `MOVD32_rx` — `movd r32, xmm`. The ModRM reg field is the xmm.
pub(crate) fn movd32_rx(mc: &mut Assembler, dst: u8, src: u8) {
    op_rr(mc, RexKind::Nw, 0x66, &[0x0F, 0x7E], src, dst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynasmrt::dynasm;

    fn finish(mc: Assembler) -> Vec<u8> {
        mc.finalize().unwrap()
    }

    fn enc(f: impl FnOnce(&mut Assembler)) -> Vec<u8> {
        let mut mc = Assembler::new(0);
        f(&mut mc);
        finish(mc)
    }

    fn assert_same(got: &[u8], expected: &[u8]) {
        assert_eq!(got, expected, "encoder {got:02x?} dynasm {expected:02x?}");
    }

    #[test]
    fn test_not_push_test_memory_forms() {
        // `NOT_b`, `PUS1_m`, `PUS1_j`, `TEST_ri`/`_mi`/`_ai`/`_ji`/`_bi`/
        // `_br` and `MOV32_si` against dynasm's encodings.
        let got = enc(|mc| {
            not_b(mc, -16);
            push_m(mc, (R12, 8));
            push_m(mc, (ECX, 0x1234));
            push_j(mc, 0x1000);
            test_ri(mc, R9, 0x7f00_0000);
            test_mi(mc, (R13, -8), 3);
            test_ai(mc, (i16::from(EAX), R10, 3, 16), 0x100);
            test_ji(mc, 0x2000, 5);
            test_bi(mc, 24, 0x40);
            test_br(mc, -32, R11);
            mov32_si(mc, 8, -1);
        });
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; not QWORD [rbp - 16]
            ; push QWORD [r12 + 8]
            ; push QWORD [rcx + 0x1234]
            ; push QWORD [0x1000]
            ; test r9, 0x7f00_0000
            ; test QWORD [r13 - 8], 3
            ; test QWORD [rax + r10 * 8 + 16], 0x100
            ; test QWORD [0x2000], 5
            ; test QWORD [rbp + 24], 0x40
            ; test QWORD [rbp - 32], r11
            ; mov DWORD [rsp + 8], -1
        );
        assert_same(&got, &finish(d));
    }

    #[test]
    fn single_byte_and_fits_in_32bits() {
        assert!(single_byte(-128));
        assert!(single_byte(127));
        assert!(!single_byte(128));
        assert!(!single_byte(-129));
        assert!(fits_in_32bits(-2147483648));
        assert!(fits_in_32bits(2147483647));
        assert!(!fits_in_32bits(2147483648));
        assert!(!fits_in_32bits(-2147483649));
    }

    #[test]
    fn test_mov_ri_64() {
        let got = enc(|mc| {
            mov_ri(mc, ECX, -2);
            mov_ri(mc, R15, -3);
            mov_ri(mc, EBX, -0x8000_0003);
            mov_ri(mc, R13, -0x8000_0002);
            mov_ri(mc, ECX, 42);
            mov_ri(mc, R12, 0x8000_0042);
            mov_ri(mc, R12, 0x1_0000_0007);
        });
        let expected = [
            0x48, 0xC7, 0xC1, 0xFE, 0xFF, 0xFF, 0xFF, 0x49, 0xC7, 0xC7, 0xFD, 0xFF, 0xFF, 0xFF,
            0x48, 0xBB, 0xFD, 0xFF, 0xFF, 0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0x49, 0xBD, 0xFE, 0xFF,
            0xFF, 0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xB9, 0x2A, 0x00, 0x00, 0x00, 0x41, 0xBC, 0x42,
            0x00, 0x00, 0x80, 0x49, 0xBC, 0x07, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        ];
        assert_eq!(got, expected);
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; mov rcx, -2
            ; mov r15, -3
            ; mov rbx, -0x80000003i64
            ; mov r13, -0x80000002i64
            ; mov ecx, 42
            ; mov r12d, -2147483582
            ; mov r12, 0x100000007i64
        );
        assert_same(&got, &finish(d));
    }

    #[test]
    fn test_mov_rm_64() {
        let got = enc(|mc| {
            mov_rm(mc, EDX, (EDI, 0));
            mov_rm(mc, EDX, (R12, 0));
            mov_rm(mc, EDX, (R13, 0));
        });
        assert_eq!(
            got,
            [
                0x48, 0x8B, 0x17, 0x49, 0x8B, 0x14, 0x24, 0x49, 0x8B, 0x55, 0x00
            ]
        );
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; mov rdx, [rdi]
            ; mov rdx, [r12]
            ; mov rdx, [r13]
        );
        assert_same(&got, &finish(d));
    }

    #[test]
    fn test_mov_rm_negative_64() {
        let got = enc(|mc| mov_rm(mc, EDX, (EDI, -1)));
        assert_eq!(got, [0x48, 0x8B, 0x57, 0xFF]);
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; mov rdx, [rdi - 1]
        );
        assert_same(&got, &finish(d));
    }

    #[test]
    fn displacement_edges_and_rbp() {
        let got = enc(|mc| {
            mov_rm(mc, EDX, (EDI, 127));
            mov_rm(mc, EDX, (EDI, 128));
            mov_rm(mc, EDX, (EDI, -128));
            mov_rm(mc, EDX, (EDI, -129));
            mov_rm(mc, EDX, (EBP, 0));
            mov_rm(mc, EDX, (R12, 127));
            mov_rm(mc, EDX, (R12, 128));
            mov_rm(mc, EDX, (R13, -128));
            mov_rm(mc, EDX, (R13, -129));
            mov_mr(mc, (EBP, 0), EDX);
        });
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; mov rdx, [rdi + 127]
            ; mov rdx, [rdi + 128]
            ; mov rdx, [rdi - 128]
            ; mov rdx, [rdi - 129]
            ; mov rdx, [rbp]
            ; mov rdx, [r12 + 127]
            ; mov rdx, [r12 + 128]
            ; mov rdx, [r13 - 128]
            ; mov rdx, [r13 - 129]
            ; mov [rbp], rdx
        );
        assert_same(&got, &finish(d));
    }

    #[test]
    fn high_registers_reg_and_base_and_index() {
        let got = enc(|mc| {
            mov_rm(mc, R8, (R9, 0));
            mov_rm(mc, R15, (R12, 0));
            mov_mr(mc, (R13, 4), R10);
            mov32_rm(mc, R11, (R14, 0));
            lea_rm(mc, R15, (R14, 0));
            lea_ra(mc, R9, (i16::from(R10), R11, 3, 0));
            lea_ra(mc, EDX, (i16::from(R13), EDI, 2, 0));
            lea_ra(mc, EDX, (i16::from(R12), R8, 1, 128));
            mov_ra(mc, R8, (i16::from(ESI), R15, 2, -128));
            movzx8_rm(mc, R10, (R11, 16));
            movsx32_rm(mc, R12, (R13, 0));
            cmp_rm(mc, R14, (R15, 127));
            cmp_ri(mc, R8, -129);
            add_ri(mc, R9, 127);
            add_ri(mc, R9, 128);
        });
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; mov r8, [r9]
            ; mov r15, [r12]
            ; mov [r13 + 4], r10
            ; mov r11d, [r14]
            ; lea r15, [r14]
            ; lea r9, [r10 + r11 * 8]
            ; lea rdx, [r13 + rdi * 4]
            ; lea rdx, [r12 + r8 * 2 + 128]
            ; mov r8, [rsi + r15 * 4 - 128]
            ; movzx r10, BYTE [r11 + 16]
            ; movsxd r12, DWORD [r13]
            ; cmp r14, [r15 + 127]
            ; cmp r8, -129
            ; add r9, 127
            ; add r9, 128
        );
        assert_same(&got, &finish(d));
    }

    #[test]
    fn assert_encodes_as_64() {
        // 64-bit forms of the `assert_encodes_as` rows in `test_rx86.py`.
        // `SHL_ri`/`SHR_ri`/`SAR_ri` use opcode `D1` when the count is 1.
        // dynasm has no `D1` row, so `shl rdx, 1` is the longer `C1 /r, 1`.
        let shift_once = enc(|mc| {
            shl_ri(mc, EDX, 1);
            shr_ri(mc, EDX, 1);
            sar_ri(mc, EDX, 1);
        });
        assert_eq!(
            shift_once,
            [0x48, 0xD1, 0xE2, 0x48, 0xD1, 0xEA, 0x48, 0xD1, 0xFA]
        );
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; shl rdx, 1
            ; shr rdx, 1
            ; sar rdx, 1
        );
        let dynasm_once = finish(d);
        assert_eq!(
            dynasm_once,
            [
                0x48, 0xC1, 0xE2, 0x01, 0x48, 0xC1, 0xEA, 0x01, 0x48, 0xC1, 0xFA, 0x01
            ]
        );
        assert!(dynasm_once.len() > shift_once.len());

        let shifts = enc(|mc| {
            shl_ri(mc, EDX, 5);
            shr_ri(mc, EDX, 5);
            sar_ri(mc, EDX, 5);
        });
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; shl rdx, 5
            ; shr rdx, 5
            ; sar rdx, 5
        );
        assert_same(&shifts, &finish(d));

        let test8 = enc(|mc| test8_mi(mc, (EDX, 16), 99));
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; test BYTE [rdx + 16], 99
        );
        assert_same(&test8, &finish(d));
        assert_eq!(test8, [0xF6, 0x42, 0x10, 0x63]);

        // `SUB_mi` picks `83 /5 ib` for a byte immediate. dynasm matches
        // `m*i*` (opcode `81`, imm32) before `m*ib`.
        let sub_mi8 = enc(|mc| sub_mi(mc, (EDX, 16), 55));
        assert_eq!(sub_mi8, [0x48, 0x83, 0x6A, 0x10, 0x37]);
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; sub QWORD [rdx + 16], 55
        );
        let dynasm_sub = finish(d);
        assert_eq!(dynasm_sub, [0x48, 0x81, 0x6A, 0x10, 0x37, 0x00, 0x00, 0x00]);
        assert!(dynasm_sub.len() > sub_mi8.len());

        let imul = enc(|mc| {
            imul_rri(mc, EBX, ECX, 0x0123_4567);
            imul_rri(mc, EBX, ECX, 0x2A);
        });
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; imul rbx, rcx, 0x01234567
            ; imul rbx, rcx, 0x2A
        );
        assert_same(&imul, &finish(d));
    }

    #[test]
    fn rex_fw_is_longer_than_dynasm_without_an_empty_rex() {
        // `rex_fw` always emits `0x40`. A static byte instruction whose
        // registers sit below `spl` does not need that prefix.
        let mov8 = enc(|mc| mov8_mi(mc, (EDX, 16), 99));
        assert_eq!(mov8, [0x40, 0xC6, 0x42, 0x10, 0x63]);
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; mov BYTE [rdx + 16], 99
        );
        let dynasm_mov8 = finish(d);
        assert_eq!(dynasm_mov8, [0xC6, 0x42, 0x10, 0x63]);
        assert!(dynasm_mov8.len() < mov8.len());

        let mov8_a = enc(|mc| mov8_ai(mc, (i16::from(EBX), ECX, 2, 16), 99));
        assert_eq!(mov8_a, [0x40, 0xC6, 0x44, 0x8B, 0x10, 0x63]);
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; mov BYTE [rbx + rcx * 4 + 16], 99
        );
        assert_eq!(finish(d), [0xC6, 0x44, 0x8B, 0x10, 0x63]);
    }

    #[test]
    fn memory_and_immediate_forms_match_dynasm() {
        let got = enc(|mc| {
            mov_mr(mc, (EDI, 0), EDX);
            mov_mr(mc, (EDI, -128), EDX);
            mov_mr(mc, (EDI, 128), EDX);
            mov32_rm(mc, ECX, (EAX, 16));
            mov8_mr(mc, (R8, 0), ESI);
            movzx8_rm(mc, ECX, (EAX, 16));
            movzx16_rm(mc, ECX, (EAX, 16));
            movsx8_rm(mc, ECX, (EAX, 16));
            movsx16_rm(mc, ECX, (EAX, 16));
            movsx32_rm(mc, ECX, (EAX, 16));
            lea_ra(mc, EDX, (i16::from(ESI), EDI, 2, 0));
            lea_ra(mc, EDX, (NO_BASE_REGISTER, EDI, 2, 0xCD));
            mov_mi(mc, (EDI, 0), 0);
            mov32_mi(mc, (EDX, 16), 99);
            mov16_mi(mc, (EDX, 16), 99);
            cmp_mi(mc, (EDX, 16), 200);
            cmp_rm(mc, EDX, (EDI, 0));
            cmp_mr(mc, (EDI, 8), EDX);
            add_ri(mc, ECX, 1);
            sub_ri(mc, ECX, 200);
            and_ri(mc, ECX, -1);
            or_ri(mc, ECX, 2);
            xor_ri(mc, ECX, 128);
            sub_rm(mc, EDX, (EBP, 8));
            movsd_xm(mc, 2, (EDI, 8));
            movsd_mx(mc, (ESP, 32), 2);
            movss_xm(mc, 3, (EAX, 4));
            movss_mx(mc, (R9, 0), 3);
            movq_mx(mc, (ESP, 8), 2);
            imul_rmi(mc, EAX, (EDI, 8), 3);
            mul_b(mc, 16);
            or8_mi(mc, (EDX, 1), 0x20);
            push_b(mc, 8);
            pop_b(mc, 8);
        });
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; mov [rdi], rdx
            ; mov [rdi - 128], rdx
            ; mov [rdi + 128], rdx
            ; mov ecx, [rax + 16]
            ; mov [r8], sil
            ; movzx rcx, BYTE [rax + 16]
            ; movzx rcx, WORD [rax + 16]
            ; movsx rcx, BYTE [rax + 16]
            ; movsx rcx, WORD [rax + 16]
            ; movsxd rcx, DWORD [rax + 16]
            ; lea rdx, [rsi + rdi * 4]
            ; lea rdx, [rdi * 4 + 0xCD]
            ; mov QWORD [rdi], 0
            ; mov DWORD [rdx + 16], 99
            ; mov WORD [rdx + 16], 99
            ; cmp QWORD [rdx + 16], 200
            ; cmp rdx, [rdi]
            ; cmp QWORD [rdi + 8], rdx
            ; add rcx, 1
            ; sub rcx, 200
            ; and rcx, -1
            ; or rcx, 2
            ; xor rcx, 128
            ; sub rdx, [rbp + 8]
            ; movsd xmm2, [rdi + 8]
            ; movsd [rsp + 32], xmm2
            ; movss xmm3, [rax + 4]
            ; movss [r9], xmm3
            ; movq [rsp + 8], xmm2
            ; imul rax, [rdi + 8], 3
            ; mul QWORD [rbp + 16]
            ; or BYTE [rdx + 1], 0x20
            ; push QWORD [rbp + 8]
            ; pop QWORD [rbp + 8]
        );
        assert_same(&got, &finish(d));

        // `CMP_mi` / `CMP_ri` use `83 /r ib`. dynasm's `v*i*`
        // row (opcode `81`, imm32) is matched first, including for -128,
        // which `single_byte` still accepts.
        let cmp_imm8 = enc(|mc| {
            cmp_mi(mc, (EDX, 16), 55);
            cmp_ri(mc, R8, -128);
        });
        assert_eq!(
            cmp_imm8,
            [0x48, 0x83, 0x7A, 0x10, 0x37, 0x49, 0x83, 0xF8, 0x80]
        );
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; cmp QWORD [rdx + 16], 55
            ; cmp r8, -128
        );
        let dynasm_cmp = finish(d);
        assert!(dynasm_cmp.len() > cmp_imm8.len());
        assert_eq!(
            dynasm_cmp,
            [
                0x48, 0x81, 0x7A, 0x10, 0x37, 0x00, 0x00, 0x00, 0x49, 0x81, 0xF8, 0x80, 0xFF, 0xFF,
                0xFF,
            ]
        );
    }

    /// `(low, low)`, `(low, high)`, `(high, low)`, `(high, high)`.
    fn quad(encode: fn(&mut Assembler, u8, u8)) -> Vec<u8> {
        enc(|mc| {
            encode(mc, 1, 0);
            encode(mc, 1, 8);
            encode(mc, 9, 0);
            encode(mc, 9, 8);
        })
    }

    /// GPR destination: `(dst, src)` is `(gpr, xmm)` in the same four classes.
    fn quad_rx(encode: fn(&mut Assembler, u8, u8)) -> Vec<u8> {
        enc(|mc| {
            encode(mc, 0, 1);
            encode(mc, 8, 1);
            encode(mc, 0, 9);
            encode(mc, 8, 9);
        })
    }

    #[test]
    fn sse_xx_matches_static_dynasm() {
        // Static xmm/gpr operands are the shortest form dynasm emits.
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; addsd xmm1, xmm0 ; addsd xmm1, xmm8 ; addsd xmm9, xmm0 ; addsd xmm9, xmm8);
        assert_same(&quad(addsd_xx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; subsd xmm1, xmm0 ; subsd xmm1, xmm8 ; subsd xmm9, xmm0 ; subsd xmm9, xmm8);
        assert_same(&quad(subsd_xx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; mulsd xmm1, xmm0 ; mulsd xmm1, xmm8 ; mulsd xmm9, xmm0 ; mulsd xmm9, xmm8);
        assert_same(&quad(mulsd_xx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; divsd xmm1, xmm0 ; divsd xmm1, xmm8 ; divsd xmm9, xmm0 ; divsd xmm9, xmm8);
        assert_same(&quad(divsd_xx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; ucomisd xmm1, xmm0 ; ucomisd xmm1, xmm8 ; ucomisd xmm9, xmm0 ; ucomisd xmm9, xmm8);
        assert_same(&quad(ucomisd_xx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; sqrtsd xmm1, xmm0 ; sqrtsd xmm1, xmm8 ; sqrtsd xmm9, xmm0 ; sqrtsd xmm9, xmm8);
        assert_same(&quad(sqrtsd_xx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; movapd xmm1, xmm0 ; movapd xmm1, xmm8 ; movapd xmm9, xmm0 ; movapd xmm9, xmm8);
        assert_same(&quad(movapd_xx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; pxor xmm1, xmm0 ; pxor xmm1, xmm8 ; pxor xmm9, xmm0 ; pxor xmm9, xmm8);
        assert_same(&quad(pxor_xx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; xorpd xmm1, xmm0 ; xorpd xmm1, xmm8 ; xorpd xmm9, xmm0 ; xorpd xmm9, xmm8);
        assert_same(&quad(xorpd_xx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; andpd xmm1, xmm0 ; andpd xmm1, xmm8 ; andpd xmm9, xmm0 ; andpd xmm9, xmm8);
        assert_same(&quad(andpd_xx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; xorps xmm1, xmm0 ; xorps xmm1, xmm8 ; xorps xmm9, xmm0 ; xorps xmm9, xmm8);
        assert_same(&quad(xorps_xx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; cvtsd2ss xmm1, xmm0 ; cvtsd2ss xmm1, xmm8 ; cvtsd2ss xmm9, xmm0 ; cvtsd2ss xmm9, xmm8);
        assert_same(&quad(cvtsd2ss_xx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; cvtss2sd xmm1, xmm0 ; cvtss2sd xmm1, xmm8 ; cvtss2sd xmm9, xmm0 ; cvtss2sd xmm9, xmm8);
        assert_same(&quad(cvtss2sd_xx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; cvtsi2sd xmm1, rax ; cvtsi2sd xmm1, r8 ; cvtsi2sd xmm9, rax ; cvtsi2sd xmm9, r8);
        assert_same(&quad(cvtsi2sd_xr), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; movq xmm1, rax ; movq xmm1, r8 ; movq xmm9, rax ; movq xmm9, r8);
        assert_same(&quad(movdq_xr), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; cvttsd2si rax, xmm1 ; cvttsd2si r8, xmm1 ; cvttsd2si rax, xmm9 ; cvttsd2si r8, xmm9);
        assert_same(&quad_rx(cvttsd2si_rx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; movq rax, xmm1 ; movq r8, xmm1 ; movq rax, xmm9 ; movq r8, xmm9);
        assert_same(&quad_rx(movdq_rx), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; movd xmm1, eax ; movd xmm1, r8d ; movd xmm9, eax ; movd xmm9, r8d);
        assert_same(&quad(movd32_xr), &finish(d));
        let mut d = Assembler::new(0);
        dynasm!(d ; .arch x64 ; movd eax, xmm1 ; movd r8d, xmm1 ; movd eax, xmm9 ; movd r8d, xmm9);
        assert_same(&quad_rx(movd32_rx), &finish(d));
    }

    #[test]
    fn cvttsd2si_rb_matches_dynasm() {
        // `CVTTSD2SI_rb`: `stack_bp` with a disp8 and a disp32.
        let got = enc(|mc| {
            cvttsd2si_rb(mc, EAX, 16);
            cvttsd2si_rb(mc, R8, -0x200);
        });
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; cvttsd2si rax, QWORD [rbp + 16]
            ; cvttsd2si r8, QWORD [rbp - 0x200]
        );
        assert_same(&got, &finish(d));
    }

    #[test]
    fn bts_mr_matches_dynasm_header_displacement() {
        // WriteBarrierSlowPath: `BTS [loc_base + (-GcHeader::SIZE)], r11`.
        let got = enc(|mc| {
            bts_mr(mc, (EDI, -8), R11);
            bts_mr(mc, (R10, -8), R11);
            bts_mr(mc, (R12, -8), R11);
            bts_mr(mc, (R13, -8), R11);
        });
        let mut d = Assembler::new(0);
        dynasm!(d
            ; .arch x64
            ; bts QWORD [rdi - 8], r11
            ; bts QWORD [r10 - 8], r11
            ; bts QWORD [r12 - 8], r11
            ; bts QWORD [r13 - 8], r11
        );
        assert_same(&got, &finish(d));
    }

    #[test]
    fn test8_ji_or8_ji_and_bts_jr_encode_abs() {
        // `encode_abs` on x86-64: modrm `0x04|orbyte`, sib `0x25`, disp32.
        let addr = 0x1234_5678_i32;
        assert_eq!(
            enc(|mc| test8_ji(mc, addr, 0x63)),
            [0xF6, 0x04, 0x25, 0x78, 0x56, 0x34, 0x12, 0x63]
        );
        assert_eq!(
            enc(|mc| or8_ji(mc, addr, 0x01)),
            [0x80, 0x0C, 0x25, 0x78, 0x56, 0x34, 0x12, 0x01]
        );
        // r11 is register field 3 with REX.W|REX.R.
        assert_eq!(
            enc(|mc| bts_jr(mc, addr, R11)),
            [0x4C, 0x0F, 0xAB, 0x1C, 0x25, 0x78, 0x56, 0x34, 0x12]
        );
    }

    #[test]
    fn set_ir_forces_rex_for_the_low_byte() {
        // `SET_ir`: `rex_fw`, `0x0F`, `0x90+cond`, ModRM `0xC0|reg`.
        // sete al, setl sil, setg r8b.
        assert_eq!(
            enc(|mc| set_ir(mc, 4, EAX | BYTE_REG_FLAG)),
            [0x40, 0x0F, 0x94, 0xC0]
        );
        assert_eq!(
            enc(|mc| set_ir(mc, 12, ESI | BYTE_REG_FLAG)),
            [0x40, 0x0F, 0x9C, 0xC6]
        );
        assert_eq!(
            enc(|mc| set_ir(mc, 15, R8 | BYTE_REG_FLAG)),
            [0x41, 0x0F, 0x9F, 0xC0]
        );
    }
}
