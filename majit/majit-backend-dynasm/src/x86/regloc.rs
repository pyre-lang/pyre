//! x86/regloc.py `LocationCodeBuilder`: the instruction layer over `rx86`
//! that dispatches on location codes.
//!
//! Upstream mixes it into `codebuf.MachineCodeBlockWrapper`, so `self.mc` is
//! both the byte buffer and the owner of `_scratch_register_value`. Here it
//! wraps the dynasm `Assembler` and derefs to it: `rx86` encoders and
//! `dynasm!` keep writing through `self.mc`, while every `INSN` below picks
//! the immediate or scratch-staged encoding once.
use std::ops::{Deref, DerefMut};

use dynasmrt::{DynasmApi, dynasm};

use super::assembler::Assembler;
use super::rx86;
use crate::regloc::{Loc, RegLoc, X86_64_SCRATCH_REG};

/// A location as `INSN` reads it: `location_code()` together with the
/// matching `value_*()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operand {
    /// `'r'`: a general-purpose `RegLoc`.
    R(u8),
    /// `'x'`: an xmm `RegLoc`.
    X(u8),
    /// `'b'`: `RawEbpLoc` / `FrameLoc`.
    B(i32),
    /// `'s'`: `RawEspLoc`.
    S(i32),
    /// `'i'`: `ImmedLoc`.
    I(i64),
    /// `'j'`: an absolute address, `heap(addr)` or `ConstFloatLoc`.
    J(i64),
    /// `'m'`: `AddressLoc` `[base + ofs]`.
    M(u8, i64),
    /// `'a'`: `AddressLoc` `[base + (index << scale) + ofs]`; the base may be
    /// `rx86::NO_BASE_REGISTER`.
    A(i16, u8, u8, i64),
}

/// regloc.py `imm(x)`.
pub(crate) fn imm(value: i64) -> Operand {
    Operand::I(value)
}

/// regloc.py `heap(addr)`: the word at an absolute address.
pub(crate) fn heap(addr: i64) -> Operand {
    Operand::J(addr)
}

/// regloc.py `addr_add(reg_or_imm1, reg_or_imm2, offset, scale)`, the
/// `AddressLoc` constructor: an immediate base and an immediate scaled part
/// fold into `'j'`, an immediate base alone becomes the static offset of an
/// `'a'` with no base register, and a register base with an immediate scaled
/// part is `'m'`.
pub(crate) fn addr_add(base_loc: &Loc, scaled_loc: &Loc, static_offset: i64, scale: u8) -> Operand {
    assert!(scale < 4);
    match (Operand::from(base_loc), Operand::from(scaled_loc)) {
        (Operand::I(base), Operand::I(scaled)) => Operand::J(
            base.wrapping_add(scaled << scale)
                .wrapping_add(static_offset),
        ),
        (Operand::I(base), Operand::R(scaled)) => Operand::A(
            rx86::NO_BASE_REGISTER,
            scaled,
            scale,
            base.wrapping_add(static_offset),
        ),
        (Operand::R(base), Operand::I(scaled)) => {
            Operand::M(base, (scaled << scale).wrapping_add(static_offset))
        }
        (Operand::R(base), Operand::R(scaled)) => {
            Operand::A(i16::from(base), scaled, scale, static_offset)
        }
        _ => panic!("AddressLoc of ({base_loc:?}, {scaled_loc:?})"),
    }
}

/// regloc.py `addr_add_const(reg_or_imm1, offset)`.
pub(crate) fn addr_add_const(base_loc: &Loc, offset: i64) -> Operand {
    addr_add(base_loc, &Loc::immed(0), offset, 0)
}

/// regloc.py `RegLoc.lowest8bits` / `ImmedLoc.lowest8bits`: a register's
/// `low_byte`, which every byte instruction encodes with a forced REX prefix
/// (`sil`, `dil`, `r8b`, ... rather than `dh`, `bh`), or an immediate's low
/// byte sign-extended.
pub(crate) fn lowest8bits(loc: &Loc) -> Operand {
    match *loc {
        Loc::Reg(r) => {
            assert!(!r.is_xmm, "lowest8bits of {r:?}");
            Operand::R(rx86::low_byte(r.value))
        }
        Loc::Immed(i) | Loc::ImmedFloat(i) => Operand::I(i64::from(i.value as u8 as i8)),
        _ => panic!("lowest8bits of {loc:?}"),
    }
}

impl From<RegLoc> for Operand {
    fn from(reg: RegLoc) -> Self {
        if reg.is_xmm {
            Operand::X(reg.value)
        } else {
            Operand::R(reg.value)
        }
    }
}

impl From<&Loc> for Operand {
    fn from(loc: &Loc) -> Self {
        match *loc {
            Loc::Reg(r) => r.into(),
            Loc::Frame(f) => Operand::B(f.ebp_loc.value),
            Loc::Ebp(e) => Operand::B(e.value),
            Loc::Esp(e) => Operand::S(e.value),
            Loc::Immed(i) | Loc::ImmedFloat(i) => Operand::I(i.value),
            Loc::ConstFloat(c) => Operand::J(c.value as i64),
            Loc::Addr(a) if a.index == 255 => Operand::M(a.base, i64::from(a.offset)),
            Loc::Addr(a) => Operand::A(i16::from(a.base), a.index, a.scale, i64::from(a.offset)),
        }
    }
}

impl From<Loc> for Operand {
    fn from(loc: Loc) -> Self {
        Operand::from(&loc)
    }
}

impl Operand {
    /// `AddressLoc.find_unused_reg`: a register the address does not read.
    fn find_unused_reg(self) -> u8 {
        let (eax, ecx, edx) = (rx86::EAX, rx86::ECX, rx86::EDX);
        match self {
            Operand::M(base, _) if base == eax => edx,
            Operand::A(base, index, _, _) if base == i16::from(eax) => {
                if index == edx {
                    ecx
                } else {
                    edx
                }
            }
            Operand::A(base, index, _, _) if index == eax => {
                if base == i16::from(edx) {
                    ecx
                } else {
                    edx
                }
            }
            _ => eax,
        }
    }

    fn is_scratch(self) -> bool {
        self == Operand::R(X86_64_SCRATCH_REG.value)
    }
}

/// Whether `rx86` encodes `name` with an absolute `'j'` operand
/// (`has_implementation_for`). Any `NAME_j` has a `NAME_m` fallback, which
/// `INSN` reaches through `_addr_as_reg_offset` when this is false.
fn has_j_form(name: BinOp, loc1: Operand, loc2: Operand) -> bool {
    matches!(
        (name, loc1, loc2),
        (
            BinOp::TEST | BinOp::TEST8 | BinOp::OR8,
            Operand::J(_),
            Operand::I(_)
        ) | (BinOp::BTS, Operand::J(_), Operand::R(_))
            | (
                BinOp::MOVSD
                    | BinOp::ADDSD
                    | BinOp::SUBSD
                    | BinOp::MULSD
                    | BinOp::DIVSD
                    | BinOp::UCOMISD
                    | BinOp::XORPD
                    | BinOp::ANDPD,
                Operand::X(_),
                Operand::J(_)
            )
    )
}

fn mem32(base: u8, ofs: i64) -> (u8, i32) {
    (base, ofs as i32)
}

fn addr32(base: i16, index: u8, scale: u8, ofs: i64) -> (i16, u8, u8, i32) {
    (base, index, scale, ofs as i32)
}

/// `_binaryop(name)`: the instructions `INSN(loc1, loc2)` covers.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOp {
    MOV,
    MOV8,
    MOV16,
    MOV32,
    MOVZX8,
    MOVSX8,
    MOVZX16,
    MOVSX16,
    MOVSX32,
    ADD,
    SUB,
    AND,
    OR,
    XOR,
    IMUL,
    CMP,
    TEST,
    TEST8,
    OR8,
    BTS,
    SHL,
    SHR,
    SAR,
    LEA,
    CMOVNS,
    MOVSD,
    MOVAPD,
    MOVDQ,
    MOVD32,
    ADDSD,
    SUBSD,
    MULSD,
    DIVSD,
    UCOMISD,
    XORPD,
    ANDPD,
    CVTTSD2SI,
    CVTSD2SS,
    CVTSS2SD,
    MOVUPS,
}

impl BinOp {
    /// `name.startswith('MOV')`.
    fn is_mov(self) -> bool {
        matches!(
            self,
            BinOp::MOV
                | BinOp::MOV8
                | BinOp::MOV16
                | BinOp::MOV32
                | BinOp::MOVZX8
                | BinOp::MOVSX8
                | BinOp::MOVZX16
                | BinOp::MOVSX16
                | BinOp::MOVSX32
                | BinOp::MOVSD
                | BinOp::MOVAPD
                | BinOp::MOVDQ
                | BinOp::MOVD32
                | BinOp::MOVUPS
        )
    }
}

/// `_unaryop(name)` / `_relative_unaryop(name)`.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnOp {
    INC,
    NEG,
    NOT,
    MUL,
    PUSH,
    POP,
    CALL,
    JMP,
}

/// x86/regloc.py `LocationCodeBuilder`, over the `Assembler` it writes to.
pub(crate) struct LocationCodeBuilder {
    mc: Assembler,
    /// `_scratch_register_value`: what `X86_64_SCRATCH_REG` is known to
    /// hold. `-1` means unknown.
    _scratch_register_value: i64,
}

impl Deref for LocationCodeBuilder {
    type Target = Assembler;

    fn deref(&self) -> &Assembler {
        &self.mc
    }
}

impl DerefMut for LocationCodeBuilder {
    fn deref_mut(&mut self) -> &mut Assembler {
        &mut self.mc
    }
}

impl Default for LocationCodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// The INSN entry points keep the RPython spelling (`mc.MOV`, `mc.CALL`).
#[allow(non_snake_case)]
impl LocationCodeBuilder {
    /// `MachineCodeBlockWrapper()`: an empty buffer with r11 unknown.
    pub(crate) fn new() -> Self {
        LocationCodeBuilder {
            mc: Assembler::new(0),
            _scratch_register_value: -1,
        }
    }

    /// The assembled bytes, handed to `codebuf` for `materialize`.
    pub(crate) fn into_inner(self) -> Assembler {
        self.mc
    }

    /// Hand the bytes assembled so far to `codebuf` and continue in a fresh
    /// buffer, the next helper of a family built by one assembler.
    pub(crate) fn take(&mut self) -> Assembler {
        std::mem::replace(self, LocationCodeBuilder::new()).mc
    }

    // ── _binaryop ──

    pub(crate) fn MOV(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOV, loc1.into(), loc2.into());
    }
    pub(crate) fn MOV8(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOV8, loc1.into(), loc2.into());
    }
    pub(crate) fn MOV16(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOV16, loc1.into(), loc2.into());
    }
    pub(crate) fn MOV32(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOV32, loc1.into(), loc2.into());
    }
    pub(crate) fn MOVZX8(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOVZX8, loc1.into(), loc2.into());
    }
    pub(crate) fn MOVSX8(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOVSX8, loc1.into(), loc2.into());
    }
    pub(crate) fn MOVZX16(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOVZX16, loc1.into(), loc2.into());
    }
    pub(crate) fn MOVSX16(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOVSX16, loc1.into(), loc2.into());
    }
    pub(crate) fn MOVSX32(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOVSX32, loc1.into(), loc2.into());
    }
    pub(crate) fn ADD(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::ADD, loc1.into(), loc2.into());
    }
    pub(crate) fn SUB(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::SUB, loc1.into(), loc2.into());
    }
    pub(crate) fn AND(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::AND, loc1.into(), loc2.into());
    }
    pub(crate) fn OR(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::OR, loc1.into(), loc2.into());
    }
    pub(crate) fn XOR(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::XOR, loc1.into(), loc2.into());
    }
    pub(crate) fn IMUL(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::IMUL, loc1.into(), loc2.into());
    }
    pub(crate) fn CMP(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::CMP, loc1.into(), loc2.into());
    }
    pub(crate) fn TEST(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::TEST, loc1.into(), loc2.into());
    }
    pub(crate) fn TEST8(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::TEST8, loc1.into(), loc2.into());
    }
    pub(crate) fn OR8(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::OR8, loc1.into(), loc2.into());
    }
    pub(crate) fn BTS(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::BTS, loc1.into(), loc2.into());
    }
    pub(crate) fn SHL(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::SHL, loc1.into(), loc2.into());
    }
    pub(crate) fn SHR(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::SHR, loc1.into(), loc2.into());
    }
    pub(crate) fn SAR(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::SAR, loc1.into(), loc2.into());
    }
    pub(crate) fn LEA(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::LEA, loc1.into(), loc2.into());
    }
    pub(crate) fn CMOVNS(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::CMOVNS, loc1.into(), loc2.into());
    }
    pub(crate) fn MOVSD(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOVSD, loc1.into(), loc2.into());
    }
    pub(crate) fn MOVAPD(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOVAPD, loc1.into(), loc2.into());
    }
    pub(crate) fn MOVDQ(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOVDQ, loc1.into(), loc2.into());
    }
    pub(crate) fn MOVD32(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOVD32, loc1.into(), loc2.into());
    }
    pub(crate) fn ADDSD(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::ADDSD, loc1.into(), loc2.into());
    }
    pub(crate) fn SUBSD(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::SUBSD, loc1.into(), loc2.into());
    }
    pub(crate) fn MULSD(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MULSD, loc1.into(), loc2.into());
    }
    pub(crate) fn DIVSD(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::DIVSD, loc1.into(), loc2.into());
    }
    pub(crate) fn UCOMISD(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::UCOMISD, loc1.into(), loc2.into());
    }
    pub(crate) fn XORPD(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::XORPD, loc1.into(), loc2.into());
    }
    pub(crate) fn ANDPD(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::ANDPD, loc1.into(), loc2.into());
    }
    pub(crate) fn CVTTSD2SI(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::CVTTSD2SI, loc1.into(), loc2.into());
    }
    pub(crate) fn CVTSD2SS(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::CVTSD2SS, loc1.into(), loc2.into());
    }
    pub(crate) fn CVTSS2SD(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::CVTSS2SD, loc1.into(), loc2.into());
    }
    pub(crate) fn MOVUPS(&mut self, loc1: impl Into<Operand>, loc2: impl Into<Operand>) {
        self.INSN(BinOp::MOVUPS, loc1.into(), loc2.into());
    }

    // ── _unaryop / _relative_unaryop ──

    pub(crate) fn INC(&mut self, loc: impl Into<Operand>) {
        self.INSN_unary(UnOp::INC, loc.into());
    }
    pub(crate) fn NEG(&mut self, loc: impl Into<Operand>) {
        self.INSN_unary(UnOp::NEG, loc.into());
    }
    pub(crate) fn NOT(&mut self, loc: impl Into<Operand>) {
        self.INSN_unary(UnOp::NOT, loc.into());
    }
    pub(crate) fn MUL(&mut self, loc: impl Into<Operand>) {
        self.INSN_unary(UnOp::MUL, loc.into());
    }
    pub(crate) fn PUSH(&mut self, loc: impl Into<Operand>) {
        self.INSN_unary(UnOp::PUSH, loc.into());
    }
    pub(crate) fn POP(&mut self, loc: impl Into<Operand>) {
        self.INSN_unary(UnOp::POP, loc.into());
    }
    pub(crate) fn CALL(&mut self, loc: impl Into<Operand>) {
        self.INSN_relative(UnOp::CALL, loc.into());
    }
    pub(crate) fn JMP(&mut self, loc: impl Into<Operand>) {
        self.INSN_relative(UnOp::JMP, loc.into());
    }

    /// `_binaryop`'s `insn_with_64_bit_immediate`. Returns true when the
    /// instruction is fully emitted; false after `_load_scratch(val2)`, when
    /// the caller retries with the scratch register as `loc2`.
    fn insn_with_64_bit_immediate(&mut self, name: BinOp, loc1: Operand, val2: i64) -> bool {
        if let (BinOp::MOV, Operand::R(reg)) = (name, loc1) {
            rx86::mov_ri(&mut self.mc, reg, val2);
            return true;
        }
        let checkvalue = match loc1 {
            Operand::J(addr) => addr,
            Operand::M(_, ofs) | Operand::A(_, _, _, ofs) => ofs,
            _ => 0,
        };
        let scratch = X86_64_SCRATCH_REG.value;
        // A `'j'` with no `NAME_j<code2>` form is reached through
        // `_addr_as_reg_offset`, which needs the scratch register too.
        let j_needs_scratch =
            matches!(loc1, Operand::J(_)) && !has_j_form(name, loc1, Operand::R(scratch));
        if !rx86::fits_in_32bits(checkvalue) || j_needs_scratch {
            // Both operands are 64-bit: the scratch register is taken by the
            // address, so the immediate goes through a pushed free register.
            let freereg = loc1.find_unused_reg();
            rx86::push_r(&mut self.mc, freereg);
            rx86::mov_ri(&mut self.mc, freereg, val2);
            self.INSN(name, loc1, Operand::R(freereg));
            dynasm!(self.mc ; .arch x64 ; pop Rq(freereg));
            return true;
        }
        self._load_scratch(val2);
        false
    }

    /// `_binaryop(name)` `INSN(loc1, loc2)`.
    fn INSN(&mut self, name: BinOp, loc1: Operand, loc2: Operand) {
        self.INSN_dispatch(name, loc1, loc2);
        if name.is_mov() && loc1.is_scratch() {
            // `_addr_as_reg_offset` may have recorded the address this MOV
            // read through r11, which the MOV then overwrote.
            self.forget_scratch_register();
        }
    }

    fn INSN_dispatch(&mut self, name: BinOp, loc1: Operand, mut loc2: Operand) {
        // You cannot pass in the scratch register as a location, except
        // with a MOV instruction.
        if name.is_mov() {
            if loc2.is_scratch() {
                assert!(
                    !matches!(loc1, Operand::J(_) | Operand::M(..) | Operand::A(..)),
                    "{name:?} from the scratch register into {loc1:?}"
                );
            }
            if loc1.is_scratch() {
                self.forget_scratch_register();
            }
        } else if loc1.is_scratch() || loc2.is_scratch() {
            panic!("{name:?} with scratch reg specified");
        }
        if let Operand::I(val2) = loc2
            && !rx86::fits_in_32bits(val2)
        {
            if self.insn_with_64_bit_immediate(name, loc1, val2) {
                return;
            }
            loc2 = Operand::R(X86_64_SCRATCH_REG.value);
        }
        // More faking out of certain operations for x86_64.
        if let Operand::J(addr) = loc1
            && (!rx86::fits_in_32bits(addr) || !has_j_form(name, loc1, loc2))
        {
            let (reg, ofs) = self._addr_as_reg_offset(addr);
            return self.invoke(name, Operand::M(reg, i64::from(ofs)), loc2);
        }
        if let Operand::J(addr) = loc2
            && (!rx86::fits_in_32bits(addr) || !has_j_form(name, loc1, loc2))
        {
            let (reg, ofs) = self._addr_as_reg_offset(addr);
            return self.invoke(name, loc1, Operand::M(reg, i64::from(ofs)));
        }
        let loc1 = self.fix_static_offset(loc1);
        let loc2 = self.fix_static_offset(loc2);
        self.invoke(name, loc1, loc2);
    }

    /// `_fix_static_offset_64_m` / `_fix_static_offset_64_a` for an address
    /// whose static offset does not fit 32 bits.
    fn fix_static_offset(&mut self, loc: Operand) -> Operand {
        match loc {
            Operand::M(base, ofs) if !rx86::fits_in_32bits(ofs) => {
                let (reg, ofs) = self._fix_static_offset_64_m((base, ofs));
                Operand::M(reg, i64::from(ofs))
            }
            Operand::A(base, index, scale, ofs) if !rx86::fits_in_32bits(ofs) => {
                let (base, index, scale, ofs) =
                    self._fix_static_offset_64_a((base, index, scale, ofs));
                Operand::A(base, index, scale, i64::from(ofs))
            }
            other => other,
        }
    }

    /// `invoke(self, codes, val1, val2)`: `NAME_<code1><code2>`. Every `'m'`
    /// and `'a'` offset and every `'j'` address fits 32 bits by now.
    fn invoke(&mut self, name: BinOp, loc1: Operand, loc2: Operand) {
        use BinOp::*;
        use Operand::*;
        let mc = &mut self.mc;
        match (name, loc1, loc2) {
            (MOV, R(a), R(b)) => {
                if a != b {
                    dynasm!(mc ; .arch x64 ; mov Rq(a), Rq(b));
                }
            }
            (MOV, R(a), I(v)) => rx86::mov_ri(mc, a, v),
            (MOV, R(a), B(o)) => rx86::mov_rb(mc, a, o),
            (MOV, R(a), S(o)) => rx86::mov_rs(mc, a, o),
            (MOV, R(a), M(b, o)) => rx86::mov_rm(mc, a, mem32(b, o)),
            (MOV, R(a), A(b, i, s, o)) => rx86::mov_ra(mc, a, addr32(b, i, s, o)),
            (MOV, B(o), R(b)) => rx86::mov_br(mc, o, b),
            (MOV, B(o), I(v)) => rx86::mov_bi(mc, o, v as i32),
            (MOV, S(o), R(b)) => rx86::mov_sr(mc, o, b),
            (MOV, S(o), I(v)) => rx86::mov_si(mc, o, v as i32),
            (MOV, M(b, o), R(r)) => rx86::mov_mr(mc, mem32(b, o), r),
            (MOV, M(b, o), I(v)) => rx86::mov_mi(mc, mem32(b, o), v as i32),
            (MOV, A(b, i, s, o), R(r)) => rx86::mov_ar(mc, addr32(b, i, s, o), r),
            (MOV, A(b, i, s, o), I(v)) => rx86::mov_ai(mc, addr32(b, i, s, o), v as i32),
            (MOV8, M(b, o), R(r)) => rx86::mov8_mr(mc, mem32(b, o), r),
            (MOV8, M(b, o), I(v)) => rx86::mov8_mi(mc, mem32(b, o), v as i32),
            (MOV8, A(b, i, s, o), R(r)) => rx86::mov8_ar(mc, addr32(b, i, s, o), r),
            (MOV8, A(b, i, s, o), I(v)) => rx86::mov8_ai(mc, addr32(b, i, s, o), v as i32),
            (MOV16, M(b, o), R(r)) => rx86::mov16_mr(mc, mem32(b, o), r),
            (MOV16, M(b, o), I(v)) => rx86::mov16_mi(mc, mem32(b, o), v as i32),
            (MOV16, A(b, i, s, o), R(r)) => rx86::mov16_ar(mc, addr32(b, i, s, o), r),
            (MOV16, A(b, i, s, o), I(v)) => rx86::mov16_ai(mc, addr32(b, i, s, o), v as i32),
            (MOV32, R(a), R(b)) => rx86::mov32_rr(mc, a, b),
            (MOV32, R(a), M(b, o)) => rx86::mov32_rm(mc, a, mem32(b, o)),
            (MOV32, R(a), A(b, i, s, o)) => rx86::mov32_ra(mc, a, addr32(b, i, s, o)),
            (MOV32, R(a), B(o)) => rx86::mov32_rm(mc, a, (rx86::EBP, o)),
            (MOV32, S(o), R(r)) => rx86::mov32_mr(mc, (rx86::ESP, o), r),
            (MOV32, S(o), I(v)) => rx86::mov32_mi(mc, (rx86::ESP, o), v as i32),
            (MOV32, B(o), I(v)) => rx86::mov32_bi(mc, o, v as i32),
            (MOV32, M(b, o), R(r)) => rx86::mov32_mr(mc, mem32(b, o), r),
            (MOV32, M(b, o), I(v)) => rx86::mov32_mi(mc, mem32(b, o), v as i32),
            (MOV32, A(b, i, s, o), R(r)) => rx86::mov32_ar(mc, addr32(b, i, s, o), r),
            (MOV32, A(b, i, s, o), I(v)) => rx86::mov32_ai(mc, addr32(b, i, s, o), v as i32),
            (MOVZX8, R(a), R(b)) => dynasm!(mc ; .arch x64 ; movzx Rq(a), Rb(b)),
            (MOVZX8, R(a), M(b, o)) => rx86::movzx8_rm(mc, a, mem32(b, o)),
            (MOVZX8, R(a), A(b, i, s, o)) => rx86::movzx8_ra(mc, a, addr32(b, i, s, o)),
            (MOVSX8, R(a), R(b)) => dynasm!(mc ; .arch x64 ; movsx Rq(a), Rb(b)),
            (MOVSX8, R(a), B(o)) => rx86::movsx8_rm(mc, a, (rx86::EBP, o)),
            (MOVSX8, R(a), M(b, o)) => rx86::movsx8_rm(mc, a, mem32(b, o)),
            (MOVSX8, R(a), A(b, i, s, o)) => rx86::movsx8_ra(mc, a, addr32(b, i, s, o)),
            (MOVZX16, R(a), R(b)) => dynasm!(mc ; .arch x64 ; movzx Rq(a), Rw(b)),
            (MOVZX16, R(a), M(b, o)) => rx86::movzx16_rm(mc, a, mem32(b, o)),
            (MOVZX16, R(a), A(b, i, s, o)) => rx86::movzx16_ra(mc, a, addr32(b, i, s, o)),
            (MOVSX16, R(a), R(b)) => dynasm!(mc ; .arch x64 ; movsx Rq(a), Rw(b)),
            (MOVSX16, R(a), B(o)) => rx86::movsx16_rm(mc, a, (rx86::EBP, o)),
            (MOVSX16, R(a), M(b, o)) => rx86::movsx16_rm(mc, a, mem32(b, o)),
            (MOVSX16, R(a), A(b, i, s, o)) => rx86::movsx16_ra(mc, a, addr32(b, i, s, o)),
            (MOVSX32, R(a), R(b)) => dynasm!(mc ; .arch x64 ; movsxd Rq(a), Rd(b)),
            (MOVSX32, R(a), B(o)) => rx86::movsx32_rm(mc, a, (rx86::EBP, o)),
            (MOVSX32, R(a), M(b, o)) => rx86::movsx32_rm(mc, a, mem32(b, o)),
            (MOVSX32, R(a), A(b, i, s, o)) => rx86::movsx32_ra(mc, a, addr32(b, i, s, o)),
            (ADD, R(a), R(b)) => dynasm!(mc ; .arch x64 ; add Rq(a), Rq(b)),
            (ADD, R(a), I(v)) => rx86::add_ri(mc, a, v as i32),
            (ADD, R(a), B(o)) => rx86::add_rb(mc, a, o),
            (SUB, R(a), R(b)) => dynasm!(mc ; .arch x64 ; sub Rq(a), Rq(b)),
            (SUB, R(a), I(v)) => rx86::sub_ri(mc, a, v as i32),
            (SUB, R(a), B(o)) => rx86::sub_rb(mc, a, o),
            (SUB, R(a), M(b, o)) => rx86::sub_rm(mc, a, mem32(b, o)),
            (SUB, M(b, o), I(v)) => rx86::sub_mi(mc, mem32(b, o), v as i32),
            (AND, R(a), R(b)) => dynasm!(mc ; .arch x64 ; and Rq(a), Rq(b)),
            (AND, R(a), I(v)) => rx86::and_ri(mc, a, v as i32),
            (AND, R(a), B(o)) => rx86::and_rb(mc, a, o),
            (OR, R(a), R(b)) => dynasm!(mc ; .arch x64 ; or Rq(a), Rq(b)),
            (OR, R(a), I(v)) => rx86::or_ri(mc, a, v as i32),
            (OR, R(a), B(o)) => rx86::or_rb(mc, a, o),
            (XOR, R(a), R(b)) => dynasm!(mc ; .arch x64 ; xor Rq(a), Rq(b)),
            (XOR, R(a), I(v)) => rx86::xor_ri(mc, a, v as i32),
            (XOR, R(a), B(o)) => rx86::xor_rb(mc, a, o),
            (IMUL, R(a), R(b)) => dynasm!(mc ; .arch x64 ; imul Rq(a), Rq(b)),
            (IMUL, R(a), I(v)) => rx86::imul_ri(mc, a, v as i32),
            (IMUL, R(a), B(o)) => rx86::imul_rb(mc, a, o),
            (CMP, R(a), R(b)) => dynasm!(mc ; .arch x64 ; cmp Rq(a), Rq(b)),
            (CMP, R(a), I(v)) => rx86::cmp_ri(mc, a, v as i32),
            (CMP, R(a), B(o)) => rx86::cmp_rb(mc, a, o),
            (CMP, R(a), M(b, o)) => rx86::cmp_rm(mc, a, mem32(b, o)),
            (CMP, B(o), R(r)) => rx86::cmp_br(mc, o, r),
            (CMP, B(o), I(v)) => rx86::cmp_bi(mc, o, v as i32),
            (CMP, M(b, o), R(r)) => rx86::cmp_mr(mc, mem32(b, o), r),
            (CMP, M(b, o), I(v)) => rx86::cmp_mi(mc, mem32(b, o), v as i32),
            (TEST, R(a), R(b)) => dynasm!(mc ; .arch x64 ; test Rq(a), Rq(b)),
            (TEST, R(a), I(v)) => rx86::test_ri(mc, a, v as i32),
            (TEST, M(b, o), I(v)) => rx86::test_mi(mc, mem32(b, o), v as i32),
            (TEST, A(b, i, s, o), I(v)) => rx86::test_ai(mc, addr32(b, i, s, o), v as i32),
            (TEST, J(addr), I(v)) => rx86::test_ji(mc, addr as i32, v as i32),
            (TEST, B(o), I(v)) => rx86::test_bi(mc, o, v as i32),
            (TEST, B(o), R(r)) => rx86::test_br(mc, o, r),
            (TEST8, M(b, o), I(v)) => rx86::test8_mi(mc, mem32(b, o), v as i32),
            (TEST8, A(b, i, s, o), I(v)) => rx86::test8_ai(mc, addr32(b, i, s, o), v as i32),
            (TEST8, J(addr), I(v)) => rx86::test8_ji(mc, addr as i32, v as i32),
            (OR8, M(b, o), I(v)) => rx86::or8_mi(mc, mem32(b, o), v as i32),
            (OR8, J(addr), I(v)) => rx86::or8_ji(mc, addr as i32, v as i32),
            (BTS, M(b, o), R(r)) => rx86::bts_mr(mc, mem32(b, o), r),
            (BTS, J(addr), R(r)) => rx86::bts_jr(mc, addr as i32, r),
            (SHL, R(a), I(v)) => rx86::shl_ri(mc, a, v as i32),
            (SHR, R(a), I(v)) => rx86::shr_ri(mc, a, v as i32),
            (SAR, R(a), I(v)) => rx86::sar_ri(mc, a, v as i32),
            (SHL, R(a), R(rx86::ECX)) => dynasm!(mc ; .arch x64 ; shl Rq(a), cl),
            (SHR, R(a), R(rx86::ECX)) => dynasm!(mc ; .arch x64 ; shr Rq(a), cl),
            (SAR, R(a), R(rx86::ECX)) => dynasm!(mc ; .arch x64 ; sar Rq(a), cl),
            (LEA, R(a), B(o)) => rx86::lea_rm(mc, a, (rx86::EBP, o)),
            (LEA, R(a), S(o)) => rx86::lea_rs(mc, a, o),
            (LEA, R(a), M(b, o)) => rx86::lea_rm(mc, a, mem32(b, o)),
            (LEA, R(a), A(b, i, s, o)) => rx86::lea_ra(mc, a, addr32(b, i, s, o)),
            (CMOVNS, R(a), R(b)) => rx86::cmovns_rr(mc, a, b),
            (MOVSD, X(a), B(o)) => rx86::movsd_xb(mc, a, o),
            (MOVSD, X(a), S(o)) => rx86::movsd_xs(mc, a, o),
            (MOVSD, X(a), M(b, o)) => rx86::movsd_xm(mc, a, mem32(b, o)),
            (MOVSD, X(a), A(b, i, s, o)) => rx86::movsd_xa(mc, a, addr32(b, i, s, o)),
            (MOVSD, X(a), J(addr)) => rx86::movsd_xj(mc, a, addr as i32),
            (MOVSD, B(o), X(r)) => rx86::movsd_bx(mc, o, r),
            (MOVSD, S(o), X(r)) => rx86::movsd_sx(mc, o, r),
            (MOVSD, M(b, o), X(r)) => rx86::movsd_mx(mc, mem32(b, o), r),
            (MOVSD, A(b, i, s, o), X(r)) => rx86::movsd_ax(mc, addr32(b, i, s, o), r),
            (MOVAPD, X(a), X(b)) => rx86::movapd_xx(mc, a, b),
            (ADDSD, X(a), X(b)) => rx86::addsd_xx(mc, a, b),
            (ADDSD, X(a), B(o)) => rx86::addsd_xb(mc, a, o),
            (ADDSD, X(a), M(b, o)) => rx86::addsd_xm(mc, a, mem32(b, o)),
            (ADDSD, X(a), J(addr)) => rx86::addsd_xj(mc, a, addr as i32),
            (SUBSD, X(a), X(b)) => rx86::subsd_xx(mc, a, b),
            (SUBSD, X(a), B(o)) => rx86::subsd_xb(mc, a, o),
            (SUBSD, X(a), M(b, o)) => rx86::subsd_xm(mc, a, mem32(b, o)),
            (SUBSD, X(a), J(addr)) => rx86::subsd_xj(mc, a, addr as i32),
            (MULSD, X(a), X(b)) => rx86::mulsd_xx(mc, a, b),
            (MULSD, X(a), B(o)) => rx86::mulsd_xb(mc, a, o),
            (MULSD, X(a), M(b, o)) => rx86::mulsd_xm(mc, a, mem32(b, o)),
            (MULSD, X(a), J(addr)) => rx86::mulsd_xj(mc, a, addr as i32),
            (DIVSD, X(a), X(b)) => rx86::divsd_xx(mc, a, b),
            (DIVSD, X(a), B(o)) => rx86::divsd_xb(mc, a, o),
            (DIVSD, X(a), M(b, o)) => rx86::divsd_xm(mc, a, mem32(b, o)),
            (DIVSD, X(a), J(addr)) => rx86::divsd_xj(mc, a, addr as i32),
            (UCOMISD, X(a), X(b)) => rx86::ucomisd_xx(mc, a, b),
            (UCOMISD, X(a), B(o)) => rx86::ucomisd_xb(mc, a, o),
            (UCOMISD, X(a), M(b, o)) => rx86::ucomisd_xm(mc, a, mem32(b, o)),
            (UCOMISD, X(a), J(addr)) => rx86::ucomisd_xj(mc, a, addr as i32),
            (XORPD, X(a), X(b)) => rx86::xorpd_xx(mc, a, b),
            (XORPD, X(a), M(b, o)) => rx86::xorpd_xm(mc, a, mem32(b, o)),
            (XORPD, X(a), J(addr)) => rx86::xorpd_xj(mc, a, addr as i32),
            (ANDPD, X(a), X(b)) => rx86::andpd_xx(mc, a, b),
            (ANDPD, X(a), M(b, o)) => rx86::andpd_xm(mc, a, mem32(b, o)),
            (ANDPD, X(a), J(addr)) => rx86::andpd_xj(mc, a, addr as i32),
            (MOVDQ, R(a), X(b)) => rx86::movdq_rx(mc, a, b),
            (MOVDQ, X(a), R(b)) => rx86::movdq_xr(mc, a, b),
            (MOVD32, R(a), X(b)) => rx86::movd32_rx(mc, a, b),
            (MOVD32, X(a), R(b)) => rx86::movd32_xr(mc, a, b),
            (CVTTSD2SI, R(a), X(b)) => rx86::cvttsd2si_rx(mc, a, b),
            (CVTTSD2SI, R(a), B(o)) => rx86::cvttsd2si_rb(mc, a, o),
            (CVTSD2SS, X(a), X(b)) => rx86::cvtsd2ss_xx(mc, a, b),
            (CVTSS2SD, X(a), X(b)) => rx86::cvtss2sd_xx(mc, a, b),
            (MOVUPS, M(b, o), X(r)) => rx86::movups_mx(mc, mem32(b, o), r),
            (MOVUPS, A(b, i, s, o), X(r)) => rx86::movups_ax(mc, addr32(b, i, s, o), r),
            (MOVD32, X(a), B(o)) => dynasm!(mc ; .arch x64 ; movd Rx(a), DWORD [rbp + o]),
            _ => panic!("_missing_binary_insn: {name:?} has no encoding for ({loc1:?}, {loc2:?})"),
        }
    }

    /// `_unaryop(name)` `INSN(loc)`.
    fn INSN_unary(&mut self, name: UnOp, loc: Operand) {
        assert!(!loc.is_scratch(), "{name:?} with scratch reg specified");
        let loc = match loc {
            Operand::I(val) if !rx86::fits_in_32bits(val) => {
                // for 'PUSH(imm)'
                self._load_scratch(val);
                Operand::R(X86_64_SCRATCH_REG.value)
            }
            Operand::J(addr) if !rx86::fits_in_32bits(addr) => {
                let (reg, ofs) = self._addr_as_reg_offset(addr);
                Operand::M(reg, i64::from(ofs))
            }
            other => self.fix_static_offset(other),
        };
        self.invoke_unary(name, loc);
    }

    /// `_relative_unaryop(name)` `INSN(loc)`: `CALL` / `JMP`. An immediate
    /// target is `_load_scratch` then `NAME_r(X86_64_SCRATCH_REG)`. Either
    /// way r11 is forgotten afterwards.
    fn INSN_relative(&mut self, name: UnOp, loc: Operand) {
        let loc = match loc {
            Operand::I(val) => {
                self._load_scratch(val);
                Operand::R(X86_64_SCRATCH_REG.value)
            }
            other => other,
        };
        self.invoke_unary(name, loc);
        // This is for CALL and JMP, so it's correct to forget the value of
        // the R11 register here.
        self.forget_scratch_register();
    }

    fn invoke_unary(&mut self, name: UnOp, loc: Operand) {
        use Operand::*;
        use UnOp::*;
        let mc = &mut self.mc;
        match (name, loc) {
            (INC, M(b, o)) => rx86::inc_m(mc, mem32(b, o)),
            (INC, J(addr)) => rx86::inc_j(mc, addr as i32),
            (NEG, R(r)) => dynasm!(mc ; .arch x64 ; neg Rq(r)),
            (NOT, R(r)) => dynasm!(mc ; .arch x64 ; not Rq(r)),
            (NOT, B(o)) => rx86::not_b(mc, o),
            (MUL, R(r)) => dynasm!(mc ; .arch x64 ; mul Rq(r)),
            (MUL, B(o)) => rx86::mul_b(mc, o),
            (PUSH, R(r)) => rx86::push_r(mc, r),
            (PUSH, B(o)) => rx86::push_b(mc, o),
            (PUSH, M(b, o)) => rx86::push_m(mc, mem32(b, o)),
            (PUSH, J(addr)) => rx86::push_j(mc, addr as i32),
            (PUSH, I(v)) if rx86::single_byte(v) => rx86::push_i8(mc, v as i8),
            (PUSH, I(v)) => rx86::push_i32(mc, v as i32),
            (POP, R(r)) => dynasm!(mc ; .arch x64 ; pop Rq(r)),
            (POP, B(o)) => rx86::pop_b(mc, o),
            (CALL, R(r)) => dynasm!(mc ; .arch x64 ; call Rq(r)),
            (CALL, B(o)) => rx86::call_b(mc, o),
            (CALL, S(o)) => rx86::call_s(mc, o),
            (JMP, R(r)) => dynasm!(mc ; .arch x64 ; jmp Rq(r)),
            _ => panic!("{name:?} has no encoding for {loc:?}"),
        }
    }

    /// `_addr_as_reg_offset`: a 64-bit address as an offset from the
    /// scratch register. Reuse the value r11 is known to hold when the
    /// difference fits a signed disp32; otherwise `MOV_ri` it and remember.
    pub(crate) fn _addr_as_reg_offset(&mut self, addr: i64) -> (u8, i32) {
        let scratch = X86_64_SCRATCH_REG.value;
        if self._scratch_register_value != -1 {
            let offset = addr.wrapping_sub(self._scratch_register_value);
            if rx86::fits_in_32bits(offset) {
                return (scratch, offset as i32);
            }
        }
        self._scratch_register_value = addr;
        rx86::mov_ri(&mut self.mc, scratch, addr);
        (scratch, 0)
    }

    /// `_fix_static_offset_64_m`: `LEA` the base plus the materialized
    /// offset into `X86_64_SCRATCH_REG`, which is then forgotten, and
    /// address `[r11]`.
    pub(crate) fn _fix_static_offset_64_m(&mut self, args: (u8, i64)) -> (u8, i32) {
        let (basereg, static_offset) = args;
        let scratch = X86_64_SCRATCH_REG.value;
        let (reg, ofs) = self._addr_as_reg_offset(static_offset);
        self.forget_scratch_register();
        rx86::lea_ra(&mut self.mc, scratch, (i16::from(basereg), reg, 0, ofs));
        (scratch, 0)
    }

    /// `_fix_static_offset_64_a`: the materialized offset becomes the base;
    /// with a real base it is first `LEA`d with it into
    /// `X86_64_SCRATCH_REG`, keeping the scaled index.
    pub(crate) fn _fix_static_offset_64_a(
        &mut self,
        args: (i16, u8, u8, i64),
    ) -> (i16, u8, u8, i32) {
        let (basereg, scalereg, scale, static_offset) = args;
        let scratch = X86_64_SCRATCH_REG.value;
        let (mut reg, mut ofs) = self._addr_as_reg_offset(static_offset);
        if basereg != rx86::NO_BASE_REGISTER {
            self.forget_scratch_register();
            rx86::lea_ra(&mut self.mc, scratch, (basereg, reg, 0, ofs));
            reg = scratch;
            ofs = 0;
        }
        (i16::from(reg), scalereg, scale, ofs)
    }

    /// `_load_scratch`: put `value` in `X86_64_SCRATCH_REG`. Nothing is
    /// emitted when r11 already holds it, `LEA r11, [r11 + d]` covers a
    /// difference that fits a signed disp32, and `MOV_ri` the rest.
    pub(crate) fn _load_scratch(&mut self, value: i64) {
        let scratch = X86_64_SCRATCH_REG.value;
        if self._scratch_register_value != -1 {
            if self._scratch_register_value == value {
                return;
            }
            let offset = value.wrapping_sub(self._scratch_register_value);
            if rx86::fits_in_32bits(offset) {
                rx86::lea_rm(&mut self.mc, scratch, (scratch, offset as i32));
                self._scratch_register_value = value;
                return;
            }
        }
        self._scratch_register_value = value;
        rx86::mov_ri(&mut self.mc, scratch, value);
    }

    /// `forget_scratch_register`. Every point where the fallthrough's r11
    /// value stops describing the code that follows: a `CALL`/`JMP`, a
    /// join label, a write of r11 by an instruction this builder did not
    /// dispatch.
    pub(crate) fn forget_scratch_register(&mut self) {
        self._scratch_register_value = -1;
    }

    pub(crate) fn get_scratch_register_known_value(&self) -> i64 {
        self._scratch_register_value
    }

    pub(crate) fn restore_scratch_register_known_value(&mut self, saved_value: i64) {
        self._scratch_register_value = saved_value;
    }

    pub(crate) fn load_scratch_if_known(&mut self, saved_value: i64) {
        if saved_value != -1 {
            self._load_scratch(saved_value);
        }
    }
}
