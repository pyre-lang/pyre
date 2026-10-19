/// regloc.py: Location abstractions for registers, frame slots, immediates.
///
/// This module adds support for "locations", which can be either
/// a RegLoc, FrameLoc, ImmedLoc, or AddressLoc.
use crate::arch::WORD;

/// regloc.py AssemblerLocation — location code identifies the kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocationCode {
    /// 'r' — general-purpose register
    Reg,
    /// 'x' — XMM register
    Xmm,
    /// 'b' — RBP-relative (frame slot)
    Ebp,
    /// 's' — RSP-relative (stack)
    Esp,
    /// 'i' — immediate
    Immed,
    /// 'j' — absolute address immediate
    AddrImmed,
}

/// regloc.py RegLoc — a register location (GPR or XMM).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegLoc {
    pub value: u8,
    pub is_xmm: bool,
}

impl RegLoc {
    pub const fn new(regnum: u8, is_xmm: bool) -> Self {
        RegLoc {
            value: regnum,
            is_xmm,
        }
    }

    pub fn location_code(&self) -> LocationCode {
        if self.is_xmm {
            LocationCode::Xmm
        } else {
            LocationCode::Reg
        }
    }

    pub fn is_core_reg(&self) -> bool {
        !self.is_xmm
    }
    pub fn is_float(&self) -> bool {
        self.is_xmm
    }

    pub fn get_width(&self) -> usize {
        if self.is_xmm { 8 } else { WORD }
    }
}

/// regloc.py RawEbpLoc — RBP-relative memory location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawEbpLoc {
    pub value: i32,
    pub is_float: bool,
}

impl RawEbpLoc {
    pub fn new(offset: i32) -> Self {
        RawEbpLoc {
            value: offset,
            is_float: false,
        }
    }

    pub fn location_code(&self) -> LocationCode {
        LocationCode::Ebp
    }

    pub fn get_width(&self) -> usize {
        if self.is_float { 8 } else { WORD }
    }

    pub fn add_offset(&self, ofs: i32) -> Self {
        RawEbpLoc {
            value: self.value + ofs,
            is_float: self.is_float,
        }
    }
}

/// regloc.py RawEspLoc — RSP-relative memory location: an outgoing call
/// argument word, or the thread-local slot `THREADLOCAL_OFS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawEspLoc {
    pub value: i32,
    pub is_float: bool,
}

impl RawEspLoc {
    pub fn new(value: i32, is_float: bool) -> Self {
        // accessing values < 0 is forbidden on x86-32.
        assert!(value >= 0, "RawEspLoc({value}) below the stack pointer");
        RawEspLoc { value, is_float }
    }

    pub fn location_code(&self) -> LocationCode {
        LocationCode::Esp
    }

    pub fn get_width(&self) -> usize {
        if self.is_float { 8 } else { WORD }
    }
}

/// regloc.py FrameLoc — frame slot location (position-aware RawEbpLoc).
/// `position` is a jitframe slot index; `u32` keeps [`Loc`] in 16 B so
/// `vec![loc]` does not mint the 24-byte class on every consider_*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameLoc {
    pub ebp_loc: RawEbpLoc,
    pub position: u32,
}

impl FrameLoc {
    pub fn new(position: usize, ebp_offset: i32, is_float: bool) -> Self {
        FrameLoc {
            ebp_loc: RawEbpLoc {
                value: ebp_offset,
                is_float,
            },
            position: u32::try_from(position).expect("frame slot fits u32"),
        }
    }

    pub fn is_stack(&self) -> bool {
        true
    }
    pub fn get_position(&self) -> usize {
        self.position as usize
    }
}

/// regloc.py ImmedLoc — immediate integer (or float-bit) value.
/// Float-ness lives on [`Loc`] (`Immed` / `ImmedFloat`) so the payload
/// stays 8 B and [`Loc`] stays 16 B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImmedLoc {
    pub value: i64,
}

/// `regloc.py` `ConstFloatLoc`: absolute address of an 8-byte literal in
/// the machine data block. Location code `'j'`, an
/// `ImmediateAssemblerLocation`, width 8. The address is what
/// `X86XMMRegisterManager.convert_to_imm` returns; the bits live at
/// `value`, not in this word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstFloatLoc {
    pub value: usize,
}

impl ImmedLoc {
    pub fn new(value: i64) -> Self {
        ImmedLoc { value }
    }

    pub fn location_code(&self) -> LocationCode {
        LocationCode::Immed
    }
}

/// regloc.py AddressLoc — memory address with base + index*scale + offset.
#[derive(Debug, Clone, Copy)]
pub struct AddressLoc {
    pub base: u8,    // base register number
    pub index: u8,   // index register number (255 = none)
    pub scale: u8,   // scale (0,1,2,3 for 1,2,4,8)
    pub offset: i32, // displacement
}

/// regloc.py: unified location enum for function arguments.
#[derive(Debug, Clone, Copy)]
pub enum Loc {
    Reg(RegLoc),
    Ebp(RawEbpLoc),
    Esp(RawEspLoc),
    Frame(FrameLoc),
    Immed(ImmedLoc),
    ImmedFloat(ImmedLoc),
    /// `ConstFloatLoc`. Address-bearing; not the float's bit pattern.
    ConstFloat(ConstFloatLoc),
    Addr(AddressLoc),
}

impl Loc {
    #[inline]
    pub fn immed(value: i64) -> Self {
        Loc::Immed(ImmedLoc { value })
    }

    #[inline]
    pub fn immed_float(value: i64) -> Self {
        Loc::ImmedFloat(ImmedLoc { value })
    }

    #[inline]
    pub fn as_immed(&self) -> Option<(i64, bool)> {
        match self {
            Loc::Immed(i) => Some((i.value, false)),
            Loc::ImmedFloat(i) => Some((i.value, true)),
            _ => None,
        }
    }
}

/// Matches either spelling of a frame-pointer location, binding its
/// `RawEbpLoc`.
///
/// `regloc.py class FrameLoc(RawEbpLoc)` — the two are one type upstream,
/// separated here only because Rust has no inheritance. Every operand that
/// addresses through the frame pointer accepts both (`mov`'s `'b'` code), so a
/// match naming just one spelling silently excludes the other.
macro_rules! ebp_loc_pat {
    ($e:ident) => {
        $crate::regloc::Loc::Frame($crate::regloc::FrameLoc { ebp_loc: $e, .. })
            | $crate::regloc::Loc::Ebp($e)
    };
}

pub(crate) use ebp_loc_pat;

impl Loc {
    pub fn is_reg(&self) -> bool {
        matches!(self, Loc::Reg(_))
    }
    /// `regloc.py RawEbpLoc.is_stack` returns True, and `FrameLoc` inherits
    /// it — a frame slot is a raw ebp location that also knows its stack
    /// position, so both spellings are stack locations.
    ///
    /// Naming only `Frame` here makes an `Ebp` operand look like a register to
    /// the parallel move, which then skips the scratch register and hands the
    /// backend a memory-to-memory move no machine encodes.
    pub fn is_stack(&self) -> bool {
        matches!(self, Loc::Frame(_) | Loc::Ebp(_))
    }
    /// `regloc.py AssemblerLocation.is_memory_reference`: location codes
    /// `'b'`, `'s'`, `'j'`, `'a'`, `'m'`. A `RawEspLoc` is one, although its
    /// `is_stack` is False.
    pub fn is_memory_reference(&self) -> bool {
        matches!(
            self,
            Loc::Frame(_) | Loc::Ebp(_) | Loc::Esp(_) | Loc::ConstFloat(_) | Loc::Addr(_)
        )
    }
    /// `ImmediateAssemblerLocation`: `ImmedLoc` and `ConstFloatLoc`.
    pub fn is_immed(&self) -> bool {
        matches!(
            self,
            Loc::Immed(_) | Loc::ImmedFloat(_) | Loc::ConstFloat(_)
        )
    }

    pub fn as_reg(&self) -> Option<RegLoc> {
        match self {
            Loc::Reg(r) => Some(*r),
            _ => None,
        }
    }
}

// regloc.py:336-339 — pre-defined register singletons.

pub const fn gpr(n: u8) -> RegLoc {
    RegLoc::new(n, false)
}
pub const fn xmm(n: u8) -> RegLoc {
    RegLoc::new(n, true)
}

/// regloc.py:338
pub const EAX: RegLoc = gpr(0);
pub const ECX: RegLoc = gpr(1);
pub const EDX: RegLoc = gpr(2);
pub const EBX: RegLoc = gpr(3);
pub const ESP: RegLoc = gpr(4);
pub const EBP: RegLoc = gpr(5);
pub const ESI: RegLoc = gpr(6);
pub const EDI: RegLoc = gpr(7);
pub const R8: RegLoc = gpr(8);
pub const R9: RegLoc = gpr(9);
pub const R10: RegLoc = gpr(10);
pub const R11: RegLoc = gpr(11);
pub const R12: RegLoc = gpr(12);
pub const R13: RegLoc = gpr(13);
pub const R14: RegLoc = gpr(14);
pub const R15: RegLoc = gpr(15);

pub const XMM0: RegLoc = xmm(0);
pub const XMM1: RegLoc = xmm(1);
pub const XMM2: RegLoc = xmm(2);
pub const XMM3: RegLoc = xmm(3);
pub const XMM4: RegLoc = xmm(4);
pub const XMM5: RegLoc = xmm(5);
pub const XMM6: RegLoc = xmm(6);
pub const XMM7: RegLoc = xmm(7);
pub const XMM8: RegLoc = xmm(8);
pub const XMM9: RegLoc = xmm(9);
pub const XMM10: RegLoc = xmm(10);
pub const XMM11: RegLoc = xmm(11);
pub const XMM12: RegLoc = xmm(12);
pub const XMM13: RegLoc = xmm(13);
pub const XMM14: RegLoc = xmm(14);
pub const XMM15: RegLoc = xmm(15);

/// regloc.py X86_64_SCRATCH_REG = r11
pub const X86_64_SCRATCH_REG: RegLoc = R11;

/// regloc.py:348 X86_64_SCRATCH_REG_2 = r12 (used as secondary scratch)
pub const X86_64_SCRATCH_REG_2: RegLoc = R12;

/// regloc.py:355-359 X86_64_XMM_SCRATCH_REG — xmm5 on Win64 (the XMM
/// allocator pool is xmm0..xmm4, regalloc.py:130), xmm15 elsewhere
/// (pool is xmm0..xmm14, regalloc.py:125). Always outside the
/// allocation pool, so emit-time materialization through it can never
/// clobber an allocated operand.
#[cfg(target_os = "windows")]
pub const X86_64_XMM_SCRATCH_REG: RegLoc = XMM5;
#[cfg(not(target_os = "windows"))]
pub const X86_64_XMM_SCRATCH_REG: RegLoc = XMM15;

#[cfg(test)]
mod loc_size_tests {
    #[test]
    fn loc_leaves_the_24_byte_class() {
        assert!(
            std::mem::size_of::<super::Loc>() <= 16,
            "Loc is {} B; vec![loc] must stay out of the 24-byte class",
            std::mem::size_of::<super::Loc>()
        );
        assert!(std::mem::size_of::<super::FrameLoc>() <= 12);
        assert!(std::mem::size_of::<super::ImmedLoc>() <= 8);
    }
}
