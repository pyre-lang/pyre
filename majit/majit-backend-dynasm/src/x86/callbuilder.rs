//! Port of `rpython/jit/backend/x86/callbuilder.py`.
//!
//! ABI argument-register tables (`CallBuilder64.ARGUMENTS_GPR` /
//! `ARGUMENTS_XMM`), the stack-alignment helpers (`align_stack_words`,
//! `CallBuilderX86.subtract_esp_aligned`), `follow_jump`, and the per-call
//! builder itself: `CallBuilderX86` and `CallBuilder64` collapse into one
//! [`CallBuilder64`], because x86-64 is the only `CallBuilder` this backend
//! builds (`CallBuilder = CallBuilder64` under `IS_X86_64`). The
//! `AbstractCallBuilder` drivers (`emit`, `emit_no_collect`,
//! `emit_call_release_gil`) are its methods as well.

use dynasmrt::dynasm;

use super::arch::{PASS_ON_MY_FRAME, THREADLOCAL_OFS, WORD};
use super::assembler::Assembler386;
#[cfg(target_os = "windows")]
use super::regloc::imm;
use super::rx86;
use crate::jump::{RegallocMoves, remap_frame_layout};
use crate::regloc::{
    EAX, ECX, EDX, Loc, R8, R9, R10, R12, RawEspLoc, RegLoc, X86_64_SCRATCH_REG,
    X86_64_XMM_SCRATCH_REG, XMM0, XMM1, XMM2, XMM3, ebp_loc_pat,
};
#[cfg(not(target_os = "windows"))]
use crate::regloc::{EDI, ESI, XMM4, XMM5, XMM6, XMM7};

/// `callbuilder.py` `CALL_ALIGN`: stack words rounded up to 16 bytes.
pub const CALL_ALIGN: i32 = 16 / WORD as i32;

/// `callbuilder.py` `align_stack_words`.
pub fn align_stack_words(words: i32) -> i32 {
    (words + CALL_ALIGN - 1) & !(CALL_ALIGN - 1)
}

/// `CallBuilder64.prepare_arguments` `on_stack` before
/// `subtract_esp_aligned(on_stack - stack_max)`. Win64 starts at 4 shadow
/// words; SysV at 0. `CallBuilderX86.stack_max` is `PASS_ON_MY_FRAME`,
/// already reserved by `_call_header`. Callers subtract that scratch from
/// this count; they do not pass the raw outgoing area to
/// `subtract_esp_aligned`.
pub fn outgoing_stack_words(stack_slots: usize) -> i32 {
    #[cfg(target_os = "windows")]
    {
        4 + stack_slots as i32
    }
    #[cfg(not(target_os = "windows"))]
    {
        stack_slots as i32
    }
}

/// Bytes `CallBuilderX86.subtract_esp_aligned` subtracts for `count`
/// extra stack words (`on_stack - PASS_ON_MY_FRAME`). `count <= 0` is a
/// no-op: the outgoing area fits in the scratch.
pub fn aligned_esp_bytes(count: i32) -> i32 {
    if count > 0 {
        align_stack_words(count) * WORD as i32
    } else {
        0
    }
}

/// callbuilder.py `CallBuilder64.ARGUMENTS_GPR`
#[cfg(not(target_os = "windows"))]
pub const ARGUMENTS_GPR: &[RegLoc] = &[EDI, ESI, EDX, ECX, R8, R9];
#[cfg(target_os = "windows")]
pub const ARGUMENTS_GPR: &[RegLoc] = &[ECX, EDX, R8, R9];

/// callbuilder.py `CallBuilder64.ARGUMENTS_XMM`
#[cfg(not(target_os = "windows"))]
pub const ARGUMENTS_XMM: &[RegLoc] = &[XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7];
#[cfg(target_os = "windows")]
pub const ARGUMENTS_XMM: &[RegLoc] = &[XMM0, XMM1, XMM2, XMM3];

/// `follow_jump`: if `addr` starts with `E9 rel32`, step to that target
/// and repeat. Any other leading byte, including `EB`, stops the walk.
/// `CallBuilderX86.__init__` does this for an immediate `fnloc`.
pub fn follow_jump(addr: i64) -> i64 {
    let mut addr = addr as u64;
    loop {
        let ptr = addr as *const u8;
        let byte = unsafe { ptr.read() };
        if byte != 0xE9 {
            break;
        }
        let rel = unsafe { ptr.add(1).cast::<i32>().read_unaligned() };
        addr = addr.wrapping_add(5).wrapping_add(i64::from(rel) as u64);
    }
    addr as i64
}
/// `history.py` `FLOAT`: the `restype` of a double result.
const FLOAT: char = 'f';

/// `regloc.py` `AssemblerLocation.is_float`.
fn loc_is_float(loc: &Loc) -> bool {
    match loc {
        Loc::Reg(r) => r.is_xmm,
        ebp_loc_pat!(e) => e.is_float,
        Loc::Esp(e) => e.is_float,
        Loc::ImmedFloat(_) | Loc::ConstFloat(_) => true,
        Loc::Immed(_) | Loc::Addr(_) => false,
    }
}

/// `callbuilder.py` `CallBuilderX86` + `CallBuilder64`: one per emitted
/// call, borrowing the assembler for the duration of the emission.
pub(crate) struct CallBuilder64<'m, 'a> {
    asm: &'m mut Assembler386<'a>,
    /// The followed immediate from `__init__`, or the argument slot
    /// `prepare_arguments` moved a non-immediate fnloc into.
    fnloc: Option<Loc>,
    arglocs: Vec<Loc>,
    resloc: Option<Loc>,
    /// `get_result_type()`: `'i'`, `'r'`, `'f'`, `'v'` or `'S'`.
    restype: char,
    ressize: usize,
    pub(crate) ressign: bool,
    /// `get_arg_types()`: the arg-class string, which alone tells a
    /// singlefloat (`'S'`) from an integer.
    pub(crate) argtypes: String,
    is_call_release_gil: bool,
    fnloc_is_immediate: bool,
    /// `get_current_esp()`, `start_frame_size - mc._frame_size`: negative
    /// while rsp sits below the body rsp.
    current_esp: i32,
    tlofs_reg: Option<RegLoc>,
    result_value_saved_early: bool,
    next_arg_gpr: usize,
    next_arg_xmm: usize,
    #[cfg_attr(not(target_os = "windows"), allow(dead_code))]
    win64_arg_gpr: u8,
    #[cfg_attr(not(target_os = "windows"), allow(dead_code))]
    win64_arg_xmm: u8,
    /// `push_gcmap` published the gcmap the regalloc left pending, so
    /// `pop_gcmap` clears it.
    gcmap_pushed: bool,
}

impl<'m, 'a> CallBuilder64<'m, 'a> {
    /// `CallBuilderX86.__init__`. Avoid tons of issues with a non-immediate
    /// fnloc by sticking it as an extra argument if needed.
    pub(crate) fn new(
        asm: &'m mut Assembler386<'a>,
        fnloc: Loc,
        arglocs: &[Loc],
        resloc: Option<Loc>,
        restype: char,
        ressize: usize,
    ) -> Self {
        let mut arglocs = arglocs.to_vec();
        let fnloc = match fnloc {
            Loc::Immed(i) => Some(Loc::immed(follow_jump(i.value))),
            other => {
                arglocs.push(other);
                None
            }
        };
        CallBuilder64 {
            asm,
            fnloc_is_immediate: fnloc.is_some(),
            fnloc,
            arglocs,
            resloc,
            restype,
            ressize,
            ressign: false,
            argtypes: String::new(),
            is_call_release_gil: false,
            current_esp: 0,
            tlofs_reg: None,
            result_value_saved_early: false,
            next_arg_gpr: 0,
            next_arg_xmm: 0,
            win64_arg_gpr: 0,
            win64_arg_xmm: 0,
            gcmap_pushed: false,
        }
    }

    /// `AbstractCallBuilder.emit_no_collect`: emit a call that cannot
    /// collect.
    pub(crate) fn emit_no_collect(&mut self) {
        self.prepare_arguments();
        self.emit_raw_call();
        self.restore_stack_pointer(0);
        self.load_result();
    }

    /// `AbstractCallBuilder.emit`: emit a regular call; not for
    /// CALL_RELEASE_GIL.
    pub(crate) fn emit(&mut self) {
        self.prepare_arguments();
        self.push_gcmap();
        self.emit_raw_call();
        self.restore_stack_pointer(0);
        self.pop_gcmap();
        self.load_result();
    }

    /// `AbstractCallBuilder.emit_call_release_gil`, including the calls
    /// around the raw call. `save_err` is a combination of `RFFI_*ERR*`.
    ///
    /// The JIT keeps the GIL across the call: neither
    /// `call_releasegil_addr_and_move_real_arguments` nor the `rpy_fastgil`
    /// half of `move_real_result_and_call_reacqgil_addr` is emitted.
    pub(crate) fn emit_call_release_gil(&mut self, save_err: i64) {
        self.select_call_release_gil_mode();
        self.prepare_arguments();
        self.push_gcmap_for_call_release_gil();
        self.write_real_errno(save_err);
        self.emit_raw_call();
        self.restore_stack_pointer(0);
        self.read_real_errno(save_err);
        self.move_real_result_and_call_reacqgil_addr();
        self.pop_gcmap();
        self.load_result();
    }

    /// `AbstractCallBuilder.select_call_release_gil_mode`.
    fn select_call_release_gil_mode(&mut self) {
        self.is_call_release_gil = true;
    }

    /// `CallBuilderX86.subtract_esp_aligned`.
    fn subtract_esp_aligned(&mut self, count: i32) {
        if count > 0 {
            let align = align_stack_words(count);
            self.sub_esp(align * WORD as i32);
        }
    }

    /// `mc.SUB_ri(esp.value, bytes)`, which moves `mc._frame_size`.
    fn sub_esp(&mut self, bytes: i32) {
        rx86::sub_ri(&mut self.asm.mc, rx86::ESP, bytes);
        self.current_esp -= bytes;
    }

    /// `mc.ADD_ri(esp.value, bytes)`, which moves `mc._frame_size`.
    fn add_esp(&mut self, bytes: i32) {
        rx86::add_ri(&mut self.asm.mc, rx86::ESP, bytes);
        self.current_esp += bytes;
    }

    /// `CallBuilderX86.get_current_esp`.
    fn get_current_esp(&self) -> i32 {
        self.current_esp
    }

    /// `CallBuilderX86.restore_stack_pointer`.
    fn restore_stack_pointer(&mut self, target_esp: i32) {
        let current_esp = self.get_current_esp();
        if current_esp != target_esp {
            self.add_esp(target_esp - current_esp);
        }
    }

    /// `CallBuilder64.load_result`, falling back to
    /// `CallBuilderX86.load_result`, which uses `load_from_mem` to do the
    /// zero- or sign-extension.
    fn load_result(&mut self) {
        let Some(Loc::Reg(resloc)) = self.resloc else {
            return;
        };
        if self.restype == 'S' {
            // singlefloat return: use MOVD to load the target register
            // from the lower 32 bits of XMM0
            self.asm.mc.MOVD32(resloc, XMM0);
            return;
        }
        if self.ressize == 0 {
            return; // void result
        }
        let srcloc = if self.restype == FLOAT { XMM0 } else { EAX };
        if self.ressize >= WORD && resloc == srcloc {
            return; // no need for any MOV
        }
        if self.ressize >= WORD {
            self.asm.regalloc_mov(&Loc::Reg(srcloc), &Loc::Reg(resloc));
            return;
        }
        // `load_from_mem` from a register: MOVSX / MOVZX of the low part.
        let (dst, src) = (resloc, srcloc);
        match (self.ressize, self.ressign) {
            (1, true) => self.asm.mc.MOVSX8(dst, src),
            (1, false) => self.asm.mc.MOVZX8(dst, src),
            (2, true) => self.asm.mc.MOVSX16(dst, src),
            (2, false) => self.asm.mc.MOVZX16(dst, src),
            (4, true) => self.asm.mc.MOVSX32(dst, src),
            (4, false) => self.asm.mc.MOV32(dst, src),
            (size, _) => panic!("load_from_mem: size {size} not in {{1, 2, 4, WORD}}"),
        }
    }

    /// `CallBuilderX86.push_gcmap`: we push *now* the gcmap, describing
    /// the status of GC registers after the rearrangements done just
    /// before, ignoring the return value eax. The regalloc built it
    /// (`get_gcmap([eax])`) and left it pending for this call.
    fn push_gcmap(&mut self) {
        debug_assert!(!self.is_call_release_gil);
        self.gcmap_pushed = self.asm.push_pending_call_gcmap();
    }

    /// `AbstractCallBuilder.push_gcmap_for_call_release_gil`. Every GC
    /// value is out of the registers by now (`save_all_regs`), so the
    /// pending gcmap is the `noregs` one.
    fn push_gcmap_for_call_release_gil(&mut self) {
        debug_assert!(self.is_call_release_gil);
        self.gcmap_pushed = self.asm.push_pending_call_gcmap();
    }

    /// `CallBuilderX86.pop_gcmap`: `_reload_frame_if_necessary`, then
    /// `pop_gcmap`.
    fn pop_gcmap(&mut self) {
        self.asm
            .pop_pending_call_gcmap_after_collect(self.gcmap_pushed);
    }

    /// `CallBuilderX86.get_tlofs_reg`: load the THREADLOCAL_OFS from the
    /// stack into a callee-saved register. Further calls just return the
    /// same register, by assuming it is indeed saved.
    fn get_tlofs_reg(&mut self) -> RegLoc {
        debug_assert!(self.is_call_release_gil);
        if let Some(reg) = self.tlofs_reg {
            return reg;
        }
        // pick a register saved across calls
        let reg = R12;
        let ofs = THREADLOCAL_OFS as i32 - self.get_current_esp();
        rx86::mov_rs(&mut self.asm.mc, reg.value, ofs);
        self.tlofs_reg = Some(reg);
        reg
    }

    /// `CallBuilderX86.write_real_errno`. This occurs just before
    /// `emit_raw_call()`.
    fn write_real_errno(&mut self, save_err: i64) {
        use majit_jitcode::rffi::{RFFI_ALT_ERRNO, RFFI_READSAVED_ERRNO, RFFI_ZERO_ERRNO_BEFORE};
        use majit_rlib::rthread::{
            TLFIELD_ALT_ERRNO_OFS, TLFIELD_P_ERRNO_OFS, TLFIELD_RPY_ERRNO_OFS,
        };

        #[cfg(target_os = "windows")]
        if save_err & majit_jitcode::rffi::RFFI_READSAVED_LASTERROR != 0 {
            // must call SetLastError(). On win64 we save (again) the
            // registers containing arguments, and restore them afterwards.
            use majit_rlib::rthread::{TLFIELD_ALT_LASTERROR_OFS, TLFIELD_RPY_LASTERROR_OFS};
            let set_last_error_addr = majit_rlib::rwin32::_SetLastError as *const () as i64;
            let lasterror = if save_err & RFFI_ALT_ERRNO != 0 {
                TLFIELD_ALT_LASTERROR_OFS
            } else {
                TLFIELD_RPY_LASTERROR_OFS
            } as i32;
            let tlofsreg = self.get_tlofs_reg(); // => r12, callee-saved
            self.win64_save_register_args();
            rx86::mov_rm(&mut self.asm.mc, ECX.value, (tlofsreg.value, lasterror));
            self.asm.mc.CALL(imm(follow_jump(set_last_error_addr)));
            self.win64_restore_register_args();
        }

        if save_err & RFFI_READSAVED_ERRNO != 0 {
            // Just before a call, read '*_errno' and write it into the
            // real 'errno'. Most registers are free here, including the
            // callee-saved ones, except 'ebx' and except the ones used to
            // pass the arguments on x86-64.
            let rpy_errno = if save_err & RFFI_ALT_ERRNO != 0 {
                TLFIELD_ALT_ERRNO_OFS
            } else {
                TLFIELD_RPY_ERRNO_OFS
            } as i32;
            let p_errno = TLFIELD_P_ERRNO_OFS as i32;
            let tlofsreg = self.get_tlofs_reg(); // => r12, callee-saved
            let tmpreg = R10; // edx is used for 3rd argument
            rx86::mov_rm(&mut self.asm.mc, tmpreg.value, (tlofsreg.value, p_errno));
            rx86::mov32_rm(&mut self.asm.mc, EAX.value, (tlofsreg.value, rpy_errno));
            rx86::mov32_mr(&mut self.asm.mc, (tmpreg.value, 0), EAX.value);
        } else if save_err & RFFI_ZERO_ERRNO_BEFORE != 0 {
            // Same, but write zero.
            let p_errno = TLFIELD_P_ERRNO_OFS as i32;
            let tlofsreg = self.get_tlofs_reg(); // => r12, callee-saved
            rx86::mov_rm(&mut self.asm.mc, EAX.value, (tlofsreg.value, p_errno));
            rx86::mov32_mi(&mut self.asm.mc, (EAX.value, 0), 0);
        }
    }

    /// `CallBuilderX86.read_real_errno`. This occurs after
    /// `emit_raw_call()` and after `restore_stack_pointer()`.
    fn read_real_errno(&mut self, save_err: i64) {
        use majit_jitcode::rffi::{RFFI_ALT_ERRNO, RFFI_SAVE_ERRNO};
        use majit_rlib::rthread::{
            TLFIELD_ALT_ERRNO_OFS, TLFIELD_P_ERRNO_OFS, TLFIELD_RPY_ERRNO_OFS,
        };

        if save_err & RFFI_SAVE_ERRNO != 0 {
            // Just after a call, read the real 'errno' and save a copy of
            // it inside our thread-local '*_errno'. Most registers are
            // free here, including the callee-saved ones, except 'ebx'.
            // The tlofs register might have been loaded earlier and is
            // callee-saved, so it does not need to be reloaded.
            let rpy_errno = if save_err & RFFI_ALT_ERRNO != 0 {
                TLFIELD_ALT_ERRNO_OFS
            } else {
                TLFIELD_RPY_ERRNO_OFS
            } as i32;
            let p_errno = TLFIELD_P_ERRNO_OFS as i32;
            let tlofsreg = self.get_tlofs_reg(); // => r12 (possibly reused)
            rx86::mov_rm(&mut self.asm.mc, ECX.value, (tlofsreg.value, p_errno));
            rx86::mov32_rm(&mut self.asm.mc, ECX.value, (ECX.value, 0));
            rx86::mov32_mr(&mut self.asm.mc, (tlofsreg.value, rpy_errno), ECX.value);
        }

        #[cfg(target_os = "windows")]
        {
            use majit_jitcode::rffi::{RFFI_SAVE_LASTERROR, RFFI_SAVE_WSALASTERROR};
            use majit_rlib::rthread::{TLFIELD_ALT_LASTERROR_OFS, TLFIELD_RPY_LASTERROR_OFS};
            if save_err & (RFFI_SAVE_LASTERROR | RFFI_SAVE_WSALASTERROR) != 0 {
                let get_last_error_addr = if save_err & RFFI_SAVE_LASTERROR != 0 {
                    majit_rlib::rwin32::_GetLastError as *const () as i64
                } else {
                    majit_rlib::_rsocket_rffi::_WSAGetLastError as *const () as i64
                };
                let lasterror = if save_err & RFFI_ALT_ERRNO != 0 {
                    TLFIELD_ALT_LASTERROR_OFS
                } else {
                    TLFIELD_RPY_LASTERROR_OFS
                } as i32;
                self.save_result_value(true); // save eax/edx/xmm0
                self.result_value_saved_early = true;
                // `save_result_value` stored into [ESP], which is the Win64
                // home space of a function called right here, and the x64
                // ABI lets that callee spill its argument registers there.
                // The call gets its own 32-byte home space below, as
                // `win64_save_register_args` gives `SetLastError`.
                self.sub_esp(4 * WORD as i32);
                self.asm.mc.CALL(imm(follow_jump(get_last_error_addr)));
                self.add_esp(4 * WORD as i32);
                let tlofsreg = self.get_tlofs_reg(); // => r12 (possibly reused)
                rx86::mov32_mr(&mut self.asm.mc, (tlofsreg.value, lasterror), EAX.value);
            }
        }
    }

    /// The part of `CallBuilderX86.move_real_result_and_call_reacqgil_addr`
    /// that does not touch `rpy_fastgil`: restore the result
    /// `read_real_errno` saved early.
    fn move_real_result_and_call_reacqgil_addr(&mut self) {
        if self.result_value_saved_early {
            self.restore_result_value(true);
        }
    }

    /// `CallBuilder64.save_result_value`: temporarily save the result
    /// value into [ESP].
    #[cfg_attr(not(target_os = "windows"), allow(dead_code))]
    fn save_result_value(&mut self, _save_edx: bool) {
        if self.ressize == 0 {
            return; // void return
        }
        if self.restype == FLOAT {
            // and not 'S'
            rx86::movsd_sx(&mut self.asm.mc, 0, XMM0.value);
            return;
        }
        if self.restype == 'S' {
            // singlefloat return: use MOVD to store the lower 32 bits
            // of XMM0 into [ESP]
            dynasm!(self.asm.mc ; .arch x64 ; movd DWORD [rsp], xmm0);
        } else {
            rx86::mov_sr(&mut self.asm.mc, 0, EAX.value);
        }
    }

    /// `CallBuilder64.restore_result_value`: opposite of
    /// `save_result_value()`.
    fn restore_result_value(&mut self, _save_edx: bool) {
        if self.ressize == 0 {
            return; // void return
        }
        if self.restype == FLOAT {
            // and not 'S'
            rx86::movsd_xs(&mut self.asm.mc, XMM0.value, 0);
            return;
        }
        if self.restype == 'S' {
            dynasm!(self.asm.mc ; .arch x64 ; movd xmm0, DWORD [rsp]);
        } else {
            rx86::mov_rs(&mut self.asm.mc, EAX.value, 0);
        }
    }

    /// `CallBuilder64._unused_gpr`.
    fn _unused_gpr(&mut self, _hint: &Loc) -> Option<RegLoc> {
        let i = self.next_arg_gpr;
        self.next_arg_gpr = i + 1;
        if cfg!(target_os = "windows") {
            self.next_arg_xmm = self.next_arg_gpr;
        }
        let res = *ARGUMENTS_GPR.get(i)?;
        if cfg!(target_os = "windows") {
            self.win64_arg_gpr |= 1 << i;
        }
        Some(res)
    }

    /// `CallBuilder64._unused_xmm`.
    fn _unused_xmm(&mut self) -> Option<RegLoc> {
        let i = self.next_arg_xmm;
        self.next_arg_xmm = i + 1;
        if cfg!(target_os = "windows") {
            self.next_arg_gpr = self.next_arg_xmm;
        }
        let res = *ARGUMENTS_XMM.get(i)?;
        if cfg!(target_os = "windows") {
            self.win64_arg_xmm |= 1 << i;
        }
        Some(res)
    }

    /// `CallBuilder64.win64_save_register_args`. Arguments 0 to 3 are in
    /// argument registers; further arguments are already in the stack
    /// beyond 4 unused words that will be the shadow space for the CALL.
    /// We save arguments 0 to 3 inside that space, then reserve another
    /// shadow space for the call to SetLastError that follows.
    #[cfg(target_os = "windows")]
    fn win64_save_register_args(&mut self) {
        debug_assert_eq!(ARGUMENTS_GPR.len(), 4);
        debug_assert_eq!(ARGUMENTS_XMM.len(), 4);
        for i in 0..4 {
            let ofs = (i * WORD) as i32;
            if self.win64_arg_gpr & (1 << i) != 0 {
                rx86::mov_sr(&mut self.asm.mc, ofs, ARGUMENTS_GPR[i].value);
            } else if self.win64_arg_xmm & (1 << i) != 0 {
                rx86::movsd_sx(&mut self.asm.mc, ofs, ARGUMENTS_XMM[i].value);
            }
        }
        self.sub_esp(4 * WORD as i32);
    }

    /// `CallBuilder64.win64_restore_register_args`.
    #[cfg(target_os = "windows")]
    fn win64_restore_register_args(&mut self) {
        self.add_esp(4 * WORD as i32);
        for i in 0..4 {
            let ofs = (i * WORD) as i32;
            if self.win64_arg_gpr & (1 << i) != 0 {
                rx86::mov_rs(&mut self.asm.mc, ARGUMENTS_GPR[i].value, ofs);
            } else if self.win64_arg_xmm & (1 << i) != 0 {
                rx86::movsd_xs(&mut self.asm.mc, ARGUMENTS_XMM[i].value, ofs);
            }
        }
    }

    /// `CallBuilder64.prepare_arguments`.
    fn prepare_arguments(&mut self) {
        let mut src_locs: Vec<Loc> = Vec::new();
        let mut dst_locs: Vec<Loc> = Vec::new();
        let mut xmm_src_locs: Vec<Loc> = Vec::new();
        let mut xmm_dst_locs: Vec<Loc> = Vec::new();
        let mut singlefloats: Vec<(Loc, Loc)> = Vec::new();

        let arglocs = std::mem::take(&mut self.arglocs);
        let argtypes = std::mem::take(&mut self.argtypes);
        let is_singlefloat = |i: usize| argtypes.as_bytes().get(i) == Some(&b'S');

        // Win64: shadow parameters (space reserved by the caller)
        let mut on_stack: i32 = if cfg!(target_os = "windows") { 4 } else { 0 };
        let stack_target = |on_stack: &mut i32, is_float: bool| {
            let tgt = Loc::Esp(RawEspLoc::new(*on_stack * WORD as i32, is_float));
            *on_stack += 1;
            tgt
        };
        for (i, loc) in arglocs.iter().enumerate() {
            debug_assert!(
                argtypes
                    .as_bytes()
                    .get(i)
                    .is_none_or(|&class| (class == b'f') == loc_is_float(loc)),
                "call argument {i} at {loc:?} disagrees with its class in {argtypes:?}"
            );
            if loc_is_float(loc) {
                let tgt = match self._unused_xmm() {
                    Some(reg) => Loc::Reg(reg),
                    None => stack_target(&mut on_stack, true),
                };
                xmm_src_locs.push(*loc);
                xmm_dst_locs.push(tgt);
            } else if is_singlefloat(i) {
                // Singlefloat argument
                let tgt = match self._unused_xmm() {
                    Some(reg) => Loc::Reg(reg),
                    None => stack_target(&mut on_stack, false),
                };
                singlefloats.push((*loc, tgt));
            } else {
                let tgt = match self._unused_gpr(loc) {
                    Some(reg) => Loc::Reg(reg),
                    None => stack_target(&mut on_stack, false),
                };
                src_locs.push(*loc);
                dst_locs.push(tgt);
            }
        }

        if !self.fnloc_is_immediate {
            // the last "argument" prepared above
            let fnloc = *dst_locs
                .last()
                .expect("a non-immediate fnloc is the last argument");
            self.fnloc = Some(fnloc);
        }

        if cfg!(debug_assertions) {
            // assert that we got the right stack depth
            let floats = arglocs
                .iter()
                .enumerate()
                .filter(|(i, arg)| loc_is_float(arg) || is_singlefloat(*i))
                .count() as i32;
            let all_args = arglocs.len() as i32;
            let stack_depth = if cfg!(target_os = "windows") {
                all_args.max(4)
            } else {
                (all_args - floats - ARGUMENTS_GPR.len() as i32).max(0)
                    + (floats - ARGUMENTS_XMM.len() as i32).max(0)
            };
            assert_eq!(stack_depth, on_stack);
        }

        self.subtract_esp_aligned(on_stack - PASS_ON_MY_FRAME as i32);

        // Handle register arguments: first remap the xmm arguments
        let tmpreg = Loc::Reg(X86_64_XMM_SCRATCH_REG);
        remap_frame_layout(&mut *self.asm, &xmm_src_locs, &xmm_dst_locs, tmpreg);
        // Load the singlefloat arguments from main regs or stack to xmm regs
        for (src, dst) in singlefloats {
            match dst {
                Loc::Esp(dst) => self.mov32_to_esp(src, dst),
                Loc::Reg(dst) => {
                    let src = if let Loc::Immed(_) = src {
                        let scratch = Loc::Reg(X86_64_SCRATCH_REG);
                        self.asm.regalloc_mov(&src, &scratch);
                        scratch
                    } else {
                        src
                    };
                    self.movd32_to_xmm(dst, src);
                }
                other => unreachable!("singlefloat argument destination {other:?}"),
            }
        }
        // Finally remap the arguments in the main regs
        let tmpreg = Loc::Reg(X86_64_SCRATCH_REG);
        remap_frame_layout(&mut *self.asm, &src_locs, &dst_locs, tmpreg);

        self.arglocs = arglocs;
        self.argtypes = argtypes;
    }

    /// `prepare_arguments` for a singlefloat that spilled to the stack:
    /// `MOV32(dst, src)`, through the scratch register for a frame slot.
    fn mov32_to_esp(&mut self, src: Loc, dst: RawEspLoc) {
        let dst = Loc::Esp(dst);
        match src {
            // XXX too much special logic
            Loc::Frame(_) | Loc::Ebp(_) => {
                self.asm.mc.MOV32(X86_64_SCRATCH_REG, &src);
                self.asm.mc.MOV32(&dst, X86_64_SCRATCH_REG);
            }
            Loc::Reg(r) if r.is_xmm => panic!("singlefloat stack argument location {src:?}"),
            _ => self.asm.mc.MOV32(&dst, &src),
        }
    }

    /// `prepare_arguments`'s `MOVD32(dst, src)` of a singlefloat into its
    /// argument xmm register.
    fn movd32_to_xmm(&mut self, dst: RegLoc, src: Loc) {
        self.asm.mc.MOVD32(dst, &src);
    }

    /// `CallBuilder64.emit_raw_call`: `CALL(self.fnloc)`.
    fn emit_raw_call(&mut self) {
        let fnloc = self.fnloc.expect("emit_raw_call before prepare_arguments");
        self.asm.mc.CALL(&fnloc);
    }
}
