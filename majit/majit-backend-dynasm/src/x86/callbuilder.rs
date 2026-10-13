//! Port of `rpython/jit/backend/x86/callbuilder.py`.
//!
//! ABI argument-register tables (`CallBuilder64.ARGUMENTS_GPR` /
//! `ARGUMENTS_XMM`) and the stack-alignment helpers
//! (`align_stack_words`, `CallBuilderX86.subtract_esp_aligned`).

use super::arch::WORD;
use crate::regloc::{
    ECX, EDI, EDX, ESI, R8, R9, RegLoc, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
};

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
