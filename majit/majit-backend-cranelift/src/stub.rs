//! Out-of-line guard-failure stubs and `patch_jump_for_descr`.
//!
//! `x86/assembler.py generate_quick_failure` / `write_pending_failure_recoveries`
//! emit a per-guard stub after the loop body; `patch_pending_failure_recoveries`
//! stores the stub address on `faildescr.adr_jump_offset`. `patch_jump_for_descr`
//! (x86) / `patch_trace` (aarch64) later overwrite that stub with a jump to the
//! attached bridge. Dynasm ports the same pair as `generate_quick_failure` and
//! `write_redirect_branch`.
//!
//! Cranelift cannot lay the stub down inside the loop function (Ion would
//! allocate across it), so each guard gets a 20-byte trampoline in the
//! assembler arena. The loop's failure block `return_call`s that trampoline
//! (`CallConv::Tail`); the trampoline is a `JMP`/`B` to either the shared
//! `_build_failure_recovery` helper or, once a bridge is attached, the
//! bridge body. The hop is `JMP`/`B`, not `BL`: `return_call` already tore
//! the loop frame down, and a linked branch would leak a return address
//! (`emit_attached_bridge_dispatch` previously did that under closing_jump).

use majit_ir::{DescrRef, FailDescr};

/// Bytes reserved for one trampoline. Fits the aarch64 far veneer
/// (`MOVZ`/`MOVK`×3 + `BR x16`, 20 bytes) and the x86_64 far form
/// (`MOV r11, imm64; JMP r11`, 13 bytes).
pub(crate) const STUB_SIZE: usize = 20;

fn with_writable<F: FnOnce()>(addr: *mut u8, len: usize, f: F) {
    {
        let _writing = majit_backend::AssemblerWriting::enter();
        f();
    }
    majit_backend::flush_instruction_cache(addr as *const u8, len);
}

/// Encode a Tail-compatible redirect at `at` that lands on `target`.
///
/// x86: `assembler.py patch_jump_for_descr` `JMP rel32` / `MOV r11; JMP r11`.
/// aarch64: `assembler.py patch_trace` overwrites the stub, but with `B`
/// rather than `BL` — see the module comment.
pub(crate) fn encode_redirect_branch(at: usize, target: usize) -> [u8; STUB_SIZE] {
    let mut bytes = [0u8; STUB_SIZE];
    #[cfg(target_arch = "x86_64")]
    {
        let offset = target as isize - (at as isize + 5);
        if offset >= i32::MIN as isize && offset <= i32::MAX as isize {
            bytes[0] = 0xE9;
            bytes[1..5].copy_from_slice(&(offset as i32).to_le_bytes());
        } else {
            bytes[0] = 0x49;
            bytes[1] = 0xBB;
            bytes[2..10].copy_from_slice(&(target as u64).to_le_bytes());
            bytes[10] = 0x41;
            bytes[11] = 0xFF;
            bytes[12] = 0xE3;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        assert!(
            at & 0b11 == 0 && target & 0b11 == 0,
            "AArch64 redirect branch endpoints must be 4-byte aligned: at={at:#x}, target={target:#x}"
        );
        let offset = target as isize - at as isize;
        const B_REACH: isize = 1 << 27;
        let words: [u32; 5] = if (-B_REACH..B_REACH).contains(&offset) {
            let imm26 = ((offset >> 2) & 0x03FF_FFFF) as u32;
            [
                0x1400_0000 | imm26,
                0xD503_201F,
                0xD503_201F,
                0xD503_201F,
                0xD503_201F,
            ]
        } else {
            let v = target as u64;
            let rd = 16u32;
            let imm16 = |shift: u32| (((v >> shift) & 0xFFFF) as u32) << 5;
            [
                0xD280_0000 | imm16(0) | rd,
                0xF280_0000 | (1 << 21) | imm16(16) | rd,
                0xF280_0000 | (2 << 21) | imm16(32) | rd,
                0xF280_0000 | (3 << 21) | imm16(48) | rd,
                0xD61F_0000 | (rd << 5),
            ]
        };
        for (i, w) in words.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (at, target);
        panic!("guard-stub patching is implemented for x86_64 and aarch64");
    }
    bytes
}

/// Write a redirect at `at`. `at` must name `STUB_SIZE` writable, executable
/// bytes.
pub(crate) fn write_redirect_branch(at: usize, target: usize) {
    let bytes = encode_redirect_branch(at, target);
    with_writable(at as *mut u8, STUB_SIZE, || unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, STUB_SIZE);
    });
}

/// `x86/callbuilder.py follow_jump`: if `addr` starts with another jump,
/// step to that target and repeat. x86 walks `JMP rel32` (`E9`); aarch64
/// walks an unlinked `B` (`0x14`). A far veneer (x86 `MOV r11` / aarch64
/// `MOVZ x16`) stops the walk, matching `follow_jump` ignoring anything
/// but `E9`.
pub(crate) fn follow_jump(mut addr: usize) -> usize {
    loop {
        #[cfg(target_arch = "x86_64")]
        {
            let byte = unsafe { (addr as *const u8).read() };
            if byte != 0xE9 {
                return addr;
            }
            let rel = unsafe { (addr as *const u8).add(1).cast::<i32>().read_unaligned() };
            addr = addr.wrapping_add(5).wrapping_add(rel as i64 as usize);
        }
        #[cfg(target_arch = "aarch64")]
        {
            let word = unsafe { (addr as *const u32).read() };
            if word & 0xFC00_0000 != 0x1400_0000 {
                return addr;
            }
            let imm26 = word & 0x03FF_FFFF;
            let offset = ((imm26 as i32) << 6) >> 6 << 2;
            addr = (addr as isize).wrapping_add(offset as isize) as usize;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            return addr;
        }
    }
}

/// Bytes a `redirect_call_assembler` overlay occupies.
///
/// `x86/assembler.py redirect_call_assembler` asserts `get_relative_pos() == 5`
/// (32-bit) or `<= 13` (64-bit). `aarch64/assembler.py redirect_call_assembler`
/// writes one `B` (4 bytes) or a far veneer. Guard trampolines pad to
/// [`STUB_SIZE`]; a live loop entry must not be NOP-padded.
pub(crate) fn entry_redirect_len(at: usize, target: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        let offset = target as isize - (at as isize + 5);
        if offset >= i32::MIN as isize && offset <= i32::MAX as isize {
            5
        } else {
            13
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let offset = target as isize - at as isize;
        const B_REACH: isize = 1 << 27;
        if (-B_REACH..B_REACH).contains(&offset) {
            4
        } else {
            STUB_SIZE
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (at, target);
        STUB_SIZE
    }
}

/// `x86/assembler.py redirect_call_assembler` `mc.copy_to_raw_memory(oldadr)`.
///
/// Overlay a JMP/B at a compiled loop entry. Writes only
/// [`entry_redirect_len`] bytes so the rest of the live prologue is left
/// intact for threads already past the first instruction.
pub(crate) fn write_entry_redirect(at: usize, target: usize) {
    let bytes = encode_redirect_branch(at, target);
    let len = entry_redirect_len(at, target);
    with_writable(at as *mut u8, len, || unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, len);
    });
}

/// Default trampoline: jump to the shared `_build_failure_recovery` helper.
pub(crate) fn emit_recovery_stub(at: usize, recovery: usize) {
    write_redirect_branch(at, recovery);
}

/// `x86/assembler.py patch_pending_failure_recoveries`.
///
/// `assemble_loop` / `assemble_bridge` collect `(faildescr, addr)` in
/// `pending_guard_tokens` while emitting stubs, then stamp
/// `faildescr.adr_jump_offset` only after `materialize_loop` succeeded.
/// Call this once the trampoline blocks are owned by the compiled
/// loop/bridge that will be published and no later compile step can
/// fail. Dropping `pending` on an error path leaves each descr's
/// previous offset in place.
pub(crate) fn patch_pending_failure_recoveries(pending_guard_tokens: &[(DescrRef, usize)]) {
    for (descr, addr) in pending_guard_tokens {
        if let Some(fd) = descr.as_fail_descr() {
            fd.set_adr_jump_offset(*addr);
        }
    }
}

/// `x86/assembler.py patch_jump_for_descr` / `aarch64/assembler.py patch_trace`.
///
/// `adr_jump_offset` is the trampoline address (`generate_quick_failure` /
/// `patch_pending_failure_recoveries`). Overwrite it with a jump to the
/// bridge body and clear the slot ("patched").
pub(crate) fn patch_jump_for_descr(descr: &dyn FailDescr, adr_new_target: usize) {
    let stub_addr = descr.adr_jump_offset();
    assert!(stub_addr != 0, "guard already patched");
    write_redirect_branch(stub_addr, adr_new_target);
    descr.set_adr_jump_offset(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follow_jump_walks_a_near_jump_chain() {
        let mut buf = vec![0u8; 64];
        let at = buf.as_mut_ptr() as usize;
        let mid = at + 16;
        let end = at + 32;
        let first = encode_redirect_branch(at, mid);
        let first_len = entry_redirect_len(at, mid);
        unsafe {
            std::ptr::copy_nonoverlapping(first.as_ptr(), at as *mut u8, first_len);
        }
        let second = encode_redirect_branch(mid, end);
        let second_len = entry_redirect_len(mid, end);
        unsafe {
            std::ptr::copy_nonoverlapping(second.as_ptr(), mid as *mut u8, second_len);
        }
        assert_eq!(follow_jump(at), end);
        assert_eq!(follow_jump(end), end);
    }

    #[test]
    fn entry_redirect_len_matches_jump_encoding() {
        let at = 0x1000usize;
        let near = 0x1100usize;
        #[cfg(target_arch = "x86_64")]
        assert_eq!(entry_redirect_len(at, near), 5);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(entry_redirect_len(at, near), 4);
        let far = at.wrapping_add(1usize << 40);
        #[cfg(target_arch = "x86_64")]
        assert_eq!(entry_redirect_len(at, far), 13);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(entry_redirect_len(at, far), STUB_SIZE);
    }

    #[test]
    fn near_redirect_encodes_a_direct_jump() {
        let at = 0x1000usize;
        let target = 0x1100usize;
        let bytes = encode_redirect_branch(at, target);
        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(bytes[0], 0xE9);
            let rel = i32::from_le_bytes(bytes[1..5].try_into().unwrap());
            assert_eq!(rel, (target as isize - (at as isize + 5)) as i32);
        }
        #[cfg(target_arch = "aarch64")]
        {
            let word = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
            assert_eq!(word & 0xFC00_0000, 0x1400_0000);
            let imm26 = word & 0x03FF_FFFF;
            let offset = ((imm26 as i32) << 6) >> 6 << 2;
            assert_eq!(offset as isize, target as isize - at as isize);
        }
    }

    #[test]
    fn patch_pending_failure_recoveries_does_not_stamp_until_commit() {
        let descr = majit_backend::make_resume_guard_descr_typed(Vec::new());
        let fd = descr.as_fail_descr().expect("resume guard is a FailDescr");
        fd.set_adr_jump_offset(0x1111);
        {
            let pending = vec![(descr.clone(), 0xDEAD_BEEFusize)];
            let _ = pending;
        }
        assert_eq!(fd.adr_jump_offset(), 0x1111);
        patch_pending_failure_recoveries(&[(descr.clone(), 0x2222)]);
        assert_eq!(fd.adr_jump_offset(), 0x2222);
    }
}
