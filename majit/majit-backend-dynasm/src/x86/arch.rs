//! Port of `rpython/jit/backend/x86/arch.py`.
//!
//! Frame layout constants for the x86_64 backend.  pyre's
//! `dynasm/arch.rs` re-exports these so existing call sites continue
//! to use `crate::arch::*` while the per-arch source matches upstream.

/// arch.py:11
pub const WORD: usize = 8;

/// arch.py:13 — `IS_X86_64`.
pub const IS_X86_64: bool = true;

/// arch.py `WIN64 = sys.platform == "win32" and sys.maxint > 2**32`.
/// pyre is Rust-compiled per target, so the check maps to cfg.
#[cfg(target_os = "windows")]
pub const WIN64: bool = true;
#[cfg(not(target_os = "windows"))]
pub const WIN64: bool = false;

//        +--------------------+    <== aligned to 16 bytes
//        |   return address   |
//        +--------------------+           ----------------------.
//        |    saved regs      |                FRAME_FIXED_SIZE |
//        +--------------------+       --------------------.     |
//        |   scratch          |          PASS_ON_MY_FRAME |     |
//        |      space         |                           |     |
//        +--------------------+    <== aligned to 16 -----' ----'
//
// All the rest of the data is in a GC-managed variable-size jitframe.
// This jitframe object's address is always stored in the frame
// register.  The object layout itself lives in llsupport/jitframe.py;
// arch.py only defines the managed-register prefix inside `jf_frame`.

/// arch.py:42-48.
///   `rbp + rbx + r12 + r13 + r14 + r15 + threadlocal + 12 extra = 19`
/// on non-Win64, plus 4 for vmprof.
///   `rbp + rbx + rsi + rdi + r12 + 12 extra = 17` on Win64,
/// plus 4 for vmprof.
#[cfg(not(target_os = "windows"))]
pub const FRAME_FIXED_SIZE: usize = 19 + 4;
#[cfg(target_os = "windows")]
pub const FRAME_FIXED_SIZE: usize = 17 + 4;

/// arch.py:49.
pub const PASS_ON_MY_FRAME: usize = 12;

/// arch.py:50 — 13 GPR + 15 XMM.
pub const JITFRAME_FIXED_SIZE: usize = 28;

/// arch.py:51-58 — threadlocal_addr offset in frame.
///   non-Win64: `(FRAME_FIXED_SIZE - 1) * WORD`  (moved in from %esi)
///   Win64:     `(FRAME_FIXED_SIZE + 2) * WORD`  (shadow store slot)
#[cfg(not(target_os = "windows"))]
pub const THREADLOCAL_OFS: usize = (FRAME_FIXED_SIZE - 1) * WORD;
#[cfg(target_os = "windows")]
pub const THREADLOCAL_OFS: usize = (FRAME_FIXED_SIZE + 2) * WORD;

/// arch.py:59-60 — additional Win64 shadow-store slots.
#[cfg(target_os = "windows")]
pub const SHADOWSTORE2_OFS: usize = (FRAME_FIXED_SIZE + 3) * WORD;
#[cfg(target_os = "windows")]
pub const SHADOWSTORE3_OFS: usize = (FRAME_FIXED_SIZE + 4) * WORD;

/// arch.py `DEFAULT_FRAME_BYTES`: return address + `FRAME_FIXED_SIZE` words.
pub const DEFAULT_FRAME_BYTES: usize = (1 + FRAME_FIXED_SIZE) * WORD;

// `arch.py` `assert PASS_ON_MY_FRAME >= 12`. `FRAME_FIXED_SIZE` is odd, so
// `SUB rsp, FRAME_FIXED_SIZE * WORD` turns the 8-mod-16 entry rsp into a
// 0-mod-16 body. `(1 + FRAME_FIXED_SIZE) * WORD` stays 0 mod 16.
const _: () = assert!(PASS_ON_MY_FRAME >= 12);
const _: () = assert!(DEFAULT_FRAME_BYTES % 16 == 0);
const _: () = assert!((FRAME_FIXED_SIZE * WORD) % 16 == 8);
