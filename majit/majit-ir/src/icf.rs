//! Distinguisher so LLVM MergeFunctions cannot fold two residual-call
//! targets that would otherwise have identical bodies.
//!
//! RPython's genc emits one C function per graph (`FunctionCodeGenerator`)
//! and never merges two graphs; `pyjitpl.py` `MetaInterpStaticData.bytecode_for_address`
//! asserts the address is absent. `drain_list_append` keeps a forwarding call
//! for the same reason: one published address must name one function
//! (`registered_paths_sharing_an_address_are_alias_spellings`).
//!
//! `core::hint::black_box(ptr)` lowers to inline asm that may load through
//! the pointer and therefore clobbers all memory. `dont_look_inside` /
//! oopspec / call-surface bodies are not `#[inline(never)]`, so that
//! clobber was inlined into interpreter hot loops (dict ops, shadow-stack
//! push/pop).
//!
//! Native x86_64: `asm!` with `nomem, nostack, preserves_flags` whose
//! template is a multi-byte NOP whose 32-bit displacement is the low 32
//! bits of the unique `const` immediate (`0F 1F 80 <disp32>`). LLVM
//! MergeFunctions sees distinct IR; a folding linker (MSVC `/OPT:ICF`)
//! sees distinct machine bytes. Inlined copies still do not clobber
//! memory: a NOP's memory operand is not accessed. Other native targets
//! keep the comment-only template (distinct IR, zero instructions). The
//! token lives in this crate so Charon extracts a Call (the same shape as
//! `black_box`) rather than an `asm!` terminator the ULLBC reader maps to
//! `TermKind::Unknown`. wasm32 has no stable inline `asm!`; a volatile
//! read of the const local is a single-location load.
//!
//! The translator erases that Call (`Lowering::is_icf_identity_token`):
//! upstream `dont_look_inside` (`rpython/rlib/jit.py`) is metadata only, and
//! `Transformer.rewrite_op_debug_assert` (`jtransform.py`) returns `[]` for
//! an op with no effect on the traced program.

/// FNV-1a of the helper path, used as the unique `const` asm immediate.
pub const fn icf_path_hash(s: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
        i += 1;
    }
    hash
}

/// Keep this helper body distinct from every other published helper.
///
/// Native x86_64 emits a NOP carrying the immediate so a folding linker
/// (MSVC `/OPT:ICF`) also sees distinct bytes; other native targets keep
/// the comment form.
#[inline(always)]
pub fn icf_identity_token<const ID: u64>() {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: the template is a multi-byte NOP; the memory operand is
        // not accessed. nomem/nostack/preserves_flags, so the block neither
        // reads nor writes memory.
        unsafe {
            core::arch::asm!(
                "nop dword ptr [rax + {0}]",
                const (ID as u32) as i32,
                options(nomem, nostack, preserves_flags)
            );
        }
    }
    #[cfg(all(not(target_arch = "wasm32"), not(target_arch = "x86_64")))]
    {
        // SAFETY: the template is a comment; nomem/nostack/preserves_flags,
        // so the block neither reads nor writes memory.
        unsafe {
            core::arch::asm!(
                "/* {0} */",
                const ID,
                options(nomem, nostack, preserves_flags)
            );
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let id = ID;
        let _ = unsafe { core::ptr::read_volatile(&raw const id) };
    }
}

/// Per-instance [`icf_identity_token`] for a generic helper: the asm
/// comment names `icf_type_marker::<T>`'s symbol, so each
/// monomorphization carries a distinct operand.
#[inline(always)]
pub fn icf_identity_token_of<T: ?Sized>() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        // SAFETY: the template is a comment; nomem/nostack/preserves_flags,
        // so the block neither reads nor writes memory.
        unsafe {
            core::arch::asm!(
                "/* {0} */",
                sym icf_type_marker::<T>,
                options(nomem, nostack, preserves_flags)
            );
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let f: fn() = icf_type_marker::<T>;
        let _ = unsafe { core::ptr::read_volatile(&raw const f) };
    }
}

/// Marker whose symbol [`icf_identity_token_of`] names with `sym`.
/// The body names that same symbol so MergeFunctions cannot fold two
/// empty monomorphizations onto one address.
#[doc(hidden)]
#[inline(never)]
pub fn icf_type_marker<T: ?Sized>() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        // SAFETY: the template is a comment; nomem/nostack/preserves_flags,
        // so the block neither reads nor writes memory.
        unsafe {
            core::arch::asm!(
                "/* {0} */",
                sym icf_type_marker::<T>,
                options(nomem, nostack, preserves_flags)
            );
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let f: fn() = icf_type_marker::<T>;
        let _ = unsafe { core::ptr::read_volatile(&raw const f) };
    }
}

/// Expand to [`icf_identity_token`] so macro and hand-written sites share
/// one helper. `token` is a string literal or `concat!` of path pieces.
#[macro_export]
macro_rules! icf_identity {
    ($token:expr) => {
        $crate::icf_identity_token::<{ $crate::icf_path_hash($token) }>()
    };
}
