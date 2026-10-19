/// rpython/jit/backend/x86/ parity: x86-specific backend.
///
/// RPython class hierarchy:
///   BaseAssembler (llsupport/assembler.py)
///     └── Assembler386 (x86/assembler.py)
pub mod arch;
pub mod assembler;
pub mod callbuilder;
pub mod cpu_ext;
pub mod regalloc;
pub mod reghint;
pub mod regloc;
pub mod rx86;
