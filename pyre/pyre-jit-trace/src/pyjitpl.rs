//! `semantic_fallthrough_pc` — the tracer's auto-advance PC rule.
//!
//! The trait meta-interpreter (`PyreMetaInterp`, RPython MetaInterp
//! pyjitpl.py) is retired; the FBW walker
//! (`jitcode_dispatch`) is the sole tracer.  Only the shared fallthrough-PC
//! rule the codewriter and runtime both key off survives in this module.

use pyre_interpreter::CodeObject;
use pyre_interpreter::bytecode::Instruction;

/// The PC the tracer auto-advances to after `pc` (skipping
/// `ExtendedArg` / `Resume` / `Nop` / `Cache` / `NotTaken`), i.e. the
/// value the tracer stores in `MIFrame::fallthrough_pc`.  `pub` so the
/// codewriter's splice resume-coverage gate can compute a can-raise op's
/// `after_residual_call` resume PC the same way the runtime does
/// (trace_opcode.rs `resume_pc = self.fallthrough_pc`); the sparse
/// resume resolver reuses it so the can-raise fallthrough resume marker
/// keys off the SAME pc the runtime records in the guard's resume data.
pub fn semantic_fallthrough_pc(code: &CodeObject, pc: usize) -> usize {
    let mut next_pc = pc.saturating_add(1);
    loop {
        match pyre_interpreter::decode_instruction_at(code, next_pc) {
            Some((
                Instruction::ExtendedArg
                | Instruction::Resume { .. }
                | Instruction::Nop
                | Instruction::Cache
                | Instruction::NotTaken,
                _,
            )) => next_pc += 1,
            _ => return next_pc,
        }
    }
}

/// Interpreter lasti after a Python-call jitcode op.
///
/// `semantic_fallthrough_pc` always advances one word first. When the
/// containing Python pc is already the next real opcode (the residual's
/// `py_exact` run was re-keyed onto the fallthrough), that extra step
/// skips the first instruction after the CALL. Cache words inside the
/// CALL still need a trivia skip; a CALL opcode still needs fallthrough.
pub fn resume_py_pc_after_python_call(code: &CodeObject, containing_py: usize) -> usize {
    match pyre_interpreter::decode_instruction_at(code, containing_py) {
        Some((
            Instruction::ExtendedArg
            | Instruction::Resume { .. }
            | Instruction::Nop
            | Instruction::Cache
            | Instruction::NotTaken,
            _,
        )) => crate::jitcode_dispatch::skip_python_trivia_forward(code, containing_py),
        Some((
            Instruction::Call { .. } | Instruction::CallKw { .. } | Instruction::CallFunctionEx,
            _,
        )) => semantic_fallthrough_pc(code, containing_py),
        _ => containing_py,
    }
}
