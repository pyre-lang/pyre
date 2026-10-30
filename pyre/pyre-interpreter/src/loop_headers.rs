//! Bytecode loop-header analysis and the portal-nameable predicate.
//!
//! Upstream fixes loop headers while the codewriter builds the graph's
//! JitCode (`rpython/jit/codewriter/jtransform.py handle_jit_marker__loop_header`); here the
//! scan is interpreter-side bytecode analysis so the accessor is nameable
//! by the fnaddr table.

use std::collections::HashSet;
use std::sync::atomic::Ordering;

use rustpython_compiler_core::bytecode::{Instruction, OpArg, OpArgState};

use crate::CodeObject;
use crate::pyopcode::{jump_target_forward, skip_caches};

/// Single-source-of-truth backward-jump target calculation used by both
/// the loop-header pre-scan (`pypy/module/pypyjit/interp_jit.py jump_absolute`) and
/// the emitter (`jtransform.py` `handle_jit_marker__loop_header`).
///
/// Returns the target PC for `JumpBackward` (with `skip_caches` on the
/// next-PC base) and `JumpBackwardNoInterrupt` (direct `py_pc + 1 - delta`
/// arithmetic to match the interpreter's dispatch in pyopcode.rs).
/// Returns `None` for any non-backward-jump opcode.
pub fn backward_jump_target(
    code: &CodeObject,
    py_pc: usize,
    instr: Instruction,
    op_arg: OpArg,
) -> Option<usize> {
    match instr {
        Instruction::JumpBackward { delta } => Some(
            skip_caches(&code.instructions, py_pc + 1).saturating_sub(delta.get(op_arg).as_usize()),
        ),
        Instruction::JumpBackwardNoInterrupt { delta } => {
            Some((py_pc + 1).saturating_sub(delta.get(op_arg).as_usize()))
        }
        _ => None,
    }
}

/// Control-flow successors of every code-unit index: fall-through (except
/// after an unconditional transfer), forward and backward jump targets, and
/// the exception edge from each protected pc to its handler landing.  Built in
/// one sequential decode pass so an `EXTENDED_ARG` prefix folds into the
/// following opcode's argument, matching the edge model of
/// `find_branch_target_pcs`.
pub fn code_successors(code: &CodeObject) -> Vec<Vec<usize>> {
    let num_instrs = code.instructions.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); num_instrs];
    let mut scan_state = OpArgState::default();
    for pc in 0..num_instrs {
        let (instr, op_arg) = scan_state.get(code.instructions[pc]);
        if let Some(target) = backward_jump_target(code, pc, instr, op_arg)
            && target < num_instrs
        {
            succ[pc].push(target);
        }
        let forward_delta = match instr {
            Instruction::PopJumpIfFalse { delta }
            | Instruction::PopJumpIfTrue { delta }
            | Instruction::PopJumpIfNone { delta }
            | Instruction::PopJumpIfNotNone { delta }
            | Instruction::JumpForward { delta }
            | Instruction::ForIter { delta } => Some(delta.get(op_arg).as_usize()),
            _ => None,
        };
        if let Some(delta) = forward_delta {
            let target = jump_target_forward(&code.instructions, pc + 1, delta);
            if target < num_instrs {
                succ[pc].push(target);
            }
        }
        let terminates = matches!(
            instr,
            Instruction::JumpForward { .. }
                | Instruction::JumpBackward { .. }
                | Instruction::JumpBackwardNoInterrupt { .. }
                | Instruction::ReturnValue
                | Instruction::RaiseVarargs { .. }
                | Instruction::Reraise { .. }
        );
        if !terminates {
            let fallthrough = pc + 1;
            if fallthrough < num_instrs {
                succ[pc].push(fallthrough);
            }
        }
    }
    for entry in crate::pycode::decode_exceptiontable(&code.exceptiontable) {
        let start = entry.start as usize / 2;
        let end = (entry.end as usize / 2).min(num_instrs);
        let handler = entry.target as usize / 2;
        if handler < num_instrs {
            for pc in start..end {
                succ[pc].push(handler);
            }
        }
    }
    succ
}

/// True when `target_pc` dominates `source_pc`: every control-flow path from
/// the code entry (pc 0) to `source_pc` passes through `target_pc`.  Answered
/// by a forward reachability from the entry that never enters `target_pc` — if
/// `source_pc` stays unreachable, `target_pc` dominates it.
pub fn target_dominates(succ: &[Vec<usize>], target_pc: usize, source_pc: usize) -> bool {
    let num_instrs = succ.len();
    if target_pc == source_pc || target_pc == 0 {
        return true;
    }
    if source_pc >= num_instrs {
        return false;
    }
    let mut seen = vec![false; num_instrs];
    let mut stack = vec![0usize];
    seen[0] = true;
    while let Some(pc) = stack.pop() {
        for &next in &succ[pc] {
            if next == target_pc || seen[next] {
                continue;
            }
            seen[next] = true;
            stack.push(next);
        }
    }
    !seen[source_pc]
}

/// True when a backward jump returns from an exception handler to earlier
/// code rather than closing a loop.
///
/// `pypy/module/pypyjit/interp_jit.py` `PyFrame.jump_absolute` calls
/// `can_enter_jit` on every backward `JUMP_ABSOLUTE`. Remaining-sibling
/// after `except` is `JUMP_FORWARD` on PyPy (`pyopcode.py` `JUMP_FORWARD`)
/// and never reaches `jump_absolute`. 3.15 places the handler at the end
/// of the bytecode, so that rejoin is `JUMP_BACKWARD_NO_INTERRUPT` — a
/// backward offset whose target does not dominate the jump (the handler
/// is reached through the exception edge). SEND/YIELD await loops use
/// the same opcode with a dominating target and still tick.
/// `find_loop_header_pcs` already uses this domination test;
/// `jump_absolute` on a true loop back-edge (`JUMP_BACKWARD`) still
/// always ticks.
///
/// Residual like `code_pc_is_loop_header`: the CFG walk is a predicate the
/// fnaddr table can name. Leaving it open pulled the no-interrupt arm of
/// `execute_jump_backward` into the two-phase Skip set.
#[majit_macros::dont_look_inside]
pub fn backward_jump_is_handler_only_target(
    code: &CodeObject,
    source_pc: usize,
    target_pc: usize,
) -> bool {
    if let Some(info) = handler_only_info_for(code) {
        return info
            .handler_only_edges
            .binary_search(&(source_pc, target_pc))
            .is_ok();
    }
    !target_dominates(&code_successors(code), target_pc, source_pc)
}

fn handler_only_info_for(code: &CodeObject) -> Option<&'static LoopHeaderInfo> {
    let w_code = crate::pycode::live_code_wrapper(code as *const CodeObject as *const ());
    let info = ensure_loop_header_info(w_code)?;
    Some(unsafe { &*info })
}

/// Scan `code` for JUMP_BACKWARD targets — the PCs where
/// `transform_graph_to_jitcode` would emit `BC_JUMP_TARGET` and where
/// `jit_merge_point` is evaluated.
///
/// The flow-graph scan is `warmspot.py` `_find_jit_marker`, via
/// `find_can_enter_jit` and `find_loop_headers`
/// (`WarmRunnerDesc.rewrite_can_enter_jits`).
/// `jtransform.py` `Transformer.handle_jit_marker__loop_header` only
/// rewrites one already-present marker into a `loop_header` op. Bytecode
/// carries no `jit_marker`, so this walk resolves `JUMP_BACKWARD` targets.
pub fn find_loop_header_pcs(code: &CodeObject) -> HashSet<usize> {
    let num_instrs = code.instructions.len();
    let succ = code_successors(code);
    let mut loop_header_pcs: HashSet<usize> = HashSet::new();
    let mut scan_state = OpArgState::default();
    for scan_pc in 0..num_instrs {
        let (scan_instr, scan_arg) = scan_state.get(code.instructions[scan_pc]);
        if let Some(target) = backward_jump_target(code, scan_pc, scan_instr, scan_arg)
            && target < num_instrs
            && target_dominates(&succ, target, scan_pc)
        {
            loop_header_pcs.insert(target);
        }
    }
    loop_header_pcs
}

/// Per-code loop-header scan, owned by the `PyCode` wrapper.
///
/// `header_pcs` is `find_loop_header_pcs` sorted for binary search.
/// `handler_only_edges` is each backward jump `(source_pc, target_pc)` whose
/// target does not dominate the jump — remaining-sibling
/// `JUMP_BACKWARD_NO_INTERRUPT` after `except` (`pyopcode.py` `JUMP_FORWARD`
/// on PyPy). Sorted for binary search from `execute_jump_backward`.
pub struct LoopHeaderInfo {
    pub header_pcs: Vec<usize>,
    pub handler_only_edges: Vec<(usize, usize)>,
}

/// Host `LoopHeaderInfo` is a `Box::into_raw`'d host allocation, never a GC object.
#[doc(hidden)]
#[allow(non_upper_case_globals, dead_code)]
pub const _alloc_flavor_LoopHeaderInfo: &str = "raw";

/// Publish `LoopHeaderInfo` on `w_code` at the first query.
///
/// A null wrapper, or a null/unaligned `code_ptr` (the case in which
/// `w_code_new_owned` leaves `globals_caches` null), builds nothing and
/// returns `None`. The loser of `compare_exchange` drops its box.
fn ensure_loop_header_info(w_code: pyre_object::PyObjectRef) -> Option<*const LoopHeaderInfo> {
    if w_code.is_null() {
        return None;
    }
    let pycode = unsafe { &*(w_code as *const crate::pycode::PyCode) };
    let code_ptr = pycode.code_ptr;
    let align_mask = std::mem::align_of::<CodeObject>() as i64 - 1;
    if code_ptr.is_null() || (code_ptr as i64) & align_mask != 0 {
        return None;
    }
    let existing = pycode.loop_header_info.load(Ordering::Acquire);
    if !existing.is_null() {
        return Some(existing);
    }
    let code = unsafe { &*(code_ptr as *const CodeObject) };
    let num_instrs = code.instructions.len();
    let succ = code_successors(code);
    let mut header_pcs: Vec<usize> = Vec::new();
    let mut handler_only_edges: Vec<(usize, usize)> = Vec::new();
    let mut scan_state = OpArgState::default();
    for scan_pc in 0..num_instrs {
        let (scan_instr, scan_arg) = scan_state.get(code.instructions[scan_pc]);
        if let Some(target) = backward_jump_target(code, scan_pc, scan_instr, scan_arg)
            && target < num_instrs
        {
            if target_dominates(&succ, target, scan_pc) {
                header_pcs.push(target);
            } else {
                handler_only_edges.push((scan_pc, target));
            }
        }
    }
    header_pcs.sort_unstable();
    header_pcs.dedup();
    handler_only_edges.sort_unstable();
    let built = Box::into_raw(Box::new(LoopHeaderInfo {
        header_pcs,
        handler_only_edges,
    }));
    match pycode.loop_header_info.compare_exchange(
        std::ptr::null_mut(),
        built,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => Some(built),
        Err(winner) => {
            drop(unsafe { Box::from_raw(built) });
            Some(winner)
        }
    }
}

/// Two-word predicate for the portal arm: is `pc` a loop header of `w_code`?
///
/// Collapses the per-code header scan into one residual the fnaddr table can
/// name. The sorted set lives on `PyCode.loop_header_info`. A null wrapper,
/// or a null or unaligned `code_ptr`, answers false.
#[majit_macros::dont_look_inside]
pub fn code_pc_is_loop_header(w_code: pyre_object::PyObjectRef, pc: usize) -> bool {
    let Some(info) = ensure_loop_header_info(w_code) else {
        return false;
    };
    let info = unsafe { &*info };
    info.header_pcs.binary_search(&pc).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_exec;
    use rustpython_compiler_core::bytecode::ConstantData;

    fn first_function_code(source: &str) -> CodeObject {
        let module = compile_exec(source).expect("test source should compile");
        module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } => Some((**code).clone()),
                _ => None,
            })
            .expect("expected nested function code object")
    }

    fn wrapper_for(code: &CodeObject) -> pyre_object::PyObjectRef {
        crate::box_code_object(code.clone())
    }

    fn dominating_backedge_targets(code: &CodeObject) -> Vec<usize> {
        let succ = code_successors(code);
        let mut targets = Vec::new();
        let mut scan_state = OpArgState::default();
        for pc in 0..code.instructions.len() {
            let (instr, op_arg) = scan_state.get(code.instructions[pc]);
            if let Some(target) = backward_jump_target(code, pc, instr, op_arg) {
                if target < code.instructions.len() && target_dominates(&succ, target, pc) {
                    targets.push(target);
                }
            }
        }
        targets.sort_unstable();
        targets.dedup();
        targets
    }

    #[test]
    fn nested_loops_have_two_header_pcs() {
        let code = first_function_code(
            "def f(n):\n    s = 0\n    for i in range(n):\n        for j in range(n):\n            s += 1\n    return s\n",
        );
        let w_code = wrapper_for(&code);
        let headers = find_loop_header_pcs(&code);
        let expected = dominating_backedge_targets(&code);
        assert_eq!(headers.len(), 2);
        assert_eq!(expected.len(), 2);
        for &pc in &expected {
            assert!(headers.contains(&pc), "missing header {pc}");
            assert!(code_pc_is_loop_header(w_code, pc));
        }
        for pc in 0..code.instructions.len() {
            assert_eq!(code_pc_is_loop_header(w_code, pc), headers.contains(&pc));
        }
    }

    #[test]
    fn while_with_continue_has_one_header_pc() {
        let code = first_function_code(
            "def f(n):\n    s = 0\n    i = 0\n    while i < n:\n        i += 1\n        if i % 2 == 0:\n            continue\n        s += i\n    return s\n",
        );
        let w_code = wrapper_for(&code);
        let headers = find_loop_header_pcs(&code);
        let expected = dominating_backedge_targets(&code);
        assert_eq!(headers.len(), 1);
        assert_eq!(expected.len(), 1);
        let header = expected[0];
        assert!(headers.contains(&header));
        assert!(code_pc_is_loop_header(w_code, header));
        let mut continue_targets = Vec::new();
        let mut scan_state = OpArgState::default();
        for pc in 0..code.instructions.len() {
            let (instr, op_arg) = scan_state.get(code.instructions[pc]);
            if let Some(target) = backward_jump_target(&code, pc, instr, op_arg) {
                continue_targets.push(target);
            }
        }
        assert!(
            continue_targets.iter().any(|&t| t == header),
            "continue must land on the while header"
        );
        for &target in &continue_targets {
            assert_eq!(code_pc_is_loop_header(w_code, target), target == header);
        }
    }

    #[test]
    fn for_loop_with_try_except_keeps_its_header() {
        let code = first_function_code(
            "def run(n):\n    acc = 0\n    for i in range(n):\n        try:\n            raise ValueError\n        except ValueError:\n            acc += 1\n    return acc\n",
        );
        let w_code = wrapper_for(&code);
        let headers = find_loop_header_pcs(&code);
        let expected = dominating_backedge_targets(&code);
        assert_eq!(expected.len(), 1);
        assert_eq!(headers.len(), 1);
        assert!(headers.contains(&expected[0]));
        assert!(code_pc_is_loop_header(w_code, expected[0]));

        let succ = code_successors(&code);
        let mut handler_only = Vec::new();
        let mut scan_state = OpArgState::default();
        for pc in 0..code.instructions.len() {
            let (instr, op_arg) = scan_state.get(code.instructions[pc]);
            if let Some(target) = backward_jump_target(&code, pc, instr, op_arg) {
                if !target_dominates(&succ, target, pc) {
                    handler_only.push(target);
                }
            }
        }
        for target in handler_only {
            assert!(
                !code_pc_is_loop_header(w_code, target),
                "handler-rejoin target {target} must not be a loop header"
            );
        }
    }

    #[test]
    fn remaining_sibling_no_interrupt_is_handler_only() {
        // `named_reraise_sibling_hot` `run`: after `except ValueError`
        // the remaining-sibling is `JUMP_BACKWARD_NO_INTERRUPT` to the
        // next try, not the `FOR_ITER` header. PyPy encodes that rejoin
        // as `JUMP_FORWARD`. The last except in a for-loop body uses
        // `JUMP_BACKWARD` to the header and still ticks.
        let code = first_function_code(
            "def run(n):\n    total = 0\n    for i in range(n):\n        try:\n            total += 1\n        except ValueError:\n            total += 2\n        try:\n            total += 3\n        except ValueError:\n            total += 7\n    return total\n",
        );
        let w_code = wrapper_for(&code);
        let wrapped = unsafe { &*(crate::pycode::w_code_get_ptr(w_code) as *const CodeObject) };
        let succ = code_successors(&code);
        let mut no_interrupt_handler_only = 0;
        let mut ordinary_backedges = 0;
        let mut scan_state = OpArgState::default();
        for pc in 0..code.instructions.len() {
            let (instr, op_arg) = scan_state.get(code.instructions[pc]);
            let Some(target) = backward_jump_target(&code, pc, instr, op_arg) else {
                continue;
            };
            match instr {
                Instruction::JumpBackwardNoInterrupt { .. } => {
                    assert!(
                        backward_jump_is_handler_only_target(&code, pc, target),
                        "remaining-sibling {pc}->{target} must be handler-only"
                    );
                    assert_eq!(
                        backward_jump_is_handler_only_target(wrapped, pc, target),
                        backward_jump_is_handler_only_target(&code, pc, target),
                    );
                    assert!(
                        !target_dominates(&succ, target, pc),
                        "remaining-sibling target {target} must not dominate {pc}"
                    );
                    no_interrupt_handler_only += 1;
                }
                Instruction::JumpBackward { .. } => {
                    assert!(
                        !backward_jump_is_handler_only_target(&code, pc, target),
                        "for-loop back-edge {pc}->{target} must dominate"
                    );
                    assert_eq!(
                        backward_jump_is_handler_only_target(wrapped, pc, target),
                        backward_jump_is_handler_only_target(&code, pc, target),
                    );
                    ordinary_backedges += 1;
                }
                _ => {}
            }
        }
        assert!(
            no_interrupt_handler_only >= 1,
            "expected a remaining-sibling JUMP_BACKWARD_NO_INTERRUPT"
        );
        // Success-path JUMP_BACKWARD to FOR_ITER plus the last except's
        // JUMP_BACKWARD to the same header; both dominate and tick.
        assert!(ordinary_backedges >= 1);
        assert_eq!(find_loop_header_pcs(&code).len(), 1);
    }

    #[test]
    fn while_except_no_interrupt_rejoins_after_the_handler() {
        // Remaining-sibling after `except` in a `while` jumps to the
        // statement after the try (`i += 1`), not the while header.
        let code = first_function_code(
            "def f(n):\n    acc = 0\n    i = 0\n    while i < n:\n        try:\n            acc += 1\n        except ValueError:\n            acc += 2\n        i += 1\n    return acc\n",
        );
        let mut scan_state = OpArgState::default();
        let mut saw_handler_only = false;
        let mut saw_while_backedge = false;
        for pc in 0..code.instructions.len() {
            let (instr, op_arg) = scan_state.get(code.instructions[pc]);
            let Some(target) = backward_jump_target(&code, pc, instr, op_arg) else {
                continue;
            };
            match instr {
                Instruction::JumpBackwardNoInterrupt { .. } => {
                    assert!(backward_jump_is_handler_only_target(&code, pc, target));
                    saw_handler_only = true;
                }
                Instruction::JumpBackward { .. } => {
                    assert!(!backward_jump_is_handler_only_target(&code, pc, target));
                    saw_while_backedge = true;
                }
                _ => {}
            }
        }
        assert!(saw_handler_only);
        assert!(saw_while_backedge);
        assert_eq!(find_loop_header_pcs(&code).len(), 1);
    }
}
