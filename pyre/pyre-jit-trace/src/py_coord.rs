//! JitCode-PC-keyed Python coordinate helpers.
//!
//! These helpers read the codewriter-built twin tables materialized at
//! lowering time where the codewriter natively knows the Python PC:
//! `block_head_py_by_jit_pc` for exact block-head markers,
//! `py_exact_by_jit_pc` for the opcode that emitted each contiguous run, and
//! `py_floor_by_jit_pc` for first-offset-per-PC floor segments. Together with
//! the forward-carried `SnapshotFrame.py_pc` resume channel, these tables are
//! the only Python-coordinate sources; runtime consumers never project a
//! JitCode PC through a Python-keyed map.

/// Resolve a JitCode offset to the Python instruction coordinate containing it.
///
/// Exact block-head marker matches win first. Otherwise the per-emission-run
/// owner (`py_exact_by_jit_pc`) names the opcode that actually emitted those
/// bytes, including a block drained after the body. The first-offset-per-PC
/// floor is the last resort: it collapses such a late block onto whichever
/// PC last opened a segment. The coordinate is the one the per-bytecode
/// `last_instr` store in `pyopcode.py dispatch_bytecode` would have left in
/// the frame.
///
/// Dispatch-loop head bytes (`interp_jit.py PyFrame.dispatch`) sit before
/// any opcode case; the floor starts at `(0, 0)`, so those bytes resolve to
/// the entry green. A blackhole resumed at the merge point uses the live
/// `next_instr` register for the `switch` case.
pub(crate) fn containing_py_pc_for_jitcode_pc(
    metadata: &crate::PyJitCodeMetadata,
    jit_pc: usize,
) -> u32 {
    if let Ok(i) = metadata
        .block_head_py_by_jit_pc
        .binary_search_by_key(&jit_pc, |&(off, _)| off)
    {
        return metadata.block_head_py_by_jit_pc[i].1;
    }
    if !metadata.py_exact_by_jit_pc.is_empty() {
        if let Some(py) =
            crate::pyjitcode::exact_py_pc_for_jitcode_pc(&metadata.py_exact_by_jit_pc, jit_pc)
        {
            return py;
        }
    }
    if !metadata.py_floor_by_jit_pc.is_empty() {
        return crate::pyjitcode::floor_segment_for_jitcode_pc(
            &metadata.py_floor_by_jit_pc,
            jit_pc,
        )
        .map(|(_, py)| py)
        .expect("drained JitCode PC floor pivot must begin at byte offset zero");
    }
    0
}

/// Name the Python instruction a JitCode PC falls inside.
///
/// Every walk outcome that maps to `TraceAction::AbortPermanent` reports the
/// same `DispatchError` name, so the bytecode is the only thing that tells one
/// permanent decline from another. Diagnostics that have to attribute an
/// outcome to an opcode go through here rather than projecting the offset
/// themselves, so the census and the residual-decline probe cannot drift onto
/// different coordinate systems.
///
/// `None` for the name means the caller had no code object or the coordinate
/// is past the end of the bytecode; the Python PC is still returned.
pub(crate) fn py_op_name_for_jitcode_pc(
    metadata: &crate::PyJitCodeMetadata,
    code: Option<&pyre_interpreter::CodeObject>,
    jit_pc: usize,
) -> (usize, Option<String>) {
    let py_pc = containing_py_pc_for_jitcode_pc(metadata, jit_pc) as usize;
    let name = code
        .and_then(|code| pyre_interpreter::decode_instruction_at(code, py_pc))
        .map(|(instruction, _)| format!("{instruction:?}"));
    (py_pc, name)
}

/// Resolve `MetaInterpStaticData.jitcodes[jitcode_index]` to the containing
/// Python instruction coordinate for a stored JitCode offset.
pub fn containing_py_pc_for_jitcode_pc_public(jitcode_index: i32, offset: i32) -> Option<i32> {
    let payload = crate::state::pyjitcode_for_jitcode_index(jitcode_index)?;
    Some(containing_py_pc_for_jitcode_pc(&payload.metadata, offset as usize) as i32)
}

/// Return the per-emission-run Python owner of a stored JitCode offset.
///
/// Unlike the floor tier, this preserves a Python instruction that emits in
/// multiple disjoint JitCode regions. Empty skeleton metadata returns `None`.
pub fn exact_py_pc_for_jitcode_pc_public(jitcode_index: i32, offset: i32) -> Option<i32> {
    let payload = crate::state::pyjitcode_for_jitcode_index(jitcode_index)?;
    crate::pyjitcode::exact_py_pc_for_jitcode_pc(
        &payload.metadata.py_exact_by_jit_pc,
        offset as usize,
    )
    .map(|py| py as i32)
}

/// Advance a Python instruction coordinate past resume trivia when code is available.
pub fn skip_python_trivia_forward_public(
    jitcode_index: i32,
    raw_py_pc: i32,
) -> Option<(i32, bool)> {
    let payload = crate::state::pyjitcode_for_jitcode_index(jitcode_index)?;
    if payload.code_ptr.is_null() {
        return Some((raw_py_pc, false));
    }
    let code = unsafe { &*payload.code_ptr };
    Some((
        crate::jitcode_dispatch::skip_python_trivia_forward(code, raw_py_pc as usize) as i32,
        true,
    ))
}

/// Translate a resume-frame pc word to a trivia-normalized Python coordinate.
///
/// A negative word (`NO_JITCODE_PC`) has no Python coordinate; it
/// passes through so the caller's `pc < 0` screen rejects it (the internal
/// metadata lookups below are bounds-checked and never index with the word, so
/// no wrap results).
pub fn trivia_normalized_py_pc_for_jitcode_pc(jitcode_index: i32, pc_word: i32) -> i32 {
    let fallback = pc_word;
    match containing_py_pc_for_jitcode_pc_public(jitcode_index, pc_word)
        .and_then(|raw_py_pc| skip_python_trivia_forward_public(jitcode_index, raw_py_pc))
        .map(|(py_pc, _)| py_pc)
    {
        Some(py_pc) => py_pc,
        None => fallback,
    }
}

/// Jitcode offset and containing Python pc of the Python-call op that ends
/// at `pc` (`MIFrame.pc` immediately after the call).
fn python_call_ending_at(jitcode_index: i32, pc: usize) -> Option<(usize, usize)> {
    let pjc = crate::state::pyjitcode_for_jitcode_index(jitcode_index)?;
    if pjc.code_ptr.is_null() || crate::state::jitcode_is_transparent_helper(jitcode_index) {
        return None;
    }
    let call_pc = pjc.loop_scan().python_call_ending_at(pc)?;
    let call_py = containing_py_pc_for_jitcode_pc(&pjc.metadata, call_pc) as usize;
    Some((call_pc, call_py))
}

/// Python resume pc when `pc` is the byte immediately after a Python-call
/// jitcode op (`inline_call_return_marker` / trailing `-live-`).
///
/// The `forward_py_pc` twin names the CALL (or a cache word inside it).
/// Interpreter hand-off uses `next_instr`-style lasti, so the coordinate is
/// [`crate::pyjitpl::resume_py_pc_after_python_call`] of the CALL's
/// containing Python pc — cache words skip forward, a CALL opcode
/// fallthroughs, and a run already re-keyed onto the next opcode stays.
fn after_call_resume_py_pc(jitcode_index: i32, pc: usize) -> Option<u32> {
    let pjc = crate::state::pyjitcode_for_jitcode_index(jitcode_index)?;
    let (_, call_py) = python_call_ending_at(jitcode_index, pc)?;
    let code = unsafe { &*pjc.code_ptr };
    Some(crate::pyjitpl::resume_py_pc_after_python_call(code, call_py) as u32)
}

/// Containing Python pc of the CALL a resume word sits immediately after.
///
/// [`resume_py_pc_for_jitcode_word`] names the CALL's fallthrough for lasti.
/// `get_list_of_active_boxes` (`pyjitpl.py`) `in_a_call` still measures
/// operand-stack height at the CALL itself, so reconstruct uses this
/// coordinate for `stack_depth_at`.
pub(crate) fn call_site_py_pc_for_resume_word(jitcode_index: i32, pc_word: i32) -> Option<u32> {
    if pc_word < 0 {
        return None;
    }
    python_call_ending_at(jitcode_index, pc_word as usize).map(|(_, call_py)| call_py as u32)
}

/// Forward Python instruction coordinate for a carried resume JitCode word.
///
/// A word that is the byte after a Python-call jitcode op is the caller's
/// `in_a_call` / `after_residual_call` resume (`MIFrame.pc`). That maps to
/// the CALL's semantic fallthrough, not the containing CALL / cache word
/// the `forward_py_pc` twin reports.
///
/// Every other word reads the codewriter-built `forward_py_pc` twin first.
/// Empty-twin JitCodes and negative sentinel words resolve through the
/// trivia-normalized containing lookup.
///
/// A negative word (`NO_JITCODE_PC` `-1`) would cast to a huge `usize` twin
/// key and floor-resolve to a bogus predecessor entry, so it is delegated
/// straight to `trivia_normalized_py_pc_for_jitcode_pc`, which resolves it
/// identically to the pre-cutover call.
pub fn resume_py_pc_for_jitcode_word(jitcode_index: i32, pc_word: i32) -> i32 {
    if pc_word < 0 {
        return trivia_normalized_py_pc_for_jitcode_pc(jitcode_index, pc_word);
    }
    if let Some(ft) = after_call_resume_py_pc(jitcode_index, pc_word as usize) {
        return ft as i32;
    }
    match crate::state::pyjitcode_for_jitcode_index(jitcode_index)
        .and_then(|pjc| pjc.forward_py_pc_for_jitcode_pc(pc_word as usize))
    {
        Some(twin) => twin as i32,
        None => trivia_normalized_py_pc_for_jitcode_pc(jitcode_index, pc_word),
    }
}
