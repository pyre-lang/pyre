//! JitCode-PC-keyed Python coordinate helpers.
//!
//! These helpers read the codewriter-built twin tables materialized at
//! lowering time where the codewriter natively knows the Python PC:
//! `block_head_py_by_jit_pc` for exact block-head markers and
//! `py_floor_by_jit_pc` for containing floor segments. Together with the
//! forward-carried `SnapshotFrame.py_pc` resume channel, these tables are the
//! only Python-coordinate sources; runtime consumers never project a JitCode
//! PC through a Python-keyed map.

#[allow(dead_code)]
pub(crate) fn floor_boundary_at_or_after(
    metadata: &crate::PyJitCodeMetadata,
    jit_pc: usize,
) -> Option<(usize, u32)> {
    let table = &metadata.py_floor_by_jit_pc;
    let idx = table.partition_point(|&(off, _)| (off as usize) < jit_pc);
    table.get(idx).map(|&(off, py)| (off as usize, py))
}

pub(crate) fn first_floor_boundary_for_py(
    metadata: &crate::PyJitCodeMetadata,
    py_pc: u32,
) -> Option<(usize, u32)> {
    metadata
        .py_floor_by_jit_pc
        .iter()
        .find(|&&(_, py)| py == py_pc)
        .map(|&(off, py)| (off as usize, py))
}

/// Resolve a JitCode offset to the Python instruction coordinate containing it.
///
/// Exact block-head marker matches win first. Otherwise the floor segment is
/// the Python instruction whose lowering region contains the JitCode PC, the
/// coordinate the per-bytecode `last_instr` store in
/// `pyopcode.py dispatch_bytecode` would have left in the frame.
pub(crate) fn containing_py_pc_for_jitcode_pc(
    metadata: &crate::PyJitCodeMetadata,
    jit_pc: usize,
) -> u32 {
    if !metadata.py_floor_by_jit_pc.is_empty() {
        let pivot = metadata
            .block_head_py_by_jit_pc
            .binary_search_by_key(&jit_pc, |&(off, _)| off)
            .ok()
            .map(|i| metadata.block_head_py_by_jit_pc[i].1)
            .or_else(|| {
                crate::pyjitcode::floor_segment_for_jitcode_pc(&metadata.py_floor_by_jit_pc, jit_pc)
                    .map(|(_, py)| py)
            })
            .expect("drained JitCode PC floor pivot must begin at byte offset zero");
        return pivot;
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

/// `PYRE_M73_BACKXLAT_TWIN_AUDIT` asserts the codewriter-built forward py_pc
/// twin equals the [`trivia_normalized_py_pc_for_jitcode_pc`] containing lookup
/// at every non-negative resume word before [`resume_py_pc_for_jitcode_word`]
/// trusts the twin. Off in production; the gated assert is the only added work.
fn m73_backxlat_twin_audit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_M73_BACKXLAT_TWIN_AUDIT").is_some())
}

pub(crate) fn emptytwin_census_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_M73_EMPTYTWIN_CENSUS").is_some())
}

pub(crate) fn note_empty_twin_fallback(site: &'static str, jitcode_index: i32, jit_pc: i32) {
    if emptytwin_census_enabled() {
        eprintln!("[m73-emptytwin] site={site} jitcode={jitcode_index} pc={jit_pc}");
    }
}

/// Forward Python instruction coordinate for a carried resume JitCode word.
///
/// Reads the codewriter-built `forward_py_pc` twin first. Empty-twin JitCodes
/// and negative sentinel words resolve through the trivia-normalized containing
/// lookup.
///
/// A negative word (`NO_JITCODE_PC` `-1`) would cast to a huge `usize` twin
/// key and floor-resolve to a bogus predecessor entry, so it is delegated
/// straight to `trivia_normalized_py_pc_for_jitcode_pc`, which resolves it
/// identically to the pre-cutover call.
pub fn resume_py_pc_for_jitcode_word(jitcode_index: i32, pc_word: i32) -> i32 {
    if pc_word < 0 {
        return trivia_normalized_py_pc_for_jitcode_pc(jitcode_index, pc_word);
    }
    match crate::state::pyjitcode_for_jitcode_index(jitcode_index)
        .and_then(|pjc| pjc.forward_py_pc_for_jitcode_pc(pc_word as usize))
    {
        Some(twin) => {
            if m73_backxlat_twin_audit_enabled() {
                assert_eq!(
                    twin as i32,
                    trivia_normalized_py_pc_for_jitcode_pc(jitcode_index, pc_word),
                    "PYRE_M73_BACKXLAT_TWIN_AUDIT: forward py_pc twin diverged at jitcode {jitcode_index} pc {pc_word}"
                );
            }
            twin as i32
        }
        None => {
            note_empty_twin_fallback("resume_word", jitcode_index, pc_word);
            trivia_normalized_py_pc_for_jitcode_pc(jitcode_index, pc_word)
        }
    }
}
