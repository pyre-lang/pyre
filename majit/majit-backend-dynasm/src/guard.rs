/// assembler.py ResumeGuardDescr parity: fail descriptor; the patchable
/// jump offset (`history.py _attrs_` `adr_jump_offset`) lives on the
/// metainterp `ResumeGuardDescr` (`majit-metainterp/src/compile.rs`) and
/// is accessed here via `meta_resume_fd()` forwarding.

// Descr-by-address recovery is now a pure `Arc::from_raw` against the
// `FailDescrCell` wrapper baked at codegen time —
// `majit_ir::recover_fail_descr_cell(addr)` in `majit-ir/src/descr.rs`.
// `history.py AbstractDescr.show(cpu, descr_gcref) =
// cast_gcref_to_instance(...)` parity.  The strong refcount is held by
// `CompiledLoopToken.asmmemmgr_gcreftracers` (`model.py`).

// RECOVERY_LAYOUT_TABLE removed: `recovery_layout` is not in
// `AbstractFailDescr._attrs_` (`history.py`).  Resume code
// (`resume.py`) decodes recovery on demand from
// `rd_numb / rd_consts / rd_virtuals / rd_pendingfields` via
// `ResumeGuardDescr.get_resumestorage`.  At deopt,
// `FailDescrLayout.recovery_layout` is `None`.

// FAIL_ARG_LOCS_TABLE removed: duplicates `rd_locs`
// (`history.py:132 _attrs_`).  All readers (`lib.rs:handle_fail_*` and
// `runner.rs::execute_token`) now decode `descr.rd_locs()` directly
// via `decode_rd_loc_slot`, matching PyPy's
// `llmodel.py _decode_pos`.

/// Re-export the shared per-cpu descr attachment types so existing
/// `crate::guard::{AttachedDescrPtrs, CpuDescrAttachments, CpuDescrHandle}`
/// imports keep resolving while the canonical definitions live in
/// `majit-backend` (shared with cranelift).
///
/// `rpython/jit/backend/model.py AbstractCPU` — descr attachments are a
/// cross-backend base-class concern; `compile.py:665-674
/// make_and_attach_done_descrs` binds them on each `cpu` instance
/// regardless of backend.
pub use majit_backend::{AttachedDescrPtrs, CpuDescrAttachments, CpuDescrCell, CpuDescrHandle};

/// Re-export the slot decoder for the same reason: `_decode_pos` is a method
/// on `AbstractLLCPU` (`llmodel.py`), not a per-arch entry, so its
/// definition sits in `majit-backend`'s `llmodel` beside the direct readers it
/// feeds, and `crate::guard::decode_rd_loc_slot` keeps resolving here.
pub use majit_backend::llmodel::decode_rd_loc_slot;

// `DynasmFailDescr` removed.  PyPy's dynasm-equivalent
// `assembler.py` carries no per-emission descr wrapper — every guard
// op's `op.descr` is the same `ResumeGuardDescr` Arc that the metainterp
// stamps, and every FINISH emission writes the cpu-attached singleton
// (`compile.py make_and_attach_done_descrs`) directly into
// `jf_descr`.  After the Unified-Descr port the backend no longer
// owned any field worth wrapping, and the singleton-direct push
// eliminated the last constructor of the wrapper.  All readers reach
// the descr through `DescrRef` and dispatch trait methods on
// `&dyn FailDescr`.

/// `compile.py:849` build a FailDescrLayout snapshot for any FailDescr
/// descr identity (post Phase C-1 cascade — `find_descr_by_ptr` returns
/// `DescrRef`, so the layout extractor takes the trait object plus the
/// data-pointer address used to key backend-static side-tables
/// (`SOURCE_OP_INDEX_TABLE`, `RECOVERY_LAYOUT_TABLE`)).
///
/// `fail_index` / `trace_id` are explicit caller arguments rather than
/// reads off `fd.fail_index_per_trace()` / `fd.trace_id()` so the
/// `fail_descrs` Vec position can be the identity source.  Singleton
/// FINISH descrs (`compile.py _DoneWithThisFrameDescr` /
/// `ExitFrameWithExceptionDescrRef`) share an Arc across emissions and
/// answer the trait-default `0` for both methods; the layout pipeline
/// must read position from the Vec, not from the descr.
pub fn layout_for_fail_descr(
    descr: &majit_ir::DescrRef,
    fail_index: u32,
    trace_id: u64,
) -> majit_backend::FailDescrLayout {
    let fd = descr
        .as_fail_descr()
        .expect("layout_for_fail_descr requires a Descr that exposes the FailDescr trait");
    let fail_arg_types = fd.fail_arg_types();
    majit_backend::FailDescrLayout {
        fail_index,
        fail_arg_types: fail_arg_types.to_vec(),
        is_finish: fd.is_finish(),
        is_exception_exit: fd.is_exit_frame_with_exception(),
        trace_id,
        source_op_index: fd.source_op_index(),
        frame_stack: None,
        // Backend does not cache recovery_layout. Production deopt
        // decodes the descr's `rd_numb` (`ResumeGuardDescr.get_resumestorage`).
        recovery_layout: None,
        trace_info: None,
        descr: Some(descr.clone()),
    }
}

// `Debug`, `impl Descr`, `impl FailDescr` for the removed
// `DynasmFailDescr` all went away with the struct.  The metainterp
// `AbstractFailDescr` Arc carries the canonical `FailDescr` impl now
// reached through `op.descr` and `DescrRef::as_fail_descr`.

/// `BaseAssembler.store_info_on_descr`.
///
/// Encode `fail_locs` once. When every position equals
/// `_previous_rd_locs`, return that handle instead of a new allocation.
pub(crate) fn store_info_on_descr(
    previous: &mut majit_ir::RdLocs,
    faillocs: &[Option<crate::regloc::Loc>],
    slot_for_loc: fn(&crate::regloc::Loc) -> Option<u16>,
) -> majit_ir::RdLocs {
    let mut shared = previous.len() == faillocs.len();
    let mut positions: Option<Vec<u16>> = if shared {
        None
    } else {
        Some(vec![0; faillocs.len()])
    };
    for (i, fl) in faillocs.iter().enumerate() {
        let position = match fl {
            None => 0xFFFF,
            Some(loc) => slot_for_loc(loc).unwrap_or_else(|| {
                panic!("store_info_on_descr: cannot encode fail-arg loc {loc:?} (gen_regs.index)")
            }),
        };
        if shared {
            if previous[i] == position {
                continue;
            }
            let mut owned = previous.as_slice().to_vec();
            owned[i] = position;
            positions = Some(owned);
            shared = false;
            continue;
        }
        positions.as_mut().expect("rd_locs buffer")[i] = position;
    }
    if shared {
        return previous.clone();
    }
    let locs = majit_ir::RdLocs::from_vec(positions.expect("rd_locs buffer"));
    *previous = locs.clone();
    locs
}
