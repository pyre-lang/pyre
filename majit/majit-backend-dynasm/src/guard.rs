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
// `ResumeGuardDescr.get_resumestorage`.

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
    let mut positions = smallvec::SmallVec::<[u16; 32]>::with_capacity(faillocs.len());
    for fl in faillocs {
        let position = match fl {
            None => 0xFFFF,
            Some(loc) => slot_for_loc(loc).unwrap_or_else(|| {
                panic!("store_info_on_descr: cannot encode fail-arg loc {loc:?} (gen_regs.index)")
            }),
        };
        positions.push(position);
    }
    if previous.as_slice() == positions.as_slice() {
        return previous.clone();
    }
    let locs = majit_ir::RdLocs::from_exact_iter(positions.iter().copied());
    *previous = locs.clone();
    locs
}

#[cfg(test)]
mod store_info_on_descr_tests {
    use super::store_info_on_descr;
    use crate::regloc::{Loc, RegLoc};

    fn slot_reg(loc: &Loc) -> Option<u16> {
        match loc {
            Loc::Reg(r) => Some(r.value as u16),
            _ => None,
        }
    }

    #[test]
    fn equal_consecutive_guards_share_allocation() {
        let mut previous = majit_ir::RdLocs::new();
        let locs = [
            Some(Loc::Reg(RegLoc::new(1, false))),
            None,
            Some(Loc::Reg(RegLoc::new(3, false))),
        ];
        let first = store_info_on_descr(&mut previous, &locs, slot_reg);
        let second = store_info_on_descr(&mut previous, &locs, slot_reg);
        assert!(majit_ir::RdLocs::ptr_eq(&first, &second));
        assert_eq!(first.as_slice(), &[1, 0xFFFF, 3]);
    }

    #[test]
    fn empty_and_mismatch_allocate() {
        let mut previous = majit_ir::RdLocs::new();
        let empty = store_info_on_descr(&mut previous, &[], slot_reg);
        assert!(empty.is_empty());
        assert!(majit_ir::RdLocs::ptr_eq(&empty, &previous));

        let first = [Some(Loc::Reg(RegLoc::new(1, false)))];
        let second = [Some(Loc::Reg(RegLoc::new(2, false)))];
        let a = store_info_on_descr(&mut previous, &first, slot_reg);
        let b = store_info_on_descr(&mut previous, &second, slot_reg);
        assert!(!majit_ir::RdLocs::ptr_eq(&a, &b));
        assert_eq!(a.as_slice(), &[1]);
        assert_eq!(b.as_slice(), &[2]);
    }

    #[test]
    fn large_positions_round_trip() {
        let mut previous = majit_ir::RdLocs::new();
        let locs: Vec<Option<Loc>> = (0..40)
            .map(|i| {
                if i == 7 {
                    None
                } else {
                    Some(Loc::Reg(RegLoc::new(i as u8, false)))
                }
            })
            .collect();
        let encoded = store_info_on_descr(&mut previous, &locs, slot_reg);
        assert_eq!(encoded.len(), 40);
        assert_eq!(encoded[7], 0xFFFF);
        assert_eq!(encoded[39], 39);
        let again = store_info_on_descr(&mut previous, &locs, slot_reg);
        assert!(majit_ir::RdLocs::ptr_eq(&encoded, &again));
    }
}
