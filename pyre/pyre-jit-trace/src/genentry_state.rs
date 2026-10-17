//! jd2 (`generatorentry_driver`) symbolic + `JitState` scaffolding.
//!
//! `generator.py` `generatorentry_driver = jit.JitDriver(greens=['pycode'],
//! reds=['gen', 'w_arg'], get_printable_location=..., name='generatorentry')`
//! drives `send_ex` for a generator body `should_not_inline` refuses to
//! inline (two or more yields). `split_portal` copies the loop to
//! `generatorentry_portal_portal` and leaves the original as the runner
//! stub, so the registered main jitcode is that copy.
//!
//! * [`GenEntrySym`] — the merge-point banks of the portal: `gen`, `w_arg`.
//! * [`GenEntryJitState`] — `Meta = PyreMeta`, `Sym = GenEntrySym`, `Env = PyreEnv`.

use majit_ir::{GcRef, GreenKey, GreenType, OpRef, Type, Value};
use majit_metainterp::{JitCodeSym, JitDriverStaticData, JitState};

use crate::state::{PyreEnv, PyreMeta};
use pyre_object::{PY_NULL, PyObjectRef};

/// Registered main-jitcode key after `split_portal` copies the loop.
pub const GENENTRY_PORTAL_KEY: &str = "baseobjspace::generatorentry_portal_portal";

/// jd2 symbolic state at the `generatorentry_driver` merge point.
///
/// `generator.py` names the reds `gen` and `w_arg`; both are object
/// references, so the extracted banks are two ref registers.
#[allow(dead_code)]
pub struct GenEntrySym {
    /// `jit_merge_point(pycode=...)` green.
    pub pycode: PyObjectRef,
    /// The generator being resumed (`gen=self`).
    pub r#gen: OpRef,
    /// The value sent in (`w_arg`; `w_None` for `next`).
    pub w_arg: OpRef,
}

impl JitCodeSym for GenEntrySym {
    fn total_slots(&self) -> usize {
        0
    }

    fn loop_header_pc(&self) -> usize {
        // The marker's byte offset shifts with the extracted portal's op
        // layout, so discover it. The portal carries exactly one merge point
        // (`send_ex` has no `can_enter_jit`).
        let canonical = crate::jitcode_runtime::portal_jitcode_for_key(GENENTRY_PORTAL_KEY)
            .expect("jd2 portal jitcode must be registered");
        crate::jitcode_runtime::decoded_ops(&canonical.code)
            .find(|op| op.opname == "jit_merge_point")
            .expect("jd2 portal body must contain a jit_merge_point")
            .pc
    }
}

/// jd2 `JitState`: reuses [`PyreMeta`] and the empty [`PyreEnv`], driving
/// [`GenEntrySym`]. Novable: the generator frame is reached through `gen`,
/// not carried as a virtualizable of this driver.
pub struct GenEntryJitState {
    /// The merge-point green pinned for this driver activation.
    pub pycode: PyObjectRef,
}

/// `jitdrivers_sd` index of `generatorentry_driver`. Part of the cell
/// identity: jd0's cell for `(next_instr=0, is_being_profiled=false, pycode)`
/// is a different `JitCell` (`warmstate.py`).
pub const GENERATORENTRY_JD_INDEX: i64 = 2;

/// jd2 greens `(pycode, driver index)` as a `GreenKey`.
pub fn genentry_green_key(pycode: PyObjectRef) -> GreenKey {
    GreenKey::with_types(
        vec![pycode as i64, GENERATORENTRY_JD_INDEX],
        vec![GreenType::Ref, GreenType::Int],
    )
}

/// Hash of the jd2 greens `(pycode, driver index)`.
pub fn genentry_green_hash(pycode: PyObjectRef) -> u64 {
    majit_metainterp::green_key_hash_typed(
        &[pycode as i64, GENERATORENTRY_JD_INDEX],
        &[GreenType::Ref, GreenType::Int],
    )
}

/// `warmstate.py JitCell.get_jitcell`: resolve once and carry the cell key.
/// `make_key` runs only when the bucket holds more than one cell.
pub fn genentry_resolved_cell_key(
    warm: &majit_metainterp::warmstate::WarmEnterState,
    pycode: PyObjectRef,
) -> u64 {
    let hash = genentry_green_hash(pycode);
    warm.resolve_cell_key(hash, || genentry_green_key(pycode))
}

/// Reds in merge-point bank order (red R: `gen`, `w_arg`).
pub fn genentry_live_values(w_gen: PyObjectRef, w_arg: PyObjectRef) -> Vec<Value> {
    vec![
        Value::Ref(GcRef(w_gen as usize)),
        Value::Ref(GcRef(w_arg as usize)),
    ]
}

impl GenEntryJitState {
    /// jd2 (`generatorentry_driver`) portal descriptor: greens `pycode`,
    /// reds `gen`, `w_arg`, no virtualizable, `no_loop_header` set because
    /// the source has no `can_enter_jit`.
    pub fn generatorentry_driver_descriptor() -> JitDriverStaticData {
        crate::state::PyreJitState::generatorentry_driver_descriptor()
    }
}

impl JitState for GenEntryJitState {
    type Meta = PyreMeta;
    type Sym = GenEntrySym;
    type Env = PyreEnv;

    fn build_meta(&self, header_pc: usize, env: &Self::Env) -> Self::Meta {
        let _ = (header_pc, env);
        PyreMeta {
            num_locals: 0,
            ns_len: 0,
            namespace_dependent: false,
            valuestackdepth: 0,
            array_capacity: 0,
            trace_extra_reds: 0,
            has_virtualizable: false,
            slot_types: Vec::new(),
        }
    }

    fn extract_live(&mut self, meta: &Self::Meta) -> Vec<i64> {
        // The live reds are the tracer's InputArgs; this driver projects no
        // interpreter frame.
        let _ = meta;
        Vec::new()
    }

    fn create_sym(meta: &Self::Meta, header_pc: usize) -> Self::Sym {
        let _ = (meta, header_pc);
        GenEntrySym {
            pycode: PY_NULL,
            r#gen: OpRef::input_arg_typed(0, Type::Ref),
            w_arg: OpRef::input_arg_typed(1, Type::Ref),
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        // One novable shape per pycode.
        let _ = meta;
        true
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        // No interpreter frame to restore into; the reds live in the compiled
        // trace's registers.
        let _ = (meta, values);
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        vec![sym.r#gen, sym.w_arg]
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        let _ = (sym, meta);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_metainterp::warmstate::WarmEnterState;

    #[test]
    fn genentry_resolved_cell_key_reads_the_warmstate_it_is_handed() {
        // several_yields CALL_ASSEMBLER and eval.rs genentry_merge_point
        // resolve on jitdrivers_sd[2].warmstate. A hash-only occupant on
        // that table mints a sibling for the typed greens; portal slot 0
        // has no such cell, so it still answers the raw hash.
        let portal = WarmEnterState::new(1039);
        let mut jd2 = WarmEnterState::new(1039);
        let pycode = 0x51ED as PyObjectRef;
        let hash = genentry_green_hash(pycode);
        jd2.disable_noninlinable_function(hash);
        let typed = genentry_green_key(pycode);
        let minted = jd2.ensure_cell_key(&typed);
        assert_ne!(
            minted, hash,
            "typed genentry greens mint a sibling beside the hash-only cell"
        );
        assert_eq!(
            genentry_resolved_cell_key(&jd2, pycode),
            minted,
            "jd2 comparekey walks to the typed cell"
        );
        assert_eq!(
            genentry_resolved_cell_key(&portal, pycode),
            hash,
            "portal slot 0 has no genentry cell for these greens"
        );
    }
}
