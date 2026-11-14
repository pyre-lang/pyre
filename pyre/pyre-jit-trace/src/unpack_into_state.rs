//! jd5 (`unpack_into`) symbolic + `JitState` scaffolding.
//!
//! `generator.py` `_create_unpack_into`:
//! `jitdriver = JitDriver(greens=['pycode'], reds='auto', name='unpack_into')`
//! drives `unpack_into_w`. This module supplies the fifth-driver types for
//! the generic LLBC meta-tracer, via [`UnpackIntoJitState`]'s [`JitState`]
//! implementation.
//!
//! * [`UnpackIntoSym`] — the extracted `reds='auto'` of the split portal:
//!   `root_base`, `list_slot`, and `RootScope.save_point`. The source pins
//!   `[w_gen, w_list, pycode]`. `split_portal` copies the loop to
//!   `unpack_into_w_portal_portal`.
//! * [`UnpackIntoJitState`] — `Meta = PyreMeta`, `Sym = UnpackIntoSym`,
//!   `Env = PyreEnv`.
//!
//! The interp-level twin `unpack_into` (a `Vec` of `W_Root`) is still a
//! no-op merge point; this slot is `unpack_into_w` only
//! (`listobject.py ListStrategy.extend` / `space.is_generator`).

use majit_ir::{GreenKey, GreenType, OpRef, Type, Value};
use majit_metainterp::{JitCodeSym, JitDriverStaticData, JitState};

use crate::state::{PyreEnv, PyreMeta};
use pyre_object::{PY_NULL, PyObjectRef};

/// Registered main-jitcode key after `split_portal` copies the loop.
pub const UNPACK_INTO_PORTAL_KEY: &str = "baseobjspace::unpack_into_w_portal_portal";

/// `jitdrivers_sd` index of `unpack_into`: registered right after
/// `filter_jitdriver` (`jitdriver_sd.index`).
pub const UNPACK_INTO_JD_INDEX: usize = 5;

/// jd5 symbolic state carried across the `unpack_into` back-edge.
///
/// `generator.py` `reds='auto'` names `self` and `results`. The extracted
/// Rust portal cannot keep those objects as traced locals, so they live
/// on the shadow stack and `reds='auto'` names `root_base`, `list_slot`,
/// and `RootScope.save_point`.
pub struct UnpackIntoSym {
    /// `generator.py` `jit_merge_point(pycode=pycode)` green.
    pub pycode: PyObjectRef,
    /// Shadow-stack slot of the pinned generator.
    pub root_base: OpRef,
    /// Shadow-stack slot of the pinned list (`root_base + 1`).
    pub list_slot: OpRef,
    /// `RootScope.save_point`. Equal to `root_base` in this portal
    /// (`push_roots` then `pin_roots` with nothing in between).
    pub save_point: OpRef,
}

impl JitCodeSym for UnpackIntoSym {
    fn total_slots(&self) -> usize {
        0
    }

    fn loop_header_pc(&self) -> usize {
        let canonical = crate::jitcode_runtime::portal_jitcode_for_key(UNPACK_INTO_PORTAL_KEY)
            .expect("jd5 portal jitcode must be registered");
        crate::jitcode_runtime::decoded_ops(&canonical.code)
            .find(|op| op.opname == "jit_merge_point")
            .expect("jd5 drain body must contain a jit_merge_point")
            .pc
    }

    fn set_redboxes(&mut self, redboxes: &[(OpRef, Type)]) {
        // pyjitpl.py opimpl_jit_merge_point → reached_loop_header redboxes
        if let Some(&(op, _)) = redboxes.first() {
            self.save_point = op;
        }
        if let Some(&(op, _)) = redboxes.get(1) {
            self.root_base = op;
        }
        if let Some(&(op, _)) = redboxes.get(2) {
            self.list_slot = op;
        }
    }
}

/// jd5 `JitState`: reuses [`PyreMeta`] and the empty [`PyreEnv`], driving
/// [`UnpackIntoSym`]. Novable, so the meta carries no virtualizable payload.
pub struct UnpackIntoJitState {
    /// The merge-point Ref green pinned for this driver activation.
    pub pycode: PyObjectRef,
}

/// Shadow-stack slot of the generator pinned by `unpack_into_w_portal`.
///
/// The portal pins `[w_gen, w_list, pycode]`. Call this only while those
/// three pins are still the top of the stack.
pub fn jd5_root_base() -> i64 {
    pyre_object::gc_roots::shadow_stack_len().saturating_sub(3) as i64
}

/// Loop-carried reds in merge-point bank order (three Ints).
///
/// `jtransform.py` `sort_vars` keeps declaration order when every red is
/// the int bank. The portal declares `_roots` / `root_base` / `list_slot`,
/// so the extracted registers are save_point, root_base, list_slot.
pub fn unpack_into_live_values_at(root_base: i64) -> Vec<Value> {
    vec![
        Value::Int(root_base),
        Value::Int(root_base),
        Value::Int(root_base + 1),
    ]
}

/// jd5 greens `(pycode)` as a `GreenKey` (`generator.py` `greens=['pycode']`).
pub fn unpack_into_green_key(pycode: PyObjectRef) -> GreenKey {
    GreenKey::with_types(vec![pycode as i64], vec![GreenType::Ref])
}

/// Hash of the jd5 greens `(pycode)`.
pub fn unpack_into_green_hash(pycode: PyObjectRef) -> u64 {
    majit_metainterp::green_key_hash_typed(&[pycode as i64], &[GreenType::Ref])
}

/// `warmstate.py JitCell.get_jitcell`: resolve once and carry the cell key.
pub fn unpack_into_resolved_cell_key(
    warm: &majit_metainterp::warmstate::WarmEnterState,
    pycode: PyObjectRef,
) -> u64 {
    let hash = unpack_into_green_hash(pycode);
    warm.resolve_cell_key(hash, || unpack_into_green_key(pycode))
}

impl UnpackIntoJitState {
    /// jd5 (`unpack_into`) portal descriptor.
    /// `generator.py` `greens=['pycode'], reds='auto', name='unpack_into'`.
    /// The extracted body names `root_base`, `list_slot`, and
    /// `RootScope.save_point`.
    pub fn unpack_into_driver_descriptor() -> JitDriverStaticData {
        let mut sd = JitDriverStaticData::new(
            vec![("pycode", Type::Ref)],
            vec![
                ("save_point", Type::Int),
                ("root_base", Type::Int),
                ("list_slot", Type::Int),
            ],
        );
        // generator.py jitdriver = jit.JitDriver(name='unpack_into', ...)
        sd.name = "unpack_into".into();
        // generator.py `unpack_into` = reds='auto', only a `jit_merge_point`
        // in the `while True` drain (no `can_enter_jit`, no `loop_header`).
        // warmspot.py rewrite_can_enter_jits leaves `no_loop_header` at its
        // `True` default for such a driver, so `opimpl_jit_merge_point`
        // (pyjitpl.py) auto-adds the loop header unconditionally.
        sd.no_loop_header = true;
        sd.frame_value_count_fn = Some(crate::state::build_time_frame_value_count_at);
        sd
    }
}

impl JitState for UnpackIntoJitState {
    type Meta = PyreMeta;
    type Sym = UnpackIntoSym;
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
        let _ = meta;
        Vec::new()
    }

    fn create_sym(meta: &Self::Meta, header_pc: usize) -> Self::Sym {
        let _ = (meta, header_pc);
        UnpackIntoSym {
            pycode: PY_NULL,
            save_point: OpRef::input_arg_typed(0, Type::Int),
            root_base: OpRef::input_arg_typed(1, Type::Int),
            list_slot: OpRef::input_arg_typed(2, Type::Int),
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        let _ = meta;
        true
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        let _ = (meta, values);
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        vec![sym.save_point, sym.root_base, sym.list_slot]
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        let _ = (sym, meta);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::GreenKey;
    use majit_metainterp::warmstate::WarmEnterState;

    #[test]
    fn unpack_into_green_key_matches_descriptor_green_args_spec() {
        let p = 0x51ED as PyObjectRef;
        let jd5 = UnpackIntoJitState::unpack_into_driver_descriptor();
        assert_eq!(
            unpack_into_green_key(p),
            GreenKey::with_types(vec![p as i64], jd5.green_args_spec()),
        );
    }

    #[test]
    fn unpack_into_resolved_cell_key_reads_the_warmstate_it_is_handed() {
        let portal = WarmEnterState::new(1039);
        let mut jd5 = WarmEnterState::new(1039);
        let pycode = 0x51ED as PyObjectRef;
        let typed = unpack_into_green_key(pycode);
        let hash = typed.get_uhash();
        let squatter = colliding_squatter_key(&typed);
        assert_eq!(squatter.get_uhash(), hash);
        assert_ne!(squatter, typed);
        jd5.disable_noninlinable_function_for_key(&squatter);
        let minted = jd5.ensure_cell_key(&typed);
        assert_ne!(
            minted, hash,
            "the squatter keeps the raw hash; the typed cell is minted"
        );
        assert_eq!(
            unpack_into_resolved_cell_key(&jd5, pycode),
            minted,
            "jd5 comparekey walks to the typed cell"
        );
        assert_eq!(
            unpack_into_resolved_cell_key(&portal, pycode),
            hash,
            "portal slot 0 has no unpack_into cell for these greens"
        );
    }

    fn colliding_squatter_key(typed: &GreenKey) -> GreenKey {
        use majit_ir::{GREEN_UHASH_MULT, GREEN_UHASH_SEED, green_uhash_step};
        let hash = typed.get_uhash();
        let mut inverse: u64 = 1;
        for _ in 0..6 {
            inverse =
                inverse.wrapping_mul(2u64.wrapping_sub(GREEN_UHASH_MULT.wrapping_mul(inverse)));
        }
        let first = 1i64;
        let prefix = green_uhash_step(GREEN_UHASH_SEED, GreenType::Int, first);
        GreenKey::new(vec![first, (prefix ^ hash.wrapping_mul(inverse)) as i64])
    }
}
