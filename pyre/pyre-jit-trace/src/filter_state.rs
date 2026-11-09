//! jd4 (`filter_jitdriver`) symbolic + `JitState` scaffolding.
//!
//! `functional.py`
//! `filter_jitdriver = JitDriver(name='filter',
//! greens=['reverse', 'callable_greenkey', 'greenkey'], reds='auto')`
//! drives `_filter_jitdriver`. This module supplies the fourth-driver
//! types for the generic LLBC meta-tracer, via [`FilterJitState`]'s
//! [`JitState`] implementation.
//!
//! * [`FilterSym`] — the extracted `reds='auto'` of the split portal:
//!   `root_base`, `pred_slot`, and `RootScope.save_point`. The source pins
//!   `[w_iterable, w_predicate, callable_greenkey, greenkey]`.
//!   `split_portal` copies the loop to `filter_jitdriver_portal_portal`.
//! * [`FilterJitState`] — `Meta = PyreMeta`, `Sym = FilterSym`, `Env = PyreEnv`.

use majit_ir::{GreenKey, GreenType, OpRef, Type, Value};
use majit_metainterp::{JitCodeSym, JitDriverStaticData, JitState};

use crate::state::{PyreEnv, PyreMeta};
use pyre_object::{PY_NULL, PyObjectRef};

/// Registered main-jitcode key after `split_portal` copies the loop.
pub const FILTER_PORTAL_KEY: &str = "baseobjspace::filter_jitdriver_portal_portal";

/// `jitdrivers_sd` index of `filter_jitdriver`: registered right after
/// `_do_extend_jitdriver` (`jitdriver_sd.index`).
pub const FILTER_JD_INDEX: usize = 4;

/// jd4 symbolic state carried across the `filter_jitdriver` back-edge.
///
/// `functional.py` `reds='auto'` names the Python objects `w_iterable` and
/// `w_predicate`. The extracted Rust portal cannot keep those objects as
/// traced locals, so they live on the shadow stack and `reds='auto'` names
/// `root_base`, `pred_slot`, and `RootScope.save_point`.
pub struct FilterSym {
    /// `functional.py` `jit_merge_point(reverse=reverse)`.
    pub reverse: i64,
    /// `functional.py` `jit_merge_point(callable_greenkey=...)`.
    pub callable_greenkey: PyObjectRef,
    /// `functional.py` `jit_merge_point(greenkey=greenkey)`.
    pub greenkey: PyObjectRef,
    /// Shadow-stack slot of the pinned iterator.
    pub root_base: OpRef,
    /// Shadow-stack slot of the pinned predicate (`root_base + 1`).
    pub pred_slot: OpRef,
    /// `RootScope.save_point`. Equal to `root_base` in this portal
    /// (`push_roots` then `pin_roots` with nothing in between).
    pub save_point: OpRef,
}

impl JitCodeSym for FilterSym {
    fn total_slots(&self) -> usize {
        0
    }

    fn loop_header_pc(&self) -> usize {
        let canonical = crate::jitcode_runtime::portal_jitcode_for_key(FILTER_PORTAL_KEY)
            .expect("jd4 portal jitcode must be registered");
        crate::jitcode_runtime::decoded_ops(&canonical.code)
            .find(|op| op.opname == "jit_merge_point")
            .expect("jd4 skip body must contain a jit_merge_point")
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
            self.pred_slot = op;
        }
    }
}

/// jd4 `JitState`: reuses [`PyreMeta`] and the empty [`PyreEnv`], driving
/// [`FilterSym`]. Novable, so the meta carries no virtualizable payload.
pub struct FilterJitState {
    /// The merge-point Int green pinned for this driver activation.
    pub reverse: i64,
    /// The merge-point Ref green for the predicate's pycode or type.
    pub callable_greenkey: PyObjectRef,
    /// The merge-point Ref green pinned for this driver activation.
    pub greenkey: PyObjectRef,
}

/// Shadow-stack slot of the iterator pinned by `filter_jitdriver_portal`.
///
/// The portal pins `[w_iterable, w_predicate, callable_greenkey, greenkey]`.
/// Call this only while those four pins are still the top of the stack.
pub fn jd4_root_base() -> i64 {
    pyre_object::gc_roots::shadow_stack_len().saturating_sub(4) as i64
}

/// Loop-carried reds in merge-point bank order (three Ints).
///
/// `jtransform.py` `sort_vars` keeps declaration order when every red is
/// the int bank. The portal declares `_roots` / `root_base` / `pred_slot`,
/// so the extracted registers are save_point, root_base, pred_slot.
pub fn jd4_live_values_at(root_base: i64) -> Vec<Value> {
    vec![
        Value::Int(root_base),
        Value::Int(root_base),
        Value::Int(root_base + 1),
    ]
}

/// jd4 greens `(reverse, callable_greenkey, greenkey)` as a `GreenKey`
/// (`functional.py` `greens=['reverse', 'callable_greenkey', 'greenkey']`).
pub fn filter_green_key(
    reverse: i64,
    callable_greenkey: PyObjectRef,
    greenkey: PyObjectRef,
) -> GreenKey {
    GreenKey::with_types(
        vec![reverse, callable_greenkey as i64, greenkey as i64],
        vec![GreenType::Int, GreenType::Ref, GreenType::Ref],
    )
}

/// Hash of the jd4 greens `(reverse, callable_greenkey, greenkey)`.
pub fn filter_green_hash(
    reverse: i64,
    callable_greenkey: PyObjectRef,
    greenkey: PyObjectRef,
) -> u64 {
    majit_metainterp::green_key_hash_typed(
        &[reverse, callable_greenkey as i64, greenkey as i64],
        &[GreenType::Int, GreenType::Ref, GreenType::Ref],
    )
}

/// `warmstate.py JitCell.get_jitcell`: resolve once and carry the cell key.
pub fn filter_resolved_cell_key(
    warm: &majit_metainterp::warmstate::WarmEnterState,
    reverse: i64,
    callable_greenkey: PyObjectRef,
    greenkey: PyObjectRef,
) -> u64 {
    let hash = filter_green_hash(reverse, callable_greenkey, greenkey);
    warm.resolve_cell_key(hash, || {
        filter_green_key(reverse, callable_greenkey, greenkey)
    })
}

impl FilterJitState {
    /// jd4 (`filter_jitdriver`) portal descriptor.
    /// `functional.py` `greens=['reverse', 'callable_greenkey', 'greenkey'], reds='auto'`.
    /// The extracted body names `root_base`, `pred_slot`, and
    /// `RootScope.save_point`.
    pub fn filter_driver_descriptor() -> JitDriverStaticData {
        let mut sd = JitDriverStaticData::new(
            vec![
                ("reverse", Type::Int),
                ("callable_greenkey", Type::Ref),
                ("greenkey", Type::Ref),
            ],
            vec![
                ("save_point", Type::Int),
                ("root_base", Type::Int),
                ("pred_slot", Type::Int),
            ],
        );
        // functional.py filter_jitdriver = jit.JitDriver(name='filter', ...)
        sd.name = "filter".into();
        // functional.py `filter_jitdriver` = reds='auto', only a
        // `jit_merge_point` in the `while True` skip (no `can_enter_jit`, no
        // `loop_header`). warmspot.py rewrite_can_enter_jits leaves `no_loop_header` at its
        // `True` default for such a driver, so `opimpl_jit_merge_point`
        // (pyjitpl.py) auto-adds the loop header unconditionally.
        sd.no_loop_header = true;
        sd.frame_value_count_fn = Some(crate::state::build_time_frame_value_count_at);
        sd
    }
}

impl JitState for FilterJitState {
    type Meta = PyreMeta;
    type Sym = FilterSym;
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
        FilterSym {
            reverse: 0,
            callable_greenkey: PY_NULL,
            greenkey: PY_NULL,
            save_point: OpRef::input_arg_typed(0, Type::Int),
            root_base: OpRef::input_arg_typed(1, Type::Int),
            pred_slot: OpRef::input_arg_typed(2, Type::Int),
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
        vec![sym.save_point, sym.root_base, sym.pred_slot]
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        let _ = (sym, meta);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_jitcode::jitcode::BhDescr;
    use majit_metainterp::{JitCode, RuntimeBhDescr};

    #[test]
    fn jd4_build_time_descrs_resolve_through_global_pool() {
        let canonical = crate::jitcode_runtime::portal_jitcode_for_key(FILTER_PORTAL_KEY)
            .expect("jd4's extracted main JitCode must be registered");
        let merge_points: Vec<_> = crate::jitcode_runtime::decoded_ops(&canonical.code)
            .filter(|op| op.opname == "jit_merge_point")
            .collect();
        assert_eq!(merge_points.len(), 1);
        let banks =
            majit_metainterp::decode_jit_merge_point_banks(&canonical.code, merge_points[0].pc);
        assert_eq!(
            banks.green_i.len(),
            1,
            "extracted portal carries reverse as a green Int"
        );
        assert_eq!(
            banks.green_r.len(),
            2,
            "extracted portal carries callable_greenkey and greenkey as green Refs"
        );
        assert_eq!(banks.green_f.len(), 0);
        assert_eq!(
            banks.red_i.len(),
            3,
            "extracted portal carries root_base, pred_slot, and save_point as red Ints"
        );
        assert_eq!(
            banks.red_r.len(),
            0,
            "RootScope.save_point lives in the int bank"
        );
        assert_eq!(banks.red_f.len(), 0);

        crate::jitcode_runtime::install_global_build_descr_pool();
        let jitcode = JitCode::from_canonical((*canonical).clone());
        assert!(jitcode.exec.descrs.is_empty());

        let inline = crate::jitcode_runtime::decoded_ops(&canonical.code)
            .find(|op| op.opname.starts_with("inline_call_"))
            .expect("jd4 body has an inline call");
        let callee_idx = canonical.code[inline.pc + 1] as usize
            | ((canonical.code[inline.pc + 2] as usize) << 8);
        assert!(
            jitcode
                .descr_at(callee_idx)
                .and_then(RuntimeBhDescr::as_jitcode)
                .is_some(),
            "inline_call callee must resolve to a JitCode through the global pool"
        );

        let n = crate::jitcode_runtime::all_descrs().len();
        assert!(n > 0, "build_time descr pool must be non-empty");
        let mut jitcodes = 0usize;
        let mut calls = 0usize;
        for i in 0..n {
            let d = jitcode
                .descr_at(i)
                .expect("every global build-time descr slot must resolve");
            if d.as_jitcode().is_some() {
                jitcodes += 1;
            }
            if matches!(d.as_bh_descr(), Some(BhDescr::Call { .. })) {
                calls += 1;
            }
        }
        assert!(jitcodes > 0, "pool must resolve inline-call callees");
        assert!(calls > 0, "pool must resolve residual-call descrs");
    }
}
