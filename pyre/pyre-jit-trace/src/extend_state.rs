//! jd3 (`list._do_extend_from_iterable`) symbolic + `JitState` scaffolding.
//!
//! `listobject.py`
//! `_do_extend_jitdriver = JitDriver(greens=['strategy_type', 'greenkey'],
//! reds='auto', name='list__do_extend_from_iterable')`
//! drives `_do_extend_from_iterable`. This module supplies the third-driver
//! types for the generic LLBC meta-tracer, via [`ExtendJitState`]'s
//! [`JitState`] implementation.
//!
//! * [`ExtendSym`] — the extracted `reds='auto'` of the split portal:
//!   `root_base`, `items_slot`, the loop-carried append count `i`, and
//!   `RootScope.save_point`. The source pins `[w_iterator, w_list, greenkey]`
//!   matching [`crate::unpack_state::UnpackSym`]. `split_portal` copies the
//!   loop to `list_extend_portal_portal`.
//! * [`ExtendJitState`] — `Meta = PyreMeta`, `Sym = ExtendSym`, `Env = PyreEnv`.
//!
//! `strategy_type` is an Int green. Upstream `type(w_list.strategy)` is the
//! strategy class object (a Ref). pyre's `ListStrategy` is a `#[repr(u8)]`
//! enum, so the discriminant is the green; `ListStrategy::class_name`
//! keeps the printable spelling.

use majit_ir::{GreenKey, GreenType, OpRef, Type, Value};
use majit_metainterp::{JitCodeSym, JitDriverStaticData, JitState};

use crate::state::{PyreEnv, PyreMeta};
use pyre_object::{PY_NULL, PyObjectRef};

/// Registered main-jitcode key after `split_portal` copies the loop.
pub const LIST_EXTEND_PORTAL_KEY: &str = "type_methods::list_extend_portal_portal";

/// `jitdrivers_sd` index of `_do_extend_jitdriver`: registered right
/// after `generatorentry` (`jitdriver_sd.index`).
pub const LIST_EXTEND_JD_INDEX: usize = 3;

/// jd3 symbolic state carried across the `_do_extend_jitdriver` back-edge.
///
/// `listobject.py` `reds='auto'` names the Python objects `w_list` and
/// `w_iterator` and the count `i`. The extracted Rust portal cannot keep
/// those objects as traced locals, so they live on the shadow stack and
/// `reds='auto'` names `root_base`, `items_slot`, `i`, and
/// `RootScope.save_point`. JUMP / enter must match the extracted banks.
pub struct ExtendSym {
    /// `listobject.py` `jit_merge_point(strategy_type=type(w_list.strategy))`.
    pub strategy_type: i64,
    /// `listobject.py` `jit_merge_point(greenkey=greenkey)` green.
    pub greenkey: PyObjectRef,
    /// Shadow-stack slot of the pinned iterator.
    pub root_base: OpRef,
    /// Shadow-stack slot of the pinned list (`root_base + 1`).
    pub items_slot: OpRef,
    /// Loop-carried append count (`listobject.py` `i`).
    pub count: OpRef,
    /// `RootScope.save_point`. Equal to `root_base` in this portal
    /// (`push_roots` then `pin_roots` with nothing in between).
    pub save_point: OpRef,
}

impl JitCodeSym for ExtendSym {
    fn total_slots(&self) -> usize {
        0
    }

    fn loop_header_pc(&self) -> usize {
        let canonical = crate::jitcode_runtime::portal_jitcode_for_key(LIST_EXTEND_PORTAL_KEY)
            .expect("jd3 portal jitcode must be registered");
        crate::jitcode_runtime::decoded_ops(&canonical.code)
            .find(|op| op.opname == "jit_merge_point")
            .expect("jd3 drain body must contain a jit_merge_point")
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
            self.items_slot = op;
        }
        if let Some(&(op, _)) = redboxes.get(3) {
            self.count = op;
        }
    }
}

/// jd3 `JitState`: reuses [`PyreMeta`] and the empty [`PyreEnv`], driving
/// [`ExtendSym`]. Novable, so the meta carries no virtualizable payload.
pub struct ExtendJitState {
    /// The merge-point Int green pinned for this driver activation.
    pub strategy_type: i64,
    /// The merge-point Ref green pinned for this driver activation.
    pub greenkey: PyObjectRef,
}

/// Shadow-stack slot of the iterator pinned by `list_extend_portal`.
///
/// The portal pins `[w_iterator, w_list, greenkey]`. Call this only while
/// those three pins are still the top of the stack. The merge-point hook
/// prefers the portal's live `root_base` local over this reconstruction.
pub fn jd3_root_base() -> i64 {
    pyre_object::gc_roots::shadow_stack_len().saturating_sub(3) as i64
}

/// Loop-carried reds in merge-point bank order (four Ints).
///
/// `jtransform.py` `sort_vars` keeps declaration order when every red is
/// the int bank. The portal declares `_roots` / `root_base` / `list_slot`
/// / `i`, so the extracted registers are save_point, root_base, list_slot,
/// count.
pub fn jd3_live_values_at(root_base: i64, count: i64) -> Vec<Value> {
    vec![
        Value::Int(root_base),
        Value::Int(root_base),
        Value::Int(root_base + 1),
        Value::Int(count),
    ]
}

/// jd3 greens `(strategy_type, greenkey)` as a `GreenKey`
/// (`listobject.py` `greens=['strategy_type', 'greenkey']`).
pub fn list_extend_green_key(strategy_type: i64, greenkey: PyObjectRef) -> GreenKey {
    GreenKey::with_types(
        vec![strategy_type, greenkey as i64],
        vec![GreenType::Int, GreenType::Ref],
    )
}

/// Hash of the jd3 greens `(strategy_type, greenkey)`.
pub fn list_extend_green_hash(strategy_type: i64, greenkey: PyObjectRef) -> u64 {
    majit_metainterp::green_key_hash_typed(
        &[strategy_type, greenkey as i64],
        &[GreenType::Int, GreenType::Ref],
    )
}

/// `warmstate.py JitCell.get_jitcell`: resolve once and carry the cell key.
pub fn list_extend_resolved_cell_key(
    warm: &majit_metainterp::warmstate::WarmEnterState,
    strategy_type: i64,
    greenkey: PyObjectRef,
) -> u64 {
    let hash = list_extend_green_hash(strategy_type, greenkey);
    warm.resolve_cell_key(hash, || list_extend_green_key(strategy_type, greenkey))
}

impl ExtendJitState {
    /// jd3 (`_do_extend_jitdriver`) portal descriptor.
    /// `listobject.py` `greens=['strategy_type', 'greenkey'], reds='auto'`.
    /// The extracted body names `root_base`, `items_slot`, `i`, and
    /// `RootScope.save_point`.
    pub fn list_extend_driver_descriptor() -> JitDriverStaticData {
        let mut sd = JitDriverStaticData::new(
            vec![("strategy_type", Type::Int), ("greenkey", Type::Ref)],
            vec![
                ("save_point", Type::Int),
                ("root_base", Type::Int),
                ("items_slot", Type::Int),
                ("count", Type::Int),
            ],
        );
        // listobject.py _do_extend_jitdriver = jit.JitDriver(name='list__do_extend_from_iterable', ...)
        sd.name = "list__do_extend_from_iterable".into();
        // listobject.py `_do_extend_jitdriver` = reds='auto', only a
        // `jit_merge_point` in the `while True` drain (no `can_enter_jit`, no
        // `loop_header`). warmspot.py rewrite_can_enter_jits leaves `no_loop_header` at its
        // `True` default for such a driver, so `opimpl_jit_merge_point`
        // (pyjitpl.py) auto-adds the loop header unconditionally.
        sd.no_loop_header = true;
        sd.frame_value_count_fn = Some(crate::state::build_time_frame_value_count_at);
        sd
    }
}

impl JitState for ExtendJitState {
    type Meta = PyreMeta;
    type Sym = ExtendSym;
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
        ExtendSym {
            strategy_type: 0,
            greenkey: PY_NULL,
            save_point: OpRef::input_arg_typed(0, Type::Int),
            root_base: OpRef::input_arg_typed(1, Type::Int),
            items_slot: OpRef::input_arg_typed(2, Type::Int),
            count: OpRef::input_arg_typed(3, Type::Int),
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
        vec![sym.save_point, sym.root_base, sym.items_slot, sym.count]
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
    fn jd3_build_time_descrs_resolve_through_global_pool() {
        let canonical = crate::jitcode_runtime::portal_jitcode_for_key(LIST_EXTEND_PORTAL_KEY)
            .expect("jd3's extracted main JitCode must be registered");
        let merge_points: Vec<_> = crate::jitcode_runtime::decoded_ops(&canonical.code)
            .filter(|op| op.opname == "jit_merge_point")
            .collect();
        assert_eq!(merge_points.len(), 1);
        let banks =
            majit_metainterp::decode_jit_merge_point_banks(&canonical.code, merge_points[0].pc);
        assert_eq!(
            banks.green_i.len(),
            1,
            "extracted portal carries strategy_type as a green Int"
        );
        assert_eq!(banks.green_r.len(), 1);
        assert_eq!(banks.green_f.len(), 0);
        assert_eq!(
            banks.red_i.len(),
            4,
            "extracted portal carries root_base, items_slot, i, and save_point as red Ints"
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
            .expect("jd3 body has an inline call");
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
