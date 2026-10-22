//! Dormant jd1 (`unpackiterable_driver`) symbolic + `JitState` scaffolding.
//!
//! `baseobjspace.py`
//! `unpackiterable_driver = JitDriver(greens=['greenkey'], reds='auto', ...)`
//! drives the unknown-length unpack loop `unpackiterable_portal`
//! (`baseobjspace.py`, merge at `:1012`). This module supplies the
//! *dormant* second-driver types for the generic LLBC meta-tracer, via
//! [`UnpackJitState`]'s [`JitState`] implementation, without touching jd0's
//! `state.rs`:
//!
//! * [`UnpackSym`] — the extracted `reds='auto'` state of the Rust portal:
//!   `root_base`, `items_slot`, and the live `RootScope` save point.
//! * [`UnpackJitState`] — `Meta = PyreMeta`, `Sym = UnpackSym`, `Env = PyreEnv`.
//!
//! Dormant means gated, not unbuilt: the descriptor builder, its second
//! `register_jitdriver_sd` registration in `build_jit_driver_pair`, the
//! merge-point invocation (`eval.rs drive_unpack_iterable_trace`) and the
//! blackhole entry (`resume_in_blackhole_from_exit_layout`) all exist.  What
//! keeps them off the hot path is `PYRE_JD1`, which is opt-in because pyre
//! drives jd1 through the same `MetaInterp.tracing` slot as the bytecode
//! portal — see `eval.rs jd1_experiment_enabled` for why that is not yet
//! RPython's independent second driver.

use majit_ir::{OpRef, Type, Value};
use majit_metainterp::{JitCodeSym, JitDriverStaticData, JitState};

use crate::state::{PyreEnv, PyreMeta};
use pyre_object::{PY_NULL, PyObjectRef};

/// jd1 symbolic state carried across the `unpackiterable_driver` back-edge.
///
/// `baseobjspace.py` `reds='auto'` names the Python objects `w_iterator`
/// and `items`.  The extracted Rust portal cannot: there is no RPython GC
/// transform to keep those objects as traced locals, so they live on the
/// shadow stack and `reds='auto'` names `root_base`, `items_slot`, and the
/// `RootScope` save point.  Binding the objects as merge-point reds rematerializes
/// the slots after the marker and silently declines `try_fuse_drain_match`
/// (`front/result_exc.rs`), leaving `StopIteration` / `PyErrorKind::eq`
/// residuals.  JUMP / enter must match the extracted banks.
///
/// Convergence: an RPython-style GC transform that keeps `w_iterator` /
/// `items` as traced locals, so `reds='auto'` names them without a
/// shadow-stack pin and the drain fusion still sees
/// `next(shadow_stack_get(root_base))` as the last op of its block.
#[allow(dead_code)]
pub struct UnpackSym {
    /// baseobjspace.py jit_merge_point(greenkey=greenkey) green.
    pub greenkey: PyObjectRef,
    /// Shadow-stack slot of the pinned iterator.
    pub root_base: OpRef,
    /// Shadow-stack slot of the pinned `items` list (`root_base + 1`).
    pub items_slot: OpRef,
    /// `RootScope.save_point`. `RootScope` is `#[repr(transparent)]` over
    /// that `usize`, so rtyper classifies the live local as a red Int.
    pub roots_cell: OpRef,
}

impl JitCodeSym for UnpackSym {
    fn total_slots(&self) -> usize {
        0
    }

    fn loop_header_pc(&self) -> usize {
        // The `jit_merge_point` byte offset shifts with the extracted drain
        // body's op layout (e.g. how the `unpackiterable_driver` receiver read
        // lowers), so discover it rather than hardcode.  The drain loop carries
        // exactly one merge point (`baseobjspace.py`, no `can_enter_jit`).
        let canonical =
            crate::jitcode_runtime::portal_jitcode_for_key("baseobjspace::unpackiterable_portal")
                .expect("jd1 portal jitcode must be registered");
        crate::jitcode_runtime::decoded_ops(&canonical.code)
            .find(|op| op.opname == "jit_merge_point")
            .expect("jd1 drain body must contain a jit_merge_point")
            .pc
    }
}

/// jd1 `JitState`: reuses [`PyreMeta`] (the "shape of the traced code") and the
/// empty [`PyreEnv`], driving [`UnpackSym`]. Novable, so the meta carries no
/// virtualizable payload and no extra reds.
pub struct UnpackJitState {
    /// The merge-point green pinned for this driver activation.
    pub greenkey: PyObjectRef,
}

/// Shadow-stack slot of the iterator pinned by `unpackiterable_portal`.
///
/// Call this only while those two pins are still the top of the stack.
/// `drive_unpack_iterable_trace` captures the value before
/// `ResidualExceptionScope::park` can push exception roots on top.
pub fn jd1_root_base() -> i64 {
    pyre_object::gc_roots::shadow_stack_len().saturating_sub(2) as i64
}

/// Loop-carried reds in merge-point bank order (red I, then red R).
///
/// `unpackiterable_portal` does `push_roots` then `pin_roots` of the three
/// objects with nothing in between, so the `RootScope` save point equals
/// `root_base`. The extracted merge names that word as a third red Int.
pub fn jd1_live_values_at(root_base: i64) -> Vec<Value> {
    vec![
        Value::Int(root_base),
        Value::Int(root_base + 1),
        Value::Int(root_base),
    ]
}

/// `jitdrivers_sd` index of `unpackiterable_driver`: registered right
/// after the Python portal (`jitdriver_sd.index`).
pub const UNPACKITERABLE_JD_INDEX: usize = 1;

impl UnpackJitState {
    /// jd1 (`unpackiterable_driver`) portal descriptor.
    /// `baseobjspace.py` `greens=['greenkey'], reds='auto'`. The extracted
    /// body names `root_base`, `items_slot`, and the `RootScope` save point.
    pub fn unpackiterable_driver_descriptor() -> JitDriverStaticData {
        let mut sd = JitDriverStaticData::new(
            vec![("greenkey", Type::Ref)],
            vec![
                ("root_base", Type::Int),
                ("items_slot", Type::Int),
                ("roots_cell", Type::Int),
            ],
        );
        // baseobjspace.py unpackiterable_driver = jit.JitDriver(name='unpackiterable', ...)
        sd.name = "unpackiterable".into();
        // baseobjspace.py `unpackiterable_driver` = reds='auto', only a
        // `jit_merge_point` in the `while True` drain (no `can_enter_jit`, no
        // `loop_header`). warmspot.py rewrite_can_enter_jits leaves `no_loop_header` at its
        // `True` default for such a driver, so `opimpl_jit_merge_point`
        // (pyjitpl.py) auto-adds the loop header unconditionally — the
        // first trace closes the loop on the `goto` back-edge instead of
        // unrolling the whole drain to the StopIteration finish.
        sd.no_loop_header = true;
        // Guard metadata decodes `-live-` from `MetaInterpStaticData.jitcodes`.
        // The callback name stays so this driver does not move in the same patch.
        sd.frame_value_count_fn = Some(crate::state::build_time_frame_value_count_at);
        sd
    }
}

impl JitState for UnpackJitState {
    type Meta = PyreMeta;
    type Sym = UnpackSym;
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
        // The live reds are the tracer's InputArgs, not interpreter-frame
        // values; the dormant driver holds no concrete frame to project them
        // from.
        let _ = meta;
        Vec::new()
    }

    fn create_sym(meta: &Self::Meta, header_pc: usize) -> Self::Sym {
        let _ = (meta, header_pc);
        UnpackSym {
            greenkey: PY_NULL,
            root_base: OpRef::input_arg_typed(0, Type::Int),
            items_slot: OpRef::input_arg_typed(1, Type::Int),
            roots_cell: OpRef::input_arg_typed(2, Type::Int),
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        // Single novable shape per greenkey.
        let _ = meta;
        true
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        // No interpreter frame to restore into; the reds live in the compiled
        // loop's registers.
        let _ = (meta, values);
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        vec![sym.root_base, sym.items_slot, sym.roots_cell]
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

    /// jd1's extracted main JitCode carries an empty per-jitcode `exec.descrs`;
    /// its `d`/`j` argcodes resolve through the shared global build-time descr
    /// pool (`install_global_build_descr_pool` -> `JitCode::descr_at`),
    /// mirroring RPython's single shared `Assembler.descrs`. This checks that
    /// resolution wiring — not the full trace walk, which executes each
    /// residual (`self.next`, `items.append`) for real and so needs the live
    /// runtime with patched fn-addresses (a later activation slice), not a bare
    /// unit test.
    #[test]
    fn jd1_build_time_descrs_resolve_through_global_pool() {
        let canonical =
            crate::jitcode_runtime::portal_jitcode_for_key("baseobjspace::unpackiterable_portal")
                .expect("jd1's extracted main JitCode must be registered");
        // The extracted body is the walkable unpack loop: exactly one merge
        // point, whose byte offset depends on the drain body's op layout
        // (`UnpackSym::loop_header_pc` discovers it rather than hardcoding).
        let merge_points: Vec<_> = crate::jitcode_runtime::decoded_ops(&canonical.code)
            .filter(|op| op.opname == "jit_merge_point")
            .collect();
        assert_eq!(merge_points.len(), 1);
        let banks =
            majit_metainterp::decode_jit_merge_point_banks(&canonical.code, merge_points[0].pc);
        assert_eq!(banks.green_i.len(), 0);
        assert_eq!(banks.green_r.len(), 1);
        assert_eq!(banks.green_f.len(), 0);
        assert_eq!(
            banks.red_i.len(),
            3,
            "extracted portal carries root_base, items_slot, and the RootScope save point as red Ints"
        );
        assert_eq!(
            banks.red_r.len(),
            0,
            "RootScope is a transparent Int save point, so the extracted portal has no object red"
        );
        assert_eq!(banks.red_f.len(), 0);

        crate::jitcode_runtime::install_global_build_descr_pool();
        let jitcode = JitCode::from_canonical((*canonical).clone());
        // A build-time jitcode has no per-jitcode descr pool; every resolution
        // below therefore exercises the global fallback in `descr_at`.
        assert!(jitcode.exec.descrs.is_empty());

        // jd1's first inline-call names its callee in the leading 2-byte `d`
        // operand; `descr_at` must map it to a JitCode through the global
        // pool. Every `inline_call_*` spelling puts `d` first (`dR>r`,
        // `dIR`, `dIRF>f`, ...), so the family is what this reads -- the body
        // currently inlines `drain_append_at` as `inline_call_ir_v/dIR`.
        let inline = crate::jitcode_runtime::decoded_ops(&canonical.code)
            .find(|op| op.opname.starts_with("inline_call_"))
            .expect("jd1 body has an inline call");
        let callee_idx = canonical.code[inline.pc + 1] as usize
            | ((canonical.code[inline.pc + 2] as usize) << 8);
        assert!(
            jitcode
                .descr_at(callee_idx)
                .and_then(RuntimeBhDescr::as_jitcode)
                .is_some(),
            "inline_call callee must resolve to a JitCode through the global pool"
        );

        // Every slot in the shared pool resolves, and it carries both the
        // JitCode (inline-call) and Call (residual-call) descr families the jd1
        // loop body reads back via `descr_at` — the residual-call descr read
        // that previously panicked on the empty per-jitcode pool.
        let n = crate::jitcode_runtime::all_descrs().len();
        assert!(n > 0, "build-time descr pool must be non-empty");
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
