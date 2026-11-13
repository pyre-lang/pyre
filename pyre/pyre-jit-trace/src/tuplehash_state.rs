//! jd3 (`hash_driver`) symbolic + `JitState` scaffolding.
//!
//! `tupleobject.py` `hash_driver = jit.JitDriver(greens=['w_type'],
//! reds='auto', get_printable_location=..., name='tuple.hash')` drives
//! `_descr_hash_jitdriver`. `split_portal` copies the loop to
//! `tuple_hash_portal_portal` and leaves the original as the runner
//! stub, so the registered main jitcode is that copy (`warmspot.py`
//! `split_graph_and_record_jitdriver`).
//!
//! * [`TupleHashSym`] — the extracted `reds='auto'` state of the portal:
//!   `len`, `acc`, `i`, and `obj`.
//! * [`TupleHashJitState`] — `Meta = PyreMeta`, `Sym = TupleHashSym`, `Env = PyreEnv`.

use majit_ir::{GcRef, GreenKey, GreenType, OpRef, Type, Value};
use majit_metainterp::{JitCodeSym, JitDriverStaticData, JitState};

use crate::state::{PyreEnv, PyreMeta};
use pyre_object::{PY_NULL, PyObjectRef};

/// Registered main-jitcode key after `split_portal` copies the loop.
pub const TUPLEHASH_PORTAL_KEY: &str = "builtins::tuple_hash_portal_portal";

/// jd3 symbolic state at the `hash_driver` merge point.
///
/// `tupleobject.py` `_descr_hash_jitdriver` names the live set
/// `reds='auto'`; autodetect records three Int reds then one Ref red
/// (`len`, `acc`, `i`, `obj`) as `jd.red_args_types`.
pub struct TupleHashSym {
    /// `jit_merge_point(w_type=...)` green.
    pub w_type: PyObjectRef,
    /// Tuple length, loop-carried across the merge.
    pub len: OpRef,
    /// xxHash accumulator (`acc`).
    pub acc: OpRef,
    /// Walk index (`i`).
    pub i: OpRef,
    /// The tuple being hashed (`obj`).
    pub obj: OpRef,
}

impl JitCodeSym for TupleHashSym {
    fn total_slots(&self) -> usize {
        0
    }

    fn loop_header_pc(&self) -> usize {
        // The marker's byte offset shifts with the extracted portal's op
        // layout, so discover it. The portal carries exactly one merge point
        // (`_descr_hash_jitdriver` has no `can_enter_jit`).
        let canonical = crate::jitcode_runtime::portal_jitcode_for_key(TUPLEHASH_PORTAL_KEY)
            .expect("jd3 portal jitcode must be registered");
        crate::jitcode_runtime::decoded_ops(&canonical.code)
            .find(|op| op.opname == "jit_merge_point")
            .expect("jd3 portal body must contain a jit_merge_point")
            .pc
    }

    fn set_redboxes(&mut self, redboxes: &[(OpRef, Type)]) {
        // pyjitpl.py `opimpl_jit_merge_point` → `reached_loop_header(greenboxes, redboxes)`:
        // the boxes held by the merge point's red registers, in `(I, R, F)`
        // operand order (`handle_jit_marker__jit_merge_point` / `make_three_lists`).
        // `hash_driver` reds='auto' is `len`, `acc`, `i`, then `obj`.
        match redboxes {
            [
                (len, Type::Int),
                (acc, Type::Int),
                (i, Type::Int),
                (obj, Type::Ref),
            ] => {
                self.len = *len;
                self.acc = *acc;
                self.i = *i;
                self.obj = *obj;
            }
            other => panic!(
                "tuple.hash merge-point reds are len, acc, i, obj in (I,I,I,R) order, got {other:?}"
            ),
        }
    }

    fn loop_carried_boxes(&self, vable_boxes: &[(OpRef, Type)]) -> Option<Vec<(OpRef, Type)>> {
        // pyjitpl.py `reached_loop_header`:
        //   live_arg_boxes = greenboxes + redboxes
        //   live_arg_boxes += self.virtualizable_boxes
        //   live_arg_boxes.pop()
        let mut boxes = vec![
            (self.len, Type::Int),
            (self.acc, Type::Int),
            (self.i, Type::Int),
            (self.obj, Type::Ref),
        ];
        let n = vable_boxes.len().saturating_sub(1);
        boxes.extend_from_slice(&vable_boxes[..n]);
        Some(boxes)
    }
}

/// jd3 `JitState`: reuses [`PyreMeta`] and the empty [`PyreEnv`], driving
/// [`TupleHashSym`]. Novable: the tuple is a red, not a virtualizable of
/// this driver.
pub struct TupleHashJitState {
    /// The merge-point green pinned for this driver activation.
    pub w_type: PyObjectRef,
}

/// `jitdrivers_sd` index of `hash_driver`.
pub const TUPLEHASH_JD_INDEX: usize = 3;

/// jd3 greens `(w_type)` as a `GreenKey` (`tupleobject.py` `greens=['w_type']`).
pub fn tuple_hash_green_key(w_type: PyObjectRef) -> GreenKey {
    GreenKey::with_types(vec![w_type as i64], vec![GreenType::Ref])
}

/// `tupleobject.py` `greens=['w_type']` — one Ref green.
pub const TUPLE_HASH_GREEN_TYPES: [GreenType; 1] = [GreenType::Ref];

/// Hash of the jd3 greens `(w_type)`.
#[inline]
pub fn tuple_hash_green_hash(w_type: PyObjectRef) -> u64 {
    majit_metainterp::green_key_hash_typed(&[w_type as i64], &TUPLE_HASH_GREEN_TYPES)
}

/// `warmstate.py JitCell.get_jitcell`: resolve once and carry the cell key.
///
/// `JitCell.get_jitcell(*greenargs)` compares the live greens in place, so
/// this builds no [`GreenKey`]. Function-entry uses the same
/// `resolve_cell_key_with_greenargs` door (`eval.rs`
/// `tick_function_entry_from_portal`).
#[inline]
pub fn tuple_hash_resolved_cell_key(
    warm: &majit_metainterp::warmstate::WarmEnterState,
    w_type: PyObjectRef,
) -> u64 {
    let greenargs = [w_type as i64];
    let hash = tuple_hash_green_hash(w_type);
    warm.resolve_cell_key_with_greenargs(hash, &greenargs, &TUPLE_HASH_GREEN_TYPES)
}

/// Reds in merge-point bank order (red I: `len`, `acc`, `i`; red R: `obj`).
///
/// Stack array so the merge-point tick never heap-allocates them.
/// `warmstate.py maybe_compile_and_run` builds `execute_args` only when
/// raising `EnterJitAssembler`.
#[inline]
pub fn tuple_hash_live_values(len: i64, acc: i64, i: i64, obj: PyObjectRef) -> [Value; 4] {
    [
        Value::Int(len),
        Value::Int(acc),
        Value::Int(i),
        Value::Ref(GcRef(obj as usize)),
    ]
}

/// `llmodel.py execute_token` words for the same reds (`unspecialize_value`).
#[inline]
pub fn tuple_hash_raw_reds(len: i64, acc: i64, i: i64, obj: PyObjectRef) -> [i64; 4] {
    [len, acc, i, obj as usize as i64]
}

impl TupleHashJitState {
    /// jd3 (`hash_driver`) portal descriptor.
    /// `tupleobject.py` `greens=['w_type'], reds='auto'`. The extracted
    /// body names `len`, `acc`, `i`, and `obj`.
    pub fn tuple_hash_driver_descriptor() -> JitDriverStaticData {
        let mut sd = JitDriverStaticData::new(
            vec![("w_type", Type::Ref)],
            vec![
                ("len", Type::Int),
                ("acc", Type::Int),
                ("i", Type::Int),
                ("obj", Type::Ref),
            ],
        );
        // tupleobject.py hash_driver = jit.JitDriver(name='tuple.hash', ...)
        sd.name = "tuple.hash".into();
        // tupleobject.py `hash_driver` = reds='auto', only a
        // `jit_merge_point` in `_descr_hash_jitdriver` (no `can_enter_jit`,
        // no `loop_header`). warmspot.py rewrite_can_enter_jits leaves
        // `no_loop_header` at its `True` default for such a driver, so
        // `opimpl_jit_merge_point` (pyjitpl.py) auto-adds the loop header
        // unconditionally — the first trace closes the loop on the `goto`
        // back-edge instead of unrolling the whole walk to the finish.
        sd.no_loop_header = true;
        // Guard metadata decodes `-live-` from `MetaInterpStaticData.jitcodes`.
        sd.frame_value_count_fn = Some(crate::state::build_time_frame_value_count_at);
        sd
    }
}

impl JitState for TupleHashJitState {
    type Meta = PyreMeta;
    type Sym = TupleHashSym;
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
        TupleHashSym {
            w_type: PY_NULL,
            len: OpRef::input_arg_typed(0, Type::Int),
            acc: OpRef::input_arg_typed(1, Type::Int),
            i: OpRef::input_arg_typed(2, Type::Int),
            obj: OpRef::input_arg_typed(3, Type::Ref),
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        // One novable shape per w_type.
        let _ = meta;
        true
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        // No interpreter frame to restore into; the reds live in the compiled
        // loop's registers.
        let _ = (meta, values);
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        vec![sym.len, sym.acc, sym.i, sym.obj]
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        let _ = (sym, meta);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{GreenKey, Type};
    use majit_jitcode::jitcode::BhDescr;
    use majit_metainterp::warmstate::WarmEnterState;
    use majit_metainterp::{JitCode, JitCodeSym, JitState, RuntimeBhDescr};

    /// jd3's extracted main JitCode carries an empty per-jitcode `exec.descrs`;
    /// its `d`/`j` argcodes resolve through the shared global build-time descr
    /// pool (`install_global_build_descr_pool` -> `JitCode::descr_at`),
    /// mirroring RPython's single shared `Assembler.descrs`.
    #[test]
    fn jd3_build_time_descrs_resolve_through_global_pool() {
        let canonical = crate::jitcode_runtime::portal_jitcode_for_key(TUPLEHASH_PORTAL_KEY)
            .expect("jd3's extracted main JitCode must be registered");
        // The extracted body is the walkable hash loop: exactly one merge
        // point, whose byte offset depends on the body's op layout
        // (`TupleHashSym::loop_header_pc` discovers it rather than hardcoding).
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
            "extracted portal carries len, acc, i as red Ints"
        );
        assert_eq!(
            banks.red_r.len(),
            1,
            "extracted portal carries obj as the one object red"
        );
        assert_eq!(banks.red_f.len(), 0);
        assert_eq!(
            banks.red_i,
            vec![0, 1, 2],
            "red I registers are len, acc, i"
        );
        assert_eq!(banks.red_r, vec![1], "red R register is obj");

        crate::jitcode_runtime::install_global_build_descr_pool();
        let jitcode = JitCode::from_canonical((*canonical).clone());
        // A build-time jitcode has no per-jitcode descr pool; every resolution
        // below therefore exercises the global fallback in `descr_at`.
        assert!(jitcode.exec.descrs.is_empty());

        // jd3's first inline-call names its callee in the leading 2-byte `d`
        // operand; `descr_at` must map it to a JitCode through the global
        // pool. Every `inline_call_*` spelling puts `d` first (`dR>r`,
        // `dIR`, `dIRF>f`, ...), so the family is what this reads.
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

        // Every slot in the shared pool resolves, and it carries both the
        // JitCode (inline-call) and Call (residual-call) descr families the
        // jd3 loop body reads back via `descr_at`.
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

    #[test]
    fn tuplehash_resolved_cell_key_reads_the_warmstate_it_is_handed() {
        // eval.rs tuple-hash merge and CALL_ASSEMBLER resolve on
        // jitdrivers_sd[TUPLEHASH_JD_INDEX].warmstate. A colliding typed
        // squatter occupies the bucket so the tuple.hash cell is minted
        // (cell_key != hash). Portal slot 0 has no such cell, so it still
        // answers the raw hash.
        let portal = WarmEnterState::new(1039);
        let mut jd3 = WarmEnterState::new(1039);
        let w_type = 0x51ED as PyObjectRef;
        let typed = tuple_hash_green_key(w_type);
        let hash = typed.get_uhash();
        // Occupy the production hash with a two-Int squatter so `typed`
        // mints. Same get_uhash, different comparekey (`warmstate.py` fold).
        let squatter = colliding_squatter_key(&typed);
        assert_eq!(squatter.get_uhash(), hash);
        assert_ne!(squatter, typed);
        jd3.disable_noninlinable_function_for_key(&squatter);
        let minted = jd3.ensure_cell_key(&typed);
        assert_ne!(
            minted, hash,
            "the squatter keeps the raw hash; the typed cell is minted"
        );
        assert_eq!(
            tuple_hash_resolved_cell_key(&jd3, w_type),
            minted,
            "jd3 comparekey walks to the typed cell"
        );
        assert_eq!(
            tuple_hash_resolved_cell_key(&portal, w_type),
            hash,
            "portal slot 0 has no tuple.hash cell for these greens"
        );
    }

    #[test]
    fn tuplehash_set_redboxes_updates_collect_jump_args() {
        // pyjitpl.py `opimpl_jit_merge_point` reads the live `%I`/`%R`
        // boxes and `reached_loop_header` / `compile_loop` uses
        // `live_arg_boxes = greenboxes + redboxes`. The CloseLoop JUMP is
        // `collect_jump_args`, so `set_redboxes` must replace the portal
        // InputArgs with the loop-carried values the body just wrote.
        let len = OpRef::input_arg_typed(0, Type::Int);
        let acc0 = OpRef::input_arg_typed(1, Type::Int);
        let i0 = OpRef::input_arg_typed(2, Type::Int);
        let obj = OpRef::input_arg_typed(3, Type::Ref);
        let mut sym = TupleHashSym {
            w_type: PY_NULL,
            len,
            acc: acc0,
            i: i0,
            obj,
        };
        assert_eq!(
            TupleHashJitState::collect_jump_args(&sym),
            vec![len, acc0, i0, obj]
        );
        let acc1 = OpRef::input_arg_typed(14, Type::Int);
        let i1 = OpRef::input_arg_typed(15, Type::Int);
        JitCodeSym::set_redboxes(
            &mut sym,
            &[
                (len, Type::Int),
                (acc1, Type::Int),
                (i1, Type::Int),
                (obj, Type::Ref),
            ],
        );
        assert_eq!(
            TupleHashJitState::collect_jump_args(&sym),
            vec![len, acc1, i1, obj]
        );
    }

    #[test]
    fn tuplehash_backedge_copies_new_acc_and_i_into_merge_point_reds() {
        // flatten.py `insert_renamings` on the portal back-edge copies
        // the updated acc/i into the merge-point red registers. The
        // CloseLoop path then reads those registers (`prepare_list_of_boxes`).
        let canonical = crate::jitcode_runtime::portal_jitcode_for_key(TUPLEHASH_PORTAL_KEY)
            .expect("jd3's extracted main JitCode must be registered");
        let ops: Vec<_> = crate::jitcode_runtime::decoded_ops(&canonical.code).collect();
        let merge = ops
            .iter()
            .find(|op| op.opname == "jit_merge_point")
            .expect("jd3 portal body must contain a jit_merge_point");
        let banks = majit_metainterp::decode_jit_merge_point_banks(&canonical.code, merge.pc);
        let acc_reg = banks.red_i[1] as u8;
        let i_reg = banks.red_i[2] as u8;
        let back_edge = ops
            .iter()
            .find(|op| {
                op.opname == "goto"
                    && majit_jitcode::jitcode::read_label_at(&canonical.code, op.pc + 1) <= merge.pc
            })
            .expect("portal loop has a goto back-edge to the merge point");
        let dests: Vec<u8> = ops
            .iter()
            .rev()
            .skip_while(|op| op.pc >= back_edge.pc)
            .take_while(|op| op.opname == "int_copy")
            .map(|op| canonical.code[op.next_pc - 1])
            .collect();
        assert!(
            dests.contains(&acc_reg),
            "back-edge must copy new acc into red I register {acc_reg}, copies={dests:?}"
        );
        assert!(
            dests.contains(&i_reg),
            "back-edge must copy new i into red I register {i_reg}, copies={dests:?}"
        );
    }

    /// Two-Int key whose `get_uhash` equals `typed`'s.
    ///
    /// The multiplicative inverse of `GREEN_UHASH_MULT` undoes the last
    /// `green_uhash_step`, so the tuple.hash cell mints a cell key. The
    /// squatter's shape need not match: the cell chain is bucketed by
    /// hash and filtered by `comparekey`.
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
