//! Float state-field lowering shape tests.

pub type Bytecode = [u8];

fn all_jitcode_bodies(dispatch_jc: &majit_metainterp::JitCode) -> Vec<Vec<u8>> {
    let mut bodies = vec![dispatch_jc.code.clone()];
    bodies.extend(
        dispatch_jc
            .exec
            .descrs
            .iter()
            .filter_map(|descr| descr.as_jitcode())
            .map(|sub| sub.code.clone()),
    );
    bodies
}

mod scalar {
    use super::{Bytecode, all_jitcode_bodies};
    use majit_metainterp::jitcode::insns::{BC_LOAD_STATE_FIELD_FLOAT, BC_STORE_STATE_FIELD_FLOAT};
    use majit_metainterp::{Assembler, JitDriver};

    struct FloatScalarState {
        a: i64,
        f: f64,
    }

    const OP_NOP: u8 = 0;
    const OP_TOUCH_F: u8 = 1;

    #[majit_macros::jit_interp(
        state = FloatScalarState,
        env = Bytecode,
        state_fields = { a: int, f: float },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn float_scalar_minimal(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<FloatScalarState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = FloatScalarState { a: 0, f: 0.0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                OP_TOUCH_F => state.f += 1.5,
                _ => break,
            }
        }
        state.a
    }

    #[test]
    fn float_scalar_lowers_load_store_and_live_f() {
        use majit_metainterp::JitState as _;

        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![1], vec![], vec![0]);
        __prebuild_jitcode_liveness_float_scalar_minimal(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_float_scalar_minimal(&mut asm, 0i64)
            .expect("dispatch lower must succeed for float scalar fixture");
        let bodies = all_jitcode_bodies(&dispatch_jc);
        assert!(
            bodies
                .iter()
                .any(|body| body.contains(&BC_LOAD_STATE_FIELD_FLOAT)),
            "float read must lower to load_state_field_float; bodies: {bodies:?}"
        );
        assert!(
            bodies
                .iter()
                .any(|body| body.contains(&BC_STORE_STATE_FIELD_FLOAT)),
            "float write must lower to store_state_field_float; bodies: {bodies:?}"
        );

        let mut state = FloatScalarState { a: 7, f: 3.5 };
        let meta = state.build_meta(0, &[OP_NOP]);
        let live = state.extract_live(&meta);
        let types = state.live_value_types(&meta);
        let (_live_i, _live_r, live_f) = meta.canonical_liveness_slots();
        assert_eq!(live, vec![7, 3.5f64.to_bits() as i64]);
        assert_eq!(types, vec![majit_ir::Type::Int, majit_ir::Type::Float]);
        assert_eq!(live_f, vec![0]);
    }
}

// Finding F reproduction: a float state write positioned as a top-level
// dispatch-loop statement (not an opcode-match arm body) is filtered by
// `lower_dispatch_body`'s `stmt_modifies_jit_state` skip gate. Until float
// scalars were added to `expr_is_jit_state_place`, that predicate returned
// false for `state.f`, so the write was silently dropped from the dispatch
// JitCode and compiled execution left the field stale. The module groups the
// fixture; it is no longer forced, because `__JitSym_<fn>` / `__JitMeta_<fn>`
// now carry the annotated function's name.
mod scalar_toplevel {
    use super::{Bytecode, all_jitcode_bodies};
    use majit_metainterp::jitcode::insns::BC_STORE_STATE_FIELD_FLOAT;
    use majit_metainterp::{Assembler, JitDriver};

    struct FloatToplevelState {
        a: i64,
        f: f64,
    }

    const OP_NOP: u8 = 0;

    #[majit_macros::jit_interp(
        state = FloatToplevelState,
        env = Bytecode,
        state_fields = { a: int, f: float },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn float_scalar_toplevel_write(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<FloatToplevelState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = FloatToplevelState { a: 0, f: 0.0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            // Top-level float write — reaches the `stmt_modifies_jit_state`
            // gate in `lower_dispatch_body` rather than the arm-inline path.
            state.f += 1.5;
            match opcode {
                OP_NOP => {}
                _ => break,
            }
        }
        state.a
    }

    #[test]
    fn toplevel_float_write_is_not_dropped_from_dispatch_body() {
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![1], vec![], vec![0]);
        __prebuild_jitcode_liveness_float_scalar_toplevel_write(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_float_scalar_toplevel_write(&mut asm, 0i64)
            .expect("dispatch lower must succeed for the top-level float write fixture");
        let bodies = all_jitcode_bodies(&dispatch_jc);
        assert!(
            bodies
                .iter()
                .any(|body| body.contains(&BC_STORE_STATE_FIELD_FLOAT)),
            "a top-level float write must lower to store_state_field_float; bodies: {bodies:?}"
        );
    }
}

mod virt_array {
    use super::{Bytecode, all_jitcode_bodies};
    use majit_metainterp::jitcode::insns::{BC_GETARRAYITEM_VABLE_F, BC_SETARRAYITEM_VABLE_F};
    use majit_metainterp::virt_array::VirtArray;
    use majit_metainterp::{Assembler, JitDriver};

    struct FloatArrayState {
        regs: VirtArray<f64>,
    }

    const OP_NOP: u8 = 0;
    const OP_TOUCH_REGS: u8 = 2;

    #[majit_macros::jit_interp(
        state = FloatArrayState,
        env = Bytecode,
        state_fields = { regs: [float; virt] },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn float_array_minimal(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<FloatArrayState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = FloatArrayState {
            regs: VirtArray::filled(0.0, 2),
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                OP_TOUCH_REGS => state.regs[0] += 1.25,
                _ => break,
            }
        }
        0
    }

    #[test]
    fn float_virt_array_uses_float_vinfo_and_vable_ops() {
        let info = <FloatArrayState as majit_metainterp::JitState>::__build_virtualizable_info()
            .expect("float virt array should build vinfo");
        assert_eq!(info.array_fields.len(), 1);
        assert_eq!(info.array_fields[0].item_type, majit_ir::Type::Float);

        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![1, 2], vec![1], vec![]);
        __prebuild_jitcode_liveness_float_array_minimal(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_float_array_minimal(&mut asm, 0i64)
            .expect("dispatch lower must succeed for float virt array fixture");
        let bodies = all_jitcode_bodies(&dispatch_jc);
        assert!(
            bodies
                .iter()
                .any(|body| body.contains(&BC_GETARRAYITEM_VABLE_F)),
            "float virt array read must lower to getarrayitem_vable_f; bodies: {bodies:?}"
        );
        assert!(
            bodies
                .iter()
                .any(|body| body.contains(&BC_SETARRAYITEM_VABLE_F)),
            "float virt array write must lower to setarrayitem_vable_f; bodies: {bodies:?}"
        );
    }
}

// Finding E: a `float(f32)` scalar declares an f32 struct field. Extraction
// took `self.f.to_bits()` (32-bit f32 bits), but the restore path reads it
// back with `f64::from_bits(_ as u64) as f32`, so the value was corrupted.
// Extraction/init now widen to f64 first, matching restore.
mod scalar_f32 {
    use super::Bytecode;
    use majit_metainterp::JitDriver;

    struct F32State {
        a: i64,
        f: f32,
    }

    const OP_NOP: u8 = 0;

    #[majit_macros::jit_interp(
        state = F32State,
        env = Bytecode,
        state_fields = { a: int, f: float(f32) },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn f32_scalar_minimal(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<F32State> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = F32State { a: 0, f: 0.0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                _ => break,
            }
        }
        state.a
    }

    #[test]
    fn f32_scalar_extracts_f64_widened_bits() {
        use majit_metainterp::JitState as _;
        let mut state = F32State { a: 0, f: 1.5f32 };
        let meta = state.build_meta(0, &[OP_NOP]);
        let live = state.extract_live(&meta);
        // The f32 field is encoded as its f64 widening, so the restore path
        // `f64::from_bits(_ as u64) as f32` recovers the original value.
        assert_eq!(live, vec![0, (1.5f32 as f64).to_bits() as i64]);
        // Guard against the pre-fix encoding, which stored the raw 32-bit
        // f32 bit pattern and round-tripped to a bogus f64.
        assert_ne!(live[1], 1.5f32.to_bits() as i64);
    }
}

// Finding D: with two float scalars {a, b} their canonical identity slots
// are float f0/f1. `alloc_reg` draws float working registers from the flat
// `next_reg`, which was floored past only the int/ref identity ends. For a
// float-only state those ends (int end = 1, ref end = 0) are below the float
// end (2), so a read of `state.a` allocated a temp at f1 — `b`'s identity
// slot — and `load_state_field_float`'s register-bank copy on blackhole
// resume overwrote `b`. Flooring `next_reg` past `float_identity_end()` too
// keeps float temps above the reserved prefix.
mod scalar_float_slot_reserve {
    use super::{Bytecode, all_jitcode_bodies};
    use majit_metainterp::jitcode::insns::BC_LOAD_STATE_FIELD_FLOAT;
    use majit_metainterp::{Assembler, JitDriver};

    struct TwoFloatState {
        a: f64,
        b: f64,
    }

    const OP_NOP: u8 = 0;
    const OP_MIX: u8 = 1;

    #[majit_macros::jit_interp(
        state = TwoFloatState,
        env = Bytecode,
        state_fields = { a: float, b: float },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn two_float_minimal(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<TwoFloatState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = TwoFloatState { a: 0.0, b: 0.0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                // Reads `a` into a float temp, then writes `b`.
                OP_MIX => state.b = state.a + 1.5,
                _ => break,
            }
        }
        0
    }

    #[test]
    fn float_temp_does_not_alias_a_sibling_identity_slot() {
        // Two float scalars → identity slots f0 (a) and f1 (b);
        // `float_identity_end()` == 2. Every `load_state_field_float` dest is
        // a working register that must sit at or above that end.
        const FLOAT_IDENTITY_END: u8 = 2;
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![], vec![], vec![0, 1]);
        __prebuild_jitcode_liveness_two_float_minimal(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_two_float_minimal(&mut asm, 0i64)
            .expect("dispatch lower must succeed for the two-float fixture");
        let bodies = all_jitcode_bodies(&dispatch_jc);
        // `load_state_field_float/df`: [opcode][field_idx:u16 LE][dest:u8].
        // Match only a plausible field index (high byte 0, index < 2) so an
        // operand byte that happens to equal the opcode is not misread.
        let mut seen_load = false;
        for body in &bodies {
            let mut i = 0;
            while i + 3 < body.len() {
                if body[i] == BC_LOAD_STATE_FIELD_FLOAT && body[i + 2] == 0 && body[i + 1] < 2 {
                    let dest = body[i + 3];
                    assert!(
                        dest >= FLOAT_IDENTITY_END,
                        "load_state_field_float dest f{dest} aliases a reserved \
                         float identity slot [0, {FLOAT_IDENTITY_END}); body: {body:?}"
                    );
                    seen_load = true;
                }
                i += 1;
            }
        }
        assert!(
            seen_load,
            "fixture must emit at least one load_state_field_float; bodies: {bodies:?}"
        );
    }
}

// The virtualizable's element boxes are a strict SUFFIX of the loop-carried
// list upstream: `live_arg_boxes = greenboxes + redboxes` and then
// `live_arg_boxes += self.virtualizable_boxes; live_arg_boxes.pop()`
// (pyjitpl.py:2981-2989). `+=` appends, so no element can precede a red.
//
// The entry contract is built the same way — `JitDriver::extend_compiled_live_values`
// does `live_values.extend(extra_values)` after every red. The closing JUMP
// spliced the elements between the vable identity and the ref/float scalars, so
// for a state declaring BOTH a `[.. ; virt]` array and a ref or float scalar the
// two sides carried the same arity with different slot meanings —
// `jump.numargs() == label.numargs()` (compile.py:334) passes and nothing
// downstream catches it. Pin the suffix relation directly.
mod virt_array_with_float_scalar {
    use super::Bytecode;
    use majit_metainterp::virt_array::VirtArray;
    use majit_metainterp::{JitDriver, JitState, decode_jit_merge_point_banks};

    struct MixedState {
        sp: i64,
        cells: Vec<i64>,
        acc: f64,
        stack: VirtArray<i64>,
    }

    const OP_NOP: u8 = 0;
    const OP_STEP: u8 = 1;

    #[majit_macros::jit_interp(
        state = MixedState,
        env = Bytecode,
        // The fixed `[int]` array is what makes this shape reachable: a state
        // that is exactly one virt array plus a float scalar is already refused
        // by the recursive-portal fresh-allocation gate, so the ordering bug
        // could only be hit alongside a fixed array or a ref scalar.
        state_fields = { sp: int, cells: [int], acc: float, stack: [int; virt] },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn mixed_virt_and_float(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<MixedState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = MixedState {
            sp: 0,
            cells: vec![0; 2],
            acc: 0.0,
            stack: VirtArray::filled(0, 2),
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                OP_STEP => {
                    state.stack[0] += 1;
                    state.acc += 1.5;
                    // Read-only: a MUTATED plain `[int]` is refused outright
                    // (it is not restored on deopt), so the fixed array is
                    // loop-carried but never written.
                    state.sp = state.sp + 1 + state.cells[0];
                }
                _ => break,
            }
        }
        state.sp
    }

    #[test]
    fn element_boxes_are_a_strict_suffix_of_the_reds() {
        let state = MixedState {
            sp: 0,
            cells: vec![0; 2],
            acc: 0.0,
            stack: VirtArray::filled(0, 2),
        };
        let program: &Bytecode = &[OP_NOP, OP_STEP];
        let meta = state.build_meta(0, program);
        let sym = <MixedState as JitState>::create_sym(&meta, 0);

        // `TraceCtx::collect_virtualizable_typed_boxes()` shape: the element
        // block in per-array order, identity LAST.
        let e0 = majit_ir::OpRef::input_arg_typed(90, majit_ir::Type::Int);
        let e1 = majit_ir::OpRef::input_arg_typed(91, majit_ir::Type::Int);
        let identity = majit_ir::OpRef::input_arg_typed(92, majit_ir::Type::Ref);
        let boxes = [
            (e0, majit_ir::Type::Int),
            (e1, majit_ir::Type::Int),
            (identity, majit_ir::Type::Ref),
        ];

        let reds = <MixedState as JitState>::collect_jump_args(&sym);
        let close = <MixedState as JitState>::collect_jump_args_with_boxes(&sym, &boxes);

        assert!(
            reds.is_empty(),
            "plain reds live in the portal frame, not on __JitSym"
        );
        assert_eq!(
            close,
            vec![identity, e0, e1],
            "without a portal frame the JUMP is the vable identity plus the \
             element suffix (`live_arg_boxes += virtualizable_boxes; pop`)"
        );
    }

    #[test]
    fn closing_jump_from_portal_includes_scalar_reds() {
        // Without a portal frame, `collect_jump_args_with_boxes` is the
        // vable identity plus the element suffix (`live_arg_boxes +=
        // virtualizable_boxes; pop`). Marker reds (cells) come from the
        // `jit_merge_point` operands.
        let state = MixedState {
            sp: 0,
            cells: vec![0; 2],
            acc: 0.0,
            stack: VirtArray::filled(0, 2),
        };
        let program: &Bytecode = &[OP_NOP, OP_STEP];
        let meta = state.build_meta(0, program);
        let sym = <MixedState as JitState>::create_sym(&meta, 0);

        let e0 = majit_ir::OpRef::input_arg_typed(90, majit_ir::Type::Int);
        let e1 = majit_ir::OpRef::input_arg_typed(91, majit_ir::Type::Int);
        let identity = majit_ir::OpRef::input_arg_typed(92, majit_ir::Type::Ref);
        let boxes = [
            (e0, majit_ir::Type::Int),
            (e1, majit_ir::Type::Int),
            (identity, majit_ir::Type::Ref),
        ];

        let close = <MixedState as JitState>::collect_jump_args_with_boxes(&sym, &boxes);
        assert_eq!(close, vec![identity, e0, e1]);
    }

    #[test]
    fn merge_point_marker_lists_plain_int_array_as_one_red() {
        // Marker lists the `[int]` items-pointer then length
        // (`handle_jit_marker__jit_merge_point` / `make_three_lists`).
        // The `-live-` before the merge point keeps both so a
        // guard resume restores them. `cells.len() == 3` so a missing
        // length slot cannot be mistaken for a per-item red.
        let state = MixedState {
            sp: 0,
            cells: vec![0; 3],
            acc: 0.0,
            stack: VirtArray::filled(0, 2),
        };
        let program: &Bytecode = &[OP_NOP, OP_STEP];
        let mut driver: JitDriver<MixedState> = JitDriver::new(1);
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
        let jc = driver
            .dispatch_jitcode()
            .expect("install_canonical_liveness registers the dispatch JitCode");
        let mp = jc
            .exec
            .jit_merge_point_offset
            .expect("dispatch JitCode has a jit_merge_point");
        let banks = decode_jit_merge_point_banks(&jc.code, mp);
        assert!(
            banks.red_i.contains(&1),
            "marker lists the [int] items-pointer at int_identity_base; reds_i={:?}",
            banks.red_i
        );
        assert!(
            banks.red_i.contains(&2),
            "marker lists the [int] length after the pointer; reds_i={:?}",
            banks.red_i
        );
        assert!(
            !banks.red_i.contains(&3),
            "cells are not per-item reds; reds_i={:?}",
            banks.red_i
        );

        const SIZE_LIVE_OP: usize = 3;
        let live_pc = mp
            .checked_sub(SIZE_LIVE_OP)
            .expect("op3 -live- precedes jit_merge_point");
        let meta = driver.meta_interp();
        let all_liveness = meta.staticdata.liveness_info.snapshot_vec();
        let op_live = meta.staticdata.op_live as u8;
        let indices = majit_metainterp::resume::read_frame_liveness_reg_indices(
            jc,
            live_pc,
            op_live,
            &all_liveness,
        );
        assert!(
            indices.int.iter().any(|&r| r == 1),
            "guard resume live set includes the [int] items-pointer (int_regs[1]); live_i={:?}",
            indices.int
        );
        assert!(
            indices.int.iter().any(|&r| r == 2),
            "guard resume live set includes the [int] length (int_regs[2]); live_i={:?}",
            indices.int
        );
    }

    /// The entry contract this suffix has to match: `live_value_types` is the
    /// reds only, and `JitDriver::extend_compiled_live_values` appends the
    /// elements after all of them.
    #[test]
    fn the_entry_contract_carries_no_element_in_its_red_block() {
        let mut state = MixedState {
            sp: 0,
            cells: vec![0; 2],
            acc: 0.0,
            stack: VirtArray::filled(0, 2),
        };
        let program: &Bytecode = &[OP_NOP, OP_STEP];
        let meta = state.build_meta(0, program);
        // `virtualizable.py VirtualizableInfo.__init__`: `sp` and `acc` sit
        // on the virtualizable with `stack`, so they are not independent
        // JitDriver reds. The red block is the `[int]` items-pointer then
        // length plus the one identity, independent of `cells.len()`.
        assert_eq!(
            state.live_value_types(&meta),
            vec![
                majit_ir::Type::Int, // cells items-pointer
                majit_ir::Type::Int, // cells length
                majit_ir::Type::Ref, // vable identity
            ],
            "no `stack` element belongs in the red block",
        );
        let live = state.extract_live(&meta);
        assert_eq!(live.len(), 3);
        assert_eq!(
            live[0],
            state.cells.as_ptr() as i64,
            "the [int] first red is the Vec items pointer"
        );
        assert_eq!(
            live[1],
            state.cells.len() as i64,
            "the [int] second red is the Vec length"
        );
    }

    #[test]
    fn create_sym_holds_layout_not_red_inputargs() {
        let state = MixedState {
            sp: 7,
            cells: vec![11, 13],
            acc: 1.5,
            stack: VirtArray::filled(0, 2),
        };
        let program: &Bytecode = &[OP_NOP, OP_STEP];
        let meta = state.build_meta(0, program);
        let mut sym = <MixedState as JitState>::create_sym(&meta, 0);
        state.initialize_sym(&mut sym, &meta);

        // Plain reds live in the portal frame, including the `[int]`
        // length (`extract_live` ptr then len). `__JitSym` keeps only
        // virt-array length mirrors (fresh-callee capacity);
        // `warmstate.py` `maybe_compile_and_run` does not copy a plain
        // `[int]` length onto the dispatcher handle.
        assert_eq!(sym.stack_len_value, 2);
        assert_eq!(
            <MixedState as JitState>::count_bound_sym_inputargs(&sym),
            Some(0)
        );
    }
}
