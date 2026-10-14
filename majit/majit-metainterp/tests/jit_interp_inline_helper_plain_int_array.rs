//! A JitCode whose only state access is `state.cells[i]`.
//!
//! A plain `[int]` read lowers to `getarrayitem_raw_i` off the ptr identity
//! slot, not to `load_state_field`. `MIFrame.setup_call` /
//! `bhimpl_inline_call_ir_v` give the callee only the declared argument boxes,
//! so the ptr (and len) must be explicit int arguments of the inline call.

use majit_metainterp::jitcode::insns::{BC_GETARRAYITEM_RAW_I, BC_LOAD_STATE_FIELD};
use majit_metainterp::{Assembler, JitCode, JitCodeBuilder, JitDriver};

#[cfg(all(feature = "cranelift", not(feature = "dynasm")))]
use majit_backend_cranelift::CraneliftBackend as TestBackend;
#[cfg(feature = "dynasm")]
use majit_backend_dynasm::runner::DynasmBackend as TestBackend;

pub type Bytecode = [u8];

const OP_READ: u8 = 1;

struct CellReadState {
    cells: Vec<i64>,
}

#[majit_macros::jit_interp(
    state = CellReadState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { cells: [int] },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_cell_read(program: &Bytecode, threshold: u32, cells: Vec<i64>) -> i64 {
    let mut driver: JitDriver<CellReadState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = CellReadState { cells };
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
            OP_READ => return state.cells[0],
            _ => break,
        }
    }
    0
}

fn build_dispatch_jitcode() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_dispatch_cell_read(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_dispatch_cell_read(&mut asm, 0i64).expect("dispatch lower must succeed")
}

fn dispatch_tree(jc: &JitCode) -> impl Iterator<Item = &JitCode> {
    std::iter::once(jc).chain(
        jc.exec
            .descrs
            .iter()
            .filter_map(|descr| descr.as_jitcode().map(|sub| sub.as_ref())),
    )
}

#[test]
fn a_cells_only_lowered_jitcode_emits_getarrayitem_raw_i() {
    let dispatch_jc = build_dispatch_jitcode();
    majit_metainterp::assert_no_degraded_dispatch_arms("CellReadState");

    let readers: Vec<&JitCode> = dispatch_tree(&dispatch_jc)
        .filter(|code| code.code.contains(&BC_GETARRAYITEM_RAW_I))
        .collect();
    assert!(
        !readers.is_empty(),
        "the OP_READ arm must emit getarrayitem_raw_i"
    );
    for code in &readers {
        assert!(
            !code.code.contains(&BC_LOAD_STATE_FIELD),
            "the body's only state access is state.cells[i]; code={:?}",
            code.code
        );
    }
}

#[test]
fn interpreter_cells_read_matches_the_seeded_value() {
    let cells = vec![11i64, 22, 33];
    let program = [OP_READ];
    let got = dispatch_cell_read(&program, 0, cells.clone());
    assert_eq!(got, cells[0]);
}

/// The lowerer inlines a `return state.cells[i]` arm into the dispatch
/// JitCode (`try_inline_dispatch_arm`). This helper is the same emit the
/// lowerer writes — `getarrayitem_raw_i` off the ptr identity slot — and
/// the parent passes ptr/len as declared int arguments (`MIFrame.setup_call`,
/// `bhimpl_inline_call_ir_i`). `inline_call_grouped` requires a positional
/// I-list, so the identity slots at i1/i2 are preceded by i0.
#[test]
fn blackhole_inline_call_passes_cells_ptr_as_int_args() {
    let cells = vec![11i64, 22, 33];
    let expected = cells[0];

    let mut helper = JitCodeBuilder::new();
    helper.ensure_i_regs(4);
    helper.load_const_i_value(3, 0);
    let descr_idx = helper.add_raw_int_array_descr_signed(std::mem::size_of::<i64>(), true);
    helper.getarrayitem_raw_i(0, 1, 3, descr_idx);
    helper.int_return(0);
    let helper = helper.finish();
    assert!(
        !helper.reads_identity_slots(),
        "a cells-only helper has no load_state_field"
    );

    let mut parent = JitCodeBuilder::new();
    parent.ensure_i_regs(4);
    let sub_idx = parent.add_sub_jitcode(helper);
    // greens = [pc, program]: pc occupies i0, cells ptr/len sit at i1/i2.
    parent.inline_call_ir_i(sub_idx, &[(0, 0), (1, 1), (2, 2)], &[], Some(0));
    parent.int_return(0);
    let parent_jc = std::sync::Arc::new(parent.finish());

    let mut bh_insns: indexmap::IndexMap<String, u8> =
        majit_metainterp::jitcode::wellknown_bh_insns()
            .iter()
            .map(|(key, value)| ((*key).to_string(), *value))
            .collect();
    bh_insns.extend(
        majit_metainterp::jitcode::extension_insns()
            .iter()
            .map(|(key, value)| ((*key).to_string(), *value)),
    );
    let cpu = TestBackend::new();
    let mut bh_builder = majit_metainterp::blackhole::build_inline_call_only_bh_builder(&[(
        "recursive_call_v/iIRFIRF",
        majit_metainterp::jitcode::insns::BC_RECURSIVE_CALL_VOID,
    )]);
    bh_builder.set_cpu(&cpu);
    bh_builder.setup_insns(&bh_insns);
    bh_builder.setup_cached_control_opcodes(
        majit_metainterp::jitcode::insns::BC_LIVE as i32,
        majit_metainterp::jitcode::insns::BC_CATCH_EXCEPTION as i32,
        majit_metainterp::jitcode::insns::BC_RVMPROF_CODE as i32,
    );
    majit_metainterp::blackhole::wire_bhimpl_handlers(&mut bh_builder);
    let mut bh = bh_builder.acquire_interp();
    bh.setposition(parent_jc, 0);
    // greens = [pc, program]: pc occupies i0, cells ptr/len sit at i1/i2.
    bh.state_field_layout = majit_metainterp::blackhole::StateFieldLayout::new(0, 1, 0, 1);
    bh.registers_i[1] = cells.as_ptr() as i64;
    bh.registers_i[2] = cells.len() as i64;
    let _ = bh.run();
    assert_eq!(
        bh.tmpreg_i, expected,
        "declared inline_call int args must plant the cells pointer so \
         getarrayitem_raw_i reads cells[0]"
    );
}
