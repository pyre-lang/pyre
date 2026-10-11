//! A plain `[int]` read inside a `split_dispatch` arm sub-JitCode.
//!
//! `lower_state_array_read` emits `getarrayitem_raw_i` off the ptr identity
//! register of the *current* JitCode frame. `MIFrame.setup_call` /
//! `fill_registers` and `bhimpl_inline_call_ir_v` give the callee only the
//! declared argument boxes, so that ptr must be a declared int argument of
//! the arm call.

use core::sync::atomic::{AtomicU32, Ordering};
use majit_metainterp::jitcode::insns::{
    BC_GETARRAYITEM_RAW_I, BC_INLINE_CALL, BC_INLINE_CALL_IR_I, BC_INLINE_CALL_IR_V,
};
use majit_metainterp::{Assembler, JitCode, JitDriver};

pub type Bytecode = [u8];

const OP_ADD: u8 = 1;
const OP_LOOP: u8 = 2;
const CELL_LEN: usize = 4;
const N: usize = 40;

struct SplitCells {
    acc: i64,
    cells: Vec<i64>,
    ticks: i64,
}

static COMPILES: AtomicU32 = AtomicU32::new(0);

#[majit_macros::jit_interp(
    state = SplitCells,
    env = Bytecode,
    split_dispatch = true,
    greens = [pc, program],
    state_fields = { acc: int, cells: [int], ticks: int },
)]
#[allow(unused_assignments, unused_variables)]
fn split_cells_sum(program: &Bytecode, threshold: u32, cells: Vec<i64>, ticks: i64) -> i64 {
    let mut driver: JitDriver<SplitCells> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = SplitCells {
        acc: 0,
        cells,
        ticks,
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);
        let opcode = program[pc];
        match opcode {
            OP_ADD => {
                state.acc = state.acc + state.cells[0];
                pc += 1;
            }
            OP_LOOP => {
                state.ticks = state.ticks - 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
                pc += 1;
            }
            _ => break,
        }
    }
    state.acc
}

fn build_dispatch_jitcode() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_split_cells_sum(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_split_cells_sum(&mut asm, 0i64).expect("dispatch lower must succeed")
}

fn dispatch_subs(jc: &JitCode) -> impl Iterator<Item = &JitCode> {
    jc.exec
        .descrs
        .iter()
        .filter_map(|descr| descr.as_jitcode().map(|sub| sub.as_ref()))
}

#[test]
fn split_dispatch_arm_is_a_sub_jitcode_that_reads_cells() {
    let dispatch_jc = build_dispatch_jitcode();
    majit_metainterp::assert_no_degraded_dispatch_arms("SplitCells");

    let dispatch_has_inline = dispatch_jc.code.contains(&BC_INLINE_CALL)
        || dispatch_jc.code.contains(&BC_INLINE_CALL_IR_I)
        || dispatch_jc.code.contains(&BC_INLINE_CALL_IR_V);
    assert!(
        dispatch_has_inline,
        "split_dispatch must emit an arm inline_call; dispatch={:?}",
        dispatch_jc.code
    );

    let readers: Vec<&JitCode> = dispatch_subs(&dispatch_jc)
        .filter(|code| code.code.contains(&BC_GETARRAYITEM_RAW_I))
        .collect();
    assert!(
        !readers.is_empty(),
        "the OP_ADD arm sub-JitCode must emit getarrayitem_raw_i; dispatch={:?} subs={:?}",
        dispatch_jc.code,
        dispatch_subs(&dispatch_jc)
            .map(|c| &c.code)
            .collect::<Vec<_>>()
    );
}

#[test]
fn compiled_split_dispatch_cells_read_matches_the_interpreter() {
    let cells = vec![7i64; CELL_LEN];
    let program = vec![OP_ADD, OP_LOOP];
    let expected = cells[0] * N as i64;

    let interp = split_cells_sum(&program, u32::MAX, cells.clone(), N as i64);
    assert_eq!(
        interp, expected,
        "interpreter adds cells[0] once per loop trip"
    );

    COMPILES.store(0, Ordering::Relaxed);
    let compiled = split_cells_sum(&program, 2, cells, N as i64);
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "the split_dispatch loop must compile"
    );
    assert_eq!(
        compiled, interp,
        "compiled sub-JitCode state.cells[0] must match the interpreter"
    );
}
