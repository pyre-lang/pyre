//! A loop-carried store to a plain `[int]` writes the native buffer in
//! place (`jtransform.py` `rewrite_op_setarrayitem` raw arm:
//! `setarrayitem_raw_i`). Compiled execution must match the interpreter
//! on both the result and the final `cells` contents.
//!
//! Before the store lowering, `validate_state_fields` refused this
//! program at expansion.

use core::sync::atomic::{AtomicU32, Ordering};
use majit_metainterp::jitcode::insns::BC_SETARRAYITEM_RAW_I;
use majit_metainterp::{Assembler, JitCode, JitDriver};

pub type Bytecode = [u8];

const PROGRAM: [u8; 1] = [0];
const N: i64 = 40;

struct StoreCells {
    cells: Vec<i64>,
    i: i64,
    n: i64,
    len: i64,
}

static COMPILES: AtomicU32 = AtomicU32::new(0);

fn seed_cells() -> Vec<i64> {
    vec![3i64, 5, 7, 11]
}

/// `cells[i] = cells[i] + cells[(i + 1) % len]`, then advance `i`.
fn expected_cells(mut cells: Vec<i64>, n: i64) -> Vec<i64> {
    let len = cells.len();
    let mut i = 0usize;
    for _ in 0..n {
        let next = (i + 1) % len;
        cells[i] += cells[next];
        i = next;
    }
    cells
}

fn cells_fingerprint(cells: &[i64]) -> i64 {
    cells.iter().enumerate().fold(0i64, |h, (k, &c)| {
        h.wrapping_mul(1_000_003)
            .wrapping_add(c.wrapping_mul((k as i64) + 1))
    })
}

#[majit_macros::jit_interp(
    state = StoreCells,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { cells: [int], i: int, n: int, len: int },
)]
#[allow(unused_assignments, unused_variables)]
fn neighbor_add(program: &Bytecode, threshold: u32, cells: Vec<i64>, n: i64) -> i64 {
    let mut driver: JitDriver<StoreCells> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = StoreCells {
        len: cells.len() as i64,
        cells,
        i: 0,
        n,
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while state.n > 0 {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        let i = state.i as usize;
        let next = state.i + 1i64;
        let next = if next < state.len { next } else { 0i64 };
        state.cells[i] = state.cells[i] + state.cells[next as usize];
        state.i = next;
        state.n = state.n - 1i64;
    }
    cells_fingerprint(&state.cells)
}

fn build_dispatch_jitcode() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_neighbor_add(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_neighbor_add(&mut asm, 0i64).expect("dispatch lower must succeed")
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
fn a_loop_carried_plain_int_store_emits_setarrayitem_raw_i() {
    let dispatch_jc = build_dispatch_jitcode();
    majit_metainterp::assert_no_degraded_dispatch_arms("StoreCells");
    let writers: Vec<&JitCode> = dispatch_tree(&dispatch_jc)
        .filter(|code| code.code.contains(&BC_SETARRAYITEM_RAW_I))
        .collect();
    assert!(
        !writers.is_empty(),
        "the traced body must emit setarrayitem_raw_i; dispatch={:?}",
        dispatch_jc.code
    );
}

#[test]
fn compiled_plain_int_array_store_matches_the_interpreter() {
    let cells = seed_cells();
    let expected = expected_cells(cells.clone(), N);
    let expected_fp = cells_fingerprint(&expected);

    let interp = neighbor_add(&PROGRAM, u32::MAX, cells.clone(), N);
    assert_eq!(
        interp, expected_fp,
        "interpreter neighbor-add must match the plain walk"
    );

    COMPILES.store(0, Ordering::Relaxed);
    let compiled = neighbor_add(&PROGRAM, 2, cells, N);
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "the loop must compile"
    );
    assert_eq!(
        compiled, interp,
        "compiled neighbor-add must match the interpreter on the cells fingerprint"
    );
}
