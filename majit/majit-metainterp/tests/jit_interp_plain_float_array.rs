//! A plain `[float]` (`Vec<f64>`) is ptr then len, the same two INT
//! identity reds as `[int]`. `jtransform.py` `rewrite_op_getarrayitem`
//! selects `getarrayitem_raw_f` from `getkind(ARRAY.OF)`. The loop
//! compiles and the result matches the interpreter bit-for-bit.
//!
//! Before the validator accepted `[float]`, this program failed to
//! expand (`unsupported: cells: [float]`).

use core::sync::atomic::{AtomicU32, Ordering};
use majit_metainterp::jitcode::insns::BC_GETARRAYITEM_RAW_F;
use majit_metainterp::{Assembler, JitCode, JitDriver};

pub type Bytecode = [u8];

const PROGRAM: [u8; 1] = [0];
const N: i64 = 40;

struct FloatCells {
    acc: f64,
    cells: Vec<f64>,
    i: i64,
    n: i64,
    len: i64,
}

static COMPILES: AtomicU32 = AtomicU32::new(0);

fn seed_cells() -> Vec<f64> {
    vec![0.25, 0.5, 0.75, 1.0]
}

fn expected_sum(cells: &[f64], n: i64) -> f64 {
    let mut acc = 0.0f64;
    let mut i = 0usize;
    for _ in 0..n {
        acc += cells[i];
        i += 1;
        if i == cells.len() {
            i = 0;
        }
    }
    acc
}

#[majit_macros::jit_interp(
    state = FloatCells,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: float, cells: [float], i: int, n: int, len: int },
)]
#[allow(unused_assignments, unused_variables)]
fn cells_sum(program: &Bytecode, threshold: u32, cells: Vec<f64>, n: i64) -> i64 {
    let mut driver: JitDriver<FloatCells> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = FloatCells {
        acc: 0.0,
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
        state.acc = state.acc + state.cells[i];
        let next = state.i + 1i64;
        state.i = if next < state.len { next } else { 0i64 };
        state.n = state.n - 1i64;
    }
    state.acc.to_bits() as i64
}

fn build_dispatch_jitcode() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![0]);
    __prebuild_jitcode_liveness_cells_sum(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_cells_sum(&mut asm, 0i64).expect("dispatch lower must succeed")
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
fn a_plain_float_read_emits_getarrayitem_raw_f() {
    let dispatch_jc = build_dispatch_jitcode();
    majit_metainterp::assert_no_degraded_dispatch_arms("FloatCells");
    let readers: Vec<&JitCode> = dispatch_tree(&dispatch_jc)
        .filter(|code| code.code.contains(&BC_GETARRAYITEM_RAW_F))
        .collect();
    assert!(
        !readers.is_empty(),
        "the traced body must emit getarrayitem_raw_f; dispatch={:?}",
        dispatch_jc.code
    );
}

#[test]
fn compiled_plain_float_array_sum_matches_the_interpreter() {
    let cells = seed_cells();
    let expected = expected_sum(&cells, N).to_bits() as i64;

    let interp = cells_sum(&PROGRAM, u32::MAX, cells.clone(), N);
    assert_eq!(
        interp, expected,
        "interpreter cells sum must match the plain walk bit-for-bit"
    );

    COMPILES.store(0, Ordering::Relaxed);
    let compiled = cells_sum(&PROGRAM, 2, cells, N);
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "the loop must compile"
    );
    assert_eq!(
        compiled, interp,
        "compiled cells sum must match the interpreter bit-for-bit"
    );
}
