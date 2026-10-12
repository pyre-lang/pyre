//! A compiled plain `[int]` read at the last valid index matches the
//! interpreter. `rlist.py` `rtype_getitem` uses `dum_nocheck`; the JIT
//! sees the bare `getarrayitem_raw_i` item read.

use core::sync::atomic::{AtomicU32, Ordering};
use majit_metainterp::JitDriver;

pub type Bytecode = [u8];

const PROGRAM: [u8; 1] = [0];
const CELL_LEN: usize = 8;
const LAST: i64 = 42;
const N: i64 = 20;

fn cells_last_distinct() -> Vec<i64> {
    let mut cells = vec![1i64; CELL_LEN];
    cells[CELL_LEN - 1] = LAST;
    cells
}

struct LastCells {
    acc: i64,
    cells: Vec<i64>,
    pos: i64,
    n: i64,
}

static COMPILES: AtomicU32 = AtomicU32::new(0);

#[majit_macros::jit_interp(
    state = LastCells,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, cells: [int], pos: int, n: int },
)]
#[allow(unused_assignments, unused_variables)]
fn last_cells_sum(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<LastCells> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = LastCells {
        acc: 0,
        cells: cells_last_distinct(),
        pos: 0,
        n,
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while state.pos < state.n {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        state.acc = state.acc + state.cells[CELL_LEN - 1];
        state.pos = state.pos + 1i64;
    }
    state.acc
}

#[test]
fn compiled_plain_int_array_last_index_matches_the_interpreter() {
    let state = LastCells {
        acc: 0,
        cells: cells_last_distinct(),
        pos: 0,
        n: N,
    };
    let mut driver: JitDriver<LastCells> = JitDriver::new(2);
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, &PROGRAM)
            .install_canonical_liveness(&mut driver);
    }
    assert!(
        driver.dispatch_jitcode().is_some(),
        "plain [int] ptr+len identity must install a dispatch JitCode"
    );

    let interp = last_cells_sum(&PROGRAM, u32::MAX, N);
    assert_eq!(interp, N * LAST, "interpreter reads cells[len - 1]");

    COMPILES.store(0, Ordering::Relaxed);
    let compiled = last_cells_sum(&PROGRAM, 2, N);
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "the loop must compile"
    );
    assert_eq!(
        compiled, interp,
        "compiled last-index read must match the interpreter"
    );
}
