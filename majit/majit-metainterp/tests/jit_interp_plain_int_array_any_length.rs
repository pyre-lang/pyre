//! A plain `[int]` of any runtime length is ptr then len
//! (`jtransform.py` `rewrite_op_getarrayitem` raw arm: the Vec / slice
//! element buffer is a raw `{ptr, len}` view). The dispatch JitCode installs
//! and the hot loop compiles past the old 16-cell identity reserve.

use core::sync::atomic::{AtomicU32, Ordering};
use majit_metainterp::JitDriver;

pub type Bytecode = [u8];

const PROGRAM: [u8; 1] = [0];
const CELL_LEN: usize = 20;
const N: i64 = 40;

struct OversizedCells {
    acc: i64,
    cells: Vec<i64>,
    pos: i64,
    n: i64,
}

static COMPILES: AtomicU32 = AtomicU32::new(0);

#[majit_macros::jit_interp(
    state = OversizedCells,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, cells: [int], pos: int, n: int },
)]
#[allow(unused_assignments, unused_variables)]
fn oversized_cells_count(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<OversizedCells> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = OversizedCells {
        acc: 0,
        cells: vec![1i64; CELL_LEN],
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
        state.acc = state.acc + state.cells[0];
        state.pos = state.pos + 1i64;
    }
    state.acc
}

#[test]
fn cells_longer_than_sixteen_install_and_compile() {
    let state = OversizedCells {
        acc: 0,
        cells: vec![1i64; CELL_LEN],
        pos: 0,
        n: N,
    };
    let mut driver: JitDriver<OversizedCells> = JitDriver::new(2);
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, &PROGRAM)
            .install_canonical_liveness(&mut driver);
    }
    assert!(
        driver.dispatch_jitcode().is_some(),
        "CELL_LEN={CELL_LEN} is ptr then len; install must register \
         the dispatch JitCode"
    );

    COMPILES.store(0, Ordering::Relaxed);
    let got = oversized_cells_count(&PROGRAM, 2, N);
    assert_eq!(got, N, "each step adds cells[0]=1, so acc == n");
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "a 20-cell plain [int] must compile the hot loop"
    );
}
