//! A loop-carried store to a plain `[int]` survives a mid-run guard
//! failure. The store writes the native buffer in place
//! (`jtransform.py` `rewrite_op_setarrayitem` raw arm), so blackhole /
//! interpreter resume must see the same `cells` as a pure interpreter.
//!
//! The rare branch is data-dependent and taken only after the hot path
//! has compiled.
//!
//! Before the store lowering, `validate_state_fields` refused this
//! program at expansion.

use core::sync::atomic::{AtomicU32, Ordering};
use majit_metainterp::JitDriver;

pub type Bytecode = [u8];

const PROGRAM: [u8; 1] = [0];
const N: i64 = 40;
const SWITCH_AT: i64 = 20;
const HOT_ADD: i64 = 5;
const RARE_ADD: i64 = 7;

struct GuardFailCells {
    cells: Vec<i64>,
    pos: i64,
    n: i64,
    switch_at: i64,
}

static COMPILES: AtomicU32 = AtomicU32::new(0);

fn seed_cells() -> Vec<i64> {
    vec![3i64, HOT_ADD, 0, 0]
}

fn expected_cells(mut cells: Vec<i64>, n: i64, switch_at: i64) -> Vec<i64> {
    for pos in 0..n {
        if pos >= switch_at {
            cells[0] += RARE_ADD;
        } else {
            cells[0] += cells[1];
        }
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
    state = GuardFailCells,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { cells: [int], pos: int, n: int, switch_at: int },
)]
#[allow(unused_assignments, unused_variables)]
fn hot_then_rare(
    program: &Bytecode,
    threshold: u32,
    cells: Vec<i64>,
    n: i64,
    switch_at: i64,
) -> i64 {
    let mut driver: JitDriver<GuardFailCells> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = GuardFailCells {
        cells,
        pos: 0,
        n,
        switch_at,
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
        if state.pos >= state.switch_at {
            state.cells[0] = state.cells[0] + RARE_ADD;
        } else {
            state.cells[0] = state.cells[0] + state.cells[1];
        }
        state.pos = state.pos + 1i64;
    }
    cells_fingerprint(&state.cells)
}

#[test]
fn compiled_plain_int_store_survives_a_late_guard_failure() {
    let cells = seed_cells();
    let expected = expected_cells(cells.clone(), N, SWITCH_AT);
    let expected_fp = cells_fingerprint(&expected);
    assert_ne!(
        expected[0],
        cells[0] + N * HOT_ADD,
        "the rare branch must change the final cells[0]"
    );

    let interp = hot_then_rare(&PROGRAM, u32::MAX, cells.clone(), N, SWITCH_AT);
    assert_eq!(
        interp, expected_fp,
        "interpreter must apply the rare branch after switch_at"
    );

    COMPILES.store(0, Ordering::Relaxed);
    let compiled = hot_then_rare(&PROGRAM, 2, cells, N, SWITCH_AT);
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "the hot path must compile before the rare branch is taken"
    );
    assert_eq!(
        compiled, interp,
        "after the compiled hot path guard-fails into the rare branch, \
         cells must still match the interpreter"
    );
}
