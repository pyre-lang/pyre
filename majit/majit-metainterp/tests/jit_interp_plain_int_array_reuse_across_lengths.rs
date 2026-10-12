//! A loop compiled while `cells.len() == 4` must also run when it is
//! entered with `cells.len() == 8`.
//!
//! `warmstate.py` `maybe_compile_and_run` selects the JIT cell by greens
//! alone and passes reds as runtime arguments. `jtransform.py`
//! `handle_jit_marker__jit_merge_point` does not turn an ordinary red
//! into a compatibility key. The length identity is a red
//! (`int_identity_base + plain_int_scalar_count + 2*k + 1`), carried
//! like any other red.

use core::sync::atomic::{AtomicU32, Ordering};
use majit_metainterp::JitDriver;

pub type Bytecode = [u8];

const PROGRAM: [u8; 1] = [0];
const N: i64 = 40;

struct Cells {
    acc: i64,
    cells: Vec<i64>,
    pos: i64,
    n: i64,
    i: i64,
    len: i64,
}

static COMPILES: AtomicU32 = AtomicU32::new(0);
static ENTRIES: AtomicU32 = AtomicU32::new(0);
static COMPILES_AFTER_FIRST: AtomicU32 = AtomicU32::new(0);
static ENTRIES_AFTER_FIRST: AtomicU32 = AtomicU32::new(0);

fn expected_sum(cells: &[i64], n: i64) -> i64 {
    let sum: i64 = cells.iter().copied().sum();
    sum * (n / cells.len() as i64)
}

fn cells_len4() -> Vec<i64> {
    vec![1, 2, 3, 4]
}

fn cells_len8() -> Vec<i64> {
    vec![10, 20, 30, 40, 50, 60, 70, 80]
}

#[majit_macros::jit_interp(
    state = Cells,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, cells: [int], pos: int, n: int, i: int, len: int },
)]
#[allow(unused_assignments, unused_variables)]
fn two_length_sum(
    program: &Bytecode,
    threshold: u32,
    cells_a: Vec<i64>,
    cells_b: Vec<i64>,
    n: i64,
) -> (i64, i64) {
    let mut driver: JitDriver<Cells> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    driver.set_on_compiled_entry(|_gk, _pc| {
        ENTRIES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = Cells {
        acc: 0,
        len: cells_a.len() as i64,
        cells: cells_a,
        pos: 0,
        n,
        i: 0,
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
        state.acc = state.acc + state.cells[state.i as usize];
        state.i = state.i + 1i64;
        if state.i >= state.len {
            state.i = 0i64;
        }
        state.pos = state.pos + 1i64;
    }
    let first = state.acc;
    COMPILES_AFTER_FIRST.store(COMPILES.load(Ordering::Relaxed), Ordering::Relaxed);
    ENTRIES_AFTER_FIRST.store(ENTRIES.load(Ordering::Relaxed), Ordering::Relaxed);

    state.acc = 0;
    state.len = cells_b.len() as i64;
    state.cells = cells_b;
    state.pos = 0;
    state.n = n;
    state.i = 0;
    while state.pos < state.n {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        state.acc = state.acc + state.cells[state.i as usize];
        state.i = state.i + 1i64;
        if state.i >= state.len {
            state.i = 0i64;
        }
        state.pos = state.pos + 1i64;
    }
    (first, state.acc)
}

#[test]
fn compiled_loop_reused_when_plain_int_array_length_changes() {
    let cells_a = cells_len4();
    let cells_b = cells_len8();
    let expect_a = expected_sum(&cells_a, N);
    let expect_b = expected_sum(&cells_b, N);

    let (interp_a, interp_b) =
        two_length_sum(&PROGRAM, u32::MAX, cells_a.clone(), cells_b.clone(), N);
    assert_eq!(
        interp_a, expect_a,
        "interpreter sums every cell of length 4"
    );
    assert_eq!(
        interp_b, expect_b,
        "interpreter sums every cell of length 8"
    );

    COMPILES.store(0, Ordering::Relaxed);
    ENTRIES.store(0, Ordering::Relaxed);
    COMPILES_AFTER_FIRST.store(0, Ordering::Relaxed);
    ENTRIES_AFTER_FIRST.store(0, Ordering::Relaxed);
    let (compiled_a, compiled_b) = two_length_sum(&PROGRAM, 2, cells_a, cells_b, N);
    assert_eq!(
        (compiled_a, compiled_b),
        (interp_a, interp_b),
        "compiled sums must match the interpreter at both lengths"
    );

    let compiles_first = COMPILES_AFTER_FIRST.load(Ordering::Relaxed);
    let compiles_total = COMPILES.load(Ordering::Relaxed);
    let entries_first = ENTRIES_AFTER_FIRST.load(Ordering::Relaxed);
    let entries_total = ENTRIES.load(Ordering::Relaxed);
    assert!(
        compiles_first >= 1,
        "the loop must compile at length 4; compiles_first={compiles_first}"
    );
    assert_eq!(
        compiles_total, compiles_first,
        "length 8 must reuse that loop, not compile a new one keyed on the length; \
         compiles_first={compiles_first} compiles_total={compiles_total}"
    );
    assert!(
        entries_total > entries_first,
        "length 8 must enter the compiled loop; \
         entries_first={entries_first} entries_total={entries_total}"
    );
}
