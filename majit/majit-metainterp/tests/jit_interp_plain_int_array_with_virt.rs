//! Mixed layout: a plain `[int]` longer than 16 cells beside a `[int; virt]`.
//! The loop compiles and the result matches the interpreter.

use core::sync::atomic::{AtomicU32, Ordering};
use majit_metainterp::JitDriver;
use majit_metainterp::virt_array::VirtArray;

pub type Bytecode = [u8];

const PROGRAM: [u8; 1] = [0];
const CELL_LEN: usize = 20;
const N: i64 = 40;

struct MixedOverReserve {
    acc: i64,
    cells: Vec<i64>,
    iregs: VirtArray<i64>,
}

static COMPILES: AtomicU32 = AtomicU32::new(0);

#[majit_macros::jit_interp(
    state = MixedOverReserve,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, cells: [int], iregs: [int; virt] },
)]
#[allow(unused_assignments, unused_variables)]
fn mixed_over_reserve_count(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<MixedOverReserve> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = MixedOverReserve {
        acc: 0,
        cells: vec![1i64; CELL_LEN],
        iregs: VirtArray::from_slice(&[0i64, n]),
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while state.iregs[0] < state.iregs[1] {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        state.acc = state.acc + state.cells[0];
        state.iregs[0] = state.iregs[0] + 1i64;
    }
    state.acc
}

#[test]
fn plain_int_over_sixteen_beside_virt_array_compiles() {
    let state = MixedOverReserve {
        acc: 0,
        cells: vec![1i64; CELL_LEN],
        iregs: VirtArray::from_slice(&[0i64, N]),
    };
    let mut driver: JitDriver<MixedOverReserve> = JitDriver::new(2);
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, &PROGRAM)
            .install_canonical_liveness(&mut driver);
    }
    assert!(
        driver.dispatch_jitcode().is_some(),
        "mixed [int] len={CELL_LEN} + [int; virt] must install a dispatch JitCode"
    );

    COMPILES.store(0, Ordering::Relaxed);
    let got = mixed_over_reserve_count(&PROGRAM, 2, N);
    assert_eq!(got, N, "each step adds cells[0]=1, so acc == n");
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "mixed [int] len>16 + [int; virt] must compile the hot loop"
    );
}
