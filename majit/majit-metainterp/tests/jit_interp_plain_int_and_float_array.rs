//! Mixed layout: a plain `[int]` beside a plain `[float]`. Each array is
//! two INT identity slots (items pointer then length), so the 2-slot-per-
//! array layout is exercised with mixed element kinds.
//! `jtransform.py` `rewrite_op_getarrayitem` emits `getarrayitem_raw_i`
//! and `getarrayitem_raw_f` from each array's item type.

use core::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};
use majit_metainterp::jitcode::insns::{BC_GETARRAYITEM_RAW_F, BC_GETARRAYITEM_RAW_I};
use majit_metainterp::{Assembler, JitCode, JitDriver};

pub type Bytecode = [u8];

const PROGRAM: [u8; 1] = [0];
const N: i64 = 40;

struct MixedCells {
    acc_i: i64,
    ints: Vec<i64>,
    floats: Vec<f64>,
    acc_f: f64,
    i: i64,
    n: i64,
    len: i64,
}

static COMPILES: AtomicU32 = AtomicU32::new(0);
static LAST_ACC_I: AtomicI64 = AtomicI64::new(0);
static LAST_ACC_F_BITS: AtomicU64 = AtomicU64::new(0);

fn seed_ints() -> Vec<i64> {
    vec![1i64, 2, 3, 4]
}

fn seed_floats() -> Vec<f64> {
    vec![0.25, 0.5, 0.75, 1.0]
}

fn expected_sums(ints: &[i64], floats: &[f64], n: i64) -> (i64, f64) {
    let mut acc_i = 0i64;
    let mut acc_f = 0.0f64;
    let mut i = 0usize;
    for _ in 0..n {
        acc_i += ints[i];
        acc_f += floats[i];
        i += 1;
        if i == ints.len() {
            i = 0;
        }
    }
    (acc_i, acc_f)
}

#[majit_macros::jit_interp(
    state = MixedCells,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        acc_i: int,
        ints: [int],
        floats: [float],
        acc_f: float,
        i: int,
        n: int,
        len: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn mixed_cells_sum(
    program: &Bytecode,
    threshold: u32,
    ints: Vec<i64>,
    floats: Vec<f64>,
    n: i64,
) -> i64 {
    let mut driver: JitDriver<MixedCells> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = MixedCells {
        acc_i: 0,
        acc_f: 0.0,
        len: ints.len() as i64,
        ints,
        floats,
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
        state.acc_i = state.acc_i + state.ints[i];
        state.acc_f = state.acc_f + state.floats[i];
        let next = state.i + 1i64;
        state.i = if next < state.len { next } else { 0i64 };
        state.n = state.n - 1i64;
    }
    LAST_ACC_I.store(state.acc_i, Ordering::SeqCst);
    LAST_ACC_F_BITS.store(state.acc_f.to_bits(), Ordering::SeqCst);
    state.acc_i
}

fn snapshot() -> (i64, u64) {
    (
        LAST_ACC_I.load(Ordering::SeqCst),
        LAST_ACC_F_BITS.load(Ordering::SeqCst),
    )
}

fn build_dispatch_jitcode() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![0]);
    __prebuild_jitcode_liveness_mixed_cells_sum(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_mixed_cells_sum(&mut asm, 0i64).expect("dispatch lower must succeed")
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
fn mixed_plain_arrays_are_two_int_identity_slots_each() {
    let state = MixedCells {
        acc_i: 0,
        ints: seed_ints(),
        floats: seed_floats(),
        acc_f: 0.0,
        i: 0,
        n: N,
        len: 4,
    };
    {
        use majit_metainterp::JitState as _;
        let layout = state.state_field_layout();
        assert_eq!(layout.num_scalars, 4);
        assert_eq!(layout.num_plain_int_arrays, 2);
        assert_eq!(layout.num_float_scalars, 1);
        assert_eq!(layout.total_slots(), 8);
        assert_eq!(
            layout.array_identity_slots(0),
            (layout.int_scalar_base + 4, layout.int_scalar_base + 5)
        );
        assert_eq!(
            layout.array_identity_slots(1),
            (layout.int_scalar_base + 6, layout.int_scalar_base + 7)
        );
    }
}

#[test]
fn mixed_plain_arrays_emit_raw_i_and_raw_f() {
    let dispatch_jc = build_dispatch_jitcode();
    majit_metainterp::assert_no_degraded_dispatch_arms("MixedCells");
    let tree: Vec<&JitCode> = dispatch_tree(&dispatch_jc).collect();
    assert!(
        tree.iter()
            .any(|code| code.code.contains(&BC_GETARRAYITEM_RAW_I)),
        "the [int] read must emit getarrayitem_raw_i; dispatch={:?}",
        dispatch_jc.code
    );
    assert!(
        tree.iter()
            .any(|code| code.code.contains(&BC_GETARRAYITEM_RAW_F)),
        "the [float] read must emit getarrayitem_raw_f; dispatch={:?}",
        dispatch_jc.code
    );
}

#[test]
fn compiled_mixed_plain_arrays_match_the_interpreter() {
    let ints = seed_ints();
    let floats = seed_floats();
    let (expected_i, expected_f) = expected_sums(&ints, &floats, N);

    let interp_i = mixed_cells_sum(&PROGRAM, u32::MAX, ints.clone(), floats.clone(), N);
    let interp = snapshot();
    assert_eq!(interp_i, expected_i);
    assert_eq!(interp.0, expected_i);
    assert_eq!(interp.1, expected_f.to_bits());

    COMPILES.store(0, Ordering::Relaxed);
    let compiled_i = mixed_cells_sum(&PROGRAM, 2, ints, floats, N);
    let compiled = snapshot();
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "the loop must compile"
    );
    assert_eq!(compiled_i, interp_i);
    assert_eq!(
        compiled, interp,
        "compiled (acc_i, acc_f bits) must match the interpreter bit-for-bit"
    );
}
