//! A portal whose post-loop suffix contains an unlowerable loop before an
//! int return.
//!
//! `lower_for_loop` only unrolls a literal range, so the iterator `for` stays
//! out of the JitCode and the walk ends in `void_return`. The native suffix
//! then owns that prefix (`pyjitpl.py finishframe` / `DoneWithThisFrame*`
//! returning to the interpreter). Finish has to write every live red back
//! first (`compile_done_with_this_frame` / `write_from_resume_data_partial`)
//! or the suffix reads the trace-start `acc` and drops the last iteration.
//! Lowering only the tail would emit `int_return` and skip the native suffix,
//! so the prefix's side effects would vanish on the JIT path.

use std::sync::atomic::{AtomicUsize, Ordering};

use majit_metainterp::jitcode::insns::{BC_INT_RETURN, BC_VOID_RETURN};
use majit_metainterp::{Assembler, JitCode, JitDriver};

pub type Bytecode = [u8];

const PROGRAM: [u8; 1] = [0];

static COMPILES: AtomicUsize = AtomicUsize::new(0);
static PREFIX_HITS: AtomicUsize = AtomicUsize::new(0);
static PROBE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

struct SuffixLoopState {
    acc: i64,
    pos: i64,
    n: i64,
}

#[majit_macros::jit_interp(
    state = SuffixLoopState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, pos: int, n: int },
)]
#[allow(unused_assignments, unused_variables)]
fn suffix_loop_int_return(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<SuffixLoopState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_, _, _, _| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = SuffixLoopState { acc: 0, pos: 0, n };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while state.pos < state.n {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        state.acc += 1i64;
        state.pos += 1i64;
    }
    // Unlowerable: `lower_for_loop` only unrolls a literal range.
    for _ in [0i64].iter() {
        PREFIX_HITS.fetch_add(1, Ordering::Relaxed);
    }
    state.acc
}

fn run(threshold: u32, n: i64) -> (i64, usize) {
    let _guard = PROBE_LOCK.lock();
    COMPILES.store(0, Ordering::Relaxed);
    PREFIX_HITS.store(0, Ordering::Relaxed);
    let got = suffix_loop_int_return(&PROGRAM, threshold, n);
    (got, COMPILES.load(Ordering::Relaxed))
}

fn run_with_hits(threshold: u32, n: i64) -> (i64, usize, usize) {
    let _guard = PROBE_LOCK.lock();
    COMPILES.store(0, Ordering::Relaxed);
    PREFIX_HITS.store(0, Ordering::Relaxed);
    let got = suffix_loop_int_return(&PROGRAM, threshold, n);
    (
        got,
        COMPILES.load(Ordering::Relaxed),
        PREFIX_HITS.load(Ordering::Relaxed),
    )
}

fn install() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![0]);
    __prebuild_jitcode_liveness_suffix_loop_int_return(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_suffix_loop_int_return(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture")
}

#[test]
fn dispatch_jitcode_emits_void_return_after_suffix_loop() {
    let dispatch_jc = install();
    let code = &dispatch_jc.code;
    assert!(
        code.contains(&BC_VOID_RETURN),
        "an unlowerable suffix prefix must leave BC_VOID_RETURN so the \
         native suffix runs that prefix; got bytes {code:?}"
    );
    assert!(
        !code.contains(&BC_INT_RETURN),
        "lowering only the tail would drop the prefix; got bytes {code:?}"
    );
}

/// `n == threshold` compiles on the last header tick. `void_return` Finish
/// has to write the post-body `acc` (`n`) back before the native suffix
/// runs; a merge-point / wiped stash leaves `n - 1`.
#[test]
fn last_iteration_finish_carries_the_int() {
    for n in [2i64, 3, 5, 8] {
        let (got, compiles) = run(n as u32, n);
        assert!(
            compiles >= 1,
            "n={n}: no loop compiled, so Finish never ran and this case \
             tests nothing"
        );
        assert_eq!(got, n, "n={n}: Finish dropped the returned int");
    }
}

#[test]
fn every_threshold_and_trip_count_answers_correctly() {
    for threshold in [2u32, 4, 8] {
        for n in 2i64..30 {
            let (got, _) = run(threshold, n);
            assert_eq!(got, n, "threshold={threshold} n={n}");
        }
    }
}

/// The suffix prefix is an unlowerable iterator `for` with an observable
/// side effect. The interpreter runs it; a tail-only `int_return` JitCode
/// would skip it. JIT and interpreter must agree on both the counter and
/// the returned `acc`.
#[test]
fn unlowerable_suffix_prefix_side_effect_matches_interpreter() {
    for n in [2i64, 3, 5, 8] {
        let (interp_got, interp_compiles, interp_hits) = run_with_hits(u32::MAX, n);
        assert_eq!(
            interp_compiles, 0,
            "n={n}: a huge threshold must stay on the interpreter"
        );
        assert_eq!(interp_got, n, "n={n}: interpreter acc");
        assert_eq!(
            interp_hits, 1,
            "n={n}: interpreter must run the suffix prefix once"
        );

        let (jit_got, jit_compiles, jit_hits) = run_with_hits(n as u32, n);
        assert!(
            jit_compiles >= 1,
            "n={n}: no loop compiled, so the JIT path never ran"
        );
        assert_eq!(jit_got, interp_got, "n={n}: JIT acc diverged");
        assert_eq!(
            jit_hits, interp_hits,
            "n={n}: JIT dropped or double-ran the suffix prefix"
        );
    }
}
