//! An `-> i64` portal's driver takes `warmspot.py` `getkind` from the
//! signature once, at `JitDriver::new`. The steady compiled entry then
//! polls `DoneWithThisFrameDescrInt`.

use std::sync::atomic::{AtomicBool, Ordering};

use majit_metainterp::JitDriver;

pub type Bytecode = [u8];

const PROGRAM: [u8; 1] = [0];

static INT_FINISH: AtomicBool = AtomicBool::new(false);
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct AccState {
    acc: i64,
    pos: i64,
    n: i64,
}

#[majit_macros::jit_interp(
    state = AccState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, pos: int, n: int },
)]
#[allow(unused_assignments, unused_variables)]
fn count_portal(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<AccState> = JitDriver::new(threshold);
    let from_signature = driver.has_raw_int_finish();
    // The same kind, spelled again. `set_result_type` stays valid.
    driver.set_result_type(majit_ir::Type::Int);
    INT_FINISH.store(
        from_signature && driver.has_raw_int_finish(),
        Ordering::Relaxed,
    );
    let mut pc: usize = 0;
    let mut state = AccState { acc: 0, pos: 0, n };
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
    state.acc
}

#[test]
fn i64_portal_steady_entry_uses_the_int_finish() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    INT_FINISH.store(false, Ordering::Relaxed);
    let cold = count_portal(&PROGRAM, u32::MAX, 40);
    assert!(
        INT_FINISH.load(Ordering::Relaxed),
        "JitDriver::new did not take the portal's i64 kind"
    );
    let warm = count_portal(&PROGRAM, 2, 40);
    assert_eq!(warm, cold);
    assert!(
        INT_FINISH.load(Ordering::Relaxed),
        "the steady driver lost the int finish kind"
    );
}
