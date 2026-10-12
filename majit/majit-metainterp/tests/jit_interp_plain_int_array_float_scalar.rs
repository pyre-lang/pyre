//! A state with one int scalar, one plain `[int]`, and one float scalar.
//! `StateFieldLayout::num_scalars` is the real int scalar count; the `[int]`
//! is ptr then len after those scalars. An abort that converts into the
//! blackhole must not consume the float as those ptr/len words.

use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use majit_metainterp::JitState as _;

pub type Bytecode = [u8];

const ADD_CELL: u8 = 1; // acc += cells[0]
const ADD_F: u8 = 2; // f += 0.5
const STEP: u8 = 3; // acc += cells[0]; f += 0.5
const BR_LT: u8 = 4; // [BR_LT, target]: jump if acc < cells[1]
const RETURN: u8 = 5;

const CELL: i64 = 3;
const N: i64 = 20;
const F0: f64 = 0.25;
const F_STEP: f64 = 0.5;
/// Loop bound stored in `cells[1]`: each iteration adds `2 * CELL` to `acc`.
const ACC_LIMIT: i64 = 2 * CELL * N;

static LAST_ACC: AtomicI64 = AtomicI64::new(0);
static LAST_F_BITS: AtomicU64 = AtomicU64::new(0);

struct MixedCells {
    acc: i64,
    cells: Vec<i64>,
    f: f64,
}

#[majit_macros::jit_interp(
    state = MixedCells,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, cells: [int], f: float },
)]
#[allow(unused_assignments, unused_variables)]
fn mixed_cells_mainloop(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<MixedCells> =
        majit_metainterp::JitDriver::new(threshold);
    // Cuts the trace off inside an opcode arm (`TraceCtx::is_too_long`).
    driver
        .meta_interp_mut()
        .warm_state_for_driver(0)
        .set_param("trace_limit", 2);
    let mut pc: usize = 0;
    let mut state = MixedCells {
        acc: 0,
        cells: vec![CELL, ACC_LIMIT],
        f: F0,
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);

        let opcode = program[pc];
        pc += 1;

        match opcode {
            ADD_CELL => {
                state.acc = state.acc + state.cells[0];
            }
            ADD_F => {
                // Compound assign with a float literal is the form the
                // dispatch lowerer expresses (`jit_interp_float_state_field`).
                state.f += 0.5;
            }
            STEP => {
                state.acc = state.acc + state.cells[0];
                state.f += 0.5;
            }
            BR_LT => {
                let target = program[pc] as usize;
                pc += 1;
                if state.acc < state.cells[1] {
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            RETURN => break,
            _ => {}
        }
    }

    LAST_ACC.store(state.acc, Ordering::SeqCst);
    LAST_F_BITS.store(state.f.to_bits(), Ordering::SeqCst);
    state.acc
}

/// One loop iteration: STEP, ADD_CELL, ADD_F, then back while `acc < cells[1]`.
/// Each iteration adds `2 * CELL` to `acc` and `2 * F_STEP` to `f`.
fn loop_program() -> Vec<u8> {
    vec![STEP, ADD_CELL, ADD_F, BR_LT, 0, RETURN]
}

fn snapshot() -> (i64, u64) {
    (
        LAST_ACC.load(Ordering::SeqCst),
        LAST_F_BITS.load(Ordering::SeqCst),
    )
}

#[test]
fn layout_num_scalars_is_the_real_int_scalar_count() {
    let state = MixedCells {
        acc: 1,
        cells: vec![CELL, ACC_LIMIT],
        f: F0,
    };
    let layout = state.state_field_layout();
    assert_eq!(layout.num_scalars, 1);
    assert_eq!(layout.num_plain_int_arrays, 1);
    assert_eq!(layout.total_slots(), 3);
    assert_eq!(
        layout.array_identity_slots(0),
        (layout.int_scalar_base + 1, layout.int_scalar_base + 2)
    );
}

#[test]
fn abort_blackhole_preserves_the_float_scalar() {
    // `trace_limit` 2 forces `TraceCtx::is_too_long` inside an opcode arm
    // of the bytecode-dispatch loop, the same abort
    // `abort_blackhole_virt_array` uses to reach `run_pending_abort_blackhole`.
    let program = loop_program();
    let _ = mixed_cells_mainloop(&program, 3);
    let traced = snapshot();
    let _ = mixed_cells_mainloop(&program, u32::MAX);
    let interp = snapshot();
    assert_eq!(
        traced, interp,
        "traced (acc, f bits) must match the interpreter run"
    );
    assert_eq!(traced.0, ACC_LIMIT);
    assert_eq!(
        traced.1,
        (F0 + 2.0 * F_STEP * (N as f64)).to_bits(),
        "float scalar must survive abort-blackhole seeding; got bits {:#x}",
        traced.1
    );
}
