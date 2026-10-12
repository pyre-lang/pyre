//! A heap virtualizable (`virtualizable_fields` on a `ref(_)` state field)
//! beside a plain `[int]` that the loop reads as `state.cells[i]`.
//!
//! Each `[int]` is ptr then len, so the object's live index is a fixed
//! function of the declared layout (`make_virtualizable_infos`).

use core::sync::atomic::{AtomicU32, Ordering};
use majit_metainterp::JitDriver;
use majit_metainterp::JitState as _;

pub type Bytecode = [u8];

const OP_ACC: u8 = 1;
const OP_STEP: u8 = 2;
const OP_JUMP_BACK: u8 = 3;
const OP_HALT: u8 = 4;

const PROGRAM: [u8; 5] = [OP_ACC, OP_STEP, OP_JUMP_BACK, 0, OP_HALT];
const CELL_LEN: usize = 40;
const N: i64 = 40;

#[repr(C)]
struct Frame {
    token: usize,
    acc: i64,
}

const FRAME_TOKEN_OFFSET: usize = core::mem::offset_of!(Frame, token);
const FRAME_ACC_OFFSET: usize = core::mem::offset_of!(Frame, acc);

struct HeapVableAndCells {
    i: i64,
    n: i64,
    cells: Vec<i64>,
    frame: usize,
}

static COMPILES: AtomicU32 = AtomicU32::new(0);

fn seeded_cells() -> Vec<i64> {
    (0..CELL_LEN as i64).map(|k| k + 1).collect()
}

fn expected_acc() -> i64 {
    (1..=N).sum()
}

#[majit_macros::jit_interp(
    state = HeapVableAndCells,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        i: int,
        n: int,
        cells: [int],
        frame: ref(Frame),
    },
    virtualizable_fields = {
        var: frame,
        token_offset: FRAME_TOKEN_OFFSET,
        fields: { acc: int @ FRAME_ACC_OFFSET },
    },
)]
#[allow(unused_assignments, unused_variables)]
fn heap_vable_cells_sum(program: &Bytecode, threshold: u32, frame: usize, n: i64) -> i64 {
    let mut driver: JitDriver<HeapVableAndCells> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = HeapVableAndCells {
        i: 0,
        n,
        cells: seeded_cells(),
        frame,
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
            OP_ACC => {
                let i = state.i as usize;
                state.frame.acc = state.frame.acc + state.cells[i];
            }
            OP_STEP => {
                state.i = state.i + 1i64;
            }
            OP_JUMP_BACK => {
                let target = program[pc] as usize;
                pc += 1;
                if state.i < state.n {
                    if target < pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            OP_HALT => break,
            _ => break,
        }
    }
    state.frame.acc
}

/// Int scalars, then ptr then len for `cells`, then the heap virtualizable.
#[test]
fn heap_vable_live_index_sits_past_the_plain_int_array() {
    let mut frame = Frame { token: 0, acc: 0 };
    let frame_ptr = &mut frame as *mut Frame as usize;
    let state = HeapVableAndCells {
        i: 0,
        n: N,
        cells: seeded_cells(),
        frame: frame_ptr,
    };
    let meta = state.build_meta(0, &PROGRAM);
    let types = state.live_value_types(&meta);
    let live = state.extract_live(&meta);
    assert_eq!(
        types,
        vec![
            majit_ir::Type::Int, // i
            majit_ir::Type::Int, // n
            majit_ir::Type::Int, // cells items-pointer
            majit_ir::Type::Int, // cells length
            majit_ir::Type::Ref, // frame
        ],
        "the heap virtualizable follows ptr then len, independent of cells.len()"
    );
    assert_eq!(live.len(), types.len());
    assert_eq!(live[2], state.cells.as_ptr() as i64);
    assert_eq!(live[3], state.cells.len() as i64);
    assert_eq!(live[4], frame_ptr as i64);

    let info = <HeapVableAndCells as majit_metainterp::JitState>::__build_virtualizable_info()
        .expect("virtualizable_fields builds a vinfo");
    assert_eq!(
        info.identity_live_index,
        Some(4),
        "make_virtualizable_infos names the object in the declared live list"
    );

    let mut driver: JitDriver<HeapVableAndCells> = JitDriver::new(u32::MAX);
    state
        .build_meta(0, &PROGRAM)
        .install_canonical_liveness(&mut driver);
    assert!(
        driver.dispatch_jitcode().is_some(),
        "heap virtualizable + plain [int] must install a dispatch JitCode"
    );
    let contract = driver
        .flat_entry_contract()
        .expect("a heap virtualizable arms the flat entry contract");
    assert_eq!(contract.index_of_virtualizable, 4);
}

#[test]
fn heap_vable_beside_plain_int_array_compiles_and_matches_the_interpreter() {
    let mut cold_frame = Frame { token: 0, acc: 0 };
    let cold = heap_vable_cells_sum(
        &PROGRAM,
        u32::MAX,
        &mut cold_frame as *mut Frame as usize,
        N,
    );
    assert_eq!(
        cold,
        expected_acc(),
        "interpreter sums cells[i] into frame.acc"
    );
    assert_eq!(
        cold_frame.acc, cold,
        "the interpreter writes the virtualizable field on the heap object"
    );

    COMPILES.store(0, Ordering::Relaxed);
    let mut warm_frame = Frame { token: 0, acc: 0 };
    let warm = heap_vable_cells_sum(&PROGRAM, 3, &mut warm_frame as *mut Frame as usize, N);
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "the loop must compile: the heap virtualizable's live index is fixed \
         (compiles={} warm={warm} cold={cold})",
        COMPILES.load(Ordering::Relaxed),
    );
    assert_eq!(warm, cold, "compiled sum must match the interpreter");
    assert_eq!(
        warm_frame.acc, cold_frame.acc,
        "compiled code must leave the virtualizable's fields matching the interpreter"
    );
}
