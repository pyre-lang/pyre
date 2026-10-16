//! Heap virtualizable plus another `ref(_)` state field.
//!
//! `warmspot.py` `make_virtualizable_infos` (`index_of_virtualizable`):
//! the object is one red at its declared position among the ref scalars.
//! An extra `ref(_)` before or after that object must occupy its own slot.

use core::sync::atomic::{AtomicU32, Ordering};
use majit_metainterp::JitDriver;

pub type Bytecode = [u8];

const OP_ACC: u8 = 1;
const OP_JUMP_BACK: u8 = 2;
const OP_HALT: u8 = 3;
const PROGRAM: [u8; 4] = [OP_ACC, OP_JUMP_BACK, 0, OP_HALT];
const N: i64 = 40;
const MARK: i64 = 111;

#[repr(C)]
struct Extra {
    token: usize,
    mark: i64,
}

#[repr(C)]
struct Frame {
    token: usize,
    acc: i64,
}

const FRAME_TOKEN_OFFSET: usize = core::mem::offset_of!(Frame, token);
const FRAME_ACC_OFFSET: usize = core::mem::offset_of!(Frame, acc);

struct HeapVableThenExtra {
    i: i64,
    n: i64,
    frame: usize,
    extra: usize,
}

struct HeapVableExtraFirst {
    i: i64,
    n: i64,
    extra: usize,
    frame: usize,
}

static COMPILES_AFTER: AtomicU32 = AtomicU32::new(0);
static COMPILES_BEFORE: AtomicU32 = AtomicU32::new(0);

#[majit_macros::jit_interp(
    state = HeapVableThenExtra,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        i: int,
        n: int,
        frame: ref(Frame),
        extra: ref(Extra),
    },
    virtualizable_fields = {
        var: frame,
        token_offset: FRAME_TOKEN_OFFSET,
        fields: { acc: int @ FRAME_ACC_OFFSET },
    },
)]
#[allow(unused_assignments, unused_variables)]
fn extra_after_vable(
    program: &Bytecode,
    threshold: u32,
    extra: usize,
    frame: usize,
    n: i64,
) -> i64 {
    let mut driver: JitDriver<HeapVableThenExtra> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES_AFTER.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = HeapVableThenExtra {
        i: 0,
        n,
        extra,
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
                state.frame.acc = state.frame.acc + state.extra.mark;
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

#[majit_macros::jit_interp(
    state = HeapVableExtraFirst,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        i: int,
        n: int,
        extra: ref(Extra),
        frame: ref(Frame),
    },
    virtualizable_fields = {
        var: frame,
        token_offset: FRAME_TOKEN_OFFSET,
        fields: { acc: int @ FRAME_ACC_OFFSET },
    },
)]
#[allow(unused_assignments, unused_variables)]
fn extra_before_vable(
    program: &Bytecode,
    threshold: u32,
    extra: usize,
    frame: usize,
    n: i64,
) -> i64 {
    let mut driver: JitDriver<HeapVableExtraFirst> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES_BEFORE.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = HeapVableExtraFirst {
        i: 0,
        n,
        extra,
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
                state.frame.acc = state.frame.acc + 1i64;
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

#[test]
fn extra_ref_after_heap_vable_is_read_from_its_own_slot() {
    let info = <HeapVableThenExtra as majit_metainterp::JitState>::__build_virtualizable_info()
        .expect("heap vable builds a vinfo");
    assert_eq!(
        info.identity_live_index,
        Some(2),
        "int scalars i,n then the vable among the ref fields"
    );
    assert_eq!(
        info.identity_ref_bank_index,
        Some(1),
        "program at r0; the vable is the first ref scalar"
    );

    let mut extra = Extra {
        token: 0,
        mark: MARK,
    };
    let mut frame = Frame { token: 0, acc: 0 };
    let extra_ptr = &mut extra as *mut Extra as usize;
    let frame_ptr = &mut frame as *mut Frame as usize;

    let cold = extra_after_vable(&PROGRAM, u32::MAX, extra_ptr, frame_ptr, N);
    assert_eq!(cold, MARK * N, "interpreter must add Extra.mark");
    assert_eq!(frame.acc, MARK * N);

    extra.mark = MARK;
    frame.acc = 0;
    COMPILES_AFTER.store(0, Ordering::Relaxed);
    let warm = extra_after_vable(&PROGRAM, 3, extra_ptr, frame_ptr, N);
    assert!(
        COMPILES_AFTER.load(Ordering::Relaxed) >= 1,
        "loop must compile so the extra-ref slot is observed (compiles={})",
        COMPILES_AFTER.load(Ordering::Relaxed)
    );
    assert_eq!(warm, cold, "compiled Extra.mark must match the interpreter");
    assert_eq!(frame.acc, MARK * N);
    assert_eq!(extra.mark, MARK);
}

#[test]
fn extra_ref_before_heap_vable_expands_and_hits_the_frame() {
    let info = <HeapVableExtraFirst as majit_metainterp::JitState>::__build_virtualizable_info()
        .expect("heap vable builds a vinfo");
    assert_eq!(
        info.identity_live_index,
        Some(3),
        "int scalars i,n then extra then the vable"
    );
    assert_eq!(
        info.identity_ref_bank_index,
        Some(2),
        "program at r0, extra at r1; the vable is the second ref scalar"
    );

    let mut extra = Extra {
        token: 0,
        mark: MARK,
    };
    let mut frame = Frame { token: 0, acc: 0 };
    let extra_ptr = &mut extra as *mut Extra as usize;
    let frame_ptr = &mut frame as *mut Frame as usize;

    let cold = extra_before_vable(&PROGRAM, u32::MAX, extra_ptr, frame_ptr, N);
    assert_eq!(cold, N, "interpreter must increment frame.acc");
    assert_eq!(frame.acc, N);
    assert_eq!(extra.mark, MARK, "interpreter must leave Extra.mark alone");

    extra.mark = MARK;
    frame.acc = 0;
    COMPILES_BEFORE.store(0, Ordering::Relaxed);
    let warm = extra_before_vable(&PROGRAM, 3, extra_ptr, frame_ptr, N);
    assert!(
        COMPILES_BEFORE.load(Ordering::Relaxed) >= 1,
        "loop must compile (compiles={})",
        COMPILES_BEFORE.load(Ordering::Relaxed)
    );
    assert_eq!(warm, cold, "compiled frame.acc must match interpreter");
    assert_eq!(frame.acc, N, "compiled writes must land on Frame");
    assert_eq!(extra.mark, MARK, "compiled writes must not land on Extra");
}
