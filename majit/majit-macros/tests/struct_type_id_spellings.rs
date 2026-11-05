//! One Rust struct spelled two ways must be one lltype.
//!
//! `descr.py` `get_size_descr` / `get_field_descr` key on the STRUCT object.
//! A virtual stored through `crate::spell::Num` and read through `Num` used
//! to mint two field descrs, so `GetfieldGcF` found no slot and folded to 0.0.

mod spell {
    #[repr(C)]
    pub struct Num {
        pub v: f64,
    }
}

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use majit_ir::OpCode;
use majit_metainterp::JitDriver;
use majit_metainterp::virt_array::VirtArray;
use spell::Num;

static COMPILES: AtomicUsize = AtomicUsize::new(0);
static COMPILED: Mutex<Vec<OpCode>> = Mutex::new(Vec::new());

fn alloc_num(v: f64) -> *mut crate::spell::Num {
    Box::into_raw(Box::new(crate::spell::Num { v }))
}

#[majit_macros::jit_inline(
    int_fields = { crate::spell::Num::v => f64 },
    struct_allocs = { crate::spell::Num => alloc_num },
    headerless_structs = { crate::spell::Num },
)]
fn make_num() -> *mut Num {
    let w = crate::spell::Num { v: 3.5 };
    w as *mut Num
}

#[majit_macros::jit_inline(
    ref_params = { w: ref(Num) },
    int_fields = { Num::v => f64 },
    headerless_structs = { Num },
)]
fn read_num(w: *mut Num) -> f64 {
    unsafe { (*w).v }
}

struct State {
    acc: i64,
    ticks: i64,
    regs: VirtArray<f64>,
}

pub type Bytecode = [u8];

const OP_READ: u8 = 1;
const OP_TICK: u8 = 2;
const OP_HALT: u8 = 3;
const PROGRAM: [u8; 3] = [OP_READ, OP_TICK, OP_HALT];
const TICKS: i64 = 40;
const STORED: f64 = 3.5;

#[majit_macros::jit_interp(
    state = State,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, ticks: int, regs: [float; virt] },
    calls = { make_num => inline_ref, read_num => inline_float },
    int_fields = { Num::v => f64, crate::spell::Num::v => f64 },
    headerless_structs = { Num, crate::spell::Num },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch(program: &Bytecode, threshold: u32, ticks: i64) -> i64 {
    let mut driver: JitDriver<State> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
        *COMPILED.lock().unwrap() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = State {
        acc: 0,
        ticks,
        regs: VirtArray::filled(0.0, 1),
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
            OP_READ => {
                let w = make_num();
                state.regs[0] = read_num(w);
                state.acc = state.regs[0].to_bits() as i64;
            }
            OP_TICK => {
                state.ticks -= 1;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            OP_HALT => break,
            _ => break,
        }
    }
    state.acc
}

#[test]
fn two_spellings_of_a_virtual_float_fold_to_the_stored_value() {
    let expected = STORED.to_bits() as i64;
    let cold = dispatch(&PROGRAM, u32::MAX, TICKS);
    assert_eq!(
        cold, expected,
        "the interpreter must return the stored float"
    );

    let warm = dispatch(&PROGRAM, 4, TICKS);
    assert_eq!(
        warm, expected,
        "compiled getfield folded to 0.0 (bits {warm:#x}) instead of {STORED}"
    );
    assert!(
        COMPILES.load(Ordering::Relaxed) > 0,
        "nothing compiled, so the warm answer was the interpreter's too"
    );
    let body = COMPILED.lock().unwrap().clone();
    assert!(
        !body.contains(&OpCode::GetfieldGcF),
        "GetfieldGcF survived, so the two spellings did not share a slot: {body:?}"
    );
}
