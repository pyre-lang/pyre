//! The entry buffer `extract_live_values_into` fills is the pairing of the two
//! word-form readers, `extract_live` and `live_value_types`, field for field.
//!
//! The override writes each field as a `Value` directly rather than filling
//! the two word buffers and zipping them, so the two forms are produced by
//! different generated code and can drift apart: a field emitted in one order
//! by one and another by the other, or a float carried as bits by one and as
//! a value by the other. This pins them equal.
//!
//! `virtualizable.py VirtualizableInfo.__init__` puts int/ref/float scalars
//! on the same object as the virt array, so they are not independent
//! JitDriver reds. The red block is the one identity; compiled entry
//! reloads the statics with `GETFIELD` (`compile.py
//! patch_new_loop_to_load_virtualizable_fields`).

use majit_metainterp::virt_array::VirtArray;
use majit_metainterp::{JitDriver, JitState};

pub type Bytecode = [u8];

struct Cell {
    #[allow(dead_code)] // linked-cell shape for the virt-array entry test
    next: *mut Cell,
    value: i64,
}

struct EveryKindState {
    total: i64,
    regs: VirtArray<i64>,
    head: usize,
    weight: f64,
}

const OP_STEP: u8 = 1;

#[majit_macros::jit_interp(
    state = EveryKindState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        total: int,
        regs: [int; virt],
        head: ref(Cell),
        weight: float,
    },
    ref_fields = { Cell::next => Cell },
    int_fields = { Cell::value => i64 },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_every_kind(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<EveryKindState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = EveryKindState {
        total: 0,
        regs: VirtArray::filled(0i64, 2),
        head: 0,
        weight: 0.0,
    };
    loop {
        jit_merge_point!();
        let opcode = program[pc];
        match opcode {
            OP_STEP => {
                state.total += 1;
                pc += 1;
            }
            _ => return state.total,
        }
    }
}

#[test]
fn the_entry_values_are_the_word_form_paired() {
    let program: Vec<u8> = vec![OP_STEP, 0];
    let mut cell = Cell {
        next: std::ptr::null_mut(),
        value: 7,
    };
    let mut state = EveryKindState {
        total: -3,
        regs: VirtArray::from_slice(&[11, i64::MIN]),
        head: &mut cell as *mut Cell as usize,
        weight: -0.75,
    };
    let meta = state.build_meta(0, &program);

    let paired = state.extract_live_values(&meta);

    let mut out = Vec::new();
    let mut raw = Vec::new();
    let mut types = Vec::new();
    state.extract_live_values_into(&meta, &mut out, &mut raw, &mut types);

    assert_eq!(
        out, paired,
        "the direct form and the paired word form must carry the same values in \
         the same order"
    );
    // The red block is the virtualizable identity alone. `total`, `head`
    // and `weight` ride on that object and are reloaded at compiled
    // entry, not carried as reds.
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0],
        majit_ir::Value::Ref(majit_ir::GcRef(&state as *const EveryKindState as usize))
    );
    let _ = cell.value;
}
