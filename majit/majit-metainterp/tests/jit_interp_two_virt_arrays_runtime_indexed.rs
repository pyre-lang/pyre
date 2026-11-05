//! Two virtualizable arrays, each addressed by a RUNTIME index.
//!
//! `virtualizable.py`'s accessors reach an element as "the array's base slot
//! plus the element index", and the base slot of array N is the summed length
//! of arrays 0..N-1 (`get_index_in_array`). One array cannot observe that sum:
//! its base is always the static field count. Two can, and only if something
//! actually indexes the second one at run time — a constant index folds before
//! the sum is ever consulted.
//!
//! The two fixtures that already stand leave exactly this square empty.
//! `jit_interp_virt_array_stack_has_no_memory_ops` indexes ONE array by a
//! running depth; `jit_interp_two_virt_arrays_with_scalar` declares TWO arrays
//! but reaches both at constant index 0. So "two arrays" and "runtime index"
//! are each covered alone and neither is covered together.
//!
//! The shape here is a two-pool stack machine: `vals` is a ring holding each
//! pool's top `CAP` words at `pool * CAP + (h & CAP_MASK)`, and `depths` parks a
//! pool's height while another pool is selected. Both indices are reds. The
//! lengths differ (`POOLS * CAP` against `POOLS`) so a base computed as anything
//! other than the real sum lands inside the wrong array rather than out of
//! bounds, which is the failure that stays silent.
//!
//! The assertion is compiled-equals-interpreted. A cold run pins the answer with
//! the JIT unable to reach its threshold; the warm run must agree.
use core::sync::atomic::{AtomicU32, Ordering};

use majit_metainterp::virt_array::VirtArray;

pub type Bytecode = [u8];

/// Two pools, so `depths` has something to park.
const POOLS: usize = 2;
/// A power of two: a pool's slot for height `h` is `h & CAP_MASK`.
const CAP: usize = 4;
const CAP_MASK: usize = CAP - 1;

const OP_PUSH: u8 = 1;
const OP_DRAIN: u8 = 2;
const OP_SWITCH: u8 = 3;
const OP_BACK: u8 = 4;
const OP_RET: u8 = 5;

static COMPILES: AtomicU32 = AtomicU32::new(0);

struct TwoPools {
    /// `pool * CAP + (h & CAP_MASK)`.
    vals: VirtArray<i64>,
    /// One parked height per pool. Shorter than `vals`, so the two arrays
    /// cannot share a base by coincidence.
    depths: VirtArray<i64>,
    sel: usize,
    sp: usize,
    counter: i64,
    acc: i64,
}

#[majit_macros::jit_interp(
    state = TwoPools,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        vals: [int; virt],
        depths: [int; virt],
        sel: int(usize),
        sp: int(usize),
        counter: int,
        acc: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, iterations: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<TwoPools> =
        majit_metainterp::JitDriver::new(threshold);
    driver.set_on_compile_loop(|_green_key, _before, _after, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = TwoPools {
        vals: VirtArray::filled(0i64, POOLS * CAP),
        depths: VirtArray::filled(0i64, POOLS),
        sel: 0usize,
        sp: 0usize,
        counter: iterations,
        acc: 0i64,
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
            OP_PUSH => {
                // The operand is what distinguishes the two pools' contents, so
                // a read that lands in the wrong pool changes the answer.
                let value = program[pc] as i64;
                pc += 1;
                let slot = state.sel * CAP + (state.sp & CAP_MASK);
                state.vals[slot] = value;
                state.sp += 1;
            }
            OP_DRAIN => {
                state.sp -= 1;
                let slot = state.sel * CAP + (state.sp & CAP_MASK);
                state.acc += state.vals[slot];
            }
            OP_SWITCH => {
                // Park the outgoing pool's height and take the incoming one's
                // back. Both reach `depths` at a runtime index.
                state.depths[state.sel] = state.sp as i64;
                state.sel = (state.sel + 1) & (POOLS - 1);
                state.sp = state.depths[state.sel] as usize;
            }
            OP_BACK => {
                state.counter -= 1;
                if state.counter != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            OP_RET => break,
            _ => unreachable!(),
        }
    }
    state.acc
}

/// Each iteration stacks pool 0 three deep, leaves it parked across a switch,
/// works pool 1, and only then drains pool 0 back down.
///
/// The depth is what this fixture is for. A machine that pushes one word reaches
/// its array at two indices, and two indices is not a runtime index — the
/// promotion that stands in for one is exercised by how MANY values the index
/// takes. Pool 0 walks slots 0..3 and pool 1 slots 4..6, so seven distinct
/// element slots and both `depths` entries are reached at run time.
fn program() -> Vec<u8> {
    vec![
        OP_PUSH, 7, // pool 0: sp 0 -> 1
        OP_PUSH, 8, // sp 1 -> 2
        OP_PUSH, 9,         // sp 2 -> 3
        OP_SWITCH, // park pool 0 at 3, select pool 1
        OP_PUSH, 5, // pool 1: sp 0 -> 1
        OP_PUSH, 6,         // sp 1 -> 2
        OP_DRAIN,  // 6
        OP_DRAIN,  // 5
        OP_SWITCH, // park pool 1 at 0, select pool 0 and take 3 back
        OP_DRAIN,  // 9
        OP_DRAIN,  // 8
        OP_DRAIN,  // 7
        OP_BACK, OP_RET,
    ]
}

/// 6 + 5 + 9 + 8 + 7. Every term comes from a different element slot, so a read
/// that lands one slot off changes the sum rather than repeating a value.
const PER_ITERATION: i64 = 35;

#[test]
fn two_runtime_indexed_virt_arrays_agree_with_the_interpreter() {
    let program = program();
    const ITERATIONS: i64 = 200;

    COMPILES.store(0, Ordering::Relaxed);
    // Unreachable threshold, so this run fixes the answer without the JIT.
    let cold = mainloop(&program, ITERATIONS, u32::MAX);
    assert_eq!(cold, ITERATIONS * PER_ITERATION, "cold interpretation");
    assert_eq!(
        COMPILES.load(Ordering::Relaxed),
        0,
        "the cold run must not have compiled anything"
    );

    let warm = mainloop(&program, ITERATIONS, 3);
    assert_eq!(
        warm, cold,
        "a second virtualizable array reached at a runtime index must read the \
         same words compiled as interpreted"
    );
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "two runtime-indexed `[int; virt]` arrays did not compile the hot loop"
    );
}
