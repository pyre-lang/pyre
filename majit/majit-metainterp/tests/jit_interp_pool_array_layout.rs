//! A `pool_arrays` base whose items are not where the descr used to assume.
//!
//! `add_ptr_array_descr` hard-coded `base_size = size_of::<usize>()` and
//! `len_offset = Some(0)` — a `{ len, items.. }` header the declaration never
//! stated and the consumer's struct had to be built to match. Reordering that
//! header, or dropping it, compiled clean on both sides and left the element
//! read at `base + 8 + i*8`.
//!
//! Nothing could have caught it. The struct is the consumer's own, so the
//! macro sees whatever it is handed; no `offset_of!` was emitted, so the
//! assumption never met the layout. The two facts only ever met at run time,
//! as a read of the wrong slot.
//!
//! This machine puts the items LAST, which is the arrangement the old
//! hard-code gets wrong: with a `[i64; 2]` ahead of them, an assumed base of
//! one word reads `items[i]` from inside that padding.

use majit_metainterp::{Assembler, JitDriver};
use std::sync::atomic::{AtomicUsize, Ordering};

static COMPILES: AtomicUsize = AtomicUsize::new(0);

#[repr(C)]
struct Slot {
    value: i64,
}

/// Items last, and behind more than one word — so an assumed `base_size` of
/// `size_of::<usize>()` lands inside `pad` rather than on an element.
#[repr(C)]
struct Pools {
    len: usize,
    pad: [i64; 2],
    items: [*mut Slot; 4],
}

/// The marker call. Its concrete body is the fallback for a build with no
/// `pool_arrays` declaration, and it is what the lowering replaces with a
/// `getarrayitem_gc_r` when the declaration matches.
fn pool_get(base: usize, index: i64) -> usize {
    let pools = base as *const Pools;
    unsafe { (*pools).items[index as usize] as usize }
}

struct PoolState {
    pools: usize,
    selected: usize,
    total: i64,
    /// The loop counter. The element read has to be inside a compiled trace to
    /// be graded here at all, and a straight-line program never compiles one.
    ticks: i64,
}

pub type Bytecode = [u8];

const OP_SELECT: u8 = 1;
const OP_READ: u8 = 2;
const OP_TICK: u8 = 3;
const OP_HALT: u8 = 4;

#[majit_macros::jit_interp(
    state = PoolState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        pools: ref(Pools),
        selected: ref(Slot),
        total: int,
        ticks: int,
    },
    pool_arrays = { pools.items[len] => pool_get -> Slot },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_pool(program: &Bytecode, threshold: u32, pools: usize, ticks: i64) -> i64 {
    let mut driver: JitDriver<PoolState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = PoolState {
        pools,
        selected: 0,
        total: 0,
        ticks,
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
            OP_SELECT => {
                state.selected = pool_get(state.pools, 2i64);
            }
            OP_READ => {
                state.total = state.total + state.selected.value;
            }
            OP_TICK => {
                state.ticks -= 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            OP_HALT => {
                break;
            }
            _ => break,
        }
    }
    state.total
}

fn build_pools(slots: &mut [Slot; 4]) -> Pools {
    Pools {
        len: 4,
        pad: [-1, -1],
        items: [
            &mut slots[0] as *mut Slot,
            &mut slots[1] as *mut Slot,
            &mut slots[2] as *mut Slot,
            &mut slots[3] as *mut Slot,
        ],
    }
}

/// The element read must resolve through the declaration, not through a
/// header shape.
///
/// `items` sits behind `len` and a two-word `pad`, so the assumed one-word
/// base lands on `pad[0]` and reads `-1` as a `*mut Slot`.
///
/// The concrete path indexes the real field whatever the descr says, so the
/// sum alone does not grade the JIT — a program that never compiled a trace
/// returns 30 under any layout the descr claims. The compile count is what
/// makes the sum evidence, and it is asserted first for that reason: without
/// it this test passes on the hard-coded build it exists to reject.
#[test]
fn the_element_read_follows_the_declared_items_offset() {
    let mut slots = [
        Slot { value: 10 },
        Slot { value: 20 },
        Slot { value: 30 },
        Slot { value: 40 },
    ];
    let mut pools = build_pools(&mut slots);
    const TICKS: i64 = 200;
    let program = [OP_SELECT, OP_READ, OP_TICK, OP_HALT];

    let before = COMPILES.load(Ordering::Relaxed);
    let total = dispatch_pool(&program, 8, &mut pools as *mut Pools as usize, TICKS);
    assert!(
        COMPILES.load(Ordering::Relaxed) > before,
        "no trace compiled, so the element read below never went through a \
         descr and the assertion on it grades nothing",
    );
    assert_eq!(
        total,
        30 * TICKS,
        "index 2 must reach `items[2]`, not a word two fields earlier",
    );
}

/// …and the offsets the portal bakes in must be the struct's own.
///
/// Reading the jitcode rather than the run is what separates "the declaration
/// was used" from "the concrete fallback ran and happened to agree". A machine
/// whose trace never compiled would pass the test above unchanged.
#[test]
fn the_portal_bakes_in_the_structs_own_offsets() {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_dispatch_pool(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let dispatch_jc =
        __dispatch_jitcode_dispatch_pool(&mut asm, 0i64).expect("dispatch lower must succeed");

    let arrays: Vec<(usize, Option<usize>)> = dispatch_jc
        .exec
        .descrs
        .iter()
        .filter_map(|descr| descr.as_bh_descr())
        .filter_map(|descr| match descr {
            majit_metainterp::blackhole::BhDescr::Array {
                base_size,
                len_offset,
                is_array_of_pointers: true,
                ..
            } => Some((*base_size, *len_offset)),
            _ => None,
        })
        .collect();
    assert!(
        arrays.contains(&(
            core::mem::offset_of!(Pools, items),
            Some(core::mem::offset_of!(Pools, len)),
        )),
        "the pool-array descr must carry this struct's offsets \
         (items={}, len={}); found {arrays:?}",
        core::mem::offset_of!(Pools, items),
        core::mem::offset_of!(Pools, len),
    );
}
