//! One logical stack split between a virtualizable ring and a heap chain
//! reached through a `ref(T)` state field.
//!
//! `jit_interp_two_virt_arrays_runtime_indexed` covers two `[int; virt]` arrays
//! at runtime indices and nothing else: every value it ever reads lives in a
//! box, so the whole machine is answerable without touching memory. That is
//! also its blind spot. A ring only holds the top `CAP` words; a stack deeper
//! than that keeps its remainder somewhere else, and reaching that somewhere
//! else brings in the two things the boxed-only machine never has —
//!
//! * a `ref(T)` state scalar, which shares the flat virtualizable slot space
//!   with the int scalars and the array elements but is carried in the ref
//!   register bank, and
//! * residual calls sitting BETWEEN element accesses, so the boxes have to
//!   survive a call boundary rather than only a guard.
//!
//! The interesting property is that the two halves of the stack are kept
//! consistent by a RED condition (`sp > CAP` / `sp >= CAP`). A trace recorded
//! while the ring alone was enough compiles that condition to a guard and
//! contains no chain traffic at all. Everything about whether the split stays
//! coherent therefore rests on the state the guard restores: if `sp` comes back
//! disagreeing with the chain by even one, the interpreter either pops a chain
//! that is empty or leaks a word into one that should be.
//!
//! So the assertion is not only compiled-equals-interpreted on the sum. The
//! chain must also be empty at the end of the run: every word the ring spilled
//! has to come back exactly once.
use core::sync::atomic::{AtomicU32, Ordering};

use majit_metainterp::virt_array::VirtArray;

pub type Bytecode = [u8];

const POOLS: usize = 2;
/// A power of two, and small, so the program crosses the ring boundary in both
/// directions several times per iteration instead of once per run.
const CAP: usize = 4;
const CAP_MASK: usize = CAP - 1;

const OP_PUSH: u8 = 1;
const OP_DRAIN: u8 = 2;
const OP_SWITCH: u8 = 3;
const OP_BACK: u8 = 4;
const OP_RET: u8 = 5;

static COMPILES: AtomicU32 = AtomicU32::new(0);

/// The overflow half of the stack. Headed by a raw chain so a push and a pop
/// are the plainest possible residual calls.
#[repr(C)]
struct Chain {
    head: *mut Node,
    size: i64,
}

#[repr(C)]
struct Node {
    value: i64,
    next: *mut Node,
}

extern "C" fn chain_push(chain: usize, value: i64) {
    let chain = unsafe { &mut *(chain as *mut Chain) };
    let node = Box::into_raw(Box::new(Node {
        value,
        next: chain.head,
    }));
    chain.head = node;
    chain.size += 1;
}

extern "C" fn chain_pop(chain: usize) -> i64 {
    let chain = unsafe { &mut *(chain as *mut Chain) };
    assert!(
        !chain.head.is_null(),
        "the interpreter reached the chain with nothing on it: the restored \
         `sp` claims the ring has spilled words that were never pushed"
    );
    let node = unsafe { Box::from_raw(chain.head) };
    chain.head = node.next;
    chain.size -= 1;
    node.value
}

struct SplitStack {
    /// `pool * CAP + (h & CAP_MASK)` — the top `CAP` words of each pool.
    vals: VirtArray<i64>,
    /// One parked height per pool. Shorter than `vals`.
    depths: VirtArray<i64>,
    sel: usize,
    /// The selected pool's total height, ring and chain together. The chain
    /// holds `max(0, sp - CAP)` words.
    sp: usize,
    counter: i64,
    acc: i64,
    /// Declared last, after both arrays: a ref-kind slot at the far end of the
    /// flat layout is the one whose index depends on every length before it.
    chain: usize,
}

#[majit_macros::jit_interp(
    state = SplitStack,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        vals: [int; virt],
        depths: [int; virt],
        sel: int(usize),
        sp: int(usize),
        counter: int,
        acc: int,
        chain: ref(Chain),
    },
    calls = {
        chain_push => residual_void,
        chain_pop => residual_int,
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, iterations: i64, threshold: u32, chain: usize) -> i64 {
    let mut driver: majit_metainterp::JitDriver<SplitStack> =
        majit_metainterp::JitDriver::new(threshold);
    driver.set_on_compile_loop(|_green_key, _before, _after, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = SplitStack {
        vals: VirtArray::filled(0i64, POOLS * CAP),
        depths: VirtArray::filled(0i64, POOLS),
        sel: 0usize,
        sp: 0usize,
        counter: iterations,
        acc: 0i64,
        chain,
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
                let value = program[pc] as i64;
                pc += 1;
                state.sp += 1;
                // `sp` is the height after the push, so the new word takes the
                // slot of height `sp - 1`.
                let free_slot = state.sel * CAP + ((state.sp - 1) & CAP_MASK);
                if state.sp > CAP {
                    // The ring is full: the word in this slot is the pool's
                    // oldest and leaves for the chain.
                    chain_push(state.chain, state.vals[free_slot]);
                }
                state.vals[free_slot] = value;
            }
            OP_DRAIN => {
                let top_slot = state.sel * CAP + ((state.sp - 1) & CAP_MASK);
                state.acc += state.vals[top_slot];
                state.sp -= 1;
                // The slot just vacated is the one height `sp - CAP` wants
                // back; `sp & CAP_MASK` names it because the ring wraps.
                let refill_slot = state.sel * CAP + (state.sp & CAP_MASK);
                if state.sp >= CAP {
                    state.vals[refill_slot] = chain_pop(state.chain);
                }
            }
            OP_SWITCH => {
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

/// Pool 0 is driven `CAP + 2` deep, so two words spill to the chain; the run
/// then leaves it parked across a switch, works pool 1 entirely inside its
/// ring, and only afterwards drains pool 0 back down through both refills.
///
/// Pool 1 never spills. That asymmetry is deliberate: the chain belongs to
/// pool 0, and a machine where both pools spilled would need one chain each
/// before the answer meant anything.
fn program() -> Vec<u8> {
    vec![
        OP_PUSH, 1, // pool 0, heights 0..5 — the first two spill on the way up
        OP_PUSH, 2, OP_PUSH, 3, OP_PUSH, 4, OP_PUSH, 5, // height 4: spills height 0
        OP_PUSH, 6,         // height 5: spills height 1
        OP_SWITCH, // park pool 0 at 6, select pool 1
        OP_PUSH, 7, OP_PUSH, 8, OP_DRAIN,  // 8
        OP_DRAIN,  // 7
        OP_SWITCH, // park pool 1 at 0, take pool 0's 6 back
        OP_DRAIN,  // 6, then refill slot 1 from the chain
        OP_DRAIN,  // 5, then refill slot 0 from the chain
        OP_DRAIN,  // 4
        OP_DRAIN,  // 3
        OP_DRAIN,  // 2 — came back from the chain
        OP_DRAIN,  // 1 — came back from the chain
        OP_BACK, OP_RET,
    ]
}

/// 1+2+3+4+5+6 from pool 0 and 7+8 from pool 1. Every term is distinct, so a
/// read that lands one slot off changes the sum instead of repeating a value.
const PER_ITERATION: i64 = 36;

fn run(program: &[u8], iterations: i64, threshold: u32) -> (i64, i64) {
    let mut chain = Chain {
        head: core::ptr::null_mut(),
        size: 0,
    };
    let acc = mainloop(
        program,
        iterations,
        threshold,
        &mut chain as *mut Chain as usize,
    );
    (acc, chain.size)
}

#[test]
fn a_ring_that_spills_to_a_ref_chain_agrees_with_the_interpreter() {
    let program = program();
    const ITERATIONS: i64 = 200;

    COMPILES.store(0, Ordering::Relaxed);
    // Unreachable threshold, so this run fixes the answer without the JIT.
    let (cold, cold_left) = run(&program, ITERATIONS, u32::MAX);
    assert_eq!(cold, ITERATIONS * PER_ITERATION, "cold interpretation");
    assert_eq!(cold_left, 0, "cold run left words on the chain");
    assert_eq!(
        COMPILES.load(Ordering::Relaxed),
        0,
        "the cold run must not have compiled anything"
    );

    let (warm, warm_left) = run(&program, ITERATIONS, 3);
    assert_eq!(
        warm, cold,
        "a virtualizable ring spilling through a `ref(T)` state field must read \
         the same words compiled as interpreted"
    );
    assert_eq!(
        warm_left, 0,
        "the compiled run left words on the chain: the ring and the chain \
         disagree about how deep the stack is"
    );
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "the split-stack loop did not compile"
    );
}
