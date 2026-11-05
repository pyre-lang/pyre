//! A virtualizable ring whose index ADVANCES, spilling every evicted word to a
//! chain reached through a `ref(T)` state field.
//!
//! `jit_interp_virt_array_spills_to_ref_chain` covers the same split stack, but
//! its program returns to the same depth every iteration, so the ring slot each
//! access reaches is the same on every trip. A vable array's index is promoted,
//! and a promoted value that never changes is free: the guard is recorded once
//! and passes forever. That is the blind spot. A stack that only grows reaches
//! a DIFFERENT slot each trip, so the promote's guard fails every iteration,
//! and each failure has to resume at the boundary of the opcode that computed
//! the index — before the eviction the same opcode still owes the chain.
//!
//! It measured 201 of 1992 before the fix. It is the virtualizable-array face
//! of the defect
//! `jit_interp_bridge_drops_opcode_tail` reproduces with one scalar field, and
//! it is kept because the ring is what makes the guard fail EVERY trip: the
//! promoted index is a different value each time, so the branch behind it never
//! settles.
//!
//! What makes a missed resume invisible without this fixture is that nothing
//! observable goes wrong at the moment it happens: the ring keeps answering,
//! the chain stays internally consistent, and the loop keeps running. Only the
//! COUNT of words that reached the chain records it. So the assertion is
//! arithmetic on the whole run rather than a crash: after `N` pushes the chain
//! must hold exactly `N - CAP` words, and their sum must be the sum of the
//! first `N - CAP` values pushed — one skipped eviction changes both.
use core::sync::atomic::{AtomicU32, Ordering};

use majit_metainterp::virt_array::VirtArray;

pub type Bytecode = [u8];

/// A power of two, and small, so the ring index takes every one of its values
/// within a few trips rather than once per run.
const CAP: usize = 8;
const CAP_MASK: usize = CAP - 1;

const OP_PUSH: u8 = 1;
const OP_BACK: u8 = 2;
const OP_RET: u8 = 3;

static COMPILES: AtomicU32 = AtomicU32::new(0);

/// The overflow half of the stack. It keeps a count and a sum instead of nodes
/// because the test grades how many words arrived and which, not their order.
#[repr(C)]
struct Chain {
    size: i64,
    sum: i64,
}

extern "C" fn chain_push(chain: usize, value: i64) {
    let chain = unsafe { &mut *(chain as *mut Chain) };
    chain.size += 1;
    chain.sum += value;
}

struct GrowingStack {
    /// The top `CAP` words: the word at absolute height `h` is at
    /// `(h - 1) & CAP_MASK`.
    vals: VirtArray<i64>,
    /// The whole height, ring and chain together. The chain holds
    /// `max(0, sp - CAP)` words.
    sp: usize,
    counter: i64,
    /// Declared after the array, so its slot index depends on the array's
    /// declared length.
    chain: usize,
}

#[majit_macros::jit_interp(
    state = GrowingStack,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        vals: [int; virt],
        sp: int(usize),
        counter: int,
        chain: ref(Chain),
    },
    calls = {
        chain_push => residual_void,
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, iterations: i64, threshold: u32, chain: usize) -> i64 {
    let mut driver: majit_metainterp::JitDriver<GrowingStack> =
        majit_metainterp::JitDriver::new(threshold);
    driver.set_on_compile_loop(|_green_key, _before, _after, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = GrowingStack {
        vals: VirtArray::filled(0i64, CAP),
        sp: 0usize,
        counter: iterations,
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
                // A red, and a different one every trip, so a word that never
                // reaches the chain is missing from the sum rather than hidden
                // by a repeat.
                let value = state.counter;
                state.sp += 1;
                let free_slot = (state.sp - 1) & CAP_MASK;
                if state.sp > CAP {
                    // The ring is full: the word in this slot is the oldest
                    // one it holds and leaves for the chain. The eviction and
                    // the store below are the same opcode's work, so a guard
                    // between them owes its resume the whole pair.
                    chain_push(state.chain, state.vals[free_slot]);
                }
                state.vals[free_slot] = value;
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

    state.sp as i64
}

fn program() -> Vec<u8> {
    vec![OP_PUSH, OP_BACK, OP_RET]
}

fn run(program: &[u8], iterations: i64, threshold: u32) -> (i64, i64, i64) {
    let mut chain = Chain { size: 0, sum: 0 };
    let height = mainloop(
        program,
        iterations,
        threshold,
        &mut chain as *mut Chain as usize,
    );
    (height, chain.size, chain.sum)
}

#[test]

fn an_advancing_ring_index_spills_every_word_it_evicts() {
    let program = program();
    // Deep enough that the ring index has cycled hundreds of times after the
    // loop is compiled, so a per-iteration loss accumulates into a plain
    // arithmetic difference.
    const ITERATIONS: i64 = 2000;

    COMPILES.store(0, Ordering::Relaxed);
    // Unreachable threshold, so this run fixes the answer without the JIT.
    let cold = run(&program, ITERATIONS, u32::MAX);
    assert_eq!(
        COMPILES.load(Ordering::Relaxed),
        0,
        "the cold run must not have compiled anything"
    );
    assert_eq!(cold.0, ITERATIONS, "cold height");
    assert_eq!(
        cold.1,
        ITERATIONS - CAP as i64,
        "cold run: every word but the ring's last {CAP} belongs to the chain"
    );

    let warm = run(&program, ITERATIONS, 3);
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "the growing-ring loop did not compile"
    );
    assert_eq!(
        warm.1, cold.1,
        "compiled run spilled a different NUMBER of words than the interpreter: \
         a push whose ring-index promote failed did not finish its eviction"
    );
    assert_eq!(
        warm.2, cold.2,
        "compiled run spilled different WORDS than the interpreter"
    );
    assert_eq!(warm.0, cold.0, "compiled height");
}
