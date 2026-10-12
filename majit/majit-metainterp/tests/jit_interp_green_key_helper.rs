//! `__majit_green_key_<fn>` must name the cell the merge point files under.
//!
//! A door outside the mainloop — anything asking "is this position already
//! compiled?" — has to build the merge point's key itself, and before the macro
//! emitted this the only way was to write the layout out by hand: the seed, one
//! `green_uhash_step` per declared green, in declaration order. A consumer
//! that got a slot wrong still hashed to something, just not to the cell it
//! meant, and the failure is silent and total: the probe answers no forever, so
//! the door never enters an artifact that exists and every counter stays
//! plausible.
//!
//! Asserting the hash against a literal would only pin whatever the helper
//! currently computes. What decides it is the driver: run until a loop compiles
//! at a known back edge, then ask `has_compiled_loop` with the helper's hash.
//! That is the question a door asks, against the key the compiler actually
//! filed.

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use majit_metainterp::JitDriver;
use majit_metainterp::virt_array::VirtArray;

/// What the run's own probe answered, and how many loops it compiled.
///
/// Statics rather than a tuple return: a `#[jit_interp]` mainloop hands back
/// one value, and the shape of that value is part of what the macro lowers.
static PROBE_FOUND: AtomicBool = AtomicBool::new(false);
static LOOPS_COMPILED: AtomicUsize = AtomicUsize::new(0);

pub type Bytecode = [u8];

/// `regs[0] -= 1`, advance.
const OP_DEC: u8 = 1;
/// Back edge: jump to 0 while `regs[0] != 0`, else fall past.
const OP_BACK: u8 = 2;
const OP_END: u8 = 3;

/// Where the back edge lands, and so the position whose key the loop is filed
/// under.
const LOOP_HEADER: usize = 0;

struct GreenKeyState {
    regs: VirtArray<i64>,
}

#[majit_macros::jit_interp(
    state = GreenKeyState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_green_key(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<GreenKeyState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = GreenKeyState {
        regs: VirtArray::filled(0i64, 1),
    };
    state.regs[0] = program[program.len() - 1] as i64;
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    loop {
        jit_merge_point!(driver, program, pc; state);
        let opcode = program[pc];
        match opcode {
            OP_DEC => {
                state.regs[0] = state.regs[0] - 1;
                pc = pc + 1;
            }
            OP_BACK => {
                if state.regs[0] != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = LOOP_HEADER;
                    continue;
                } else {
                    pc = pc + 1;
                }
            }
            _ => break,
        }
    }
    // The probe under test, asked from inside so it can reach the driver this
    // run compiled against — a door in a real embedder holds its own.
    let (hash, slots) = __majit_green_key_dispatch_green_key(LOOP_HEADER, program);
    assert_eq!(
        slots.len(),
        2,
        "greens = [pc, program] is two declared slots; a third is the retired prepended-target layout"
    );
    PROBE_FOUND.store(driver.has_compiled_loop(hash), Ordering::Relaxed);
    LOOPS_COMPILED.store(driver.get_stats().loops_compiled, Ordering::Relaxed);
    state.regs[0]
}

/// `regs[0]` counts down from the trailing byte, so the loop runs it to zero.
fn program() -> Vec<u8> {
    vec![OP_DEC, OP_BACK, OP_END, 40]
}

#[test]
fn the_generated_green_key_names_the_cell_the_merge_point_filed() {
    let answer = dispatch_green_key(&program(), 3);
    assert_eq!(answer, 0, "the loop must have run to completion");
    assert!(
        LOOPS_COMPILED.load(Ordering::Relaxed) > 0,
        "nothing compiled, so this says nothing about the key"
    );
    assert!(
        PROBE_FOUND.load(Ordering::Relaxed),
        "a loop compiled at pc={LOOP_HEADER}, but the key \
         `__majit_green_key_dispatch_green_key` builds for that position does not \
         find it — the helper's slot order or hash recipe disagrees with the \
         merge point's"
    );
}

/// The untraced tier must reach the same answer, so the assertion above is
/// about the KEY and not about the loop having run at all.
#[test]
fn the_untraced_tier_agrees() {
    let answer = dispatch_green_key(&program(), u32::MAX);
    assert_eq!(answer, 0);
    assert!(
        !PROBE_FOUND.load(Ordering::Relaxed),
        "nothing compiles at an unreachable threshold"
    );
}
