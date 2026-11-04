//! A virtualizable array used as an operand stack, indexed by a runtime depth.
//!
//! `pyjitpl.py _get_arrayitem_vable_index` promotes the index of a
//! virtualizable array access with `implement_guard_value` and then answers
//! from `virtualizable_boxes`. So a stack whose index is the machine's own
//! depth — never a constant in the source — still lowers to boxes rather than
//! to memory: the element accesses leave no `getarrayitem`/`setarrayitem`
//! behind at all.
//!
//! That is the property a heap array cannot have. `optimizeopt/heap.rs`'s
//! array-item cache answers for constant indices only, so the same machine
//! written over a heap array keeps every store and every load.
//!
//! The second fixture holds the program fixed and varies only the DECLARED
//! length of the array, which is what a virtualizable's box count scales with
//! (`virtualizable.py get_total_size`). The compiled loop must not
//! grow with it.

use core::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::Mutex;

use majit_ir::OpCode;
use majit_metainterp::virt_array::VirtArray;

pub type Bytecode = [u8];

/// `stack[sp] = 1; sp += 1`
const OP_PUSH1: u8 = 1;
/// `stack[sp] = 2; sp += 1`
const OP_PUSH2: u8 = 2;
/// `sp -= 1; stack[sp - 1] += stack[sp]`
const OP_ADD: u8 = 3;
/// `sp -= 1; acc += stack[sp]`
const OP_DRAIN: u8 = 4;
/// `counter -= 1`; back edge to 0 while non-zero
const OP_BACK: u8 = 5;
/// yield `acc`
const OP_RET: u8 = 6;

/// One iteration pushes two values, adds them, and drains the sum into `acc`,
/// so the stack is balanced across the back edge and `acc` grows by 3.
fn balanced_program() -> Vec<u8> {
    vec![OP_PUSH1, OP_PUSH2, OP_ADD, OP_DRAIN, OP_BACK, OP_RET]
}

/// Opcodes that read or write memory the trace could have kept in a box.
fn memory_ops(opcodes: &[OpCode]) -> usize {
    opcodes
        .iter()
        .filter(|op| {
            matches!(
                op,
                OpCode::SetarrayitemGc
                    | OpCode::GetarrayitemGcI
                    | OpCode::GetarrayitemGcR
                    | OpCode::GetarrayitemGcF
            )
        })
        .count()
}

macro_rules! stack_machine {
    ($name:ident, $decl:tt, $container:ty, $init:expr) => {
        pub mod $name {
            use super::{
                AtomicUsize, Bytecode, Mutex, OP_ADD, OP_BACK, OP_DRAIN, OP_PUSH1, OP_PUSH2,
                OP_RET, Ordering, VirtArray,
            };

            /// The optimized opcode list of the last loop this machine compiled.
            pub static COMPILED: Mutex<Vec<majit_ir::OpCode>> = Mutex::new(Vec::new());
            pub static COMPILES: AtomicUsize = AtomicUsize::new(0);
            /// Ops recorded before the optimizer ran, for the last loop.
            pub static RECORDED: AtomicUsize = AtomicUsize::new(0);

            struct StackMachine {
                stack: $container,
                sp: usize,
                counter: i64,
                acc: i64,
            }

            #[majit_macros::jit_interp(
                state = StackMachine,
                env = Bytecode,
                greens = [pc, program],
                state_fields = {
                    stack: $decl,
                    sp: int(usize),
                    counter: int,
                    acc: int,
                },
            )]
            #[allow(unused_assignments, unused_variables)]
            pub fn mainloop(program: &Bytecode, iterations: i64, threshold: u32) -> i64 {
                let mut driver: majit_metainterp::JitDriver<StackMachine> =
                    majit_metainterp::JitDriver::new(threshold);
                driver.set_on_compile_loop(|_green_key, before, _after, opcodes| {
                    COMPILES.fetch_add(1, Ordering::Relaxed);
                    RECORDED.store(before, Ordering::Relaxed);
                    *COMPILED.lock() = opcodes.to_vec();
                });
                let mut pc: usize = 0;
                let mut state = StackMachine {
                    stack: $init,
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
                        OP_PUSH1 => {
                            state.stack[state.sp] = 1;
                            state.sp += 1;
                        }
                        OP_PUSH2 => {
                            state.stack[state.sp] = 2;
                            state.sp += 1;
                        }
                        OP_ADD => {
                            let top = state.stack[state.sp - 1];
                            state.sp -= 1;
                            state.stack[state.sp - 1] += top;
                        }
                        OP_DRAIN => {
                            state.sp -= 1;
                            state.acc += state.stack[state.sp];
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
        }
    };
}

stack_machine!(
    eight_slots,
    [int; virt],
    VirtArray<i64>,
    VirtArray::filled(0i64, 8)
);
stack_machine!(
    five_hundred_slots,
    [int; virt],
    VirtArray<i64>,
    VirtArray::filled(0i64, 512)
);
// A plain `[int]` array is not a control here: `jit_interp` refuses one that is
// loop-carried, because a loop-carried plain element does not read its heap
// value back. The control for "a heap array keeps its accesses" is the aheui
// array backend, whose logo loop holds 2055 `setarrayitem` and 1306
// `getarrayitem` against this shape's none.

/// The machine computes the same answer warm and cold.
#[test]
fn a_vable_stack_machine_interprets_its_program() {
    let program = balanced_program();
    assert_eq!(eight_slots::mainloop(&program, 5, u32::MAX), 15, "cold");
    assert_eq!(eight_slots::mainloop(&program, 200, 3), 600, "warm");
}

/// The payoff: a stack indexed by a runtime depth leaves no element access in
/// the compiled loop.
///
/// A zero that cannot fail proves nothing, so the same program over a plain
/// heap array is the control: it must report a nonzero count, and the loop
/// must be non-trivial in both arms.
#[test]
fn a_vable_stack_indexed_by_a_runtime_depth_lowers_to_no_memory_ops() {
    let program = balanced_program();
    assert_eq!(eight_slots::mainloop(&program, 400, 3), 1200);
    assert!(
        eight_slots::COMPILES.load(Ordering::Relaxed) > 0,
        "the fixture must compile a loop for the census to mean anything"
    );

    let vable = eight_slots::COMPILED.lock().clone();
    eprintln!(
        "vable stack loop: {} recorded -> {} ops, {} of them memory",
        eight_slots::RECORDED.load(Ordering::Relaxed),
        vable.len(),
        memory_ops(&vable),
    );
    // An empty loop would report zero memory ops vacuously.
    assert!(
        vable.iter().any(|op| matches!(op, OpCode::IntAdd)),
        "the loop must still carry the machine's arithmetic; loop was {vable:?}"
    );
    // `patch_new_loop_to_load_virtualizable_fields` prepends GETFIELD /
    // GETARRAYITEM after optimize, so the assembled list can be longer
    // than the recording. The optimizer's work is the zero memory-ops
    // count below, not a shorter op list.
    assert_eq!(
        memory_ops(&vable),
        0,
        "vable stack element accesses must not survive as memory ops; loop was {vable:?}"
    );
}

/// The declared length drives the virtualizable's box count, not the loop.
#[test]
fn the_compiled_loop_does_not_grow_with_the_declared_array_length() {
    let program = balanced_program();
    assert_eq!(eight_slots::mainloop(&program, 400, 3), 1200);
    assert_eq!(five_hundred_slots::mainloop(&program, 400, 3), 1200);
    let small = eight_slots::COMPILED.lock().len();
    let large = five_hundred_slots::COMPILED.lock().len();
    assert_eq!(
        small, large,
        "a 512-slot declaration compiled {large} ops against the 8-slot {small}"
    );
}
