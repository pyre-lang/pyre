//! A recursive-portal-capable state whose `[.. ; virt]` field is block-backed.
//!
//! The recursive `CALL_ASSEMBLER` portal cannot `New` a host state through the
//! IR, so the generated interpreter emits a pair of host helpers the compiled
//! caller calls residually: one allocates a fresh state sized at the caller's
//! live capacity, the other drops it. That allocator is generated only for the
//! single-virt-array shape declared here — no ref scalars, no float scalars, no
//! opaque carriers and no fixed arrays.
//!
//! The fixture exists because that allocator builds the whole state by struct
//! literal, so each field's initializer has to be spelled in the container the
//! field was declared with. A `[.. ; virt]` field may be declared with either a
//! `Vec` or a block-backed `VirtArray`, and only the block-backed spelling
//! catches an initializer that hardcodes one of the two: with a `Vec` field a
//! `vec![]` initializer compiles whether or not it consulted the declaration.

use majit_metainterp::virt_array::VirtArray;

pub type Bytecode = [u8];

const OP_PUSH: u8 = 1; // regs[top] = top as i64; top += 1
const OP_ADD: u8 = 2; // top -= 2; regs[top] += regs[top + 1]; top += 1
const OP_JUMP_BACK: u8 = 3; // [OP_JUMP_BACK, target]: loop while regs[0] < limit
const OP_RETURN: u8 = 4; // yield regs[top - 1]

fn accumulate_program() -> Vec<u8> {
    vec![OP_PUSH, OP_ADD, OP_JUMP_BACK, 0, OP_RETURN]
}

struct Machine {
    top: i64,
    regs: VirtArray<i64>,
}

#[majit_macros::jit_interp(
    state = Machine,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        top: int,
        regs: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn mainloop(program: &Bytecode, limit: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<Machine> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = Machine {
        top: 1,
        regs: VirtArray::filled(0i64, 8),
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
                state.regs[state.top as usize] = 1;
                state.top += 1;
            }
            OP_ADD => {
                state.top -= 2;
                let rhs = state.regs[(state.top + 1) as usize];
                state.regs[state.top as usize] += rhs;
                state.top += 1;
            }
            OP_JUMP_BACK => {
                let target = program[pc] as usize;
                pc += 1;
                if state.regs[0] < limit {
                    if target < pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                }
            }
            OP_RETURN => return state.regs[(state.top - 1) as usize],
            _ => unreachable!(),
        }
    }
    state.regs[0]
}

/// The machine answers the same warm and cold, so the fixture is not vacuous:
/// the generated interpreter it compiles is the one under test.
#[test]
fn a_block_backed_recursive_capable_machine_interprets_the_same_program() {
    let program = accumulate_program();
    assert_eq!(mainloop(&program, 5, u32::MAX), 5, "never-warm run");
    assert_eq!(mainloop(&program, 120, 3), 120, "warm run");
}

/// The host allocator builds a fresh state at a requested capacity and the
/// matching free reclaims it. Reaching the pair at all is the regression: the
/// allocator is generated for this state shape, so if its array initializer
/// were spelled as a `Vec` the fixture would not compile.
#[test]
fn the_generated_fresh_allocator_sizes_a_block_backed_array() {
    use majit_metainterp::{JitCodeSym as _, JitState as _};
    let program = accumulate_program();
    let caller = Machine {
        top: 3,
        regs: VirtArray::from_slice(&[9i64, 8, 7, 0, 0, 0]),
    };
    let meta = caller.build_meta(0, program.as_slice());
    let mut sym = <Machine as majit_metainterp::JitState>::create_sym(&meta, 0);
    caller.initialize_sym(&mut sym, &meta);
    let (alloc_fp, free_fp) = sym
        .recursive_fresh_alloc_free_targets()
        .expect("a single-virt-array state supports portal alloc/free");
    let alloc: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(alloc_fp) };
    let free: extern "C" fn(i64) = unsafe { core::mem::transmute(free_fp) };

    let cap: i64 = 12;
    let raw = alloc(cap);
    assert_ne!(raw, 0, "fresh alloc must return a non-null pointer");
    unsafe {
        let fresh = &*(raw as *const Machine);
        assert_eq!(fresh.top, 0, "fresh scalars are zeroed");
        assert_eq!(
            fresh.regs.len(),
            cap as usize,
            "fresh array sized at the requested capacity",
        );
        assert!(
            fresh.regs.iter().all(|&x| x == 0),
            "fresh array zero-initialised",
        );
    }
    free(raw);
    // The compiled guard-fail path may reach the free with nothing allocated.
    free(0);
}

/// The fresh reds path builds the same state through the same declared backing,
/// at the capacity the caller's array carries.
#[test]
fn the_fresh_reds_path_reallocates_at_the_callers_capacity() {
    use majit_metainterp::{JitCodeSym as _, JitState as _};
    let program = accumulate_program();
    let caller = Machine {
        top: 3,
        regs: VirtArray::from_slice(&[9i64, 8, 7, 0, 0, 0]),
    };
    let meta = caller.build_meta(0, program.as_slice());
    let mut sym = <Machine as majit_metainterp::JitState>::create_sym(&meta, 0);
    caller.initialize_sym(&mut sym, &meta);
    let (_values, owner) = sym
        .recursive_fresh_entry_reds()
        .expect("a state with no ref scalars and no opaque carriers has fresh reds");
    let fresh = owner
        .downcast_ref::<Machine>()
        .expect("the owner is the fresh state the reds name");
    assert_eq!(fresh.top, 0, "fresh frame starts empty");
    assert_eq!(
        fresh.regs.len(),
        caller.regs.len(),
        "fresh array re-allocated at the caller's captured capacity",
    );
}
