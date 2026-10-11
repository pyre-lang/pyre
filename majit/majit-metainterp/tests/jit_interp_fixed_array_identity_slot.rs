//! Verifies virtualizable identity handling when a plain `[int]` array appears
//! before a virtualizable array. The `[int]` is ptr then len, so the
//! identity slot is a constant (`warmspot.py` `index_of_virtualizable`).

use core::sync::atomic::{AtomicU32, Ordering};

pub type Bytecode = [u8];

const OP_ACC: u8 = 1; // iregs[2] += cells[0]  (the fixed array, read-only)
const OP_INC: u8 = 2; // iregs[0] += 1
const OP_JUMP_BACK: u8 = 3; // [OP_JUMP_BACK, target]: loop while iregs[0] < iregs[1]
const OP_RETURN: u8 = 4; // yield iregs[0]

/// `iregs = [counter, limit, accumulator]`.
fn count_program() -> Vec<u8> {
    vec![OP_ACC, OP_INC, OP_JUMP_BACK, 0, OP_RETURN]
}

/// What every machine here must return for a given limit.
fn expected(limit: i64) -> i64 {
    limit
}

/// A plain `[int]` array between the loop-dead `int` scalar and the
/// `[int; virt]` array. `ret` sits on the virtualizable with `iregs`
/// (`virtualizable.py VirtualizableInfo.__init__`), so `extract_live`
/// emits `[cells items-pointer, cells len, identity]`. The identity is at
/// flat slot 2 — not at 0, and not at `cells.len()`.
mod fixed_array_before_virt_array {
    use super::{AtomicU32, Bytecode, OP_ACC, OP_INC, OP_JUMP_BACK, OP_RETURN, Ordering};
    use majit_metainterp::JitState as _;
    use majit_metainterp::virt_array::VirtArray;

    static COMPILES: AtomicU32 = AtomicU32::new(0);

    struct FixedAndVirt {
        ret: i64,
        cells: Vec<i64>,
        iregs: VirtArray<i64>,
    }

    #[majit_macros::jit_interp(
        state = FixedAndVirt,
        env = Bytecode,
        greens = [pc, program],
        state_fields = {
            ret: int,
            cells: [int],
            iregs: [int; virt],
        },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn mainloop(program: &Bytecode, limit: i64, threshold: u32) -> i64 {
        let mut driver: majit_metainterp::JitDriver<FixedAndVirt> =
            majit_metainterp::JitDriver::new(threshold);
        driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
            COMPILES.fetch_add(1, Ordering::Relaxed);
        });
        let mut pc: usize = 0;
        let mut state = FixedAndVirt {
            ret: 0,
            cells: vec![0i64, 0i64],
            iregs: VirtArray::from_slice(&[0i64, limit, 0i64]),
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
                // Read-only: a MUTATED plain `[int]` is refused outright (it is
                // not restored on deopt), so the fixed array is loop-carried but
                // never written.
                OP_ACC => state.iregs[2] = state.iregs[2] + state.cells[0],
                OP_INC => state.iregs[0] = state.iregs[0] + 1,
                OP_JUMP_BACK => {
                    let target = program[pc] as usize;
                    pc += 1;
                    if state.iregs[0] < state.iregs[1] {
                        if target < pc {
                            can_enter_jit!(driver, target, &mut state, program, || {});
                        }
                        pc = target;
                        continue;
                    }
                }
                OP_RETURN => {
                    state.ret = state.iregs[0];
                    return state.ret;
                }
                _ => panic!("bad opcode {opcode}"),
            }
        }
        state.ret
    }

    /// Ground truth for the slot the identity occupies. `ret` is a
    /// virtualizable static, so the red block is the plain `[int]`
    /// items-pointer then length then the identity. No `iregs` element
    /// belongs in the red block, and ptr+len is two slots regardless of
    /// `cells.len()`.
    #[test]
    fn the_identity_sits_past_the_plain_int_array_pointer() {
        let state = FixedAndVirt {
            ret: 0,
            cells: vec![0i64, 0i64],
            iregs: VirtArray::from_slice(&[0i64, 4, 0i64]),
        };
        let program: &Bytecode = &super::count_program();
        let meta = state.build_meta(0, program);

        assert_eq!(
            state.live_value_types(&meta),
            vec![
                majit_ir::Type::Int, // cells items-pointer
                majit_ir::Type::Int, // cells length
                majit_ir::Type::Ref, // vable identity
            ],
            "the identity must follow ptr then len, so its flat slot is 2 — \
             independent of cells.len()",
        );
        let live = state.extract_live(&meta);
        assert_eq!(live.len(), 3);
        assert_eq!(
            live[0],
            state.cells.as_ptr() as i64,
            "the [int] first red is the Vec items pointer"
        );
        assert_eq!(
            live[1],
            state.cells.len() as i64,
            "the [int] second red is the Vec length"
        );
    }

    #[test]
    fn a_fixed_array_beside_a_virt_array_compiles_the_hot_loop() {
        let program = super::count_program();
        let limit = 50i64;
        COMPILES.store(0, Ordering::Relaxed);

        let got = mainloop(&program, limit, 3);

        assert_eq!(
            got,
            super::expected(limit),
            "scalar carried the wrong result out"
        );
        assert!(
            COMPILES.load(Ordering::Relaxed) >= 1,
            "declaring a plain `[int]` array beside a `[int; virt]` one must \
             still compile the hot loop: identity_live_index is the constant \
             after ptr then len"
        );
    }
}
