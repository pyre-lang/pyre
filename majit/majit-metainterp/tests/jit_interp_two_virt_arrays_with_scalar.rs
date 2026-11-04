//! Verifies that virtualizable identity lookup uses its declared input slot
//! rather than assuming slot zero.
//!
//! State-field layouts place integer scalars before the virtualizable identity.
//! The fixture varies array count and scalar liveness to ensure a scalar is not
//! misclassified as the identity reference.

use core::sync::atomic::{AtomicU32, Ordering};

pub type Bytecode = [u8];

const OP_ACC: u8 = 1; // advance the machine's accumulator
const OP_INC: u8 = 2; // iregs[0] += 1
const OP_JUMP_BACK: u8 = 3; // [OP_JUMP_BACK, target]: loop while iregs[0] < iregs[1]
const OP_RETURN: u8 = 4; // yield iregs[0]

/// `iregs = [counter, limit]`; the accumulator is one element or one scalar.
fn count_program() -> Vec<u8> {
    vec![OP_ACC, OP_INC, OP_JUMP_BACK, 0, OP_RETURN]
}

/// What every machine here must return for a given limit.
fn expected(limit: i64) -> i64 {
    limit
}

/// Control with two virtualizable arrays and no scalar preceding the identity.
mod two_arrays_no_scalar {
    use super::{AtomicU32, Bytecode, OP_ACC, OP_INC, OP_JUMP_BACK, OP_RETURN, Ordering};
    use majit_metainterp::virt_array::VirtArray;

    static COMPILES: AtomicU32 = AtomicU32::new(0);

    struct TwoArrays {
        iregs: VirtArray<i64>,
        fregs: VirtArray<f64>,
    }

    #[majit_macros::jit_interp(
        state = TwoArrays,
        env = Bytecode,
        greens = [pc, program],
        state_fields = {
            iregs: [int; virt],
            fregs: [float; virt],
        },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn mainloop(program: &Bytecode, limit: i64, threshold: u32) -> i64 {
        let mut driver: majit_metainterp::JitDriver<TwoArrays> =
            majit_metainterp::JitDriver::new(threshold);
        driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
            COMPILES.fetch_add(1, Ordering::Relaxed);
        });
        let mut pc: usize = 0;
        let mut state = TwoArrays {
            iregs: VirtArray::from_slice(&[0i64, limit]),
            fregs: VirtArray::from_slice(&[0.0f64]),
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
                OP_ACC => state.fregs[0] += 1.25,
                OP_INC => state.iregs[0] += 1,
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
                OP_RETURN => break,
                _ => panic!("bad opcode {opcode}"),
            }
        }
        // A constant index into a transferred virt array is readable after the
        // loop; the walk-final element value arrives via
        // `writeback_virt_array_state_fields`.
        state.iregs[0]
    }

    #[test]
    fn two_virt_arrays_without_a_scalar_compile_the_hot_loop() {
        let program = super::count_program();
        let limit = 50i64;
        COMPILES.store(0, Ordering::Relaxed);

        let got = mainloop(&program, limit, 3);

        assert_eq!(
            got,
            super::expected(limit),
            "two-array machine answered wrong"
        );
        assert!(
            COMPILES.load(Ordering::Relaxed) >= 1,
            "the two-array control never compiled its hot loop, so the \
             experiments below have no baseline to be compared against"
        );
    }
}

/// One virt array and two scalars. `colscalar` in `majit/examples/cel` is this
/// shape and runs today; it is repeated here so the failing machine below
/// differs from a passing one in exactly one declaration.
mod one_array_with_scalar {
    use super::{AtomicU32, Bytecode, OP_ACC, OP_INC, OP_JUMP_BACK, OP_RETURN, Ordering};
    use majit_metainterp::virt_array::VirtArray;

    static COMPILES: AtomicU32 = AtomicU32::new(0);

    struct OneArrayScalar {
        iregs: VirtArray<i64>,
        acc: i64,
        ret: i64,
    }

    #[majit_macros::jit_interp(
        state = OneArrayScalar,
        env = Bytecode,
        greens = [pc, program],
        state_fields = {
            iregs: [int; virt],
            acc: int,
            ret: int,
        },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn mainloop(program: &Bytecode, limit: i64, threshold: u32) -> i64 {
        let mut driver: majit_metainterp::JitDriver<OneArrayScalar> =
            majit_metainterp::JitDriver::new(threshold);
        driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
            COMPILES.fetch_add(1, Ordering::Relaxed);
        });
        let mut pc: usize = 0;
        let mut state = OneArrayScalar {
            iregs: VirtArray::from_slice(&[0i64, limit]),
            acc: 0,
            ret: 0,
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
                OP_ACC => state.acc += 2,
                OP_INC => state.iregs[0] += 1,
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

    #[test]
    fn one_virt_array_with_scalars_compiles_the_hot_loop() {
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
            "one virt array beside two `int` scalars did not compile — that \
             combination has shipping consumers, so a failure here means the \
             fixture is wrong, not the combination"
        );
    }
}

/// Two virt arrays AND a scalar — the combination with no consumer. Identical to
/// `two_arrays_no_scalar` except that the result leaves the loop in a declared
/// `int` scalar instead of in element 0 of the int array.
mod two_arrays_with_scalar {
    use super::{AtomicU32, Bytecode, OP_ACC, OP_INC, OP_JUMP_BACK, OP_RETURN, Ordering};
    use majit_metainterp::virt_array::VirtArray;

    static COMPILES: AtomicU32 = AtomicU32::new(0);

    struct TwoArraysScalar {
        iregs: VirtArray<i64>,
        fregs: VirtArray<f64>,
        ret: i64,
    }

    #[majit_macros::jit_interp(
        state = TwoArraysScalar,
        env = Bytecode,
        greens = [pc, program],
        state_fields = {
            iregs: [int; virt],
            fregs: [float; virt],
            ret: int,
        },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn mainloop(program: &Bytecode, limit: i64, threshold: u32) -> i64 {
        let mut driver: majit_metainterp::JitDriver<TwoArraysScalar> =
            majit_metainterp::JitDriver::new(threshold);
        driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
            COMPILES.fetch_add(1, Ordering::Relaxed);
        });
        let mut pc: usize = 0;
        let mut state = TwoArraysScalar {
            iregs: VirtArray::from_slice(&[0i64, limit]),
            fregs: VirtArray::from_slice(&[0.0f64]),
            ret: 0,
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
                OP_ACC => state.fregs[0] += 1.25,
                OP_INC => state.iregs[0] += 1,
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

    #[test]
    fn two_virt_arrays_with_a_scalar_compile_the_hot_loop() {
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
            "declaring one `int` scalar beside two `[.. ; virt]` arrays stopped \
             the hot loop from compiling at all; every trace aborts with \
             VirtualStatesCantMatch on not-virtual slot 0 (expected Ref, got Int)"
        );
    }
}

/// Two virt arrays of the SAME item type, plus a scalar. `dualtape`
/// (`majit/examples/dualtape/src/jit_interp.rs`'s `DualState`) declares this shape
/// (`pa: int, a: [int; virt], pb: int, b: [int; virt]`) and its tests pass — but
/// they assert result equality only, and
/// `MAJIT_LOG=1 cargo test -p dualtape --features cranelift`
/// emits no `[jit]` line at all, so that suite never reaches tracing and says
/// nothing about whether the shape compiles. This machine asks the question it
/// leaves open, and separates "two arrays" from "two arrays of mixed types".
mod two_int_arrays_with_scalar {
    use super::{AtomicU32, Bytecode, OP_ACC, OP_INC, OP_JUMP_BACK, OP_RETURN, Ordering};
    use majit_metainterp::virt_array::VirtArray;

    static COMPILES: AtomicU32 = AtomicU32::new(0);

    struct TwoIntArraysScalar {
        iregs: VirtArray<i64>,
        aregs: VirtArray<i64>,
        ret: i64,
    }

    #[majit_macros::jit_interp(
        state = TwoIntArraysScalar,
        env = Bytecode,
        greens = [pc, program],
        state_fields = {
            iregs: [int; virt],
            aregs: [int; virt],
            ret: int,
        },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn mainloop(program: &Bytecode, limit: i64, threshold: u32) -> i64 {
        let mut driver: majit_metainterp::JitDriver<TwoIntArraysScalar> =
            majit_metainterp::JitDriver::new(threshold);
        driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
            COMPILES.fetch_add(1, Ordering::Relaxed);
        });
        let mut pc: usize = 0;
        let mut state = TwoIntArraysScalar {
            iregs: VirtArray::from_slice(&[0i64, limit]),
            aregs: VirtArray::from_slice(&[0i64]),
            ret: 0,
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
                OP_ACC => state.aregs[0] += 2,
                OP_INC => state.iregs[0] += 1,
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

    #[test]
    fn two_int_virt_arrays_with_a_scalar_compile_the_hot_loop() {
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
            "two same-typed `[int; virt]` arrays beside one `int` scalar did not \
             compile the hot loop"
        );
    }
}

/// PREDICTION (a): ONE virt array plus a single LOOP-DEAD scalar. If the array
/// count were the discriminator this would compile; if the discriminator is
/// "flat slot 0 holds a passthrough scalar", it fails.
mod one_array_dead_scalar {
    use super::{AtomicU32, Bytecode, OP_ACC, OP_INC, OP_JUMP_BACK, OP_RETURN, Ordering};
    use majit_metainterp::virt_array::VirtArray;

    static COMPILES: AtomicU32 = AtomicU32::new(0);

    struct OneArrayDeadScalar {
        iregs: VirtArray<i64>,
        ret: i64,
    }

    #[majit_macros::jit_interp(
        state = OneArrayDeadScalar,
        env = Bytecode,
        greens = [pc, program],
        state_fields = {
            iregs: [int; virt],
            ret: int,
        },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn mainloop(program: &Bytecode, limit: i64, threshold: u32) -> i64 {
        let mut driver: majit_metainterp::JitDriver<OneArrayDeadScalar> =
            majit_metainterp::JitDriver::new(threshold);
        driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
            COMPILES.fetch_add(1, Ordering::Relaxed);
        });
        let mut pc: usize = 0;
        let mut state = OneArrayDeadScalar {
            iregs: VirtArray::from_slice(&[0i64, limit, 0i64]),
            ret: 0,
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
                OP_ACC => state.iregs[2] += 2,
                OP_INC => state.iregs[0] += 1,
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

    #[test]
    fn one_virt_array_with_one_loop_dead_scalar_compiles_the_hot_loop() {
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
            "one virt array beside ONE loop-dead scalar did not compile"
        );
    }
}

/// PREDICTION (b): TWO virt arrays, and the only scalar is reassigned on every
/// iteration, so it does not reach the loop-close Jump as its own inputarg. If
/// the array count were the discriminator this would fail; if the discriminator
/// is the passthrough scalar at slot 0, it compiles.
mod two_arrays_live_scalar {
    use super::{AtomicU32, Bytecode, OP_ACC, OP_INC, OP_JUMP_BACK, OP_RETURN, Ordering};
    use majit_metainterp::virt_array::VirtArray;

    static COMPILES: AtomicU32 = AtomicU32::new(0);

    struct TwoArraysLiveScalar {
        acc: i64,
        iregs: VirtArray<i64>,
        fregs: VirtArray<f64>,
    }

    #[majit_macros::jit_interp(
        state = TwoArraysLiveScalar,
        env = Bytecode,
        greens = [pc, program],
        state_fields = {
            acc: int,
            iregs: [int; virt],
            fregs: [float; virt],
        },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn mainloop(program: &Bytecode, limit: i64, threshold: u32) -> i64 {
        let mut driver: majit_metainterp::JitDriver<TwoArraysLiveScalar> =
            majit_metainterp::JitDriver::new(threshold);
        driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
            COMPILES.fetch_add(1, Ordering::Relaxed);
        });
        let mut pc: usize = 0;
        let mut state = TwoArraysLiveScalar {
            acc: 0,
            iregs: VirtArray::from_slice(&[0i64, limit]),
            fregs: VirtArray::from_slice(&[0.0f64]),
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
                OP_ACC => state.acc += 2,
                OP_INC => state.iregs[0] += 1,
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
                OP_RETURN => break,
                _ => panic!("bad opcode {opcode}"),
            }
        }
        state.iregs[0]
    }

    #[test]
    fn two_virt_arrays_with_a_loop_live_scalar_compile_the_hot_loop() {
        let program = super::count_program();
        let limit = 50i64;
        COMPILES.store(0, Ordering::Relaxed);

        let got = mainloop(&program, limit, 3);

        assert_eq!(
            got,
            super::expected(limit),
            "two-array machine answered wrong"
        );
        assert!(
            COMPILES.load(Ordering::Relaxed) >= 1,
            "two virt arrays beside ONE loop-live scalar did not compile"
        );
    }
}
