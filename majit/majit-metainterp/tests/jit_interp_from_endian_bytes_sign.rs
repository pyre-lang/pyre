//! `<IntTy>::from_le_bytes` must carry the narrow type's wrapping into the trace.
//!
//! The recogniser desugars the call into the shift/or chain that spells the
//! same widening, and the chain is lowered in the machine-word integer bank.
//! For a WORD-width target that is exact. For a narrower one it is not: Rust
//! evaluates `(b as i16) << 8` in `i16`, so a byte that reaches the sign bit
//! wraps, while the word bank keeps every bit and the two tiers answer
//! differently. `i16::from_le_bytes([0xff, 0xff])` is `-1` in Rust and `65535`
//! in a chain that never narrows.
//!
//! The assertion is a two-tier comparison rather than a literal: the untraced
//! tier runs the arm as native Rust, so it is the language's own answer, and
//! the compiled tier must reproduce it. An arm that degraded would also agree,
//! because a degraded arm is executed natively too, so the degraded set is
//! pinned empty for the same run.

use core::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::Mutex;

use majit_metainterp::JitDriver;
use majit_metainterp::virt_array::VirtArray;

/// The mainloop hands back one value, so the decoded ones come out through
/// statics -- which the tests in this binary would race on, because they run
/// on their own threads. `run` holds this for the call AND for reading the
/// statics back, so a run and its readout are one unit.
static RUN_LOCK: Mutex<()> = Mutex::new(());

static LOOPS_COMPILED: AtomicUsize = AtomicUsize::new(0);
static DECODED_I16: AtomicI64 = AtomicI64::new(0);
static DECODED_I32: AtomicI64 = AtomicI64::new(0);
static DECODED_U16: AtomicI64 = AtomicI64::new(0);

pub type Bytecode = [u8];

/// `regs[0] -= 1`, advance.
const OP_DEC: u8 = 1;
/// Decode the two following bytes as a little-endian `i16` into `regs[1]`.
const OP_DECODE_I16: u8 = 2;
/// Decode the four following bytes as a big-endian `i32` into `regs[2]`.
const OP_DECODE_I32: u8 = 3;
/// Decode the two following bytes as a little-endian `u16` into `regs[3]`.
const OP_DECODE_U16: u8 = 4;
/// Back edge: jump to 0 while `regs[0] != 0`, else fall past.
const OP_BACK: u8 = 5;
const OP_END: u8 = 6;

const LOOP_HEADER: usize = 0;

struct EndianState {
    regs: VirtArray<i64>,
}

#[majit_macros::jit_interp(
    state = EndianState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_endian(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<EndianState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = EndianState {
        regs: VirtArray::filled(0i64, 4),
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
                state.regs[0] -= 1;
                pc += 1;
            }
            OP_DECODE_I16 => {
                state.regs[1] = i16::from_le_bytes([program[pc + 1], program[pc + 2]]) as i64;
                pc += 3;
            }
            OP_DECODE_I32 => {
                state.regs[2] = i32::from_be_bytes([
                    program[pc + 1],
                    program[pc + 2],
                    program[pc + 3],
                    program[pc + 4],
                ]) as i64;
                pc += 5;
            }
            OP_DECODE_U16 => {
                state.regs[3] = u16::from_le_bytes([program[pc + 1], program[pc + 2]]) as i64;
                pc += 3;
            }
            OP_BACK => {
                if state.regs[0] != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = LOOP_HEADER;
                    continue;
                } else {
                    pc += 1;
                }
            }
            _ => break,
        }
    }
    DECODED_I16.store(state.regs[1], Ordering::Relaxed);
    DECODED_I32.store(state.regs[2], Ordering::Relaxed);
    DECODED_U16.store(state.regs[3], Ordering::Relaxed);
    LOOPS_COMPILED.store(driver.get_stats().loops_compiled, Ordering::Relaxed);
    state.regs[0]
}

/// Every decoded value has its sign bit set, which is the case the word bank
/// cannot reproduce without a final narrowing.
fn program() -> Vec<u8> {
    vec![
        OP_DEC,
        OP_DECODE_I16,
        0xff,
        0xff,
        OP_DECODE_I32,
        0xff,
        0xff,
        0xff,
        0xfe,
        OP_DECODE_U16,
        0x00,
        0x80,
        OP_BACK,
        OP_END,
        40,
    ]
}

/// What one run at the given threshold decoded, and how many loops it compiled.
struct Run {
    decoded: (i64, i64, i64),
    loops_compiled: usize,
}

fn run(threshold: u32) -> Run {
    let _serialized = RUN_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let answer = dispatch_endian(&program(), threshold);
    assert_eq!(answer, 0, "the loop must have run to completion");
    Run {
        decoded: (
            DECODED_I16.load(Ordering::Relaxed),
            DECODED_I32.load(Ordering::Relaxed),
            DECODED_U16.load(Ordering::Relaxed),
        ),
        loops_compiled: LOOPS_COMPILED.load(Ordering::Relaxed),
    }
}

#[test]
fn the_compiled_tier_decodes_what_rust_decodes() {
    let untraced = run(u32::MAX);
    assert_eq!(
        untraced.loops_compiled, 0,
        "nothing compiles at an unreachable threshold"
    );
    assert_eq!(
        untraced.decoded,
        (
            i16::from_le_bytes([0xff, 0xff]) as i64,
            i32::from_be_bytes([0xff, 0xff, 0xff, 0xfe]) as i64,
            u16::from_le_bytes([0x00, 0x80]) as i64,
        ),
        "the untraced tier is native Rust and defines the answer"
    );

    let compiled = run(3);
    assert!(
        compiled.loops_compiled > 0,
        "nothing compiled, so this says nothing about the lowering"
    );
    assert_eq!(
        compiled.decoded, untraced.decoded,
        "the compiled tier decoded (i16, i32, u16) = {:?} where Rust decodes {:?}",
        compiled.decoded, untraced.decoded,
    );
}

/// Guard the guard: an arm that refused to lower would run natively and agree
/// with the untraced tier for the wrong reason.
#[test]
fn every_decoder_arm_lowers() {
    let _ = run(3);
    let mut arms: Vec<String> = majit_metainterp::degraded_dispatch_arms()
        .into_iter()
        .filter(|entry| entry.interp == "EndianState")
        .map(|entry| entry.arm.to_string())
        .collect();
    arms.sort();
    assert!(
        arms.is_empty(),
        "a degraded arm executes natively, so the tier comparison above would \
         pass without lowering anything; recorded={arms:?}",
    );
}

/// P1: `u64::from_le_bytes(..) as f64` must take the unsigned int-to-float
/// lowering. The recogniser desugars the call to a shift/or chain ending in
/// `((chain) as u64)`, but signedness for the float cast is read off the
/// original `Expr::Call`, which is unsigned only when the path names `u16` /
/// `u32` / `u64`. With the high bit set, the signed lowering answers
/// `-2^62` after `/ 2.0` and the unsigned one `+2^62`. The round trip stays
/// in the int bank because that is the existing state's register file.
mod uint_as_float {
    use super::*;

    static RUN_LOCK: Mutex<()> = Mutex::new(());
    static LOOPS_COMPILED: AtomicUsize = AtomicUsize::new(0);
    static DECODED: AtomicI64 = AtomicI64::new(0);

    const OP_DEC: u8 = 1;
    /// `(u64::from_le_bytes(next 8 bytes) as f64 / 2.0) as i64` into `regs[1]`.
    const OP_DECODE_U64_AS_F64: u8 = 2;
    const OP_BACK: u8 = 3;
    const OP_END: u8 = 4;
    const LOOP_HEADER: usize = 0;

    struct UintAsFloatState {
        regs: VirtArray<i64>,
    }

    #[majit_macros::jit_interp(
        state = UintAsFloatState,
        env = Bytecode,
        greens = [pc, program],
        state_fields = {
            regs: [int; virt],
        },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_uint_as_float(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<UintAsFloatState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = UintAsFloatState {
            regs: VirtArray::filled(0i64, 2),
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
                    state.regs[0] -= 1;
                    pc += 1;
                }
                OP_DECODE_U64_AS_F64 => {
                    state.regs[1] = (u64::from_le_bytes([
                        program[pc + 1],
                        program[pc + 2],
                        program[pc + 3],
                        program[pc + 4],
                        program[pc + 5],
                        program[pc + 6],
                        program[pc + 7],
                        program[pc + 8],
                    ]) as f64
                        / 2.0) as i64;
                    pc += 9;
                }
                OP_BACK => {
                    if state.regs[0] != 0 {
                        can_enter_jit!(driver, 0usize, &mut state, program, || {});
                        pc = LOOP_HEADER;
                        continue;
                    } else {
                        pc += 1;
                    }
                }
                _ => break,
            }
        }
        DECODED.store(state.regs[1], Ordering::Relaxed);
        LOOPS_COMPILED.store(driver.get_stats().loops_compiled, Ordering::Relaxed);
        state.regs[0]
    }

    /// Little-endian encoding of `2^63`: the case where signed and unsigned
    /// int-to-float disagree.
    fn program() -> Vec<u8> {
        vec![
            OP_DEC,
            OP_DECODE_U64_AS_F64,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x80,
            OP_BACK,
            OP_END,
            40,
        ]
    }

    struct Run {
        decoded: i64,
        loops_compiled: usize,
    }

    fn run(threshold: u32) -> Run {
        let _serialized = RUN_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let answer = dispatch_uint_as_float(&program(), threshold);
        assert_eq!(answer, 0, "the loop must have run to completion");
        Run {
            decoded: DECODED.load(Ordering::Relaxed),
            loops_compiled: LOOPS_COMPILED.load(Ordering::Relaxed),
        }
    }

    #[test]
    fn u64_from_le_bytes_as_f64_uses_the_unsigned_lowering() {
        let untraced = run(u32::MAX);
        assert_eq!(
            untraced.loops_compiled, 0,
            "nothing compiles at an unreachable threshold"
        );
        assert_eq!(
            untraced.decoded,
            (u64::from_le_bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80]) as f64 / 2.0)
                as i64,
            "the untraced tier is native Rust and defines the answer"
        );

        let compiled = run(3);
        assert!(
            compiled.loops_compiled > 0,
            "nothing compiled, so this says nothing about the lowering"
        );
        assert_eq!(
            compiled.decoded, untraced.decoded,
            "the compiled tier decoded {} where Rust decodes {}",
            compiled.decoded, untraced.decoded,
        );
    }

    #[test]
    fn the_uint_as_float_decoder_arm_lowers() {
        let _ = run(3);
        let mut arms: Vec<String> = majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|entry| entry.interp == "UintAsFloatState")
            .map(|entry| entry.arm.to_string())
            .collect();
        arms.sort();
        assert!(
            arms.is_empty(),
            "a degraded arm executes natively, so the tier comparison above would \
             pass without lowering anything; recorded={arms:?}",
        );
    }
}

/// P2: a namespaced user type whose last segment is a primitive name must
/// not be rewritten as that primitive. `wire::u16::from_le_bytes` is a
/// user decoder; treating it as `u16::from_le_bytes` runs the primitive's
/// semantics on the compiled tier. After the recogniser declines, the arm
/// may degrade and run natively — that is the fail-closed outcome, so the
/// degraded set is not pinned empty.
mod namespaced_decoder {
    use super::*;

    static RUN_LOCK: Mutex<()> = Mutex::new(());
    static LOOPS_COMPILED: AtomicUsize = AtomicUsize::new(0);
    static DECODED: AtomicI64 = AtomicI64::new(0);

    /// A user type that shares the primitive's last path segment but whose
    /// decoder is deliberately not the primitive reconstruction.
    pub mod wire {
        #[allow(non_camel_case_types)]
        pub struct u16;

        impl u16 {
            pub fn from_le_bytes(bytes: [u8; 2]) -> i64 {
                i64::from(bytes[0]) + 1000 * i64::from(bytes[1])
            }
        }
    }

    const OP_DEC: u8 = 1;
    const OP_DECODE_WIRE_U16: u8 = 2;
    const OP_BACK: u8 = 3;
    const OP_END: u8 = 4;
    const LOOP_HEADER: usize = 0;

    struct NamespacedEndianState {
        regs: VirtArray<i64>,
    }

    #[majit_macros::jit_interp(
        state = NamespacedEndianState,
        env = Bytecode,
        greens = [pc, program],
        state_fields = {
            regs: [int; virt],
        },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_namespaced(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<NamespacedEndianState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = NamespacedEndianState {
            regs: VirtArray::filled(0i64, 2),
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
                    state.regs[0] -= 1;
                    pc += 1;
                }
                OP_DECODE_WIRE_U16 => {
                    state.regs[1] = wire::u16::from_le_bytes([program[pc + 1], program[pc + 2]]);
                    pc += 3;
                }
                OP_BACK => {
                    if state.regs[0] != 0 {
                        can_enter_jit!(driver, 0usize, &mut state, program, || {});
                        pc = LOOP_HEADER;
                        continue;
                    } else {
                        pc += 1;
                    }
                }
                _ => break,
            }
        }
        DECODED.store(state.regs[1], Ordering::Relaxed);
        LOOPS_COMPILED.store(driver.get_stats().loops_compiled, Ordering::Relaxed);
        state.regs[0]
    }

    fn program() -> Vec<u8> {
        vec![OP_DEC, OP_DECODE_WIRE_U16, 0x01, 0x02, OP_BACK, OP_END, 40]
    }

    struct Run {
        decoded: i64,
        loops_compiled: usize,
    }

    fn run(threshold: u32) -> Run {
        let _serialized = RUN_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let answer = dispatch_namespaced(&program(), threshold);
        assert_eq!(answer, 0, "the loop must have run to completion");
        Run {
            decoded: DECODED.load(Ordering::Relaxed),
            loops_compiled: LOOPS_COMPILED.load(Ordering::Relaxed),
        }
    }

    #[test]
    fn a_namespaced_u16_decoder_is_not_the_primitive() {
        let untraced = run(u32::MAX);
        assert_eq!(
            untraced.loops_compiled, 0,
            "nothing compiles at an unreachable threshold"
        );
        assert_eq!(
            untraced.decoded,
            wire::u16::from_le_bytes([0x01, 0x02]),
            "the untraced tier is native Rust and defines the answer"
        );
        // The primitive reconstruction of the same bytes is a different value;
        // if it were not, a misrecognition would be invisible.
        assert_ne!(
            untraced.decoded,
            u16::from_le_bytes([0x01, 0x02]) as i64,
            "the user decoder must disagree with the primitive or this test \
             cannot see a misrecognition"
        );

        let compiled = run(3);
        assert_eq!(
            compiled.decoded, untraced.decoded,
            "the compiled tier decoded {} where the user decoder answers {}",
            compiled.decoded, untraced.decoded,
        );
    }
}
