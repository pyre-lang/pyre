//! `opaque(T)` carries a field the JIT never enumerates.
//!
//! A state field declared `opaque(T)` keeps its declared type on the state
//! struct and is not a red, a fail arg, or a sym slot
//! (`jit_interp/mod.rs` `StateFieldKind::Opaque`). It exists for carriers
//! whose layout is not flat ints — pools, handles, anything the JIT has no
//! representation for — and the interpreter reaches them directly.
//!
//! The declaration has three consequences, and nothing else covered any of
//! them: the field contributes no red slot, an arm that touches it degrades
//! rather than lowering something wrong, and the state shape is excluded from
//! the generic recursive-portal fresh entry (`codegen_state.rs`) because a
//! fresh frame cannot synthesize an arbitrary `T`.

use majit_metainterp::{Assembler, JitCodeSym as _, JitDriver};

pub type Bytecode = [u8];

const OP_NOP: u8 = 0;
const OP_BUMP: u8 = 1;
const OP_TOUCH: u8 = 2;
const OP_READ_LEN: u8 = 3;

/// A carrier whose layout is not flat ints, which is the whole reason
/// `opaque(T)` exists.
struct Storage {
    names: Vec<String>,
}

struct OpaqueState {
    acc: i64,
    storage: Storage,
}

#[majit_macros::jit_interp(
    state = OpaqueState,
    env = Bytecode,
    state_fields = { acc: int, storage: opaque(Storage) },
    greens = [],
)]
#[allow(unused_assignments, unused_variables)]
fn opaque_interp(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<OpaqueState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = OpaqueState {
        acc: 0,
        storage: Storage { names: Vec::new() },
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_NOP => {}
            OP_BUMP => state.acc += 1,
            OP_TOUCH => state.storage.names.push("x".to_string()),
            OP_READ_LEN => state.acc = state.storage.names.len() as i64,
            _ => break,
        }
    }
    state.acc
}

/// The control: the same shape with the carrier removed. Every assertion below
/// that reads "because of the opaque field" is paired against this.
struct PlainState {
    acc: i64,
}

#[majit_macros::jit_interp(
    state = PlainState,
    env = Bytecode,
    state_fields = { acc: int },
    greens = [],
)]
#[allow(unused_assignments, unused_variables)]
fn plain_interp(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<PlainState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = PlainState { acc: 0 };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_NOP => {}
            OP_BUMP => state.acc += 1,
            _ => break,
        }
    }
    state.acc
}

/// The carrier is not enumerated: the reds are the int scalar alone, the same
/// list the carrier-free control produces.
#[test]
fn an_opaque_carrier_contributes_no_red_slot() {
    use majit_metainterp::JitState as _;

    let program = [OP_BUMP];
    let mut carried = OpaqueState {
        acc: 7,
        storage: Storage {
            names: vec!["a".to_string()],
        },
    };
    let carried_meta = carried.build_meta(0, &program[..]);
    assert_eq!(carried.extract_live(&carried_meta), vec![7]);
    assert_eq!(
        carried.live_value_types(&carried_meta),
        vec![majit_ir::Type::Int],
    );

    let mut plain = PlainState { acc: 7 };
    let plain_meta = plain.build_meta(0, &program[..]);
    assert_eq!(
        carried.extract_live(&carried_meta),
        plain.extract_live(&plain_meta),
        "declaring a carrier must not move or add a red",
    );
}

/// An arm that reaches into the carrier degrades, with the statement it could
/// not express named. The other arms still compile, and the dispatch body
/// still installs.
#[test]
fn an_arm_touching_the_carrier_degrades_and_the_rest_installs() {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
    __prebuild_jitcode_liveness_opaque_interp(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    assert!(
        __dispatch_jitcode_opaque_interp(&mut asm, 0i64).is_some(),
        "one unlowerable arm must not reject the dispatch body",
    );

    let degraded = majit_metainterp::degraded_dispatch_arms();
    let touch = degraded
        .iter()
        .find(|arm| arm.interp == "OpaqueState" && arm.arm == "OP_TOUCH")
        .unwrap_or_else(|| panic!("OP_TOUCH must be recorded as degraded; got {degraded:?}"));
    assert!(
        touch.reason.contains("cannot express"),
        "the reason must name the statement, not a generic refusal: {}",
        touch.reason,
    );
    assert!(
        !degraded
            .iter()
            .any(|arm| arm.interp == "OpaqueState" && arm.arm == "OP_BUMP"),
        "the int arm shares the state and must still lower; got {degraded:?}",
    );
}

/// The carrier survives the run it is declared for: the interpreter writes it
/// through the degraded arm and reads it back, while the compiled arms keep
/// the int scalar correct.
#[test]
fn the_carrier_survives_a_run_that_also_compiles() {
    let program = [OP_BUMP, OP_BUMP, OP_TOUCH, OP_READ_LEN, OP_BUMP];
    assert_eq!(opaque_interp(&program, 0), 2);
}

/// `codegen_state.rs` excludes a carrier-bearing shape from the generic
/// recursive-portal fresh entry: a fresh frame cannot synthesize an arbitrary
/// `T`, so the shape falls back to the `JitCodeSym` `None` default and the
/// recursive dispatcher aborts to the interpreter.
#[test]
fn a_carrier_bearing_shape_has_no_generic_fresh_entry() {
    use majit_metainterp::JitState as _;

    let program = [OP_BUMP];
    let carried = OpaqueState {
        acc: 1,
        storage: Storage { names: Vec::new() },
    };
    let carried_sym = OpaqueState::create_sym(&carried.build_meta(0, &program[..]), 0);
    assert!(
        carried_sym.recursive_fresh_entry_reds().is_none(),
        "a shape carrying an opaque field has no generic fresh frame",
    );

    let plain = PlainState { acc: 1 };
    let plain_sym = PlainState::create_sym(&plain.build_meta(0, &program[..]), 0);
    assert!(
        plain_sym.recursive_fresh_entry_reds().is_some(),
        "control: the same shape without the carrier does have one, so the \
         assertion above is about the carrier and not about this fixture",
    );
}
