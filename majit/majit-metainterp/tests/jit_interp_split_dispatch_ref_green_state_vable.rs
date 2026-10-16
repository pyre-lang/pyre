//! A ref-valued portal green precedes the state-as-vable identity.
//!
//! `warmspot.py` `index_of_virtualizable` / `jd.num_green_args +
//! jd.index_of_virtualizable`: greens occupy the ref bank first, so the
//! extra-pushed `&state` identity is at `1 + portal_ref_greens +
//! carried_ref_greens` (r2 here).

use core::sync::atomic::{AtomicU32, Ordering};
use majit_metainterp::jitcode::insns::{
    BC_GETFIELD_VABLE_I, BC_INLINE_CALL, BC_INLINE_CALL_IR_I, BC_INLINE_CALL_IR_V,
    BC_SETFIELD_VABLE_I,
};
use majit_metainterp::virt_array::VirtArray;
use majit_metainterp::{Assembler, JitCode, JitDriver};

pub type Bytecode = [u8];

const OP_ADD: u8 = 1;
const OP_LOOP: u8 = 2;
const ADD_BY: i64 = 7;
const N: i64 = 40;

struct ScalarAcc {
    acc: i64,
    ticks: i64,
}

struct VirtAcc {
    acc: i64,
    ticks: i64,
    #[allow(dead_code)]
    tape: VirtArray<i64>,
}

static COMPILES_SPLIT: AtomicU32 = AtomicU32::new(0);
static COMPILES_VIRT: AtomicU32 = AtomicU32::new(0);

#[majit_macros::jit_interp(
    state = ScalarAcc,
    env = Bytecode,
    split_dispatch = true,
    greens = [pc, program, extra: ref],
    state_fields = { acc: int, ticks: int },
)]
#[allow(unused_assignments, unused_variables)]
fn split_ref_green(program: &Bytecode, extra: usize, threshold: u32, ticks: i64) -> i64 {
    let mut driver: JitDriver<ScalarAcc> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES_SPLIT.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = ScalarAcc { acc: 0, ticks };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);
        let opcode = program[pc];
        match opcode {
            OP_ADD => {
                state.acc = state.acc + ADD_BY;
                pc += 1;
            }
            OP_LOOP => {
                state.ticks = state.ticks - 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
                pc += 1;
            }
            _ => break,
        }
    }
    let _ = extra;
    state.acc
}

#[majit_macros::jit_interp(
    state = VirtAcc,
    env = Bytecode,
    greens = [pc, program, extra: ref],
    state_fields = { acc: int, ticks: int, tape: [int; virt] },
)]
#[allow(unused_assignments, unused_variables)]
fn virt_ref_green(program: &Bytecode, extra: usize, threshold: u32, ticks: i64) -> i64 {
    let mut driver: JitDriver<VirtAcc> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES_VIRT.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = VirtAcc {
        acc: 0,
        ticks,
        tape: VirtArray::from_slice(&[0i64]),
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
        match opcode {
            OP_ADD => {
                state.acc = state.acc + ADD_BY;
                pc += 1;
            }
            OP_LOOP => {
                state.ticks = state.ticks - 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
                pc += 1;
            }
            _ => break,
        }
    }
    let _ = extra;
    state.acc
}

fn program() -> Vec<u8> {
    vec![OP_ADD, OP_LOOP]
}

fn build_split_dispatch_jitcode() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_split_ref_green(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_split_ref_green(&mut asm, 0i64).expect("dispatch lower must succeed")
}

fn dispatch_subs(jc: &JitCode) -> impl Iterator<Item = &JitCode> {
    jc.exec
        .descrs
        .iter()
        .filter_map(|descr| descr.as_jitcode().map(|sub| sub.as_ref()))
}

fn has_inline_call(code: &[u8]) -> bool {
    code.contains(&BC_INLINE_CALL)
        || code.contains(&BC_INLINE_CALL_IR_I)
        || code.contains(&BC_INLINE_CALL_IR_V)
}

fn has_vable_int_ops(code: &[u8]) -> bool {
    code.contains(&BC_GETFIELD_VABLE_I) && code.contains(&BC_SETFIELD_VABLE_I)
}

#[test]
fn split_dispatch_identity_sits_after_the_ref_green() {
    let info = <ScalarAcc as majit_metainterp::JitState>::__build_virtualizable_info()
        .expect("split_dispatch scalars take the state-as-vable path");
    assert_eq!(
        info.identity_ref_bank_index,
        Some(2),
        "program at r0, extra green at r1; extra-pushed identity at r2"
    );
}

#[test]
fn split_dispatch_with_ref_green_emits_vable_ops() {
    let dispatch_jc = build_split_dispatch_jitcode();
    majit_metainterp::assert_no_degraded_dispatch_arms("ScalarAcc");
    assert!(
        has_inline_call(&dispatch_jc.code),
        "split_dispatch must emit an arm inline_call; dispatch={:?}",
        dispatch_jc.code
    );
    let arm_vable = dispatch_subs(&dispatch_jc).any(|sub| has_vable_int_ops(&sub.code));
    assert!(
        has_vable_int_ops(&dispatch_jc.code) || arm_vable,
        "arms must emit getfield_vable_i / setfield_vable_i; dispatch={:?} subs={:?}",
        dispatch_jc.code,
        dispatch_subs(&dispatch_jc)
            .map(|c| &c.code)
            .collect::<Vec<_>>()
    );
}

#[test]
fn split_dispatch_ref_green_compiled_loop_matches_interpreter() {
    let extra = 0x1000usize;
    let cold = split_ref_green(&program(), extra, u32::MAX, N);
    assert_eq!(cold, ADD_BY * N);

    COMPILES_SPLIT.store(0, Ordering::Relaxed);
    let warm = split_ref_green(&program(), extra, 3, N);
    assert!(
        COMPILES_SPLIT.load(Ordering::Relaxed) >= 1,
        "loop must compile (compiles={})",
        COMPILES_SPLIT.load(Ordering::Relaxed)
    );
    assert_eq!(warm, cold, "compiled acc must match interpreter");
}

#[test]
fn virt_array_identity_sits_after_the_ref_green() {
    let info = <VirtAcc as majit_metainterp::JitState>::__build_virtualizable_info()
        .expect("[.. ; virt] takes the state-as-vable path");
    assert_eq!(
        info.identity_ref_bank_index,
        Some(2),
        "program at r0, extra green at r1; extra-pushed identity at r2"
    );
}

#[test]
fn virt_array_ref_green_compiled_loop_matches_interpreter() {
    let extra = 0x1000usize;
    let cold = virt_ref_green(&program(), extra, u32::MAX, N);
    assert_eq!(cold, ADD_BY * N);

    COMPILES_VIRT.store(0, Ordering::Relaxed);
    let warm = virt_ref_green(&program(), extra, 3, N);
    assert!(
        COMPILES_VIRT.load(Ordering::Relaxed) >= 1,
        "loop must compile (compiles={})",
        COMPILES_VIRT.load(Ordering::Relaxed)
    );
    assert_eq!(warm, cold, "compiled acc must match interpreter");
}
