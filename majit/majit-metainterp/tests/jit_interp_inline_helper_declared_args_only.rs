//! A `cells: [int]` machine calling `#[jit_inline] fn helper(x: i64)` via
//! `inline_int`.
//!
//! `Transformer.rewrite_call` / `handle_regular_call` emit `inline_call_*`
//! from `op.args[1:]` only (the callee graph's declared arguments).
//! `MIFrame.setup_call` copies those boxes. The I-list is just `x`, matching
//! the helper's `arg_classes`, so `check_inline_call_arg_classes` accepts the
//! assembly and the compiled loop matches the interpreter.

use core::sync::atomic::{AtomicU32, Ordering};
use majit_macros::jit_inline;
use majit_metainterp::jitcode::insns::BC_INLINE_CALL_IR_I;
use majit_metainterp::{Assembler, JitCode, JitDriver};

pub type Bytecode = [u8];

const OP_ADD: u8 = 1;
const OP_LOOP: u8 = 2;
const N: usize = 40;

#[jit_inline]
fn helper(x: i64) -> i64 {
    x + 1i64
}

struct CellsHelperState {
    cells: Vec<i64>,
    acc: i64,
    ticks: i64,
}

static COMPILES: AtomicU32 = AtomicU32::new(0);

#[majit_macros::jit_interp(
    state = CellsHelperState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { cells: [int], acc: int, ticks: int },
    calls = { helper => inline_int },
)]
#[allow(unused_assignments, unused_variables)]
fn cells_helper_sum(program: &Bytecode, threshold: u32, cells: Vec<i64>, ticks: i64) -> i64 {
    let mut driver: JitDriver<CellsHelperState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = CellsHelperState {
        cells,
        acc: 0,
        ticks,
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
                state.acc = helper(state.acc);
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
    state.acc
}

fn build_dispatch_jitcode() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_cells_helper_sum(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_cells_helper_sum(&mut asm, 0i64).expect("dispatch lower must succeed")
}

fn walk_jitcodes<'a>(jc: &'a JitCode, out: &mut Vec<&'a JitCode>) {
    out.push(jc);
    for descr in &jc.exec.descrs {
        if let Some(sub) = descr.as_jitcode() {
            walk_jitcodes(sub, out);
        }
    }
}

fn helper_i_list_lens(jc: &JitCode, helper_name: &str) -> Vec<u8> {
    let mut lens = Vec::new();
    collect_helper_i_list_lens(jc, helper_name, &mut lens);
    lens
}

fn collect_helper_i_list_lens(jc: &JitCode, helper_name: &str, lens: &mut Vec<u8>) {
    let code = &jc.code;
    let mut i = 0;
    while i + 3 < code.len() {
        if code[i] == BC_INLINE_CALL_IR_I {
            let descr = u16::from_le_bytes([code[i + 1], code[i + 2]]) as usize;
            let n = code[i + 3];
            if let Some(sub) = jc.exec.descrs.get(descr).and_then(|d| d.as_jitcode())
                && sub.name() == helper_name
            {
                lens.push(n);
            }
        }
        i += 1;
    }
    for descr in &jc.exec.descrs {
        if let Some(sub) = descr.as_jitcode() {
            collect_helper_i_list_lens(sub, helper_name, lens);
        }
    }
}

#[test]
fn helper_inline_call_i_list_matches_arg_classes_and_assembles() {
    let dispatch_jc = build_dispatch_jitcode();
    majit_metainterp::assert_no_degraded_dispatch_arms("CellsHelperState");

    let mut reached = Vec::new();
    walk_jitcodes(&dispatch_jc, &mut reached);
    let helper = reached
        .iter()
        .copied()
        .find(|jc| jc.name() == "helper")
        .expect("the inline helper jitcode must be registered");
    helper.try_body().expect("helper body");
    let arg_classes = &helper.calldescr().arg_classes;
    assert_eq!(arg_classes, "i", "helper(x: i64) declares one int argument");

    let lens = helper_i_list_lens(&dispatch_jc, "helper");
    assert!(
        !lens.is_empty(),
        "dispatch must emit inline_call_ir_i into helper; dispatch={:?} names={:?}",
        dispatch_jc.code,
        reached.iter().map(|jc| jc.name()).collect::<Vec<_>>()
    );
    assert!(
        lens.iter().all(|&n| n == 1),
        "I-list must hold only x, matching arg_classes {arg_classes:?}; lens={lens:?}"
    );
}

#[test]
fn compiled_helper_on_a_cells_machine_matches_the_interpreter() {
    let cells = vec![7i64; 4];
    let program = vec![OP_ADD, OP_LOOP];
    let expected = N as i64;

    let interp = cells_helper_sum(&program, u32::MAX, cells.clone(), N as i64);
    assert_eq!(
        interp, expected,
        "each trip does acc = helper(acc) = acc + 1"
    );

    COMPILES.store(0, Ordering::Relaxed);
    let compiled = cells_helper_sum(&program, 2, cells, N as i64);
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "the loop must compile"
    );
    assert_eq!(
        compiled, interp,
        "compiled helper(x) on a cells machine must match the interpreter"
    );
}
