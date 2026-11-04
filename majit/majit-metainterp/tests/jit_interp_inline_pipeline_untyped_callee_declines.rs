//! An `inline_pipeline_int` call whose host-resolved callee does not end in a
//! typed return degrades its arm instead of aborting the process.
//!
//! The callee of an `inline_pipeline_*` call is named by the call's last path
//! segment and handed over by the host at install time
//! (`jitcode_lower/lower_value.rs`), so whether its body ends in a typed return
//! is not knowable when the macro runs. When it does not, the parent's
//! `BC_INLINE_CALL` has no return kind to write. That is a rejection, and the
//! lowering has a rejection path: the arm builder answers `None`, the arm
//! degrades to an abort stub with a recorded reason, and every other arm still
//! compiles. `make_jitcodes()` / `pyjitpl.py finish_setup()` install only
//! completed jitcodes; a callee the host resolved badly is not supposed to take
//! the process down.
//!
//! The body below is a bare `void_return`, which is what a host resolving the
//! name to a value-less function hands over. `trailing_return_info()` rejects
//! it, and this fixture pins where that rejection lands.

use std::sync::{Arc, OnceLock};

use majit_metainterp::{Assembler, EmbeddedJitCodeTable, JitDriver};

pub type Bytecode = [u8];

const OP_NOP: u8 = 0;
const OP_UNTYPED: u8 = 1;

/// A committed body ending in `void_return` — no typed return for the caller
/// to read a kind off.
fn build_time_table() -> &'static EmbeddedJitCodeTable {
    static TABLE: OnceLock<&'static EmbeddedJitCodeTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let core = majit_jitcode::jitcode::JitCode::new("pipeline_untyped");
        core.set_body(majit_jitcode::jitcode::JitCodeBody {
            code: vec![majit_metainterp::jitcode::insns::BC_VOID_RETURN],
            ..Default::default()
        });
        core.set_index(0);
        let table = EmbeddedJitCodeTable::materialize(&[Arc::new(core)], Vec::new());
        table.install_as_global_pool();
        table
    })
}

#[allow(non_snake_case)]
fn __majit_pipeline_jitcode(name: &str) -> Arc<majit_metainterp::JitCode> {
    build_time_table()
        .by_name(name)
        .unwrap_or_else(|| panic!("no build-time jitcode named {name}"))
        .clone()
}

fn __majit_pipeline_liveness_prebuild(_assembler: &mut majit_metainterp::Assembler) {}

/// The concrete path. It returns a value; only the jitcode the host hands over
/// for it does not.
fn pipeline_untyped(value: i64) -> i64 {
    value + 1
}

struct CountingState {
    a: i64,
}

#[majit_macros::jit_interp(
    state = CountingState,
    env = Bytecode,
    state_fields = { a: int },
    greens = [],
    calls = {
        pipeline_untyped => inline_pipeline_int,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn counting_interp(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<CountingState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = CountingState { a: 1 };
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
            OP_UNTYPED => state.a = pipeline_untyped(state.a),
            _ => break,
        }
    }
    state.a
}

/// Every jitcode reachable from `root` through its `j` slots, `root` included.
fn reachable(root: &Arc<majit_metainterp::JitCode>) -> Vec<Arc<majit_metainterp::JitCode>> {
    let mut found = vec![Arc::clone(root)];
    let mut cursor = 0;
    while cursor < found.len() {
        let current = Arc::clone(&found[cursor]);
        cursor += 1;
        for descr in &current.exec.descrs {
            if let Some(sub) = descr.as_jitcode()
                && !found.iter().any(|j| Arc::ptr_eq(j, sub))
            {
                found.push(Arc::clone(sub));
            }
        }
    }
    found
}

/// The subject: the untyped callee is refused, the refusal is recorded against
/// the arm that made the call, and the install completes.
#[test]
fn an_untyped_pipeline_callee_degrades_its_arm_and_installs() {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
    __prebuild_jitcode_liveness_counting_interp(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let dispatch = Arc::new(
        __dispatch_jitcode_counting_interp(&mut asm, 0i64)
            .expect("one rejected arm must not reject the whole dispatch body"),
    );

    let degraded = majit_metainterp::degraded_dispatch_arms();
    assert!(
        degraded
            .iter()
            .any(|arm| arm.interp == "CountingState" && arm.arm == "OP_UNTYPED"),
        "the arm that inline-called the untyped callee must be recorded as \
         degraded, so the rejection is reportable rather than silent; recorded \
         {degraded:?}",
    );

    let reached = reachable(&dispatch);
    let names: Vec<&str> = reached.iter().map(|jc| jc.name()).collect();
    assert!(
        !names.contains(&"pipeline_untyped"),
        "the callee has no return kind for the parent's INLINE_CALL, so it \
         must not have been spliced; reached {names:?}",
    );
}

/// The control: the arm degrading does not stop the interpreter. The concrete
/// path runs, which is the same lifecycle any other rejected arm gets.
#[test]
fn a_degraded_arm_leaves_the_concrete_path_running() {
    let program = [OP_UNTYPED, OP_UNTYPED, OP_NOP];
    assert_eq!(counting_interp(&program, 0), 3);
}
