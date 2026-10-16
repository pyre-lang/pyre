//! An `inline_pipeline_*` call resolves a callee a build-time table already
//! numbered.
//!
//! The three `inline_pipeline_*` call policies name their callee by the call's
//! last path segment and ask the host for it through `__majit_pipeline_jitcode`
//! (`jitcode_lower/lower_value.rs`). Unlike the `inline_*` policies, the
//! returned jitcode is not built here: it comes out of a table the host
//! serialized after running the codewriter, so it already carries the
//! `jitcode.index` its own operands are written against (`codewriter.py`).
//!
//! That index is the whole point of this fixture. Installing the dispatch
//! JitCode flattens everything it can inline-call into the flat registry
//! `resume.py:1338-1340` indexes, and the numbering there used to start at 0
//! and stamp every jitcode it discovered. `JitCode::set_index` is set-once and
//! asserts, so the first `inline_pipeline_*` call over a build-time callee
//! aborted the install with `index already set to N, cannot reassign to 1`
//! rather than tracing. Nothing else covers this: the other inline policies
//! build their callee on the spot, so it reaches the walk unnumbered.
//!
//! `greens = []` is deliberate and the macro's empty-green-set report is the
//! expected line for it: what is under test is what the install path does with
//! the callee's index, and no trace is recorded here.

use std::sync::{Arc, OnceLock};

use majit_metainterp::{Assembler, EmbeddedJitCodeTable, JitDriver};

pub type Bytecode = [u8];

const OP_NOP: u8 = 0;
const OP_DOUBLE: u8 = 1;
const OP_VOID: u8 = 2;

/// The callee the fixture inline-calls. Its body is what the host would have
/// deserialized, so it is spelled here the way the blob spells it: a committed
/// body ending in a typed return, and an index assigned at build time.
fn build_time_table() -> &'static EmbeddedJitCodeTable {
    static TABLE: OnceLock<&'static EmbeddedJitCodeTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        // Two entries so the callee does NOT sit at the slot a fresh numbering
        // would hand it. With the callee at index 0 the collision cannot show:
        // the old walk numbered the dispatch jitcode 0 and would have written
        // 1, which happens to be free.
        let filler = canonical_jitcode("pipeline_filler", 0);
        let callee = canonical_jitcode("pipeline_double", 1);
        let void_callee = canonical_void_jitcode("pipeline_void", 2);
        let table = EmbeddedJitCodeTable::materialize(&[filler, callee, void_callee], Vec::new());
        table.install_as_global_pool();
        table
    })
}

fn canonical_void_jitcode(name: &str, index: usize) -> Arc<majit_jitcode::jitcode::JitCode> {
    let core = majit_jitcode::jitcode::JitCode::new(name);
    core.set_body(majit_jitcode::jitcode::JitCodeBody {
        code: vec![majit_metainterp::jitcode::insns::BC_VOID_RETURN],
        ..Default::default()
    });
    core.set_index(index);
    Arc::new(core)
}

/// `int_return/i` reading int register 0 — the minimum
/// `trailing_return_info()` accepts, which the `inline_pipeline_*` lowering
/// requires of its callee.
fn canonical_jitcode(name: &str, index: usize) -> Arc<majit_jitcode::jitcode::JitCode> {
    let core = majit_jitcode::jitcode::JitCode::new(name);
    core.set_body(majit_jitcode::jitcode::JitCodeBody {
        code: vec![majit_metainterp::jitcode::insns::BC_INT_RETURN, 0],
        c_num_regs_i: 1,
        ..Default::default()
    });
    core.set_index(index);
    Arc::new(core)
}

/// The hook the `inline_pipeline_*` lowering emits a call to. The host owns
/// both the table and this resolution; nothing below it knows the names.
#[allow(non_snake_case)]
fn __majit_pipeline_jitcode(name: &str) -> Arc<majit_metainterp::JitCode> {
    build_time_table()
        .by_name(name)
        .unwrap_or_else(|| panic!("no build-time jitcode named {name}"))
        .clone()
}

fn __majit_pipeline_liveness_prebuild(_assembler: &mut majit_metainterp::Assembler) {}

/// The concrete (non-tracing) path. The jitcode above stands in for its traced
/// form, exactly as a pipeline-built callee stands in for the real function.
fn pipeline_double(value: i64) -> i64 {
    value * 2
}

fn pipeline_void() {}

struct DoublingState {
    a: i64,
}

#[majit_macros::jit_interp(
    state = DoublingState,
    env = Bytecode,
    state_fields = { a: int },
    greens = [],
    calls = {
        pipeline_double => inline_pipeline_int,
        pipeline_void => inline_pipeline_void,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn doubling_interp(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<DoublingState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = DoublingState { a: 1 };
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
            OP_DOUBLE => state.a = pipeline_double(state.a),
            OP_VOID => pipeline_void(),
            _ => break,
        }
    }
    state.a
}

#[test]
fn a_traced_void_pipeline_callee_returns_to_its_caller() {
    let program = [OP_VOID, OP_DOUBLE, OP_NOP];
    assert_eq!(doubling_interp(&program, 0), 2);
}

/// The subject: installing the driver over a dispatch JitCode that
/// inline-calls a build-time callee completes, and every jitcode in the flat
/// registry sits at the index it names.
///
/// A threshold above the program's length keeps the tracer out of it — the
/// registry is built at install time, which is where the numbering happened.
#[test]
fn a_build_time_callee_survives_the_dispatch_registry_numbering() {
    let program = [OP_DOUBLE, OP_DOUBLE, OP_NOP];
    assert_eq!(
        doubling_interp(&program, 1_000_000),
        4,
        "the concrete path must still double twice",
    );

    let table = build_time_table();
    assert_eq!(
        table.by_name("pipeline_double").map(|jc| jc.index()),
        Some(1),
        "the callee keeps the index its own operands are written against; \
         the registry numbering must not have reassigned it",
    );
}

/// Every jitcode reachable from `root` through `j` slots, `root` included, in
/// the order the registry walk finds them.
fn reachable(root: &Arc<majit_metainterp::JitCode>) -> Vec<Arc<majit_metainterp::JitCode>> {
    let mut found = vec![Arc::clone(root)];
    let mut cursor = 0;
    while cursor < found.len() {
        let current = Arc::clone(&found[cursor]);
        cursor += 1;
        for descr in &current.exec.descrs {
            if let Some(sub) = descr.as_jitcode() {
                if !found.iter().any(|j| Arc::ptr_eq(j, sub)) {
                    found.push(Arc::clone(sub));
                }
            }
        }
    }
    found
}

/// The control: the build-time callee really is spliced, and really does reach
/// the registry walk.
///
/// Without this the test above passes for a fixture whose call was lowered as
/// a residual — the callee would never reach the walk, and the numbering it is
/// meant to exercise would never run.
///
/// The splice is one level down: the dispatch JitCode inline-calls the
/// per-opcode arm, and the arm inline-calls the callee. Asserting on the
/// dispatch's own descrs would look only at the arms and find nothing, which
/// is not the same fact.
#[test]
fn the_dispatch_jitcode_inline_calls_the_build_time_callee() {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
    __prebuild_jitcode_liveness_doubling_interp(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let dispatch = Arc::new(
        __dispatch_jitcode_doubling_interp(&mut asm, 0i64)
            .expect("dispatch lower must succeed for the pipeline fixture"),
    );
    let reached = reachable(&dispatch);
    let names: Vec<&str> = reached.iter().map(|jc| jc.name()).collect();
    assert!(
        names.contains(&"pipeline_double"),
        "`inline_pipeline_int` must splice the host-resolved callee somewhere \
         the registry walk reaches; reached {names:?}",
    );
    assert!(
        reached.iter().any(|jc| jc
            .code
            .iter()
            .any(|b| majit_metainterp::jitcode::is_inline_call_opcode(*b))),
        "the callee must arrive through inline_call_*, not a residual call",
    );
    // The callee is reached at depth 2, so this fixture also pins that the
    // walk does not stop at the dispatch's own slots.
    assert!(
        !dispatch
            .exec
            .descrs
            .iter()
            .filter_map(majit_metainterp::RuntimeBhDescr::as_jitcode)
            .any(|callee| callee.name() == "pipeline_double"),
        "the callee is expected one level below the dispatch (inside the \
         per-opcode arm); if it moved up, this fixture stopped covering the \
         depth it was written for",
    );
}

/// OP_DOUBLE / OP_VOID stay on the sub-JitCode path because they call
/// `inline_pipeline_*` helpers (`arm_body_has_infer_call`). Those callees
/// used to emit `load_state_field` / `store_state_field` against the
/// parent's identity slots. The state-as-vable gate now covers that arm
/// shape, so `rewrite_op_getfield` / `rewrite_op_setfield` land on the
/// live virtualizable (`opimpl_getfield_vable_*` /
/// `opimpl_setfield_vable_*`) and no callee JitCode reads identity slots.
#[test]
fn doubling_state_callee_jitcodes_do_not_read_identity_slots() {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
    __prebuild_jitcode_liveness_doubling_interp(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let dispatch = Arc::new(
        __dispatch_jitcode_doubling_interp(&mut asm, 0i64)
            .expect("dispatch lower must succeed for the pipeline fixture"),
    );
    let reached = reachable(&dispatch);
    let identity_callees: Vec<&str> = reached
        .iter()
        .skip(1)
        .filter(|jc| jc.reads_identity_slots())
        .map(|jc| jc.name())
        .collect();
    assert!(
        identity_callees.is_empty(),
        "DoublingState infer-call arms must use getfield_vable / \
         setfield_vable, not load_state_field / store_state_field; \
         callees with identity slots: {identity_callees:?}"
    );
}
