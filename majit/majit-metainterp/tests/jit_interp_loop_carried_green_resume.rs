//! A loop-carried green written inside a traced arm must come back on every
//! exit that returns to the native loop or its epilogue.
//!
//! `kind` is a function-scope `let mut` green, the shape `aheui.py mainloop`
//! uses for `is_queue`. The SET arm writes it during the tracing walk; Halt
//! then `Finish`es. A portal returning `bool` has no finish projection, so
//! the native epilogue runs and reads `kind`. A resume that carries only the
//! pc leaves the local at its pre-trace value and the epilogue answers
//! false.
//!
//! `warmspot.py handle_jitexception` rebuilds every portal green from
//! `jitexc.py ContinueRunningNormally` before the portal continues.

use std::sync::atomic::{AtomicUsize, Ordering};

use majit_ir::OpRef;
use majit_metainterp::{
    Assembler, JitArgKind, JitCode, JitCodeBuilder, JitDriver, TraceAction, trace_jitcode_with_args,
};

pub type Bytecode = [u8];

const OP_ADD: u8 = 1;
const OP_DEC: u8 = 2;
const OP_BACK: u8 = 3;
const OP_SET: u8 = 4;
const OP_END: u8 = 5;

static COMPILES: AtomicUsize = AtomicUsize::new(0);

static PROBE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

struct KindResumeState {
    acc: i64,
    cnt: i64,
}

#[majit_macros::jit_interp(
    state = KindResumeState,
    env = Bytecode,
    greens = [pc, kind, program],
    state_fields = {
        acc: int,
        cnt: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_kind_resume(program: &Bytecode, threshold: u32, n: i64) -> bool {
    let mut driver: JitDriver<KindResumeState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_, _, _, _| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut kind: usize = 0;
    let mut state = KindResumeState { acc: 0, cnt: n };
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
            OP_ADD => {
                state.acc = state.acc + state.cnt;
            }
            OP_DEC => {
                state.cnt = state.cnt - 1;
            }
            OP_BACK => {
                let target = program[pc] as usize;
                pc += 1;
                if state.cnt > 0 {
                    if target < pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            OP_SET => {
                kind = 1;
            }
            _ => break,
        }
    }
    kind == 1
}

fn program() -> Vec<u8> {
    vec![OP_ADD, OP_DEC, OP_BACK, 0, OP_SET, OP_END]
}

fn interpret(n: i64) -> bool {
    let mut cnt = n;
    loop {
        cnt -= 1;
        if cnt > 0 {
            continue;
        }
        break;
    }
    let kind = 1usize;
    kind == 1
}

fn run(threshold: u32, n: i64) -> (bool, usize) {
    let _guard = PROBE_LOCK.lock();
    COMPILES.store(0, Ordering::Relaxed);
    let got = dispatch_kind_resume(&program(), threshold, n);
    (got, COMPILES.load(Ordering::Relaxed))
}

fn install() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![0]);
    __prebuild_jitcode_liveness_dispatch_kind_resume(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_dispatch_kind_resume(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture")
}

fn install_interleaved() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![0]);
    __prebuild_jitcode_liveness_dispatch_interleaved(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_dispatch_interleaved(&mut asm, 0i64)
        .expect("interleaved dispatch lower must succeed")
}

fn gk(raw: i64) -> majit_ir::GreenKey {
    majit_ir::GreenKey::new(vec![raw])
}

#[test]
fn dispatch_lowers() {
    let _ = install();
}

/// `n == threshold + 1` keeps the walk recording when BACK falls through to
/// SET and Halt. SET runs only inside the walk, so the native `kind` is
/// still 0 unless the Finish resume writes the live green back.
#[test]
fn a_finish_resume_restores_a_loop_carried_green() {
    for (threshold, n) in [(2u32, 3i64), (4, 5), (8, 9)] {
        let (got, compiles) = run(threshold, n);
        assert!(
            compiles >= 1,
            "threshold={threshold} n={n}: no loop compiled, so the walk never \
             reached SET and this case tests nothing"
        );
        assert_eq!(
            got,
            interpret(n),
            "threshold={threshold} n={n}: SET wrote kind inside the walk and \
             Halt finished; the native epilogue still saw the pre-trace kind"
        );
    }
}

/// Item 1: declaration order is not JitCode register order.
///
/// `a` and `b` are loop-head `let`s, so they occupy prefix registers.
/// `kind` is the loop-carried green and lives in its `join_merge`
/// header register (`i1`), while declaration order is
/// `[pc, a, kind, b, program]` — `kind` is `green_int[2]`. Copying
/// `registers_i` as the bank writes `registers_i[2]` into `kind`.
///
/// `n` is well above `threshold` so the ADD/DEC/BACK loop compiles and
/// later fails its `cnt > 0` guard. The blackhole then runs SET and
/// Halt (`DoneWithThisFrame`).
struct InterleavedState {
    acc: i64,
    cnt: i64,
}

#[majit_macros::jit_interp(
    state = InterleavedState,
    env = Bytecode,
    greens = [pc, a, kind, b, program],
    state_fields = {
        acc: int,
        cnt: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_interleaved(program: &Bytecode, threshold: u32, n: i64) -> bool {
    let mut driver: JitDriver<InterleavedState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_, _, _, _| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut kind: usize = 0;
    let a: usize = 7;
    let b: usize = 9;
    let mut state = InterleavedState { acc: 0, cnt: n };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        let a: usize = 7;
        let b: usize = 9;
        jit_merge_point!(driver, program, pc; state);
        let _ = (a, b);
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_ADD => {
                state.acc = state.acc + state.cnt;
            }
            OP_DEC => {
                state.cnt = state.cnt - 1;
            }
            OP_BACK => {
                let target = program[pc] as usize;
                pc += 1;
                if state.cnt > 0 {
                    if target < pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            OP_SET => {
                kind = 1;
            }
            _ => break,
        }
    }
    kind == 1
}

fn run_interleaved(threshold: u32, n: i64) -> (bool, usize) {
    let _guard = PROBE_LOCK.lock();
    COMPILES.store(0, Ordering::Relaxed);
    let got = dispatch_interleaved(&program(), threshold, n);
    (got, COMPILES.load(Ordering::Relaxed))
}

#[test]
fn interleaved_kind_is_not_int_register_2() {
    let jc = install_interleaved();
    let (gi, _, _) = jc
        .merge_point_green_regs()
        .expect("dispatch JitCode has a merge point");
    assert_eq!(gi.len(), 4, "int greens are pc, a, kind, b; got {gi:?}");
    assert_eq!(gi[0], 0, "pc is dispatch i0");
    assert_ne!(
        gi[2], 2,
        "kind is declaration-order green_int[2] but its join_merge \
         register must not be i2 (that slot is a later int). gi={gi:?}"
    );
}

#[test]
fn a_blackhole_finish_maps_kind_by_declaration_not_register() {
    for (threshold, n) in [(2u32, 10i64), (3, 16), (4, 20)] {
        let (got, compiles) = run_interleaved(threshold, n);
        assert!(
            compiles >= 1,
            "threshold={threshold} n={n}: no loop compiled, so the \
             compiled-guard blackhole never ran SET"
        );
        assert!(
            got,
            "threshold={threshold} n={n}: SET wrote kind=1; the resume \
             banks must restore it by declaration order, not by copying \
             registers_i[2] into kind"
        );
    }
}

/// Item 3: a compiled guard failure after SET must resume with the live
/// kind, not the loop-header kind spliced onto a post-guard pc.
///
/// Same program as the Finish fixture: the compiled ADD/DEC/BACK loop
/// fails `cnt > 0`, the blackhole runs SET, and the resume must keep
/// kind=1. Header greens plus a post-guard pc would restore 0.
#[test]
fn a_compiled_guard_failure_keeps_the_kind_set_wrote() {
    for (threshold, n) in [(2u32, 12i64), (3, 18)] {
        let (got, compiles) = run(threshold, n);
        assert!(
            compiles >= 1,
            "threshold={threshold} n={n}: no loop compiled"
        );
        assert!(
            got,
            "threshold={threshold} n={n}: the cnt>0 guard failed after \
             the compiled loop; SET ran in the blackhole and kind must \
             stay 1 (header greens with a post-guard pc would restore 0)"
        );
    }
}

/// Item 4: Abort after SET must re-read the live frame greens, not
/// publish the previous merge-point values.
struct AbortState {
    acc: i64,
    cnt: i64,
}

#[majit_macros::jit_interp(
    state = AbortState,
    env = Bytecode,
    greens = [pc, kind, program],
    state_fields = {
        acc: int,
        cnt: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_abort_after_set(program: &Bytecode, threshold: u32, n: i64) -> bool {
    let mut driver: JitDriver<AbortState> = JitDriver::new(threshold);
    driver.set_param("trace_limit", 8);
    driver.set_on_compile_loop(|_, _, _, _| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut kind: usize = 0;
    let mut state = AbortState { acc: 0, cnt: n };
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
            OP_ADD => {
                state.acc = state.acc + state.cnt;
            }
            OP_DEC => {
                state.cnt = state.cnt - 1;
            }
            OP_BACK => {
                let target = program[pc] as usize;
                pc += 1;
                if state.cnt > 0 {
                    if target < pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            OP_SET => {
                kind = 1;
            }
            _ => break,
        }
    }
    kind == 1
}

fn run_abort(threshold: u32, n: i64) -> bool {
    let _guard = PROBE_LOCK.lock();
    COMPILES.store(0, Ordering::Relaxed);
    dispatch_abort_after_set(&program(), threshold, n)
}

#[test]
fn an_abort_after_set_still_restores_kind() {
    for (threshold, n) in [(2u32, 3i64), (4, 5)] {
        let got = run_abort(threshold, n);
        assert!(
            got,
            "threshold={threshold} n={n}: SET wrote kind inside the \
             aborted walk; the published banks must re-read the live \
             frame so the epilogue sees 1"
        );
    }
}

/// Header-revisit CloseLoop: a walk that returns Continue after a green
/// write, then `adopt_live_greens_as_close`.
///
/// Dispatch JitCode loops until CloseLoop or Finish, so the generated
/// `generate_merge_wrapper` fast path is not a deterministic `#[jit_interp]`
/// outcome. This walk is the same sequence: `BC_JIT_MERGE_POINT` stamps
/// kind=0, a later `int_copy` writes kind=1, the portal frame ends, and
/// the header-revisit adopt must publish 1 (`bhimpl_jit_merge_point`).
#[test]
fn a_header_revisit_after_continue_rereads_kind() {
    let _guard = PROBE_LOCK.lock();
    let mut builder = JitCodeBuilder::new();
    builder.jit_merge_point(0, &[0, 1], &[], &[], &[], &[], &[]);
    builder.load_const_i_value(1, 1);
    let jitcode = builder.finish();

    let mut driver: JitDriver<KindResumeState> = JitDriver::new(1);
    let mut state = KindResumeState { acc: 0, cnt: 1 };
    let program = program();
    let key = gk(0);
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, &program)
            .install_canonical_liveness(&mut driver);
    }
    driver.force_start_tracing(
        key.get_uhash(),
        Some((&key.values, &key.types)),
        0,
        &mut state,
        &program[..],
    );
    assert!(
        driver.is_tracing(),
        "force_start_tracing must start a trace"
    );

    let mut walk_continued = false;
    driver.merge_point(|meta, sym| {
        let ctx = meta.trace_ctx().expect("tracing");
        let action = trace_jitcode_with_args(
            ctx,
            sym,
            &jitcode,
            0,
            |_pc| 0,
            &[
                (JitArgKind::Int, OpRef::ConstInt(0), 0),
                (JitArgKind::Int, OpRef::ConstInt(0), 0),
            ],
        );
        walk_continued = matches!(action, TraceAction::Continue);
        action
    });
    assert!(
        walk_continued,
        "the portal frame must end after the kind write, not CloseLoop/Finish"
    );
    assert!(
        driver.is_tracing(),
        "Continue leaves the trace live for the header-revisit CloseLoop"
    );

    let mut published = None;
    driver.merge_point(|meta, _sym| {
        meta.close_header_revisit(0);
        published = meta.trace_ctx().map(|ctx| ctx.portal_resume_args());
        TraceAction::Abort
    });
    let args = published.expect("header-revisit adopt must see a live TraceCtx");
    assert_eq!(
        args.green_int,
        vec![0, 1],
        "header-revisit must publish the kind written after the last \
         BC_JIT_MERGE_POINT, not the merge-point stamp (0)"
    );
}

/// Header-revisit CloseLoop builds `live_arg_boxes` from this merge
/// point's red operand registers (`reached_loop_header`), so JUMP
/// arity matches LABEL arity with a nonempty red list.
#[test]
fn a_header_revisit_close_matches_label_arity() {
    let _guard = PROBE_LOCK.lock();
    let mut builder = JitCodeBuilder::new();
    builder.ensure_i_regs(4);
    builder.jit_merge_point(0, &[0, 1], &[], &[], &[2, 3], &[], &[]);
    builder.load_const_i_value(1, 1);
    let jitcode = builder.finish();

    let mut driver: JitDriver<KindResumeState> = JitDriver::new(1);
    let mut state = KindResumeState { acc: 7, cnt: 3 };
    let program = program();
    let key = gk(0);
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, &program)
            .install_canonical_liveness(&mut driver);
    }
    driver.force_start_tracing(
        key.get_uhash(),
        Some((&key.values, &key.types)),
        0,
        &mut state,
        &program[..],
    );
    assert!(
        driver.is_tracing(),
        "force_start_tracing must start a trace"
    );

    let acc = OpRef::input_arg_typed(0, majit_ir::Type::Int);
    let cnt = OpRef::input_arg_typed(1, majit_ir::Type::Int);
    let mut walk_continued = false;
    driver.merge_point(|meta, sym| {
        let ctx = meta.trace_ctx().expect("tracing");
        let action = trace_jitcode_with_args(
            ctx,
            sym,
            &jitcode,
            0,
            |_pc| 0,
            &[
                (JitArgKind::Int, OpRef::ConstInt(0), 0),
                (JitArgKind::Int, OpRef::ConstInt(0), 0),
                (JitArgKind::Int, acc, 7),
                (JitArgKind::Int, cnt, 3),
            ],
        );
        walk_continued = matches!(action, TraceAction::Continue);
        action
    });
    assert!(
        walk_continued,
        "the first header visit must Continue so CloseLoop is the revisit"
    );
    assert!(
        driver.is_tracing(),
        "Continue leaves the trace live for the header-revisit CloseLoop"
    );

    let mut jump_arity = None;
    driver.merge_point(|meta, _sym| {
        meta.close_header_revisit(0);
        let ctx = meta.trace_ctx().expect("tracing");
        let boxes = ctx
            .close_jump_boxes
            .as_ref()
            .expect("reached_loop_header must stash close_jump_boxes");
        assert!(
            !boxes.is_empty(),
            "header-revisit close must carry a nonempty red list"
        );
        jump_arity = Some(boxes.len());
        // The generated wrapper returns CloseLoop next. This walk is too
        // short to compile (`GUARD_NOT_INVALIDATED`); Abort after the
        // `live_arg_boxes` stash, matching `a_header_revisit_after_continue_rereads_kind`.
        TraceAction::Abort
    });
    let jump_n = jump_arity.expect("header-revisit must publish JUMP boxes");
    // LABEL is the setup_call InputArgs for acc and cnt.
    assert_eq!(
        jump_n, 2,
        "JUMP arity must equal LABEL arity (acc, cnt reds)"
    );
}

/// cel's float-bank portal: `greens = [pc, program]`, two `[; virt]` banks,
/// no extra mut green. A compiled `cnt > 0` guard failure must end the
/// upstream way (`blackhole.py resume_in_blackhole` / `bhimpl_jit_merge_point`):
/// `ContinueRunningNormally`, `DoneWithThisFrame*`, or an escaping exception.
/// A missing handoff used to panic on a torn-down `TraceCtx`.
struct PcOnlyVirtState {
    regs: majit_metainterp::virt_array::VirtArray<i64>,
    fregs: majit_metainterp::virt_array::VirtArray<f64>,
    ret: i64,
}

#[majit_macros::jit_interp(
    state = PcOnlyVirtState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
        fregs: [float; virt],
        ret: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_pc_only_virt(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<PcOnlyVirtState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_, _, _, _| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = PcOnlyVirtState {
        regs: majit_metainterp::virt_array::VirtArray::from_slice(&[0, n]),
        fregs: majit_metainterp::virt_array::VirtArray::filled(0.0, 1),
        ret: 0,
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    loop {
        jit_merge_point!(driver, program, pc; state);
        if pc >= program.len() {
            break;
        }
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_ADD => {
                state.regs[0] = state.regs[0] + state.regs[1];
            }
            OP_DEC => {
                state.regs[1] = state.regs[1] - 1;
            }
            OP_BACK => {
                let target = program[pc] as usize;
                pc += 1;
                if state.regs[1] > 0 {
                    if target < pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            OP_END => {
                state.ret = state.regs[0];
                break;
            }
            _ => break,
        }
    }
    state.ret
}

fn virt_program() -> Vec<u8> {
    vec![OP_ADD, OP_DEC, OP_BACK, 0, OP_END]
}

fn interpret_virt(n: i64) -> i64 {
    let mut acc = 0i64;
    let mut cnt = n;
    loop {
        acc += cnt;
        cnt -= 1;
        if cnt > 0 {
            continue;
        }
        break;
    }
    acc
}

fn run_pc_only_virt(threshold: u32, n: i64) -> (i64, usize) {
    let _guard = PROBE_LOCK.lock();
    COMPILES.store(0, Ordering::Relaxed);
    let got = dispatch_pc_only_virt(&virt_program(), threshold, n);
    (got, COMPILES.load(Ordering::Relaxed))
}

#[test]
fn a_pc_only_virt_guard_failure_ends_in_continue_running_normally() {
    for (threshold, n) in [(2u32, 12i64), (3, 18)] {
        let (got, compiles) = run_pc_only_virt(threshold, n);
        assert!(
            compiles >= 1,
            "threshold={threshold} n={n}: no loop compiled"
        );
        assert_eq!(
            got,
            interpret_virt(n),
            "threshold={threshold} n={n}: a compiled cnt>0 guard failure \
             with greens=[pc, program] and [; virt] banks must resume from \
             ContinueRunningNormally / DoneWithThisFrame / exception, \
             not panic on a missing TraceCtx"
        );
    }
}

/// A mutating residual after a mid-opcode guard, then a deterministic
/// bridge abort (`request_walk_abort` after the residual has already run).
///
/// `bridge_from_guard_resume_position`'s no-handoff `None` used to rebuild
/// the guard deadframe and `resume_in_blackhole` from there
/// (`ResumeGuardDescr.handle_fail` else-arm). The walk had already executed
/// the residual and `start_bridge_tracing(..., true)` had applied pending
/// fields, so that restart popped the chain twice.
/// `MetaInterp.aborted_tracing` / `convert_and_run_from_pyjitpl` convert
/// the live framestack instead.
const ABORT_AFTER_RESIDUAL_CAP: usize = 8;

#[repr(C)]
struct AbortAfterResidualChain {
    size: i64,
}

extern "C" fn abort_after_residual_push(chain: usize, value: i64) {
    let chain = unsafe { &mut *(chain as *mut AbortAfterResidualChain) };
    chain.size += 1;
    let _ = value;
    majit_metainterp::note_residual_committed();
}

extern "C" fn abort_after_residual_abort(chain: usize) {
    let _ = chain;
    majit_metainterp::request_walk_abort();
}

struct AbortAfterResidualState {
    top: i64,
    sp: usize,
    counter: i64,
    chain: usize,
}

#[majit_macros::jit_interp(
    state = AbortAfterResidualState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        top: int,
        sp: int(usize),
        counter: int,
        chain: ref(AbortAfterResidualChain),
    },
    calls = {
        abort_after_residual_push => residual_void,
        abort_after_residual_abort => residual_void,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_abort_after_residual(
    program: &Bytecode,
    iterations: i64,
    threshold: u32,
    chain: usize,
) -> i64 {
    let mut driver: JitDriver<AbortAfterResidualState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_, _, _, _| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = AbortAfterResidualState {
        top: 0,
        sp: 0,
        counter: iterations,
        chain,
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
            OP_ADD => {
                let value = state.counter;
                state.sp = state.sp + 1;
                if state.sp > ABORT_AFTER_RESIDUAL_CAP {
                    abort_after_residual_push(state.chain, state.top);
                    abort_after_residual_abort(state.chain);
                }
                state.top = value;
            }
            OP_DEC => {
                state.counter = state.counter - 1;
                if state.counter != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            _ => break,
        }
    }
    state.sp as i64
}

fn run_abort_after_residual(iterations: i64, threshold: u32) -> (i64, i64, usize) {
    let _ = majit_metainterp::take_walk_abort();
    let _ = majit_metainterp::take_residual_committed();
    COMPILES.store(0, Ordering::Relaxed);
    let mut chain = AbortAfterResidualChain { size: 0 };
    let height = dispatch_abort_after_residual(
        &[OP_ADD, OP_DEC, OP_END],
        iterations,
        threshold,
        &mut chain as *mut AbortAfterResidualChain as usize,
    );
    (height, chain.size, COMPILES.load(Ordering::Relaxed))
}

#[test]
fn a_mutating_residual_then_deterministic_bridge_abort_is_not_replayed() {
    let _guard = PROBE_LOCK.lock();
    const ITERATIONS: i64 = 2000;
    let expected = ITERATIONS - ABORT_AFTER_RESIDUAL_CAP as i64;

    let cold = run_abort_after_residual(ITERATIONS, u32::MAX);
    assert_eq!(cold.2, 0, "the cold run must not have compiled anything");
    assert_eq!(
        cold.1, expected,
        "cold run: every push past the first {ABORT_AFTER_RESIDUAL_CAP} \
         displaces one word"
    );

    let aborts_before = majit_metainterp::symbolic_residual_trace_aborts();
    let warm = run_abort_after_residual(ITERATIONS, 3);
    assert!(
        warm.2 >= 1,
        "the loop did not compile, so the guard-resume walk never ran"
    );
    assert!(
        majit_metainterp::symbolic_residual_trace_aborts() > aborts_before,
        "the guard-resume walk never consumed request_walk_abort, so the \
         deterministic-bridge-abort handoff was not reached"
    );
    assert_eq!(
        warm.1, cold.1,
        "a mutating residual followed by a deterministic bridge abort ran \
         a different NUMBER of times compiled than interpreted; the no-handoff \
         None restarted from the guard deadframe and replayed the residual"
    );
    assert_eq!(warm.0, cold.0, "compiled height");
}
