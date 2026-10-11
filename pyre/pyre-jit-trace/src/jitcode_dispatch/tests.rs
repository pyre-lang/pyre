use super::*;
use crate::jitcode_runtime::{insns_opname_to_byte, named_jitcode};
use majit_ir::Type;
use majit_metainterp::{
    JitCodeSym, NonstandardVable, TraceAction, VableArrayStore, make_fail_descr,
};

static STATIC_REFUSAL_PREFIX_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[test]
fn session_roots_cover_nested_attempts_and_vec_frame_retirement() {
    let _runtime = crate::trace_ctx_for_test(0);
    let _stw = majit_gc::gc_sync::quiesce_mutators();
    let outer = std::cell::RefCell::new(WalkSession::default());
    let inner = std::cell::RefCell::new(WalkSession::default());
    let outer_roots = WalkSessionRoots::new(&outer);
    let inner_roots = WalkSessionRoots::new(&inner);
    let parent = |word: usize| InlineParentFrame {
        jitcode_index: 0,
        call_jitcode_pc: None,
        call_stack_overrides: vec![(0, word as pyre_object::PyObjectRef)],
        blackhole: Some(InlineParentBlackhole {
            resume_pc: 0,
            int_values: Vec::new(),
            ref_values: vec![(0, (word + 0x10) as pyre_object::PyObjectRef)],
            float_values: Vec::new(),
        }),
        resume_coord: ParentResumeCoord::Backxlat(0),
        resume_marker_jit_pc: None,
        boxes: vec![OpRef::const_ptr(majit_ir::GcRef(word + 0x20))],
        registers_r: None,
        registers_i: None,
        registers_f: None,
        frame_state: None,
        caller_py_pc: None,
    };
    let frames = vec![
        InlineFrameGuard::enter(&outer, 0, false, vec![parent(0x1000)]),
        InlineFrameGuard::enter(&outer, 0, false, vec![parent(0x2000)]),
    ];
    let inner_frame = InlineFrameGuard::enter(&inner, 0, false, vec![parent(0x3000)]);
    let forward = || {
        let mut seen = Vec::new();
        majit_gc::shadow_stack::walk_my_extra_areas(|root| {
            if (0x1000..0x5000).contains(&root.0) {
                seen.push(root.0);
                root.0 += 0x80;
            }
        });
        seen.sort_unstable();
        seen
    };
    assert_eq!(
        forward(),
        [
            0x1000, 0x1010, 0x1020, 0x2000, 0x2010, 0x2020, 0x3000, 0x3010, 0x3020
        ]
    );
    drop(inner_frame);
    drop(inner_roots);
    assert_eq!(forward(), [0x1080, 0x1090, 0x10a0, 0x2080, 0x2090, 0x20a0]);
    // Bridge reconstruction stores frame guards in a Vec; their drop order
    // must not control the session owner's root registration lifetime.
    drop(frames);
    assert!(outer.borrow().at_portal());
    outer.borrow_mut().tmpreg_r = OpRef::const_ptr(majit_ir::GcRef(0x4000));
    assert_eq!(forward(), [0x4000]);
    assert_eq!(
        outer.borrow().tmpreg_r,
        OpRef::const_ptr(majit_ir::GcRef(0x4080))
    );
    drop(outer_roots);
    assert!(forward().is_empty());
}

#[test]
fn session_root_callback_rejects_a_borrow_across_collection() {
    let session = std::cell::RefCell::new(WalkSession::default());
    let held = session.borrow();
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        // Check the borrow contract without unwinding through a global
        // root traversal.
        walk_session_roots(
            (&session as *const std::cell::RefCell<WalkSession>).cast(),
            &mut |_| {},
        );
    }));
    assert!(failed.is_err());
    drop(held);
}

fn finish_payload_of(outcome: &DispatchOutcome) -> Option<(OpRef, Type)> {
    match outcome {
        DispatchOutcome::Terminate {
            finish_arg,
            finish_arg_type,
        } => Some((*finish_arg, *finish_arg_type)),
        _ => None,
    }
}

#[test]
fn finish_payload_root_walker_writes_back_forwarded_const_ptr() {
    let _runtime = crate::trace_ctx_for_test(0);
    let _stw = majit_gc::gc_sync::quiesce_mutators();
    let session = std::cell::RefCell::new(WalkSession {
        finish_payload: Some((OpRef::const_ptr(majit_ir::GcRef(0x1000)), Type::Ref)),
        ..WalkSession::default()
    });
    let _roots = WalkSessionRoots::new(&session);
    let _stw = majit_gc::gc_sync::quiesce_mutators();
    majit_gc::shadow_stack::walk_my_extra_areas(|gcref| {
        if gcref.0 == 0x1000 {
            gcref.0 = 0x2000;
        }
    });
    assert_eq!(
        session.borrow().finish_payload,
        Some((OpRef::const_ptr(majit_ir::GcRef(0x2000)), Type::Ref))
    );
}

extern "C" fn count_static_refusal_prefix() {
    STATIC_REFUSAL_PREFIX_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

struct StaticRefusalSym;

impl JitCodeSym for StaticRefusalSym {
    fn total_slots(&self) -> usize {
        0
    }

    fn loop_header_pc(&self) -> usize {
        0
    }
}

/// The generated dispatch loop re-runs its source arm when a trace start
/// returns no handoff. The static refusal must therefore happen before the
/// bound prefix call: the arm itself then performs that side effect once.
#[test]
fn unbound_symbolic_residual_refuses_before_a_bound_residual_side_effect() {
    let before = STATIC_REFUSAL_PREFIX_CALLS.load(std::sync::atomic::Ordering::Relaxed);
    let mut builder = majit_metainterp::JitCodeBuilder::new();
    let bound = builder.add_fn_ptr(count_static_refusal_prefix as *const ());
    builder.residual_call_void_canonical_via_target_with_effect_info(
        bound,
        &[],
        majit_ir::EffectInfo::const_new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    );
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments([
        "unbound_symbolic_residual_refuses_before_a_bound_residual_side_effect",
    ]);
    let unbound = builder.add_fn_ptr(symbolic as usize as *const ());
    builder.residual_call_void_canonical_via_target_with_effect_info(
        unbound,
        &[],
        majit_ir::EffectInfo::const_new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    );
    builder.void_return();
    let jitcode = builder.finish();

    let scan = jitcode.reachable_symbolic_residuals();
    assert_eq!(scan.visited_jitcodes, 1);
    assert_eq!(scan.targets, vec![symbolic]);

    let mut ctx = TraceCtx::for_test_types(&[]);
    let action =
        majit_metainterp::trace_jitcode(&mut ctx, &mut StaticRefusalSym, &jitcode, 0, |_pc| 0);
    assert!(matches!(action, TraceAction::Abort));
    assert_eq!(
        STATIC_REFUSAL_PREFIX_CALLS.load(std::sync::atomic::Ordering::Relaxed),
        before,
        "the reachable-set refusal must run before the bound residual prefix",
    );

    // `None` from the generated merge wrapper falls through to this source-arm
    // execution. It is the only execution of the prefix after the preflight.
    count_static_refusal_prefix();
    assert_eq!(
        STATIC_REFUSAL_PREFIX_CALLS.load(std::sync::atomic::Ordering::Relaxed),
        before + 1,
        "the source arm must apply the observable side effect exactly once",
    );
}

/// Pyre's runtime fnaddr patch must leave the portal closure free of symbolic
/// residual targets. Keep the visited count and cold-scan duration visible so
/// this trace-start gate has a measured cost on the table it protects.
#[test]
fn portal_reachable_symbolic_residual_scan_is_empty() {
    let _ = crate::jitcode_runtime::all_jitcodes();
    crate::jitcode_runtime::install_global_build_descr_pool();
    let canonical = crate::jitcode_runtime::portal_jitcode()
        .expect("the build-time table registers the eval portal");
    let portal = crate::jitcode_runtime::get_runtime_jitcode_by_index(canonical.index())
        .expect("the eval portal index resolves to a runtime jitcode");

    let started = std::time::Instant::now();
    let scan = portal.reachable_symbolic_residuals();
    let elapsed = started.elapsed();
    eprintln!(
        "[symbolic-residual-scan] visited={} elapsed_ns={} unbound={}",
        scan.visited_jitcodes,
        elapsed.as_nanos(),
        scan.targets.len(),
    );
    let fresh = majit_metainterp::jitcode::compute_reachable_symbolic_residuals(&portal);
    assert_eq!(
        scan, &fresh,
        "the build-time table must name the same closure a body scan walks",
    );

    assert!(
        scan.visited_jitcodes > 0,
        "the portal itself must be visited"
    );
    assert_eq!(
        scan.targets,
        Vec::<i64>::new(),
        "jit_fnaddr bindings must resolve every portal-reachable residual",
    );
}

/// jd1's `unpackiterable_portal` is a second driver, not the eval portal
/// above. A `||` closure in its drain livevar reloads used to mint
/// `target:closure.call` hashes that this scan (and the walk) refuse.
#[test]
fn unpackiterable_portal_reachable_symbolic_residual_scan_is_empty() {
    let _ = crate::jitcode_runtime::all_jitcodes();
    crate::jitcode_runtime::install_global_build_descr_pool();
    let canonical =
        crate::jitcode_runtime::portal_jitcode_for_key("baseobjspace::unpackiterable_portal")
            .expect("the build-time table registers the unpackiterable portal");
    let portal = crate::jitcode_runtime::get_runtime_jitcode_by_index(canonical.index())
        .expect("the unpackiterable portal index resolves to a runtime jitcode");

    let scan = portal.reachable_symbolic_residuals();
    let fresh = majit_metainterp::jitcode::compute_reachable_symbolic_residuals(&portal);
    assert_eq!(
        scan, &fresh,
        "the build-time table must name the same closure a body scan walks",
    );
    assert!(
        scan.visited_jitcodes > 0,
        "the unpackiterable portal itself must be visited"
    );
    assert_eq!(
        scan.targets,
        Vec::<i64>::new(),
        "named shadow_stack_get / next / drain residuals must resolve; \
         a leftover target:closure.call hash aborts the jd1 walk",
    );
}

#[test]
fn propagated_subwalk_abort_cannot_rebind_its_pc_to_a_caller_frame() {
    let mut session = WalkSession::default();

    assert!(session.claim_abort_coordinate(true));
    assert!(!session.claim_abort_coordinate(true));
    assert!(!session.claim_abort_coordinate(false));
}

/// One `residual_call_r_i/iRd>i` with an empty argument varlist, whose funcbox
/// operand names the Int-bank slot `funcbox`.
fn residual_call_with_funcbox(funcbox: u8) -> Vec<u8> {
    let byte = *insns_opname_to_byte()
        .get("residual_call_r_i/iRd>i")
        .expect("`residual_call_r_i/iRd>i` must be in insns table");
    vec![byte, funcbox, 0, 0, 0, 0]
}

fn goto_if_not(cond: u8, target: u16) -> Vec<u8> {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not/iL")
        .expect("`goto_if_not/iL` must be in insns table");
    vec![byte, cond, target as u8, (target >> 8) as u8]
}

/// One `inline_call_r_v/dR` with an empty argument varlist, naming descriptor
/// `descr`.  The callee is whatever the `callee_summary` closure answers for
/// that index, which is how a callee's verdict is stated without a jitcode
/// table.
fn inline_call_with_descr(descr: u16) -> Vec<u8> {
    let byte = *insns_opname_to_byte()
        .get("inline_call_r_v/dR")
        .expect("`inline_call_r_v/dR` must be in insns table");
    vec![byte, descr as u8, (descr >> 8) as u8, 0]
}

fn void_return() -> Vec<u8> {
    vec![
        *insns_opname_to_byte()
            .get("void_return/")
            .expect("`void_return/` must be in insns table"),
    ]
}

fn int_copy_const(src: u8, dst: u8) -> Vec<u8> {
    let byte = *insns_opname_to_byte()
        .get("int_copy/c>i")
        .expect("`int_copy/c>i` must be in insns table");
    vec![byte, src, dst]
}

fn arraylen_gc(array: u8, descr: u16, dst: u8) -> Vec<u8> {
    let byte = *insns_opname_to_byte()
        .get("arraylen_gc/rd>i")
        .expect("`arraylen_gc/rd>i` must be in insns table");
    vec![byte, array, descr as u8, (descr >> 8) as u8, dst]
}

/// The generated `__majit_wrap_*` gateways all put their un-lowerable call — the
/// `#[dont_look_inside]` arity-error formatter — on the arm the argument-count
/// check rejects into.  The walk is execution-driven, so a call with the right
/// count never steps there; declining the wrapper for it refuses a body the
/// walk would have recorded.  The blocker is still reported, on the leg that
/// says a rewind covers it.
#[test]
fn a_blocker_on_an_arm_reached_without_an_effect_is_not_a_decline() {
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments(["__len"]);
    let mut code = goto_if_not(0, 11);
    code.extend(residual_call_with_funcbox(1));
    code.extend(void_return());
    code.extend(void_return());
    assert_eq!(code.len(), 12, "the reject arm ends at the second return");

    let summary = super::inline_call::summarize_body_blockers(&code, 1, &[symbolic], |_| None);
    assert_eq!(summary.blocker_effect_free, Some(symbolic));
    assert_eq!(
        summary.blocker_after_effect, None,
        "nothing ran before the blocker, so the descent rolls back"
    );
    assert!(!summary.may_execute_effect);
}

/// The same blocker behind an executed residual call is a decline: that call
/// applied an effect the rollback cannot undo, so an abort there would re-run
/// the Python call and apply it twice.
#[test]
fn a_blocker_behind_an_executed_call_is_a_decline() {
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments(["__len"]);
    let real = 0x1234_5678i64;
    assert!(!majit_jitcode::codewriter::call::is_symbolic_fnaddr(real));

    let mut code = residual_call_with_funcbox(1);
    code.extend(goto_if_not(0, 17));
    code.extend(residual_call_with_funcbox(2));
    code.extend(void_return());
    code.extend(void_return());
    assert_eq!(code.len(), 18);

    let summary =
        super::inline_call::summarize_body_blockers(&code, 1, &[real, symbolic], |_| None);
    assert_eq!(summary.blocker_after_effect, Some(symbolic));
    assert_eq!(
        summary.blocker_effect_free, None,
        "the blocker belongs to one leg, and recording it on both would let a \
         decline read as a rewind"
    );
    assert!(summary.may_execute_effect);
    assert!(!summary.body_not_walked);
}

/// A constant condition is the same condition the generated walk reads.  The
/// scan must follow only its executed successor; joining the dead fallthrough
/// would turn an effect and blocker that cannot execute into a false decline.
#[test]
fn a_known_goto_condition_does_not_scan_the_dead_arm() {
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments(["__len"]);
    let real = 0x1234_5678i64;

    // 0: i0 = const <src>; 3: goto_if_not i0 -> 20
    // 7: residual(real); 13: residual(symbolic); 19: return; 20: return
    //
    // `int_copy/c>i` carries its source inline as one signed byte, so the
    // condition is spelled by that byte and there is no `constants_i` slot to
    // vary: the two bodies differ only in it.
    let body = |src: u8| {
        let mut code = int_copy_const(src, 0);
        code.extend(goto_if_not(0, 20));
        code.extend(residual_call_with_funcbox(2));
        code.extend(residual_call_with_funcbox(3));
        code.extend(void_return());
        code.extend(void_return());
        assert_eq!(code.len(), 21);
        code
    };

    let summary =
        super::inline_call::summarize_body_blockers(&body(0), 1, &[0, real, symbolic], |_| None);
    assert_eq!(summary.blocker_after_effect, None);
    assert_eq!(summary.blocker_effect_free, None);
    assert!(!summary.may_execute_effect);

    // Flip only the condition.  The same body now executes the fallthrough,
    // and the blocker after the real residual call must remain a decline.
    let executed =
        super::inline_call::summarize_body_blockers(&body(1), 1, &[0, real, symbolic], |_| None);
    assert_eq!(executed.blocker_after_effect, Some(symbolic));
    assert!(executed.may_execute_effect);
}

/// Generated builtin gateways receive an argument slice in r0 and branch on
/// its length before entering the typed body.  The call site knows that length,
/// so the path-sensitive scan must not join the arity-error arm back in.
#[test]
fn a_known_wrapper_array_length_selects_only_the_executed_arity_arm() {
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments(["__len"]);
    let real = 0x1234_5678i64;

    // 0: i0 = arraylen(r0); 5: goto_if_not i0 -> 22
    // 9: residual(real); 15: residual(symbolic); 21: return; 22: return
    let mut code = arraylen_gc(0, 0, 0);
    code.extend(goto_if_not(0, 22));
    code.extend(residual_call_with_funcbox(1));
    code.extend(residual_call_with_funcbox(2));
    code.extend(void_return());
    code.extend(void_return());
    assert_eq!(code.len(), 23);

    let scan = |len| {
        super::inline_call::summarize_body_blockers_with(
            &code,
            1,
            &[real, symbolic],
            |_, _| None,
            &mut |_, _| true,
            &mut |_| false,
            &mut |_| None,
            &[(0, len)],
        )
    };
    let empty = scan(0);
    assert_eq!(empty.blocker_after_effect, None);
    assert!(!empty.may_execute_effect);

    let nonempty = scan(1);
    assert_eq!(nonempty.blocker_after_effect, Some(symbolic));
    assert!(nonempty.may_execute_effect);
}

/// An executed residual call the effectinfo names effect-free -- elidable,
/// loop-invariant, or `not_in_trace` -- applies nothing a rollback would have
/// to undo, so a blocker behind it stays on the rewind leg.  The scan asks
/// through the `call_effect_free` hook; the helper without the hook keeps the
/// conservative reading.
#[test]
fn a_blocker_behind_an_effect_free_call_is_not_a_decline() {
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments(["__len"]);
    let real = 0x1234_5678i64;

    let mut code = residual_call_with_funcbox(1);
    code.extend(residual_call_with_funcbox(2));
    code.extend(void_return());

    let mut asked = Vec::new();
    let summary = super::inline_call::summarize_body_blockers_with(
        &code,
        1,
        &[real, symbolic],
        |_, _| None,
        &mut |_, _| true,
        &mut |descr_index| {
            asked.push(descr_index);
            true
        },
        &mut |_| None,
        &[],
    );
    assert_eq!(
        asked,
        vec![0],
        "only the real call's descr is asked; the blocker stops the walk"
    );
    assert_eq!(summary.blocker_effect_free, Some(symbolic));
    assert_eq!(summary.blocker_after_effect, None);
    assert!(!summary.may_execute_effect);

    let conservative =
        super::inline_call::summarize_body_blockers(&code, 1, &[real, symbolic], |_| None);
    assert_eq!(conservative.blocker_after_effect, Some(symbolic));
}

/// A field write into an object this body allocated is no effect: a rewind
/// drops the allocation.  The same write into a register the body did not
/// fill from a `new*` keeps the conservative reading.
#[test]
fn a_write_into_a_fresh_allocation_is_not_an_effect() {
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments(["__len"]);
    let table = insns_opname_to_byte();
    let new = *table
        .get("new/d>r")
        .expect("`new/d>r` must be in insns table");
    let setfield = *table
        .get("setfield_gc_i/rid")
        .expect("`setfield_gc_i/rid` must be in insns table");

    // r1 = new; r1.field = i0; residual(symbolic)
    let mut code = vec![new, 0, 0, 1, setfield, 1, 0, 0, 0];
    code.extend(residual_call_with_funcbox(1));
    code.extend(void_return());
    let summary = super::inline_call::summarize_body_blockers(&code, 1, &[symbolic], |_| None);
    assert_eq!(summary.blocker_effect_free, Some(symbolic));
    assert_eq!(summary.blocker_after_effect, None);
    assert!(!summary.may_execute_effect);

    // r2 was never allocated here: the write is an effect.
    let mut code = vec![new, 0, 0, 1, setfield, 2, 0, 0, 0];
    code.extend(residual_call_with_funcbox(1));
    code.extend(void_return());
    let summary = super::inline_call::summarize_body_blockers(&code, 1, &[symbolic], |_| None);
    assert_eq!(summary.blocker_after_effect, Some(symbolic));
    assert!(summary.may_execute_effect);
}

/// A `switch` whose arm table is known continues only at its arms and the
/// fallthrough, so the state at one arm's switch does not reach a region no
/// arm names; without the table every instruction start is a successor.
#[test]
fn a_switch_with_a_known_table_does_not_leak_its_state_to_every_start() {
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments(["__len"]);
    let real = 0x1234_5678i64;
    let switch = *insns_opname_to_byte()
        .get("switch/id")
        .expect("`switch/id` must be in insns table");

    // 0: residual(real)  6: switch i0 d0  10: return  11: residual(symbolic)  17: return
    let mut code = residual_call_with_funcbox(1);
    code.extend([switch, 0, 0, 0]);
    code.extend(void_return());
    code.extend(residual_call_with_funcbox(2));
    code.extend(void_return());
    assert_eq!(code.len(), 18);

    let known = super::inline_call::summarize_body_blockers_with(
        &code,
        1,
        &[real, symbolic],
        |_, _| None,
        &mut |_, _| true,
        &mut |_| false,
        &mut |_| Some(vec![10]),
        &[],
    );
    assert_eq!(
        known.blocker_after_effect, None,
        "pc 11 is no arm and no fallthrough"
    );
    assert_eq!(known.blocker_effect_free, None);

    let widened =
        super::inline_call::summarize_body_blockers(&code, 1, &[real, symbolic], |_| None);
    assert_eq!(widened.blocker_after_effect, Some(symbolic));
}

/// A callee's blocker is judged by what has run in the caller, not by what had
/// run in the callee: the same body reached with an effect behind it holds a
/// blocker no rollback covers.
#[test]
fn a_callees_rewindable_blocker_is_not_rewindable_behind_an_effect() {
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments(["__len"]);
    let real = 0x1234_5678i64;
    let callee = majit_jitcode::codewriter::jitcode::DescentBlockerSummary {
        blocker_effect_free: Some(symbolic),
        ..Default::default()
    };

    let mut code = inline_call_with_descr(7);
    code.extend(void_return());
    let clean = super::inline_call::summarize_body_blockers(&code, 1, &[real], |d| {
        (d == 7).then_some(callee)
    });
    assert_eq!(clean.blocker_effect_free, Some(symbolic));
    assert_eq!(
        clean.blocker_after_effect, None,
        "nothing ran before the descent entered the callee"
    );

    let mut code = residual_call_with_funcbox(0);
    code.extend(inline_call_with_descr(7));
    code.extend(void_return());
    let behind = super::inline_call::summarize_body_blockers(&code, 1, &[real], |d| {
        (d == 7).then_some(callee)
    });
    assert_eq!(behind.blocker_after_effect, Some(symbolic));
    assert_eq!(behind.blocker_effect_free, None);
}

fn raise_r(reg: u8) -> Vec<u8> {
    let byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    vec![byte, reg]
}

/// `live/` carries no argcodes but dispatch skips `liveness::OFFSET_SIZE`
/// bytes of offset, so the encoding is one opcode plus that many zeroes.
fn live() -> Vec<u8> {
    let mut v = vec![
        *insns_opname_to_byte()
            .get("live/")
            .expect("`live/` must be in insns table"),
    ];
    v.extend(std::iter::repeat_n(
        0u8,
        majit_jitcode::liveness::OFFSET_SIZE,
    ));
    v
}

fn catch_exception(target: u16) -> Vec<u8> {
    let byte = *insns_opname_to_byte()
        .get("catch_exception/L")
        .expect("`catch_exception/L` must be in insns table");
    vec![byte, target as u8, (target >> 8) as u8]
}

/// A raise caught in this frame is not a frame exit.  `finishframe_lookahead_at`
/// skips the `live/` and takes the `catch_exception/L` behind it, so the handler
/// runs forward over bytes no fallthrough reaches — treating the raise as
/// terminal would leave that whole region, and any blocker in it, unwalked.
#[test]
fn a_raise_caught_in_this_frame_reaches_its_handler() {
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments(["__len"]);
    let mut code = raise_r(0);
    code.extend(live());
    let handler = (code.len() + 3 + 1) as u16;
    code.extend(catch_exception(handler));
    code.extend(void_return());
    assert_eq!(code.len(), handler as usize, "the handler follows the exit");
    code.extend(residual_call_with_funcbox(1));
    code.extend(void_return());

    let summary = super::inline_call::summarize_body_blockers(&code, 1, &[symbolic], |_| None);
    assert_eq!(
        summary.blocker_after_effect,
        Some(symbolic),
        "the handler is entered with the join of every state in the region, so \
         its blocker is one no rollback covers"
    );
    assert!(
        !summary.may_execute_effect,
        "the handler's seeded effect classifies the blocker behind it; the body \
         itself still executes none"
    );
    assert!(!summary.body_not_walked);
}

/// The join a handler is entered with makes its registers unknown, but not the
/// constant slots behind them: nothing writes those, so a funcbox read straight
/// out of one holds at every point the handler can be entered from.  Blanking
/// the whole Int bank instead would leave every blocker in every handler
/// unnamed, which reads as a body that has none.
#[test]
fn a_blocker_in_a_handler_reads_its_funcbox_from_the_surviving_constants() {
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments(["__len"]);
    let handler = 3 + 1;
    let mut code = catch_exception(handler);
    code.extend(void_return());
    assert_eq!(
        code.len(),
        handler as usize,
        "the handler follows the fallthrough"
    );
    code.extend(residual_call_with_funcbox(1));
    code.extend(void_return());

    let summary = super::inline_call::summarize_body_blockers(&code, 1, &[symbolic], |_| None);
    assert_eq!(summary.blocker_after_effect, Some(symbolic));
    assert_eq!(
        summary.blocker_effect_free, None,
        "the handler runs behind whatever raised into it"
    );
    assert!(!summary.body_not_walked);
}

/// An `inline_call` whose descr does not resolve to a JitCode is the same
/// condition one level down: a callee the scan holds nothing about.  Falling
/// through would read it as executing no effect and holding no blocker — the
/// most permissive answer there is, reached by not looking — and would let the
/// blocker behind it be classified as effect-free, which says a rewind covers
/// it.
#[test]
fn an_inline_call_whose_callee_does_not_resolve_is_a_decline() {
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments(["__len"]);
    let mut code = inline_call_with_descr(7);
    code.extend(residual_call_with_funcbox(1));
    code.extend(void_return());

    let summary = super::inline_call::summarize_body_blockers(&code, 1, &[symbolic], |_| None);
    assert!(summary.body_not_walked);
    assert_eq!(
        summary.blocker_effect_free, None,
        "an unresolved callee may have executed an effect, so nothing behind \
         it is rewindable"
    );
    assert_eq!(summary.blocker_after_effect, Some(symbolic));
    assert!(summary.may_execute_effect);
}

/// Every `JitCode` descr names a body the jitcode table holds.
///
/// The descent scan answers `body_not_walked` — which declines the descent —
/// when `get_jitcode_ref_by_index` produces nothing, and that function answers
/// `None` only past the end of the table.  So whether that arm is ever taken
/// is exactly this question, and the generated tables settle it rather than a
/// run does.  Asserted rather than measured once: the comment on the arm says
/// "unreachable in a consistent build", and this is the consistency it means.
#[test]
fn every_jitcode_descr_names_an_installed_body() {
    let jitcodes = crate::jitcode_runtime::jitcode_count();
    let descrs = crate::jitcode_runtime::descr_ref_table();
    let mut named = 0usize;
    let mut out_of_range = Vec::new();
    for slot in 0..descrs.len() {
        let Some(index) = descrs
            .at(slot)
            .and_then(|descr| descr.as_jitcode_descr().map(|jc| jc.jitcode_index()))
        else {
            continue;
        };
        named += 1;
        if index >= jitcodes {
            out_of_range.push((slot, index));
        }
    }
    // A table holding no JitCode descr at all would pass the assertion below
    // while proving nothing, which is the shape of a test that cannot fail.
    assert!(
        named > 0,
        "no JitCode descr among {} descrs -- this test proves nothing here",
        descrs.len()
    );
    assert!(
        out_of_range.is_empty(),
        "{} of {named} JitCode descrs name an index outside the {jitcodes}-entry \
         jitcode table: {out_of_range:?}",
        out_of_range.len(),
    );
    eprintln!("[density] {named} JitCode descrs over {jitcodes} jitcodes");
}

/// A body the decoder cannot read to the end declines on that alone.  The
/// instruction starts would otherwise name a prefix, and the `push!` gate drops
/// targets that are not starts — so every path out of the undecodable byte
/// would be missing from the graph, and a path the scan never walks reports no
/// blocker.  Naming none is what an all-`None` summary already means, hence the
/// separate flag.
#[test]
fn a_body_that_does_not_decode_to_the_end_is_a_decline() {
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments(["__len"]);
    let mut code = residual_call_with_funcbox(0);
    let goto = goto_if_not(0, 5);
    code.extend(&goto[..goto.len() - 1]);
    assert_eq!(code.len(), 9, "the branch is one byte short of its label");

    let summary = super::inline_call::summarize_body_blockers(&code, 1, &[symbolic], |_| None);
    assert!(summary.body_not_walked);
    assert!(summary.may_execute_effect);
    assert_eq!(summary.blocker_effect_free, None);
    assert_eq!(
        summary.blocker_after_effect, None,
        "the decline is the unread region, so there is no blocker to name"
    );
}

/// The `s.add(x)` substitution replaces a residual the executed-effect odometer
/// counts (`CallFn`) with one it judges by descr shape alone.  A `Ref` result
/// clears the `Void` write proxy, so the helper tag is the only thing left
/// saying an insert happened — and the element's `__hash__` cannot stand in for
/// it, since a cpyext `tp_hash` runs native code and moves no frame-entry
/// odometer.  Drop the tag and a nested abort rewinds past a completed insert.
#[test]
fn the_substituted_set_add_descr_still_reads_as_a_live_heap_write() {
    let descr = super::specialize::set_add_method_descr();
    let call_descr = descr
        .as_call_descr()
        .expect("the substitution installs a call descr");
    assert_eq!(
        call_descr.result_type(),
        Type::Ref,
        "`set.add` returns None, so the Void write proxy cannot see this store"
    );
    assert!(
        super::residual_call::helper_kind_writes_live_heap(
            call_descr.get_extra_info().runtime_helper
        ),
        "an untagged descr takes the insert out of the executed-effect odometer"
    );
}

/// The zero-argument `super()` substitution has the same two obligations, and
/// for it they are load-bearing on a shape the corpus runs: the leaf reaches
/// `_super_check`'s `__class__` lookup, which a property answers with arbitrary
/// Python.  The `Ref` result clears the `Void` write proxy, so the helper tag
/// is again the only thing that keeps the call on the executed-effect
/// odometer; and the effect has to stay may-force, because that is what puts
/// `select_residual_call_opcode` on the branch that emits the vrefs bracket and
/// the trailing `GuardNotForced` this substitution relies on.
#[test]
fn the_substituted_bare_super_descr_is_a_may_force_live_heap_write() {
    let descr = super::specialize::bare_super_from_frame_descr();
    let call_descr = descr
        .as_call_descr()
        .expect("the substitution installs a call descr");
    assert_eq!(
        call_descr.result_type(),
        Type::Ref,
        "the proxy comes back as the result, so the Void write proxy is blind to it"
    );
    assert!(
        super::residual_call::helper_kind_writes_live_heap(
            call_descr.get_extra_info().runtime_helper
        ),
        "an untagged descr lets a nested abort rewind past a `__class__` property that ran"
    );
    assert!(
        call_descr
            .get_extra_info()
            .check_forces_virtual_or_virtualizable(),
        "a non-forcing effect would drop the vrefs bracket and the GuardNotForced"
    );
}

#[test]
fn specialised_pair_unpack_recognises_the_float_layout() {
    use super::specialize::{SpecialisedPairKind, specialised_pair_kind};
    use pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_FF_TYPE;

    assert_eq!(
        specialised_pair_kind(&SPECIALISED_TUPLE_FF_TYPE),
        Some(SpecialisedPairKind::Float),
    );
}

fn test_fbw_mode() -> FbwWalkMode<crate::state::PyreSym> {
    FbwWalkMode::default()
}

/// Install the minimal `-live-`-anchored outer JitCode required by
/// per-opcode guard tests. Production obtains this coordinate from the
/// codewriter; these isolated dispatcher fixtures supply the same shape.
fn test_outer_resume_jitcode_index() -> u32 {
    thread_local! {
        static INDEX: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
    }
    if let Some(index) = INDEX.with(|slot| slot.get()) {
        return index;
    }
    let runtime_jc = majit_metainterp::jitcode::JitCode::new("guard_resume_test");
    runtime_jc.set_body(majit_jitcode::jitcode::JitCodeBody {
        code: vec![crate::state::op_live(), 0, 0],
        startpoints: Some([0_usize].into_iter().collect()),
        ..Default::default()
    });
    let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
    pyjit.jitcode = std::sync::Arc::new(runtime_jc);
    pyjit.metadata.is_drained = true;
    // Forward `(jitcode_pc, py_pc)` resume markers a real drained JitCode
    // carries: guard capture reads the innermost frame's Python pc forward
    // from this table (never projected backward). The single-instruction
    // synthetic body maps every JitCode offset to py_pc 0.
    pyjit.metadata.forward_py_pc_marker_by_jit_pc = vec![(0, 0)];
    pyjit.metadata.forward_py_pc_pred_by_jit_pc = vec![(0, 0)];
    let jitcode = crate::state::install_jitcode_for(std::ptr::null(), std::sync::Arc::new(pyjit))
        as *const crate::state::JitCode;
    let index = unsafe { (*jitcode).index as u32 };
    INDEX.with(|slot| slot.set(Some(index)));
    index
}

#[test]
fn after_residual_guard_uses_trailing_live_before_fallthrough_twin() {
    let int_add = *insns_opname_to_byte()
        .get("int_add/ii>i")
        .expect("int_add/ii>i must be in the insns table");
    let live = crate::state::op_live();
    let runtime_jc = majit_metainterp::jitcode::JitCode::new("after_residual_marker_test");
    runtime_jc.set_body(majit_jitcode::jitcode::JitCodeBody {
        // The immediate marker at pc=4 models the call's trailing `-live-`;
        // pc=7 models the later normal-fallthrough marker a Python-PC twin can
        // resolve to after `catch_exception/L`.
        code: vec![int_add, 0, 1, 2, live, 0, 0, live, 0, 0],
        startpoints: Some([0_usize, 4, 7].into_iter().collect()),
        ..Default::default()
    });
    let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
    pyjit.jitcode = std::sync::Arc::new(runtime_jc);
    pyjit.metadata.after_residual_call_resume_pred_by_jit_pc = vec![(0, Some(7))];

    assert_eq!(
        super::resume_snapshot::after_residual_guard_marker(&pyjit, 0, Some(0)),
        Some(4),
        "handler liveness must come from the trailing live before catch_exception",
    );
}

#[test]
fn vstack_permuted_for_iter_entry_uses_block_head_target() {
    let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
    pyjit.metadata.n_py_instrs = 19;
    pyjit.metadata.block_head_py_by_jit_pc = vec![(110, 18)];
    pyjit.metadata.py_floor_by_jit_pc = vec![(0, 0), (100, 17), (120, 18)];

    assert_eq!(vstack_containing_py_pc(&pyjit.metadata, 110), 17);
    assert_eq!(vstack_initial_py_pc(&pyjit.metadata, 110, true), 18);
    assert_eq!(vstack_initial_py_pc(&pyjit.metadata, 110, false), 17);
    assert_eq!(vstack_step_py_pc(&pyjit.metadata, 110, 18), 18);
    assert_eq!(vstack_step_py_pc(&pyjit.metadata, 110, 17), 17);
    // Entering this marker from another block must not apply the containing
    // predecessor opcode (17); the first real target op advances to 18.
    assert_eq!(vstack_step_py_pc(&pyjit.metadata, 110, 12), 12);
    assert_eq!(vstack_step_py_pc(&pyjit.metadata, 120, 18), 18);
}

#[test]
fn exact_py_pc_preserves_a_later_disjoint_emission_region() {
    let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
    pyjit.metadata.py_floor_by_jit_pc = vec![(0, 0), (10, 11), (20, 21)];
    pyjit.metadata.py_exact_by_jit_pc = vec![(0, 0), (10, 11), (20, 21), (30, 11)];

    assert_eq!(vstack_containing_py_pc(&pyjit.metadata, 35), 21);
    assert_eq!(
        crate::pyjitcode::exact_py_pc_for_jitcode_pc(&pyjit.metadata.py_exact_by_jit_pc, 35),
        Some(11),
    );
}

#[test]
fn exact_vstack_step_uses_depth_of_its_own_python_opcode() {
    let code = pyre_interpreter::compile_exec("answer = f(1)").unwrap();
    let code = Box::into_raw(Box::new(code));
    let live = crate::liveness::liveness_for(code);
    let depths = live.depth_at_py_pc();
    let (floor_pc, &floor_depth) = depths.iter().enumerate().find(|(_, d)| **d > 0).unwrap();
    let (exact_pc, &exact_depth) = depths.iter().enumerate().find(|(_, d)| **d == 0).unwrap();
    let mut pyjit = crate::PyJitCode::skeleton(code);
    pyjit.metadata.py_floor_by_jit_pc = vec![(0, floor_pc as u32)];
    pyjit.metadata.depth_containing_by_jit_pc = vec![(0, floor_depth)];
    pyjit.metadata.py_exact_by_jit_pc = vec![(0, exact_pc as u32)];
    assert_eq!(
        vstack_step_depth(&pyjit, 0, exact_pc as u32),
        exact_depth as usize
    );
    assert_eq!(
        vstack_step_depth(&pyjit, 0, floor_pc as u32),
        floor_depth as usize
    );
    let frame = ActiveResumeFrame(std::sync::Arc::new(pyjit));
    let (pc, _, depth) = frame
        .vstack_step_coordinate_for_jitcode_pc(0, u32::MAX)
        .unwrap();
    assert_eq!((pc, depth), (exact_pc as u32, exact_depth as usize));
}

/// Build a fresh `TraceCtx`. Uses the public `for_test_types` +
/// `const_ref` / `make_fail_descr` factories so the fixture stays
/// out of `pub(crate)` API.
fn fresh_trace_ctx() -> TraceCtx {
    let _ = test_outer_resume_jitcode_index();
    TraceCtx::for_test_types(&[Type::Ref])
}

/// Recorded non-const Int box. `IntAdd` is arity 2 (`opencoder.py _op_start`).
fn dummy_int_add_box(tc: &mut TraceCtx) -> OpRef {
    let z = tc.const_int(0);
    tc.record_op(majit_ir::OpCode::IntAdd, &[z, z])
}

/// Recorded non-const box used as a Ref register. `PtrEq` is arity 2.
fn dummy_ptr_eq_box(tc: &mut TraceCtx) -> OpRef {
    let null = tc.const_null();
    tc.record_op(majit_ir::OpCode::PtrEq, &[null, null])
}

#[test]
fn parentless_populated_callee_does_not_publish_a_lone_resume_frame() {
    use crate::state::PyreSym;
    use pyre_interpreter::{compile_exec, w_code_get_ptr, w_code_new};

    // Give the callee a real, permanently-live CodeObject and a populated
    // `-live-`-anchored JitCode.  This deliberately crosses the old
    // `code_ptr.is_null()` fixture bail: the assertion is about the publish
    // path, not merely its parent-count predicate.
    let code = compile_exec("None").expect("test code should compile");
    let w_code = w_code_new(Box::into_raw(Box::new(code)) as *const ()) as *const ();
    let raw_code = unsafe {
        w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const pyre_interpreter::CodeObject
    };
    let mut builder = majit_metainterp::JitCodeBuilder::default();
    let live_patch = builder.live_placeholder();
    builder.patch_live_offset(live_patch, 0);
    let mut insns = indexmap::IndexMap::new();
    insns.insert(
        "live/".to_string(),
        majit_metainterp::jitcode::insns::BC_LIVE,
    );
    crate::assembler::publish_state(&insns, &[0, 0, 0], 3, 1);
    let mut pyjit = crate::PyJitCode::skeleton(raw_code);
    pyjit.jitcode = std::sync::Arc::new(builder.finish());
    pyjit.metadata.is_drained = true;
    pyjit.metadata.forward_py_pc_marker_by_jit_pc = vec![(0, 0)];
    let installed = crate::state::install_jitcode_for(w_code, std::sync::Arc::new(pyjit))
        as *const crate::state::JitCode;
    assert!(!installed.is_null());

    let mut tc = fresh_trace_ctx();
    let cond = tc.const_int(1);
    tc.record_guard(majit_ir::OpCode::GuardTrue, &[cond], 0);
    let mut snapshot_sym = PyreSym::new_uninit(OpRef::NONE);
    let mut mode = test_fbw_mode();
    mode.snapshot_sym = &snapshot_sym;
    mode.inline_subwalk = true;
    let session = std::cell::RefCell::new(WalkSession::default());
    session.borrow_mut().framestack.push(InlineFrame {
        is_portal: false,
        w_code: w_code as usize,
        // What `InlineFrameGuard::new` stamps on the first Python portal frame
        // pushed onto an empty framestack: `MetaInterp.newframe` consumes
        // `call_id` 0, no `DEBUG_MERGE_POINT` has been emitted yet, and a
        // forward inline call carries the portal greenkey.
        recursion_greenkey: true,
        call_id: 0,
        debug_merge_point_py_pc: None,
        parents: Vec::new(),
        entry_executed_effects: 0,
        live: None,
    });
    let regs_r = Vec::new();
    let regs_i = Vec::new();
    let regs_f = Vec::new();
    let concrete_r = Vec::new();
    let mut concrete_i = Vec::new();
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: (concrete_r).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: mode,
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::new(regs_f.iter().copied()),

        concrete_registers_i: &mut concrete_i,
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: false,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    assert_eq!(
        super::resume_snapshot::walker_capture_snapshot_for_last_guard(&mut wc, 0),
        Err(DispatchError::GuardResumeCoordinateUnavailable { pc: 0 }),
        "a callee-only image would make frames[0] disagree between the two resume decoders",
    );
    // Keep the pointer live through the call; PyCode itself is intentionally
    // permanent, matching the production `w_code_new` ownership contract.
    std::hint::black_box(&mut snapshot_sym);
}

/// The `ec` recovery every unseeded portal red falls back on
/// (`seed_execution_context_for_walk`, `MIFrame::ensure_execution_context`,
/// `setup_reconstructed_callee_frame`) must survive as a call in the trace.
/// It takes no arguments, so an elidable descr would make the pure pass fold
/// it to the recording thread's ExecutionContext, and every later thread
/// entering the compiled trace would read that thread's exception state.
#[test]
fn execution_context_recovery_records_a_non_elidable_call() {
    let mut tc = fresh_trace_ctx();
    let ops_before = tc.num_ops();
    let ec = crate::helpers::emit_current_execution_context(&mut tc, "ExecutionContext::Test");

    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "the recovery must record exactly the call",
    );
    let call_op = tc.ops().last().expect("recorded op");
    assert_eq!(call_op.opcode, majit_ir::OpCode::CallR);
    assert_eq!(
        call_op.getarglist().len(),
        1,
        "arglist is the funcptr alone: `getexecutioncontext()` takes no arguments",
    );
    assert_eq!(
        ec.inline_const_to_value(),
        None,
        "a folded constant would bake the recording thread's context",
    );
    let descr = call_op.getdescr().expect("calldescr");
    let call_descr = descr.as_call_descr().expect("call descr");
    assert!(!call_descr.get_extra_info().check_is_elidable());
    assert!(!call_descr.get_extra_info().check_can_raise(false));
}

/// `GcRef::NO_CONCRETE` is "no runtime value". `opref_concrete` must answer
/// `Null` for that sentinel, the same as `concrete_from_recorded_opref`.
#[test]
fn opref_concrete_keeps_a_no_concrete_ref_null() {
    use crate::state::ConcreteValue;

    assert_eq!(
        super::opref_concrete(Some(majit_ir::Value::Ref(majit_ir::GcRef::NO_CONCRETE))),
        ConcreteValue::Null,
    );
    let live = 0x1110usize;
    assert_eq!(
        super::opref_concrete(Some(majit_ir::Value::Ref(majit_ir::GcRef(live)))),
        ConcreteValue::Ref(live as pyre_object::PyObjectRef),
    );
    assert_eq!(super::opref_concrete(None), ConcreteValue::Null);
}

/// `do_residual_call` records the portal's `ec` red (`interp_jit.py`
/// `PyPyJitDriver.reds`). The `CALL_MAY_FORCE_R` operand is that box, not a
/// `ConstPtr` of the tracing thread's `getexecutioncontext()`.
#[test]
fn portal_may_force_records_the_callee_execution_context() {
    use pyre_interpreter::compile_exec;

    extern "C" fn portal_runner_stub() -> i64 {
        0
    }

    let pair = crate::state::ensure_trace_test_driver();
    {
        let meta = pair.0.meta_interp_mut();
        let wired = meta.staticdata.jitdrivers_sd.iter().any(|jd| {
            jd.num_greens() > 0 && jd.portal_runner_adr != 0 && jd.portal_calldescr.is_some()
        });
        if !wired {
            let mut jd = majit_metainterp::JitDriverStaticData::new(
                vec![("pycode", Type::Ref)],
                vec![("frame", Type::Ref), ("ec", Type::Ref)],
            );
            jd.portal_runner_adr = portal_runner_stub as *const () as i64;
            meta.register_jitdriver_sd(jd);
            meta.finish_setup_descrs_for_jitdrivers();
        }
    }

    let raw_code = compile_exec("None").expect("test code should compile");
    let mut frame = pyre_interpreter::pyframe::PyFrame::new(raw_code);
    let frame_ptr = (&mut *frame) as *mut pyre_interpreter::pyframe::PyFrame as usize;

    let jitcode_index = test_outer_resume_jitcode_index();
    let mut tc = TraceCtx::for_test_types(&[Type::Ref, Type::Ref]);
    let callee_frame = OpRef::input_arg_ref(0);
    let callee_ec = OpRef::input_arg_ref(1);
    let sentinel = 0xECu64;
    tc.set_opref_concrete(
        callee_frame,
        majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr)),
    );
    tc.set_opref_concrete(
        callee_ec,
        majit_ir::Value::Ref(majit_ir::GcRef(sentinel as usize)),
    );

    let session = std::cell::RefCell::new(WalkSession::default());
    let regs_r = Vec::new();
    let regs_i = Vec::new();
    let regs_f = Vec::new();
    let mut concrete_i = Vec::new();
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData::default()),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::new(regs_f.iter().copied()),
        concrete_registers_i: &mut concrete_i,
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: false,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: Some(0),
        outer_jitcode_index: jitcode_index,
        pending_guard_snapshot_error: None,
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_reorder_ceiling: u32::MAX,
        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    let recorded = super::inline_call::record_walker_loop_callee_portal_call(
        &mut wc,
        0,
        None,
        callee_frame,
        callee_ec,
        None,
        0,
        std::ptr::null(),
        false,
    )
    .expect("portal may-force residual")
    .expect("portal runner is wired");

    let call = wc
        .trace_ctx
        .ops()
        .iter()
        .find(|op| op.opcode == majit_ir::OpCode::CallMayForceR)
        .expect("CALL_MAY_FORCE_R");
    let args = call.getarglist();
    let recorded_ec = args.last().expect("ec red").to_opref();
    assert_eq!(
        recorded_ec, callee_ec,
        "the may-force residual must record the callee ec red"
    );
    assert!(
        !recorded_ec.is_constant(),
        "a ConstPtr would bake the tracing thread's execution context"
    );
    assert_eq!(
        wc.trace_ctx.concrete_of_opref(recorded_ec),
        Some(majit_ir::Value::Ref(majit_ir::GcRef(sentinel as usize))),
        "the callee ec's concrete stays the value the sub-walk stored"
    );
    let _ = (recorded, frame);
}

/// The globals guard reads a frame's namespace override with a plain
/// `GETFIELD_GC_R` on the live `debugdata` box, so the descr it uses has to
/// name `FrameDebugData.w_globals` and has to stay mutable: `pyframe.py
/// set_w_globals` rebinds the field after the object exists, and an immutable
/// descr would let the read fold back to the recording frame's namespace —
/// which is the pin the guard exists to break.
#[test]
fn frame_debug_data_w_globals_descr_reads_the_mutable_override_field() {
    let descr = crate::descr::frame_debug_data_w_globals_descr();
    let field = descr
        .as_field_descr()
        .expect("w_globals resolves to a FieldDescr");
    assert_eq!(
        field.offset(),
        pyre_interpreter::pyframe::FRAME_DEBUG_DATA_W_GLOBALS_OFFSET,
    );
    assert!(
        !field.is_immutable(),
        "an immutable descr would fold the read back to the recorded namespace",
    );
}

/// `FRAME_DEBUG_DATA_DESCR_GROUP`'s accessors name their field by POSITION,
/// so a field added ahead of one of them re-points it at its neighbour and
/// nothing fails to compile. Assert the offset each accessor resolves to
/// against the layout constant, which is the only channel that disagrees.
#[test]
fn frame_debug_data_accessors_resolve_to_their_own_offsets() {
    for (name, descr, offset) in [
        (
            "w_globals",
            crate::descr::frame_debug_data_w_globals_descr(),
            pyre_interpreter::pyframe::FRAME_DEBUG_DATA_W_GLOBALS_OFFSET,
        ),
        (
            "w_locals",
            crate::descr::frame_debug_data_w_locals_descr(),
            pyre_interpreter::pyframe::FRAME_DEBUG_DATA_W_LOCALS_OFFSET,
        ),
        (
            "w_extra_locals",
            crate::descr::frame_debug_data_w_extra_locals_descr(),
            pyre_interpreter::pyframe::FRAME_DEBUG_DATA_W_EXTRA_LOCALS_OFFSET,
        ),
    ] {
        let field = descr
            .as_field_descr()
            .unwrap_or_else(|| panic!("{name} resolves to a FieldDescr"));
        assert_eq!(field.offset(), offset, "{name} reads the wrong group index");
    }
}

#[test]
fn builtin_wrapper_args_slice_names_the_object_gcarray_descr() {
    // `cpu.arraydescrof(ARRAY)` is one descr per ARRAY: the wrapper's length
    // and item reads of its `&[PyObjectRef]` argument name the descr the
    // walker builds that array with, so its heap-cache seeds are hits.
    let wrapper = named_jitcode("__majit_wrap_random").expect("random builtin wrapper jitcode");
    let expected = crate::state::pyobject_gcarray_descr().index();
    let first = crate::jitcode_runtime::decoded_ops(&wrapper.code)
        .next()
        .expect("wrapper first op");
    assert_eq!(first.key, "arraylen_gc/rd>i");
    assert_eq!(item_pool_descr_index(&wrapper.code, first.pc + 2), expected);

    let getitem = crate::jitcode_runtime::decoded_ops(&wrapper.code)
        .find(|op| {
            op.key == "getarrayitem_gc_r/rid>r" && wrapper.code.get(op.pc + 1).copied() == Some(0)
        })
        .expect("wrapper getarrayitem(r0)");
    assert_eq!(
        item_pool_descr_index(&wrapper.code, getitem.pc + 3),
        expected
    );
}

#[test]
fn gateway_positional_count_uses_rpython_signed() {
    let count =
        named_jitcode("leading_non_null_count").expect("gateway positional-count helper jitcode");
    assert_eq!(count.calldescr.result_type, 'i');
    assert!(
        count.calldescr.result_signed,
        "PyPy argument.py keeps num_args/upfront/input_argcount as RPython Signed"
    );
}

#[test]
fn signature_bound_wrapper_reads_argument_slice_with_distinct_item_descr() {
    let wrapper =
        named_jitcode("__majit_wrap_getrandbits").expect("getrandbits builtin wrapper jitcode");
    // `getrandbits` registers with a `Signature`, so the call path resolves
    // keywords into positional PY_NULL-padded slots before the wrapper runs.
    // The argument preamble counts the leading non-null slots of the wrapper's
    // own argument array; it does not peel a `split_builtin_kwargs(args)`
    // result and read that instead.
    //
    // Asked of the array the count reads rather than of the count's position:
    // `descr_check` now precedes the preamble, so the count is no longer the
    // wrapper's first op, and its position tracks only how far ahead of it the
    // receiver tests sit.  The array is the property that separates the two
    // readings, and a single `R` operand carries it — the byte after the
    // one-entry length byte that follows the two descr bytes.
    let count = crate::jitcode_runtime::decoded_ops(&wrapper.code)
        .find(|op| inline_call_callee_name(&wrapper.code, op) == Some("leading_non_null_count"))
        .expect("signature-bound wrapper counts its leading non-null argument slots");
    assert_eq!(
        wrapper.code[count.pc + 3],
        1,
        "the count takes one argument"
    );
    let args_reg = crate::jitcode_runtime::decoded_ops(&wrapper.code)
        .find(|op| op.key == "arraylen_gc/rd>i")
        .map(|op| wrapper.code[op.pc + 1])
        .expect("wrapper reads an array length");
    assert_eq!(
        wrapper.code[count.pc + 4],
        args_reg,
        "the count reads the wrapper's own argument array"
    );

    // `descr_check` runs ahead of the count, and its refusal leaves the
    // wrapper through `receiver_mismatch` rather than being worded here: the
    // helper is opaque, so the test it guards reaches the jitcode as a
    // residual call standing before the preamble.  A wrapper that bound its
    // receiver after the count would report an argument count for a call with
    // no receiver to apply the arguments to.  (Which helper the residual
    // enters is asserted on the lowered graph, where the call still carries
    // its path: `majit-translate`'s `gateway_wrapper_refusals_all_residualize`.)
    assert!(
        crate::jitcode_runtime::decoded_ops(&wrapper.code)
            .any(|op| op.key.starts_with("residual_call") && op.pc < count.pc),
        "the receiver tests run before the argument preamble"
    );

    let args_descr_index = crate::state::pyobject_gcarray_descr().index();
    let getitem = crate::jitcode_runtime::decoded_ops(&wrapper.code)
        .find(|op| {
            op.key == "getarrayitem_gc_r/rid>r"
                && item_pool_descr_index(&wrapper.code, op.pc + 3) == args_descr_index
        })
        .expect("wrapper argument-slice item read");
    let slice_reg = wrapper.code[getitem.pc + 1];
    // The slice's colour is not pinned either, for the same reason the entry
    // call's form is not: it is an allocation outcome, not a property of the
    // wrapper.  Whichever register holds the slice, the bound check governing
    // an extraction is the nearest `arraylen_gc` on that register *before* the
    // read, so select it that way.
    //
    // Taking the nearest preceding one is what keeps this honest when the
    // splitter inlines far enough to read `args.len()` off the wrapper input
    // before the split: that read can share the slice's register once the
    // wrapper input is dead, and the first `arraylen_gc` in the code would
    // then name it rather than the slice's.
    crate::jitcode_runtime::decoded_ops(&wrapper.code)
        .filter(|op| {
            op.key == "arraylen_gc/rd>i"
                && op.pc < getitem.pc
                && wrapper.code[op.pc + 1] == slice_reg
        })
        .last()
        .expect("wrapper reads the argument-slice length before extracting from it");
}

/// Descr-pool index encoded little-endian at `at`, resolved to its
/// `all_descrs` index.
fn item_pool_descr_index(code: &[u8], at: usize) -> u32 {
    let pool_index = code[at] as usize | ((code[at + 1] as usize) << 8);
    crate::jitcode_runtime::all_descr_refs()[pool_index].index()
}

/// Name of the jitcode an `inline_call_*` enters, `None` for any other op.
/// Every `inline_call_*` key opens with its `d` operand, so the descr-pool
/// index is the two bytes after the opcode.
fn inline_call_callee_name(
    code: &[u8],
    op: &crate::jitcode_runtime::DecodedOp,
) -> Option<&'static str> {
    if !op.opname.starts_with("inline_call_") {
        return None;
    }
    let pool_index = code[op.pc + 1] as usize | ((code[op.pc + 2] as usize) << 8);
    match crate::jitcode_runtime::all_descrs().get(pool_index)? {
        majit_jitcode::codewriter::jitcode::BhDescr::JitCode { jitcode_index, .. } => {
            crate::jitcode_runtime::all_jitcodes()
                .get(*jitcode_index)
                .map(|jitcode| jitcode.name.as_str())
        }
        _ => None,
    }
}

#[test]
fn random_core_residuals_use_registered_genrand32_address() {
    // `_random` is a `pyre-module` module, so its residual address reaches
    // `jit_trace_fnaddrs` through the optional-module hooks.
    pyre_module::register();
    let bindings = pyre_interpreter::jit_trace_fnaddrs();
    let expected = bindings
        .iter()
        .find_map(|&(path, address)| {
            (path == "module::_random::Random::genrand32").then_some(address)
        })
        .expect("genrand32 runtime fnaddr");
    let cast_uint_to_float = bindings
        .iter()
        .find_map(|&(path, address)| {
            (path == "majit_metainterp::cast_uint_to_float").then_some(address)
        })
        .expect("cast_uint_to_float runtime fnaddr");
    let random = crate::jitcode_runtime::all_jitcodes()
        .iter()
        .find(|jitcode| {
            jitcode.name == "random"
                && crate::jitcode_runtime::decoded_ops(&jitcode.code)
                    .filter(|op| op.key == "residual_call_r_i/iRd>i")
                    .count()
                    == 2
        })
        .expect("rrandom Random::random jitcode");

    assert_eq!(
        random
            .constants_i
            .iter()
            .filter(|&&address| address == expected)
            .count(),
        1,
        "the two genrand32 calls share one runtime-patched constant-pool address"
    );
    assert_eq!(
        random
            .constants_i
            .iter()
            .filter(|&&address| address == cast_uint_to_float)
            .count(),
        1,
        "the two unsigned-to-float conversions share the registered support-helper address"
    );
}

#[test]
fn user_binop_forward_dunder_covers_fraction_arithmetic_without_inplace_shortcuts() {
    use pyre_interpreter::bytecode::BinaryOperator;

    assert_eq!(
        user_binop_forward_dunder(BinaryOperator::Subtract),
        Some("__sub__")
    );
    assert_eq!(
        user_binop_forward_dunder(BinaryOperator::TrueDivide),
        Some("__truediv__")
    );
    assert_eq!(
        user_binop_forward_dunder(BinaryOperator::InplaceSubtract),
        None
    );
    assert_eq!(
        user_binop_forward_dunder(BinaryOperator::InplaceTrueDivide),
        None
    );
}

#[test]
fn user_binop_reflected_dunder_covers_fraction_arithmetic_without_inplace_shortcuts() {
    use pyre_interpreter::bytecode::BinaryOperator;

    assert_eq!(
        user_binop_reflected_dunder(BinaryOperator::Add),
        Some("__radd__")
    );
    assert_eq!(
        user_binop_reflected_dunder(BinaryOperator::Subtract),
        Some("__rsub__")
    );
    assert_eq!(
        user_binop_reflected_dunder(BinaryOperator::TrueDivide),
        Some("__rtruediv__")
    );
    assert_eq!(
        user_binop_reflected_dunder(BinaryOperator::InplaceSubtract),
        None
    );
    assert_eq!(
        user_binop_reflected_dunder(BinaryOperator::InplaceTrueDivide),
        None
    );
    assert_eq!(user_binop_reflected_dunder(BinaryOperator::Subscr), None);
}

#[test]
fn rewind_dunder_admits_only_terminal_nested_raise_shape() {
    use crate::pyjitcode::InlineBodyFacts;

    let facts = |contains_raise, straight_line, nested_call, has_exception_table| InlineBodyFacts {
        contains_raise,
        has_abort_permanent: false,
        exc_override_straight_line: straight_line,
        exc_override_sample_safe: false,
        index_sample_safe: false,
        exc_override_has_nested_call: nested_call,
        owns_loop_header: false,
        has_exception_table,
    };

    assert!(dunder_body_facts_admissible_on_rewind(facts(
        false, true, false, false,
    )));
    assert!(dunder_body_facts_admissible_on_rewind(facts(
        true, true, true, false,
    )));
    assert!(!dunder_body_facts_admissible_on_rewind(facts(
        false, true, true, false,
    )));
    assert!(!dunder_body_facts_admissible_on_rewind(facts(
        true, false, true, false,
    )));
    assert!(!dunder_body_facts_admissible_on_rewind(facts(
        true, true, true, true,
    )));
}

/// Build a `done_with_this_frame_descr_ref` for tests. Mirrors the
/// production fallback at `pyjitpl.rs` (`make_fail_descr_typed`)
/// when the staticdata singleton was never attached.
fn done_descr_ref_for_tests() -> DescrRef {
    make_fail_descr(1)
}

#[test]
fn inline_caller_frame_declines_only_a_returning_handler() {
    let insns = crate::jitcode_runtime::insns_opname_to_byte();
    let catch_exception = insns["catch_exception/L"];
    let void_return = insns["void_return/"];
    let reraise = insns["reraise/"];
    let handler_returns = [catch_exception, 3, 0, void_return];
    assert_eq!(
        decline_inline_caller_frame_for_catch_marker(Some(0), &handler_returns, true),
        Err(InlineCallerFrameDecline::TryBlockCatchMarker),
    );
    let handler_reraises = [catch_exception, 3, 0, reraise];
    assert_eq!(
        decline_inline_caller_frame_for_catch_marker(Some(0), &handler_reraises, true),
        Ok(()),
    );
    assert_eq!(
        decline_inline_caller_frame_for_catch_marker(Some(0), &handler_returns, false),
        Ok(()),
    );
}

/// `ensure_residual_call_args_bound` backs the unbound-arg abort path
/// for all three residual-call shapes (iRd / iIRd / iIRFd); they all
/// funnel through this helper, so one direct test covers the guard
/// that otherwise lets `OpRef::NONE` reach the backend's
/// `resolve_opref` (a process abort).
#[test]
fn ensure_residual_call_args_bound_rejects_unbound_arg() {
    // Fully-bound funcbox + args slice passes.
    let bound = [OpRef::int_op(1), OpRef::int_op(2), OpRef::int_op(3)];
    assert!(ensure_residual_call_args_bound(&bound, 7).is_ok());

    // An unbound arg surfaces its position.
    let unbound = [OpRef::int_op(1), OpRef::NONE, OpRef::int_op(3)];
    assert!(matches!(
        ensure_residual_call_args_bound(&unbound, 7),
        Err(DispatchError::ResidualCallArgUnbound {
            pc: 7,
            arg_index: 1
        })
    ));

    // The funcbox slot (index 0) is guarded too.
    let unbound_func = [OpRef::NONE, OpRef::int_op(2)];
    assert!(matches!(
        ensure_residual_call_args_bound(&unbound_func, 4),
        Err(DispatchError::ResidualCallArgUnbound {
            pc: 4,
            arg_index: 0
        })
    ));
}

/// Build distinct `OpRef` constants for register slots so dataflow
/// assertions don't get false positives from shared identity. Each
/// slot holds `const_ref(0xC0DE_0000 + i)` for `i in 0..count`.
fn distinct_const_refs(ctx: &mut TraceCtx, count: usize) -> Vec<OpRef> {
    (0..count)
        .map(|i| ctx.const_ref(0xC0DE_0000_i64 + i as i64))
        .collect()
}

/// Companion of [`distinct_const_refs`] that mints Int-typed
/// ConstInt OpRefs.  Use this when a fixture needs to populate
/// integer-register slots — the heapcache array path keys on the
/// `ConstInt.getint()` value, so Ref-typed mints don't satisfy
/// `getarrayitem_cache`'s ConstInt precondition.
fn distinct_const_ints(ctx: &mut TraceCtx, count: usize) -> Vec<OpRef> {
    (0..count)
        .map(|i| ctx.const_int(1_000 + i as i64))
        .collect()
}

/// Default `sub_jitcode_lookup` for tests that don't exercise
/// `inline_call_r_r` recursion. Returns `None` for every index;
/// any test that hits the inline_call handler with this lookup
/// will see `DispatchError::SubJitCodeNotFound`.
fn no_sub_jitcodes(_idx: usize) -> Option<SubJitCodeBody> {
    None
}

fn switch_descr_pool(entries: &[(i64, usize)]) -> Vec<DescrRef> {
    let dict = entries.iter().copied().collect();
    vec![std::sync::Arc::new(crate::descr::PyreSwitchDescr::new(dict)) as DescrRef]
}

#[test]
fn trace_too_long_snapshot_waits_for_frame_transition_outcomes() {
    assert!(trace_too_long_abort_safe(
        &DispatchOutcome::Continue,
        false,
        0,
    ));
    assert!(!trace_too_long_abort_safe(
        &DispatchOutcome::SubReturn { result: None },
        false,
        0,
    ));
    assert!(!trace_too_long_abort_safe(
        &DispatchOutcome::SubRaise {
            exc: OpRef::input_arg_ref(0),
            exc_concrete: ConcreteValue::Ref(0xCAFEusize as _),
        },
        false,
        0,
    ));
    assert!(!trace_too_long_abort_safe(
        &DispatchOutcome::Continue,
        false,
        1,
    ));
    assert!(trace_too_long_abort_safe(
        &DispatchOutcome::Continue,
        true,
        1,
    ));
}

/// Concrete-shadow round-trip: a `WalkContext` built with
/// `concrete_registers_r` exposes each slot's `ConcreteValue` via
/// `read_ref_reg_concrete` indexed by the same byte the symbolic
/// `read_ref_reg` consults.  Verifies the parallel-slice plumbing
/// (slot N's OpRef in `registers_r` shares slot N in
/// `concrete_registers_r`).
#[test]
fn read_ref_reg_concrete_returns_slot_matching_symbolic_read() {
    let exc_obj_ptr: pyre_object::PyObjectRef = 0xDEAD_BEEFusize as _;
    let descr_pool: Vec<DescrRef> = Vec::new();
    let mut tc = TraceCtx::for_test_types(&[Type::Ref, Type::Ref, Type::Ref]);
    let mut oprefs = distinct_const_refs(&mut tc, 3);
    // A non-constant input box has no intrinsic `TraceCtx` concrete payload;
    // only the parallel MIFrame-style register shadow knows its value.
    oprefs[1] = OpRef::input_arg_ref(1);
    // Model a collection that forwarded the recorder Box while the raw
    // concrete-register shadow retained its old address.
    oprefs[2] = OpRef::input_arg_ref(2);
    let forwarded_obj_ptr = 0xF0A0_0000usize;
    let stale_obj_ptr = 0xDEAD_0000usize;
    tc.set_opref_concrete(oprefs[2], Value::Ref(majit_ir::GcRef(forwarded_obj_ptr)));
    let regs_r = oprefs.clone();
    let concrete = vec![
        ConcreteValue::Null,
        ConcreteValue::Ref(exc_obj_ptr),
        ConcreteValue::Ref(stale_obj_ptr as pyre_object::PyObjectRef),
    ];
    // Snapshot expected values before `&mut concrete` enters wc —
    // the assertion below cannot read `concrete[reg_idx]` while wc
    // holds the mutable borrow.
    let expected: Vec<ConcreteValue> = concrete.clone();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: (concrete).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    // Synthesize a 2-byte op fixture: `<opcode_byte> <reg_idx>`.
    // `read_ref_reg_concrete` reads `code[op.pc + 1 + operand_offset]`
    // exactly like `read_ref_reg`, so encoding the reg byte at pc+1
    // suffices.
    for reg_idx in 0..3 {
        let code = [0u8, reg_idx as u8];
        let op = DecodedOp {
            key: "fixture/r",
            opname: "fixture",
            argcodes: "r",
            pc: 0,
            next_pc: 2,
        };
        assert_eq!(
            read_ref_reg_concrete(&code, &op, 0, &wc),
            expected[reg_idx],
            "reg {} concrete shadow must match the parallel slot",
            reg_idx,
        );
    }
    let code = [0u8, 1u8];
    let op = DecodedOp {
        key: "fixture/r",
        opname: "fixture",
        argcodes: "r",
        pc: 0,
        next_pc: 2,
    };
    assert_eq!(
        wc.trace_ctx
            .concrete_of_opref(wc.registers_r.get(1).expect("ref register in range")),
        None
    );
    assert_eq!(
        super::vable_ops::vable_value_concrete(
            &code,
            &op,
            0,
            &wc,
            'r',
            wc.registers_r.get(1).expect("ref register in range")
        ),
        Some(Value::Ref(majit_ir::GcRef(exc_obj_ptr as usize))),
        "vable writes must preserve the concrete half of a non-constant register Box",
    );
    assert_eq!(
        super::vable_ops::vable_effective_value_concrete(
            &code,
            &op,
            0,
            &wc,
            'r',
            wc.registers_r.get(1).expect("ref register in range"),
            wc.registers_r.get(0).expect("ref register in range"),
        ),
        Some(Value::Ref(majit_ir::GcRef(0xC0DE_0000))),
        "a TOS override must resolve its own box instead of the stale encoded register shadow",
    );
    let forwarded_code = [0u8, 2u8];
    assert_eq!(
        super::vable_ops::vable_value_concrete(
            &forwarded_code,
            &op,
            0,
            &wc,
            'r',
            wc.registers_r.get(2).expect("ref register in range"),
        ),
        Some(Value::Ref(majit_ir::GcRef(forwarded_obj_ptr))),
        "a forwarded recorder Box must win over the stale raw Ref shadow",
    );

    let runtime_jc = majit_metainterp::jitcode::JitCode::new("trace_too_long_arbitrary_pc");
    runtime_jc.set_body(majit_jitcode::jitcode::JitCodeBody {
        code: vec![0, 0],
        c_num_regs_r: 3,
        // Two single-byte ops, so pc 1 is a post-step position that starts an
        // instruction and carries no `-live-` marker.
        startpoints: Some([0_usize, 1].into_iter().collect()),
        ..Default::default()
    });
    let miframe = super::residual_call::build_trace_too_long_single_frame_miframe(
        &wc,
        std::sync::Arc::new(runtime_jc),
        1,
    )
    .expect("an arbitrary post-step pc must not require a -live- marker");
    assert_eq!(miframe.pc, 1);
    assert_eq!(
        miframe.ref_value_for_blackhole(1),
        Some(exc_obj_ptr as i64),
        "the complete-bank image must retain non-constant register concrete values",
    );
    assert_eq!(
        miframe.ref_value_for_blackhole(2),
        Some(forwarded_obj_ptr as i64),
        "the complete-bank image must prefer the GC-forwarded Box payload",
    );

    // A pc the jitcode does not name as an instruction start is refused: the
    // blackhole `setposition`s onto it and dispatches, so it would read an
    // operand byte as an opcode.
    let mid_instruction_jc = majit_metainterp::jitcode::JitCode::new("trace_too_long_mid_insn");
    mid_instruction_jc.set_body(majit_jitcode::jitcode::JitCodeBody {
        code: vec![0, 0],
        c_num_regs_r: 3,
        startpoints: Some([0_usize].into_iter().collect()),
        ..Default::default()
    });
    assert!(
        super::residual_call::build_trace_too_long_single_frame_miframe(
            &wc,
            std::sync::Arc::new(mid_instruction_jc),
            1,
        )
        .is_none(),
        "a resume pc inside an instruction must decline",
    );
}

#[test]
fn vable_store_tracks_live_null_without_changing_the_recorded_trace() {
    let mut tc = TraceCtx::for_test_types(&[Type::Ref]);
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    let array_len = 3;
    let slot_count = info.num_static_extra_boxes + array_len;
    let null = Value::Ref(majit_ir::GcRef::NULL);
    let vable = tc.const_ref(1);
    let initial_boxes = vec![tc.const_null(); slot_count];
    let initial_values = vec![null; slot_count];
    tc.install_virtualizable_info(info.clone());
    tc.init_virtualizable_boxes(
        &info,
        vable,
        Value::Ref(majit_ir::GcRef(1)),
        &initial_boxes,
        &initial_values,
        &[array_len],
    );
    let fdescr = info.array_pointer_field_descr(0);
    let adescr = info.array_descrs[0].clone();
    let flat_base = info.num_static_extra_boxes;
    assert!((0..=slot_count).all(|i| !tc.virtualizable_slot_stored_live_null(i)));

    let index0 = tc.const_int(0);
    let const_null = tc.const_null();
    let ops_before = tc.num_ops();
    let nonstandard = match tc.begin_nonstandard_virtualizable(0, vable, &fdescr) {
        NonstandardVable::Decided(n) => n,
        NonstandardVable::PendingEq { .. } => {
            panic!("standard identity must take Step 3 Decided(false)")
        }
    };
    assert!(matches!(
        tc.vable_setarrayitem_checked(
            nonstandard,
            0,
            vable,
            index0,
            0,
            fdescr.clone(),
            adescr.clone(),
            const_null,
            null,
            true,
        ),
        VableArrayStore::Stored(Some(_))
    ));
    assert!(tc.virtualizable_slot_stored_live_null(flat_base));
    assert!(
        (0..=slot_count)
            .filter(|&i| i != flat_base)
            .all(|i| !tc.virtualizable_slot_stored_live_null(i))
    );
    assert_eq!(
        tc.virtualizable_entry_at(flat_base).map(|entry| entry.0),
        Some(const_null)
    );
    assert!(const_null.is_constant());
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "the side-table marker records no op"
    );

    let nonstandard = match tc.begin_nonstandard_virtualizable(0, vable, &fdescr) {
        NonstandardVable::Decided(n) => n,
        NonstandardVable::PendingEq { .. } => {
            panic!("standard identity must take Step 3 Decided(false)")
        }
    };
    assert!(matches!(
        tc.vable_setarrayitem_checked(
            nonstandard,
            0,
            vable,
            index0,
            0,
            fdescr.clone(),
            adescr.clone(),
            const_null,
            null,
            false,
        ),
        VableArrayStore::Stored(Some(_))
    ));
    assert!(!tc.virtualizable_slot_stored_live_null(flat_base));

    let index1 = tc.const_int(1);
    let non_null = tc.const_ref(2);
    let nonstandard = match tc.begin_nonstandard_virtualizable(0, vable, &fdescr) {
        NonstandardVable::Decided(n) => n,
        NonstandardVable::PendingEq { .. } => {
            panic!("standard identity must take Step 3 Decided(false)")
        }
    };
    assert!(matches!(
        tc.vable_setarrayitem_checked(
            nonstandard,
            0,
            vable,
            index1,
            1,
            fdescr.clone(),
            adescr.clone(),
            non_null,
            Value::Ref(majit_ir::GcRef(2)),
            true,
        ),
        VableArrayStore::Stored(Some(_))
    ));
    assert!(!tc.virtualizable_slot_stored_live_null(flat_base + 1));

    let nonstandard = match tc.begin_nonstandard_virtualizable(0, vable, &fdescr) {
        NonstandardVable::Decided(n) => n,
        NonstandardVable::PendingEq { .. } => {
            panic!("standard identity must take Step 3 Decided(false)")
        }
    };
    assert!(matches!(
        tc.vable_setarrayitem_checked(
            nonstandard,
            0,
            vable,
            index0,
            0,
            fdescr,
            adescr,
            const_null,
            null,
            true,
        ),
        VableArrayStore::Stored(Some(_))
    ));
    assert!(tc.virtualizable_slot_stored_live_null(flat_base));
    tc.set_virtualizable_entry_at(flat_base, const_null, null);
    assert!(!tc.virtualizable_slot_stored_live_null(flat_base));
}

/// `getfield_vable_*` must abort to `VableBoxNotSeeded` when the
/// box register is unseeded (`OpRef::NONE`) rather than feed
/// `u32::MAX` into the heapcache flag vector (a 16 GiB resize).
#[test]
fn getfield_vable_with_none_obj_surfaces_vable_box_not_seeded() {
    let descr_pool: Vec<DescrRef> = Vec::new();
    let mut tc = fresh_trace_ctx();
    let regs_r = vec![OpRef::NONE];
    let regs_i = vec![OpRef::NONE];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    // `getfield_vable_i/rd>i`: operand 0 (the box) sits at code[pc+1].
    let code = [0u8, 0x00, 0x00, 0x00, 0x00];
    let op = DecodedOp {
        key: "getfield_vable_i/rd>i",
        opname: "getfield_vable_i",
        argcodes: "rd>i",
        pc: 0,
        next_pc: 5,
    };
    assert_eq!(
        getfield_vable_via_metainterp(&code, &op, &mut wc, 'i'),
        Err(DispatchError::VableBoxNotSeeded { pc: 0 })
    );
}

/// `setfield_vable_*` carries the same unseeded-box guard.
#[test]
fn setfield_vable_with_none_obj_surfaces_vable_box_not_seeded() {
    let descr_pool: Vec<DescrRef> = Vec::new();
    let mut tc = fresh_trace_ctx();
    let regs_r = vec![OpRef::NONE];
    let regs_i = vec![OpRef::NONE];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    // `setfield_vable_i/rid`: operand 0 (the box) sits at code[pc+1].
    let code = [0u8, 0x00, 0x00, 0x00, 0x00];
    let op = DecodedOp {
        key: "setfield_vable_i/rid",
        opname: "setfield_vable_i",
        argcodes: "rid",
        pc: 0,
        next_pc: 5,
    };
    assert_eq!(
        setfield_vable_via_metainterp(&code, &op, &mut wc, 'i'),
        Err(DispatchError::VableBoxNotSeeded { pc: 0 })
    );
}

/// `getarrayitem_vable_*` / `setarrayitem_vable_*` / `arraylen_vable`
/// carry the same unseeded-box guard as the scalar field handlers:
/// an `OpRef::NONE` vable must abort to `VableBoxNotSeeded` rather than
/// resize the heapcache flag vector to 16 GiB.
#[test]
fn array_vable_handlers_with_none_obj_surface_vable_box_not_seeded() {
    // operand 0 (the box) sits at code[pc+1] for all three argcodes.
    let code = [0u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    for (key, opname, argcodes) in [
        (
            "getarrayitem_vable_i/riXdd>i",
            "getarrayitem_vable_i",
            "riXdd>i",
        ),
        (
            "setarrayitem_vable_i/riXdd",
            "setarrayitem_vable_i",
            "riXdd",
        ),
        ("arraylen_vable/rdd>i", "arraylen_vable", "rdd>i"),
    ] {
        let descr_pool: Vec<DescrRef> = Vec::new();
        let mut tc = fresh_trace_ctx();
        let regs_r = vec![OpRef::NONE];
        let regs_i = vec![OpRef::NONE];
        let session = std::cell::RefCell::new(WalkSession::default());
        let mut wc = WalkContext {
            frame_state: WalkFrameState::new(WalkFrameStateData {
                callee_shadow: None,
                concrete_registers_r: ([]).to_vec(),
                outer_active_boxes: Vec::new(),
                vstack_boxes: Vec::new(),
                vstack_last_ref: OpRef::NONE,
                vstack_reorder_saved: None,
                ..Default::default()
            }),
            inline_callee_consts: None,
            fbw_mode: test_fbw_mode(),
            session: &session,
            registers_r: &RegisterBank::new(regs_r.iter().copied()),
            registers_i: &RegisterBank::new(regs_i.iter().copied()),
            registers_f: &RegisterBank::default(),

            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            entry_py_pc: EntryPyPc::Py(0),
            outer_resume_marker_jit_pc: None,
            outer_jitcode_index: 0,
            raw_descrs: RawDescrPool::Global,
            is_authoritative_executor: false,

            pending_guard_snapshot_error: None,

            vstack_depth: 0,
            vstack_cur_pypc: 0,
            vstack_valid: false,

            vstack_reorder_ceiling: u32::MAX,

            vstack_handler_landing_py: None,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
        };
        let op = DecodedOp {
            key,
            opname,
            argcodes,
            pc: 0,
            next_pc: code.len(),
        };
        let result = match opname {
            "getarrayitem_vable_i" => getarrayitem_vable_via_metainterp(&code, &op, &mut wc, 'i'),
            "setarrayitem_vable_i" => setarrayitem_vable_via_metainterp(&code, &op, &mut wc, 'i'),
            "arraylen_vable" => arraylen_vable_via_metainterp(&code, &op, &mut wc),
            _ => unreachable!(),
        };
        assert_eq!(
            result,
            Err(DispatchError::VableBoxNotSeeded { pc: 0 }),
            "{opname} must abort VableBoxNotSeeded on an unseeded vable register",
        );
    }
}

/// A seeded vable whose index register carries no concrete value must abort
/// to `VableArrayIndexNotConcrete`.
///
/// `pyjitpl.py _get_arrayitem_vable_index` reaches the slot through
/// `indexbox.getint()` after `implement_guard_value(indexbox, pc)`, so the
/// index is a constant by the time the slot is chosen.  pyre promotes the same
/// way (`walker_promote_vable_array_index`) but has to read the concrete value
/// out of the walker first, and an `OpRef` with no recorded concrete cannot
/// supply one — the array slot would otherwise be picked from a value the
/// trace never pinned.
///
/// The abort itself never fires on the synth corpus (measured: 0, against
/// `VableEscapedDuringResidualCall`'s 149 events over 20 of 446 fixtures — the
/// `[fbw-census]` rows count events, not fixtures, and exist only for a
/// fixture that declined at least once, so they carry no corpus-wide
/// denominator).
/// That is the reason to pin it rather than not: nothing else would notice if
/// a refactor made this path unreachable, or made it fire where the promote
/// should have carried the index.
#[test]
fn array_vable_handlers_with_unpinned_index_surface_index_not_concrete() {
    // operand 0 (the box) at code[pc+1], operand 1 (the index) at code[pc+2].
    let code = [0u8, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00];
    for (key, opname, argcodes) in [
        (
            "getarrayitem_vable_i/riXdd>i",
            "getarrayitem_vable_i",
            "riXdd>i",
        ),
        (
            "setarrayitem_vable_i/riXdd",
            "setarrayitem_vable_i",
            "riXdd",
        ),
    ] {
        let descr_pool: Vec<DescrRef> = Vec::new();
        let mut tc = fresh_trace_ctx();
        // Ref reg 0 is seeded, so the `VableBoxNotSeeded` guard above lets this
        // through; int reg 1 holds an `OpRef` the walker never gave a concrete.
        let regs_r = vec![OpRef::input_arg_ref(0)];
        let regs_i = vec![OpRef::NONE, OpRef::NONE];
        let session = std::cell::RefCell::new(WalkSession::default());
        let mut wc = WalkContext {
            frame_state: WalkFrameState::new(WalkFrameStateData {
                callee_shadow: None,
                concrete_registers_r: ([]).to_vec(),
                outer_active_boxes: Vec::new(),
                vstack_boxes: Vec::new(),
                vstack_last_ref: OpRef::NONE,
                vstack_reorder_saved: None,
                ..Default::default()
            }),
            inline_callee_consts: None,
            fbw_mode: test_fbw_mode(),
            session: &session,
            registers_r: &RegisterBank::new(regs_r.iter().copied()),
            registers_i: &RegisterBank::new(regs_i.iter().copied()),
            registers_f: &RegisterBank::default(),

            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            entry_py_pc: EntryPyPc::Py(0),
            outer_resume_marker_jit_pc: None,
            outer_jitcode_index: 0,
            raw_descrs: RawDescrPool::Global,
            is_authoritative_executor: false,

            pending_guard_snapshot_error: None,

            vstack_depth: 0,
            vstack_cur_pypc: 0,
            vstack_valid: false,

            vstack_reorder_ceiling: u32::MAX,

            vstack_handler_landing_py: None,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
        };
        let op = DecodedOp {
            key,
            opname,
            argcodes,
            pc: 0,
            next_pc: code.len(),
        };
        let result = match opname {
            "getarrayitem_vable_i" => getarrayitem_vable_via_metainterp(&code, &op, &mut wc, 'i'),
            "setarrayitem_vable_i" => setarrayitem_vable_via_metainterp(&code, &op, &mut wc, 'i'),
            _ => unreachable!(),
        };
        assert!(
            matches!(
                result,
                Err(DispatchError::VableArrayIndexNotConcrete { pc: 0, .. })
            ),
            "{opname} must abort VableArrayIndexNotConcrete on an unpinned index, got {result:?}",
        );
    }
}

/// `_opimpl_getarrayitem_vable` (pyjitpl.py) and
/// `_opimpl_setarrayitem_vable` (:1236-1247) take the
/// `_nonstandard_virtualizable` decision FIRST, and their non-standard branch
/// reaches `getarrayitem_gc_*` / `setarrayitem_gc_any` with the index box as it
/// stands. The promote lives at the head of `_get_arrayitem_vable_index`
/// (:1201-1216), which only the standard branch enters. So a non-standard
/// access must mint no GUARD_VALUE on the index — promoting there pins a value
/// upstream leaves free, over-specializing the trace on an access that reads
/// ordinary heap.
///
/// The non-standard branch mints a GUARD_VALUE of its own (the Step 4 PTR_EQ
/// promote), so the subject here is which BOX a guard names, not how many
/// guards there are.
#[test]
fn a_nonstandard_vable_array_access_does_not_promote_the_index() {
    for opname in ["getarrayitem_vable_i", "setarrayitem_vable_i"] {
        let mut tc = TraceCtx::for_test_types(&[Type::Ref]);
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        let array_len = 3;
        let slot_count = info.num_static_extra_boxes + array_len;
        // The standard virtualizable, i.e. `virtualizable_boxes[-1]`.
        let standard = tc.const_ref(1);
        let initial_boxes = vec![tc.const_null(); slot_count];
        let initial_values = vec![Value::Ref(majit_ir::GcRef::NULL); slot_count];
        tc.install_virtualizable_info(info.clone());
        tc.init_virtualizable_boxes(
            &info,
            standard,
            Value::Ref(majit_ir::GcRef(1)),
            &initial_boxes,
            &initial_values,
            &[array_len],
        );
        // A different frame: `_nonstandard_virtualizable` reaches its Step 4
        // PTR_EQ, reads unequal, and takes the non-standard branch.
        //
        // It must NOT be a `ConstPtr`. Step 4 runs its PTR_EQ through
        // `execute_and_record`, so a constant on both sides folds the
        // comparison away and mints no guard — which would leave the
        // `num_guards() > guards_before` line below measuring nothing. A
        // virtualizable in a register is a recorded box in production anyway;
        // the concrete pointer arrives from the frame's ref values, not from
        // the box being constant.
        let vable = OpRef::input_arg_ref(0);
        tc.set_opref_concrete(vable, Value::Ref(majit_ir::GcRef(2)));
        // A non-constant index carrying a recording-time concrete — the only
        // shape for which the promote is not already a no-op.
        let zero = tc.const_int(0);
        let index = tc.record_op(majit_ir::OpCode::IntAdd, &[zero, zero]);
        tc.set_opref_concrete(index, Value::Int(0));
        assert!(!index.is_constant(), "the index box must be promotable");

        let bh_array = majit_jitcode::jitcode::BhDescr::from_array_descr(
            info.array_descrs[0]
                .as_array_descr()
                .expect("the fixture's array descr"),
        );
        let raw_pool = vec![
            majit_metainterp::jitcode::RuntimeBhDescr::Descr(Box::new(
                majit_jitcode::jitcode::BhDescr::VableArray { index: 0 },
            )),
            majit_metainterp::jitcode::RuntimeBhDescr::Descr(Box::new(bh_array)),
        ];
        // `ridd>i` (read): r-reg(0) + i-reg(1) + fdescr(0) + adescr(1) + dst(2).
        // `riidd` (write): r-reg(0) + i-reg(1) + value i-reg(2) + fdescr(0) +
        // adescr(1).
        let code: [u8; 8] = if opname == "getarrayitem_vable_i" {
            [0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x02]
        } else {
            [0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x01, 0x00]
        };
        let stored_value = tc.const_int(7);
        let regs_r = vec![vable];
        let regs_i = vec![zero, index, stored_value];
        let descr_pool: Vec<DescrRef> = Vec::new();
        let guards_before = tc.num_guards();
        let session = std::cell::RefCell::new(WalkSession::default());
        let mut wc = WalkContext {
            frame_state: WalkFrameState::new(WalkFrameStateData {
                callee_shadow: None,
                concrete_registers_r: ([]).to_vec(),
                outer_active_boxes: Vec::new(),
                vstack_boxes: Vec::new(),
                vstack_last_ref: OpRef::NONE,
                vstack_reorder_saved: None,
                ..Default::default()
            }),
            inline_callee_consts: None,
            fbw_mode: test_fbw_mode(),
            session: &session,
            registers_r: &RegisterBank::new(regs_r.iter().copied()),
            registers_i: &RegisterBank::new(regs_i.iter().copied()),
            registers_f: &RegisterBank::default(),

            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            raw_descrs: RawDescrPool::PerFn(&raw_pool),
            is_authoritative_executor: false,
            trace_ctx: &mut tc,
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            entry_py_pc: EntryPyPc::Py(0),
            outer_resume_marker_jit_pc: None,
            outer_jitcode_index: 0,

            pending_guard_snapshot_error: None,

            vstack_depth: 0,
            vstack_cur_pypc: 0,
            vstack_valid: false,

            vstack_reorder_ceiling: u32::MAX,

            vstack_handler_landing_py: None,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
        };
        let op = DecodedOp {
            key: opname,
            opname,
            argcodes: "",
            pc: 0,
            next_pc: code.len(),
        };
        let result = match opname {
            "getarrayitem_vable_i" => getarrayitem_vable_via_metainterp(&code, &op, &mut wc, 'i'),
            _ => setarrayitem_vable_via_metainterp(&code, &op, &mut wc, 'i'),
        };
        drop(wc);
        assert!(
            tc.num_guards() > guards_before,
            "{opname} must still mint the Step 4 PTR_EQ promote, or this test \
             would pass on a walk that never took the non-standard branch",
        );
        // Read the guard before the dispatch outcome: an unconditional promote
        // records its GUARD_VALUE and only then tries to build a snapshot, so
        // checking the outcome first would report the snapshot failure and
        // leave the promote itself unnamed.
        let promoted_index = tc.ops().iter().any(|recorded| {
            recorded.opcode == majit_ir::OpCode::GuardValue
                && recorded
                    .getarglist()
                    .first()
                    .is_some_and(|arg| arg.to_opref() == index)
        });
        assert!(
            !promoted_index,
            "{opname} promoted the index on the non-standard branch; \
             pyjitpl.py:1220/:1239 reach the heap op with the index box unpromoted",
        );
        assert_eq!(
            result.map(|(outcome, _)| outcome),
            Ok(DispatchOutcome::Continue),
            "{opname} must dispatch through the non-standard branch",
        );
    }
}

#[test]
#[ignore = "T3 audit probe — dumps runtime opnames + walker-handled set + \
                per-opname JitCode hit count. Run with \
                `cargo test -p pyre-jit-trace --features dynasm --lib \
                t3_audit_opname_gap_inventory -- --ignored --nocapture` to \
                produce a project memory entry; not a permanent test."]
fn t3_audit_opname_gap_inventory() {
    use crate::jitcode_runtime::{all_jitcodes, insns_byte_to_opname, insns_opname_to_byte};

    // 1) Runtime opnames (pyre's actual codewriter emission set).
    let runtime_names: std::collections::BTreeSet<String> =
        insns_opname_to_byte().keys().cloned().collect();

    // 2) Walker-handled opnames — parsed from the embedded `handle`
    // function's string literals.  Source-of-truth scan against
    // the files themselves so this probe stays accurate as handlers
    // land/leave.  `arith.rs` carries the `regular_record_table!` arms
    // (the `int_*` / `float_*` / `ptr_*` families) as opname literals, and
    // `residual_call.rs` carries the residual-call body match arms, so both
    // are scanned alongside `mod.rs`.
    let source = concat!(
        include_str!("mod.rs"),
        "\n",
        include_str!("arith.rs"),
        "\n",
        include_str!("residual_call.rs"),
    );
    let mut handled: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // Heuristic: scan the literal patterns that appear ONLY in
    // dispatch arms of `handle()` — they look like
    // `"<opname>/[argcodes]" => ...`.  Filter to entries that are
    // also in the runtime table to drop test-fixture literals.
    // An arm may list several keys on one line (`"a/i" | "b/c" => ...`),
    // so every literal on the line counts, not just the leading one —
    // reading only the first silently under-reports the handled set.
    for line in source.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with('"') {
            continue;
        }
        for (idx, key) in trimmed.split('"').enumerate() {
            // Odd indices are the quoted spans; even ones are the
            // separators between them.
            if idx % 2 == 0 {
                continue;
            }
            // Must contain '/' (separates opname from argcodes); skip
            // anything that doesn't look like an opname/argcodes literal.
            if !key.contains('/') {
                continue;
            }
            if runtime_names.contains(key) {
                handled.insert(key.to_string());
            }
        }
    }

    let unhandled: Vec<&String> = runtime_names.difference(&handled).collect();

    // 3) Per-opname JitCode hit count — for each unhandled opname,
    // count how many JitCodes contain its opcode byte.  Higher
    // counts = higher likelihood of blocking the next opcode
    // entering the shadow allow-list.
    let opname_to_byte = insns_opname_to_byte();
    let byte_to_opname = insns_byte_to_opname();
    let all_jcs = all_jitcodes();

    // For accurate counts the byte must be at a true OP position,
    // not an operand position.  We need to walk each JitCode using
    // `decoded_ops` to enumerate true op bytes.
    let mut hit_counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for jc in all_jcs {
        for op in crate::jitcode_runtime::decoded_ops(&jc.code) {
            let key = op.key;
            if handled.contains(key) {
                continue;
            }
            *hit_counts.entry(key.to_string()).or_insert(0) += 1;
        }
    }

    eprintln!();
    eprintln!("=== T3 AUDIT: runtime opnames ===");
    eprintln!("total runtime opnames: {}", runtime_names.len());
    eprintln!("walker-handled opnames: {}", handled.len());
    eprintln!("unhandled opnames: {}", unhandled.len());

    eprintln!();
    eprintln!("=== T3 AUDIT: unhandled opnames ranked by JitCode hit count ===");
    let mut ranked: Vec<(String, usize)> = hit_counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (name, count) in &ranked {
        let byte = opname_to_byte
            .get(name)
            .map(|b| format!("0x{b:02x}"))
            .unwrap_or_else(|| "?".to_string());
        eprintln!("  {count:>5}  {byte}  {name}");
    }

    eprintln!();
    eprintln!("=== T3 AUDIT: unhandled opnames with ZERO JitCode hits ===");
    for name in &unhandled {
        if !ranked.iter().any(|(n, _)| n == *name) {
            let byte = opname_to_byte
                .get(*name)
                .map(|b| format!("0x{b:02x}"))
                .unwrap_or_else(|| "?".to_string());
            eprintln!("  {byte}  {name}");
        }
    }

    eprintln!();
    eprintln!("=== T3 AUDIT: walker-handled opnames (for cross-check) ===");
    for name in &handled {
        eprintln!("  {name}");
    }

    // Sanity: byte_to_opname must invert opname_to_byte.
    assert_eq!(byte_to_opname.len(), opname_to_byte.len());
}

#[test]
fn regular_record_table_is_the_sole_dispatcher() {
    // The `regular_record_table!` macro (arith.rs) owns dispatch for the
    // uniform `int_*` / `float_*` / `ptr_*` record families. Guard the
    // invariant that those opnames route ONLY through the table:
    // (a) it is non-empty with unique keys, (b) every key is a real
    // runtime opname, (c) none still appears as a `"opname" =>` arm in
    // `handle` (which the pre-match table lookup would shadow into dead
    // code).
    let keys = REGULAR_RECORD_KEYS;
    assert!(!keys.is_empty(), "table must route at least one opname");

    // (a) unique keys.
    let unique: std::collections::BTreeSet<&str> = keys.iter().copied().collect();
    assert_eq!(
        unique.len(),
        keys.len(),
        "REGULAR_RECORD_KEYS has duplicates"
    );

    // (b) every key is a real codewriter-emitted opname, except the
    // documented dormant `int_same_as/i>i` forward-prep arm: RPython
    // `jtransform.py rewrite_op_same_as` strips `same_as` before
    // assembly, so it is intentionally absent from the runtime table while
    // its dispatch entry is kept for the walker's forward-prep path.
    let runtime = insns_opname_to_byte();
    for key in keys {
        if *key == "int_same_as/i>i" {
            continue;
        }
        assert!(
            runtime.contains_key(*key),
            "table key {key:?} is not a runtime opname",
        );
    }

    // (c) no key survives as a dispatch arm in `handle`.
    let handle_src = include_str!("mod.rs");
    for key in keys {
        let arm = format!("\"{key}\" =>");
        assert!(
            !handle_src.contains(&arm),
            "opname {key:?} still has a dead dispatch arm in mod.rs; \
             it is dispatched by regular_record_table!",
        );
    }
}

fn drive_int_add_jump_if_ovf(
    lhs_value: i64,
    rhs_value: i64,
) -> (Vec<OpCode>, usize, bool, u32, OpRef, OpRef, usize) {
    let opname = "int_add_jump_if_ovf/Lii>i";
    let ovf_byte = *insns_opname_to_byte()
        .get(opname)
        .expect("int_add_jump_if_ovf must be in the runtime instruction table");
    let live_byte = *insns_opname_to_byte()
        .get("live/")
        .expect("live must be in the runtime instruction table");
    // Lii>i: target 9, source registers 0/1, destination register 2.
    // Separate live tails at 6 and 9 make fallthrough and jump observable.
    let code = [ovf_byte, 9, 0, 0, 1, 2, live_byte, 0, 0, live_byte, 0, 0];
    let _ = test_outer_resume_jitcode_index();
    let mut tc = TraceCtx::for_test_types(&[Type::Int, Type::Int]);
    let lhs = OpRef::input_arg_int(0);
    let rhs = OpRef::input_arg_int(1);
    tc.set_opref_concrete(lhs, Value::Int(lhs_value));
    tc.set_opref_concrete(rhs, Value::Int(rhs_value));
    let regs_i = vec![lhs, rhs, OpRef::NONE];
    let mut concrete_i = vec![
        ConcreteValue::Int(lhs_value),
        ConcreteValue::Int(rhs_value),
        ConcreteValue::Null,
    ];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut concrete_i,
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: Some(0),
        outer_jitcode_index: test_outer_resume_jitcode_index(),

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("int_add_jump_if_ovf must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    let dst = wc.registers_i.get(2).expect("int register in range");
    drop(wc);
    let ops = tc.ops().to_vec();
    let opcodes = ops.iter().map(|op| op.opcode).collect();
    let guard_num_args = ops[1].num_args();
    let guard_has_snapshot = ops[1].rd_resume_position() >= 0;
    let resume_pos = ops[1].rd_resume_position();
    let resbox = ops[0].pos().get();
    let guard_resume_pc = tc
        .get_snapshot(resume_pos)
        .expect("overflow guard snapshot must exist")
        .frames
        .last()
        .expect("overflow guard snapshot must contain its frame")
        .pc;
    (
        opcodes,
        guard_num_args,
        guard_has_snapshot,
        guard_resume_pc,
        resbox,
        dst,
        next_pc,
    )
}

#[test]
fn int_add_jump_if_ovf_no_overflow_records_guard_no_overflow_and_writes_dst() {
    let (opcodes, guard_num_args, guard_has_snapshot, guard_resume_pc, resbox, dst, next_pc) =
        drive_int_add_jump_if_ovf(40, 2);
    assert_eq!(opcodes, vec![OpCode::IntAddOvf, OpCode::GuardNoOverflow],);
    assert_eq!(guard_num_args, 0, "GuardNoOverflow is operand-less");
    assert!(guard_has_snapshot);
    assert_eq!(
        guard_resume_pc, 0,
        "the guard resumes at the overflow opcode"
    );
    assert_eq!(next_pc, 6, "no overflow continues at op.next_pc");
    assert_eq!(dst, resbox, "the no-overflow continue writes resbox to dst");
}

#[test]
fn int_add_jump_if_ovf_overflow_records_guard_overflow_and_jumps() {
    let (opcodes, guard_num_args, guard_has_snapshot, guard_resume_pc, _, dst, next_pc) =
        drive_int_add_jump_if_ovf(i64::MAX, 1);
    assert_eq!(opcodes, vec![OpCode::IntAddOvf, OpCode::GuardOverflow],);
    assert_eq!(guard_num_args, 0, "GuardOverflow is operand-less");
    assert!(guard_has_snapshot);
    assert_eq!(
        guard_resume_pc, 0,
        "the guard resumes at the overflow opcode"
    );
    assert_eq!(dst, OpRef::NONE, "the overflow jump does not write dst");
    assert_eq!(next_pc, 9, "overflow jumps to the handler target");
}

#[test]
fn int_ovf_jump_constant_operands_fold_without_recording_an_ovf_op() {
    let byte = *insns_opname_to_byte()
        .get("int_add_jump_if_ovf/Lii>i")
        .expect("int_add_jump_if_ovf must be in the runtime instruction table");
    let code = [byte, 6, 0, 0, 1, 2];
    let mut tc = fresh_trace_ctx();
    let lhs = tc.const_int(40);
    let rhs = tc.const_int(2);
    let mut regs_i = [lhs, rhs, OpRef::NONE];
    let (outcome, next_pc) = run_hint_step(&code, &mut tc, &mut [], &mut [], &mut regs_i)
        .expect("constant overflow arithmetic must fold");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 6);
    assert_eq!(
        tc.num_ops(),
        0,
        "folding emits neither an ovf op nor a guard"
    );
    assert_eq!(regs_i[2].inline_const_to_value(), Some(Value::Int(42)));
}

#[test]
fn int_ovf_jump_declines_when_an_operand_is_not_concrete() {
    let byte = *insns_opname_to_byte()
        .get("int_add_jump_if_ovf/Lii>i")
        .expect("int_add_jump_if_ovf must be in the runtime instruction table");
    let code = [byte, 6, 0, 0, 1, 2];
    let mut tc = TraceCtx::for_test_types(&[Type::Int, Type::Int]);
    let lhs = OpRef::input_arg_int(0);
    let rhs = OpRef::input_arg_int(1);
    let mut regs_i = [lhs, rhs, OpRef::NONE];
    let err = run_hint_step(&code, &mut tc, &mut [], &mut [], &mut regs_i)
        .expect_err("unstamped overflow operands cannot guess the no-overflow arm");
    match err {
        DispatchError::UnsupportedOpname { key, .. } => {
            assert_eq!(key, "int_*_ovf (operands not concrete)");
        }
        other => panic!("expected unsupported overflow operands, got {other:?}"),
    }
}

#[test]
fn int_ovf_jump_recovers_overflow_from_int_bank_shadow() {
    let byte = *insns_opname_to_byte()
        .get("int_add_jump_if_ovf/Lii>i")
        .expect("int_add_jump_if_ovf must be in the runtime instruction table");
    let live_byte = *insns_opname_to_byte()
        .get("live/")
        .expect("live must be in the runtime instruction table");
    let code = [byte, 9, 0, 0, 1, 2, live_byte, 0, 0, live_byte, 0, 0];
    let mut tc = TraceCtx::for_test_types(&[Type::Int, Type::Int]);
    let lhs = OpRef::input_arg_int(0);
    let rhs = OpRef::input_arg_int(1);
    let mut regs_i = [lhs, rhs, OpRef::NONE];
    let mut concrete_i = [
        ConcreteValue::Int(i64::MAX),
        ConcreteValue::Int(1),
        ConcreteValue::Null,
    ];
    let (outcome, next_pc) = run_hint_step_full(
        &code,
        &mut tc,
        &mut [],
        &mut [],
        &mut regs_i,
        &mut concrete_i,
        &mut [],
        &[],
    )
    .expect("int-bank shadow recovers the overflow decision");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 9, "overflow jumps to the handler target");
    let opcodes: Vec<_> = tc.ops().iter().map(|op| op.opcode).collect();
    assert_eq!(opcodes, vec![OpCode::IntAddOvf, OpCode::GuardOverflow]);
}

/// Drive one of the `d>r` struct-allocation handlers (`new`,
/// `new_with_vtable`): both record the descr alone and write the
/// allocation into the ref bank.
///
/// Returns the concrete stamped onto the recorded allocation op, which is
/// `None` without a cpu (no `bh_new` to execute).
fn drive_alloc_with_descr(
    opname: &str,
    expected_opcode: OpCode,
    cpu: Option<&dyn majit_backend::Backend>,
) -> Option<Value> {
    let nwv_byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in the runtime instruction table"));
    let live_byte = *insns_opname_to_byte()
        .get("live/")
        .expect("live must be in the runtime instruction table");
    // `d>r`: 2B descr index (0) + 1B dst register (0); live tail at 4.
    let code = [nwv_byte, 0, 0, 0, live_byte, 0, 0];
    let _ = test_outer_resume_jitcode_index();
    let descr_pool = vec![crate::descr::w_int_size_descr()];
    let mut tc = TraceCtx::for_test_types(&[]);
    tc.set_cpu(cpu);
    let regs_r = vec![OpRef::NONE];
    let concrete_r = vec![ConcreteValue::Null];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: (concrete_r).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: Some(0),
        outer_jitcode_index: test_outer_resume_jitcode_index(),

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).unwrap_or_else(|_| panic!("`{opname}` must dispatch"));
    assert_eq!(outcome, DispatchOutcome::Continue);
    let dst = wc.registers_r.get(0).expect("ref register in range");
    drop(wc);

    let ops = tc.ops();
    assert_eq!(
        ops.iter().map(|op| op.opcode).collect::<Vec<_>>(),
        vec![expected_opcode],
    );
    assert_eq!(
        ops[0].num_args(),
        0,
        "the allocation records the descr alone, with no box operands"
    );
    assert_eq!(
        dst,
        ops[0].pos().get(),
        "the `>r` decorator writes the allocation into the ref bank"
    );
    assert_eq!(next_pc, 4, "`d>r` consumes a 2B descr plus a 1B dst");
    tc.lookup_opref_concrete(dst)
}

#[test]
fn alloc_records_the_alloc_and_writes_the_ref_dst() {
    let cases = [
        (
            "new_with_vtable",
            "new_with_vtable/d>r",
            OpCode::NewWithVtable,
        ),
        ("new", "new/d>r", OpCode::New),
    ];
    for (name, opname, opcode) in cases {
        assert!(opname.starts_with(name), "case {name}");
        drive_alloc_with_descr(opname, opcode, None);
    }
}

/// Backend stub for the allocation-rooting tests: `bh_new*` hands back one
/// caller-owned block so the handler observes a real, dereferenceable
/// pointer (`new_with_vtable` writes `w_class` into it).
struct AllocTestCpu {
    block: i64,
}
impl majit_backend::Backend for AllocTestCpu {
    fn bh_new(&self, _sizedescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        self.block
    }
    fn bh_new_with_vtable(&self, _sizedescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        self.block
    }
    fn compile_loop(
        &mut self,
        _inputargs: &[majit_ir::InputArgRc],
        _ops: &[majit_ir::OpRc],
        _token: &majit_backend::JitCellToken,
    ) -> Result<majit_backend::AsmInfo, majit_backend::BackendError> {
        unimplemented!("AllocTestCpu::compile_loop")
    }
    fn compile_bridge(
        &mut self,
        _fail_descr: &dyn majit_ir::FailDescr,
        _inputargs: &[majit_ir::InputArgRc],
        _ops: &[majit_ir::OpRc],
        _original_token: &majit_backend::JitCellToken,
        _previous_tokens: &[std::sync::Arc<majit_backend::JitCellToken>],
        _caller_recovery_layout: Option<&majit_backend::ExitRecoveryLayout>,
    ) -> Result<majit_backend::AsmInfo, majit_backend::BackendError> {
        unimplemented!("AllocTestCpu::compile_bridge")
    }
    fn execute_token(
        &self,
        _token: &majit_backend::JitCellToken,
        _args: &[majit_ir::Value],
    ) -> majit_backend::DeadFrame {
        unimplemented!("AllocTestCpu::execute_token")
    }
    fn get_latest_descr<'a>(
        &'a self,
        _frame: &'a majit_backend::DeadFrame,
    ) -> &'a dyn majit_ir::FailDescr {
        unimplemented!("AllocTestCpu::get_latest_descr")
    }
    fn get_latest_descr_arc(
        &self,
        _frame: &majit_backend::DeadFrame,
    ) -> std::sync::Arc<dyn majit_ir::descr::Descr> {
        unimplemented!("AllocTestCpu::get_latest_descr_arc")
    }
    fn get_int_value(&self, _frame: &majit_backend::DeadFrame, _index: usize) -> i64 {
        unimplemented!("AllocTestCpu::get_int_value")
    }
    fn get_value_direct(&self, frame: &majit_backend::DeadFrame, slot: usize) -> i64 {
        // AllocTestCpu's slot space is the dense fail-value vector.
        self.get_int_value(frame, slot)
    }
    fn get_float_value(&self, _frame: &majit_backend::DeadFrame, _index: usize) -> f64 {
        unimplemented!("AllocTestCpu::get_float_value")
    }
    fn get_ref_value(&self, _frame: &majit_backend::DeadFrame, _index: usize) -> majit_ir::GcRef {
        unimplemented!("AllocTestCpu::get_ref_value")
    }
    fn invalidate_loop(&self, _token: &majit_backend::JitCellToken) {
        unimplemented!("AllocTestCpu::invalidate_loop")
    }
}

/// The object `execute_new[_with_vtable]` allocates is rooted by being
/// stamped onto the op recorded for it: `MetaInterp::walk_active_trace_refs`
/// forwards every recorder `Op` `value` cell holding a `Value::Ref`
/// (`history.py` `*FrontendOp(pos, value)`), and that stamp is the
/// only thing putting a walk-time allocation in the root set — the walker's
/// `concrete_registers_r` shadow is a borrowed slice, not a root.
///
/// Losing the stamp would leave the allocation reachable only from that
/// shadow, so pin it here rather than relying on a side list of executed
/// allocations, which would duplicate a root the op graph already owns.
fn alloc_result_is_stamped_onto_its_recorded_op(opname: &str, expected_opcode: OpCode) {
    // Real backing storage: `new_with_vtable` writes `w_class` into the
    // returned block, so a synthetic address would be a wild store.
    let block: Box<[usize; 32]> = Box::new([0; 32]);
    let cpu = AllocTestCpu {
        block: block.as_ref().as_ptr() as i64,
    };
    let stamped = drive_alloc_with_descr(opname, expected_opcode, Some(&cpu));
    assert_eq!(
        stamped,
        Some(Value::Ref(majit_ir::GcRef(cpu.block as usize))),
        "`{opname}` must stamp the allocation onto its recorded op — that cell \
         is the GC root the walk relies on",
    );
}

#[test]
fn new_stamps_its_allocation_onto_the_recorded_op() {
    alloc_result_is_stamped_onto_its_recorded_op("new/d>r", OpCode::New);
}

#[test]
fn new_with_vtable_stamps_its_allocation_onto_the_recorded_op() {
    alloc_result_is_stamped_onto_its_recorded_op("new_with_vtable/d>r", OpCode::NewWithVtable);
}

/// Step one record-only opcode (the heapcache hints, the raw-memory pair)
/// over caller-supplied banks.
fn run_hint_step(
    code: &[u8],
    tc: &mut TraceCtx,
    regs_r: &mut [OpRef],
    concrete_r: &mut [ConcreteValue],
    regs_i: &mut [OpRef],
) -> Result<(DispatchOutcome, usize), DispatchError> {
    run_hint_step_with_descrs(code, tc, regs_r, concrete_r, regs_i, &[])
}

fn run_hint_step_with_descrs(
    code: &[u8],
    tc: &mut TraceCtx,
    regs_r: &mut [OpRef],
    concrete_r: &mut [ConcreteValue],
    regs_i: &mut [OpRef],
    descr_pool: &[DescrRef],
) -> Result<(DispatchOutcome, usize), DispatchError> {
    run_hint_step_full(
        code,
        tc,
        regs_r,
        concrete_r,
        regs_i,
        &mut [],
        &mut [],
        descr_pool,
    )
}

/// The banks an arm that reads the Int-bank concrete shadow or the Float
/// bank needs, which the two shorthands above leave empty.
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per register bank the walk context carries; \
              bundling them would hide which bank a test leaves empty"
)]
fn run_hint_step_full(
    code: &[u8],
    tc: &mut TraceCtx,
    regs_r: &mut [OpRef],
    concrete_r: &mut [ConcreteValue],
    regs_i: &mut [OpRef],
    concrete_i: &mut [ConcreteValue],
    regs_f: &mut [OpRef],
    descr_pool: &[DescrRef],
) -> Result<(DispatchOutcome, usize), DispatchError> {
    with_hint_walk_context(
        tc,
        regs_r,
        concrete_r,
        regs_i,
        concrete_i,
        regs_f,
        descr_pool,
        |wc| step(code, 0, wc),
    )
}

/// Run `run` against the walk context [`run_hint_step_full`] builds, then
/// copy the register banks back.
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per register bank the walk context carries; \
              bundling them would hide which bank a test leaves empty"
)]
fn with_hint_walk_context<R>(
    tc: &mut TraceCtx,
    regs_r: &mut [OpRef],
    concrete_r: &mut [ConcreteValue],
    regs_i: &mut [OpRef],
    concrete_i: &mut [ConcreteValue],
    regs_f: &mut [OpRef],
    descr_pool: &[DescrRef],
    run: impl FnOnce(&mut WalkContext<'_, '_, crate::state::PyreSym>) -> R,
) -> R {
    // Guard-emitting arms need a resolvable outer resume coordinate for the
    // snapshot; record-only arms ignore these two fields.
    let outer_jitcode_index = test_outer_resume_jitcode_index();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: concrete_r.to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::new(regs_f.iter().copied()),

        concrete_registers_i: concrete_i,
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: Some(0),
        outer_jitcode_index,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let result = run(&mut wc);
    regs_f.copy_from_slice(&wc.registers_f.to_vec());
    regs_i.copy_from_slice(&wc.registers_i.to_vec());
    regs_r.copy_from_slice(&wc.registers_r.to_vec());
    concrete_r.copy_from_slice(&wc.frame_state.borrow().concrete_registers_r);
    result
}

#[test]
fn assert_not_none_records_when_the_operand_has_a_concrete() {
    let byte = *insns_opname_to_byte()
        .get("assert_not_none/r")
        .expect("`assert_not_none/r` must be in insns table");
    let code = [byte, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = dummy_ptr_eq_box(&mut tc);
    let mut regs_r = [operand];
    let mut concrete_r = [ConcreteValue::Ref(
        0xdead_beef_usize as *mut pyre_object::pyobject::PyObject,
    )];
    let (outcome, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("`assert_not_none/r` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 2, "`r` consumes a single register byte");
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::AssertNotNone);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![operand],
    );
}

#[test]
fn assert_not_none_declines_when_the_operand_has_no_concrete() {
    let byte = *insns_opname_to_byte()
        .get("assert_not_none/r")
        .expect("`assert_not_none/r` must be in insns table");
    let code = [byte, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = dummy_ptr_eq_box(&mut tc);
    let ops_before = tc.num_ops();
    let mut regs_r = [operand];
    // No shadow for the slot: the walker never observed this pointer, so
    // it cannot stand behind the non-null check the recorder performs.
    let mut concrete_r = [ConcreteValue::Null];
    let err = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect_err("a shadow-less operand must decline instead of asserting");
    assert_eq!(
        err,
        DispatchError::UnsupportedOpname {
            pc: 0,
            key: "assert_not_none/r",
        },
    );
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "a declined step must not leave a recorded op behind"
    );
}

#[test]
fn assert_not_none_uses_known_nonnull_heapcache_without_a_ref_shadow() {
    let byte = *insns_opname_to_byte()
        .get("assert_not_none/r")
        .expect("`assert_not_none/r` must be in insns table");
    let code = [byte, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = tc.record_op(majit_ir::OpCode::New, &[]);
    tc.heap_cache_mut().new_object(operand);
    let ops_before = tc.num_ops();
    let mut regs_r = [operand];
    let mut concrete_r = [ConcreteValue::Null];
    let (outcome, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("a heapcache-known allocation needs no ref shadow");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 2);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "the known-nonnull fast path records no AssertNotNone"
    );
}

#[test]
fn int_isconstant_writes_constint_and_records_no_ir() {
    let byte = *insns_opname_to_byte()
        .get("int_isconstant/i>i")
        .expect("`int_isconstant/i>i` must be in insns table");
    let code = [byte, 0x00, 0x01];
    let mut tc = fresh_trace_ctx();
    let src = tc.const_int(5);
    let ops_before = tc.num_ops();
    let mut regs_i = [src, OpRef::NONE];
    let mut concrete_i = [ConcreteValue::Int(5), ConcreteValue::Null];
    let (outcome, next_pc) = run_hint_step_full(
        &code,
        &mut tc,
        &mut [],
        &mut [],
        &mut regs_i,
        &mut concrete_i,
        &mut [],
        &[],
    )
    .expect("`int_isconstant/i>i` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 3);
    assert_eq!(regs_i[1], tc.const_int(1));
    assert_eq!(concrete_i[1], ConcreteValue::Int(1));
    assert_eq!(tc.num_ops(), ops_before, "isconstant records no IR");
}

#[test]
fn ref_isconstant_is_false_for_a_recorded_box() {
    let byte = *insns_opname_to_byte()
        .get("ref_isconstant/r>i")
        .expect("`ref_isconstant/r>i` must be in insns table");
    let code = [byte, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let src = tc.record_op(majit_ir::OpCode::New, &[]);
    let ops_before = tc.num_ops();
    let mut regs_r = [src];
    let mut concrete_r = [ConcreteValue::Null];
    let mut regs_i = [OpRef::NONE];
    let mut concrete_i = [ConcreteValue::Null];
    let (outcome, next_pc) = run_hint_step_full(
        &code,
        &mut tc,
        &mut regs_r,
        &mut concrete_r,
        &mut regs_i,
        &mut concrete_i,
        &mut [],
        &[],
    )
    .expect("`ref_isconstant/r>i` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 3);
    assert_eq!(regs_i[0], tc.const_int(0));
    assert_eq!(concrete_i[0], ConcreteValue::Int(0));
    assert_eq!(tc.num_ops(), ops_before, "isconstant records no IR");
}

#[test]
fn ref_isvirtual_follows_the_heapcache() {
    let byte = *insns_opname_to_byte()
        .get("ref_isvirtual/r>i")
        .expect("`ref_isvirtual/r>i` must be in insns table");
    let code = [byte, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let src = tc.record_op(majit_ir::OpCode::New, &[]);
    tc.heap_cache_mut().new_object(src);
    let ops_before = tc.num_ops();
    let mut regs_r = [src];
    let mut concrete_r = [ConcreteValue::Null];
    let mut regs_i = [OpRef::NONE];
    let mut concrete_i = [ConcreteValue::Null];
    let (outcome, next_pc) = run_hint_step_full(
        &code,
        &mut tc,
        &mut regs_r,
        &mut concrete_r,
        &mut regs_i,
        &mut concrete_i,
        &mut [],
        &[],
    )
    .expect("`ref_isvirtual/r>i` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 3);
    assert_eq!(regs_i[0], tc.const_int(1));
    assert_eq!(concrete_i[0], ConcreteValue::Int(1));
    assert_eq!(tc.num_ops(), ops_before, "isvirtual records no IR");
}

#[test]
fn fused_int_compare_folds_same_box_and_constant_operands_without_guards() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_int_eq/iiL")
        .expect("`goto_if_not_int_eq/iiL` must be in insns table");
    let code = [byte, 0x00, 0x00, 0x09, 0x00];
    let mut same_tc = TraceCtx::for_test_types(&[Type::Int]);
    let same = OpRef::input_arg_int(0);
    let mut same_regs = [same];
    let (_, same_next) = run_hint_step(&code, &mut same_tc, &mut [], &mut [], &mut same_regs)
        .expect("a same-box equality has a static direction");
    assert_eq!(same_next, code.len());
    assert_eq!(
        same_tc.num_ops(),
        0,
        "same-box equality records no compare or guard"
    );

    let code = [byte, 0x00, 0x01, 0x09, 0x00];
    let mut const_tc = fresh_trace_ctx();
    let lhs = const_tc.const_int(4);
    let rhs = const_tc.const_int(5);
    let mut const_regs = [lhs, rhs];
    let (_, const_next) = run_hint_step(&code, &mut const_tc, &mut [], &mut [], &mut const_regs)
        .expect("constant equality has a static direction");
    assert_eq!(const_next, 9);
    assert_eq!(
        const_tc.num_ops(),
        0,
        "constant equality records no compare or guard"
    );

    let ptr_byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_ne/rrL")
        .expect("`goto_if_not_ptr_ne/rrL` must be in insns table");
    let ptr_code = [ptr_byte, 0x00, 0x00, 0x09, 0x00];
    let mut ptr_tc = TraceCtx::for_test_types(&[Type::Ref]);
    let same_ptr = OpRef::input_arg_ref(0);
    let mut ptr_regs = [same_ptr];
    let (_, ptr_next) = run_hint_step(&ptr_code, &mut ptr_tc, &mut ptr_regs, &mut [], &mut [])
        .expect("a same-box pointer inequality has a static direction");
    assert_eq!(ptr_next, 9);
    assert_eq!(
        ptr_tc.num_ops(),
        0,
        "same-box pointer inequality records no compare or guard"
    );
}

#[test]
fn fused_int_is_zero_folds_a_const_operand_without_recording() {
    // `opimpl_goto_if_not_int_is_zero`'s condbox comes from
    // `self.execute(rop.INT_IS_ZERO, box)` — a `Const` operand folds the
    // condition without recording, and `opimpl_goto_if_not` emits no guard
    // for a `Const` condbox.
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_int_is_zero/iL")
        .expect("`goto_if_not_int_is_zero/iL` must be in insns table");
    let code = [byte, 0x00, 0x09, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = tc.const_int(5);
    let mut regs_i = [operand];
    let (_, next_pc) = run_hint_step(&code, &mut tc, &mut [], &mut [], &mut regs_i)
        .expect("a const int_is_zero condition has a static direction");
    assert_eq!(next_pc, 9, "is_zero(5) = 0 takes the not-taken label");
    assert_eq!(
        tc.num_ops(),
        0,
        "const int_is_zero records no IntIsZero and no guard"
    );
}

#[test]
fn fused_ptr_compare_uses_ref_shadows_for_the_runtime_direction() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_eq/rrL")
        .expect("`goto_if_not_ptr_eq/rrL` must be in insns table");
    let code = [byte, 0x00, 0x01, 0x09, 0x00];
    let mut tc = TraceCtx::for_test_types(&[Type::Ref, Type::Ref]);
    let lhs = OpRef::input_arg_ref(0);
    let rhs = OpRef::input_arg_ref(1);
    let mut regs_r = [lhs, rhs];
    let ptr = 0xdead_beef_usize as *mut pyre_object::pyobject::PyObject;
    let mut concrete_r = [ConcreteValue::Ref(ptr), ConcreteValue::Ref(ptr)];
    let (_, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("ref shadows determine the pointer comparison direction");
    assert_eq!(next_pc, code.len());
    assert_eq!(
        tc.ops().iter().map(|op| op.opcode).collect::<Vec<_>>(),
        vec![OpCode::PtrEq, OpCode::GuardTrue]
    );
}

#[test]
fn ptr_eq_folds_from_the_ref_shadow_when_box_value_is_absent() {
    // A non-fused `ptr_eq/rr>i` whose operands carry a concrete only in the
    // ref-register shadow (box_value absent — the inline-callee-param case)
    // must still fold the bool result, matching the fused twin and
    // ptr_nullity. Without the shadow fallback the bool stayed symbolic and a
    // downstream goto_if_not declined.
    let byte = *insns_opname_to_byte()
        .get("ptr_eq/rr>i")
        .expect("`ptr_eq/rr>i` must be in insns table");
    // `rr>i`: 1B r-src1 + 1B r-src2 + 1B i-dst.
    let code = [byte, 0x00, 0x01, 0x00];
    let mut tc = TraceCtx::for_test_types(&[Type::Ref, Type::Ref]);
    let a = OpRef::input_arg_ref(0);
    let b = OpRef::input_arg_ref(1);
    assert_eq!(
        tc.box_value(a),
        None,
        "an input-arg ref has no box_value carrier"
    );
    let mut regs_r = [a, b];
    let mut regs_i = [OpRef::None];
    let pa = 0xdead_0000_usize as *mut pyre_object::pyobject::PyObject;
    let pb = 0xbeef_0000_usize as *mut pyre_object::pyobject::PyObject;
    let mut concrete_r = [ConcreteValue::Ref(pa), ConcreteValue::Ref(pb)];
    let (_, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut regs_i)
        .expect("ptr_eq must record from the ref shadow");
    assert_eq!(next_pc, 4);
    assert_eq!(
        tc.concrete_of_opref(regs_i[0]),
        Some(majit_ir::Value::Int(0)),
        "distinct shadow pointers fold PtrEq to false without a box_value carrier",
    );
}

#[test]
fn hint_force_virtualizable_is_a_noop_without_virtualizable_info() {
    let byte = *insns_opname_to_byte()
        .get("hint_force_virtualizable/r")
        .expect("`hint_force_virtualizable/r` must be in insns table");
    // `r`: 1B ref reg.
    let code = [byte, 0x00];
    let mut tc = fresh_trace_ctx();
    let vable = dummy_ptr_eq_box(&mut tc);
    let ops_before = tc.num_ops();
    let mut regs_r = [vable];
    let mut concrete_r = [ConcreteValue::Null];
    let (outcome, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("`hint_force_virtualizable/r` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 2, "`r` consumes a single register byte");
    // `gen_store_back_in_vable` returns before emitting anything when the
    // jitdriver carries no virtualizable info -- the upstream `vinfo is None`
    // gate -- so the hint writes back nothing here.
    assert_eq!(tc.num_ops(), ops_before);
}

#[test]
fn goto_if_not_ptr_nonzero_guards_nonnull_and_falls_through() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_nonzero/rL")
        .expect("`goto_if_not_ptr_nonzero/rL` must be in insns table");
    // `rL`: 1B ref reg + 2B label (target 9).
    let code = [byte, 0x00, 0x09, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = dummy_ptr_eq_box(&mut tc);
    let mut regs_r = [operand];
    let mut concrete_r = [ConcreteValue::Ref(
        0xdead_beef_usize as *mut pyre_object::pyobject::PyObject,
    )];
    let (outcome, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("`goto_if_not_ptr_nonzero/rL` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 4,
        "a non-null pointer falls through past the 3 operand bytes"
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(
        last.opcode,
        majit_ir::OpCode::GuardNonnull,
        "`_establish_nullity` guards the observed non-nullness"
    );
    assert!(
        tc.heap_cache().is_nullity_known(operand),
        "the nullity must be stamped into the heapcache"
    );
}

#[test]
fn goto_if_not_ptr_iszero_takes_the_branch_when_nonnull() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_iszero/rL")
        .expect("`goto_if_not_ptr_iszero/rL` must be in insns table");
    let code = [byte, 0x00, 0x09, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = dummy_ptr_eq_box(&mut tc);
    let mut regs_r = [operand];
    let mut concrete_r = [ConcreteValue::Ref(
        0xdead_beef_usize as *mut pyre_object::pyobject::PyObject,
    )];
    let (_, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("`goto_if_not_ptr_iszero/rL` must dispatch");
    assert_eq!(
        next_pc, 9,
        "iszero jumps exactly where nonzero falls through"
    );
}

#[test]
fn goto_if_not_ptr_nonzero_guards_isnull_and_takes_the_branch() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_nonzero/rL")
        .expect("`goto_if_not_ptr_nonzero/rL` must be in insns table");
    let code = [byte, 0x00, 0x09, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = dummy_ptr_eq_box(&mut tc);
    let mut regs_r = [operand];
    let mut concrete_r = [ConcreteValue::Ref(std::ptr::null_mut())];
    let (_, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("`goto_if_not_ptr_nonzero/rL` must dispatch");
    assert_eq!(next_pc, 9, "a null pointer takes the branch");
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::GuardIsnull);
}

#[test]
fn goto_if_not_ptr_nonzero_declines_without_a_concrete() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_nonzero/rL")
        .expect("`goto_if_not_ptr_nonzero/rL` must be in insns table");
    let code = [byte, 0x00, 0x09, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = dummy_ptr_eq_box(&mut tc);
    let ops_before = tc.num_ops();
    let mut regs_r = [operand];
    let mut concrete_r = [ConcreteValue::Null];
    let err = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect_err("an unobserved pointer must decline, not pick a direction");
    assert_eq!(
        err,
        DispatchError::UnsupportedOpname {
            pc: 0,
            key: "goto_if_not_ptr_nonzero/rL",
        },
    );
    assert_eq!(tc.num_ops(), ops_before, "a declined step records nothing");
}

#[test]
fn goto_if_not_ptr_nonzero_uses_known_nonnull_heapcache_without_a_ref_shadow() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_nonzero/rL")
        .expect("`goto_if_not_ptr_nonzero/rL` must be in insns table");
    // `rL`: 1B ref reg + 2B label (target 9).
    let code = [byte, 0x00, 0x09, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = tc.record_op(majit_ir::OpCode::New, &[]);
    tc.heap_cache_mut().new_object(operand);
    let ops_before = tc.num_ops();
    let mut regs_r = [operand];
    // The Ref shadow is absent, yet the heapcache already proved non-nullness.
    // `_establish_nullity` answers from the cache before demanding a pointer.
    let mut concrete_r = [ConcreteValue::Null];
    let (outcome, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("a heapcache-known non-null box needs no ref shadow");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 4,
        "a known non-null pointer falls through without reading a shadow"
    );
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "the heapcache-known fast path emits no nullity guard"
    );
}

#[test]
fn raw_load_i_records_the_load_against_the_descr() {
    let byte = *insns_opname_to_byte()
        .get("raw_load_i/iid>i")
        .expect("`raw_load_i/iid>i` must be in insns table");
    // `iid>i`: 1B base + 1B offset + 2B descr + 1B dst.
    let code = [byte, 0x00, 0x01, 0x00, 0x00, 0x02];
    let descr_pool = vec![crate::descr::w_int_size_descr()];
    let mut tc = fresh_trace_ctx();
    let base = tc.const_int(0x1000);
    let offset = tc.const_int(8);
    let mut regs_i = [base, offset, OpRef::NONE];
    let (outcome, next_pc) =
        run_hint_step_with_descrs(&code, &mut tc, &mut [], &mut [], &mut regs_i, &descr_pool)
            .expect("`raw_load_i/iid>i` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 6,
        "`iid>i` consumes 3 register bytes plus a 2B descr"
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::RawLoadI);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![base, offset],
        "raw loads record the address pair only; the descr rides alongside",
    );
    assert_eq!(
        regs_i[2],
        last.pos().get(),
        "the `>i` decorator writes the dst"
    );
}

/// JUMP_BACKWARD's eval-breaker tick is `raw_load_i` of a constant address
/// through the descr `add_raw_int_array_descr_signed` interns, then
/// `uint_ge` against `JIT_BREAKER_FLOOR`, then `goto_if_not`.
/// `pyjitpl.py MIFrame.opimpl_raw_load_i` runs `execute_with_descr`, which
/// executes the load so the result box carries the word (`box.getint()`).
/// The walk must stamp that word onto the recorded op the same way, or
/// `uint_ge`/`goto_if_not` see a Null shadow and abort with
/// `GotoIfNotValueNotConcrete`.
#[test]
fn raw_load_i_stamps_the_loaded_word_onto_the_result_box() {
    let load_byte = *insns_opname_to_byte()
        .get("raw_load_i/iid>i")
        .expect("`raw_load_i/iid>i` must be in insns table");
    // `iid>i`: 1B base + 1B offset + 2B descr + 1B dst.
    let code = [load_byte, 0x00, 0x01, 0x00, 0x00, 0x02];
    let item_size = majit_ir::eval_breaker_word::EVAL_BREAKER_WORD_SIZE;
    let descr = crate::descr::raw_carray_descr(majit_ir::Type::Int, item_size, true);
    assert!(
        descr.as_array_descr().is_some(),
        "the interned raw-int array descr must answer `as_array_descr`, \
         matching `add_raw_int_array_descr_signed`"
    );
    let descr_pool = vec![descr];
    let mut cell = [0u8; 16];
    // Unarmed eval-breaker word: below `JIT_BREAKER_FLOOR`.
    let loaded: i64 = 0;
    unsafe {
        std::ptr::write_unaligned(cell.as_mut_ptr().cast::<i64>(), loaded);
    }
    let addr = cell.as_ptr() as i64;
    let mut tc = fresh_trace_ctx();
    let base = tc.const_int(addr);
    let offset = tc.const_int(0);
    let mut regs_i = [base, offset, OpRef::NONE];
    let mut concrete_i = [
        ConcreteValue::Int(addr),
        ConcreteValue::Int(0),
        ConcreteValue::Null,
    ];
    let (outcome, next_pc) = run_hint_step_full(
        &code,
        &mut tc,
        &mut [],
        &mut [],
        &mut regs_i,
        &mut concrete_i,
        &mut [],
        &descr_pool,
    )
    .expect("`raw_load_i/iid>i` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 6,
        "`iid>i` consumes 3 register bytes plus a 2B descr"
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::RawLoadI);
    assert_eq!(
        concrete_i[2],
        ConcreteValue::Int(loaded),
        "the walk executes the load into the Int-bank shadow"
    );
    assert_eq!(
        tc.concrete_of_opref(regs_i[2]),
        Some(Value::Int(loaded)),
        "pyjitpl.py MIFrame.opimpl_raw_load_i stamps box.value via execute_with_descr"
    );
}

/// JUMP_BACKWARD's eval-breaker `goto_if_not` keeps both arms on that
/// opcode, so `get_list_of_active_boxes` (`pyjitpl.py`) reads the
/// terminator's own `-live-`.  A `FOR_ITER` not-taken arm is a different
/// opcode.  A kept-stack `POP_JUMP_IF_*` / short-circuit trampoline can
/// share the guard's own Python opcode and must NOT take this path.
#[test]
fn tick_goto_if_not_stays_in_the_jump_backward_opcode() {
    // Measured on `simple.py` `def f`: poll `goto_if_not` at jit 564,
    // not-taken arm at 568, both exact-py JUMP_BACKWARD.
    assert!(
        super::resume_snapshot::branch_not_taken_stays_in_guard_opcode(Some(32), Some(32), true),
        "the poll's not-taken arm is still JUMP_BACKWARD",
    );
    // FOR_ITER `goto_if_not` at jit 302, not-taken arm at 592 (END_FOR).
    assert!(
        !super::resume_snapshot::branch_not_taken_stays_in_guard_opcode(Some(14), Some(36), true),
        "FOR_ITER's exhausted arm is a different Python opcode",
    );
    assert!(
        !super::resume_snapshot::branch_not_taken_stays_in_guard_opcode(Some(14), None, true),
        "a missing other-arm py_pc is not the same-opcode poll",
    );
    assert!(
        !super::resume_snapshot::branch_not_taken_stays_in_guard_opcode(Some(8), Some(8), false),
        "a same-opcode POP_JUMP_IF_* / short-circuit branch is not the tick",
    );
}

/// The walk executes a raw store as well as recording it, so the target has
/// to be memory the test owns and every operand has to carry a concrete —
/// `_opimpl_raw_store` goes through `execute_and_record`, and skipping the
/// store would hand the rest of the walked body a buffer nothing filled.
#[test]
fn raw_store_i_records_the_store_and_performs_it() {
    let byte = *insns_opname_to_byte()
        .get("raw_store_i/iiid")
        .expect("`raw_store_i/iiid` must be in insns table");
    // `iiid`: 1B base + 1B offset + 1B value + 2B descr.
    let code = [byte, 0x00, 0x01, 0x02, 0x00, 0x00];
    let descr_pool = vec![crate::descr::raw_carray_descr(majit_ir::Type::Int, 8, true)];
    let mut cell = [0u8; 16];
    let addr = cell.as_mut_ptr() as i64;
    let mut tc = fresh_trace_ctx();
    let base = tc.const_int(addr);
    let offset = tc.const_int(8);
    let value = tc.const_int(42);
    let mut regs_i = [base, offset, value];
    let mut concrete_i = [
        ConcreteValue::Int(addr),
        ConcreteValue::Int(8),
        ConcreteValue::Int(42),
    ];
    let (outcome, next_pc) = run_hint_step_full(
        &code,
        &mut tc,
        &mut [],
        &mut [],
        &mut regs_i,
        &mut concrete_i,
        &mut [],
        &descr_pool,
    )
    .expect("`raw_store_i/iiid` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 6,
        "`iiid` consumes 3 register bytes plus a 2B descr"
    );
    // SAFETY: the slot the store just wrote, at the descr's item width.
    let stored = unsafe { std::ptr::read_unaligned(cell.as_ptr().add(8).cast::<i64>()) };
    assert_eq!(stored, 42, "the walk performs the store it records");
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::RawStore);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![base, offset, value],
    );
    assert_eq!(
        regs_i,
        [base, offset, value],
        "a raw store leaves every register untouched"
    );
}

#[test]
fn setarrayitem_raw_i_records_the_store_and_performs_it() {
    // `setarrayitem_raw_i/iiid` — pyjitpl.py `_opimpl_setarrayitem_raw_any`
    // / `opimpl_setarrayitem_raw_i` via `execute_with_descr(SETARRAYITEM_RAW)`.
    // Operand layout `iiid`: 1B array + 1B index + 1B value + 2B descr.
    let byte = *insns_opname_to_byte()
        .get("setarrayitem_raw_i/iiid")
        .expect("`setarrayitem_raw_i/iiid` must be in insns table");
    let code = [byte, 0x00, 0x01, 0x02, 0x00, 0x00];
    let descr_pool = vec![crate::descr::raw_carray_descr(majit_ir::Type::Int, 8, true)];
    let mut cell = [0u8; 16];
    let addr = cell.as_mut_ptr() as i64;
    let mut tc = fresh_trace_ctx();
    let array = tc.const_int(addr);
    let index = tc.const_int(1);
    let value = tc.const_int(42);
    let mut regs_i = [array, index, value];
    let mut concrete_i = [
        ConcreteValue::Int(addr),
        ConcreteValue::Int(1),
        ConcreteValue::Int(42),
    ];
    let (outcome, next_pc) = run_hint_step_full(
        &code,
        &mut tc,
        &mut [],
        &mut [],
        &mut regs_i,
        &mut concrete_i,
        &mut [],
        &descr_pool,
    )
    .expect("`setarrayitem_raw_i/iiid` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 6,
        "`iiid` consumes 3 register bytes plus a 2B descr"
    );
    // `raw_carray_descr` has `base_size == 0`, so index 1 lands at +8.
    let stored = unsafe { std::ptr::read_unaligned(cell.as_ptr().add(8).cast::<i64>()) };
    assert_eq!(stored, 42, "the walk performs the store it records");
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::SetarrayitemRaw);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![array, index, value],
        "SetarrayitemRaw args must be [array, index, value]",
    );
    assert_eq!(
        regs_i,
        [array, index, value],
        "a raw array store leaves every register untouched"
    );
}

/// Float twin of the above: the value comes from the Float bank and the
/// descr names a `CArray(Float)`.
#[test]
fn raw_store_f_records_the_store_and_performs_it() {
    let byte = *insns_opname_to_byte()
        .get("raw_store_f/iifd")
        .expect("`raw_store_f/iifd` must be in insns table");
    // `iifd`: 1B base + 1B offset + 1B float-bank value + 2B descr.
    let code = [byte, 0x00, 0x01, 0x00, 0x00, 0x00];
    let descr_pool = vec![crate::descr::raw_carray_descr(
        majit_ir::Type::Float,
        8,
        false,
    )];
    let mut cell = [0u8; 16];
    let addr = cell.as_mut_ptr() as i64;
    let mut tc = fresh_trace_ctx();
    let base = tc.const_int(addr);
    let offset = tc.const_int(8);
    let value = tc.const_float(1.5f64.to_bits() as i64);
    let mut regs_i = [base, offset];
    let mut concrete_i = [ConcreteValue::Int(addr), ConcreteValue::Int(8)];
    let mut regs_f = [value];
    let (outcome, next_pc) = run_hint_step_full(
        &code,
        &mut tc,
        &mut [],
        &mut [],
        &mut regs_i,
        &mut concrete_i,
        &mut regs_f,
        &descr_pool,
    )
    .expect("`raw_store_f/iifd` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 6,
        "`iifd` consumes 3 register bytes plus a 2B descr"
    );
    // SAFETY: the slot the store just wrote, at the descr's item width.
    let stored = unsafe { std::ptr::read_unaligned(cell.as_ptr().add(8).cast::<f64>()) };
    assert_eq!(stored, 1.5, "the walk performs the store it records");
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::RawStore);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![base, offset, value],
    );
    assert_eq!(
        regs_f,
        [value],
        "a raw store leaves every register untouched"
    );
}

#[test]
fn record_exact_class_records_the_hint_with_both_operands() {
    let byte = *insns_opname_to_byte()
        .get("record_exact_class/ri")
        .expect("`record_exact_class/ri` must be in insns table");
    // `ri`: 1B ref reg + 1B int reg holding the class vtable address.
    let code = [byte, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = dummy_ptr_eq_box(&mut tc);
    let cls = tc.const_int(0x4000);
    let mut regs_r = [operand];
    let mut regs_i = [cls];
    let mut concrete_r = [ConcreteValue::Null];
    let (outcome, next_pc) =
        run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut regs_i)
            .expect("`record_exact_class/ri` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 3, "`ri` consumes two register bytes");
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::RecordExactClass);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![operand, cls],
    );
}

#[test]
fn switch_id_jumps_to_target_or_falls_through() {
    let switch_byte = *insns_opname_to_byte()
        .get("switch/id")
        .expect("`switch/id` must be in insns table");
    let code = [
        switch_byte,
        0x00, // i register 0
        0x00,
        0x00, // d descr index 0
    ];
    let cases = [("hit", 5_i64, Some(17_usize)), ("miss", 7, None)];
    for (name, key, jump_to) in cases {
        let mut tc = fresh_trace_ctx();
        let value = tc.const_int(key);
        let regs_i = vec![value];
        let descr_pool = switch_descr_pool(&[(5, 17), (9, 23)]);
        let _descr = done_descr_ref_for_tests();
        let session = std::cell::RefCell::new(WalkSession::default());
        let mut wc = WalkContext {
            frame_state: WalkFrameState::new(WalkFrameStateData {
                callee_shadow: None,
                concrete_registers_r: ([]).to_vec(),
                outer_active_boxes: Vec::new(),
                vstack_boxes: Vec::new(),
                vstack_last_ref: OpRef::NONE,
                vstack_reorder_saved: None,
                ..Default::default()
            }),
            inline_callee_consts: None,
            fbw_mode: test_fbw_mode(),
            session: &session,
            registers_r: &RegisterBank::default(),
            registers_i: &RegisterBank::new(regs_i.iter().copied()),
            registers_f: &RegisterBank::default(),

            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            raw_descrs: RawDescrPool::Global,
            is_authoritative_executor: false,
            trace_ctx: &mut tc,
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            entry_py_pc: EntryPyPc::Py(0),
            outer_resume_marker_jit_pc: None,
            outer_jitcode_index: 0,

            pending_guard_snapshot_error: None,

            vstack_depth: 0,
            vstack_cur_pypc: 0,
            vstack_valid: false,

            vstack_reorder_ceiling: u32::MAX,

            vstack_handler_landing_py: None,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
        };

        let stepped = step(&code, 0, &mut wc);
        assert!(stepped.is_ok(), "case {name}");
        let (outcome, next_pc) = stepped.unwrap();
        assert_eq!(outcome, DispatchOutcome::Continue, "case {name}");
        assert_eq!(next_pc, jump_to.unwrap_or(code.len()), "case {name}");
    }
}

#[test]
fn switch_id_requires_concrete_int_value() {
    let switch_byte = *insns_opname_to_byte()
        .get("switch/id")
        .expect("`switch/id` must be in insns table");
    let code = [
        switch_byte,
        0x00, // i register 0
        0x00,
        0x00, // d descr index 0
    ];
    let mut tc = fresh_trace_ctx();
    let regs_i = vec![OpRef::input_arg_int(0)];
    let descr_pool = switch_descr_pool(&[(5, 17)]);
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    let err = step(&code, 0, &mut wc).expect_err("non-constant switch value must not guess");

    assert_eq!(
        err,
        DispatchError::SwitchValueNotConcrete {
            pc: 0,
            value: OpRef::input_arg_int(0),
        }
    );
}

#[test]
fn goto_if_not_truthy_records_guard_true_and_falls_through() {
    // `goto_if_not/iL` with a concrete non-zero Int: emit GuardTrue,
    // do NOT take the jump (pc advances past the 3-byte operand
    // block).  RPython `pyjitpl.py opimpl_goto_if_not`
    // `if switchcase: opnum = rop.GUARD_TRUE; ... if not switchcase: self.pc = target`.
    let goto_if_byte = *insns_opname_to_byte()
        .get("goto_if_not/iL")
        .expect("`goto_if_not/iL` must be in insns table");
    let code = [
        goto_if_byte,
        0x00, // i register 0
        0x40,
        0x00, // L target = 0x0040
    ];
    let mut tc = fresh_trace_ctx();
    let value = tc.const_int(1);
    let regs_i = vec![value];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("truthy branch must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, code.len(), "truthy branch falls through");
}

#[test]
fn goto_if_not_falsy_records_guard_false_and_jumps() {
    // `goto_if_not/iL` with a concrete zero Int: emit GuardFalse,
    // jump to the label target (pc = target).
    let goto_if_byte = *insns_opname_to_byte()
        .get("goto_if_not/iL")
        .expect("`goto_if_not/iL` must be in insns table");
    let code = [
        goto_if_byte,
        0x00, // i register 0
        0x40,
        0x00, // L target = 0x0040
    ];
    let mut tc = fresh_trace_ctx();
    let value = tc.const_int(0);
    let regs_i = vec![value];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("falsy branch must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 0x0040, "falsy branch jumps to label target");
}

#[test]
fn goto_if_not_requires_concrete_int_value() {
    // Non-constant symbolic OpRef has no concrete: must surface
    // `GotoIfNotValueNotConcrete` rather than guess a branch.
    let goto_if_byte = *insns_opname_to_byte()
        .get("goto_if_not/iL")
        .expect("`goto_if_not/iL` must be in insns table");
    let code = [
        goto_if_byte,
        0x00, // i register 0
        0x40,
        0x00, // L target = 0x0040
    ];
    let mut tc = fresh_trace_ctx();
    let regs_i = vec![OpRef::input_arg_int(0)];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    let err = step(&code, 0, &mut wc).expect_err("non-constant branch value must not guess");

    assert_eq!(
        err,
        DispatchError::GotoIfNotValueNotConcrete {
            pc: 0,
            value: OpRef::input_arg_int(0),
        }
    );
}

/// Production-like `sub_jitcode_lookup` that resolves `idx` against
/// `crate::jitcode_runtime::all_jitcodes()`. Used by the end-to-end
/// helper acceptance tests (`walk_return_value_helper_*`,
/// `walk_pop_top_helper_*`) so the walker can recurse into real
/// callee bodies. Delegates to the production
/// [`super::sub_jitcode_body_by_index`].
fn production_sub_jitcodes(idx: usize) -> Option<SubJitCodeBody> {
    super::sub_jitcode_body_by_index(idx)
}

#[test]
fn sub_jitcode_body_by_index_builds_w_list_append() {
    // The shared by-index `SubJitCodeBody` builder must resolve a
    // build-time charon body (`w_list_append`) to a well-formed body —
    // non-empty bytecode and >= 2 ref registers for the (list, value)
    // params (calldescr arg_classes 'rr').  This body is descended by the
    // shipping `lst.append` arm (`try_walker_orthodox_list_append`).
    let idx = crate::jitcode_runtime::list_append_jitcode()
        .expect("w_list_append must be present in ALL_JITCODES")
        .index();
    let body =
        super::sub_jitcode_body_by_index(idx).expect("by-index builder must resolve w_list_append");
    assert!(
        !body.code.is_empty(),
        "w_list_append body must carry assembled bytecode"
    );
    assert!(
        body.num_regs_r >= 2,
        "w_list_append takes (list, value) => >= 2 ref registers, got {}",
        body.num_regs_r
    );
    // Out-of-range index resolves to None (RPython: ALL_JITCODES miss).
    assert!(super::sub_jitcode_body_by_index(usize::MAX).is_none());
}

#[test]
fn int_truediv_and_newfloat_jitcodes_are_the_pypy_leaf() {
    // intobject.py `_truediv` + `space.newfloat`: the `/` descent walks
    // `int_truediv`, which looks inside `newfloat`.  `fuse_boxing_alloc`
    // must have rewritten that constructor to `new_with_vtable` so the
    // walk records instantiate + setfield, not residual `malloc_typed`.
    let truediv = crate::jitcode_runtime::pathed_jitcode(
        "pyre_interpreter::objspace::descroperation::_truediv",
    )
    .expect("_truediv must be a discovered jitcode");
    let newfloat = crate::jitcode_runtime::pathed_jitcode("pyre_object::floatobject::newfloat")
        .expect("newfloat must be a discovered jitcode");
    let truediv_ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&truediv.code)
        .map(|op| op.opname)
        .collect();
    let newfloat_ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&newfloat.code)
        .map(|op| op.opname)
        .collect();
    assert!(
        truediv_ops.iter().any(|op| *op == "cast_int_to_float"),
        "_truediv must lower `float(x)` to cast_int_to_float; ops={truediv_ops:?}"
    );
    assert!(
        !truediv_ops
            .iter()
            .any(|op| op.starts_with("residual_call_irf")),
        "_truediv must not residualize sitofp; ops={truediv_ops:?}"
    );
    assert!(
        truediv_ops.iter().any(|op| *op == "new_with_vtable"),
        "_truediv must box via in-graph new_with_vtable; ops={truediv_ops:?}"
    );
    assert!(
        !truediv_ops.iter().any(|op| op.starts_with("inline_call")),
        "_truediv must not inline_call newfloat; ops={truediv_ops:?}"
    );
    assert!(
        newfloat_ops.iter().any(|op| *op == "new_with_vtable"),
        "newfloat must lower to new_with_vtable; ops={newfloat_ops:?}"
    );
    assert!(
        !newfloat_ops
            .iter()
            .any(|op| op.contains("residual") || *op == "residual_call"),
        "fused newfloat must not residualise malloc; ops={newfloat_ops:?}"
    );
    assert!(
        newfloat_ops.len() < 32,
        "fused newfloat is New+setfields, not malloc_typed; ops={newfloat_ops:?}"
    );
    eprintln!("int_truediv {} ops: {truediv_ops:?}", truediv_ops.len());
    eprintln!("newfloat {} ops: {newfloat_ops:?}", newfloat_ops.len());
    let dump_calls = |code: &[u8], label: &str| {
        let mut pc = 0;
        while let Some(op) = crate::jitcode_runtime::decode_op_at(code, pc) {
            if op.opname.contains("call") {
                let mut cursor = op.pc + 1;
                let mut descr_idx = None;
                for c in op.argcodes.chars() {
                    match c {
                        'i' | 'c' | 'r' | 'f' => cursor += 1,
                        'L' | 'j' => cursor += 2,
                        'd' => {
                            let lo = code[cursor] as u16;
                            let hi = code[cursor + 1] as u16;
                            descr_idx = Some((lo | (hi << 8)) as usize);
                            cursor += 2;
                        }
                        'I' | 'R' | 'F' => {
                            let n = code[cursor] as usize;
                            cursor += 1 + n;
                        }
                        '>' => cursor += 1,
                        _ => {}
                    }
                }
                let name = descr_idx
                    .and_then(crate::jitcode_runtime::get_descr_by_index)
                    .map(|d| match d {
                        majit_jitcode::jitcode::BhDescr::JitCode { jitcode_index, .. } => {
                            crate::jitcode_runtime::get_jitcode_ref_by_index(*jitcode_index)
                                .map(|jc| format!("jitcode:{}", jc.name))
                                .unwrap_or_else(|| format!("jitcode_idx:{jitcode_index}"))
                        }
                        majit_jitcode::jitcode::BhDescr::Call { calldescr } => {
                            format!("call:{calldescr:?}")
                        }
                        other => format!("{other:?}"),
                    })
                    .unwrap_or_else(|| "?".into());
                eprintln!(
                    "{label} pc={} {} descr={descr_idx:?} -> {name}",
                    op.pc, op.key
                );
            }
            pc = op.next_pc;
        }
    };
    dump_calls(&truediv.code, "int_truediv");
    dump_calls(&newfloat.code, "newfloat");
    {
        let mut pc = 0;
        while let Some(op) = crate::jitcode_runtime::decode_op_at(&truediv.code, pc) {
            let mut cursor = op.pc + 1;
            let mut labels = Vec::new();
            for c in op.argcodes.chars() {
                match c {
                    'i' | 'c' | 'r' | 'f' => cursor += 1,
                    'd' | 'j' => cursor += 2,
                    'L' => {
                        let lo = truediv.code[cursor] as u16;
                        let hi = truediv.code[cursor + 1] as u16;
                        labels.push(lo | (hi << 8));
                        cursor += 2;
                    }
                    'I' | 'R' | 'F' => {
                        let n = truediv.code[cursor] as usize;
                        cursor += 1 + n;
                    }
                    '>' => cursor += 1,
                    _ => {}
                }
            }
            eprintln!("  [{:>3}] {} labels={labels:?}", op.pc, op.key);
            pc = op.next_pc;
        }
    }
    let int_value = crate::jitcode_runtime::pathed_jitcode(
        "pyre_interpreter::objspace::descroperation::int_value",
    )
    .or_else(|| crate::jitcode_runtime::named_jitcode("int_value"))
    .expect("int_value jitcode");
    let int_value_ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&int_value.code)
        .map(|op| op.opname)
        .collect();
    eprintln!("int_value {} ops: {int_value_ops:?}", int_value_ops.len());
    dump_calls(&int_value.code, "int_value");
    for path in [
        "pyre_object::pyobject::is_bool",
        "pyre_object::intobject::w_int_get_value",
        "pyre_object::boolobject::w_bool_get_value",
        "pyre_object::pyobject::py_type_check",
    ] {
        let Some(jc) = crate::jitcode_runtime::pathed_jitcode(path)
            .or_else(|| crate::jitcode_runtime::named_jitcode(path.rsplit("::").next().unwrap()))
        else {
            eprintln!("missing {path}");
            continue;
        };
        let ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&jc.code)
            .map(|op| op.opname)
            .collect();
        eprintln!("{path} {} ops: {ops:?}", ops.len());
        dump_calls(&jc.code, path.rsplit("::").next().unwrap());
    }
    assert!(
        truediv_ops.len() < 80,
        "_truediv must stay the unboxed leaf, not the dispatcher; ops={truediv_ops:?}"
    );
    assert!(
        !truediv_ops
            .iter()
            .any(|op| op.starts_with("inline_call_r_i")),
        "_truediv takes unboxed ints; no int_value walk; ops={truediv_ops:?}"
    );
    {
        let mut pc = 0;
        while let Some(op) = crate::jitcode_runtime::decode_op_at(&truediv.code, pc) {
            if op.opname == "goto" || op.opname.starts_with("goto_if") {
                let mut cursor = op.pc + 1;
                for c in op.argcodes.chars() {
                    match c {
                        'i' | 'c' | 'r' | 'f' => cursor += 1,
                        'd' | 'j' => cursor += 2,
                        'L' => {
                            let lo = truediv.code[cursor] as u16;
                            let hi = truediv.code[cursor + 1] as u16;
                            let target = (lo | (hi << 8)) as usize;
                            assert!(
                                target >= op.next_pc,
                                "_truediv must not back-edge ({opname} {pc} → {target}); that CFG hung the helper walk",
                                opname = op.opname,
                                pc = op.pc,
                            );
                            cursor += 2;
                        }
                        'I' | 'R' | 'F' => {
                            let n = truediv.code[cursor] as usize;
                            cursor += 1 + n;
                        }
                        '>' => cursor += 1,
                        _ => {}
                    }
                }
            }
            pc = op.next_pc;
        }
    }
}

#[test]
fn newcomplex_and_lane_getters_are_the_pypy_leaf() {
    // complexobject.py `descr__new__` allocates `W_ComplexObject(real, imag)`.
    // `complexwprop` then boxes one lane with `space.newfloat`. Both leaves
    // have to be their own jitcode: the `complex()` walk descends `newcomplex`,
    // and `LOAD_ATTR real`/`imag` descends the getter.
    let newcomplex =
        crate::jitcode_runtime::pathed_jitcode("pyre_object::complexobject::newcomplex")
            .expect("newcomplex must be a discovered jitcode");
    let newcomplex_ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&newcomplex.code)
        .map(|op| op.opname)
        .collect();
    assert!(
        newcomplex_ops.iter().any(|op| *op == "new_with_vtable"),
        "newcomplex must lower to new_with_vtable; ops={newcomplex_ops:?}"
    );
    assert!(
        !newcomplex_ops
            .iter()
            .any(|op| op.contains("residual") || *op == "residual_call"),
        "fused newcomplex must not residualise malloc; ops={newcomplex_ops:?}"
    );
    assert!(
        newcomplex_ops.len() < 32,
        "fused newcomplex is New+setfields, not malloc_typed; ops={newcomplex_ops:?}"
    );
    for path in [
        "pyre_object::complexobject::complex_descr_get_real",
        "pyre_object::complexobject::complex_descr_get_imag",
    ] {
        let getter = crate::jitcode_runtime::pathed_jitcode(path)
            .unwrap_or_else(|| panic!("{path} must be a discovered jitcode"));
        assert!(
            !getter.code.is_empty(),
            "{path} must carry assembled bytecode"
        );
    }
}

#[test]
fn newutf8_and_int_descr_str_jitcodes_are_the_pypy_leaf() {
    // `space.newutf8` / `W_UnicodeObject.__init__` (`objspace.py` /
    // `unicodeobject.py`): field stores after `malloc_typed_managed`,
    // which `fuse_boxing_alloc` rewrites to `new_with_vtable`.
    // `intobject.py descr_str` is `ll_int2dec` then that wrap.
    let newutf8 = crate::jitcode_runtime::newutf8_jitcode()
        .expect("w_str_from_storage_and_length must be a discovered jitcode");
    let descr_str = crate::jitcode_runtime::int_descr_str_jitcode()
        .expect("int descr_str must be a discovered jitcode");
    let dump = |code: &[u8], label: &str| {
        let mut pc = 0;
        while let Some(op) = crate::jitcode_runtime::decode_op_at(code, pc) {
            eprintln!("  {label} [{:>3}] {}", op.pc, op.key);
            pc = op.next_pc;
        }
    };
    let newutf8_body = super::sub_jitcode_body_by_index(newutf8.index()).expect("newutf8 body");
    let newutf8_ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&newutf8.code)
        .map(|op| op.opname)
        .collect();
    eprintln!(
        "newutf8 i={} r={} f={} {} ops: {newutf8_ops:?}",
        newutf8_body.num_regs_i,
        newutf8_body.num_regs_r,
        newutf8_body.num_regs_f,
        newutf8_ops.len()
    );
    dump(&newutf8.code, "newutf8");
    let descr_str_body =
        super::sub_jitcode_body_by_index(descr_str.index()).expect("descr_str body");
    let descr_str_ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&descr_str.code)
        .map(|op| op.opname)
        .collect();
    eprintln!(
        "descr_str i={} r={} f={} {} ops: {descr_str_ops:?}",
        descr_str_body.num_regs_i,
        descr_str_body.num_regs_r,
        descr_str_body.num_regs_f,
        descr_str_ops.len()
    );
    dump(&descr_str.code, "descr_str");
    // `ll_int2dec` is the one residual that returns the STR; `len(res)` is
    // `LLHelpers.ll_strlen`, a field read off that STR, and the result is
    // wrapped by the inlined newutf8 body.
    assert_eq!(
        descr_str_ops
            .iter()
            .filter(|op| op.starts_with("residual_call_") && op.ends_with("_r"))
            .count(),
        1,
        "descr_str must residualize only ll_int2dec as a ref call; ops={descr_str_ops:?}"
    );
    let strlen_at = descr_str_ops
        .iter()
        .position(|op| *op == "getfield_gc_i")
        .unwrap_or_else(|| {
            panic!("descr_str must read the STR length field; ops={descr_str_ops:?}")
        });
    let int2dec_at = descr_str_ops
        .iter()
        .position(|op| op.starts_with("residual_call_") && op.ends_with("_r"))
        .unwrap();
    assert!(
        int2dec_at < strlen_at,
        "the length read must follow ll_int2dec; ops={descr_str_ops:?}"
    );
    assert!(
        !descr_str_ops[int2dec_at + 1..]
            .iter()
            .any(|op| op.starts_with("residual_call_") && op.ends_with("_i")),
        "the STR length must not be a Signed residual; ops={descr_str_ops:?}"
    );
    assert!(
        descr_str_ops[strlen_at..]
            .iter()
            .any(|op| op.starts_with("inline_call_") && op.ends_with("_r")),
        "descr_str must wrap through the inlined newutf8 body; ops={descr_str_ops:?}"
    );
    assert!(
        newutf8_ops.iter().any(|op| *op == "new_with_vtable"),
        "newutf8 must box via new_with_vtable; ops={newutf8_ops:?}"
    );
    assert!(
        !newutf8_ops
            .iter()
            .any(|op| op.contains("residual") || *op == "residual_call"),
        "fused newutf8 must not residualise malloc/strlen; ops={newutf8_ops:?}"
    );
    assert!(
        newutf8_ops.len() < 32,
        "fused newutf8 is New+setfields, not malloc_typed; ops={newutf8_ops:?}"
    );
}

#[test]
fn format_int_decimal_jitcode_is_the_fill_number_leaf() {
    // `newformat.py` `format_int_or_long`: `ll_int2dec`, then
    // `_fill_number` (`n_lpadding` / `n_spadding` / `n_rpadding`), then
    // `space.newutf8`. Banks are int then ref, so
    // `(value, fill, width, align, forced_sign)` is `iriir`.
    let jc = crate::jitcode_runtime::pathed_jitcode("pyre_object::intobject::format_int_decimal")
        .expect("format_int_decimal must be a discovered jitcode");
    let body = jc
        .try_body()
        .expect("format_int_decimal body must be assembled");
    let ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&jc.code)
        .map(|op| op.opname)
        .collect();
    assert_eq!(
        body.calldescr.arg_classes, "iriir",
        "format_int_decimal arg banks; ops={ops:?}"
    );
    assert_eq!(body.calldescr.result_type, 'r', "ops={ops:?}");
    assert!(
        ops.iter().any(|op| *op == "getfield_gc_i"),
        "the STR length is a field read; ops={ops:?}"
    );
    assert!(
        ops.iter()
            .any(|op| op.starts_with("residual_call_") && op.ends_with("_r")),
        "ll_int2dec stays a ref residual; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| {
            (*op == "new_with_vtable") || (op.starts_with("inline_call_") && op.ends_with("_r"))
        }),
        "the wrap is inlined newutf8; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| op.starts_with("goto_if_not")),
        "sign split and padding stay in the body; ops={ops:?}"
    );
    let sign_split = ops
        .iter()
        .position(|op| op.starts_with("goto_if_not"))
        .expect("sign split");
    assert!(
        !ops[..sign_split]
            .iter()
            .any(|op| *op == "residual_call_r_r"),
        "ptr::null_mut of the STR sign must be a null constant, not a residual; ops={ops:?}"
    );
    assert!(
        !ops.iter()
            .any(|op| op.contains("cast_uint_to_int") || op.contains("cast_int_to_uint")),
        "usize/i64 length casts must erase; ops={ops:?}"
    );
}

#[test]
fn zip_two_tuple_next_jitcode_is_the_tuple_iter_leaf() {
    // `functional.py` `W_Zip.next_w` arity two, specialised to
    // `W_FastTupleIterObject.descr_next` over `W_TupleObject`. The product
    // is `w_specialised_tuple_oo_new` (`fuse_boxing_alloc` → `new_with_vtable`).
    let zip = crate::jitcode_runtime::pathed_jitcode(
        "pyre_interpreter::baseobjspace::zip_two_tuple_next",
    )
    .expect("zip_two_tuple_next must be a discovered jitcode");
    let body = zip
        .try_body()
        .expect("zip_two_tuple_next body must be assembled");
    let ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&zip.code)
        .map(|op| op.opname)
        .collect();
    assert_eq!(body.calldescr.arg_classes, "r", "ops={ops:?}");
    assert_eq!(body.calldescr.result_type, 'r', "ops={ops:?}");
    // The item load is `tuple_exact_item`: `wrappeditems` is `[*]`, so the
    // read is `getarrayitem_gc_r_pure`, not a residual `w_tuple_getitem`.
    let item =
        crate::jitcode_runtime::pathed_jitcode("pyre_interpreter::baseobjspace::tuple_exact_item")
            .expect("tuple_exact_item must be a discovered jitcode");
    let item_ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&item.code)
        .map(|op| op.opname)
        .collect();
    assert!(
        item_ops.iter().any(|op| *op == "getarrayitem_gc_r_pure"),
        "tuple item load must be a pure getarrayitem; ops={item_ops:?}"
    );
    assert!(
        !item_ops.iter().any(|op| op.contains("residual")),
        "tuple_exact_item must not residualise the load; ops={item_ops:?}"
    );
    let oo = crate::jitcode_runtime::pathed_jitcode(
        "pyre_object::specialisedtupleobject::w_specialised_tuple_oo_new",
    )
    .expect("w_specialised_tuple_oo_new must be a discovered jitcode");
    let oo_ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&oo.code)
        .map(|op| op.opname)
        .collect();
    assert!(
        oo_ops.iter().any(|op| *op == "new_with_vtable"),
        "the pair must box via new_with_vtable; ops={oo_ops:?}"
    );
    assert!(
        !oo_ops.iter().any(|op| op.contains("residual")),
        "w_specialised_tuple_oo_new must not residualise malloc; ops={oo_ops:?}"
    );
    assert!(
        !(ops.len() == 1 && ops[0].contains("residual")),
        "zip_two_tuple_next collapsed to one residual; ops={ops:?}"
    );
}

#[test]
fn specialised_tuple_ii_new_jitcode_is_new_with_vtable() {
    // `makespecialisedtuple2` `Cls_ii` (`specialisedtupleobject.py`).
    // `fuse_boxing_alloc` rewrites `malloc_typed_managed` to
    // `new_with_vtable` plus the `value0` / `value1` stores.
    let ii = crate::jitcode_runtime::pathed_jitcode(
        "pyre_object::specialisedtupleobject::w_specialised_tuple_ii_new",
    )
    .expect("w_specialised_tuple_ii_new must be a discovered jitcode");
    let body = ii
        .try_body()
        .expect("w_specialised_tuple_ii_new body must be assembled");
    let ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&ii.code)
        .map(|op| op.opname)
        .collect();
    assert_eq!(body.calldescr.arg_classes, "ii", "ops={ops:?}");
    assert_eq!(body.calldescr.result_type, 'r', "ops={ops:?}");
    assert!(
        ops.iter().any(|op| *op == "new_with_vtable"),
        "the pair must box via new_with_vtable; ops={ops:?}"
    );
    assert!(
        ops.iter().filter(|op| **op == "setfield_gc_i").count() >= 2,
        "value0 and value1 are int stores; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| op.contains("residual")),
        "w_specialised_tuple_ii_new must not residualise malloc; ops={ops:?}"
    );
}

#[test]
fn float_binop_leaves_are_the_pypy_leaf() {
    // floatobject.py `descr_add` after `_to_float`: `W_FloatObject(x + y)`.
    // The descent walks `_float_add`, which must contain the fused New.
    for (path, arith) in [
        (
            "pyre_interpreter::objspace::descroperation::_float_add",
            "float_add",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_sub",
            "float_sub",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_mul",
            "float_mul",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_truediv",
            "float_truediv",
        ),
    ] {
        let jc = crate::jitcode_runtime::pathed_jitcode(path)
            .unwrap_or_else(|| panic!("{path} must be a discovered jitcode"));
        let ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&jc.code)
            .map(|op| op.opname)
            .collect();
        assert!(
            ops.iter().any(|op| *op == arith),
            "{path} must record {arith}; ops={ops:?}"
        );
        assert!(
            ops.iter().any(|op| *op == "new_with_vtable"),
            "{path} must box via in-graph new_with_vtable; ops={ops:?}"
        );
        assert!(
            !ops.iter().any(|op| op.starts_with("inline_call")),
            "{path} must not inline_call newfloat; ops={ops:?}"
        );
        assert!(
            !ops.iter().any(|op| op.starts_with("residual_call_irf")),
            "{path} must not residualize the payload; ops={ops:?}"
        );
        eprintln!("{path} {} ops: {ops:?}", ops.len());
    }
}

#[test]
fn float_cmp_leaves_are_the_pypy_leaf() {
    // floatobject.py `_compare` after `_to_float`: `space.newbool(x ? y)`.
    for (path, arith) in [
        (
            "pyre_interpreter::objspace::descroperation::_float_lt",
            "float_lt",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_le",
            "float_le",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_gt",
            "float_gt",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_ge",
            "float_ge",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_eq",
            "float_eq",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_ne",
            "float_ne",
        ),
    ] {
        let jc = crate::jitcode_runtime::pathed_jitcode(path)
            .unwrap_or_else(|| panic!("{path} must be a discovered jitcode"));
        let ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&jc.code)
            .map(|op| op.opname)
            .collect();
        assert!(
            ops.iter().any(|op| *op == arith),
            "{path} must record {arith}; ops={ops:?}"
        );
        assert!(
            ops.len() < 32,
            "{path} must stay the unboxed leaf; ops={ops:?}"
        );
        assert!(
            !ops.iter().any(|op| op.starts_with("residual_call_irf")),
            "{path} must not residualize the payload; ops={ops:?}"
        );
        eprintln!("{path} {} ops: {ops:?}", ops.len());
    }
}

#[test]
fn unary_neg_leaves_are_the_pypy_leaf() {
    for (path, arith) in [
        (
            "pyre_interpreter::objspace::descroperation::_int_neg",
            "int_sub",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_neg",
            "float_neg",
        ),
        ("pyre_interpreter::objspace::descroperation::_float_pos", ""),
        (
            "pyre_interpreter::objspace::descroperation::_int_invert",
            "int_invert",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_abs",
            "float_abs",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_sqrt",
            "",
        ),
        ("pyre_interpreter::objspace::descroperation::_float_sin", ""),
        ("pyre_interpreter::objspace::descroperation::_float_cos", ""),
        ("pyre_interpreter::objspace::descroperation::_float_tan", ""),
        (
            "pyre_interpreter::objspace::descroperation::_float_atan",
            "",
        ),
        ("pyre_interpreter::objspace::descroperation::_float_exp", ""),
        (
            "pyre_interpreter::objspace::descroperation::_float_log1p",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_asin",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_acos",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_sinh",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_cosh",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_tanh",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_asinh",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_acosh",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_atanh",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_cbrt",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_exp2",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_expm1",
            "",
        ),
        ("pyre_interpreter::objspace::descroperation::_float_erf", ""),
        (
            "pyre_interpreter::objspace::descroperation::_float_erfc",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_gamma",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_lgamma",
            "",
        ),
        ("pyre_interpreter::objspace::descroperation::_float_ulp", ""),
        (
            "pyre_interpreter::objspace::descroperation::_float_degrees",
            "float_mul",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_radians",
            "float_mul",
        ),
        ("pyre_interpreter::objspace::descroperation::_float_log", ""),
        ("pyre_interpreter::objspace::descroperation::_float_pow", ""),
        (
            "pyre_interpreter::objspace::descroperation::_float_fmod",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_copysign",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_remainder",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_atan2",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_int_from_floor",
            "cast_float_to_int",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_int_from_ceil",
            "cast_float_to_int",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_int_from_trunc",
            "cast_float_to_int",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_frexp_mantissa",
            "convert_float_bytes_to_longlong",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_int_frexp_exponent",
            "convert_float_bytes_to_longlong",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_float_ldexp",
            "",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_int_isqrt",
            "cast_float_to_int",
        ),
        (
            "pyre_interpreter::objspace::descroperation::_int_abs",
            "int_sub",
        ),
    ] {
        let jc = crate::jitcode_runtime::pathed_jitcode(path)
            .unwrap_or_else(|| panic!("{path} must be a discovered jitcode"));
        let ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&jc.code)
            .map(|op| op.opname)
            .collect();
        if !arith.is_empty() {
            assert!(
                ops.iter().any(|op| *op == arith),
                "{path} must record {arith}; ops={ops:?}"
            );
        }
        assert!(
            ops.iter().any(|op| *op == "new_with_vtable"),
            "{path} must box via in-graph new_with_vtable; ops={ops:?}"
        );
        assert!(
            !ops.iter()
                .any(|op| op.starts_with("inline_call") && op.ends_with("_r")),
            "{path} must not inline_call the constructor; ops={ops:?}"
        );
        eprintln!("{path} {} ops: {ops:?}", ops.len());
    }
    // `space.newbool` is a singleton, not a boxed alloc.
    let isclose = "pyre_interpreter::objspace::descroperation::_float_isclose";
    let jc = crate::jitcode_runtime::pathed_jitcode(isclose)
        .unwrap_or_else(|| panic!("{isclose} must be a discovered jitcode"));
    let ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&jc.code)
        .map(|op| op.opname)
        .collect();
    assert!(
        ops.iter().any(|op| *op == "float_le" || *op == "float_eq"),
        "{isclose} must record the comparison; ops={ops:?}"
    );
    assert!(
        !ops.iter()
            .any(|op| *op == "new_with_vtable" || op.starts_with("residual_call")),
        "{isclose} must neither box nor call out for its bool; ops={ops:?}"
    );
    eprintln!("{isclose} {} ops: {ops:?}", ops.len());
    // baseobjspace.py `newbool`: a branch on the value and one prebuilt
    // singleton constant per arm.
    let newbool = "pyre_object::boolobject::w_bool_from";
    let jc = crate::jitcode_runtime::pathed_jitcode(newbool)
        .unwrap_or_else(|| panic!("{newbool} must be a discovered jitcode"));
    let ops: Vec<&str> = crate::jitcode_runtime::decoded_ops(&jc.code)
        .map(|op| op.opname)
        .collect();
    assert!(
        ops.iter().any(|op| *op == "ref_return"),
        "{newbool} must return a ref; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| op.starts_with("residual_call")
            || op.starts_with("inline_call")
            || *op == "new_with_vtable"),
        "{newbool} must read its singletons as constants; ops={ops:?}"
    );
    eprintln!("{newbool} {} ops: {ops:?}", ops.len());
}

#[test]
fn append_journal_rollback_rewinds_length() {
    // #171 journal infra: a walked eager `list.append` grows the
    // concrete list at trace time (the fold records the array-op IR but
    // does not mutate), so a NON-commit walk must rewind the length,
    // exactly like the STORE_SUBSCR store journal restores its displaced
    // element.  Spare-capacity gating (`w_list_can_append_without_realloc`)
    // makes the rewind a pure length set with no reallocation to undo.
    use pyre_object::listobject::{w_list_can_append_without_realloc, w_list_len};
    use pyre_object::{w_int_new, w_list_append};

    super::fbw_store_journal_reset();

    let list =
        pyre_object::listobject::w_list_new(vec![w_int_new(10), w_int_new(20), w_int_new(30)]);
    // A first append forces the backing array to grow with a growth
    // factor, leaving spare capacity so the *next* append is in-place
    // (the only shape the arm specializes).
    unsafe { w_list_append(list, w_int_new(40)) };
    let len_before = unsafe { w_list_len(list) };
    let allocated_before = unsafe { pyre_object::listobject::w_list_allocated(list) };
    assert_eq!(len_before, 4);
    assert!(
        unsafe { w_list_can_append_without_realloc(list) },
        "post-grow list must have spare capacity for the in-place append"
    );

    // Rollback path: journal push + eager append (production order, see
    // try_walker_orthodox_list_append), then a non-commit exit rewinds
    // the length.
    super::fbw_list_journal_push_append(list, len_before, allocated_before);
    unsafe { w_list_append(list, w_int_new(50)) };
    assert_eq!(unsafe { w_list_len(list) }, 5);
    super::fbw_store_journal_rollback();
    assert_eq!(
        unsafe { w_list_len(list) },
        len_before,
        "non-commit walk must rewind the eager append's length"
    );
    assert_eq!(
        unsafe { pyre_object::listobject::w_list_allocated(list) },
        allocated_before,
        "non-commit walk must restore the logical allocation"
    );

    // Commit path: the eager append stands; the log is dropped.
    super::fbw_list_journal_push_append(list, len_before, allocated_before);
    unsafe { w_list_append(list, w_int_new(60)) };
    super::fbw_store_journal_commit();
    assert_eq!(
        unsafe { w_list_len(list) },
        len_before + 1,
        "committed walk keeps the eager append"
    );
    // A subsequent rollback with the log already committed-empty is a
    // no-op (does not shrink further).
    super::fbw_store_journal_rollback();
    assert_eq!(unsafe { w_list_len(list) }, len_before + 1);
}

#[test]
fn pop_end_journal_rollback_restores_item_and_length() {
    use pyre_object::listobject::{
        ListStrategy, W_ListObject, ll_list_int_getitem_fast, w_list_allocated, w_list_len,
        w_list_new_with_strategy,
    };
    use pyre_object::{w_int_new, w_list_pop_end};

    super::fbw_store_journal_reset();
    let list = w_list_new_with_strategy(
        vec![w_int_new(10), w_int_new(20), w_int_new(30)],
        ListStrategy::Integer,
    );
    let len_before = unsafe { w_list_len(list) };
    let raw_item =
        unsafe { ll_list_int_getitem_fast(&*(list as *const W_ListObject), len_before - 1) };
    unsafe { w_list_pop_end(list) };
    super::fbw_list_journal_push_pop_end(list, len_before, w_int_new(raw_item));
    assert_eq!(unsafe { w_list_len(list) }, len_before - 1);

    super::fbw_store_journal_rollback();
    assert_eq!(unsafe { w_list_len(list) }, len_before);
    assert_eq!(
        unsafe { ll_list_int_getitem_fast(&*(list as *const W_ListObject), len_before - 1) },
        raw_item
    );
}

#[test]
fn pop_end_journal_rollback_after_strategy_switch() {
    use pyre_object::listobject::{
        ListStrategy, W_ListObject, ll_list_int_getitem_fast, w_list_len, w_list_new_with_strategy,
        w_list_uses_object_storage,
    };
    use pyre_object::{w_int_get_value, w_int_new, w_list_append, w_list_getitem, w_list_pop_end};

    super::fbw_store_journal_reset();
    let list = w_list_new_with_strategy(
        vec![w_int_new(10), w_int_new(20), w_int_new(30)],
        ListStrategy::Integer,
    );
    let len_before = unsafe { w_list_len(list) };
    let raw_item =
        unsafe { ll_list_int_getitem_fast(&*(list as *const W_ListObject), len_before - 1) };
    unsafe { w_list_pop_end(list) };
    super::fbw_list_journal_push_pop_end(list, len_before, w_int_new(raw_item));

    unsafe { w_list_append(list, pyre_object::w_str_new("strategy switch")) };
    assert!(unsafe { w_list_uses_object_storage(list) });

    super::fbw_store_journal_rollback();
    assert_eq!(unsafe { w_list_len(list) }, len_before);
    let restored = unsafe { w_list_getitem(list, (len_before - 1) as i64).unwrap() };
    assert_eq!(unsafe { w_int_get_value(restored) }, raw_item);
}

#[test]
fn interleaved_append_pop_journal_rollback_restores_original() {
    use pyre_object::listobject::{
        ListStrategy, W_ListObject, ll_list_int_getitem_fast, w_list_allocated, w_list_len,
        w_list_new_with_strategy,
    };
    use pyre_object::{w_int_new, w_list_append, w_list_pop_end};

    super::fbw_store_journal_reset();
    let original = [10, 20, 30];
    let list = w_list_new_with_strategy(
        original.into_iter().map(w_int_new).collect(),
        ListStrategy::Integer,
    );

    let len_before_append = unsafe { w_list_len(list) };
    let allocated_before_append = unsafe { w_list_allocated(list) };
    super::fbw_list_journal_push_append(list, len_before_append, allocated_before_append);
    unsafe { w_list_append(list, w_int_new(40)) };

    let len_before_pop = unsafe { w_list_len(list) };
    let raw_item =
        unsafe { ll_list_int_getitem_fast(&*(list as *const W_ListObject), len_before_pop - 1) };
    unsafe { w_list_pop_end(list) };
    super::fbw_list_journal_push_pop_end(list, len_before_pop, w_int_new(raw_item));

    let len_before_append = unsafe { w_list_len(list) };
    let allocated_before_append = unsafe { w_list_allocated(list) };
    super::fbw_list_journal_push_append(list, len_before_append, allocated_before_append);
    unsafe { w_list_append(list, w_int_new(50)) };

    let len_before_pop = unsafe { w_list_len(list) };
    let raw_item =
        unsafe { ll_list_int_getitem_fast(&*(list as *const W_ListObject), len_before_pop - 1) };
    unsafe { w_list_pop_end(list) };
    super::fbw_list_journal_push_pop_end(list, len_before_pop, w_int_new(raw_item));

    super::fbw_store_journal_rollback();
    assert_eq!(unsafe { w_list_len(list) }, original.len());
    for (index, expected) in original.into_iter().enumerate() {
        assert_eq!(
            unsafe { ll_list_int_getitem_fast(&*(list as *const W_ListObject), index) },
            expected
        );
    }
}

#[test]
fn append_journal_rollback_rewinds_object_length() {
    // #171 object-append: the orthodox fold journals object-strategy
    // appends through the SAME `FBW_LIST_EFFECT_JOURNAL` as the int spec, so a
    // non-commit rollback must rewind the strategy-correct length — the
    // `W_ListObject.length` header (`ll_list_obj_set_len`), not
    // `int_items.len`.  Rewinding via the int leaf would leave the object
    // header grown (legacy replay double-appends) and trip the
    // `w_list_int_set_len` strategy assert.
    use pyre_object::listobject::{w_list_can_append_without_realloc, w_list_len};
    use pyre_object::{w_list_append, w_none};

    super::fbw_store_journal_reset();

    // `w_list_new_object` forces the Object strategy regardless of element
    // type; None elements keep the test free of int/float boxing.
    let list = pyre_object::listobject::w_list_new_object(vec![w_none(), w_none(), w_none()]);
    // Grow once so the next append is an in-place spare-capacity store
    // (the only shape the journal records).
    unsafe { w_list_append(list, w_none()) };
    let len_before = unsafe { w_list_len(list) };
    let allocated_before = unsafe { pyre_object::listobject::w_list_allocated(list) };
    assert_eq!(len_before, 4);
    assert!(
        unsafe { w_list_can_append_without_realloc(list) },
        "post-grow object list must have spare capacity for the in-place append"
    );

    super::fbw_list_journal_push_append(list, len_before, allocated_before);
    unsafe { w_list_append(list, w_none()) };
    assert_eq!(unsafe { w_list_len(list) }, 5);
    super::fbw_store_journal_rollback();
    assert_eq!(
        unsafe { w_list_len(list) },
        len_before,
        "non-commit walk must rewind the object append's length header"
    );
}

/// Tests use the production `PyreJitCodeDescr` adapter
/// directly — the type lives at
/// `pyre-jit-trace/src/descr.rs::PyreJitCodeDescr`
/// + `descr::make_jitcode_descr(idx)` so the walker's
/// `as_jitcode_descr()` cast exercises production code, not a
/// test-local duplicate.
use crate::descr::make_jitcode_descr;

/// Build a `descr_refs` pool of length `pool_len` where the slot at
/// each `BhDescr::JitCode` index in `crate::jitcode_runtime::all_descrs()`
/// holds a `TestJitCodeDescr` carrying that descr's `jitcode_index`,
/// and every other slot holds a `make_fail_descr` placeholder.
/// Lets acceptance tests resolve `inline_call_*` descr indices
/// without standing up the full BhDescr → trait Descr adapter
/// pipeline.
fn descr_pool_with_jitcode_adapters(pool_len: usize) -> Vec<DescrRef> {
    let all_bh = crate::jitcode_runtime::all_descrs();
    (0..pool_len)
        .map(|i| match all_bh.get(i) {
            Some(majit_jitcode::jitcode::BhDescr::JitCode { jitcode_index, .. }) => {
                make_jitcode_descr(*jitcode_index)
            }
            // Residual calls surfaced during inline_call recursion (e.g.
            // the arm bodies' helper calls) resolve their descr through
            // `as_call_descr()`. Build a real CallDescr for `BhDescr::Call`
            // entries, mirroring production (`descr.rs make_call_descr_from_bh`),
            // so the walk does not surface ResidualCallDescrNotCallDescr.
            Some(majit_jitcode::jitcode::BhDescr::Call { calldescr }) => {
                crate::descr::make_call_descr_from_bh(calldescr)
            }
            _ => make_fail_descr(1 + i),
        })
        .collect()
}

#[test]
fn inline_call_recursion_writes_subreturn_into_caller_dst_register() {
    // Core acceptance: caller's `inline_call_r_r/dR>r`
    // recurses into a synthetic callee jitcode whose body is
    // simply `ref_return r0`. The callee's ref_return surfaces as
    // `SubReturn { result: Some(callee.registers_r[0]) }`; the
    // caller's inline_call handler writes that OpRef into the
    // caller's dst register. Then the caller's own `ref_return r3`
    // records the outermost Finish carrying that propagated value.
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    // Callee body: `ref_return r0`. registers_r[0] is populated
    // from the caller's R-list arg.
    let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // Caller body:
    //   inline_call_r_r/dR>r descr=7, R=[r2], >r=r5
    //   ref_return r5
    let caller_code = [
        inline_byte,
        0x07,
        0x00, // d (LE descr index = 7)
        0x01,
        0x02, // R: varlen=1, args=[r2]
        0x05, // >r: dst = r5
        ret_byte,
        0x05, // ref_return r5
    ];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let arg_value = regs_r[2];
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    // The standing exception is metainterp-level state
    // (`pyjitpl.py:2417`), so it is seeded on the session.
    {
        let mut sess = session.borrow_mut();
        sess.last_exc_value = Some(arg_value);
        sess.last_exc_value_concrete = ConcreteValue::Int(123);
    }
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    fbw_finish_payload_reset();
    let (outcome, end_pc) = walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
    assert_eq!(end_pc, caller_code.len());
    assert_eq!(
        wc.last_exc_value(),
        None,
        "normal inline return must clear the caller's stale exception",
    );
    assert_eq!(
        wc.last_exc_value_concrete(),
        ConcreteValue::Null,
        "normal inline return must clear the caller's concrete exception shadow",
    );
    let regs_r = wc.registers_r.to_vec();
    drop(wc);
    // dst register r5 must equal the arg the caller passed (since
    // callee's `ref_return r0` returns its registers_r[0] which
    // was populated from caller's R-list[0] = r2's OpRef).
    assert_eq!(
        regs_r[5], arg_value,
        "inline_call_r_r dst writeback must propagate callee's SubReturn",
    );
    // The outermost finish payload carries the same value (callee's
    // ref_return surfaced as SubReturn and recorded nothing; the
    // caller's top-level ref_return stashed the payload).
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "no op recorded: the compile consumer records the FINISH from the payload",
    );
    assert_eq!(
        finish_payload_of(&outcome),
        Some((arg_value, Type::Ref)),
        "outermost finish payload must carry the arg value the caller threaded \
             through inline_call_r_r",
    );
}

#[test]
fn inline_call_subwalk_uses_heap_frames_past_the_old_host_stack_cap() {
    // The retired recursive walker declined at depth 65.  Build a 96-frame
    // chain so this test can only finish when `run_sub_jitcode_walk` drives an
    // explicit heap stack and delivers each SubReturn to its parent CALL.
    const DEPTH: usize = 96;
    let ret = *insns_opname_to_byte().get("ref_return/r").unwrap();
    let inline = *insns_opname_to_byte().get("inline_call_r_r/dR>r").unwrap();

    let mut bodies = Vec::with_capacity(DEPTH);
    for index in 0..DEPTH {
        let code: &'static [u8] = if index + 1 == DEPTH {
            Box::leak(vec![ret, 0].into_boxed_slice())
        } else {
            let next = index + 1;
            Box::leak(
                vec![
                    inline,
                    (next & 0xff) as u8,
                    (next >> 8) as u8,
                    1,
                    0,
                    0,
                    ret,
                    0,
                ]
                .into_boxed_slice(),
            )
        };
        bodies.push(SubJitCodeBody {
            code,
            num_regs_r: 1,
            num_regs_i: 0,
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        });
    }
    let lookup = move |index: usize| bodies.get(index).cloned();
    let descr_pool: Vec<DescrRef> = (0..DEPTH).map(make_jitcode_descr).collect();

    let root_code = [inline, 0, 0, 1, 0, 0, ret, 0];
    let mut trace_ctx = fresh_trace_ctx();
    let expected = OpRef::input_arg_typed(0, Type::Ref);
    let regs_r = vec![expected];
    trace_ctx.heap_cache_mut().class_now_known(expected);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut walk_ctx = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut trace_ctx,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    fbw_finish_payload_reset();
    let (outcome, end_pc) = walk(&root_code, 0, &mut walk_ctx).unwrap();
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
    assert_eq!(end_pc, root_code.len());
    drop(walk_ctx);
    assert_eq!(regs_r[0], expected);
    assert!(
        trace_ctx.heap_cache().is_class_known(expected),
        "a child push must restore the parent's pre-CALL heap-cache knowledge"
    );
    assert_eq!(finish_payload_of(&outcome), Some((expected, Type::Ref)));
}

#[test]
fn inline_call_r_i_writes_int_subreturn_into_caller_int_bank() {
    // Acceptance: caller's `inline_call_r_i/dR>i`
    // recurses into a synthetic callee whose body is `int_return
    // r0` on the int bank. The callee's int_return surfaces as
    // `SubReturn { result: Some(callee.registers_i[0]) }`; the
    // caller's helper writes that OpRef into the caller's
    // `registers_i[dst]` (NOT registers_r — the kind discriminator
    // for this variant). RPython parity: pyjitpl.py
    // exec-generated `_opimpl_inline_call_r_i` template paired with
    // `_opimpl_any_return` for `int_return`.
    //
    // Callee shape constraint: the `_r` arglist promises only Ref
    // args, but the body needs an int register populated to source
    // the int_return. The codewriter populates that via a separate
    // op inside the callee body (e.g. `int_copy/i>i`). For the
    // walker test, we synthesize an `int_copy` that materializes
    // the int constant from a tracer-side const_int OpRef stored in
    // a high i-register, then int_returns it.
    //
    // Simpler: callee whose body is just `int_return i0`; we
    // pre-populate the callee's registers_i[0] indirectly through a
    // setter — but the walker doesn't expose that directly. So
    // instead, we use a callee whose body emits an int constant op
    // and returns it. The simplest working shape is `int_neg i0
    // ->i0; int_return i0` — but registers_i[0] starts as
    // OpRef::NONE which `int_neg` would record meaninglessly.
    //
    // Pragmatic alternative: the walker's test-side `setup` for
    // sub_body lets us choose `num_regs_i = 1`. We initialize
    // callee.registers_i[0] to a known OpRef AT SUB-WALK TIME by
    // having the caller arglist carry the int OpRef indirectly —
    // but the `dR` arglist only has Ref. So we *can't* pass the
    // int OpRef through the call.
    //
    // RPython solution: callee bodies *always* compute their int
    // results from concrete operations (int_const, int_add, etc.).
    // For walker testing, the smallest standalone body is
    // `int_const_42 i0 = 42; int_return i0` — but pyre doesn't have
    // an `int_const/c>i` opname today (constants live in the
    // jitcode's constants_i table). Without re-engineering the
    // sub_body fixture, the cleanest test is to drive the callee
    // body through `int_copy` from a callee int register that the
    // setup_call path populated (which doesn't exist for `_r_i`
    // variant — only Ref args flow in).
    //
    // Here we lean on the simpler invariant: the
    // SubReturn{Some(value)} from the *helper itself* writes into
    // the caller's `registers_i[dst]`. To exercise that branch
    // without standing up a full int-producing callee, we test the
    // helper's dst-bank dispatch logic via a callee body that
    // returns an OpRef::NONE placeholder through `int_return r0`
    // — wait, that's wrong: `int_return/i` reads from `registers_i`
    // not `registers_r`.
    //
    // Cleanest path: callee body = `[int_return_byte, 0x00]` where
    // callee's `registers_i[0]` is OpRef::NONE; the SubReturn
    // value will be NONE. The test asserts that the caller's
    // `registers_i[dst]` was written to NONE (proving the dst-bank
    // routing is correct — wrong-bank routing would write to
    // `registers_r[dst]` instead and leave `registers_i[dst]`
    // unchanged at its initial OpRef::NONE).
    //
    // The OpRef::NONE-vs-OpRef::NONE comparison is admittedly
    // weak; instead we initialize the caller's `registers_i[dst]`
    // to a distinct OpRef before the call so the assertion can
    // distinguish "no write" from "write of NONE".
    let int_ret_byte = *insns_opname_to_byte()
        .get("int_return/i")
        .expect("`int_return/i` must be in insns table");
    let inline_ri_byte = *insns_opname_to_byte()
        .get("inline_call_r_i/dR>i")
        .expect("`inline_call_r_i/dR>i` must be in insns table");
    // Callee body: `int_return i0` (registers_i[0] starts at NONE).
    let callee_code: &'static [u8] = Box::leak(Box::new([int_ret_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1, // callee accepts a Ref arg, then ignores it
        num_regs_i: 1,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // Caller body: `inline_call_r_i descr=7, R=[r2], >i=i3`
    //   opcode(1) + d(2) + R-len(1) + R[0](1) + dst(1) = 6 bytes
    let caller_code = [inline_ri_byte, 0x07, 0x00, 0x01, 0x02, 0x03];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 4);
    // Initialize registers_i[3] (dst) to a sentinel so we can
    // detect that the write happened.
    let sentinel_pre = tc.const_int(0xDEAD_BEEF);
    let regs_i: Vec<OpRef> = vec![sentinel_pre; 4];
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&caller_code, 0, &mut wc).expect("inline_call_r_i must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, caller_code.len());
    let returned_int = wc.registers_i.get(3);
    let regs_r = wc.registers_r.to_vec();
    drop(wc);
    // Callee's int_return[i0] surfaced SubReturn{Some(NONE)}; the
    // helper wrote that into caller's registers_i[3]. Sentinel is
    // gone, replaced by OpRef::NONE.
    assert_eq!(
        returned_int,
        Some(OpRef::NONE),
        "inline_call_r_i must write SubReturn value into caller registers_i[dst]",
    );
    // Wrong-bank check: registers_r[3] must remain its original
    // distinct_const_refs value (the dst-bank routing did NOT
    // write to the Ref bank).
    assert_ne!(
        regs_r[3],
        OpRef::NONE,
        "inline_call_r_i must NOT write to registers_r[dst]",
    );
}

#[test]
fn inline_call_ir_r_populates_callee_int_and_ref_banks() {
    // Acceptance: caller's `inline_call_ir_r/dIR>r` carries
    // both an I-list and an R-list. The callee's int + ref register
    // banks must both be populated (RPython
    // `pyjitpl.py setup_call(argboxes_i, argboxes_r,
    // argboxes_f)`). Smoke test: callee body is `ref_return r0` —
    // the ref arg routes through registers_r[0] back to the caller's
    // dst slot. The int arg flowing into registers_i[0] is dead but
    // proves the helper read the I-list (a regression where the
    // I-list parsing miscounted bytes would offset the R-list read
    // and we'd see the wrong ref OpRef in the dst).
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let inline_ir_r_byte = *insns_opname_to_byte()
        .get("inline_call_ir_r/dIR>r")
        .expect("`inline_call_ir_r/dIR>r` must be in insns table");
    // Callee body: `ref_return r0` (size 2).
    let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 1, // accept one int arg
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // Caller body: `inline_call_ir_r descr=7, I=[i1], R=[r2], >r=r5`
    //   opcode(1) + d(2) + I-len(1) + I[0](1) + R-len(1) + R[0](1) + dst(1) = 8 bytes
    let caller_code = [
        inline_ir_r_byte,
        0x07,
        0x00, // descr index 7 (LE)
        0x01,
        0x01, // I-list: len=1, args=[i1]
        0x01,
        0x02, // R-list: len=1, args=[r2]
        0x05, // dst = r5
    ];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let arg_ref = regs_r[2];
    let regs_i: Vec<OpRef> = (0..4)
        .map(|i| tc.const_int(0xCAFE_F00D + i as i64))
        .collect();
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) =
        step(&caller_code, 0, &mut wc).expect("inline_call_ir_r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, caller_code.len());
    let regs_r = wc.registers_r.to_vec();
    drop(wc);
    // dst register r5 must equal the caller's R-list arg (which the
    // callee returned via ref_return r0).
    assert_eq!(
        regs_r[5], arg_ref,
        "inline_call_ir_r dst writeback must propagate callee's SubReturn from ref_return r0",
    );
}

#[test]
fn compare_tag_inline_call_records_no_binary_exception_guard() {
    // Flatten lowers `x is y` to `inline_call_ir_r` of
    // `compare_value_from_tag` with COMPARE tag 8, which sits in the same
    // I-list slot a `binary_value_from_tag` call uses for its BINARY tag.
    // Read as a BINARY tag, 8 decodes as `<<` — a raising operator — so the
    // call used to record a `GUARD_NO_EXCEPTION` plus its snapshot for an
    // operation it never performs.  The callee here is neither helper, so
    // no operator guard may appear.
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let inline_ir_r_byte = *insns_opname_to_byte()
        .get("inline_call_ir_r/dIR>r")
        .expect("`inline_call_ir_r/dIR>r` must be in insns table");
    // Callee body: `ref_return r0` (size 2), two ref params.
    let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 2,
        num_regs_i: 1,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // Caller body: `inline_call_ir_r descr=7, I=[i1], R=[r2, r3], >r=r5`
    let caller_code = [
        inline_ir_r_byte,
        0x07,
        0x00, // descr index 7 (LE)
        0x01,
        0x01, // I-list: len=1, args=[i1]
        0x02,
        0x02,
        0x03, // R-list: len=2, args=[r2, r3]
        0x05, // dst = r5
    ];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let arg_ref = regs_r[2];
    // The I-list names `i1`, so THAT slot must carry COMPARE tag 8 (`is`),
    // which decodes as the raising `<<` when read as a BINARY tag.  The tag
    // is read out of the color-indexed concrete shadow, not the OpRef bank,
    // so both have to hold it.
    let mut regs_i: Vec<OpRef> = (0..4).map(|_| tc.const_int(0)).collect();
    regs_i[1] = tc.const_int(8);
    let mut concrete_i = vec![ConcreteValue::Int(0); 4];
    concrete_i[1] = ConcreteValue::Int(8);
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut concrete_i,
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    // Give the fixture a resume coordinate so a guard this dispatcher records
    // can capture its snapshot.  Without one the bogus guard surfaces as a
    // dispatch error instead of as a recorded op, which is a weaker witness.
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let (outcome, next_pc) =
        step(&caller_code, 0, &mut wc).expect("inline_call_ir_r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, caller_code.len());
    let regs_r_after = wc.registers_r.to_vec();
    drop(wc);
    assert_eq!(
        regs_r_after[5], arg_ref,
        "dst writeback must still propagate the callee's SubReturn",
    );
    assert!(
        !tc.ops()
            .iter()
            .any(|o| o.opcode == majit_ir::OpCode::GuardNoException),
        "a COMPARE tag must not be read as a raising BINARY operator",
    );
}

#[test]
fn inline_call_irf_r_populates_all_three_kind_banks() {
    // Acceptance: caller's `inline_call_irf_r/dIRF>r`
    // carries an I-list, R-list, AND F-list. Smoke test: callee
    // body is `ref_return r0` — the caller's R-list arg propagates
    // through. The I-list and F-list args are dead from the
    // callee's POV but their presence forces the helper to advance
    // operand offsets correctly through all three lists; a parsing
    // bug (e.g. F-list-len byte misaligned) would put the wrong
    // ref OpRef into the dst.
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let inline_irf_r_byte = *insns_opname_to_byte()
        .get("inline_call_irf_r/dIRF>r")
        .expect("`inline_call_irf_r/dIRF>r` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 1,
        num_regs_f: 1,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // Caller body: inline_call_irf_r descr=7, I=[i1], R=[r2], F=[f0], >r=r5
    //   opcode(1) + d(2) + I-len(1) + I[0](1) + R-len(1) + R[0](1)
    //   + F-len(1) + F[0](1) + dst(1) = 10 bytes
    let caller_code = [
        inline_irf_r_byte,
        0x07,
        0x00, // descr index 7
        0x01,
        0x01, // I-list len=1, args=[i1]
        0x01,
        0x02, // R-list len=1, args=[r2]
        0x01,
        0x00, // F-list len=1, args=[f0]
        0x05, // dst = r5
    ];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let arg_ref = regs_r[2];
    let regs_i: Vec<OpRef> = (0..4).map(|i| tc.const_int(i as i64)).collect();
    // Float bank: pyre's TraceCtx doesn't expose a const_float
    // factory in the test fixture path, but we only need *distinct*
    // OpRef values to exercise list-byte advancement; const_int +
    // type-punning into the float slot is sufficient because the
    // walker treats the bank as opaque OpRef storage.
    let regs_f: Vec<OpRef> = (0..4).map(|i| tc.const_int(0xF1F1 + i as i64)).collect();
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::new(regs_f.iter().copied()),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) =
        step(&caller_code, 0, &mut wc).expect("inline_call_irf_r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, caller_code.len());
    let regs_r = wc.registers_r.to_vec();
    drop(wc);
    // Smoking gun: dst register r5 must equal the caller's R-list
    // arg (passed through callee's `ref_return r0`). A list-byte
    // advancement bug would land a different OpRef here.
    assert_eq!(
        regs_r[5], arg_ref,
        "inline_call_irf_r must correctly advance through I/R/F lists \
             and propagate the callee's ref SubReturn",
    );
}

#[test]
fn inline_call_ir_int_arity_overflow_surfaces_typed_error() {
    // Per-bank arity check — providing more I-args than
    // the callee declared `num_regs_i` slots surfaces
    // `InlineCallIntArityMismatch`. The Ref-bank check is covered
    // by the existing `inline_call_with_more_args_than_callee_regs_surfaces_arity_mismatch`
    // test for the `_r_r` variant.
    let inline_ir_r_byte = *insns_opname_to_byte()
        .get("inline_call_ir_r/dIR>r")
        .expect("`inline_call_ir_r/dIR>r` must be in insns table");
    // Callee with num_regs_i=0 — any I-list args overflow.
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0, // overflow trigger
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // Caller body: `inline_call_ir_r descr=7, I=[i1], R=[r2], >r=r5`
    let caller_code = [
        inline_ir_r_byte,
        0x07,
        0x00,
        0x01,
        0x01, // I-list with 1 arg → overflows callee (num_regs_i=0)
        0x01,
        0x02,
        0x05,
    ];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let regs_i: Vec<OpRef> = (0..4).map(|i| tc.const_int(i as i64)).collect();
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&caller_code, 0, &mut wc).expect_err("I-list overflow must surface typed error");
    assert_eq!(
        err,
        DispatchError::InlineCallIntArityMismatch {
            pc: 0,
            provided: 1,
            callee_num_regs_i: 0,
        },
    );
}

#[test]
fn inline_call_recursion_propagates_subraise_from_callee() {
    // Top-level uncaught SubRaise: callee's `raise/r` surfaces as
    // `SubRaise { exc }` to the caller's inline_call handler. With
    // no caller-side `catch_exception/L` and is_top_level=true on
    // the outermost walker, RPython
    // `pyjitpl.py finishframe_exception` records
    // `compile_exit_frame_with_exception(last_exc_box)` — i.e.
    // FINISH(exc, exit_frame_with_exception_descr_ref) and exits
    // the trace. Walker mirrors this in `walk()`: top-level
    // SubRaise → record FINISH + Terminate.
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    // Callee body: `raise r0`
    let callee_code: &'static [u8] = Box::leak(Box::new([raise_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // Caller body: `inline_call_r_r descr=7 R=[r2] >r=r5`
    // (no follow-on `ref_return` — the SubRaise propagates straight
    // up to the caller's `walk` loop, which converts to FINISH at
    // top level.)
    let caller_code = [inline_byte, 0x07, 0x00, 0x01, 0x02, 0x05];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let arg_value = regs_r[2];
    let descr = done_descr_ref_for_tests();
    let descr_exc = make_fail_descr(2);
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    fbw_finish_payload_reset();
    let (outcome, _) = walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
    assert!(
        matches!(outcome, DispatchOutcome::Terminate { .. }),
        "top-level walk must convert uncaught SubRaise to Terminate",
    );
    drop(wc);
    // The bubbled exception is stashed as an `is_exception` finish payload,
    // NOT recorded inline: `full_body_walk_trace`'s Terminate arm builds
    // `TraceAction::Finish { exit_with_exception: true }` and the compile
    // consumer records the single FINISH against
    // `exit_frame_with_exception_descr`.  So no op is recorded in `walk()`.
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "propagated SubRaise must NOT record an inline FINISH — the payload is deferred",
    );
    assert!(
        fbw_finish_is_exception(),
        "a top-level propagated SubRaise must mark the finish payload as an exception exit",
    );
    let (finish_value, finish_ty) =
        finish_payload_of(&outcome).expect("exception finish payload must be stashed");
    assert_eq!(finish_ty, Type::Ref, "portal-exit FINISH carries Type::Ref");
    assert_eq!(
        finish_value, arg_value,
        "the stashed payload must carry the bubbled exc OpRef",
    );
    let _ = &descr_exc;
}

#[test]
fn inline_call_with_unresolvable_descr_surfaces_typed_error() {
    // Descr at the inline_call's d-slot must implement
    // `JitCodeDescr`. A `FailDescr` placeholder doesn't, so the
    // walker surfaces `ExpectedJitCodeDescr`.
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    let caller_code = [inline_byte, 0x05, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&caller_code, 0, &mut wc)
        .expect_err("FailDescr at inline_call's d-slot must hit ExpectedJitCodeDescr");
    assert_eq!(
        err,
        DispatchError::ExpectedJitCodeDescr {
            pc: 0,
            descr_index: 5,
        },
    );
}

#[test]
fn inline_call_with_missing_sub_jitcode_lookup_surfaces_typed_error() {
    // Descr resolves to JitCodeDescr but lookup returns
    // None — surface `SubJitCodeNotFound`.
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    let caller_code = [inline_byte, 0x03, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[3] = make_jitcode_descr(999_999);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&caller_code, 0, &mut wc)
        .expect_err("missing sub-jitcode must hit SubJitCodeNotFound");
    assert_eq!(
        err,
        DispatchError::SubJitCodeNotFound {
            pc: 0,
            jitcode_index: 999_999,
        },
    );
}

#[test]
fn step_through_live_opcode_advances_by_offset_size() {
    let live_byte = *insns_opname_to_byte()
        .get("live/")
        .expect("`live/` must be in insns table");
    let code = [live_byte, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("live/ must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc,
        1 + majit_jitcode::liveness::OFFSET_SIZE,
        "live/ must advance past the OFFSET_SIZE liveness slot",
    );
}

#[test]
fn step_through_ref_return_records_finish_with_descr_and_correct_arg() {
    // Top-level `ref_return/r` stashes the Finish payload for
    // `full_body_walk_trace` (the compile consumer records the FINISH
    // from `finish_args`), and the `reg` byte selects the correct
    // OpRef from `registers_r`.
    // RPython `pyjitpl.py:opimpl_ref_return → finishframe →
    // compile_done_with_this_frame → record1(FINISH, descr=token)`.
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    // Read register at byte index 3 — distinct from index 0 to
    // catch off-by-one bugs in operand decoding.
    let code = [ret_byte, 0x03];
    let mut tc = fresh_trace_ctx();
    let regs = distinct_const_refs(&mut tc, 8);
    let expected_arg = regs[3];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    fbw_finish_payload_reset();
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("ref_return/r must dispatch");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
    assert_eq!(next_pc, 2, "ref_return/r consumes 1 register byte");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "ref_return must not record the FINISH itself: the compile consumer \
             records it from the stashed finish payload",
    );
    assert_eq!(
        finish_payload_of(&outcome),
        Some((expected_arg, Type::Ref)),
        "finish payload must select registers_r[3], not registers_r[0]",
    );
}

#[test]
fn ref_return_with_out_of_range_register_surfaces_typed_error() {
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let code = [ret_byte, 0x07]; // index 7 — registers_r is empty
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("must surface RegisterOutOfRange");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "r"
        },
    );
}

#[test]
fn raise_with_unwritten_register_surfaces_register_read_unbound() {
    // A jitcode whose `raise/r` names a Ref register no op ever writes leaves
    // that slot at its `[None] * num_regs` initial value.  Carrying the
    // `OpRef::NONE` into the trace produced `Finish(_)` against
    // `ExitFrameWithExceptionDescrRef` (fail-arg type Ref), which no backend
    // can bind — dynasm panicked in `RegisterManager.loc`, cranelift in
    // `resolve_opref`.  The read declines instead.
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let code = [raise_byte, 0x01];
    let mut tc = fresh_trace_ctx();
    let session = std::cell::RefCell::new(WalkSession::default());
    let registers_r = [OpRef::NONE, OpRef::NONE];
    let concrete_registers_r = [ConcreteValue::Null, ConcreteValue::Null];
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: (concrete_registers_r).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(registers_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("must surface RegisterReadUnbound");
    assert_eq!(
        err,
        DispatchError::RegisterReadUnbound {
            pc: 0,
            reg: 1,
            bank: "r"
        },
    );
}

#[test]
fn step_through_int_return_records_finish_with_int_descr() {
    // `int_return/i` mirrors `ref_return/r` on the int bank.
    // Top-level re-boxes the int for the Type::Ref portal exit
    // (`wrapint` = NEW_WITH_VTABLE + SETFIELD_GC) and stashes the
    // boxed value as the finish payload (RPython `pyjitpl.py
    // compile_done_with_this_frame`).
    let ret_byte = *insns_opname_to_byte()
        .get("int_return/i")
        .expect("`int_return/i` must be in insns table");
    let code = [ret_byte, 0x02];
    let mut tc = fresh_trace_ctx();
    let regs_i: Vec<OpRef> = (0..4)
        .map(|i| tc.const_int(0xBEEF_0000 + i as i64))
        .collect();
    let expected_arg = regs_i[2];
    let descr_int = make_fail_descr(42);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let ops_before = wc.trace_ctx.num_ops();
    fbw_finish_payload_reset();
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("int_return/i must dispatch");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
    assert_eq!(next_pc, 2);
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 2,
        "wrapint must record NEW_WITH_VTABLE + SETFIELD_GC for the boxed payload",
    );
    let (finish_value, finish_ty) =
        finish_payload_of(&outcome).expect("finish payload must be stashed");
    assert_eq!(finish_ty, Type::Ref, "portal-exit FINISH carries Type::Ref");
    let ops = tc.ops();
    let new_box = &ops[ops.len() - 2];
    assert_eq!(new_box.opcode, majit_ir::OpCode::NewWithVtable);
    let setfield = ops.last().expect("recorded op must exist");
    assert_eq!(setfield.opcode, majit_ir::OpCode::SetfieldGc);
    assert_eq!(
        setfield
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![finish_value, expected_arg],
        "the re-boxed payload must store int_return's register into the new box",
    );
}

#[test]
fn step_through_int_return_subwalk_surfaces_subreturn_some() {
    // nested `int_return/i` propagates SubReturn{Some(value)}
    // — same shape as `ref_return/r` sub-walk. RPython
    // `pyjitpl.py finishframe → popframe` returns control to
    // caller's metainterp loop with the box in hand.
    let ret_byte = *insns_opname_to_byte()
        .get("int_return/i")
        .expect("`int_return/i` must be in insns table");
    let code = [ret_byte, 0x01];
    let mut tc = fresh_trace_ctx();
    let regs_i: Vec<OpRef> = (0..4)
        .map(|i| tc.const_int(0xCAFE_0000 + i as i64))
        .collect();
    let expected = regs_i[1];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: false,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let ops_before = wc.trace_ctx.num_ops();
    let (outcome, _) = step(&code, 0, &mut wc).expect("int_return/i must dispatch");
    assert_eq!(
        outcome,
        DispatchOutcome::SubReturn {
            result: Some(expected),
        },
        "sub-walk int_return must surface SubReturn{{Some(value)}}",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "sub-walk int_return must NOT record FINISH (control returns to caller)",
    );
}

#[test]
fn step_through_void_return_stashes_void_finish_payload() {
    // Top-level `void_return/` is the VOID portal exit (RPython
    // `pyjitpl.py compile_done_with_this_frame`, the
    // `result_type == VOID` branch — `exits = []`,
    // `token = sd.done_with_this_frame_descr_void`).  Under the
    // finish-portal route it mirrors the three
    // value-returning arms: it does NOT record the FINISH op itself
    // (the compile consumer records `FINISH([])` from the empty
    // finish_args) and stashes a `Type::Void`-marked payload so
    // `full_body_walk_trace` builds `TraceAction::Finish` with no args.
    let ret_byte = *insns_opname_to_byte()
        .get("void_return/")
        .expect("`void_return/` must be in insns table");
    let code = [ret_byte];
    let mut tc = fresh_trace_ctx();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let ops_before = wc.trace_ctx.num_ops();
    fbw_finish_payload_reset();
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("void_return/ must dispatch");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
    assert_eq!(next_pc, 1, "void_return/ has zero operand bytes");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "void_return must NOT record the FINISH itself: the compile \
             consumer records FINISH([]) from the empty finish_args",
    );
    assert_eq!(
        finish_payload_of(&outcome),
        Some((OpRef::NONE, Type::Void)),
        "void portal exit stashes a Type::Void-marked payload",
    );
}

#[test]
fn step_through_void_return_subwalk_surfaces_subreturn_none() {
    // nested `void_return/` propagates SubReturn{None} —
    // RPython `pyjitpl.py opimpl_void_return → finishframe(None)`.
    // The caller's `inline_call_*_v` variant (when one exists) does
    // not write a dst register; today the walker has no `_v`
    // inline_call handler so `SubReturn{None}` reaching an `_r_r`
    // caller surfaces `UnexpectedVoidSubReturn` (the existing typed
    // error covers that path). This test only exercises the leaf
    // sub-walk surface.
    let ret_byte = *insns_opname_to_byte()
        .get("void_return/")
        .expect("`void_return/` must be in insns table");
    let code = [ret_byte];
    let mut tc = fresh_trace_ctx();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: false,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let ops_before = wc.trace_ctx.num_ops();
    let (outcome, _) = step(&code, 0, &mut wc).expect("void_return/ must dispatch");
    assert_eq!(outcome, DispatchOutcome::SubReturn { result: None });
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "sub-walk void_return must NOT record FINISH",
    );
}

#[test]
fn raise_with_out_of_range_register_surfaces_typed_error() {
    // `raise/r` reads its operand for OOR validation
    // even though recording is deferred. Catches the same classes
    // of assembler bugs `ref_return/r` does.
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let code = [raise_byte, 0x05];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("raise/r must read its operand");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 5,
            len: 0,
            bank: "r"
        },
    );
}

#[test]
fn step_through_goto_jumps_to_label_target() {
    // `goto/L` reads its 2-byte LE label and the walker
    // returns Continue at the label target, not the linear next pc.
    // RPython `blackhole.py bhimpl_goto(target): return target`.
    let goto_byte = *insns_opname_to_byte()
        .get("goto/L")
        .expect("`goto/L` must be in insns table");
    // target = 0x002A = 42
    let code = [goto_byte, 0x2A, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("goto/L must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 42,
        "goto/L must jump to its 2-byte LE label target",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "goto/L records nothing — pure control flow",
    );
}

#[test]
fn step_through_goto_handles_high_byte_of_label() {
    // Confirm the LE decode reads both bytes (regression guard for
    // accidentally treating L as a single byte).
    let goto_byte = *insns_opname_to_byte()
        .get("goto/L")
        .expect("`goto/L` must be in insns table");
    // target = 0x0102 = 258
    let code = [goto_byte, 0x02, 0x01];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("goto/L must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 258);
}

#[test]
fn finishframe_lookahead_distinguishes_catch_rvmprof_and_nomatch() {
    // `finishframe_lookahead_at` must mirror RPython
    // `pyjitpl.py finishframe_exception` line-by-line —
    // sequential `catch_exception/L` then `rvmprof_code/ii` then
    // fall-through.
    //
    // pyre's emitted insns table currently lacks `rvmprof_code/ii`
    // (forward-prep — RPython emits it when rvmprof is enabled at
    // codewriter time). Test only the bytes that ARE in the table
    // and assert the helper shape compiles + the catch / no-match
    // arms route correctly.
    let live_byte = *insns_opname_to_byte()
        .get("live/")
        .expect("live/ must be in insns");
    let catch_byte = *insns_opname_to_byte()
        .get("catch_exception/L")
        .expect("catch_exception/L must be in insns");
    let goto_byte = *insns_opname_to_byte()
        .get("goto/L")
        .expect("goto/L must be in insns");

    // (1) live/ + catch_exception/L target=42 → CatchTarget(42).
    let code_catch = [live_byte, 0x00, 0x00, catch_byte, 0x2A, 0x00];
    assert_eq!(
        finishframe_lookahead_at(&code_catch, 0),
        FinishframeLookahead::CatchTarget(0x2A),
    );

    // (2) catch_exception/L without leading live/ → still
    //     CatchTarget (RPython's `if opcode == op_live: skip` is
    //     conditional, not required).
    let code_no_live_catch = [catch_byte, 0x10, 0x01];
    assert_eq!(
        finishframe_lookahead_at(&code_no_live_catch, 0),
        FinishframeLookahead::CatchTarget(0x110),
    );

    // (3) live/ + goto/L (NOT catch nor rvmprof) → NoMatch (the
    //     caller continues unwinding).
    let code_no_match = [live_byte, 0x00, 0x00, goto_byte, 0x00, 0x00];
    assert_eq!(
        finishframe_lookahead_at(&code_no_match, 0),
        FinishframeLookahead::NoMatch,
    );

    // (4) Position past end of code → NoMatch (decode fails).
    assert_eq!(
        finishframe_lookahead_at(&code_catch, 99),
        FinishframeLookahead::NoMatch,
    );
}

#[test]
fn step_through_catch_exception_with_active_exception_surfaces_typed_error() {
    // RPython `pyjitpl.py opimpl_catch_exception`:
    //   assert not self.metainterp.last_exc_value
    // Reaching catch_exception/L on the normal walk path with
    // last_exc_value=Some(_) violates the codewriter invariant —
    // surface as `CatchExceptionWithActiveException`.
    let catch_byte = *insns_opname_to_byte()
        .get("catch_exception/L")
        .expect("`catch_exception/L` must be in insns table");
    let code = [catch_byte, 0x2A, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs = distinct_const_refs(&mut tc, 4);
    let active_exc = regs[0];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    // The standing exception is metainterp-level state
    // (`pyjitpl.py:2417`), so it is seeded on the session.
    {
        let mut sess = session.borrow_mut();
        sess.last_exc_value = Some(active_exc);
    }
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("catch_exception/L with active exc must error");
    assert_eq!(
        err,
        DispatchError::CatchExceptionWithActiveException { pc: 0 }
    );
}

#[test]
fn step_through_catch_exception_advances_past_label_operand() {
    // `catch_exception/L` records nothing on the normal
    // walk (RPython `pyjitpl.py opimpl_catch_exception` is
    // an `assert not last_exc_value` only) and the walker advances
    // linearly past the 2-byte target.
    let catch_byte = *insns_opname_to_byte()
        .get("catch_exception/L")
        .expect("`catch_exception/L` must be in insns table");
    let code = [catch_byte, 0x2A, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("catch_exception/L must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 3,
        "catch_exception/L must advance past the 2-byte target operand",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "catch_exception/L records nothing on normal walk",
    );
}

#[test]
fn step_through_raise_records_outermost_finish_and_terminates() {
    // RPython `pyjitpl.py opimpl_raise` →
    // `finishframe_exception` (outermost-frame branch) →
    // `compile_exit_frame_with_exception` records
    // `FINISH(exc, descr=exit_frame_with_exception_descr_ref)`.
    // `raise/r` surfaces
    // `SubRaise` and `walk()`'s top-level SubRaise arm records the
    // outermost FINISH + converts to Terminate, so drive `walk()`.
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    // exc operand reads registers_r[2]
    let code = [raise_byte, 0x02];
    let mut tc = fresh_trace_ctx();
    let regs = distinct_const_refs(&mut tc, 4);
    let expected_exc = regs[2];
    let descr_done = done_descr_ref_for_tests();
    let descr_exc = make_fail_descr(99);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    fbw_finish_payload_reset();
    let (outcome, next_pc) = walk(&code, 0, &mut wc).expect("raise/r must dispatch");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
    assert_eq!(next_pc, 2);
    drop(wc);
    // The exception is stashed as an `is_exception` finish payload, NOT
    // recorded inline: `full_body_walk_trace`'s Terminate arm builds
    // `TraceAction::Finish { exit_with_exception: true }` and the compile
    // consumer records the single FINISH against
    // `exit_frame_with_exception_descr` (mirror of the value-return
    // `fbw_terminate_with_finish` path).  So no op is recorded in `walk()`.
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "raise/r must NOT record an inline FINISH — the payload is deferred",
    );
    assert!(
        fbw_finish_is_exception(),
        "top-level raise/r must mark the finish payload as an exception exit",
    );
    let (finish_value, finish_ty) =
        finish_payload_of(&outcome).expect("exception finish payload must be stashed");
    assert_eq!(finish_ty, Type::Ref, "portal-exit FINISH carries Type::Ref");
    assert_eq!(
        finish_value, expected_exc,
        "the stashed payload must carry the exception OpRef from registers_r[src]",
    );
}

#[test]
fn top_level_raise_settles_the_vable_token() {
    // `pyjitpl.py compile_exit_frame_with_exception` opens with
    // `store_token_in_vable()`, the same as `compile_done_with_this_frame`.
    // The exit therefore leaves a lazy, armed token rather than eagerly
    // writing every redirected field back.
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let code = [raise_byte, 0x02];
    let mut tc = fresh_trace_ctx();
    let mut vable_buf = vec![0u8; 65536];
    bind_fake_vable(&mut tc, &mut vable_buf);
    let regs = distinct_const_refs(&mut tc, 4);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    fbw_finish_payload_reset();
    let (outcome, _next_pc) = walk(&code, 0, &mut wc).expect("raise/r must dispatch");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
    drop(wc);
    let _ = finish_payload_of(&outcome);
    assert!(
        tc.num_ops() > ops_before,
        "the exception exit must record the virtualizable token protocol",
    );
    let last = tc.ops().last().expect("recorded op must exist").clone();
    assert_eq!(
        last.opcode,
        majit_ir::OpCode::GuardNotForced2,
        "the token protocol must end in GUARD_NOT_FORCED_2",
    );
    let token_store = tc
        .ops()
        .iter()
        .rev()
        .find(|op| op.opcode == majit_ir::OpCode::SetfieldGc)
        .expect("token protocol carries a SETFIELD_GC");
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    let recorded_descr = token_store
        .getdescr()
        .expect("token store carries a field descr");
    let field = recorded_descr
        .as_field_descr()
        .expect("token store's descr is a FieldDescr");
    assert_eq!(
        field.offset(),
        info.token_offset,
        "the tail store must target the vable_token slot",
    );
    let value = token_store
        .getarglist()
        .get(1)
        .expect("SetfieldGc args are [obj, value]")
        .to_opref();
    let token_op = tc
        .ops()
        .iter()
        .find(|op| op.pos().get() == value)
        .expect("stored token is produced by an operation");
    assert_eq!(
        token_op.opcode,
        majit_ir::OpCode::ForceToken,
        "the exception exit must store the live FORCE_TOKEN",
    );
}

#[test]
fn raise_r_emits_guard_class_when_concrete_exc_pinned_in_shadow() {
    // The concrete shadow is mutable and
    // tracked by every `registers_r[dst]` write
    // ([`write_ref_reg`]), so a `raise/r` reading the shadow finds
    // a reliable concrete pointer.  Allocate a real
    // `W_BaseException` so the deref against
    // `ob_header.ob_type` is sound; expect GuardClass + Finish
    // recorded and the heapcache class-known flag pinned.  Mirrors the
    // retired trait-side raise path.
    let exc_ptr = pyre_object::interp_exceptions::w_exception_new(
        pyre_object::interp_exceptions::ExcKind::ValueError,
        "shadow-walker probe",
    );
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let code = [raise_byte, 0x02];
    let mut tc = fresh_trace_ctx();
    // Use a non-constant OpRef so the heapcache class-known flag
    // actually pins. pyre's `is_class_known(constant)` returns
    // false (`heapcache.rs`) while `class_now_known(constant)`
    // is a no-op, so constants never round-trip through the
    // class-pinned cache.
    let exc_box = OpRef::input_arg_ref(0);
    let regs: Vec<OpRef> = vec![OpRef::NONE, OpRef::NONE, exc_box, OpRef::NONE];
    let concrete = vec![
        ConcreteValue::Null,
        ConcreteValue::Null,
        ConcreteValue::Ref(exc_ptr),
        ConcreteValue::Null,
    ];
    let descr_done = done_descr_ref_for_tests();
    let descr_exc = make_fail_descr(99);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: (concrete).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    // `raise/r` emits the GuardClass during dispatch, then surfaces
    // `SubRaise`; the top-level SubRaise arm stashes the exception as a
    // deferred `is_exception` finish payload.
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    fbw_finish_payload_reset();
    let (outcome, _next_pc) = walk(&code, 0, &mut wc).expect("raise/r must dispatch");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
    drop(wc);

    // Only the GuardClass op lands inline: `raise/r` records it during
    // dispatch (the GUARD_CLASS precedes the FINISH per
    // `pyjitpl.py`), then the top-level SubRaise stashes the
    // exception as an `is_exception` finish payload — the FINISH is
    // recorded by the FBW Terminate arm's compile consumer, not inline.
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "raise/r with pinned concrete exc must record GuardClass (FINISH deferred)",
    );
    assert!(
        fbw_finish_is_exception(),
        "top-level raise/r must mark the finish payload as an exception exit",
    );
    let (finish_value, _finish_ty) =
        finish_payload_of(&outcome).expect("exception finish payload must be stashed");
    assert_eq!(
        finish_value, exc_box,
        "the stashed payload must carry the exception OpRef",
    );
    let ops = tc.ops().to_vec();
    let guard = &ops[ops_before];
    assert_eq!(guard.opcode, majit_ir::OpCode::GuardClass);
    assert_eq!(
        (&*guard.getarglist())[0].to_opref(),
        exc_box,
        "GuardClass arg0 must be the exception OpRef",
    );
    // After the guard, the heapcache must mark the class as known
    // so a follow-on raise/r against the same exc_box wouldn't
    // re-emit GuardClass.
    assert!(
        tc.heap_cache().is_class_known(exc_box),
        "heapcache.class_now_known must fire alongside GuardClass",
    );
}

#[test]
fn step_through_reraise_at_top_level_records_outermost_finish() {
    // `reraise/` mirrors `raise/r` for the top-level
    // frame — it records `FINISH(last_exc_value,
    // exit_frame_with_exception_descr_ref)`. RPython parity:
    // `pyjitpl.py opimpl_reraise → popframe →
    // finishframe_exception` when the framestack is empty falls
    // through to `compile_exit_frame_with_exception(last_exc_box)`
    // (pyjitpl.py).
    let reraise_byte = *insns_opname_to_byte()
        .get("reraise/")
        .expect("`reraise/` must be in insns table");
    let code = [reraise_byte];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 4);
    let active_exc = regs_r[1];
    let descr_done = done_descr_ref_for_tests();
    let descr_exc = make_fail_descr(99);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    // The standing exception is metainterp-level state
    // (`pyjitpl.py:2417`), so it is seeded on the session.
    {
        let mut sess = session.borrow_mut();
        sess.last_exc_value = Some(active_exc);
    }
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    // `reraise/` surfaces
    // `SubRaise` and `walk()`'s top-level SubRaise arm records the
    // outermost FINISH + converts to Terminate.
    fbw_finish_payload_reset();
    let (outcome, next_pc) = walk(&code, 0, &mut wc).expect("reraise/ must dispatch");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
    assert_eq!(next_pc, 1, "reraise/ has no operand");
    drop(wc);
    // As with `raise/r`, the standing exception is stashed as an
    // `is_exception` payload and the outermost FINISH is recorded
    // downstream by the compile consumer, not inline in `walk()`.
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "reraise/ must NOT record an inline FINISH — the payload is deferred",
    );
    assert!(
        fbw_finish_is_exception(),
        "top-level reraise/ must mark the finish payload as an exception exit",
    );
    let (finish_value, finish_ty) =
        finish_payload_of(&outcome).expect("exception finish payload must be stashed");
    assert_eq!(finish_ty, Type::Ref, "portal-exit FINISH carries Type::Ref");
    assert_eq!(
        finish_value, active_exc,
        "the stashed payload must carry the standing last_exc_value OpRef",
    );
}

#[test]
fn step_through_reraise_without_last_exc_value_surfaces_typed_error() {
    // RPython `pyjitpl.py opimpl_reraise`:
    //   assert self.metainterp.last_exc_value
    // — reaching `reraise` without an active exception is a
    // codewriter invariant violation. Walker surfaces it as a
    // typed error rather than an arbitrary panic / silent
    // fall-through.
    let reraise_byte = *insns_opname_to_byte()
        .get("reraise/")
        .expect("`reraise/` must be in insns table");
    let code = [reraise_byte];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("reraise/ without last_exc_value must error");
    assert_eq!(err, DispatchError::ReraiseWithoutLastExcValue { pc: 0 });
}

#[test]
fn raise_at_top_level_populates_last_exc_value_before_finish() {
    // `raise/r` at top-level records FINISH and *also*
    // sets `ctx.last_exc_value` (RPython `pyjitpl.py`). The
    // post-condition matters because a future opcode in a
    // wrap-around (e.g. an unconditional `reraise/` after the
    // raise) would read it. Independently asserting the field
    // post-step locks in the side effect.
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let code = [raise_byte, 0x02];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 4);
    let exc = regs_r[2];
    let descr_done = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("raise/r must dispatch");
    assert_eq!(
        wc.last_exc_value(),
        Some(exc),
        "raise/r must populate ctx.last_exc_value before terminating",
    );
}

#[test]
fn inline_call_subraise_jumps_to_caller_catch_exception_target() {
    // Acceptance: callee's `raise/r` surfaces SubRaise to
    // the caller; caller's inline_call SubRaise arm probes
    // `op.next_pc` for `live/` + `catch_exception/L`, finds it,
    // sets `last_exc_value = exc`, and resumes at the catch target.
    // RPython parity: `pyjitpl.py finishframe_exception`
    // line-by-line — `op_live` skip then `op_catch_exception`
    // target jump.
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    let live_byte = *insns_opname_to_byte()
        .get("live/")
        .expect("`live/` must be in insns table");
    let catch_byte = *insns_opname_to_byte()
        .get("catch_exception/L")
        .expect("`catch_exception/L` must be in insns table");
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    // Callee: `raise r0`
    let callee_code: &'static [u8] = Box::leak(Box::new([raise_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = move |idx: usize| {
        if idx == 11 {
            Some(sub_body.clone())
        } else {
            None
        }
    };
    // Caller layout (matches the execute_pop_top helper shape):
    //   pc=0..6   inline_call_r_r descr=11 R=[r3] >r=r5
    //     opcode(1) + d(2) + R-len(1) + R[0](1) + dst(1)
    //   pc=6..9   live + 2-byte liveness offset (OFFSET_SIZE=2)
    //     opcode(1) + slot(2)
    //   pc=9..12  catch_exception/L target=12 (LE little-endian)
    //     opcode(1) + target(2)
    //   pc=12..14 handler body: ref_return r5
    //     opcode(1) + reg(1)
    let caller_code = vec![
        inline_byte,
        0x0B,
        0x00,
        0x01,
        0x03,
        0x05,
        live_byte,
        0x00,
        0x00,
        catch_byte,
        0x0C,
        0x00,
        ret_byte,
        0x05,
    ];
    assert_eq!(caller_code.len(), 14);
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let exc_arg = regs_r[3];
    let handler_ret = regs_r[5];
    let descr_done = done_descr_ref_for_tests();
    let descr_exc = make_fail_descr(99);
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[11] = make_jitcode_descr(11);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    fbw_finish_payload_reset();
    let (outcome, end_pc) = walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
    assert!(
        matches!(outcome, DispatchOutcome::Terminate { .. }),
        "caller must reach handler's ref_return and terminate (not bubble SubRaise)",
    );
    assert_eq!(
        end_pc, 14,
        "walker must terminate at handler's ref_return r5 (pc=12..14)",
    );
    assert_eq!(
        wc.last_exc_value(),
        Some(exc_arg),
        "caller's last_exc_value must be set to the exc OpRef from callee SubRaise",
    );
    drop(wc);
    // The outermost finish payload must carry the handler's
    // ref_return arg — r5, which still holds its pre-call
    // distinct_const_refs OpRef (caller's inline_call dst write
    // happens *only* on SubReturn, not SubRaise-then-catch).
    assert_eq!(
        finish_payload_of(&outcome),
        Some((handler_ret, Type::Ref)),
        "finish payload must exist and carry the handler's ref_return arg",
    );
}

#[test]
fn inline_call_subraise_without_caller_catch_bubbles_up_in_subwalk() {
    // Sub-walk SubRaise propagation: when the caller is itself a sub-walk
    // (`is_top_level=false`) and SubRaise reaches its `walk()`
    // loop with no `catch_exception/L` match, the loop returns
    // `SubRaise` unchanged so the parent's inline_call SubRaise arm
    // can scan its own op.next_pc for a catch handler.
    // RPython parity: `pyjitpl.py finishframe_exception` loops
    // through the framestack — only when `framestack` is exhausted
    // does it call `compile_exit_frame_with_exception`. Sub-walks
    // are not the framestack root.
    //
    // (The top-level FINISH conversion path is covered by
    // `inline_call_recursion_propagates_subraise_from_callee`
    // above.)
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    let goto_byte = *insns_opname_to_byte()
        .get("goto/L")
        .expect("`goto/L` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([raise_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = move |idx: usize| {
        if idx == 13 {
            Some(sub_body.clone())
        } else {
            None
        }
    };
    let caller_code = [
        inline_byte,
        0x0D,
        0x00,
        0x01,
        0x02,
        0x05,
        goto_byte,
        0x00,
        0x00,
    ];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let exc_arg = regs_r[2];
    let descr_done = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[13] = make_jitcode_descr(13);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        // Sub-walk frame: bubble-up behaviour, no FINISH conversion.
        is_top_level: false,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let ops_before = wc.trace_ctx.num_ops();
    let (outcome, _) = walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
    assert_eq!(
        outcome,
        DispatchOutcome::SubRaise {
            exc: exc_arg,
            exc_concrete: ConcreteValue::Null,
        },
        "sub-walk frame with no caller-side catch must bubble SubRaise through",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "sub-walk SubRaise must NOT record FINISH (only top-level converts)",
    );
}

#[test]
fn step_through_int_copy_advances_past_operand_bytes() {
    // `int_copy/i>i` reads the src `i` operand for OOR
    // validation, advances past 2 operand bytes, records nothing.
    // Dst writeback (`registers_i[dst] = registers_i[src]`) is
    // deferred — RPython `pyjitpl.py _opimpl_any_copy(box)
    // -> box` is a register rename only, no IR op.
    let int_copy_byte = *insns_opname_to_byte()
        .get("int_copy/i>i")
        .expect("`int_copy/i>i` must be in insns table");
    // src=2, dst=5 — distinct so a future writeback assertion can
    // distinguish src from dst slots.
    let code = [int_copy_byte, 0x02, 0x05];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 8);
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("int_copy/i>i must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 3,
        "int_copy/i>i must advance past src + dst register bytes",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "int_copy/i>i records no IR op (RPython parity)",
    );
}

#[test]
fn int_copy_writes_src_value_into_dst_register() {
    // Verify the dst writeback half of `int_copy/i>i`. The src
    // and dst slots must hold *different* OpRefs going in so the
    // assertion catches an accidental no-op.
    let int_copy_byte = *insns_opname_to_byte()
        .get("int_copy/i>i")
        .expect("`int_copy/i>i` must be in insns table");
    let code = [int_copy_byte, 0x02, 0x05]; // src=2, dst=5
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 8);
    let src_val_pre = regs_i[2];
    let dst_val_pre = regs_i[5];
    assert_ne!(
        src_val_pre, dst_val_pre,
        "fixture must seed src and dst with different OpRefs",
    );
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("int_copy/i>i must dispatch");
    assert_eq!(
        wc.registers_i.get(5).expect("int register in range"),
        src_val_pre,
        "int_copy must copy registers_i[src] into registers_i[dst] \
             (RPython _opimpl_any_copy + `>i` result coding)",
    );
    assert_eq!(
        wc.registers_i.get(2).expect("int register in range"),
        src_val_pre,
        "src register must remain unchanged",
    );
}

#[test]
fn int_copy_with_out_of_range_dst_register_surfaces_typed_error() {
    // dst byte indexes past `registers_i`; src is in range so the
    // src read succeeds and the dst write surfaces the OOR.
    let int_copy_byte = *insns_opname_to_byte()
        .get("int_copy/i>i")
        .expect("`int_copy/i>i` must be in insns table");
    let code = [int_copy_byte, 0x00, 0x09]; // src=0 (in range), dst=9 (OOR)
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("int_copy dst OOR must surface a typed error");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 9,
            len: 4,
            bank: "i",
        },
    );
}

#[test]
fn int_copy_with_out_of_range_src_register_surfaces_typed_error() {
    // src OOR validation parity with `raise/r`. Bank tag
    // is `"i"` to disambiguate from the Ref-bank OOR error.
    let int_copy_byte = *insns_opname_to_byte()
        .get("int_copy/i>i")
        .expect("`int_copy/i>i` must be in insns table");
    let code = [int_copy_byte, 0x07, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(), // empty — index 7 must surface OOR
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("int_copy/i>i must read its src operand");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "i",
        },
    );
}

// E1a: `ref_copy/r>r` walker arm tests are gated on the build-time
// `pipeline.insns` table picking up the `ref_copy/r>r` key. Today
// the analyzed source set (pyre-object + pyre-interpreter +
// pyre-jit/src/eval.rs) does not exercise the codewriter's
// chordal-reuse boundary that triggers `emit_ref_copy!`, so the
// key never enters `INSNS_OPNAME_TO_BYTE`. The walker arm is
// correctly wired (mirrors `int_copy/i>i`); these tests fire
// automatically once any analyzed source path emits a `ref_copy`.
//
// Broader finding: `INSNS_OPNAME_TO_BYTE` (build-time
// `pipeline.insns`) and `wellknown_bh_insns` (runtime
// `JitCodeBuilder` writers) currently use different byte
// assignments for the same key (`int_copy/i>i` is 0 in pipeline,
// `BC_MOVE_I = 21` in wellknown). Production walker dispatch over
// runtime-emitted jitcode bytes therefore needs a table-
// unification step before any `dispatch_via_miframe` invocation
// can read production bytes. Tracked separately as a
// prerequisite.
#[test]
fn step_through_ref_copy_advances_past_operand_bytes() {
    // `ref_copy/r>r` Ref-bank sibling of `int_copy/i>i`.
    // Same operand layout `r>r`: 1B src + 1B dst, no IR op recorded.
    let ref_copy_byte = *insns_opname_to_byte()
        .get("ref_copy/r>r")
        .expect("`ref_copy/r>r` must be in insns table");
    let code = [ref_copy_byte, 0x02, 0x05]; // src=2, dst=5
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("ref_copy/r>r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 3,
        "ref_copy/r>r must advance past src + dst register bytes",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "ref_copy/r>r records no IR op (RPython parity)",
    );
}

#[test]
fn ref_copy_writes_src_value_into_dst_register() {
    // Verify the dst writeback half of `ref_copy/r>r`.
    let ref_copy_byte = *insns_opname_to_byte()
        .get("ref_copy/r>r")
        .expect("`ref_copy/r>r` must be in insns table");
    let code = [ref_copy_byte, 0x02, 0x05]; // src=2, dst=5
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let src_val_pre = regs_r[2];
    let dst_val_pre = regs_r[5];
    assert_ne!(
        src_val_pre, dst_val_pre,
        "fixture must seed src and dst with different OpRefs",
    );
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: regs_r[7],
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("ref_copy/r>r must dispatch");
    assert_eq!(
        wc.registers_r.get(5).expect("ref register in range"),
        src_val_pre,
        "ref_copy must copy registers_r[src] into registers_r[dst] \
             (RPython _opimpl_any_copy + `>r` result coding)",
    );
    assert_eq!(
        wc.registers_r.get(2).expect("ref register in range"),
        src_val_pre,
        "src register must remain unchanged",
    );
    assert_eq!(
        wc.frame_state.borrow().vstack_last_ref,
        regs_r[7],
        "a link renaming is not the opcode's result, so ref_copy must leave \
             the operand-stack TOS candidate alone",
    );
}

#[test]
fn ref_copy_keeps_operand_stack_tos_candidate() {
    // `output = ''; for b in input:` flattens to GET_ITER's
    // `setarrayitem_vable_r(stack_base, iter)` followed by the block-exit
    // renaming `ref_copy('' -> r_output)`.  The renaming must not replace the
    // iterator as the TOS box `reconcile_vstack_at_boundary` lands for
    // GET_ITER.
    let ref_copy_byte = *insns_opname_to_byte()
        .get("ref_copy/r>r")
        .expect("`ref_copy/r>r` must be in insns table");
    let code = [ref_copy_byte, 0x02, 0x05]; // src=2, dst=5
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let pushed = regs_r[7];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: pushed,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("ref_copy/r>r must dispatch");
    assert_eq!(wc.registers_r.get(5), Some(regs_r[2]));
    assert_eq!(
        wc.frame_state.borrow().vstack_last_ref,
        pushed,
        "a renaming copy must leave the operand-stack TOS candidate alone",
    );
}

#[test]
fn ref_copy_with_out_of_range_dst_register_surfaces_typed_error() {
    let ref_copy_byte = *insns_opname_to_byte()
        .get("ref_copy/r>r")
        .expect("`ref_copy/r>r` must be in insns table");
    let code = [ref_copy_byte, 0x00, 0x09]; // src=0 (in range), dst=9 (OOR)
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("ref_copy dst OOR must surface a typed error");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 9,
            len: 4,
            bank: "r",
        },
    );
}

#[test]
fn ref_copy_with_out_of_range_src_register_surfaces_typed_error() {
    let ref_copy_byte = *insns_opname_to_byte()
        .get("ref_copy/r>r")
        .expect("`ref_copy/r>r` must be in insns table");
    let code = [ref_copy_byte, 0x07, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(), // empty — index 7 must surface OOR
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("ref_copy/r>r must read its src operand");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "r",
        },
    );
}

/// Drive a single `int_<binop>/ii>i` handler: the codewriter
/// encodes `[opcode, src1, src2, dst]`. Asserts the recorder
/// captured `OpCode::<expected>` with `[regs_i[src1],
/// regs_i[src2]]` and that `regs_i[dst]` was written with the
/// recorder's result OpRef.
fn drive_int_binop(opname: &str, expected_opcode: majit_ir::OpCode) {
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    // src=2, src2=4, dst=6 — chosen to be distinct so misordered
    // operand decoding surfaces in the assertion.
    let code = [byte, 0x02, 0x04, 0x06];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 8);
    let arg0 = regs_i[2];
    let arg1 = regs_i[4];
    let dst_pre = regs_i[6];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc)
        .unwrap_or_else(|e| panic!("`{opname}` must dispatch — got {:?}", e));
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 4, "`{opname}` operand layout `ii>i` = 3 bytes");
    let dst_post = wc.registers_i.get(6).expect("int register in range");
    assert_ne!(
        dst_post, dst_pre,
        "`{opname}` must write a fresh OpRef into registers_i[dst]",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "`{opname}` must record exactly one op",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(
        last.opcode, expected_opcode,
        "`{opname}` must record `{:?}`",
        expected_opcode,
    );
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![arg0, arg1],
        "`{opname}` args must be [registers_i[src1], registers_i[src2]] in source order",
    );
    assert_eq!(
        dst_post,
        last.pos().get(),
        "`{opname}` dst must hold the recorder's result OpRef (op.pos().get())",
    );
}

#[test]
fn int_binop_records_opcode_with_both_operands_and_writes_dst() {
    let cases = [
        ("int_add", "int_add/ii>i", majit_ir::OpCode::IntAdd),
        ("int_sub", "int_sub/ii>i", majit_ir::OpCode::IntSub),
        ("int_mul", "int_mul/ii>i", majit_ir::OpCode::IntMul),
        ("int_and", "int_and/ii>i", majit_ir::OpCode::IntAnd),
        (
            "int_signext",
            "int_signext/ii>i",
            majit_ir::OpCode::IntSignext,
        ),
        ("int_or", "int_or/ii>i", majit_ir::OpCode::IntOr),
        ("int_xor", "int_xor/ii>i", majit_ir::OpCode::IntXor),
        ("int_rshift", "int_rshift/ii>i", majit_ir::OpCode::IntRshift),
        (
            "uint_rshift",
            "uint_rshift/ii>i",
            majit_ir::OpCode::UintRshift,
        ),
        (
            "uint_mul_high",
            "uint_mul_high/ii>i",
            majit_ir::OpCode::UintMulHigh,
        ),
        ("uint_lt", "uint_lt/ii>i", majit_ir::OpCode::UintLt),
        ("uint_le", "uint_le/ii>i", majit_ir::OpCode::UintLe),
        ("uint_gt", "uint_gt/ii>i", majit_ir::OpCode::UintGt),
        ("uint_ge", "uint_ge/ii>i", majit_ir::OpCode::UintGe),
        ("int_eq", "int_eq/ii>i", majit_ir::OpCode::IntEq),
        ("int_ne", "int_ne/ii>i", majit_ir::OpCode::IntNe),
        ("int_lt", "int_lt/ii>i", majit_ir::OpCode::IntLt),
        ("int_le", "int_le/ii>i", majit_ir::OpCode::IntLe),
        ("int_gt", "int_gt/ii>i", majit_ir::OpCode::IntGt),
        ("int_ge", "int_ge/ii>i", majit_ir::OpCode::IntGe),
    ];
    for (name, opname, opcode) in cases {
        assert!(opname.starts_with(name), "case {name}");
        drive_int_binop(opname, opcode);
    }
}

/// Drive `int_between/iii>i` and return the recorded ops plus
/// the post-handler `dst` slot.  `inputs` describes how the
/// three operand slots are populated.
enum BetweenOperand {
    ConstInt(i64),
    NonConstWithConcrete(i64),
}

fn drive_int_between(
    b1: BetweenOperand,
    b2: BetweenOperand,
    b3: BetweenOperand,
) -> (Vec<majit_ir::OpCode>, OpRef, OpRef) {
    // `int_between/iii>i` is recorder-side only (decomposed at
    // record time) and is not currently emitted into pyre's
    // pipeline.insns table, so `insns_opname_to_byte()` lacks
    // a byte for it.  Call `int_between_record` directly with a
    // synthetic `DecodedOp` to exercise the handler.  Operand
    // layout: `[opcode, src1, src2, src3, dst]` — opcode byte is
    // unused by the handler (PC offsets into `code` from `op.pc`).
    let code = [0x00u8, 0x02, 0x04, 0x06, 0x08];
    let op = crate::jitcode_runtime::DecodedOp {
        key: "int_between/iii>i",
        opname: "int_between",
        argcodes: "iii>i",
        pc: 0,
        next_pc: 5,
    };
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 16);
    let mint = |tc: &mut TraceCtx, op: &BetweenOperand| match op {
        BetweenOperand::ConstInt(v) => tc.const_int(*v),
        BetweenOperand::NonConstWithConcrete(v) => {
            // Materialize a non-Const OpRef by recording a placeholder
            // op, then stamp its concrete value.  IntAdd with two
            // ConstInts is a valid stand-in carrier.
            let lhs = tc.const_int(0);
            let opref = tc.record_op(majit_ir::OpCode::IntAdd, &[lhs, lhs]);
            tc.set_opref_concrete(opref, majit_ir::Value::Int(*v));
            opref
        }
    };
    regs_i[2] = mint(&mut tc, &b1);
    regs_i[4] = mint(&mut tc, &b2);
    regs_i[6] = mint(&mut tc, &b3);
    let arg_b1 = regs_i[2];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) =
        int_between_record(&code, &op, &mut wc).expect("int_between_record must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 5, "operand layout `iii>i` consumes 4 bytes");
    let dst_post = wc.registers_i.get(8).expect("int register in range");
    drop(wc);
    let new_ops: Vec<majit_ir::OpCode> = tc
        .ops()
        .iter()
        .skip(ops_before)
        .map(|op| op.opcode)
        .collect();
    (new_ops, dst_post, arg_b1)
}

/// `pyjitpl.py execute_and_record` — when every argbox
/// is `ConstInt`, a pure op like `INT_SUB` / `INT_EQ` is folded
/// via `wrap_constant` and never reaches `_record_helper`.
/// `opimpl_int_between(ConstInt, ConstInt, ConstInt)` chains three
/// pure binops on all-Const inputs, so 0 ops must be recorded and
/// the destination must hold the folded ConstInt result.
#[test]
fn int_between_const_inputs_with_unit_width_takes_inteq_fast_path() {
    let (ops, dst_post, _) = drive_int_between(
        BetweenOperand::ConstInt(5),
        BetweenOperand::ConstInt(7),
        BetweenOperand::ConstInt(6),
    );
    assert!(
        ops.is_empty(),
        "all-Const inputs must fold without recording — got {ops:?}",
    );
    // `b5 = 6 - 5 = 1` → fast path → `IntEq(b2=7, b1=5) = 0`.
    assert_eq!(
        dst_post.inline_const_to_value(),
        Some(majit_ir::Value::Int(0)),
        "ConstInt(1) fast path on all-Const inputs must fold to ConstInt(0)",
    );
}

/// All-Const generic path: `pyjitpl.py` folds the three
/// pure binops without recording.  Destination must carry the
/// folded UINT_LT result.
#[test]
fn int_between_const_inputs_with_wide_range_takes_uintlt_generic_path() {
    let (ops, dst_post, _) = drive_int_between(
        BetweenOperand::ConstInt(5),
        BetweenOperand::ConstInt(7),
        BetweenOperand::ConstInt(10),
    );
    assert!(
        ops.is_empty(),
        "all-Const inputs must fold without recording — got {ops:?}",
    );
    // `b5 = 10 - 5 = 5` (not 1) → generic → `b4 = 7 - 5 = 2`,
    // `UintLt(2, 5) = 1`.
    assert_eq!(
        dst_post.inline_const_to_value(),
        Some(majit_ir::Value::Int(1)),
        "wide-range path on all-Const inputs must fold to ConstInt(1)",
    );
}

/// Regression: when `b1`/`b3` are non-constant OpRefs with
/// observed concrete values satisfying `b3-b1==1`, the recorder
/// must still take the generic `INT_SUB + UINT_LT` path.
/// Specializing to `INT_EQ` would bake a `b3-b1==1` invariant
/// without a runtime guard (`box_value` only reports the observed
/// sample), miscompiling any future execution where the same
/// boxes carry different live values.
#[test]
fn int_between_non_const_inputs_observing_unit_width_takes_generic_path() {
    let (ops, _, _) = drive_int_between(
        BetweenOperand::NonConstWithConcrete(5),
        BetweenOperand::NonConstWithConcrete(7),
        BetweenOperand::NonConstWithConcrete(6),
    );
    assert_eq!(
        ops,
        vec![
            majit_ir::OpCode::IntSub,
            majit_ir::OpCode::IntSub,
            majit_ir::OpCode::UintLt,
        ],
        "non-const inputs must NOT take INT_EQ fast path; expected \
             [IntSub(b5), IntSub(b4), UintLt] — got {ops:?}",
    );
}

/// Drive a single `float_<binop>/ff>f` handler. Same shape as
/// `drive_int_binop` but on the float bank.
fn drive_float_binop(opname: &str, expected_opcode: majit_ir::OpCode) {
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    let code = [byte, 0x02, 0x04, 0x06];
    let mut tc = fresh_trace_ctx();
    let regs_f = distinct_const_refs(&mut tc, 8);
    let arg0 = regs_f[2];
    let arg1 = regs_f[4];
    let dst_pre = regs_f[6];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::new(regs_f.iter().copied()),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc)
        .unwrap_or_else(|e| panic!("`{opname}` must dispatch — got {:?}", e));
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 4, "`{opname}` operand layout `ff>f` = 3 bytes");
    let dst_post = wc.registers_f.get(6).expect("float register in range");
    assert_ne!(
        dst_post, dst_pre,
        "`{opname}` must write a fresh OpRef into registers_f[dst]",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "`{opname}` must record exactly one op",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, expected_opcode);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![arg0, arg1]
    );
    assert_eq!(dst_post, last.pos().get());
}

#[test]
fn float_binop_records_opcode() {
    let cases = [
        ("float_add", "float_add/ff>f", majit_ir::OpCode::FloatAdd),
        ("float_sub", "float_sub/ff>f", majit_ir::OpCode::FloatSub),
        (
            "float_truediv",
            "float_truediv/ff>f",
            majit_ir::OpCode::FloatTrueDiv,
        ),
    ];
    for (name, opname, opcode) in cases {
        assert!(opname.starts_with(name), "case {name}");
        drive_float_binop(opname, opcode);
    }
}

/// Drive a single `float_<unop>/f>f` handler. Same shape pattern as
/// `drive_float_binop` minus one read.
fn drive_float_unop(opname: &str, expected_opcode: majit_ir::OpCode) {
    // `f>f` shape: 1B src + 1B dst = 2 operand bytes after opcode.
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    let code = [byte, 0x02, 0x05];
    let mut tc = fresh_trace_ctx();
    let regs_f = distinct_const_refs(&mut tc, 8);
    let arg = regs_f[2];
    let dst_pre = regs_f[5];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::new(regs_f.iter().copied()),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).unwrap_or_else(|_| panic!("`{opname}` must dispatch"));
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 3, "`{opname}` operand layout `f>f` = 2 bytes");
    let dst_post = wc.registers_f.get(5).expect("float register in range");
    assert_ne!(dst_post, dst_pre);
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, expected_opcode);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![arg],
        "`{opname}` args must be [registers_f[src]]",
    );
    assert_eq!(dst_post, last.pos().get());
}

#[test]
fn float_unop_records_opcode_and_writes_dst() {
    let cases = [
        ("float_neg", "float_neg/f>f", majit_ir::OpCode::FloatNeg),
        ("float_abs", "float_abs/f>f", majit_ir::OpCode::FloatAbs),
    ];
    for (name, opname, opcode) in cases {
        assert!(opname.starts_with(name), "case {name}");
        drive_float_unop(opname, opcode);
    }
}

/// Drive a single `int_<unop>/i>i` handler. Same shape pattern as
/// `drive_int_binop` minus one read.
fn drive_int_unop(opname: &str, expected_opcode: majit_ir::OpCode) {
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    let code = [byte, 0x02, 0x05]; // src=2, dst=5
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 8);
    let arg = regs_i[2];
    let dst_pre = regs_i[5];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc)
        .unwrap_or_else(|e| panic!("`{opname}` must dispatch — got {:?}", e));
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 3, "`{opname}` operand layout `i>i` = 2 bytes");
    let dst_post = wc.registers_i.get(5).expect("int register in range");
    assert_ne!(dst_post, dst_pre);
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, expected_opcode);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![arg]
    );
    assert_eq!(dst_post, last.pos().get());
}

#[test]
fn int_unop_records_opcode() {
    let cases = [
        ("int_neg", "int_neg/i>i", majit_ir::OpCode::IntNeg),
        ("int_invert", "int_invert/i>i", majit_ir::OpCode::IntInvert),
        (
            "int_is_true",
            "int_is_true/i>i",
            majit_ir::OpCode::IntIsTrue,
        ),
    ];
    for (name, opname, opcode) in cases {
        assert!(opname.starts_with(name), "case {name}");
        drive_int_unop(opname, opcode);
    }
}

/// `int_is_zero` is the `not a` sibling `pyjitpl.py` generates beside
/// `int_is_true`; `jtransform.py` `_rewrite_equality` is what puts it in a
/// body, by folding `int_eq(x, 0)`. Without an arm the walker answers
/// `DispatchError::UnsupportedOpname` and aborts the trace, so this asserts
/// the arm exists rather than only that the recorded op is right.
#[test]
fn int_is_zero_records_intiszero() {
    drive_int_unop("int_is_zero/i>i", majit_ir::OpCode::IntIsZero);
}

#[test]
fn int_same_as_is_eliminated_from_generated_insns_table() {
    // RPython `jtransform.py rewrite_op_same_as` removes
    // `same_as` before assembly. The walker keeps a handler arm for
    // forward-prep, but the production insns table should not contain
    // the opname unless a future codewriter path legitimately emits it.
    assert!(
        !insns_opname_to_byte().contains_key("int_same_as/i>i"),
        "`int_same_as/i>i` appeared in the generated insns table; \
             verify same_as elimination before adding a decode test"
    );
}

/// Drive `ptr_eq/rr>i` or `ptr_ne/rr>i`. Shape `rr>i`: read 2
/// r-regs, record, write to i-bank.
fn drive_ptr_compare(opname: &str, expected_opcode: majit_ir::OpCode) {
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    let code = [byte, 0x02, 0x04, 0x06]; // r-src1=2, r-src2=4, i-dst=6
    let mut tc = TraceCtx::for_test_types(&[Type::Ref, Type::Ref]);
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    // Non-const operands so the compare records: PyPy folds a ptr compare only
    // when both operands are `Const`; the distinct-const fold is covered by
    // `ptr_eq_folds_two_distinct_const_refs_without_recording`. InputArg refs
    // (declared by `for_test_types`) are the recordable non-const operand.
    regs_r[2] = OpRef::input_arg_ref(0);
    regs_r[4] = OpRef::input_arg_ref(1);
    let regs_i = distinct_const_refs(&mut tc, 8);
    let arg0 = regs_r[2];
    let arg1 = regs_r[4];
    let dst_pre = regs_i[6];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc)
        .unwrap_or_else(|e| panic!("`{opname}` must dispatch — got {:?}", e));
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 4, "`{opname}` operand layout `rr>i` = 3 bytes");
    let dst_post = wc.registers_i.get(6).expect("int register in range");
    assert_ne!(dst_post, dst_pre);
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, expected_opcode);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![arg0, arg1]
    );
    assert_eq!(dst_post, last.pos().get());
}

#[test]
fn ptr_compare_records_opcode_into_int_dst() {
    let cases = [
        ("ptr_eq", "ptr_eq/rr>i", majit_ir::OpCode::PtrEq),
        ("ptr_ne", "ptr_ne/rr>i", majit_ir::OpCode::PtrNe),
        (
            "instance_ptr_eq",
            "instance_ptr_eq/rr>i",
            majit_ir::OpCode::InstancePtrEq,
        ),
        (
            "instance_ptr_ne",
            "instance_ptr_ne/rr>i",
            majit_ir::OpCode::InstancePtrNe,
        ),
    ];
    for (name, opname, opcode) in cases {
        assert!(opname.starts_with(name), "case {name}");
        drive_ptr_compare(opname, opcode);
    }
}

#[test]
fn ptr_eq_folds_two_distinct_const_refs_without_recording() {
    // `execute_and_record` const-folds a pure op whose operands are all
    // `Const`; two distinct const refs answer `is` (PtrEq -> false) at trace
    // time without a recorded PTR_EQ.
    let byte = *insns_opname_to_byte()
        .get("ptr_eq/rr>i")
        .expect("`ptr_eq/rr>i` must be in insns table");
    let code = [byte, 0x00, 0x01, 0x00]; // `rr>i`: r-src1=0, r-src2=1, i-dst=0
    let mut tc = fresh_trace_ctx();
    let a = tc.const_ref(0xC0DE_0000);
    let b = tc.const_ref(0xC0DE_0001);
    let mut regs_r = [a, b];
    let mut regs_i = [OpRef::None];
    let mut concrete_r = [ConcreteValue::Null, ConcreteValue::Null];
    let ops_before = tc.num_ops();
    let (_, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut regs_i)
        .expect("ptr_eq on two const refs must fold");
    assert_eq!(next_pc, 4);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "distinct const-ref operands fold without recording a PtrEq",
    );
    assert_eq!(
        tc.concrete_of_opref(regs_i[0]),
        Some(majit_ir::Value::Int(0)),
        "distinct const refs are not identical -> PtrEq folds to 0",
    );
}

#[test]
fn int_binop_folds_two_const_int_operands_without_recording() {
    // `execute_and_record` const-folds a pure op whose operands are all
    // `Const` (`_all_constants` short-circuit) — the int binop loop routes
    // through `self.execute`, so an all-const `int_add` never records.
    let byte = *insns_opname_to_byte()
        .get("int_add/ii>i")
        .expect("`int_add/ii>i` must be in insns table");
    let code = [byte, 0x00, 0x01, 0x02]; // `ii>i`: src1=0, src2=1, dst=2
    let mut tc = fresh_trace_ctx();
    let lhs = tc.const_int(40);
    let rhs = tc.const_int(2);
    let mut regs_i = [lhs, rhs, OpRef::None];
    let ops_before = tc.num_ops();
    let (_, next_pc) = run_hint_step(&code, &mut tc, &mut [], &mut [], &mut regs_i)
        .expect("int_add on two const ints must fold");
    assert_eq!(next_pc, 4);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "all-const operands fold without recording an IntAdd",
    );
    assert_eq!(
        regs_i[2].inline_const_to_value(),
        Some(majit_ir::Value::Int(42)),
        "dst must hold the folded ConstInt",
    );
}

#[test]
fn int_unop_folds_a_const_int_operand_without_recording() {
    // The unary loop (`int_neg` / `int_invert` / `int_is_true`) also routes
    // through `self.execute`, so a `Const` operand folds without recording.
    let byte = *insns_opname_to_byte()
        .get("int_neg/i>i")
        .expect("`int_neg/i>i` must be in insns table");
    let code = [byte, 0x00, 0x01]; // `i>i`: src=0, dst=1
    let mut tc = fresh_trace_ctx();
    let operand = tc.const_int(7);
    let mut regs_i = [operand, OpRef::None];
    let ops_before = tc.num_ops();
    let (_, next_pc) = run_hint_step(&code, &mut tc, &mut [], &mut [], &mut regs_i)
        .expect("int_neg on a const int must fold");
    assert_eq!(next_pc, 3);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "a const operand folds without recording an IntNeg",
    );
    assert_eq!(
        regs_i[1].inline_const_to_value(),
        Some(majit_ir::Value::Int(-7)),
        "dst must hold the folded ConstInt",
    );
}

/// [`run_hint_step`] companion for float-bank steps: same lean harness
/// with caller-supplied `registers_f` (and `registers_i` for the
/// `ff>i` compares).
fn run_float_step(
    code: &[u8],
    tc: &mut TraceCtx,
    regs_f: &mut [OpRef],
    regs_i: &mut [OpRef],
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let outer_jitcode_index = test_outer_resume_jitcode_index();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::new(regs_f.iter().copied()),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: Some(0),
        outer_jitcode_index,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let result = step(code, 0, &mut wc);
    regs_f.copy_from_slice(&wc.registers_f.to_vec());
    regs_i.copy_from_slice(&wc.registers_i.to_vec());
    result
}

#[test]
fn float_binop_folds_two_const_float_operands_without_recording() {
    let byte = *insns_opname_to_byte()
        .get("float_add/ff>f")
        .expect("`float_add/ff>f` must be in insns table");
    let code = [byte, 0x00, 0x01, 0x02]; // `ff>f`: src1=0, src2=1, dst=2
    let mut tc = fresh_trace_ctx();
    let lhs = tc.const_float((40.0f64).to_bits() as i64);
    let rhs = tc.const_float((2.0f64).to_bits() as i64);
    let mut regs_f = [lhs, rhs, OpRef::None];
    let ops_before = tc.num_ops();
    let (_, next_pc) = run_float_step(&code, &mut tc, &mut regs_f, &mut [])
        .expect("float_add on two const floats must fold");
    assert_eq!(next_pc, 4);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "all-const operands fold without recording a FloatAdd",
    );
    assert_eq!(
        regs_f[2].inline_const_to_value(),
        Some(majit_ir::Value::Float(42.0)),
        "dst must hold the folded ConstFloat",
    );
}

#[test]
fn float_unop_folds_a_const_float_operand_without_recording() {
    let byte = *insns_opname_to_byte()
        .get("float_neg/f>f")
        .expect("`float_neg/f>f` must be in insns table");
    let code = [byte, 0x00, 0x01]; // `f>f`: src=0, dst=1
    let mut tc = fresh_trace_ctx();
    let operand = tc.const_float((1.5f64).to_bits() as i64);
    let mut regs_f = [operand, OpRef::None];
    let ops_before = tc.num_ops();
    let (_, next_pc) = run_float_step(&code, &mut tc, &mut regs_f, &mut [])
        .expect("float_neg on a const float must fold");
    assert_eq!(next_pc, 3);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "a const operand folds without recording a FloatNeg",
    );
    assert_eq!(
        regs_f[1].inline_const_to_value(),
        Some(majit_ir::Value::Float(-1.5)),
        "dst must hold the folded ConstFloat",
    );
}

#[test]
fn float_compare_folds_two_const_float_operands_without_recording() {
    let byte = *insns_opname_to_byte()
        .get("float_lt/ff>i")
        .expect("`float_lt/ff>i` must be in insns table");
    let code = [byte, 0x00, 0x01, 0x00]; // `ff>i`: f-src1=0, f-src2=1, i-dst=0
    let mut tc = fresh_trace_ctx();
    let lhs = tc.const_float((1.0f64).to_bits() as i64);
    let rhs = tc.const_float((2.0f64).to_bits() as i64);
    let mut regs_f = [lhs, rhs];
    let mut regs_i = [OpRef::None];
    let ops_before = tc.num_ops();
    let (_, next_pc) = run_float_step(&code, &mut tc, &mut regs_f, &mut regs_i)
        .expect("float_lt on two const floats must fold");
    assert_eq!(next_pc, 4);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "all-const operands fold without recording a FloatLt",
    );
    assert_eq!(
        regs_i[0].inline_const_to_value(),
        Some(majit_ir::Value::Int(1)),
        "1.0 < 2.0 folds to ConstInt(1)",
    );
}

#[test]
fn fused_float_compare_folds_const_operands_without_guards() {
    // The fused float compares route their condbox through
    // `self.execute(rop.FLOAT_*, b1, b2)` — all-const operands fold and the
    // `Const` condbox emits no guard.
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_float_lt/ffL")
        .expect("`goto_if_not_float_lt/ffL` must be in insns table");
    let code = [byte, 0x00, 0x01, 0x09, 0x00];
    let mut tc = fresh_trace_ctx();
    let lhs = tc.const_float((1.0f64).to_bits() as i64);
    let rhs = tc.const_float((2.0f64).to_bits() as i64);
    let mut regs_f = [lhs, rhs];
    let (_, next_pc) = run_float_step(&code, &mut tc, &mut regs_f, &mut [])
        .expect("a const float compare has a static direction");
    assert_eq!(next_pc, code.len(), "1.0 < 2.0 falls through");
    assert_eq!(
        tc.num_ops(),
        0,
        "const float compare records no FloatLt and no guard"
    );
}

#[test]
fn profiler_counts_ops_and_recorded_ops_like_execute_and_record() {
    // `execute_and_record` counts `OPS` before the `_all_constants` fold
    // and `_record_helper` adds `RECORDED_OPS` only when the op records;
    // the `b1 is b2` fast path returns before `self.execute` and counts
    // neither.
    let byte = *insns_opname_to_byte()
        .get("int_add/ii>i")
        .expect("`int_add/ii>i` must be in insns table");
    let code = [byte, 0x00, 0x01, 0x02];

    // Record path: non-const operands -> OPS + RECORDED_OPS.
    let mut tc = TraceCtx::for_test_types(&[Type::Int, Type::Int]);
    let mut regs_i = [
        OpRef::input_arg_int(0),
        OpRef::input_arg_int(1),
        OpRef::None,
    ];
    run_hint_step(&code, &mut tc, &mut [], &mut [], &mut regs_i).expect("int_add must record");
    let snap = tc.profiler().snapshot();
    assert_eq!(snap.ops, 1, "recorded int_add lands in the OPS bucket");
    assert_eq!(snap.recorded_ops, 1, "recorded int_add adds RECORDED_OPS");

    // Fold path: all-const operands -> OPS only.
    let mut fold_tc = fresh_trace_ctx();
    let lhs = fold_tc.const_int(1);
    let rhs = fold_tc.const_int(2);
    let mut fold_regs = [lhs, rhs, OpRef::None];
    run_hint_step(&code, &mut fold_tc, &mut [], &mut [], &mut fold_regs)
        .expect("int_add must fold");
    let snap = fold_tc.profiler().snapshot();
    assert_eq!(snap.ops, 1, "a folded op still lands in the OPS bucket");
    assert_eq!(snap.recorded_ops, 0, "a folded op is not RECORDED_OPS");

    // Same-box fast path: neither counter.
    let eq_byte = *insns_opname_to_byte()
        .get("int_eq/ii>i")
        .expect("`int_eq/ii>i` must be in insns table");
    let eq_code = [eq_byte, 0x00, 0x00, 0x01];
    let mut same_tc = TraceCtx::for_test_types(&[Type::Int]);
    let mut same_regs = [OpRef::input_arg_int(0), OpRef::None];
    run_hint_step(&eq_code, &mut same_tc, &mut [], &mut [], &mut same_regs)
        .expect("same-box int_eq must fold");
    let snap = same_tc.profiler().snapshot();
    assert_eq!(snap.ops, 0, "the same-box fast path skips self.execute");
    assert_eq!(snap.recorded_ops, 0);
}

#[test]
fn float_add_with_out_of_range_src_register_surfaces_typed_error() {
    let byte = *insns_opname_to_byte()
        .get("float_add/ff>f")
        .expect("`float_add/ff>f` must be in insns table");
    let code = [byte, 0x07, 0x00, 0x00]; // src=7, registers_f empty
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("float_add must read its src operand");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "f",
        },
    );
}

#[test]
fn int_add_with_out_of_range_src_register_surfaces_typed_error() {
    // OOR validation parity with int_copy. Bank tag = "i".
    let byte = *insns_opname_to_byte()
        .get("int_add/ii>i")
        .expect("`int_add/ii>i` must be in insns table");
    let code = [byte, 0x07, 0x00, 0x00]; // src=7, registers_i empty
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("int_add must read its src operand");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "i",
        },
    );
}

#[test]
fn int_add_with_out_of_range_dst_register_surfaces_typed_error() {
    // src reads succeed, dst write surfaces OOR. Catches the
    // reverse-direction encoding bugs the src-only test misses.
    let byte = *insns_opname_to_byte()
        .get("int_add/ii>i")
        .expect("`int_add/ii>i` must be in insns table");
    let code = [byte, 0x00, 0x01, 0x09]; // dst=9, registers_i.len()=4 → OOR
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("int_add dst OOR must surface a typed error");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 9,
            len: 4,
            bank: "i",
        },
    );
}

#[test]
fn unsupported_opname_surfaces_typed_error() {
    // The only remaining opname with no arm, so the only one that can
    // exercise the catch-all `UnsupportedOpname` error path.
    // `vtable_method_ptr/rd>i` is a pyre-only backend adaptation
    // (emitted by `OpKind::VtableMethodPtr` / `assembler.rs`) without a
    // PyPy analog: Python dispatch resolves through `cpu.bh_call_*` at
    // runtime rather than reifying a method pointer into the bytecode
    // stream, and its blackhole side is a deliberate bail
    // (`handler_vtable_method_ptr_bail`).
    //
    // It is NOT unreachable, and an earlier note here saying it had zero
    // JitCode hits was wrong.  Counting `body.code[pos]` over every
    // `pos in body.startpoints` of the shipped `jit_metadata.json`
    // (3459 bodies, 152888 instructions, 114 distinct opnames -- the
    // whole `insns` table, so no opname is dead) puts one occurrence in
    // `view_as_kwargs`, between a `live/` and an `int_guard_value` +
    // `residual_call_r_r`.  A `f(**d)` probe did not reach it, so
    // whether a walk gets there is unproven; the byte is chosen here
    // because nothing else can reach this arm, not because it is dead.
    let opname = "vtable_method_ptr/rd>i";
    let unsupported_byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    // Operand encoding `rd>i`: 1B r-reg + 2B descr + 1B i-reg-dst = 4B.
    let code = [unsupported_byte, 0, 0, 0, 0];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("unsupported opname must hit UnsupportedOpname");
    assert_eq!(err, DispatchError::UnsupportedOpname { pc: 0, key: opname },);
}

#[test]
fn new_array_id_records_new_array() {
    let opname = "new_array/id>r";
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    // `id>r`: 1B length reg, 2B descr, 1B dest ref reg.
    let code = [byte, 0, 0, 0, 0];
    let mut tc = fresh_trace_ctx();
    let length = tc.const_int(4);
    let descr = done_descr_ref_for_tests();
    let descr_pool = vec![descr];
    let dummy = dummy_int_add_box(&mut tc);
    let regs_r = [dummy];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: vec![ConcreteValue::Null],
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new([length]),
        registers_f: &RegisterBank::default(),
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        pending_guard_snapshot_error: None,
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_reorder_ceiling: u32::MAX,
        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    step(&code, 0, &mut wc).expect("new_array/id>r records");
    let recorded = tc
        .ops()
        .iter()
        .any(|op| op.opcode == majit_ir::OpCode::NewArray);
    assert!(recorded, "new_array/id>r must record NEW_ARRAY");
}

#[test]
fn canonical_cond_record_keys_have_walker_arms() {
    // Assembler emit now publishes these four keys. The production
    // walker must decode them; UnsupportedOpname here would abort a
    // `conditional_call_elidable!` / `record_known_result!` trace.
    let mut tc = fresh_trace_ctx();
    let dummy = dummy_int_add_box(&mut tc);
    for key in [
        "conditional_call_value_ir_i/iiIRd>i",
        "conditional_call_value_ir_r/riIRd>r",
        "record_known_result_i_ir_v/iiIRd",
        "record_known_result_r_ir_v/riIRd",
    ] {
        let byte = *insns_opname_to_byte()
            .get(key)
            .unwrap_or_else(|| panic!("`{key}` must be in insns table"));
        let code = [byte, 0, 0, 0, 0, 0, 0, 0];
        let mut regs_r = [dummy];
        let mut concrete_r = [ConcreteValue::Null];
        let mut regs_i = [dummy, dummy];
        let err = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut regs_i)
            .expect_err("{key} with no call descr cannot complete");
        assert!(
            !matches!(err, DispatchError::UnsupportedOpname { .. }),
            "{key} must have a walker arm, got {err:?}"
        );
    }
}

#[test]
fn strlen_and_strgetitem_keys_have_walker_arms() {
    // The startswith/endswith wrapper walk emits these three keys.
    // UnsupportedOpname here aborts look-inside and leaves the residual
    // call as a black box (401-guard census).
    let mut tc = fresh_trace_ctx();
    let dummy = dummy_int_add_box(&mut tc);
    for key in ["strlen/r>i", "strgetitem/ri>i", "strgetitem/rc>i"] {
        let byte = *insns_opname_to_byte()
            .get(key)
            .unwrap_or_else(|| panic!("`{key}` must be in insns table"));
        let code = [byte, 0, 0, 0, 0, 0, 0, 0];
        let mut regs_r = [dummy];
        let mut concrete_r = [ConcreteValue::Null];
        let mut regs_i = [dummy, dummy];
        let err = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut regs_i);
        assert!(
            err.as_ref()
                .err()
                .is_none_or(|e| !matches!(e, DispatchError::UnsupportedOpname { .. })),
            "{key} must have a walker arm, got {err:?}"
        );
    }
}

/// Dispatch `key` through [`handle`] with `operands` as its operand bytes.
/// A key no built JitCode emits has no byte in the insns table to step.
fn run_keyed_step(
    key: &'static str,
    operands: &[u8],
    tc: &mut TraceCtx,
    regs_r: &mut [OpRef],
    regs_i: &mut [OpRef],
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let (opname, argcodes) = key.split_once('/').expect("opname/argcodes key");
    let code: Vec<u8> = std::iter::once(0).chain(operands.iter().copied()).collect();
    let op = DecodedOp {
        key,
        opname,
        argcodes,
        pc: 0,
        next_pc: code.len(),
    };
    let mut concrete_r = vec![ConcreteValue::Null; regs_r.len()];
    with_hint_walk_context(
        tc,
        regs_r,
        &mut concrete_r,
        regs_i,
        &mut [],
        &mut [],
        &[],
        |wc| handle(&op, &code, wc),
    )
}

#[test]
fn newstr_strsetitem_copystrcontent_record_their_operands() {
    // `pyjitpl.py opimpl_newstr` / `opimpl_strsetitem` /
    // `opimpl_copystrcontent`: each form records its op over the decoded
    // operands, `c` bytes as ConstInt.  `execute` runs the store first, so
    // the string registers hold live STR buffers long enough for every
    // index below and the int registers hold concrete values.
    use pyre_object::lowlevel_string::{
        LOWLEVEL_STR_BASE_SIZE, bh_alloc_lowlevel_string, bh_free_lowlevel_string,
    };
    let mut tc = fresh_trace_ctx();
    let bufs = [
        bh_alloc_lowlevel_string(16, LOWLEVEL_STR_BASE_SIZE, 1),
        bh_alloc_lowlevel_string(16, LOWLEVEL_STR_BASE_SIZE, 1),
    ];
    let r0 = dummy_int_add_box(&mut tc);
    let r1 = dummy_int_add_box(&mut tc);
    let r = [r0, r1];
    for (reg, buf) in r.iter().zip(bufs) {
        tc.set_opref_concrete(*reg, majit_ir::Value::Ref(majit_ir::GcRef(buf as usize)));
    }
    let i0 = dummy_int_add_box(&mut tc);
    let i1 = dummy_int_add_box(&mut tc);
    let i2 = dummy_int_add_box(&mut tc);
    let i = [i0, i1, i2];
    for (n, reg) in i.iter().enumerate() {
        tc.set_opref_concrete(*reg, majit_ir::Value::Int(n as i64));
    }
    let c = OpRef::ConstInt;
    use majit_ir::OpCode::{Copystrcontent, Newstr, Strsetitem};
    let cases: Vec<(&'static str, Vec<u8>, majit_ir::OpCode, Vec<OpRef>)> = vec![
        ("newstr/i>r", vec![0, 0], Newstr, vec![i[0]]),
        ("newstr/c>r", vec![5, 0], Newstr, vec![c(5)]),
        (
            "strsetitem/rii",
            vec![0, 0, 1],
            Strsetitem,
            vec![r[0], i[0], i[1]],
        ),
        (
            "strsetitem/rci",
            vec![0, 3, 1],
            Strsetitem,
            vec![r[0], c(3), i[1]],
        ),
        (
            "strsetitem/ric",
            vec![0, 0, 97],
            Strsetitem,
            vec![r[0], i[0], c(97)],
        ),
        (
            "strsetitem/rcc",
            vec![0, 3, 97],
            Strsetitem,
            vec![r[0], c(3), c(97)],
        ),
        (
            "copystrcontent/rriii",
            vec![0, 1, 0, 1, 2],
            Copystrcontent,
            vec![r[0], r[1], i[0], i[1], i[2]],
        ),
        (
            "copystrcontent/rrcii",
            vec![0, 1, 4, 1, 2],
            Copystrcontent,
            vec![r[0], r[1], c(4), i[1], i[2]],
        ),
        (
            "copystrcontent/rrici",
            vec![0, 1, 0, 4, 2],
            Copystrcontent,
            vec![r[0], r[1], i[0], c(4), i[2]],
        ),
        (
            "copystrcontent/rriic",
            vec![0, 1, 0, 1, 4],
            Copystrcontent,
            vec![r[0], r[1], i[0], i[1], c(4)],
        ),
        (
            "copystrcontent/rrcci",
            vec![0, 1, 4, 5, 2],
            Copystrcontent,
            vec![r[0], r[1], c(4), c(5), i[2]],
        ),
        (
            "copystrcontent/rrcic",
            vec![0, 1, 4, 1, 6],
            Copystrcontent,
            vec![r[0], r[1], c(4), i[1], c(6)],
        ),
        (
            "copystrcontent/rricc",
            vec![0, 1, 0, 5, 6],
            Copystrcontent,
            vec![r[0], r[1], i[0], c(5), c(6)],
        ),
        (
            "copystrcontent/rrccc",
            vec![0, 1, 4, 5, 6],
            Copystrcontent,
            vec![r[0], r[1], c(4), c(5), c(6)],
        ),
    ];
    for (key, operands, opcode, args) in cases {
        let mut regs_r = r;
        let mut regs_i = i;
        let (outcome, next_pc) = run_keyed_step(key, &operands, &mut tc, &mut regs_r, &mut regs_i)
            .unwrap_or_else(|err| panic!("{key} must dispatch, got {err:?}"));
        assert_eq!(outcome, DispatchOutcome::Continue, "{key}");
        assert_eq!(next_pc, operands.len() + 1, "{key}");
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, opcode, "{key}");
        let recorded: Vec<OpRef> = last.getarglist().iter().map(|a| a.to_opref()).collect();
        assert_eq!(recorded, args, "{key}");
        if opcode == Newstr {
            assert_ne!(regs_r[0], r[0], "{key} writes its dst register");
        }
    }
    for buf in bufs {
        bh_free_lowlevel_string(buf, LOWLEVEL_STR_BASE_SIZE, 1);
    }
}

#[test]
fn strsetitem_and_copystrcontent_decline_a_store_they_cannot_run() {
    // `execute` runs the store before the op is recorded.  A string with no
    // concrete value, or a range outside the buffer, declines without
    // recording anything.
    use pyre_object::lowlevel_string::{
        LOWLEVEL_STR_BASE_SIZE, bh_alloc_lowlevel_string, bh_free_lowlevel_string,
    };
    let mut tc = fresh_trace_ctx();
    let buf = bh_alloc_lowlevel_string(4, LOWLEVEL_STR_BASE_SIZE, 1);
    let r0 = dummy_int_add_box(&mut tc);
    let r1 = dummy_int_add_box(&mut tc);
    let r = [r0, r1];
    tc.set_opref_concrete(r[0], majit_ir::Value::Ref(majit_ir::GcRef(buf as usize)));
    let i = [dummy_int_add_box(&mut tc)];
    // The copy length: srcstart 2 + 3 overruns the 4-char buffer.
    tc.set_opref_concrete(i[0], majit_ir::Value::Int(3));
    let cases: [(&'static str, Vec<u8>, &'static str); 4] = [
        (
            "strsetitem/rci",
            vec![1, 0, 0],
            "strsetitem (string not concrete)",
        ),
        (
            "strsetitem/rcc",
            vec![0, 4, 97],
            "strsetitem (out of range)",
        ),
        (
            "copystrcontent/rrcci",
            vec![0, 1, 0, 0, 0],
            "copystrcontent (string not concrete)",
        ),
        (
            "copystrcontent/rrcci",
            vec![0, 0, 2, 0, 0],
            "copystrcontent (out of range)",
        ),
    ];
    for (key, operands, expected) in cases {
        let before = tc.ops().len();
        let mut regs_r = r;
        let mut regs_i = i;
        let err = run_keyed_step(key, &operands, &mut tc, &mut regs_r, &mut regs_i).expect_err(key);
        match err {
            DispatchError::UnsupportedOpname { key: got, .. } => assert_eq!(got, expected, "{key}"),
            other => panic!("{key}: expected UnsupportedOpname, got {other:?}"),
        }
        assert_eq!(tc.ops().len(), before, "{key} records nothing");
    }
    bh_free_lowlevel_string(buf, LOWLEVEL_STR_BASE_SIZE, 1);
}

#[test]
fn empty_str_concat_helper_aborts_before_the_unwired_op_is_dispatched() {
    use majit_translate::codewriter::{call::CallControl, codewriter::CodeWriter};
    use majit_translate::{
        CallPath, ConcreteType, FunctionGraph, GraphTransformConfig, OpKind, ValueType,
    };

    let mut graph = FunctionGraph::new("empty_str_concat_helper");
    let lhs = graph
        .push_op_var(
            graph.startblock,
            OpKind::Input {
                name: "lhs".into(),
                ty: ValueType::Ref(None),
                class_root: None,
            },
            true,
        )
        .unwrap();
    let rhs = graph
        .push_op_var(
            graph.startblock,
            OpKind::Input {
                name: "rhs".into(),
                ty: ValueType::Ref(None),
                class_root: None,
            },
            true,
        )
        .unwrap();
    let result = graph
        .push_op_var(
            graph.startblock,
            OpKind::BinOp {
                op: "add".into(),
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();
    graph.set_return(graph.startblock, Some(result.clone()));
    FunctionGraph::set_concretetype_of_inline(&lhs, ConcreteType::GcRef);
    FunctionGraph::set_concretetype_of_inline(&rhs, ConcreteType::GcRef);
    FunctionGraph::set_concretetype_of_inline(&result, ConcreteType::GcRef);

    let config = GraphTransformConfig {
        str_concat_helper: String::new(),
        ..Default::default()
    };
    let path = CallPath::from_segments(["empty_str_concat_helper"]);
    let mut callcontrol = CallControl::new();
    callcontrol.register_function_graph(path.clone(), graph);
    let jitcode = callcontrol.get_jitcode(&path);
    let mut codewriter = CodeWriter::new();
    codewriter.drain_pending_graphs(&mut callcontrol, &config);
    let body = jitcode
        .try_body()
        .expect("the public codewriter pipeline commits the concat graph");

    let original_opname = "int_add/rr>r";
    let original_byte = *codewriter
        .assembler
        .insns()
        .get(original_opname)
        .expect("the original Ref + Ref add remains after the refusal marker");
    let original_pc = body
        .code
        .iter()
        .enumerate()
        .find_map(|(pc, candidate)| {
            (*candidate == original_byte && jitcode.is_valid_startpoint(pc)).then_some(pc)
        })
        .expect("assembled body contains the Ref + Ref add byte");
    let abort_byte = *codewriter
        .assembler
        .insns()
        .get("abort/")
        .expect("the empty helper emits the explicit abort opname");
    let abort_pc = body
        .code
        .iter()
        .enumerate()
        .find_map(|(pc, candidate)| {
            (*candidate == abort_byte && jitcode.is_valid_startpoint(pc)).then_some(pc)
        })
        .expect("assembled body contains the explicit abort byte");
    assert!(
        abort_pc < original_pc,
        "the refusal must be reached before the handler-less original op",
    );
    let decoded = crate::jitcode_runtime::DecodedOp {
        key: "abort/",
        opname: "abort",
        argcodes: "",
        pc: abort_pc,
        next_pc: abort_pc + 1,
    };

    let mut trace_ctx = fresh_trace_ctx();
    let session = std::cell::RefCell::new(WalkSession::default());
    let registers_r = vec![OpRef::NONE; body.c_num_regs_r as usize];
    let concrete_registers_r = vec![ConcreteValue::Null; body.c_num_regs_r as usize];
    let mut walk_ctx = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: (concrete_registers_r).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(registers_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut trace_ctx,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let trace_error = super::handle(&decoded, &body.code, &mut walk_ctx)
        .expect_err("the walker stops at the explicit refusal marker");
    assert_eq!(
        trace_error,
        DispatchError::AbortMarkerReached { pc: abort_pc },
    );

    let mut builder = majit_metainterp::blackhole::BlackholeInterpBuilder::new();
    builder.setup_insns(codewriter.assembler.insns());
    majit_metainterp::blackhole::wire_bhimpl_handlers(&mut builder);
    assert_eq!(builder.unwired_opnames(), vec![original_opname]);
    let mut bh = builder.acquire_interp();
    assert!(
        matches!(
            builder.dispatch_loop(&mut bh, &body.code, abort_pc),
            Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
        ),
        "the blackhole exits at the refusal marker before the unwired add",
    );
    assert!(bh.aborted);
}

/// `ptr_nonzero/r>i` records `PtrNe(box, CONST_NULL)` into the
/// int dst.  RPython parity: `pyjitpl.py opimpl_ptr_nonzero`
/// returns `self.execute(rop.PTR_NE, box, CONST_NULL)`.
#[test]
fn ptr_nonzero_records_ptrne_with_box_and_null() {
    let opname = "ptr_nonzero/r>i";
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    // Operand encoding `r>i`: 1B r-reg + 1B i-reg-dst = 2B
    let code = [byte, 0, 0];
    // A `Const` box folds (`_all_constants` — CONST_NULL is a `Const`), so
    // the record path needs a genuinely non-const InputArg operand.
    let mut tc = TraceCtx::for_test_types(&[Type::Ref]);
    let descr = done_descr_ref_for_tests();
    let box_opref = OpRef::input_arg_ref(0);
    let regs_r = [box_opref];
    let regs_i = [OpRef::None];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("ptr_nonzero must record PtrNe");
    assert!(matches!(outcome, DispatchOutcome::Continue));
    assert_eq!(next_pc, 3);
    // `get_or_insert_typed` mints a fresh OpRef on every call (see
    // `constant_pool.rs` — equality is `Const.same_constant`, not
    // OpRef identity), so we cannot compare against a freshly-minted
    // null_const.  Verify args[1] is a Ref-typed constant whose
    // pooled value is 0 instead.
    let last_args0;
    let last_args1;
    let last_opcode;
    let last_args_len;
    {
        let ops = wc.trace_ctx.ops();
        let last = ops.last().expect("ptr_nonzero must record one op");
        last_opcode = last.opcode;
        let args = last.getarglist();
        last_args_len = args.len();
        last_args0 = args[0].clone();
        last_args1 = args[1].clone();
    }
    assert_eq!(last_opcode, majit_ir::OpCode::PtrNe);
    assert_eq!(last_args_len, 2);
    assert_eq!(last_args0.to_opref(), box_opref);
    assert_eq!(
        wc.trace_ctx.const_value(last_args1.to_opref()),
        Some(0),
        "args[1] must point at the CONST_NULL pool entry (value=0)"
    );
    assert_eq!(
        wc.trace_ctx.const_type(last_args1.to_opref()),
        Some(Type::Ref)
    );
    assert_ne!(
        wc.registers_i.get(0).expect("int register in range"),
        OpRef::None
    );
}

#[test]
fn ptr_nullity_folds_a_const_box_without_recording() {
    // `execute(PTR_NE, box, CONST_NULL)` has all-`Const` operands when
    // `box` is a `Const` — `_all_constants` folds the nullity answer
    // without recording.
    let byte = *insns_opname_to_byte()
        .get("ptr_nonzero/r>i")
        .expect("`ptr_nonzero/r>i` must be in insns table");
    let code = [byte, 0, 0];

    let mut tc = fresh_trace_ctx();
    let nonnull = tc.const_ref(0xdeadbeef);
    let mut regs_r = [nonnull];
    let mut regs_i = [OpRef::None];
    let (_, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut [], &mut regs_i)
        .expect("ptr_nonzero on a const ref must fold");
    assert_eq!(next_pc, 3);
    assert_eq!(tc.num_ops(), 0, "a const box folds without recording");
    assert_eq!(
        regs_i[0].inline_const_to_value(),
        Some(majit_ir::Value::Int(1)),
        "a non-null const ref folds ptr_nonzero to 1",
    );

    let mut null_tc = fresh_trace_ctx();
    let null = null_tc.const_null();
    let mut null_regs_r = [null];
    let mut null_regs_i = [OpRef::None];
    let (_, _) = run_hint_step(
        &code,
        &mut null_tc,
        &mut null_regs_r,
        &mut [],
        &mut null_regs_i,
    )
    .expect("ptr_nonzero on the null const must fold");
    assert_eq!(null_tc.num_ops(), 0);
    assert_eq!(
        null_regs_i[0].inline_const_to_value(),
        Some(majit_ir::Value::Int(0)),
        "the null const folds ptr_nonzero to 0",
    );
}

#[test]
fn cast_int_to_float_folds_a_const_int_without_recording() {
    let byte = *insns_opname_to_byte()
        .get("cast_int_to_float/i>f")
        .expect("`cast_int_to_float/i>f` must be in insns table");
    let code = [byte, 0x00, 0x00]; // `i>f`: i-src=0, f-dst=0
    let mut tc = fresh_trace_ctx();
    let operand = tc.const_int(42);
    let mut regs_i = [operand];
    let mut regs_f = [OpRef::None];
    let (_, next_pc) = run_float_step(&code, &mut tc, &mut regs_f, &mut regs_i)
        .expect("cast_int_to_float on a const int must fold");
    assert_eq!(next_pc, 3);
    assert_eq!(tc.num_ops(), 0, "a const operand folds without recording");
    assert_eq!(
        regs_f[0].inline_const_to_value(),
        Some(majit_ir::Value::Float(42.0)),
        "dst must hold the folded ConstFloat",
    );
}

#[test]
fn cast_float_to_int_folds_a_const_float_without_recording() {
    let byte = *insns_opname_to_byte()
        .get("cast_float_to_int/f>i")
        .expect("`cast_float_to_int/f>i` must be in insns table");
    let code = [byte, 0x00, 0x00]; // `f>i`: f-src=0, i-dst=0
    let mut tc = fresh_trace_ctx();
    let operand = tc.const_float((42.75f64).to_bits() as i64);
    let mut regs_f = [operand];
    let mut regs_i = [OpRef::None];
    let (_, next_pc) = run_float_step(&code, &mut tc, &mut regs_f, &mut regs_i)
        .expect("cast_float_to_int on a const float must fold");
    assert_eq!(next_pc, 3);
    assert_eq!(tc.num_ops(), 0, "a const operand folds without recording");
    assert_eq!(
        regs_i[0].inline_const_to_value(),
        Some(majit_ir::Value::Int(42)),
        "dst must hold the folded ConstInt",
    );
}

/// A result-classified abort is still an abort.  The destination byte records
/// the unsupported operation's SSA shape; it must not turn that operation into
/// a no-op whose stale destination can be consumed later.
#[test]
fn abort_result_r_stops_the_walk() {
    let opname = "abort/>r";
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    let code = [byte, 0x05]; // dst byte = 5 (intentionally out-of-range)
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let error = step(&code, 0, &mut wc).expect_err("abort/>r must abort the walk");
    assert!(matches!(error, DispatchError::AbortMarkerReached { pc: 0 }));
    assert_eq!(
        wc.trace_ctx.num_ops(),
        ops_before,
        "abort/>r must abort before recording any IR op",
    );
}

/// `ref_guard_value/r` records `GuardValue(value, ConstPtr(concrete))`
/// when the symbolic OpRef is non-Const and a concrete pointer is
/// available in the shadow.  Mirrors `pyjitpl.py
/// implement_guard_value`.
#[test]
fn ref_guard_value_records_guardvalue_with_concrete_constant() {
    let opname = "ref_guard_value/r";
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    // Operand encoding `r`: 1B r-src only.
    let code = [byte, 0];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    // MIFrame.replace_active_box_in_frame selects the bank by the Box type.
    // Use a real Ref result; PtrEq would manufacture an IntOp in registers_r.
    let concrete_ptr: usize = 0xdead_beef;
    let source = tc.const_ref(concrete_ptr as i64);
    let value_opref = tc.record_op(majit_ir::OpCode::SameAsR, &[source]);
    let regs_r = [value_opref];
    let regs_i = [OpRef::None];
    let concrete_r = [ConcreteValue::Ref(
        concrete_ptr as *mut pyre_object::pyobject::PyObject,
    )];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: (concrete_r).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).expect("ref_guard_value must record GuardValue");
    assert!(matches!(outcome, DispatchOutcome::Continue));
    assert_eq!(next_pc, 2);
    let (last_opcode, last_args0, last_args1, last_args_len) = {
        let ops = wc.trace_ctx.ops();
        let last = ops.last().expect("ref_guard_value must record one op");
        let args = last.getarglist();
        (last.opcode, args[0].clone(), args[1].clone(), args.len())
    };
    assert_eq!(last_opcode, majit_ir::OpCode::GuardValue);
    assert_eq!(last_args_len, 2);
    assert_eq!(last_args0.to_opref(), value_opref);
    assert_eq!(
        wc.trace_ctx.const_value(last_args1.to_opref()),
        Some(concrete_ptr as i64),
        "args[1] must point at the concrete pointer in the pool",
    );
    assert_eq!(
        wc.trace_ctx.const_type(last_args1.to_opref()),
        Some(Type::Ref)
    );
    assert_eq!(
        wc.registers_r.get(0).expect("ref register in range"),
        last_args1.to_opref(),
        "register slot still holding the original OpRef must be rewritten \
             to the promoted constant (`pyjitpl.py` `replace_box`)",
    );
}

/// `int_guard_value/i` records `GuardValue(value, ConstInt(concrete))`
/// through the same bank-parameterized body as the Ref and Float variants.
#[test]
fn int_guard_value_records_guardvalue_with_concrete_constant() {
    let opname = "int_guard_value/i";
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    // Operand encoding `i`: 1B i-src only.
    let code = [byte, 0];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    // Symbolic side: a recorded op OpRef (not a Const).
    let value_opref = dummy_int_add_box(&mut tc);
    let ops_before = tc.num_ops();
    let regs_r = [OpRef::None];
    let regs_i = [value_opref];
    let mut concrete_i = [ConcreteValue::Int(42)];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut concrete_i,
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).expect("int_guard_value must record GuardValue");
    assert!(matches!(outcome, DispatchOutcome::Continue));
    assert_eq!(next_pc, 2);
    assert_eq!(
        wc.trace_ctx.num_ops(),
        ops_before + 1,
        "int_guard_value must record exactly one GuardValue op",
    );
    let (last_opcode, last_args0, last_args1, last_args_len) = {
        let ops = wc.trace_ctx.ops();
        let last = ops.last().expect("int_guard_value must record one op");
        let args = last.getarglist();
        (last.opcode, args[0].clone(), args[1].clone(), args.len())
    };
    assert_eq!(last_opcode, majit_ir::OpCode::GuardValue);
    assert_eq!(last_args_len, 2);
    assert_eq!(last_args0.to_opref(), value_opref);
    assert_eq!(wc.trace_ctx.const_value(last_args1.to_opref()), Some(42));
    assert_eq!(
        wc.trace_ctx.const_type(last_args1.to_opref()),
        Some(Type::Int)
    );
    assert_eq!(
        wc.registers_i.get(0).expect("int register in range"),
        last_args1.to_opref(),
        "register slot still holding the original OpRef must be rewritten \
         to the promoted constant (`pyjitpl.py` `replace_box`)",
    );
}

/// Symbolic OpRef already a Const → `ref_guard_value/r` is a no-op
/// (`pyjitpl.py if isinstance(box, Const): return box`).
#[test]
fn ref_guard_value_on_const_records_nothing() {
    let opname = "ref_guard_value/r";
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    let code = [byte, 0];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let value_opref = tc.const_ref(0xdead_beef);
    let baseline_ops = tc.ops().len();
    let regs_r = [value_opref];
    let regs_i = [OpRef::None];
    let concrete_r = [ConcreteValue::Ref(
        0xdead_beef as *mut pyre_object::pyobject::PyObject,
    )];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: (concrete_r).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("ref_guard_value Const arm");
    assert!(matches!(outcome, DispatchOutcome::Continue));
    assert_eq!(next_pc, 2);
    assert_eq!(
        wc.trace_ctx.ops().len(),
        baseline_ops,
        "no op should be recorded when input is already Const"
    );
    assert_eq!(
        wc.registers_r.get(0).expect("ref register in range"),
        value_opref
    );
}

#[test]
fn step_through_residual_call_r_r_records_callr_with_descr_and_args() {
    // `residual_call_r_r/iRd>r` records `OpCode::CallR`
    // with `[funcptr, ...args]` and `descr=descr_refs[d]`. RPython
    // `pyjitpl.py _opimpl_residual_call1` →
    // `do_residual_or_indirect_call → execute_and_record_varargs(
    // rop.CALL_R, [funcbox]+argboxes, descr=calldescr)`.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    // Operand encoding `iRd>r`: 1B funcptr (i-reg=2),
    // 1B varlen=2 + [r-reg=4, r-reg=7], 2B descr_index=1 (LE),
    // 1B dst-reg=0 (writeback deferred — not used by walker yet).
    let code = [
        residual_byte,
        0x02, // funcptr from registers_i[2]
        0x02, // varlen
        0x04,
        0x07, // args from registers_r[4, 7]
        0x01,
        0x00, // descr index = 1 (LE)
        0x00, // dst reg (deferred)
    ];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 4);
    let regs_r = distinct_const_refs(&mut tc, 8);
    let funcptr_expected = regs_i[2];
    let arg0_expected = regs_r[4];
    let arg1_expected = regs_r[7];
    // Build a 2-entry descr table — index 0 is a decoy (different
    // pointer), index 1 is the descr we expect the recorder to attach.
    // RPython `_build_allboxes` reads `descr.get_arg_types()` to
    // permute argboxes into ABI order; the test passes 2 R args so
    // `arg_types = [Ref, Ref]` keeps the permutation an identity
    // (allboxes = [funcbox, r0, r1]).
    let decoy = make_call_descr(
        2,
        vec![Type::Ref, Type::Ref],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    );
    let call_descr = make_call_descr(
        3,
        vec![Type::Ref, Type::Ref],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    );
    let descr_pool = vec![decoy, call_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc,
        code.len(),
        "residual_call_r_r must advance past funcptr + varlist + descr + dst",
    );
    drop(wc);
    // FailDescr placeholder has no EffectInfo (`as_call_descr() = None`),
    // so the walker takes the `no-effectinfo-fallback` branch:
    // CallR + GuardNoException (RPython parity:
    // `do_residual_call → execute_varargs(..., exc=True)` →
    // `handle_possible_exception` emits GUARD_NO_EXCEPTION).
    assert_eq!(
        tc.num_ops(),
        ops_before + 2,
        "residual_call_r_r must record CallR + GuardNoException (no-effectinfo fallback)",
    );
    let call_op = tc
        .ops()
        .iter()
        .find(|o| o.opcode == majit_ir::OpCode::CallR)
        .expect("CallR must be recorded");
    assert_eq!(
        call_op
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![funcptr_expected, arg0_expected, arg1_expected],
        "CallR args must be [funcptr, ...args] from registers_i+registers_r",
    );
    let recorded_descr = call_op.getdescr().expect("CallR must carry the calldescr");
    assert!(
        std::sync::Arc::ptr_eq(&recorded_descr, &call_descr),
        "CallR descr must be descr_refs[1] (not decoy at index 0)",
    );
    // GuardNoException follows immediately after.
    let guard_op = tc
        .ops()
        .iter()
        .find(|o| o.opcode == majit_ir::OpCode::GuardNoException)
        .expect("GuardNoException must follow CallR for raising calls");
    assert!(
        guard_op.num_args() == 0,
        "GuardNoException takes no operand args",
    );
    // `walker_capture_snapshot_for_last_guard` ports
    // `capture_resumedata(after_residual_call=True)`
    // (`pyjitpl.py`).  Every guard emitted by a
    // residual_call dispatcher now carries a snapshot whose
    // `rd_resume_position` is the freshly-allocated snapshot id
    // (`>= 0`), so the optimizer's `store_final_boxes_in_guard`
    // (`optimizeopt/mod.rs`) finds attached resume data
    // instead of panicking on the `-1` sentinel.
    assert!(
        guard_op.rd_resume_position() >= 0,
        "GuardNoException must carry an attached snapshot (rd_resume_position >= 0) after capture_resumedata port",
    );
}

/// Build a `SimpleCallDescr` for tests, parameterised by `arg_types`,
/// `result_type`, and `extraeffect`. The `_build_allboxes` permutation
/// reads `arg_types` as the callee's ABI ordering, so tests must pass
/// the exact types of the arglist they exercise.
fn make_call_descr(
    idx: u32,
    arg_types: Vec<Type>,
    result_type: Type,
    extra: majit_ir::ExtraEffect,
) -> DescrRef {
    let mut effect = majit_ir::EffectInfo::default();
    effect.extraeffect = extra;
    std::sync::Arc::new(majit_ir::SimpleCallDescr::new(
        idx,
        arg_types,
        result_type,
        false,
        std::mem::size_of::<usize>(),
        effect,
    ))
}

#[derive(Clone, Copy)]
enum SymbolicBoxStrArg {
    InternedStr,
    /// Prebuilt STR constants materialize as `_utf8` storage, not the wrapper.
    StoragePayload,
    NonStr,
    NonConstant,
}

fn run_symbolic_box_str_dispatch(
    arg_kind: SymbolicBoxStrArg,
) -> (
    Result<(DispatchOutcome, usize), DispatchError>,
    Vec<OpCode>,
    Option<majit_ir::Value>,
    bool,
    Option<usize>,
) {
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    // funcptr=i[0], one Ref arg r[0], descr=0, dst=r[1].
    let code = [residual_byte, 0, 1, 0, 0, 0, 1];
    let mut tc = fresh_trace_ctx();
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments([
        "pyre_object",
        "unicodeobject",
        "box_str_constant",
    ]);
    let regs_i = vec![tc.const_int(symbolic)];
    let (arg, expected_ptr) = match arg_kind {
        SymbolicBoxStrArg::InternedStr => {
            let obj = pyre_object::unicodeobject::box_str_constant(rustpython_wtf8::Wtf8::new(
                "__instancecheck__",
            ));
            (tc.const_ref(obj as i64), Some(obj as usize))
        }
        SymbolicBoxStrArg::StoragePayload => {
            let obj = pyre_object::unicodeobject::box_str_constant(rustpython_wtf8::Wtf8::new(
                "__index__",
            ));
            let storage = unsafe { pyre_object::unicodeobject::w_str_storage(obj) };
            (tc.const_ref(storage as i64), Some(obj as usize))
        }
        SymbolicBoxStrArg::NonStr => {
            let obj = pyre_object::intobject::w_int_new(42);
            (tc.const_ref(obj as i64), None)
        }
        SymbolicBoxStrArg::NonConstant => (OpRef::input_arg_ref(0), None),
    };
    let regs_r = vec![arg, OpRef::NONE];
    let descr_pool = vec![make_call_descr(
        31,
        vec![Type::Ref],
        Type::Ref,
        majit_ir::ExtraEffect::CannotRaise,
    )];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: FbwWalkMode {
            inline_subwalk: true,
            ..test_fbw_mode()
        },
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        is_top_level: false,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let outcome = step(&code, 0, &mut wc);
    let regs_r = wc.registers_r.to_vec();
    drop(wc);
    let dst = regs_r[1];
    let dst_value = tc.box_value(dst);
    let dst_is_constant = dst.is_constant();
    let opcodes = tc.ops().iter().map(|op| op.opcode).collect();
    (outcome, opcodes, dst_value, dst_is_constant, expected_ptr)
}

#[test]
fn symbolic_box_str_constant_residual_folds_before_recording() {
    let (outcome, opcodes, dst_value, dst_is_constant, expected_ptr) =
        run_symbolic_box_str_dispatch(SymbolicBoxStrArg::InternedStr);

    assert_eq!(
        outcome.expect("registered symbolic call must fold"),
        (DispatchOutcome::Continue, 7),
    );
    assert!(
        opcodes.is_empty(),
        "the folded residual must record no call op"
    );
    assert!(dst_is_constant, "the folded result must be a Ref constant");
    assert_eq!(
        dst_value,
        expected_ptr.map(|ptr| majit_ir::Value::Ref(majit_ir::GcRef(ptr))),
        "box_str_constant must return the existing interned object",
    );
}

#[test]
fn symbolic_box_str_constant_residual_folds_prebuilt_storage() {
    let (outcome, opcodes, dst_value, dst_is_constant, expected_ptr) =
        run_symbolic_box_str_dispatch(SymbolicBoxStrArg::StoragePayload);

    assert_eq!(
        outcome.expect("registered symbolic call must fold"),
        (DispatchOutcome::Continue, 7),
    );
    assert!(
        opcodes.is_empty(),
        "the folded residual must record no call op"
    );
    assert!(dst_is_constant, "the folded result must be a Ref constant");
    assert_eq!(
        dst_value,
        expected_ptr.map(|ptr| majit_ir::Value::Ref(majit_ir::GcRef(ptr))),
        "box_str_constant must return the existing interned object",
    );
}

#[test]
fn symbolic_box_str_constant_residual_declines_unsafe_arguments() {
    let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments([
        "pyre_object",
        "unicodeobject",
        "box_str_constant",
    ]);
    for arg_kind in [SymbolicBoxStrArg::NonStr, SymbolicBoxStrArg::NonConstant] {
        let (outcome, opcodes, _, _, _) = run_symbolic_box_str_dispatch(arg_kind);
        assert!(
            matches!(
                outcome,
                Err(DispatchError::OrthodoxSubWalkTraceUnsupported {
                    pc: 0,
                    symbolic: observed,
                }) if observed == symbolic
            ),
            "a rejected registry argument must reach the existing symbolic gate",
        );
        assert_eq!(
            opcodes,
            vec![OpCode::CallR],
            "a rejected registry argument must retain the recorded-symbolic path",
        );
    }
}

/// Convenience: legacy signature used by elidable-classification
/// tests with empty arglists (0 R args, descr arg_types=[]).
/// `result_type` defaults to `Ref` matching `_r_r` shape. Callers
/// exercising actual args must use [`make_call_descr`] directly to
/// pass matching `arg_types`.
fn call_descr_with_effect(idx: u32, extra: majit_ir::ExtraEffect) -> DescrRef {
    make_call_descr(idx, vec![], Type::Ref, extra)
}

/// Convenience: builds a `_r_r`-shape CallDescr with both
/// `extraeffect` and `oopspecindex` populated, for tests that need
/// to drive [`do_not_in_trace_call_result`] /
/// [`do_jit_force_virtual_guard`] / future oopspec-keyed guards.
fn call_descr_with_oopspec(
    idx: u32,
    extra: majit_ir::ExtraEffect,
    oopspec: majit_ir::OopSpecIndex,
) -> DescrRef {
    let mut effect = majit_ir::EffectInfo::default();
    effect.extraeffect = extra;
    effect.oopspecindex = oopspec;
    std::sync::Arc::new(majit_ir::SimpleCallDescr::new(
        idx,
        vec![],
        Type::Ref,
        false,
        std::mem::size_of::<usize>(),
        effect,
    ))
}

#[test]
fn residual_call_r_r_with_elidable_cannot_raise_records_callpurer_no_guard() {
    // RPython parity: `do_residual_call` (pyjitpl.py) reads
    // `effectinfo.check_is_elidable()` + `effectinfo.check_can_raise()`,
    // then `execute_varargs(rop.CALL_R, ..., exc, pure)`. With
    // EF_ELIDABLE_CANNOT_RAISE: `pure=True` (CALL_PURE_R) + `exc=False`
    // (no GUARD_NO_EXCEPTION).
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 1);
    let regs_r = distinct_const_refs(&mut tc, 4);
    let elidable_descr = call_descr_with_effect(7, majit_ir::ExtraEffect::ElidableCannotRaise);
    let descr_pool = vec![elidable_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "elidable+cannot-raise must record exactly CallPureR (no GuardNoException)",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(
        last.opcode,
        majit_ir::OpCode::CallPureR,
        "EF_ELIDABLE_CANNOT_RAISE must rewrite to CALL_PURE_R",
    );
}

// When the walk is the authoritative concrete leg,
// `try_execute_residual_call_via_executor` runs a `CallMayForce*`
// residual call concretely (RPython runs every residual_call during
// tracing) and stamps the recorded result OpRef so a downstream
// `goto_if_not` reads a concrete value.  Gated OFF by
// `is_authoritative_executor` so arm/shadow/probe walks never
// re-execute (which would double side effects / corrupt the live
// heap `cut_trace` cannot roll back).
extern "C" fn add2_for_walker_test(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

#[test]
fn elidable_or_memoryerror_call_executes_and_stamps_its_result() {
    // `MIFrame.execute_varargs` executes the CALL first and only then patches
    // it to CALL_PURE / handles the possible exception (`pyjitpl.py`).  The
    // `EF_ELIDABLE_OR_MEMORYERROR` bigint helpers use this exact shape: they
    // must not be left recorded-only merely because their opcode is CallPureI.
    let mut tc = fresh_trace_ctx();
    let funcbox = tc.const_int(add2_for_walker_test as *const () as i64);
    let arg0 = tc.const_int(40);
    let arg1 = tc.const_int(2);
    let allboxes = [funcbox, arg0, arg1];
    let descr = make_call_descr(
        81,
        vec![Type::Int, Type::Int],
        Type::Int,
        majit_ir::ExtraEffect::ElidableOrMemoryError,
    );
    let recorded = tc.record_op_with_descr(majit_ir::OpCode::CallPureI, &allboxes, descr.clone());
    let regs_i: Vec<OpRef> = Vec::new();
    let regs_r: Vec<OpRef> = Vec::new();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let result = try_execute_residual_call_via_executor(
        &mut wc,
        majit_ir::OpCode::CallPureI,
        &allboxes,
        descr.as_call_descr().expect("CallPureI descr"),
        recorded,
        0,
        None,
        false,
    );
    drop(wc);

    assert_eq!(result, Ok(ResidualExecOutcome::Executed(Ok(42))));
    assert_eq!(tc.box_value(recorded), Some(majit_ir::Value::Int(42)));
}

fn may_force_call_i_fixture(tc: &mut TraceCtx) -> ([OpRef; 3], DescrRef, OpRef) {
    let funcbox = tc.const_int(add2_for_walker_test as *const () as i64);
    let arg0 = tc.const_int(40);
    let arg1 = tc.const_int(2);
    let allboxes = [funcbox, arg0, arg1];
    let descr = make_call_descr(
        5,
        vec![Type::Int, Type::Int],
        Type::Int,
        majit_ir::ExtraEffect::CanRaise,
    );
    let recorded =
        tc.record_op_with_descr(majit_ir::OpCode::CallMayForceI, &allboxes, descr.clone());
    (allboxes, descr, recorded)
}

#[test]
fn authoritative_walker_executes_may_force_call_and_stamps_result() {
    let mut tc = fresh_trace_ctx();
    let (allboxes, descr, recorded) = may_force_call_i_fixture(&mut tc);
    let regs_i: Vec<OpRef> = Vec::new();
    let regs_r: Vec<OpRef> = Vec::new();
    let call_descr = descr.as_call_descr().expect("CallI descr");
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = try_execute_residual_call_via_executor(
        &mut wc,
        majit_ir::OpCode::CallMayForceI,
        &allboxes,
        call_descr,
        recorded,
        0,
        None,
        false,
    );
    drop(wc);
    assert_eq!(
        tc.box_value(recorded),
        Some(majit_ir::Value::Int(42)),
        "authoritative walker must execute add2_for_walker_test(40, 2) and stamp 42",
    );
}

#[test]
fn non_authoritative_walker_does_not_execute_may_force_call() {
    let mut tc = fresh_trace_ctx();
    let (allboxes, descr, recorded) = may_force_call_i_fixture(&mut tc);
    let regs_i: Vec<OpRef> = Vec::new();
    let regs_r: Vec<OpRef> = Vec::new();
    let call_descr = descr.as_call_descr().expect("CallI descr");
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = try_execute_residual_call_via_executor(
        &mut wc,
        majit_ir::OpCode::CallMayForceI,
        &allboxes,
        call_descr,
        recorded,
        0,
        None,
        false,
    );
    drop(wc);
    assert_eq!(
        tc.box_value(recorded),
        None,
        "non-authoritative walk (arm/shadow/probe) must NOT execute the may-force call",
    );
}

// A may-force call that RAISES (publishes on BH_LAST_EXC_VALUE) is transcribed
// onto WalkSession.last_exc_value (+ last_exc_value_concrete), while the
// blackhole carrier is consumed exactly once. The result box is NOT stamped
// (only the normal-return path stamps a result, mirroring `execute_varargs`'s
// success-only `result_box.value`).
extern "C" fn raises_for_walker_test() -> i64 {
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0xDEAD));
    0
}

#[test]
fn authoritative_walker_transcribes_may_force_raise_to_last_exc() {
    let mut tc = fresh_trace_ctx();
    let funcbox = tc.const_int(raises_for_walker_test as *const () as i64);
    let allboxes = [funcbox];
    let descr = make_call_descr(6, vec![], Type::Int, majit_ir::ExtraEffect::CanRaise);
    let recorded =
        tc.record_op_with_descr(majit_ir::OpCode::CallMayForceI, &allboxes, descr.clone());
    let regs_i: Vec<OpRef> = Vec::new();
    let regs_r: Vec<OpRef> = Vec::new();
    let call_descr = descr.as_call_descr().expect("CallI descr");
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = try_execute_residual_call_via_executor(
        &mut wc,
        majit_ir::OpCode::CallMayForceI,
        &allboxes,
        call_descr,
        recorded,
        0,
        None,
        false,
    );
    let captured_exc = wc.last_exc_value();
    let captured_concrete = wc.last_exc_value_concrete();
    drop(wc);
    assert!(
        captured_exc.is_some(),
        "a raising may-force call must transcribe the exception to last_exc_value",
    );
    assert!(
        matches!(captured_concrete, ConcreteValue::Ref(_)),
        "last_exc_value_concrete must carry the raised exception pointer",
    );
    assert_eq!(
        tc.box_value(recorded),
        None,
        "the raising path does not stamp `recorded`; the exception routes \
             via last_exc_value (only the normal-return path stamps a result)",
    );
    assert_eq!(
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get()),
        0,
        "the executor must consume the blackhole carrier exactly once",
    );
}

// pyjitpl.py / 3349-3353 vable token protocol around a
// concrete-executed may-force call: with an active standard
// virtualizable the executor sets TOKEN_TRACING_RESCALL before the
// call and probes-and-clears after it.  A token still intact means
// no force — execute + stamp as usual, token back to TOKEN_NONE.  A
// cleared token means the callee forced the virtualizable —
// `DispatchError::VableEscapedDuringResidualCall` (ABORT_ESCAPE,
// pyjitpl.py).
fn bind_fake_vable(tc: &mut TraceCtx, buf: &mut [u8]) {
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    assert!(
        info.token_offset + 8 <= buf.len(),
        "fake vable buffer must cover token_offset",
    );
    let vable_ref = tc.const_ref(buf.as_ptr() as i64);
    tc.install_virtualizable_info(info.clone());
    tc.init_virtualizable_boxes(
        &info,
        vable_ref,
        majit_ir::Value::Ref(majit_ir::GcRef(buf.as_ptr() as usize)),
        &[],
        &[],
        &[0],
    );
    tc.set_virtualizable_heap_ptr(buf.as_ptr());
}

#[test]
fn may_force_with_active_vable_executes_and_clears_token() {
    let mut tc = fresh_trace_ctx();
    let mut vable_buf = vec![0u8; 65536];
    bind_fake_vable(&mut tc, &mut vable_buf);
    let (allboxes, descr, recorded) = may_force_call_i_fixture(&mut tc);
    let regs_i: Vec<OpRef> = Vec::new();
    let regs_r: Vec<OpRef> = Vec::new();
    let call_descr = descr.as_call_descr().expect("CallI descr");
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let result = try_execute_residual_call_via_executor(
        &mut wc,
        majit_ir::OpCode::CallMayForceI,
        &allboxes,
        call_descr,
        recorded,
        0,
        None,
        false,
    );
    drop(wc);
    assert!(
        matches!(result, Ok(ResidualExecOutcome::Executed(Ok(_)))),
        "non-forcing may-force call with active vable must execute normally",
    );
    assert_eq!(
        tc.box_value(recorded),
        Some(majit_ir::Value::Int(42)),
        "active-vable may-force call must execute and stamp the result",
    );
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    let token = u64::from_le_bytes(
        vable_buf[info.token_offset..info.token_offset + 8]
            .try_into()
            .unwrap(),
    );
    assert_eq!(
        token, 0,
        "token must be cleared back to TOKEN_NONE after the call"
    );
}

static FORCING_CALLEE_TOKEN_ADDR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

extern "C" fn forces_vable_for_walker_test(a: i64, b: i64) -> i64 {
    // A force path clears the vable token (virtualizable.rs
    // force_now on TOKEN_TRACING_RESCALL).
    let addr = FORCING_CALLEE_TOKEN_ADDR.load(std::sync::atomic::Ordering::SeqCst);
    unsafe { *(addr as *mut u64) = 0 };
    a.wrapping_add(b)
}

#[test]
fn may_force_vable_escape_surfaces_typed_abort() {
    let mut tc = fresh_trace_ctx();
    let mut vable_buf = vec![0u8; 65536];
    bind_fake_vable(&mut tc, &mut vable_buf);
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    FORCING_CALLEE_TOKEN_ADDR.store(
        vable_buf.as_ptr() as usize + info.token_offset,
        std::sync::atomic::Ordering::SeqCst,
    );
    let funcbox = tc.const_int(forces_vable_for_walker_test as *const () as i64);
    let arg0 = tc.const_int(40);
    let arg1 = tc.const_int(2);
    let allboxes = [funcbox, arg0, arg1];
    let descr = make_call_descr(
        5,
        vec![Type::Int, Type::Int],
        Type::Int,
        majit_ir::ExtraEffect::CanRaise,
    );
    let recorded =
        tc.record_op_with_descr(majit_ir::OpCode::CallMayForceI, &allboxes, descr.clone());
    let regs_i: Vec<OpRef> = Vec::new();
    let regs_r: Vec<OpRef> = Vec::new();
    let call_descr = descr.as_call_descr().expect("CallI descr");
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let result = try_execute_residual_call_via_executor(
        &mut wc,
        majit_ir::OpCode::CallMayForceI,
        &allboxes,
        call_descr,
        recorded,
        7,
        None,
        false,
    );
    drop(wc);
    assert!(
        matches!(
            result,
            Err(DispatchError::VableEscapedDuringResidualCall { pc: 7 })
        ),
        "a callee that forces the vable must surface VableEscapedDuringResidualCall, got {result:?}",
    );
}

/// A `jit.not_in_trace` callee that returns normally.
///
/// `execute_residual_call` reads the raise out of `BH_LAST_EXC_VALUE`, so
/// leaving that cell alone is what "returned normally" means at this seam.
extern "C" fn not_in_trace_returns_for_walker_test() -> i64 {
    NOT_IN_TRACE_CALLS.with(|c| c.set(c.get() + 1));
    0
}

/// The same callee, raising: publishes a non-zero exception pointer on
/// `BH_LAST_EXC_VALUE` (the blackhole `bh_call_*` convention).
extern "C" fn not_in_trace_raises_for_walker_test() -> i64 {
    NOT_IN_TRACE_CALLS.with(|c| c.set(c.get() + 1));
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0xDEAD));
    0
}

thread_local! {
    /// Counts entries into the two callees above, so a test can tell
    /// "executed concretely" from "recorded symbolically and never run".
    static NOT_IN_TRACE_CALLS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Drive [`do_not_in_trace_call_result`] against a real callee address.
///
/// Returns the outcome, how many ops the trace grew by, and how many times
/// the callee actually ran.
fn run_not_in_trace(
    callee: extern "C" fn() -> i64,
    dst_bank: char,
) -> (Result<Option<DispatchOutcome>, DispatchError>, usize, u32) {
    let mut tc = fresh_trace_ctx();
    let funcbox = tc.const_int(callee as *const () as i64);
    let allboxes = [funcbox];
    let descr = call_descr_with_oopspec(
        44,
        majit_ir::ExtraEffect::CanRaise,
        majit_ir::OopSpecIndex::NotInTrace,
    );
    let call_descr = descr.as_call_descr().expect("CallDescr");
    let regs_i: Vec<OpRef> = Vec::new();
    let regs_r: Vec<OpRef> = Vec::new();
    let session = std::cell::RefCell::new(WalkSession::default());
    NOT_IN_TRACE_CALLS.with(|c| c.set(0));
    let ops_before = tc.ops().len();
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let ei = call_descr.get_extra_info();
    let outcome = do_not_in_trace_call_result(&mut wc, ei, &allboxes, call_descr, dst_bank, 0);
    drop(wc);
    let grew = tc.ops().len() - ops_before;
    (outcome, grew, NOT_IN_TRACE_CALLS.with(|c| c.get()))
}

#[test]
fn not_in_trace_normal_return_executes_concretely_and_records_no_ir() {
    // `do_not_in_trace_call` (pyjitpl.py) runs the callee through
    // `execute_varargs` and returns `None`, so `do_residual_call` hands its
    // caller no result box and the history never grows: "the trace doesn't
    // contain the call at all".
    let (outcome, ops_recorded, calls) =
        run_not_in_trace(not_in_trace_returns_for_walker_test, 'v');
    assert_eq!(
        outcome,
        Ok(Some(DispatchOutcome::Continue)),
        "a normally-returning OS_NOT_IN_TRACE callee must let the walk continue",
    );
    assert_eq!(
        calls, 1,
        "the callee must be executed concretely, exactly once"
    );
    assert_eq!(
        ops_recorded, 0,
        "OS_NOT_IN_TRACE records no IR — upstream returns None from \
         do_not_in_trace_call, so no ResOperation joins the history",
    );
}

#[test]
fn not_in_trace_raise_switches_to_blackhole_and_records_no_ir() {
    // pyjitpl.py: "cannot trace this!  it raises, so we have to follow the
    // exception-catching path, but the trace doesn't contain the call at
    // all" — `SwitchToBlackhole(Counters.ABORT_ESCAPE,
    // raising_exception=True)`.
    let (outcome, ops_recorded, calls) = run_not_in_trace(not_in_trace_raises_for_walker_test, 'v');
    assert_eq!(
        outcome,
        Ok(Some(DispatchOutcome::SwitchToBlackhole {
            reason: majit_metainterp::counters::ABORT_ESCAPE,
            raising_exception: true,
        })),
        "a raising OS_NOT_IN_TRACE callee must abort to blackhole",
    );
    assert_eq!(
        calls, 1,
        "the raise is observed by running the callee, not by guessing"
    );
    assert_eq!(
        ops_recorded, 0,
        "the aborting case records no IR either — the call is absent from the trace",
    );
}

#[test]
fn not_in_trace_with_a_destination_bank_stays_fail_loud() {
    // `jtransform.py` refuses to emit a `jit.not_in_trace` call whose result
    // is not `lltype.Void`, so only the `residual_call_*_v` spellings can
    // carry this oopspec. A destination bank here is a codewriter invariant
    // violation; the walker must not invent a value for it.
    let (outcome, _ops, calls) = run_not_in_trace(not_in_trace_returns_for_walker_test, 'r');
    assert_eq!(
        outcome,
        Err(DispatchError::NotInTraceRequiresConcreteExecution { pc: 0 }),
    );
    assert_eq!(
        calls, 0,
        "the guard must fail before running anything, so the refusal has no side effect",
    );
}

#[test]
fn residual_call_r_r_with_jit_force_virtual_oopspec_returns_typed_error() {
    // RPython parity: `pyjitpl.py` short-circuits
    // `do_residual_call` via `_do_jit_force_virtual` when
    // `effectinfo.oopspecindex == OS_JIT_FORCE_VIRTUAL`.  The
    // walker can't reproduce that short-circuit (needs concrete
    // `vref_ptr` resolver); fail-loud error keeps the path from
    // silently recording `CALL_MAY_FORCE_*` instead.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 1);
    let regs_r = distinct_const_refs(&mut tc, 4);
    let force_virtual_descr = call_descr_with_oopspec(
        42,
        majit_ir::ExtraEffect::ForcesVirtualOrVirtualizable,
        majit_ir::OopSpecIndex::JitForceVirtual,
    );
    let descr_pool = vec![force_virtual_descr];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("OS_JIT_FORCE_VIRTUAL must surface a typed error");
    assert_eq!(
        err,
        DispatchError::JitForceVirtualRequiresConcreteResolver { pc: 0 },
    );
}

#[test]
fn residual_call_r_r_with_elidable_can_raise_records_callpurer_plus_guard() {
    // EF_ELIDABLE_CAN_RAISE: `pure=True` + `exc=True` —
    // CALL_PURE_R + GUARD_NO_EXCEPTION (pyjitpl.py:execute_varargs
    // emits both).
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 1);
    let regs_r = distinct_const_refs(&mut tc, 4);
    let elidable_descr = call_descr_with_effect(8, majit_ir::ExtraEffect::ElidableCanRaise);
    let descr_pool = vec![elidable_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 2,
        "elidable+can-raise must record CallPureR + GuardNoException",
    );
    let opcodes: Vec<_> = tc.ops().iter().skip(ops_before).map(|o| o.opcode).collect();
    assert_eq!(
        opcodes,
        vec![
            majit_ir::OpCode::CallPureR,
            majit_ir::OpCode::GuardNoException
        ],
        "EF_ELIDABLE_CAN_RAISE must record CALL_PURE_R then GUARD_NO_EXCEPTION",
    );
}

#[test]
fn residual_call_r_r_with_cannot_raise_records_callr_no_guard() {
    // EF_CANNOT_RAISE: `pure=False` + `exc=False` — bare CallR,
    // no GUARD_NO_EXCEPTION.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 1);
    let regs_r = distinct_const_refs(&mut tc, 4);
    let nothrow_descr = call_descr_with_effect(9, majit_ir::ExtraEffect::CannotRaise);
    let descr_pool = vec![nothrow_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "EF_CANNOT_RAISE must record bare CallR (no GuardNoException)",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::CallR);
}

#[test]
fn residual_call_r_r_can_raise_writes_dst_before_guard_no_exception() {
    // pyjitpl.py _opimpl_residual_call*: result lands in
    // `registers_*[reg_index]` BEFORE
    // `handle_possible_exception()` records GUARD_NO_EXCEPTION.
    // `walker_capture_snapshot_for_last_guard`
    // (`pyjitpl.py capture_resumedata(after_residual_call
    // =True)`) snapshots the active registers AFTER the writeback,
    // so the dst slot's recorded OpRef rides the snapshot's
    // fail_arg list.  The structural invariant tested here is:
    // after dispatch, the dst slot holds the recorded call op's
    // OpRef, and the recorded sequence is `[CallR, GuardNoException]`
    // — i.e. the writeback ran on the record-side BEFORE the
    // guard append (and therefore before the snapshot capture).
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    // funcptr=regs_i[0], no R args, descr=0, dst=3.
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x03];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 1);
    let regs_r = distinct_const_refs(&mut tc, 8);
    let dst_pre = regs_r[3];
    let descr_pool = vec![make_call_descr(
        1,
        vec![],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    )];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
    let regs_r = wc.registers_r.to_vec();
    drop(wc);
    let opcodes: Vec<_> = tc.ops().iter().skip(ops_before).map(|o| o.opcode).collect();
    assert_eq!(
        opcodes,
        vec![OpCode::CallR, OpCode::GuardNoException],
        "CAN_RAISE residual_call_r_r must record [CallR, GuardNoException]",
    );
    let call_pos = tc
        .ops()
        .iter()
        .find(|o| o.opcode == OpCode::CallR)
        .expect("CallR must be in the trace")
        .pos()
        .get();
    assert_ne!(regs_r[3], dst_pre, "dst must be overwritten");
    assert_eq!(
        regs_r[3], call_pos,
        "registers_r[dst] must equal CallR's OpRef when GuardNoException is recorded",
    );
}

#[test]
fn residual_call_ir_r_can_raise_writes_dst_before_guard_no_exception() {
    // Same invariant as `residual_call_r_r_can_raise_...` for the
    // `_ir_*` shape (`dispatch_residual_call_iIRd_kind`): the
    // `iIRd>X` writeback must precede the GuardNoException record.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_ir_r/iIRd>r")
        .expect("`residual_call_ir_r/iIRd>r` must be in insns table");
    // funcptr=i[0], 0 i-args, 0 r-args, descr=0, dst=2.
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 1);
    let regs_r = distinct_const_refs(&mut tc, 6);
    let dst_pre = regs_r[2];
    let descr_pool = vec![make_call_descr(
        1,
        vec![],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    )];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let _ = step(&code, 0, &mut wc).expect("residual_call_ir_r/iIRd>r must dispatch");
    let regs_r = wc.registers_r.to_vec();
    drop(wc);
    let opcodes: Vec<_> = tc.ops().iter().skip(ops_before).map(|o| o.opcode).collect();
    assert_eq!(
        opcodes,
        vec![OpCode::CallR, OpCode::GuardNoException],
        "CAN_RAISE residual_call_ir_r must record [CallR, GuardNoException]",
    );
    let call_pos = tc
        .ops()
        .iter()
        .find(|o| o.opcode == OpCode::CallR)
        .expect("CallR must be in the trace")
        .pos()
        .get();
    assert_ne!(regs_r[2], dst_pre, "dst must be overwritten");
    assert_eq!(
        regs_r[2], call_pos,
        "registers_r[dst] must equal CallR's OpRef when GuardNoException is recorded",
    );
}

#[test]
fn residual_call_r_r_with_out_of_range_dst_register_surfaces_typed_error() {
    // Dst register OOR — the call was already recorded at this
    // point (RPython parity: `do_residual_or_indirect_call` records
    // first, then writes the result), but `registers_r` is empty
    // so the writeback surfaces RegisterOutOfRange.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x07]; // dst=7
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 1);
    // CallDescr required so the walker reaches the dst writeback
    // path (RPython do_residual_call invariant).
    let descr_pool = vec![make_call_descr(
        1,
        vec![],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    )];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("dst OOR must surface a typed error");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "r",
        },
    );
}

#[test]
fn residual_call_r_r_with_descr_index_out_of_range_surfaces_typed_error() {
    // descr-index OOR validation. Same shape as
    // RegisterOutOfRange, dedicated DispatchError variant for clarity.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    // descr_index=5, descr_refs.len()=2 → OOR
    let code = [residual_byte, 0x00, 0x00, 0x05, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 1);
    let descr_pool = vec![make_fail_descr(1), make_fail_descr(1)];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc)
        .expect_err("descr index 5 with pool size 2 must surface DescrIndexOutOfRange");
    assert_eq!(
        err,
        DispatchError::DescrIndexOutOfRange {
            pc: 0,
            index: 5,
            len: 2,
        },
    );
}

#[test]
fn step_through_residual_call_r_i_records_calli_with_int_dst_writeback() {
    // kind sibling of `_r_r`. Same `iRd>X` operand
    // layout, dst kind flipped to int. RPython `pyjitpl.py
    // opimpl_residual_call_r_i = _opimpl_residual_call1` shares
    // the body; `do_residual_call`'s `descr.get_normalized_result_type()`
    // dispatch (pyjitpl.py) selects `'i' → CALL_*_I`.
    // CallDescr required (RPython do_residual_call invariant);
    // walker records `OpCode::CallI` + `OpCode::GuardNoException`,
    // writes the call's OpRef into `registers_i[dst]`.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_i/iRd>i")
        .expect("`residual_call_r_i/iRd>i` must be in insns table");
    // Operand encoding `iRd>i`: 1B funcptr (i-reg=2),
    // 1B varlen=2 + [r-reg=4, r-reg=7], 2B descr_index=1 (LE),
    // 1B dst-reg=3 (writeback target into registers_i).
    let code = [
        residual_byte,
        0x02, // funcptr from registers_i[2]
        0x02, // varlen
        0x04,
        0x07, // args from registers_r[4, 7]
        0x01,
        0x00, // descr index = 1 (LE)
        0x03, // dst i-reg = 3
    ];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 8);
    let regs_r = distinct_const_refs(&mut tc, 8);
    let funcptr_expected = regs_i[2];
    let arg0_expected = regs_r[4];
    let arg1_expected = regs_r[7];
    let dst_pre = regs_i[3];
    // 2 R args + Int return → CallDescr arg_types=[Ref, Ref],
    // result_type=Int. `_build_allboxes` permutation is identity
    // (R-only argboxes match arg_types order).
    let decoy = make_call_descr(
        2,
        vec![Type::Ref, Type::Ref],
        Type::Int,
        majit_ir::ExtraEffect::CanRaise,
    );
    let call_descr = make_call_descr(
        3,
        vec![Type::Ref, Type::Ref],
        Type::Int,
        majit_ir::ExtraEffect::CanRaise,
    );
    let descr_pool = vec![decoy, call_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).expect("residual_call_r_i/iRd>i must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, code.len());
    // CallI (kind sibling of CallR) + GuardNoException recorded.
    assert_eq!(
        wc.trace_ctx.num_ops(),
        ops_before + 2,
        "_r_i must record CallI + GuardNoException (no-effectinfo fallback)",
    );
    let call_op = wc
        .trace_ctx
        .ops()
        .iter()
        .find(|o| o.opcode == majit_ir::OpCode::CallI)
        .expect("CallI must be recorded for the int-dst kind");
    assert_eq!(
        call_op
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![funcptr_expected, arg0_expected, arg1_expected],
        "CallI args must be [funcptr, ...args] from registers_i+registers_r",
    );
    let recorded_descr = call_op.getdescr().expect("CallI must carry the calldescr");
    assert!(
        std::sync::Arc::ptr_eq(&recorded_descr, &call_descr),
        "CallI descr must be descr_refs[1] (not decoy at index 0)",
    );
    // dst writeback into the int bank (NOT the r bank).
    let dst_post = wc.registers_i.get(3).expect("int register in range");
    assert_ne!(
        dst_post, dst_pre,
        "registers_i[dst] must change from its pre-call value",
    );
    assert_eq!(
        dst_post,
        call_op.pos().get(),
        "registers_i[dst] must be the recorded CallI's OpRef (op.pos().get())",
    );
}

#[test]
fn residual_call_r_i_with_elidable_cannot_raise_records_callpurei_no_guard() {
    // EF_ELIDABLE_CANNOT_RAISE on the int-kind sibling
    // must rewrite to CALL_PURE_I (not CALL_PURE_R) and skip
    // GUARD_NO_EXCEPTION. Confirms the `pure_op` selection in
    // `dispatch_residual_call_iRd_kind` follows `dst_bank`.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_i/iRd>i")
        .expect("`residual_call_r_i/iRd>i` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 4);
    let regs_r = distinct_const_refs(&mut tc, 4);
    let elidable_descr = call_descr_with_effect(7, majit_ir::ExtraEffect::ElidableCannotRaise);
    let descr_pool = vec![elidable_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("residual_call_r_i/iRd>i must dispatch");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "elidable+cannot-raise on int-kind must record exactly CallPureI",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(
        last.opcode,
        majit_ir::OpCode::CallPureI,
        "EF_ELIDABLE_CANNOT_RAISE on int-kind must rewrite to CALL_PURE_I",
    );
}

#[test]
fn step_through_residual_call_ir_r_records_callr_with_int_and_ref_args() {
    // shape sibling `_ir_r/iIRd>r`. Operand layout adds
    // an i-bank list between funcptr and the R-list. RPython
    // `_build_allboxes` permutes argboxes by `descr.get_arg_types()`
    // ABI; for an [Int, Int, Ref, Ref] callee the permutation
    // reduces to identity → allboxes = [funcbox, i0, i1, r0, r1].
    // Mixed-kind permutation is exercised by the dedicated test
    // `residual_call_ir_r_permutes_argboxes_per_arg_types_abi`.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_ir_r/iIRd>r")
        .expect("`residual_call_ir_r/iIRd>r` must be in insns table");
    // Operand encoding `iIRd>r`: 1B funcptr (i-reg=2),
    // i-list: 1B count=2 + [i-reg=5, i-reg=6],
    // r-list: 1B count=2 + [r-reg=4, r-reg=7],
    // 2B descr_index=1 (LE),
    // 1B dst-reg=0.
    let code = [
        residual_byte,
        0x02, // funcptr from registers_i[2]
        0x02, // i-list count
        0x05,
        0x06, // i-args from registers_i[5, 6]
        0x02, // r-list count
        0x04,
        0x07, // r-args from registers_r[4, 7]
        0x01,
        0x00, // descr index = 1 (LE)
        0x00, // dst r-reg = 0
    ];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 8);
    let regs_r = distinct_const_refs(&mut tc, 8);
    let funcptr_expected = regs_i[2];
    let iarg0_expected = regs_i[5];
    let iarg1_expected = regs_i[6];
    let rarg0_expected = regs_r[4];
    let rarg1_expected = regs_r[7];
    // arg_types = [Int, Int, Ref, Ref] → `_build_allboxes`
    // permutation is identity over the source-list-order argboxes.
    let decoy = make_call_descr(
        2,
        vec![Type::Int, Type::Int, Type::Ref, Type::Ref],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    );
    let call_descr = make_call_descr(
        3,
        vec![Type::Int, Type::Int, Type::Ref, Type::Ref],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    );
    let descr_pool = vec![decoy, call_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).expect("residual_call_ir_r/iIRd>r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc,
        code.len(),
        "residual_call_ir_r must advance past funcptr + i-list + r-list + descr + dst",
    );
    // CallR + GuardNoException recorded (no-effectinfo fallback).
    assert_eq!(
        wc.trace_ctx.num_ops(),
        ops_before + 2,
        "_ir_r must record CallR + GuardNoException (no-effectinfo fallback)",
    );
    let call_op = wc
        .trace_ctx
        .ops()
        .iter()
        .find(|o| o.opcode == majit_ir::OpCode::CallR)
        .expect("CallR must be recorded");
    assert_eq!(
        call_op
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![
            funcptr_expected,
            iarg0_expected,
            iarg1_expected,
            rarg0_expected,
            rarg1_expected,
        ],
        "CallR args must be [funcptr, i0, i1, r0, r1] — identity \
             permutation when descr.arg_types=[Int, Int, Ref, Ref]",
    );
    let recorded_descr = call_op.getdescr().expect("CallR must carry the calldescr");
    assert!(
        std::sync::Arc::ptr_eq(&recorded_descr, &call_descr),
        "CallR descr must be descr_refs[1] (not decoy at index 0)",
    );
    // dst writeback into registers_r[0].
    let dst_post = wc.registers_r.get(0).expect("ref register in range");
    assert_eq!(
        dst_post,
        call_op.pos().get(),
        "registers_r[dst] must be the recorded CallR's OpRef (op.pos().get())",
    );
}

#[test]
fn residual_call_ir_r_permutes_argboxes_per_arg_types_abi() {
    // The `_ir_*` shape gives
    // the walker source-list-order argboxes `[i_args..., r_args...]`,
    // but RPython `_build_allboxes` (pyjitpl.py) re-orders
    // those to match the callee's `descr.get_arg_types()` ABI. This
    // test pins the non-identity permutation.
    //
    // Setup: 2 i-args + 2 r-args + arg_types = [Ref, Int, Ref, Int].
    // Source-list-order argboxes = [i0, i1, r0, r1].
    // `_build_allboxes` walk:
    //   iter 1, kind=Ref: src_r scans flat argboxes for first Ref →
    //     positions 0 (i0, skip), 1 (i1, skip), 2 (r0, match). src_r=3.
    //   iter 2, kind=Int: src_i scans for first Int → position 0
    //     (i0, match). src_i=1.
    //   iter 3, kind=Ref: src_r at 3 → position 3 (r1, match). src_r=4.
    //   iter 4, kind=Int: src_i at 1 → position 1 (i1, match). src_i=2.
    // Final allboxes = [funcbox, r0, i0, r1, i1].
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_ir_r/iIRd>r")
        .expect("`residual_call_ir_r/iIRd>r` must be in insns table");
    let code = [
        residual_byte,
        0x02, // funcptr from registers_i[2]
        0x02, // i-list count
        0x05,
        0x06, // i-args from registers_i[5, 6]
        0x02, // r-list count
        0x04,
        0x07, // r-args from registers_r[4, 7]
        0x00,
        0x00, // descr index = 0 (LE)
        0x00, // dst r-reg = 0
    ];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 8);
    let regs_r = distinct_const_refs(&mut tc, 8);
    let funcptr = regs_i[2];
    let i0 = regs_i[5];
    let i1 = regs_i[6];
    let r0 = regs_r[4];
    let r1 = regs_r[7];
    let mixed_descr = make_call_descr(
        0,
        vec![Type::Ref, Type::Int, Type::Ref, Type::Int],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    );
    let descr_pool = vec![mixed_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let _ = step(&code, 0, &mut wc).expect("residual_call_ir_r/iIRd>r must dispatch");
    drop(wc);
    let call_op = tc
        .ops()
        .iter()
        .find(|o| o.opcode == majit_ir::OpCode::CallR)
        .expect("CallR must be recorded");
    assert_eq!(
        call_op
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![funcptr, r0, i0, r1, i1],
        "_build_allboxes must permute to match descr.arg_types \
             [Ref, Int, Ref, Int] — RPython pyjitpl.py:1960-1993",
    );
}

#[test]
fn residual_call_descr_not_call_descr_surfaces_typed_error() {
    // Walker requires CallDescr per RPython invariant
    // (pyjitpl.py do_residual_call). When the descr_pool entry
    // at the operand-encoded index lacks a CallDescr downcast (here
    // a FailDescr), the walker surfaces ResidualCallDescrNotCallDescr.
    // In production the codewriter never emits non-CallDescr; this
    // covers the test-fixture / future-deviation case.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 1);
    let regs_r = distinct_const_refs(&mut tc, 1);
    let descr_pool = vec![make_fail_descr(7)];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc)
        .expect_err("FailDescr (not CallDescr) must surface ResidualCallDescrNotCallDescr");
    assert_eq!(
        err,
        DispatchError::ResidualCallDescrNotCallDescr {
            pc: 0,
            descr_index: 0,
        },
    );
}

#[test]
fn residual_call_r_r_with_out_of_range_arg_register_surfaces_typed_error() {
    // varlist member OOR validation. Bank tag = "r" since
    // R-list reads from registers_r.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    // varlen=1, arg=9 (registers_r is empty) → OOR
    let code = [residual_byte, 0x00, 0x01, 0x09, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_i = distinct_const_refs(&mut tc, 1);
    let descr_pool = vec![make_fail_descr(1)];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc)
        .expect_err("R-list member out of range must surface RegisterOutOfRange");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 9,
            len: 0,
            bank: "r",
        },
    );
}

#[test]
#[ignore = "end-to-end walk of the execute_return_value helper reaches a goto_if_not whose value is symbolic in this unit setup (GotoIfNotValueNotConcrete { value: IntOp }); the walker needs concrete register values to decide the branch. Residual-call descrs now resolve (descr_pool_with_jitcode_adapters builds CallDescrs for BhDescr::Call); the remaining gap is a concrete-execution harness for the full helper body."]
fn walk_return_value_helper_terminates_at_first_ref_return() {
    // Acceptance: walk the ordinary `execute_return_value` JitCode
    // discovered through the portal closure end-to-end.
    // Layout (cranelift build):
    //
    //   pc=0..6   inline_call_r_r / dR>r  (recurse → SubReturn → caller dst write → Continue)
    //   pc=6..9   live /                  (continue)
    //   pc=9..11  ref_return / r          (terminate — top-level outermost)
    //   pc=11..18 (raise + ref_return tail, dead on this path)
    //
    // The helper's `inline_call_r_r` now recurses into the callee
    // jitcode via `production_sub_jitcodes` and
    // `descr_pool_with_jitcode_adapters`. The callee's
    // own `ref_return/r` surfaces as `SubReturn`; the caller writes
    // its dst register with that result and continues. The
    // caller's own `ref_return/r` at pc=9..11 then stashes the
    // outermost finish payload.
    let jc = named_jitcode("execute_return_value")
        .expect("execute_return_value must resolve through the portal closure");
    let mut tc = fresh_trace_ctx();
    // 256 distinct OpRefs (one per possible 1-byte register
    // index). `inline_call_r_r`'s recursion overwrites the dst
    // slot with the callee's `SubReturn` value, so the
    // post-recursion `ref_return/r` reads the *recorded* OpRef
    // from the sub-walk, not a `regs_r` constant. The assertion
    // therefore checks the recorded Finish's args against the
    // post-recursion register state, not a precomputed constant.
    let regs_r = distinct_const_refs(&mut tc, 256);
    let regs_i = distinct_const_refs(&mut tc, 256);
    let pool_len = crate::jitcode_runtime::all_descrs().len();
    let descr_pool = descr_pool_with_jitcode_adapters(pool_len);
    let ops_before = tc.num_ops();
    fbw_finish_payload_reset();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &production_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, end_pc) =
        walk(&jc.code, 0, &mut wc).expect("ReturnValue helper must walk to a terminator");
    assert!(
        matches!(outcome, DispatchOutcome::Terminate { .. }),
        "top-level walk must end on Terminate",
    );
    assert!(
        end_pc <= jc.code.len(),
        "walker must not run past the helper body \
             (end_pc={end_pc}, code.len()={})",
        jc.code.len(),
    );
    assert_eq!(
        end_pc, 11,
        "ReturnValue helper walker must terminate at outermost `ref_return/r` (pc=9..11)",
    );
    drop(wc);
    assert!(
        tc.num_ops() > ops_before,
        "the callee sub-walk must have contributed at least one op",
    );
    // The outermost `ref_return/r` records no FINISH itself: it stashes the
    // payload for `full_body_walk_trace`, which builds the portal-exit FINISH.
    // Sub-walks stash nothing (they surface `SubReturn`), so exactly one
    // payload — the caller's — is pending.
    assert_eq!(
        finish_payload_of(&outcome).map(|(_, ty)| ty),
        Some(Type::Ref),
        "outermost ref_return must stash a Type::Ref finish payload",
    );
}

#[test]
#[ignore = "end-to-end walk of the execute_pop_top helper reaches a goto_if_not \
        whose value is symbolic in this unit setup (GotoIfNotValueNotConcrete); \
        the walker needs concrete register values to decide the branch. \
        Residual-call descrs now resolve (descr_pool_with_jitcode_adapters \
        builds CallDescrs for BhDescr::Call); the remaining gap is a \
        concrete-execution harness for the full helper body. Bench traces never \
        reach a PopTop opcode at JIT record time, so production is \
        unaffected."]
fn walk_pop_top_helper_terminates_with_recorded_ops() {
    // Acceptance skeleton: walk the entire `execute_pop_top` helper
    // JitCode. The helper body is 25 bytes / 9 ops after the
    // jtransform `Ok` / `Err` / `Some` identity rewrite stripped
    // the trailing `int_copy + residual_call_r_r/iRd>r` wrapper
    // for the `Ok(StepResult::Continue)` return value
    // (`majit/majit-translate/src/codewriter/jtransform.rs
    //  ::rewrite_op_direct_call`).  The current sequence is:
    //
    //     inline_call_r_r/dR>r ; live/ ; catch_exception/L ;
    //     goto/L ; reraise/ ; ref_return/r ; live/ ;
    //     raise/r ; ref_return/r
    //
    // Every outer opname has a handler in this module.  The
    // remaining gap lives two levels deeper: PopTop's
    // `inline_call_r_r/dR>r` recurses into the `pop_top` callee
    // which itself recurses into a body whose first byte is
    // `getfield_vable_i/rd>i` — currently surfaced as
    // `UnsupportedOpname` by the walker.  Unignoring this test
    // requires landing the vable-aware getfield handlers (Phase
    // D-3 MIFrame integration).
    let jc = named_jitcode("execute_pop_top")
        .expect("execute_pop_top must resolve through the portal closure");
    let mut tc = fresh_trace_ctx();
    // Generously sized banks so any byte the codewriter emits is
    // in-range. 256 is the maximum register index a 1-byte slot
    // can address.
    let regs_r = distinct_const_refs(&mut tc, 256);
    let regs_i = distinct_const_refs(&mut tc, 256);
    // Descr pool: slot at each `BhDescr::JitCode` index in
    // `all_descrs()` is wrapped in a `TestJitCodeDescr` adapter so
    // `inline_call_r_r/dR>r` can resolve `as_jitcode_descr()`.
    // Other slots default to `make_fail_descr`.
    let pool_len = crate::jitcode_runtime::all_descrs().len();
    let descr_pool = descr_pool_with_jitcode_adapters(pool_len);
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &production_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, end_pc) =
        walk(&jc.code, 0, &mut wc).expect("PopTop helper must walk to a terminator");
    // PopTop's `inline_call_r_r/dR>r` recurses into the
    // codewriter-emitted callee jitcode (resolved via
    // `production_sub_jitcodes`); on success the outermost
    // `ref_return/r` lands a FINISH at top level.
    assert!(
        matches!(outcome, DispatchOutcome::Terminate { .. }),
        "top-level PopTop walk must end on Terminate",
    );
    assert!(
        end_pc <= jc.code.len(),
        "walker must not run past the arm body \
             (end_pc={end_pc}, code.len()={})",
        jc.code.len(),
    );
    drop(wc);
    let ops_after = tc.num_ops();
    assert!(
        ops_after > ops_before,
        "PopTop walk must record at least one op (FINISH from \
             ref_return at top level) — recorded {} → {}",
        ops_before,
        ops_after,
    );
    // No `residual_call_r_r/iRd>r` in the post-rewrite arm, so no
    // CallR-descr identity check applies here.  If a future arm
    // regrows the wrapper, restore the `as_call_descr().is_some()`
    // + `Arc::ptr_eq(real_call_descr)` checks that lived here in
    // the pre-`Ok` / `Err` / `Some` identity rewrite version of
    // this fixture.
}

/// A translated helper has no Python blackhole entry. The limit check waits
/// for the enclosing Python step: aborting on the helper's own instruction
/// retraces the same overflow once per helper op. The caller here is one
/// `inline_call_r_v/dR` followed by its terminator, already over the limit.
#[test]
fn helper_descent_defers_the_limit_check_to_the_enclosing_frame() {
    fn walk_past_the_limit(
        transparent_helper_subwalk: bool,
    ) -> Result<(DispatchOutcome, usize), DispatchError> {
        let void_ret = *insns_opname_to_byte()
            .get("void_return/")
            .expect("`void_return/` must be in insns table");
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_v/dR")
            .expect("`inline_call_r_v/dR` must be in insns table");
        let callee_code: &'static [u8] = Box::leak(Box::new([void_ret]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 0,
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| (idx == 7).then(|| sub_body.clone())
        };
        let caller_code = [
            inline_byte,
            0x07,
            0x00, // descr index 7
            0x01,
            0x00,     // R: len=1, arg=r0
            void_ret, // caller terminates
        ];
        let mut tc = fresh_trace_ctx();
        let regs_r = distinct_const_refs(&mut tc, 4);
        // One recorded op against a zero limit: the walk is over budget before
        // its first step, so whichever frame owns the check aborts immediately.
        dummy_ptr_eq_box(&mut tc);
        tc.set_trace_limit(0);
        assert!(tc.is_too_long(), "the walk must start over budget");
        // Bridge-shaped (pyjitpl.py:2908), so the abort's warm-state half —
        // `note_root_trace_too_long`, which reaches the driver through the
        // merge point — is a no-op and the frame that owns the check is the
        // only thing this fixture measures.
        tc.clear_merge_points();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = make_jitcode_descr(7);
        let session = std::cell::RefCell::new(WalkSession::default());
        let mut wc = WalkContext {
            frame_state: WalkFrameState::new(WalkFrameStateData {
                callee_shadow: None,
                concrete_registers_r: ([]).to_vec(),
                outer_active_boxes: Vec::new(),
                vstack_boxes: Vec::new(),
                vstack_last_ref: OpRef::NONE,
                vstack_reorder_saved: None,
                ..Default::default()
            }),
            inline_callee_consts: None,
            fbw_mode: FbwWalkMode {
                transparent_helper_subwalk,
                ..test_fbw_mode()
            },
            session: &session,
            registers_r: &RegisterBank::new(regs_r.iter().copied()),
            registers_i: &RegisterBank::default(),
            registers_f: &RegisterBank::default(),

            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            raw_descrs: RawDescrPool::Global,
            is_authoritative_executor: false,
            trace_ctx: &mut tc,
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            entry_py_pc: EntryPyPc::Py(0),
            outer_resume_marker_jit_pc: None,
            outer_jitcode_index: 0,

            pending_guard_snapshot_error: None,

            vstack_depth: 0,
            vstack_cur_pypc: 0,
            vstack_valid: false,

            vstack_reorder_ceiling: u32::MAX,

            vstack_handler_landing_py: None,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
        };
        walk(&caller_code, 0, &mut wc)
    }

    // The helper body is already over `trace_limit` and does not abort on its
    // own `void_return/`. The enclosing frame, when it is a real Python walk,
    // aborts at the `inline_call` once the helper has returned. A caller that
    // is itself a transparent helper defers too and finishes.
    let enclosing = walk_past_the_limit(false);
    assert!(
        matches!(enclosing, Err(DispatchError::TraceTooLong { pc: 0, .. })),
        "the enclosing frame aborts after the helper returns: {enclosing:?}"
    );
    assert!(
        walk_past_the_limit(true).is_ok(),
        "a transparent caller finishes the helper past trace_limit"
    );
}

#[test]
fn inline_call_with_more_args_than_callee_regs_surfaces_arity_mismatch() {
    // codewriter shape contract says `R-list.len() <=
    // callee.num_regs_r` for `inline_call_r_r/dR>r`. Walker rejects
    // overflow with a typed error instead of silently dropping
    // (the dropped args would carry symbolic OpRefs the callee
    // never reads, breaking dataflow).
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    // Callee declares num_regs_r=1 but caller passes 2 ref args.
    let callee_code: &'static [u8] = Box::leak(Box::new([0xFFu8])); // unreachable
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = move |idx: usize| {
        if idx == 5 {
            Some(sub_body.clone())
        } else {
            None
        }
    };
    // R-list = [r0, r1] but callee has only 1 slot.
    let caller_code = [inline_byte, 0x05, 0x00, 0x02, 0x00, 0x01, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[5] = make_jitcode_descr(5);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&caller_code, 0, &mut wc).expect_err("arity overflow must surface error");
    assert_eq!(
        err,
        DispatchError::InlineCallArityMismatch {
            pc: 0,
            provided: 2,
            callee_num_regs_r: 1,
        },
    );
}

// ── inline_call_*_v regression tests ──────────────────────────────
//
// Exercise the void-return contract for all three dispatch variants:
//   dispatch_inline_call_dr_kind  (inline_call_r_v/dR)
//   dispatch_inline_call_dir_kind (inline_call_ir_v/dIR)
//   dispatch_inline_call_dirf_kind(inline_call_irf_v/dIRF)

#[test]
fn inline_call_r_v_accepts_void_returning_callee() {
    // Callee body: `void_return/` — surfaces SubReturn { None }.
    // Caller: `inline_call_r_v/dR  descr=7 R=[r0]` then `void_return/`.
    let void_ret = *insns_opname_to_byte()
        .get("void_return/")
        .expect("`void_return/` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_v/dR")
        .expect("`inline_call_r_v/dR` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([void_ret]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // dR layout: 2B descr(7) + 1B R-len(1) + 1B R-arg(r0)  — no >X dst
    let caller_code = [
        inline_byte,
        0x07,
        0x00, // descr index 7
        0x01,
        0x00,     // R: len=1, arg=r0
        void_ret, // caller terminates
    ];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 4);
    let stale_exc = regs_r[0];
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    // The standing exception is metainterp-level state
    // (`pyjitpl.py:2417`), so it is seeded on the session.
    {
        let mut sess = session.borrow_mut();
        sess.last_exc_value = Some(stale_exc);
        sess.last_exc_value_concrete = ConcreteValue::Int(123);
    }
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, _) =
        walk(&caller_code, 0, &mut wc).expect("inline_call_r_v with void callee must succeed");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
    assert_eq!(wc.last_exc_value(), None);
    assert_eq!(wc.last_exc_value_concrete(), ConcreteValue::Null);
}

#[test]
fn inline_call_r_v_rejects_non_void_returning_callee() {
    // Callee body: `ref_return r0` — surfaces SubReturn { Some(_) }.
    // inline_call_r_v must reject with UnexpectedNonVoidSubReturn.
    let ref_ret = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_v/dR")
        .expect("`inline_call_r_v/dR` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([ref_ret, 0x00]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    let caller_code = [
        inline_byte,
        0x07,
        0x00, // descr index 7
        0x01,
        0x00, // R: len=1, arg=r0
    ];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = walk(&caller_code, 0, &mut wc)
        .expect_err("inline_call_r_v with non-void callee must reject");
    assert_eq!(err, DispatchError::UnexpectedNonVoidSubReturn { pc: 0 });
}

#[test]
fn inline_call_ir_v_accepts_void_returning_callee() {
    let void_ret = *insns_opname_to_byte()
        .get("void_return/")
        .expect("`void_return/` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_ir_v/dIR")
        .expect("`inline_call_ir_v/dIR` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([void_ret]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 1,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // dIR layout: 2B descr(7) + I-list(len=1, i0) + R-list(len=1, r0) — no dst
    let caller_code = [
        inline_byte,
        0x07,
        0x00, // descr index 7
        0x01,
        0x00, // I: len=1, arg=i0
        0x01,
        0x00,     // R: len=1, arg=r0
        void_ret, // caller terminates
    ];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 4);
    let regs_i = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, _) =
        walk(&caller_code, 0, &mut wc).expect("inline_call_ir_v with void callee must succeed");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
}

#[test]
fn inline_call_ir_v_rejects_non_void_returning_callee() {
    let int_ret = *insns_opname_to_byte()
        .get("int_return/i")
        .expect("`int_return/i` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_ir_v/dIR")
        .expect("`inline_call_ir_v/dIR` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([int_ret, 0x00]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 1,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    let caller_code = [
        inline_byte,
        0x07,
        0x00,
        0x01,
        0x00, // I: len=1, arg=i0
        0x01,
        0x00, // R: len=1, arg=r0
    ];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 4);
    let regs_i = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = walk(&caller_code, 0, &mut wc)
        .expect_err("inline_call_ir_v with non-void callee must reject");
    assert_eq!(err, DispatchError::UnexpectedNonVoidSubReturn { pc: 0 });
}

#[test]
fn inline_call_irf_v_accepts_void_returning_callee() {
    let void_ret = *insns_opname_to_byte()
        .get("void_return/")
        .expect("`void_return/` must be in insns table");
    let &inline_byte = insns_opname_to_byte()
        .get("inline_call_irf_v/dIRF")
        .expect("`inline_call_irf_v/dIRF` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([void_ret]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 1,
        num_regs_f: 1,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // dIRF layout: 2B descr + I-list + R-list + F-list — no dst
    let caller_code = [
        inline_byte,
        0x07,
        0x00, // descr index 7
        0x01,
        0x00, // I: len=1, arg=i0
        0x01,
        0x00, // R: len=1, arg=r0
        0x01,
        0x00,     // F: len=1, arg=f0
        void_ret, // caller terminates
    ];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 4);
    let regs_i = distinct_const_refs(&mut tc, 4);
    let regs_f = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::new(regs_f.iter().copied()),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, _) =
        walk(&caller_code, 0, &mut wc).expect("inline_call_irf_v with void callee must succeed");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
}

#[test]
fn inline_call_irf_v_rejects_non_void_returning_callee() {
    let ref_ret = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let &inline_byte = insns_opname_to_byte()
        .get("inline_call_irf_v/dIRF")
        .expect("`inline_call_irf_v/dIRF` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([ref_ret, 0x00]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 1,
        num_regs_f: 1,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    let caller_code = [
        inline_byte,
        0x07,
        0x00,
        0x01,
        0x00, // I
        0x01,
        0x00, // R
        0x01,
        0x00, // F
    ];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 4);
    let regs_i = distinct_const_refs(&mut tc, 4);
    let regs_f = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::new(regs_f.iter().copied()),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = walk(&caller_code, 0, &mut wc)
        .expect_err("inline_call_irf_v with non-void callee must reject");
    assert_eq!(err, DispatchError::UnexpectedNonVoidSubReturn { pc: 0 });
}

/// Build a `SimpleFieldDescr` with a stable index so the
/// heapcache lookup hashes consistently across the cache-miss
/// and cache-hit assertions. Default `Descr::index()` returns
/// `u32::MAX`; tests that exercise heapcache need a real index.
fn field_descr_with_index(idx: u32) -> DescrRef {
    std::sync::Arc::new(majit_ir::SimpleFieldDescr::new(
        idx,
        8, // offset
        8, // field_size
        majit_ir::Type::Int,
        false, // not immutable
    ))
}

#[test]
fn getfield_gc_i_cache_miss_records_op_and_writes_dst() {
    // First `getfield_gc_i/rd>i` invocation
    // is a heapcache miss — walker records `OpCode::GetfieldGcI`
    // with `[obj]` and `descr=descr_refs[d]`, writes the
    // recorder result into `registers_i[dst]`, and updates the
    // heapcache via `getfield_now_known(resbox)`.
    let byte = *insns_opname_to_byte()
        .get("getfield_gc_i/rd>i")
        .expect("`getfield_gc_i/rd>i` must be in insns table");
    // Operand layout `rd>i`: 1B r-reg(2) + 2B descr-index(LE 1) + 1B dst(5).
    let code = [byte, 0x02, 0x01, 0x00, 0x05];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let regs_i = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let dst_pre = regs_i[5];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("getfield_gc_i must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 5, "getfield_gc_i/rd>i operand layout = 4 bytes");
    let dst_post = wc.registers_i.get(5).expect("int register in range");
    assert_ne!(
        dst_post, dst_pre,
        "cache miss must write a fresh recorder OpRef into registers_i[dst]",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "cache miss must record exactly one GetfieldGcI op",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::GetfieldGcI);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![obj],
        "GetfieldGcI args must be [obj] (the r-reg source)",
    );
    let recorded_descr = last
        .getdescr()
        .expect("GetfieldGcI must carry the field descr");
    assert!(
        std::sync::Arc::ptr_eq(&recorded_descr, &descr),
        "GetfieldGcI descr must be descr_refs[d] (the field descr)",
    );
    assert_eq!(dst_post, last.pos().get());
}

#[test]
fn getfield_gc_i_cache_hit_returns_cached_box_without_recording() {
    // Second invocation with the same
    // (obj, descr) pair must hit the heapcache and skip IR
    // emission. RPython parity:
    //   upd = heapcache.get_field_updater(box, fielddescr)
    //   if upd.currfieldbox is not None:
    //       return upd.currfieldbox  # no execute_with_descr
    let byte = *insns_opname_to_byte()
        .get("getfield_gc_i/rd>i")
        .expect("`getfield_gc_i/rd>i` must be in insns table");
    let code = [byte, 0x02, 0x01, 0x00, 0x05];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let regs_i = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();

    // Pre-populate the heapcache as if a previous getfield had
    // already cached the field's value. RPython equivalent:
    // `heapcache.getfield_now_known(...)` after a prior fetch.
    let cached_field = tc.const_int(0xCAFE);
    tc.heapcache_getfield_now_known(obj, 1, cached_field);
    let ops_before = tc.num_ops();

    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("getfield_gc_i must dispatch");
    let dst_post = wc.registers_i.get(5).expect("int register in range");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "cache hit must NOT record any new IR op",
    );
    assert_eq!(
        dst_post, cached_field,
        "cache hit must write the cached OpRef into registers_i[dst]",
    );
}

#[test]
fn getfield_gc_r_cache_miss_records_op_and_writes_ref_dst() {
    // GetfieldGcR variant — same flow as
    // GetfieldGcI but result lands in registers_r.
    let byte = *insns_opname_to_byte()
        .get("getfield_gc_r/rd>r")
        .expect("`getfield_gc_r/rd>r` must be in insns table");
    let code = [byte, 0x02, 0x01, 0x00, 0x06];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let dst_pre = regs_r[6];
    // Use a Ref-typed field descr — sanity-check that the walker
    // doesn't introspect the descr's field_type (it just feeds
    // descr_index into the heapcache and records the op).
    let descr: DescrRef = std::sync::Arc::new(majit_ir::SimpleFieldDescr::new(
        1,
        16,
        8,
        majit_ir::Type::Ref,
        false,
    ));
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("getfield_gc_r must dispatch");
    let dst_post = wc.registers_r.get(6).expect("ref register in range");
    assert_ne!(dst_post, dst_pre);
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::GetfieldGcR);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![obj]
    );
    assert_eq!(dst_post, last.pos().get());
}

#[test]
fn getfield_gc_with_out_of_range_obj_register_surfaces_typed_error() {
    let byte = *insns_opname_to_byte()
        .get("getfield_gc_i/rd>i")
        .expect("`getfield_gc_i/rd>i` must be in insns table");
    let code = [byte, 0x07, 0x00, 0x00, 0x00]; // r-reg=7, registers_r empty
    let mut tc = fresh_trace_ctx();
    let descr = field_descr_with_index(0);
    let descr_pool = vec![descr];
    let frame_done = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("getfield_gc must validate r-reg");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "r",
        },
    );
}

#[test]
fn getfield_vable_i_routes_through_metainterp_and_writes_dst() {
    // T2 sanity: `getfield_vable_i/rd>i` delegates to
    // `walker_nonstandard_virtualizable` + `vable_getfield_int_checked`.
    // With no `virtualizable_info` bound on the trace context, begin
    // returns Decided(true) and the fallback emits a `GetfieldGcI` op +
    // writes the recorder OpRef into `registers_i[dst]` — the same
    // shape `getfield_gc_via_heapcache` produces on a cache miss.
    // The handler itself stays orthodox to RPython
    // `pyjitpl.py opimpl_getfield_vable_i`; the
    // GETFIELD_GC fallback is the checked body's decision, not
    // the walker's, so this test exercises the walker→trace_ctx
    // boundary without depending on a `virtualizable_info` fixture.
    let byte = *insns_opname_to_byte()
        .get("getfield_vable_i/rd>i")
        .expect("`getfield_vable_i/rd>i` must be in insns table");
    // Operand layout `rd>i`: 1B r-reg(2) + 2B descr-index(LE 1) + 1B dst(5).
    let code = [byte, 0x02, 0x01, 0x00, 0x05];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let regs_i = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let dst_pre = regs_i[5];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("getfield_vable_i must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 5, "getfield_vable_i/rd>i operand layout = 4 bytes");
    let dst_post = wc.registers_i.get(5).expect("int register in range");
    assert_ne!(
        dst_post, dst_pre,
        "fallback must write a fresh recorder OpRef into registers_i[dst]",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "nonstandard-vable fallback records exactly one GetfieldGcI op",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::GetfieldGcI);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![obj],
        "GetfieldGcI args must be [obj] (the r-reg source)",
    );
    let recorded_descr = last
        .getdescr()
        .expect("GetfieldGcI must carry the field descr");
    assert!(
        std::sync::Arc::ptr_eq(&recorded_descr, &descr),
        "GetfieldGcI descr must be descr_refs[d] (the field descr)",
    );
    assert_eq!(dst_post, last.pos().get());
}

#[test]
fn setfield_vable_i_routes_through_metainterp_records_setfield_gc_fallback() {
    // T2a sanity: `setfield_vable_i/rid` delegates to
    // `walker_nonstandard_virtualizable` + `vable_setfield_checked`.
    // With no `virtualizable_info` bound on the trace context, begin
    // returns Decided(true) and the fallback records a `SetfieldGc` op
    // with `[obj, value]` + the field descr — same shape
    // `setfield_gc_via_heapcache` produces.  Exercises the
    // walker -> trace_ctx boundary for the int-bank variant
    // (the `r` and `f` variants share the handler body, varying
    // only `value_bank`).
    let byte = *insns_opname_to_byte()
        .get("setfield_vable_i/rid")
        .expect("`setfield_vable_i/rid` must be in insns table");
    // Operand layout `rid`: 1B r-reg(2) + 1B i-reg(3) + 2B descr-index(LE 1).
    let code = [byte, 0x02, 0x03, 0x01, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let regs_i = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let value = regs_i[3];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("setfield_vable_i must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 5, "setfield_vable_i/rid operand layout = 4 bytes");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "nonstandard-vable fallback records exactly one SetfieldGc op",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::SetfieldGc);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![obj, value],
        "SetfieldGc args must be [obj, value]",
    );
    let recorded_descr = last
        .getdescr()
        .expect("SetfieldGc must carry the field descr");
    assert!(
        std::sync::Arc::ptr_eq(&recorded_descr, &descr),
        "SetfieldGc descr must be descr_refs[d] (the field descr)",
    );
}

#[test]
fn setfield_gc_i_redundant_write_skips_recording() {
    // When the heapcache already knows
    // valuebox is the current value of (obj, descr), the
    // SETFIELD_GC IR op must NOT be recorded. RPython parity:
    // `pyjitpl.py if upd.currfieldbox is valuebox: return`.
    let byte = *insns_opname_to_byte()
        .get("setfield_gc_i/rid")
        .expect("`setfield_gc_i/rid` must be in insns table");
    let code = [byte, 0x02, 0x03, 0x01, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let regs_i = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let valuebox = regs_i[3];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr];
    let frame_done = done_descr_ref_for_tests();
    // Pre-cache valuebox as the current field value.
    tc.heapcache_getfield_now_known(obj, 1, valuebox);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("setfield_gc_i must dispatch");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "redundant setfield (cached valuebox == new valuebox) must skip recording",
    );
}

#[test]
fn setfield_gc_i_fresh_write_records_op_and_caches_value() {
    // A fresh write (no cached value)
    // must record SETFIELD_GC and update the heapcache so a
    // subsequent redundant write hits.
    let byte = *insns_opname_to_byte()
        .get("setfield_gc_i/rid")
        .expect("`setfield_gc_i/rid` must be in insns table");
    let code = [byte, 0x02, 0x03, 0x01, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let regs_i = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let valuebox = regs_i[3];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("setfield_gc_i must dispatch");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "fresh setfield must record exactly one SetfieldGc op",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::SetfieldGc);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![obj, valuebox],
        "SetfieldGc args must be [obj, valuebox] in that order",
    );
    assert!(std::sync::Arc::ptr_eq(
        &last.getdescr().expect("SetfieldGc must carry descr"),
        &descr,
    ),);
    // Cache must now know the new field value.  Box identity-only
    // check — value payload is Void in walker-emitted writes.
    assert_eq!(
        tc.heapcache_getfield_cached(obj, 1).map(|b| b),
        Some(valuebox),
        "post-setfield, the heapcache must reflect the written value",
    );
}

#[test]
fn setfield_gc_r_records_setfieldgc_with_ref_valuebox() {
    // `rrd` shape — both box and valuebox
    // come from registers_r. SetfieldGc is type-agnostic at the
    // IR level (the descr carries the field type).
    let byte = *insns_opname_to_byte()
        .get("setfield_gc_r/rrd")
        .expect("`setfield_gc_r/rrd` must be in insns table");
    let code = [byte, 0x02, 0x05, 0x01, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let valuebox = regs_r[5];
    let descr: DescrRef = std::sync::Arc::new(majit_ir::SimpleFieldDescr::new(
        1,
        16,
        8,
        majit_ir::Type::Ref,
        false,
    ));
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("setfield_gc_r must dispatch");
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::SetfieldGc);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![obj, valuebox]
    );
}

#[test]
fn getarrayitem_gc_r_cache_miss_records_op_and_writes_dst() {
    // First `getarrayitem_gc_r/rid>r` is a
    // heapcache miss — record GetarrayitemGcR with
    // [array, index] + descr; write recorder result into r-dst
    // and update heapcache.
    let byte = *insns_opname_to_byte()
        .get("getarrayitem_gc_r/rid>r")
        .expect("`getarrayitem_gc_r/rid>r` must be in insns table");
    // Operand layout `rid>r`: 1B r-reg(2) + 1B i-reg(3) +
    // 2B descr(LE 1) + 1B r-dst(5).
    let code = [byte, 0x02, 0x03, 0x01, 0x00, 0x05];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let regs_i = distinct_const_refs(&mut tc, 8);
    let array = regs_r[2];
    let index = regs_i[3];
    let dst_pre = regs_r[5];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("getarrayitem_gc_r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 6,
        "getarrayitem_gc_r/rid>r operand layout = 5 bytes"
    );
    let dst_post = wc.registers_r.get(5).expect("ref register in range");
    assert_ne!(dst_post, dst_pre);
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::GetarrayitemGcR);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![array, index],
        "GetarrayitemGcR args must be [array, index]",
    );
    assert!(std::sync::Arc::ptr_eq(
        &last.getdescr().expect("must carry array descr"),
        &descr,
    ));
    assert_eq!(dst_post, last.pos().get());
}

/// Backend stub for the pure-getarrayitem bypass test: the required
/// methods are unreachable; `bh_getarrayitem_gc_*` trait defaults read
/// the real backing memory.
struct PureArrayTestCpu;
impl majit_backend::Backend for PureArrayTestCpu {
    fn compile_loop(
        &mut self,
        _inputargs: &[majit_ir::InputArgRc],
        _ops: &[majit_ir::OpRc],
        _token: &majit_backend::JitCellToken,
    ) -> Result<majit_backend::AsmInfo, majit_backend::BackendError> {
        unimplemented!("PureArrayTestCpu::compile_loop")
    }
    fn compile_bridge(
        &mut self,
        _fail_descr: &dyn majit_ir::FailDescr,
        _inputargs: &[majit_ir::InputArgRc],
        _ops: &[majit_ir::OpRc],
        _original_token: &majit_backend::JitCellToken,
        _previous_tokens: &[std::sync::Arc<majit_backend::JitCellToken>],
        _caller_recovery_layout: Option<&majit_backend::ExitRecoveryLayout>,
    ) -> Result<majit_backend::AsmInfo, majit_backend::BackendError> {
        unimplemented!("PureArrayTestCpu::compile_bridge")
    }
    fn execute_token(
        &self,
        _token: &majit_backend::JitCellToken,
        _args: &[majit_ir::Value],
    ) -> majit_backend::DeadFrame {
        unimplemented!("PureArrayTestCpu::execute_token")
    }
    fn get_latest_descr<'a>(
        &'a self,
        _frame: &'a majit_backend::DeadFrame,
    ) -> &'a dyn majit_ir::FailDescr {
        unimplemented!("PureArrayTestCpu::get_latest_descr")
    }
    fn get_latest_descr_arc(
        &self,
        _frame: &majit_backend::DeadFrame,
    ) -> std::sync::Arc<dyn majit_ir::descr::Descr> {
        unimplemented!("PureArrayTestCpu::get_latest_descr_arc")
    }
    fn get_int_value(&self, _frame: &majit_backend::DeadFrame, _index: usize) -> i64 {
        unimplemented!("PureArrayTestCpu::get_int_value")
    }
    fn get_value_direct(&self, frame: &majit_backend::DeadFrame, slot: usize) -> i64 {
        // PureArrayTestCpu's slot space is the dense fail-value vector.
        self.get_int_value(frame, slot)
    }
    fn get_float_value(&self, _frame: &majit_backend::DeadFrame, _index: usize) -> f64 {
        unimplemented!("PureArrayTestCpu::get_float_value")
    }
    fn get_ref_value(&self, _frame: &majit_backend::DeadFrame, _index: usize) -> majit_ir::GcRef {
        unimplemented!("PureArrayTestCpu::get_ref_value")
    }
    fn invalidate_loop(&self, _token: &majit_backend::JitCellToken) {
        unimplemented!("PureArrayTestCpu::invalidate_loop")
    }
}

/// `_opimpl_getarrayitem_gc_pure_any`: directly-constant array and index
/// operands bypass the heapcache — the load executes now, the result is
/// a `Const`, nothing is recorded, and no profiler bucket counts (the
/// bypass calls `executor.execute`, not `execute_and_record`).
#[test]
fn getarrayitem_gc_pure_const_operands_fold_without_recording_or_counting() {
    let byte = *insns_opname_to_byte()
        .get("getarrayitem_gc_i_pure/rid>i")
        .expect("`getarrayitem_gc_i_pure/rid>i` must be in insns table");
    let code = [byte, 0x02, 0x03, 0x01, 0x00, 0x05];
    let cpu = PureArrayTestCpu;
    let mut tc = fresh_trace_ctx();
    tc.set_cpu(Some(&cpu));
    // Fake GcArray block: length prefix at offset 0, items from
    // base_size 8 → item[1] == 9.
    let storage: Box<[usize; 3]> = Box::new([2, 7, 9]);
    let ptr = storage.as_ref().as_ptr() as i64;
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i = distinct_const_ints(&mut tc, 8);
    regs_r[2] = tc.const_ref(ptr);
    regs_i[3] = tc.const_int(1);
    let descr = crate::descr::make_array_descr(8, 8, Some(0), majit_ir::Type::Int, true);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let prof_before = tc.profiler().snapshot();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("getarrayitem_gc_i_pure must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 6);
    let dst_post = wc.registers_i.get(5).expect("int register in range");
    drop(wc);
    assert!(dst_post.is_constant(), "the bypass substitutes a Const");
    assert_eq!(
        dst_post.inline_const_to_value(),
        Some(majit_ir::Value::Int(9)),
        "item[1] of the backing array"
    );
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "the bypass records no ArrayitemGc op"
    );
    let prof_after = tc.profiler().snapshot();
    assert_eq!(prof_after.ops, prof_before.ops);
    assert_eq!(prof_after.recorded_ops, prof_before.recorded_ops);
    assert_eq!(prof_after.heapcached_ops, prof_before.heapcached_ops);
    drop(storage);
}

/// Headerless `GETARRAYITEM_GC` (`Box<[T]>` data, no `lendescr`).
///
/// `arraylen_sanity_load` declines that descr, so `index_in_array_bounds`
/// cannot prove the index. A constant index is still the load
/// `execute_with_descr` runs. A non-constant index, even one whose Box
/// value is a small int, stays unstamped: that is the Ref-bit-pattern
/// case `bh_getarrayitem_gc_r` SIGBUS'd on.
#[test]
fn getarrayitem_gc_headerless_const_index_stamps_the_byte() {
    let byte = *insns_opname_to_byte()
        .get("getarrayitem_gc_i/rid>i")
        .expect("`getarrayitem_gc_i/rid>i` must be in insns table");
    let code = [byte, 0x02, 0x03, 0x01, 0x00, 0x05];
    let cpu = PureArrayTestCpu;
    let storage: Box<[u8]> = Box::new([0x40, 0x01]);
    let ptr = storage.as_ptr() as i64;
    let descr = crate::descr::make_array_descr(0, 1, None, majit_ir::Type::Int, false);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr];

    let mut tc = fresh_trace_ctx();
    tc.set_cpu(Some(&cpu));
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i = distinct_const_ints(&mut tc, 8);
    regs_r[2] = tc.const_ref(ptr);
    regs_i[3] = tc.const_int(0);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        pending_guard_snapshot_error: None,
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_reorder_ceiling: u32::MAX,
        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("headerless getarrayitem");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 6);
    let dst = wc.registers_i.get(5).expect("int register in range");
    assert!(
        !dst.is_constant(),
        "a non-pure getarrayitem records the load"
    );
    assert_eq!(
        wc.trace_ctx.concrete_of_opref(dst),
        Some(majit_ir::Value::Int(0x40))
    );
    assert_eq!(wc.trace_ctx.num_ops(), ops_before + 1);
    drop(wc);
    drop(storage);

    let mut tc = fresh_trace_ctx();
    tc.set_cpu(Some(&cpu));
    let storage: Box<[u8]> = Box::new([0x40, 0x01]);
    let ptr = storage.as_ptr() as i64;
    let descr = crate::descr::make_array_descr(0, 1, None, majit_ir::Type::Int, false);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr];
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i = distinct_const_ints(&mut tc, 8);
    regs_r[2] = tc.const_ref(ptr);
    let zero = tc.const_int(0);
    let idx = tc.record_op(majit_ir::OpCode::IntAdd, &[zero, zero]);
    tc.set_opref_concrete(idx, majit_ir::Value::Int(0));
    regs_i[3] = idx;
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        pending_guard_snapshot_error: None,
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_reorder_ceiling: u32::MAX,
        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    step(&code, 0, &mut wc).expect("non-const headerless getarrayitem");
    let dst = wc.registers_i.get(5).expect("int register in range");
    assert_eq!(
        wc.trace_ctx.concrete_of_opref(dst),
        None,
        "a non-constant index into a headerless array stays unstamped"
    );
    drop(wc);
    drop(storage);
}

#[test]
fn getarrayitem_gc_r_cache_hit_returns_cached_box() {
    // Pre-cache (array, index, descr) →
    // cached_box. Second invocation must return cached_box and
    // not record an IR op.
    let byte = *insns_opname_to_byte()
        .get("getarrayitem_gc_r/rid>r")
        .expect("`getarrayitem_gc_r/rid>r` must be in insns table");
    let code = [byte, 0x02, 0x03, 0x01, 0x00, 0x05];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let regs_i = distinct_const_ints(&mut tc, 8);
    let array = regs_r[2];
    let index = regs_i[3];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr];
    let frame_done = done_descr_ref_for_tests();
    let cached = tc.const_ref(0xCAFE_F00D);
    tc.heapcache_getarrayitem_now_known(array, index, 1, cached);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("getarrayitem_gc_r must dispatch");
    let dst_post = wc.registers_r.get(5).expect("ref register in range");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "cache hit must NOT record any new IR op",
    );
    assert_eq!(
        dst_post, cached,
        "cache hit must write cached OpRef into registers_r[dst]",
    );
}

#[test]
fn setarrayitem_gc_r_records_setarrayitemgc_with_three_args() {
    // `setarrayitem_gc_r/rird` records
    // SetarrayitemGc with [array, index, value] + descr and
    // updates the heapcache via setarrayitem.
    let byte = *insns_opname_to_byte()
        .get("setarrayitem_gc_r/rird")
        .expect("`setarrayitem_gc_r/rird` must be in insns table");
    // Operand layout `rird`: 1B r-reg(2) + 1B i-reg(4) +
    // 1B r-reg(6) + 2B descr(LE 1).
    let code = [byte, 0x02, 0x04, 0x06, 0x01, 0x00];
    let mut tc = fresh_trace_ctx();
    let regs_r = distinct_const_refs(&mut tc, 8);
    let regs_i = distinct_const_ints(&mut tc, 8);
    let array = regs_r[2];
    let index = regs_i[4];
    let value = regs_r[6];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("setarrayitem_gc_r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 6,
        "setarrayitem_gc_r/rird operand layout = 5 bytes"
    );
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::SetarrayitemGc);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![array, index, value],
        "SetarrayitemGc args must be [array, index, value]",
    );
    assert!(std::sync::Arc::ptr_eq(
        &last.getdescr().expect("must carry array descr"),
        &descr,
    ));
    // Heapcache must reflect the write.  Box identity-only check —
    // value payload is Void in walker-emitted writes.
    assert_eq!(
        tc.heapcache_getarrayitem(array, index, 1).map(|b| b),
        Some(value),
        "post-setarrayitem, heapcache must reflect the written value",
    );
}

#[test]
fn dispatch_via_miframe_runs_ref_return_through_real_miframe_state() {
    // Acceptance: the bridge function takes a
    // real `MIFrame` (constructed via the same `PyreSym::new_uninit`
    // + `MIFrame { ctx, sym, .. }` shape that `state.rs`'s
    // existing tests use), pre-populates `sym.registers_r[2]` with
    // a known OpRef, then walks `ref_return r2`. Walker must
    // record `Finish([sym.registers_r[2]], descr=done_with_this_frame_descr_ref)`
    // through the *same* TraceCtx the MIFrame's `ctx` pointer
    // owns — i.e., production-shape state plumbing, no separate
    // test fixture.
    use crate::state::PyreSym;

    let mut tc = TraceCtx::for_test_types(&[majit_ir::Type::Ref]);
    let expected_arg = tc.const_ref(0xCAFE_F00D);
    let mut sym = PyreSym::new_uninit(OpRef::NONE);
    sym.registers_r_mut().replace(vec![OpRef::NONE; 8]);
    sym.registers_r_mut().set(2, expected_arg);

    let miframe = MIFrame {
        ctx: &mut tc,
        sym: &mut sym,
        fallthrough_pc: 0,
        pending_result_stack_idx: None,
        pending_result_type: None,
        pending_inline_frame: None,
        residual_call_pc: None,
        loop_close_marker_jit_pc: None,
        close_merge_point_vsd: None,
        orgpc: 0,
        concrete_frame_addr: 0,
        pre_opcode_registers_r: None,
        pre_opcode_semantic_depth: None,
    };

    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let code = [ret_byte, 0x02];
    let descr = make_fail_descr(1);
    // PyPy `setup_call(argboxes)` analog: stamp `expected_arg` at
    // `R[2]_r` so the `ref_return r2` walker handler picks it up
    // from the fresh top-level register file.  Slots 0/1 stay
    // `OpRef::NONE` since this fixture exercises only slot 2.
    let argboxes_r = [OpRef::NONE, OpRef::NONE, expected_arg];
    fbw_finish_payload_reset();
    let session = std::cell::RefCell::new(WalkSession::default());
    let (outcome, end_pc) = dispatch_via_miframe(
        unsafe { &mut *miframe.ctx },
        unsafe { &mut *miframe.sym },
        miframe.concrete_frame_addr,
        miframe.orgpc,
        &session,
        &code,
        0,
        &[],
        RawDescrPool::Global,
        false,
        &no_sub_jitcodes,
        true,
        8,
        0,
        0,
        &[],
        &[],
        &[],
        &argboxes_r,
        &[],
        &[],
    )
    .expect("dispatch_via_miframe must succeed for ref_return r2");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
    assert_eq!(end_pc, 2);

    // Drop miframe so we can inspect tc directly.
    drop(miframe);
    assert_eq!(
        finish_payload_of(&outcome),
        Some((expected_arg, Type::Ref)),
        "finish payload must be sym.registers_r[2] threaded through the MIFrame bridge",
    );
}

#[test]
fn dispatch_via_miframe_mirrors_last_exc_value_back_into_sym() {
    // When the walker's last_exc_value field
    // changes (raise/r sets it before terminating), the bridge
    // function must mirror it back to `sym.last_exc_box`. RPython
    // parity: `metainterp.last_exc_value = ...` is metainterp-level
    // state that survives across opimpl invocations.
    use crate::state::PyreSym;
    use pyre_object::interp_exceptions::{ExcKind, w_exception_new};

    let mut tc = TraceCtx::for_test_types(&[majit_ir::Type::Ref]);
    let exc = w_exception_new(ExcKind::ValueError, "boom");
    // The walk now reads the concrete exception's traceback head.
    let exc_oprep = tc.const_ref(exc as i64);
    let mut sym = PyreSym::new_uninit(OpRef::NONE);
    sym.registers_r_mut().replace(vec![OpRef::NONE; 8]);
    sym.registers_r_mut().set(3, exc_oprep);
    // Pre-condition: sym.last_exc_box is unset.
    assert!(sym.last_exc_box().is_none());

    let miframe = MIFrame {
        ctx: &mut tc,
        sym: &mut sym,
        fallthrough_pc: 0,
        pending_result_stack_idx: None,
        pending_result_type: None,
        pending_inline_frame: None,
        residual_call_pc: None,
        loop_close_marker_jit_pc: None,
        close_merge_point_vsd: None,
        orgpc: 0,
        concrete_frame_addr: 0,
        pre_opcode_registers_r: None,
        pre_opcode_semantic_depth: None,
    };

    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let code = [raise_byte, 0x03];
    let descr_done = make_fail_descr(1);
    let descr_exc = make_fail_descr(99);
    // Setup_call argbox at R[3]_r: `raise/r` reads its exc operand
    // from this slot in the fresh top-level register file.
    let argboxes_r = [OpRef::NONE, OpRef::NONE, OpRef::NONE, exc_oprep];
    let session = std::cell::RefCell::new(WalkSession::default());
    let (outcome, _) = dispatch_via_miframe(
        unsafe { &mut *miframe.ctx },
        unsafe { &mut *miframe.sym },
        miframe.concrete_frame_addr,
        miframe.orgpc,
        &session,
        &code,
        0,
        &[],
        RawDescrPool::Global,
        false,
        &no_sub_jitcodes,
        true,
        8,
        0,
        0,
        &[],
        &[],
        &[],
        &argboxes_r,
        &[],
        &[],
    )
    .expect("dispatch_via_miframe must succeed for raise r3");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
    drop(miframe);
    // Post-condition: sym.last_exc_box was mirrored from the
    // walker's last_exc_value (set by raise/r before terminate).
    assert_eq!(
        sym.last_exc_box(),
        exc_oprep,
        "sym.last_exc_box must mirror the exc OpRef the walker captured \
             via WalkContext::last_exc_value",
    );
    // Post-condition: dispatch_via_miframe also sets
    // sym.class_of_last_exc_is_const to mirror RPython's
    // `pyjitpl.py opimpl_raise: class_of_last_exc_is_const = True`.
    assert!(
        sym.class_of_last_exc_is_const(),
        "sym.class_of_last_exc_is_const must be true after a raise/r",
    );
}

#[test]
fn dispatch_via_miframe_bridge_inside_catch_answers_last_exc_value() {
    // A top-level guard-failure bridge whose entry is the `except E as name:`
    // cleanup tail starts at `last_exc_value/>r` with no standing exception
    // on the fresh MetaInterp. The walk has to seed that slot from the frame's
    // current exception (`sys_exc_value`) the way a handler-entry subwalk
    // already seeds `last_exc_value`, or the op aborts with
    // `LastExcValueWithoutActiveException` and the loop retraces.
    use crate::state::PyreSym;
    use pyre_object::interp_exceptions::{ExcKind, w_exception_new};

    let mut tc = TraceCtx::for_test_types(&[majit_ir::Type::Ref]);
    tc.is_bridge_trace = true;
    let exc = w_exception_new(ExcKind::KeyError, "e");
    let _ec = pyre_interpreter::call::ensure_executioncontext();
    let prev = pyre_interpreter::eval::get_current_exception();
    pyre_interpreter::eval::set_current_exception(exc);

    let mut sym = PyreSym::new_uninit(OpRef::NONE);
    assert!(
        sym.last_exc_box().is_none(),
        "the failing guard is not an exception guard, so last_exc_box is empty"
    );

    let last_exc_byte = *insns_opname_to_byte()
        .get("last_exc_value/>r")
        .expect("`last_exc_value/>r` must be in insns table");
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    // Handler-body entry: materialize last_exc_value into r0, then return it.
    let code = [last_exc_byte, 0x00, ret_byte, 0x00];
    fbw_finish_payload_reset();
    let session = std::cell::RefCell::new(WalkSession::default());
    let result = dispatch_via_miframe(
        &mut tc,
        &mut sym,
        0,
        0,
        &session,
        &code,
        0,
        &[],
        RawDescrPool::Global,
        false,
        &no_sub_jitcodes,
        true,
        8,
        0,
        0,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    );
    pyre_interpreter::eval::set_current_exception(prev);
    let (outcome, end_pc) =
        result.expect("catch-region bridge must answer last_exc_value/>r instead of aborting");
    assert!(matches!(outcome, DispatchOutcome::Terminate { .. }));
    assert_eq!(end_pc, code.len());
    let (finish_value, finish_ty) =
        finish_payload_of(&outcome).expect("ref_return must stash the seeded exception");
    assert_eq!(finish_ty, Type::Ref);
    assert_eq!(
        tc.box_value(finish_value),
        Some(majit_ir::Value::Ref(majit_ir::GcRef(exc as usize))),
        "last_exc_value/>r must write the frame's current exception, not a baked stand-in",
    );
    assert!(
        !sym.last_exc_box().is_none(),
        "the seed must land on last_exc_box so a follow-on reraise can read it",
    );
}

#[test]
fn dispatch_via_miframe_leaves_class_of_last_exc_is_const_unchanged_when_no_raise() {
    // When the walk does NOT raise (final last_exc remains None),
    // dispatch_via_miframe must NOT touch
    // sym.class_of_last_exc_is_const. The flag carries state from
    // a prior tracing step and must not be cleared by an unrelated
    // walk (e.g. a single ref_return-only top-level walk).
    use crate::state::PyreSym;

    let mut tc = TraceCtx::for_test_types(&[majit_ir::Type::Ref]);
    let value = tc.const_ref(0xC0FFEE);
    let mut sym = PyreSym::new_uninit(OpRef::NONE);
    sym.registers_r_mut().replace(vec![OpRef::NONE; 8]);
    sym.registers_r_mut().set(2, value);
    // Pre-condition: simulate prior raise — class_of_last_exc_is_const
    // is true and last_exc_box is set.
    sym.set_class_of_last_exc_is_const(true);
    sym.set_last_exc_box(value);

    let miframe = MIFrame {
        ctx: &mut tc,
        sym: &mut sym,
        fallthrough_pc: 0,
        pending_result_stack_idx: None,
        pending_result_type: None,
        pending_inline_frame: None,
        residual_call_pc: None,
        loop_close_marker_jit_pc: None,
        close_merge_point_vsd: None,
        orgpc: 0,
        concrete_frame_addr: 0,
        pre_opcode_registers_r: None,
        pre_opcode_semantic_depth: None,
    };
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let code = [ret_byte, 0x02];
    // Setup_call argbox at R[2]_r — the `ref_return r2` walker
    // handler picks it up from the fresh top-level register file.
    let argboxes_r = [OpRef::NONE, OpRef::NONE, value];
    let session = std::cell::RefCell::new(WalkSession::default());
    let _ = dispatch_via_miframe(
        unsafe { &mut *miframe.ctx },
        unsafe { &mut *miframe.sym },
        miframe.concrete_frame_addr,
        miframe.orgpc,
        &session,
        &code,
        0,
        &[],
        RawDescrPool::Global,
        false,
        &no_sub_jitcodes,
        true,
        8,
        0,
        0,
        &[],
        &[],
        &[],
        &argboxes_r,
        &[],
        &[],
    )
    .expect("ref_return walk must succeed");
    drop(miframe);
    // Walker preserved the carried-in class flag because no raise
    // happened during the walk.
    assert!(
        sym.class_of_last_exc_is_const(),
        "no-raise walk must not clear class_of_last_exc_is_const",
    );
}

#[test]
fn walk_undecodable_byte_surfaces_typed_error() {
    // 0xFF is unknown to the insns table (21 entries 0..=20 today).
    let code = [0xFFu8];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    assert_eq!(
        walk(&code, 0, &mut wc),
        Err(DispatchError::UndecodableOpcode { pc: 0 })
    );
}

/// `jit_merge_point/cIRFIRF` reached_loop_header behaviour: first
/// visit of a (green key, red shape) registers the merge point and
/// continues unrolling; the second visit with the same key + shape
/// closes the loop with the reds as jump args.  Mirrors
/// `pyjitpl.py` first-visit/found split.
#[test]
fn jit_merge_point_first_visit_continues_then_closes_loop() {
    let jmp_byte = *insns_opname_to_byte()
        .get("jit_merge_point/cIRFIRF")
        .expect("`jit_merge_point/cIRFIRF` must be in insns table");
    // cIRFIRF byte layout: jdindex(c) + greens(I gi, R gr, F gf) +
    // reds(I ri, R rr, F rf).  greens = portal jitdriver's
    // [next_instr(i0), pycode(r0)]; reds = [r1, r2] (loop-carried
    // refs, e.g. [frame, ec]).
    let code = [
        jmp_byte, 0x00, // c: jdindex
        0x01, 0x00, // gi: len=1, [i0 = next_instr]
        0x01, 0x00, // gr: len=1, [r0 = pycode]
        0x00, // gf: len=0
        0x00, // ri: len=0
        0x02, 0x01, 0x02, // rr: len=2, [r1, r2]
        0x00, // rf: len=0
    ];
    // Production `_compile_and_run_once` seeds
    // `current_merge_points = [(original_boxes, start)]`
    // (`pyjitpl.py` `_compile_and_run_once`); the first matching header visit closes.
    // Recorder inputargs are the two reds the header live-args carry.
    let green_key = crate::driver::make_green_key(0x1_0000 as *const (), 42, false);
    let mut tc = TraceCtx::for_test_types_with_green_key(&[Type::Ref, Type::Ref], green_key);
    tc.seed_compile_and_run_once_merge_point();
    assert!(
        tc.has_merge_point_with_shape_assert(green_key, 2),
        "seed must match header live-arg length so the first visit closes"
    );
    let next_instr = tc.const_int(42); // gi[0] = Python pc
    let pycode = tc.const_ref(0x1_0000); // gr[0] = PyCode ptr
    let red0 = tc.const_ref(0x2_0000); // rr[0]
    let red1 = tc.const_ref(0x3_0000); // rr[1]
    let regs_i = vec![next_instr];
    let regs_r = vec![pycode, red0, red1];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    // Arrival without a preceding `loop_header` stamp is a plain
    // pass-through (pyjitpl.py): nothing closes.
    let (gated, gated_next) = step(&code, 0, &mut wc).expect("gated jit_merge_point must dispatch");
    assert_eq!(gated, DispatchOutcome::Continue);
    assert_eq!(gated_next, code.len());

    // First stamped crossing: the production seed already holds this
    // key + red shape, so the visit closes (`pyjitpl.py` `_compile_and_run_once` /
    // `:3018-3022`). The reds here are constants, so
    // `remove_consts_and_duplicates` (pyjitpl.py) replaces each with a
    // freshly recorded `same_as` op before the close — the jump args
    // are runtime OpRefs wrapping the original const reds, not the consts.
    wc.trace_ctx.seen_loop_header_for_jdindex = 0;
    let (first, first_next) = step(&code, 0, &mut wc).expect("first jit_merge_point must dispatch");
    assert_eq!(first_next, code.len());
    assert_eq!(wc.trace_ctx.seen_loop_header_for_jdindex, -1);
    match first {
        DispatchOutcome::CloseLoop {
            jump_args,
            loop_header_pc,
            ..
        } => {
            assert_eq!(loop_header_pc, 42);
            assert_eq!(jump_args.len(), 2);
            for arg in &jump_args {
                assert!(!arg.is_constant(), "const red must be same_as-wrapped");
                assert_eq!(wc.trace_ctx.get_opref_type(*arg), Some(majit_ir::Type::Ref));
            }
        }
        other => panic!("expected CloseLoop from production seed, got {other:?}"),
    }
}

/// The walker JUMP used to skip a missing `virtualizable_box_at` slot, so
/// `live_arg_boxes` was one shorter than the LABEL registered from
/// `inputarg_types`. Every declared slot is now a box.
#[test]
fn append_virtualizable_boxes_includes_every_declared_slot() {
    let mut tc = TraceCtx::for_test_types(&[Type::Ref]);
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    let array_len = 3;
    let slot_count = info.num_static_extra_boxes + array_len;
    let null = Value::Ref(majit_ir::GcRef::NULL);
    let vable = tc.const_ref(1);
    let initial_boxes = vec![tc.const_null(); slot_count];
    let initial_values = vec![null; slot_count];
    tc.install_virtualizable_info(info.clone());
    tc.init_virtualizable_boxes(
        &info,
        vable,
        Value::Ref(majit_ir::GcRef(1)),
        &initial_boxes,
        &initial_values,
        &[array_len],
    );
    let red0 = tc.const_ref(0x10);
    let red1 = tc.const_ref(0x20);
    let live = super::append_virtualizable_boxes(&tc, vec![red0, red1]);
    assert_eq!(
        live.len(),
        2 + slot_count,
        "JUMP reds + virtualizable_boxes[:-1] must match the declared layout"
    );
    assert_eq!(live[0], red0);
    assert_eq!(live[1], red1);
    assert_eq!(&live[2..], initial_boxes.as_slice());
}

/// `loop_header/i` stamps `seen_loop_header_for_jdindex` from its
/// int-constant operand and records nothing (pyjitpl.py).
#[test]
fn loop_header_stamps_seen_flag() {
    let lh_byte = *insns_opname_to_byte()
        .get("loop_header/i")
        .expect("`loop_header/i` must be in insns table");
    let code = [lh_byte, 0x00]; // i: register slot 0 holds the jdindex
    let mut tc = fresh_trace_ctx();
    let jdindex = tc.const_int(0);
    let regs_i = vec![jdindex];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    assert_eq!(wc.trace_ctx.seen_loop_header_for_jdindex, -1);
    let (outcome, next) = step(&code, 0, &mut wc).expect("loop_header must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next, code.len());
    assert_eq!(wc.trace_ctx.seen_loop_header_for_jdindex, 0);
    assert_eq!(wc.trace_ctx.seen_loop_header_jit_pc, Some(0));
    assert_eq!(wc.trace_ctx.num_ops(), 0, "loop_header records nothing");
}

#[test]
fn jit_merge_point_int_form_resolves_jdindex_from_the_int_bank() {
    let byte = *insns_opname_to_byte()
        .get("jit_merge_point/iIRFIRF")
        .expect("`jit_merge_point/iIRFIRF` must be in insns table");
    let code = [
        byte, 0x02, // jdindex is in i2, not the literal value 2
        0x01, 0x00, // gi: [i0]
        0x01, 0x00, // gr: [r0]
        0x00, // gf
        0x00, // ri
        0x00, // rr
        0x00, // rf
    ];
    let pycode_ptr = 0x1_0000usize;
    let green_key = crate::driver::make_green_key(pycode_ptr as *const (), 42, false);
    // Header live args are the reds only (none on this form). Seed the
    // same empty shape production `_compile_and_run_once` seeds
    // (`pyjitpl.py` `_compile_and_run_once`).
    let mut tc = TraceCtx::for_test_types_with_green_key(&[], green_key);
    tc.seed_compile_and_run_once_merge_point();
    assert!(
        tc.has_merge_point_with_shape_assert(green_key, 0),
        "seed must match header live-arg length so the first visit closes"
    );
    let next_instr = tc.const_int(42);
    let unused = tc.const_int(99);
    let jdindex = tc.const_int(0);
    let pycode = tc.const_ref(pycode_ptr as i64);
    let regs_i = [next_instr, unused, jdindex];
    let regs_r = [pycode];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.trace_ctx.seen_loop_header_for_jdindex = 0;
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("int-form merge point must dispatch");
    assert_eq!(next_pc, code.len());
    assert_eq!(wc.trace_ctx.seen_loop_header_for_jdindex, -1);
    match outcome {
        DispatchOutcome::CloseLoop {
            jump_args,
            loop_header_pc,
            ..
        } => {
            assert_eq!(loop_header_pc, 42);
            assert_eq!(jump_args.len(), 0);
        }
        other => {
            panic!("expected CloseLoop after resolving jdindex from the int bank, got {other:?}")
        }
    }
}

/// A green list with no concrete leading element cannot form the
/// green key — surfaces `JitMergePointGreenKeyUnresolved` rather than
/// guessing.
#[test]
fn jit_merge_point_unresolved_green_key_fails_loud() {
    let jmp_byte = *insns_opname_to_byte()
        .get("jit_merge_point/cIRFIRF")
        .expect("`jit_merge_point/cIRFIRF` must be in insns table");
    let code = [
        jmp_byte, 0x00, // c
        0x01, 0x00, // gi: len=1, [i0]
        0x01, 0x00, // gr: len=1, [r0]
        0x00, // gf
        0x00, // ri
        0x00, // rr
        0x00, // rf
    ];
    let mut tc = fresh_trace_ctx();
    // i0 is a non-constant input arg → no concrete next_instr.
    let regs_i = vec![OpRef::input_arg_int(0)];
    let regs_r = vec![tc.const_ref(0x1_0000)];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    assert_eq!(
        step(&code, 0, &mut wc),
        Err(DispatchError::JitMergePointGreenKeyUnresolved { pc: 0 })
    );
}

#[test]
fn structural_midbody_anchor_requires_exact_floor_segment_start() {
    let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
    pyjit.metadata.py_floor_by_jit_pc = vec![(0, 0), (101, 28), (115, 29)];
    assert!(super::exact_floor_segment_anchor(&pyjit.metadata, 28, 101));
    assert!(!super::exact_floor_segment_anchor(&pyjit.metadata, 28, 102));
    assert!(!super::exact_floor_segment_anchor(&pyjit.metadata, 27, 101));
}

#[test]
fn portal_vable_bookkeeping_anchor_accepts_only_same_pc_same_frame_vable_writes() {
    use majit_jitcode::jitcode::BhDescr;
    use majit_metainterp::jitcode::RuntimeBhDescr;

    let setfield = *insns_opname_to_byte()
        .get("setfield_vable_i/rid")
        .expect("setfield_vable_i must exist");
    let abort = *insns_opname_to_byte()
        .get("abort_permanent/")
        .expect("abort_permanent must exist");
    let int_copy = *insns_opname_to_byte()
        .get("int_copy/c>i")
        .expect("int_copy/c>i must exist");
    let descrs = [RuntimeBhDescr::Descr(Box::new(BhDescr::VableField {
        index: 0,
    }))];
    let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
    pyjit.metadata.py_floor_by_jit_pc = vec![(0, 29)];
    let portal_gap = [setfield, 1, 7, 0, 0, abort];

    assert!(super::portal_vable_bookkeeping_anchor(
        &pyjit.metadata,
        true,
        1,
        &descrs,
        &portal_gap,
        29,
        5,
        |_| 29,
    ));
    assert!(!super::portal_vable_bookkeeping_anchor(
        &pyjit.metadata,
        true,
        1,
        &descrs,
        &portal_gap,
        29,
        5,
        |_| 28,
    ));

    let computation_gap = [int_copy, 1, 7, abort];
    assert!(!super::portal_vable_bookkeeping_anchor(
        &pyjit.metadata,
        true,
        1,
        &descrs,
        &computation_gap,
        29,
        3,
        |_| 29,
    ));

    pyjit.metadata.py_floor_by_jit_pc = vec![(0, 0), (5, 29)];
    assert!(super::portal_vable_bookkeeping_anchor(
        &pyjit.metadata,
        false,
        u16::MAX,
        &[],
        &portal_gap,
        29,
        5,
        |_| 29,
    ));
}

#[test]
fn portal_vable_bookkeeping_anchor_accepts_registered_entry_marker_prefix() {
    use majit_jitcode::jitcode::BhDescr;
    use majit_metainterp::jitcode::RuntimeBhDescr;

    let setfield = *insns_opname_to_byte()
        .get("setfield_vable_i/rid")
        .expect("setfield_vable_i must exist");
    let abort = *insns_opname_to_byte()
        .get("abort_permanent/")
        .expect("abort_permanent must exist");
    let descrs = [RuntimeBhDescr::Descr(Box::new(BhDescr::VableField {
        index: 2,
    }))];
    let code = [setfield, 1, 7, 0, 0, abort];
    let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
    // The loop-header marker at jit pc 5 names Python pc 29, while its
    // bookkeeping prefix remains in the preceding floor segment (pc 28).
    pyjit.metadata.py_floor_by_jit_pc = vec![(0, 28)];
    pyjit.metadata.merge_entry_by_green = vec![(29, 5)];

    assert!(super::portal_vable_bookkeeping_anchor(
        &pyjit.metadata,
        true,
        1,
        &descrs,
        &code,
        29,
        5,
        |_| 28,
    ));

    // The floor mismatch is admitted only at the exact registered entry.
    pyjit.metadata.merge_entry_by_green = vec![(29, 4)];
    assert!(!super::portal_vable_bookkeeping_anchor(
        &pyjit.metadata,
        true,
        1,
        &descrs,
        &code,
        29,
        5,
        |_| 28,
    ));
}

/// The operand-stack spill the walker emits ahead of an unwalkable opcode is
/// `setarrayitem_vable_r` + `setfield_vable_i(valuestackdepth)`, not the
/// `last_instr` write alone; the rebuild rewrites both, so both are admitted.
/// `getarrayitem_vable_r` is not — it writes a jitcode register the payload
/// sources live values from.
#[test]
fn portal_vable_bookkeeping_anchor_admits_this_frames_spills_and_reloads_only() {
    use majit_jitcode::jitcode::BhDescr;
    use majit_metainterp::jitcode::RuntimeBhDescr;

    let setarrayitem = *insns_opname_to_byte()
        .get("setarrayitem_vable_r/rirdd")
        .expect("setarrayitem_vable_r must exist");
    let getarrayitem = *insns_opname_to_byte()
        .get("getarrayitem_vable_r/ridd>r")
        .expect("getarrayitem_vable_r must exist");
    let setfield = *insns_opname_to_byte()
        .get("setfield_vable_i/rid")
        .expect("setfield_vable_i must exist");
    let abort = *insns_opname_to_byte()
        .get("abort_permanent/")
        .expect("abort_permanent must exist");
    // `valuestackdepth` is static field index 2, not 0.
    let descrs = [RuntimeBhDescr::Descr(Box::new(BhDescr::VableField {
        index: 2,
    }))];
    let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
    pyjit.metadata.py_floor_by_jit_pc = vec![(0, 29)];

    let mut spill = vec![setarrayitem];
    let setarrayitem_len =
        super::super::jitcode_runtime::decode_op_at(&[setarrayitem, 1, 2, 3, 0, 0, 0, 0], 0)
            .expect("setarrayitem_vable_r must decode")
            .next_pc;
    spill.extend(std::iter::repeat_n(1u8, setarrayitem_len - 1));
    let spill_end = spill.len();
    spill.extend_from_slice(&[setfield, 1, 7, 0, 0]);
    let abort_pc = spill.len();
    spill.push(abort);

    assert!(super::portal_vable_bookkeeping_anchor(
        &pyjit.metadata,
        true,
        1,
        &descrs,
        &spill,
        29,
        abort_pc,
        |_| 29,
    ));
    // A `setfield_vable_i` whose descr is not a vable field is not bookkeeping.
    let mut foreign_descr = spill.clone();
    foreign_descr[spill_end + 3] = 1;
    assert!(!super::portal_vable_bookkeeping_anchor(
        &pyjit.metadata,
        true,
        1,
        &descrs,
        &foreign_descr,
        29,
        abort_pc,
        |_| 29,
    ));
    // A spill aimed at another frame register is not this frame's bookkeeping.
    let mut foreign_frame = spill.clone();
    foreign_frame[1] = 2;
    assert!(!super::portal_vable_bookkeeping_anchor(
        &pyjit.metadata,
        true,
        1,
        &descrs,
        &foreign_frame,
        29,
        abort_pc,
        |_| 29,
    ));

    // A pure vable reload writes no frame state, only a jitcode register, and
    // the rebuild resumes the plain interpreter without a register bank — so it
    // is admitted for a weaker reason than the spill stores above.
    let mut reload = vec![getarrayitem];
    let getarrayitem_len =
        super::super::jitcode_runtime::decode_op_at(&[getarrayitem, 1, 2, 0, 0, 0, 0, 3], 0)
            .expect("getarrayitem_vable_r must decode")
            .next_pc;
    reload.extend(std::iter::repeat_n(1u8, getarrayitem_len - 1));
    let reload_abort_pc = reload.len();
    reload.push(abort);
    assert!(super::portal_vable_bookkeeping_anchor(
        &pyjit.metadata,
        true,
        1,
        &descrs,
        &reload,
        29,
        reload_abort_pc,
        |_| 29,
    ));
    // ...but only for this frame: a reload aimed at another frame register is
    // not this frame's bookkeeping.
    let mut foreign_reload = reload.clone();
    foreign_reload[1] = 2;
    assert!(!super::portal_vable_bookkeeping_anchor(
        &pyjit.metadata,
        true,
        1,
        &descrs,
        &foreign_reload,
        29,
        reload_abort_pc,
        |_| 29,
    ));
}

#[test]
fn callee_vable_ref_gates_on_frame_register_identity() {
    use majit_ir::{GcRef, Value};

    let ref_value = Value::Ref(GcRef(0x1000));
    let mut shadow = super::CalleeLocalsShadow::default();
    shadow.set_concrete(1, 3, ref_value);

    // An entry recorded through frame reg 1 resolves for that frame only.
    assert!(matches!(
        super::callee_vable_ref_at(Some(&shadow), 1, 3),
        Some(ConcreteValue::Ref(value)) if value as usize == 0x1000
    ));
    // A foreign frame register declines (no strict-fold witness either).
    assert_eq!(super::callee_vable_ref_at(Some(&shadow), 2, 3), None);

    // The strict-fold witness admits the slot even when the recorded writer
    // frame differs from the queried frame.
    shadow.fold_frame_reg = 2;
    assert!(matches!(
        super::callee_vable_ref_at(Some(&shadow), 2, 3),
        Some(ConcreteValue::Ref(value)) if value as usize == 0x1000
    ));

    // A missing slot, the `u16::MAX` sentinel, and an absent shadow decline.
    assert_eq!(super::callee_vable_ref_at(Some(&shadow), 1, 4), None);
    assert_eq!(super::callee_vable_ref_at(Some(&shadow), u16::MAX, 3), None);
    assert_eq!(super::callee_vable_ref_at(None, 1, 3), None);
}

#[test]
fn callee_locals_region_publishes_only_a_reproducible_frame() {
    use majit_ir::{GcRef, Value};

    // A slot this walk never wrote still holds what `write_boxes` would have
    // written back, so an all-absent shadow publishes.
    let mut shadow = super::CalleeLocalsShadow::default();
    assert!(super::callee_locals_region_is_publishable(&shadow, 4, 1));

    shadow.set_concrete(1, 2, Value::Ref(GcRef(0x1000)));
    assert!(super::callee_locals_region_is_publishable(&shadow, 4, 1));

    // A slot recorded through another frame register does not describe this
    // frame.
    assert!(!super::callee_locals_region_is_publishable(&shadow, 4, 2));

    // A store whose concrete half is unknown leaves `concrete` cleared while
    // `opref` still witnesses the write: the frame's value is stale.
    shadow.set_opref(3, OpRef::ref_op(7));
    shadow.set_concrete(1, 3, Value::Void);
    assert!(!super::callee_locals_region_is_publishable(&shadow, 4, 1));
    // Only within the locals region — an operand-stack slot is not flushed.
    assert!(super::callee_locals_region_is_publishable(&shadow, 3, 1));

    // The `NO_CONCRETE` sentinel is not a storable reference either.
    let mut sentinel = super::CalleeLocalsShadow::default();
    sentinel.set_concrete(1, 0, Value::Ref(GcRef::NO_CONCRETE));
    assert!(!super::callee_locals_region_is_publishable(&sentinel, 1, 1));
}

#[test]
fn dense_fallthrough_seeds_only_its_missing_result_color() {
    let keep = OpRef::ref_op(3);
    let null_ref = OpRef::const_ptr(majit_ir::GcRef(0));
    let mut registers = vec![keep, OpRef::NONE, OpRef::NONE];

    assert!(super::resume_snapshot::seed_missing_fallthrough_result(
        &mut registers,
        Some(1),
        null_ref,
    ));
    assert_eq!(registers, vec![keep, null_ref, OpRef::NONE]);

    // A defined register, a sentinel, and an out-of-range color are not
    // rewritten. Other NONE holes remain visible to the ordinary decline.
    assert!(!super::resume_snapshot::seed_missing_fallthrough_result(
        &mut registers,
        Some(0),
        null_ref,
    ));
    assert!(!super::resume_snapshot::seed_missing_fallthrough_result(
        &mut registers,
        Some(u16::MAX),
        null_ref,
    ));
    assert!(!super::resume_snapshot::seed_missing_fallthrough_result(
        &mut registers,
        Some(99),
        null_ref,
    ));
    assert_eq!(registers, vec![keep, null_ref, OpRef::NONE]);
}

/// `insert_renamings` routes a cyclic parallel move through the Int scratch
/// (`int_push/i` then `int_pop/>i`).  The pop must land the source's concrete
/// shadow in `concrete_registers_i[dst]`; leaving the destination slot's old
/// shadow behind makes every later `read_int_reg_concrete` on that register
/// report the value the move overwrote.
#[test]
fn int_scratch_move_carries_the_concrete_shadow_to_the_destination() {
    let push_byte = *insns_opname_to_byte()
        .get("int_push/i")
        .expect("int_push must be in the runtime instruction table");
    let pop_byte = *insns_opname_to_byte()
        .get("int_pop/>i")
        .expect("int_pop must be in the runtime instruction table");
    // Push r0, pop into r1: r1's pre-existing shadow must not survive.
    let code = [push_byte, 0, pop_byte, 1];
    let mut tc = TraceCtx::for_test_types(&[Type::Int, Type::Int]);
    let src = OpRef::input_arg_int(0);
    let dst_before = OpRef::input_arg_int(1);
    tc.set_opref_concrete(src, Value::Int(7));
    tc.set_opref_concrete(dst_before, Value::Int(99));
    let regs_i = vec![src, dst_before];
    let mut concrete_i = vec![ConcreteValue::Int(7), ConcreteValue::Int(99)];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut concrete_i,
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("`int_push/i` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 2);
    let (outcome, next_pc) = step(&code, next_pc, &mut wc).expect("`int_pop/>i` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 4);
    assert_eq!(
        wc.registers_i.get(1).expect("int register in range"),
        src,
        "the pop moves the source OpRef"
    );
    assert_eq!(
        wc.concrete_registers_i[1],
        ConcreteValue::Int(7),
        "the pop must overwrite the destination's stale concrete shadow",
    );
}

/// `FASTPATHS_SAME_BOXES` membership, on the non-fused (macro-table) arms.
/// The generated loop that carries `if b1 is b2: return <const>` spells only
/// the six signed int compares plus the four ref compares; the unsigned
/// compares live in the other loop and must still record their op.
#[test]
fn same_box_fastpath_covers_exactly_the_generated_fastpath_list() {
    // (opname, expected folded value) for every FASTPATHS_SAME_BOXES member
    // reachable through `binop_int_record` / `binop_ref_to_int_record`.
    let int_cases: &[(&str, i64)] = &[
        ("int_eq/ii>i", 1),
        ("int_le/ii>i", 1),
        ("int_ge/ii>i", 1),
        ("int_ne/ii>i", 0),
        ("int_lt/ii>i", 0),
        ("int_gt/ii>i", 0),
    ];
    for (opname, expected) in int_cases {
        let byte = *insns_opname_to_byte()
            .get(*opname)
            .unwrap_or_else(|| panic!("{opname} must be in the runtime instruction table"));
        // Same register for both operands: `b1 is b2`.
        let code = [byte, 0, 0, 1];
        let mut tc = TraceCtx::for_test_types(&[Type::Int]);
        let same = OpRef::input_arg_int(0);
        let mut regs_i = vec![same, OpRef::NONE];
        let (outcome, _) = run_hint_step(&code, &mut tc, &mut [], &mut [], &mut regs_i)
            .unwrap_or_else(|_| panic!("`{opname}` must dispatch"));
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert!(
            tc.ops().is_empty(),
            "`{opname}` on identical boxes must answer without recording an op",
        );
        assert_eq!(
            tc.constant_value(regs_i[1]),
            Some(*expected),
            "`{opname}` on identical boxes must fold to {expected}",
        );
    }
    // The unsigned compares share the `ii>i` shape but are NOT fast-path
    // members: they must still record.
    for opname in [
        "uint_lt/ii>i",
        "uint_le/ii>i",
        "uint_gt/ii>i",
        "uint_ge/ii>i",
    ] {
        let byte = *insns_opname_to_byte()
            .get(opname)
            .unwrap_or_else(|| panic!("{opname} must be in the runtime instruction table"));
        let code = [byte, 0, 0, 1];
        let mut tc = TraceCtx::for_test_types(&[Type::Int]);
        let same = OpRef::input_arg_int(0);
        let mut regs_i = vec![same, OpRef::NONE];
        let (outcome, _) = run_hint_step(&code, &mut tc, &mut [], &mut [], &mut regs_i)
            .unwrap_or_else(|_| panic!("`{opname}` must dispatch"));
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            tc.ops().len(),
            1,
            "`{opname}` is not a same-boxes fast-path member and must record",
        );
    }
}

/// The four ref compares are all `FASTPATHS_SAME_BOXES` members, including
/// `instance_ptr_eq` / `instance_ptr_ne`.
#[test]
fn ref_compare_same_box_fastpath_covers_the_instance_ptr_spellings() {
    let cases: &[(&str, i64)] = &[
        ("ptr_eq/rr>i", 1),
        ("ptr_ne/rr>i", 0),
        ("instance_ptr_eq/rr>i", 1),
        ("instance_ptr_ne/rr>i", 0),
    ];
    for (opname, expected) in cases {
        let byte = *insns_opname_to_byte()
            .get(*opname)
            .unwrap_or_else(|| panic!("{opname} must be in the runtime instruction table"));
        let code = [byte, 0, 0, 0];
        let mut tc = TraceCtx::for_test_types(&[Type::Ref]);
        let same = OpRef::input_arg_ref(0);
        let mut regs_r = vec![same];
        let mut concrete_r = vec![ConcreteValue::Null];
        let mut regs_i = vec![OpRef::NONE];
        let (outcome, _) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut regs_i)
            .unwrap_or_else(|_| panic!("`{opname}` must dispatch"));
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert!(
            tc.ops().is_empty(),
            "`{opname}` on identical boxes must answer without recording an op",
        );
        assert_eq!(
            tc.constant_value(regs_i[0]),
            Some(*expected),
            "`{opname}` on identical boxes must fold to {expected}",
        );
    }
}

// pyjitpl.py:2119-2121 concretely executes and stamps a Float `CALL_PURE`
// like any other.  The funcbox is the callee's own address on every live
// path — `add_fn_ptr(ptr)` is `add_call_target(ptr, ptr)` and the LLBC path
// bakes a plain `ConstInt` fnaddr — so `execute_pure_call` fetches the result
// from the floating-point return register (`executor.py cpu.bh_call_f`).

/// A real `f64`-returning callee with an integer parameter, i.e. the
/// `jit_bigint_to_f64_or_inf` shape.  The integer return register is undefined
/// after such a call, so reading it here read residue, not the result.
extern "C" fn halve_f64_for_walker_test(x: i64) -> f64 {
    (x as f64) / 2.0
}

#[test]
fn walker_folds_a_float_result_pure_call_from_the_float_return_register() {
    let mut tc = fresh_trace_ctx();
    let funcbox = tc.const_int(halve_f64_for_walker_test as *const () as i64);
    let arg0 = tc.const_int(7);
    let allboxes = [funcbox, arg0];
    let descr = make_call_descr(
        6,
        vec![Type::Int],
        Type::Float,
        majit_ir::ExtraEffect::ElidableCannotRaise,
    );
    let call_descr = descr.as_call_descr().expect("CallPureF descr");

    // Same shape as the Int sibling above, which DOES fold — so a failure here
    // is the float arm specifically, not the fixture.
    let int_descr = make_call_descr(
        7,
        vec![Type::Int, Type::Int],
        Type::Int,
        majit_ir::ExtraEffect::ElidableCannotRaise,
    );
    let int_boxes = [
        tc.const_int(add2_for_walker_test as *const () as i64),
        tc.const_int(40),
        tc.const_int(2),
    ];
    let int_call_descr = int_descr.as_call_descr().expect("CallPureI descr");

    let regs_i: Vec<OpRef> = Vec::new();
    let regs_r: Vec<OpRef> = Vec::new();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    // Record and fold one call at a time, as the dispatchers do: the fold
    // cuts back to the position taken before the record, so a second op
    // recorded in between would be cut away with it.
    let float_pos = wc.trace_ctx.get_trace_position();
    let recorded =
        wc.trace_ctx
            .record_op_with_descr(majit_ir::OpCode::CallPureF, &allboxes, descr.clone());
    let folded_f = try_fold_pure_call_via_executor(
        &mut wc,
        majit_ir::OpCode::CallPureF,
        &allboxes,
        call_descr,
        descr.clone(),
        float_pos,
        recorded,
    );
    let int_pos = wc.trace_ctx.get_trace_position();
    let int_recorded = wc.trace_ctx.record_op_with_descr(
        majit_ir::OpCode::CallPureI,
        &int_boxes,
        int_descr.clone(),
    );
    let folded_i = try_fold_pure_call_via_executor(
        &mut wc,
        majit_ir::OpCode::CallPureI,
        &int_boxes,
        int_call_descr,
        int_descr.clone(),
        int_pos,
        int_recorded,
    );
    drop(wc);

    // Every argbox is a `Const`, so `record_result_of_call_pure` takes the
    // `pyjitpl.py` arm: the call is cut back out and the result comes
    // back as the constant itself.
    assert_eq!(
        folded_i.inline_const_to_value(),
        Some(majit_ir::Value::Int(42)),
        "control: the Int sibling folds, so the fixture reaches the executor",
    );
    assert_eq!(
        folded_f.inline_const_to_value(),
        Some(majit_ir::Value::Float(3.5)),
        "halve_f64_for_walker_test(7) == 3.5; reading the integer return \
         register would stamp the argument 7 instead",
    );
}

/// `bh_load_global_fn`'s `namespace` operand (arg 0) is never dereferenced —
/// the helper resolves the globals from the executing frame or from `w_code`'s
/// own live `w_globals`. The codewriter bakes it from `w_code_get_w_globals`,
/// which is `PY_NULL` until some frame for that code object stamps it, so a
/// jitcode built before the callee's first frame carries a concrete-NULL there.
/// Aborting the walk on it is a false positive whose replay re-executes every
/// residual the walk already ran concretely.
#[test]
fn mayforce_null_ref_arg_exempts_the_unread_load_global_namespace() {
    fn descr_for(runtime_helper: majit_ir::RuntimeHelperKind) -> DescrRef {
        let mut effect = majit_ir::EffectInfo::default();
        effect.runtime_helper = runtime_helper;
        std::sync::Arc::new(majit_ir::SimpleCallDescr::new(
            9,
            vec![Type::Ref, Type::Ref, Type::Ref, Type::Int],
            Type::Ref,
            false,
            std::mem::size_of::<usize>(),
            effect,
        ))
    }

    let mut tc = fresh_trace_ctx();
    // funcbox, then `bh_load_global_fn(namespace=NULL, w_code, frame, namei)`.
    let allboxes = [
        tc.const_int(0x1234),
        tc.const_ref(0),
        tc.const_ref(0xC0DE_1000),
        tc.const_ref(0xC0DE_2000),
        tc.const_int(2),
    ];
    let load_global = descr_for(majit_ir::RuntimeHelperKind::LoadGlobal);
    let untagged = descr_for(majit_ir::RuntimeHelperKind::None);

    let regs_i: Vec<OpRef> = Vec::new();
    let regs_r: Vec<OpRef> = Vec::new();
    let session = std::cell::RefCell::new(WalkSession::default());
    let wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    assert_eq!(
        walker_abort_if_mayforce_null_ref_arg(
            majit_ir::OpCode::CallMayForceR,
            &allboxes,
            load_global.as_call_descr().expect("call descr"),
            &wc,
            186,
        ),
        Ok(()),
    );
    // Control: the same NULL in the same slot without the helper tag is still
    // the broken baked-NULL shape the guard exists for.
    assert!(matches!(
        walker_abort_if_mayforce_null_ref_arg(
            majit_ir::OpCode::CallMayForceR,
            &allboxes,
            untagged.as_call_descr().expect("call descr"),
            &wc,
            186,
        ),
        Err(DispatchError::MayForceNullRefArgUnsupported { pc: 186 }),
    ));
}

/// `bh_with_except_start_fn(exit_func, exit_self, val)`'s `exit_self` (arg 1)
/// is the receiver slot `LOAD_SPECIAL` pushes beside an already-bound
/// `__exit__`, so it is `PY_NULL` for the ordinary `with` shape and
/// `with_except_start_values` prepends it only when non-null. Aborting the walk
/// on it refuses every bridge out of a `with` handler: `while` + `with` +
/// `raise` failed one guard 1786 times and compiled 0 bridges.
#[test]
fn mayforce_null_ref_arg_exempts_the_with_except_start_receiver() {
    fn descr_for(runtime_helper: majit_ir::RuntimeHelperKind) -> DescrRef {
        let mut effect = majit_ir::EffectInfo::default();
        effect.runtime_helper = runtime_helper;
        std::sync::Arc::new(majit_ir::SimpleCallDescr::new(
            11,
            vec![Type::Ref, Type::Ref, Type::Ref],
            Type::Ref,
            false,
            std::mem::size_of::<usize>(),
            effect,
        ))
    }

    let mut tc = fresh_trace_ctx();
    // funcbox, then `bh_with_except_start_fn(exit_func, exit_self=NULL, val)`.
    let allboxes = [
        tc.const_int(0x5678),
        tc.const_ref(0xC0DE_3000),
        tc.const_ref(0),
        tc.const_ref(0xC0DE_4000),
    ];
    let with_except_start = descr_for(majit_ir::RuntimeHelperKind::WithExceptStart);
    let untagged = descr_for(majit_ir::RuntimeHelperKind::None);

    let regs_i: Vec<OpRef> = Vec::new();
    let regs_r: Vec<OpRef> = Vec::new();
    let session = std::cell::RefCell::new(WalkSession::default());
    let wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::new(regs_r.iter().copied()),
        registers_i: &RegisterBank::new(regs_i.iter().copied()),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    assert_eq!(
        walker_abort_if_mayforce_null_ref_arg(
            majit_ir::OpCode::CallMayForceR,
            &allboxes,
            with_except_start.as_call_descr().expect("call descr"),
            &wc,
            1612,
        ),
        Ok(()),
    );
    // Control: the same NULL in the same slot without the helper tag is still
    // the broken baked-NULL shape the guard exists for.
    assert!(matches!(
        walker_abort_if_mayforce_null_ref_arg(
            majit_ir::OpCode::CallMayForceR,
            &allboxes,
            untagged.as_call_descr().expect("call descr"),
            &wc,
            1612,
        ),
        Err(DispatchError::MayForceNullRefArgUnsupported { pc: 1612 }),
    ));
}

/// The checked-`PY_NULL` sentinel table is read by BOTH
/// `walker_abort_if_mayforce_null_ref_arg` (which aborts the walk) and
/// `try_execute_residual_call_via_executor` (which returns
/// `declined_symbolic`), and the two must agree — a gate-only exemption leaves
/// the executor dropping the recording iteration's call.  They share one
/// function now, so this pins the table itself rather than either caller: the
/// executor arm is otherwise unreachable from a unit test, since it runs the
/// residual concretely through the funcbox.
#[test]
fn the_mayforce_null_ref_sentinel_table_names_one_slot_per_helper() {
    use super::residual_call::mayforce_null_ref_arg_is_checked_sentinel as is_sentinel;
    use majit_ir::RuntimeHelperKind as K;

    // (helper, nargs, the exempt indices)
    let table: &[(K, usize, &[usize])] = &[
        (K::CallFn, 4, &[1]),
        (K::CallKw, 5, &[1]),
        (K::StoreDeref, 2, &[1]),
        (K::WithExceptStart, 3, &[1]),
        (K::CallFunctionEx, 4, &[1, 3]),
        (K::LoadGlobal, 4, &[0]),
        // `normalize_raise_varargs` puts `cause` last, whatever the arity.
        (K::RaiseVarargs, 3, &[2]),
        (K::RaiseVarargs, 2, &[1]),
        // An untagged residual whose funcbox names no registered leaf exempts
        // nothing — the shape the gate exists for.
        (K::None, 4, &[]),
    ];
    for &(helper, nargs, exempt) in table {
        for i in 0..nargs {
            assert_eq!(
                is_sentinel(helper, 0, i, nargs),
                exempt.contains(&i),
                "{helper:?} arg {i} of {nargs}"
            );
        }
    }

    // The registered-leaf leg: an untagged `dont_look_inside*` body carries no
    // `RuntimeHelperKind`, so its funcbox is the coordinate.
    // `dict_get_plain_applies(dict, key, tail)` checks `tail` with
    // `tail.is_null()`, and `__majit_wrap_dict_descr_get` pads the two-word
    // `d.get(k)` to three words with `PY_NULL` there.
    // The registry spells an entry either module-qualified or as the crate-root
    // re-export, so match on the leaf the table itself reads.
    let leaf_of = |path: &'static str| path.rsplit("::").next().unwrap_or(path);
    let (_, applies) = pyre_interpreter::jit_trace_fnaddrs()
        .into_iter()
        .find(|&(path, _)| leaf_of(path) == "dict_get_plain_applies")
        .expect("dict_get_plain_applies is a registered leaf");
    for i in 0..3 {
        assert_eq!(
            is_sentinel(K::None, applies, i, 3),
            i == 2,
            "dict_get_plain_applies arg {i} of 3"
        );
    }
    // That arity only: a longer call puts the caller's third word in `tail`.
    assert!(!is_sentinel(K::None, applies, 2, 4));
    // `dict_get_slow(dict, key, default, nargs)` pads the same slot, one word
    // wider, and forwards `default` only on its `nargs >= 3` arms.
    let (_, slow) = pyre_interpreter::jit_trace_fnaddrs()
        .into_iter()
        .find(|&(path, _)| leaf_of(path) == "dict_get_slow")
        .expect("dict_get_slow is a registered leaf");
    for i in 0..4 {
        assert_eq!(
            is_sentinel(K::None, slow, i, 4),
            i == 2,
            "dict_get_slow arg {i} of 4"
        );
    }
    // `jit_portal_call_3(callable, null_or_self, a0, a1, a2)`. The plain-call
    // shape leaves `null_or_self` as `PY_NULL`; only that word is checked.
    let (_, portal) = pyre_interpreter::jit_trace_fnaddrs()
        .into_iter()
        .find(|&(path, _)| leaf_of(path) == "jit_portal_call_3")
        .expect("jit_portal_call_3 is a registered leaf");
    for i in 0..5 {
        assert_eq!(
            is_sentinel(K::None, portal, i, 5),
            i == 1,
            "jit_portal_call_3 arg {i} of 5"
        );
    }
    assert!(!is_sentinel(K::None, portal, 1, 4));
    let portal_exempted: Vec<&str> = pyre_interpreter::jit_trace_fnaddrs()
        .into_iter()
        .filter(|&(_, addr)| is_sentinel(K::None, addr, 1, 5))
        .map(|(path, _)| leaf_of(path))
        .filter(|&leaf| leaf != "jit_portal_call_3")
        .collect();
    assert!(
        portal_exempted.is_empty(),
        "unexpected exempt leaves: {portal_exempted:?}"
    );
    // And that leaf only — every other published address still refuses the
    // slot, `dict_get_plain` included, which takes the same three words and
    // dereferences all of them.
    let exempted: Vec<&str> = pyre_interpreter::jit_trace_fnaddrs()
        .into_iter()
        .filter(|&(_, addr)| is_sentinel(K::None, addr, 2, 3))
        .map(|(path, _)| leaf_of(path))
        .filter(|&leaf| leaf != "dict_get_plain_applies")
        .collect();
    assert!(
        exempted.is_empty(),
        "unexpected exempt leaves: {exempted:?}"
    );
}

#[test]
fn traceback_journal_rollback_unwinds_every_walk_node() {
    // A recording walk attaches one node per frame it delivers the exception
    // through, so by the time the walk ends the exception carries SEVERAL
    // journaled nodes — the bridge handler entry that introduced the log only
    // ever pushed one. A non-commit exit has to splice all of them back out, or
    // the interpreter's replay records the same frames again and a raise that
    // re-uses one exception object hands out a doubled chain.
    use pyre_interpreter::pytraceback::{
        w_pytraceback_get_lasti, w_pytraceback_get_w_next, w_pytraceback_new,
    };
    use pyre_object::interp_exceptions::{
        ExcKind, w_exception_get_traceback, w_exception_new, w_exception_set_traceback,
    };

    // `frame` is only ever compared by identity here, and `w_code` only has to
    // be a chain-terminating slot, so both stay NULL — the splice walks
    // `w_next` and nothing dereferences either. `w_pytraceback_new` can
    // collect, so the exception is read from its root slot on each side of it.
    fn prepend(slot: usize, lasti: i64) {
        unsafe {
            let head = w_exception_get_traceback(pyre_object::gc_roots::shadow_stack_get(slot));
            let node =
                w_pytraceback_new(std::ptr::null_mut(), lasti, head, 1, pyre_object::PY_NULL);
            w_exception_set_traceback(pyre_object::gc_roots::shadow_stack_get(slot), node);
        }
    }

    fn chain(exc: pyre_object::PyObjectRef) -> Vec<i64> {
        let mut out = Vec::new();
        let mut node = unsafe { w_exception_get_traceback(exc) };
        while !node.is_null() {
            out.push(unsafe { w_pytraceback_get_lasti(node) });
            node = unsafe { w_pytraceback_get_w_next(node) };
        }
        out
    }

    super::fbw_store_journal_reset();
    let mut exc = w_exception_new(ExcKind::ValueError, "boom");
    // The node the raise itself already attached before the walk got here.
    {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(exc);
        prepend(slot, 1);
        exc = pyre_object::gc_roots::shadow_stack_get(slot);
    }
    assert_eq!(chain(exc), vec![1]);

    // Callee level first, catching level last — the order the walk records in,
    // and the reverse of the order the rollback has to undo.
    for lasti in [2i64, 3] {
        super::journaled_concrete_traceback_attach(&mut exc, |slot| prepend(slot, lasti));
    }
    assert_eq!(chain(exc), vec![3, 2, 1]);

    super::fbw_store_journal_rollback();
    assert_eq!(
        chain(exc),
        vec![1],
        "a non-commit walk must unwind EVERY node it attached, not just the last"
    );

    // Commit keeps them: a committed walk's nodes ARE this delivery's.
    for lasti in [4i64, 5] {
        super::journaled_concrete_traceback_attach(&mut exc, |slot| prepend(slot, lasti));
    }
    super::fbw_store_journal_commit();
    super::fbw_store_journal_rollback();
    assert_eq!(
        chain(exc),
        vec![5, 4, 1],
        "a committed walk keeps its nodes"
    );
}

#[test]
fn ref_var_list_offset_follows_the_argcodes_not_a_fixed_byte() {
    // The Ref-only residual shape puts its `R` list right after the funcptr
    // register, so offset 1 answers it; the mixed shape does not, because the
    // Int list between them is variable-width.  A reader that assumes 1 for
    // both takes the Int list's length byte as the Ref list's, and its
    // register indices into the Ref bank — a list of unrelated objects whose
    // LENGTH still looks plausible, which is what let the wrong operand stack
    // through the abort flush's depth check.
    let ref_only = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("residual_call_r_r/iRd>r must be in insns table");
    // opcode, i=funcptr, R: len=2 + 2 regs, d=0x0001, >r dst
    let code = [ref_only, 0x07, 0x02, 0x04, 0x05, 0x01, 0x00, 0x09];
    let op = crate::jitcode_runtime::decode_op_at(&code, 0).expect("must decode");
    assert_eq!(op.argcodes, "iRd>r");
    assert_eq!(ref_var_list_operand_offset(&code, &op), Some(1));

    let mixed = *insns_opname_to_byte()
        .get("residual_call_ir_r/iIRd>r")
        .expect("residual_call_ir_r/iIRd>r must be in insns table");
    // opcode, i=funcptr, I: len=3 + 3 regs, R: len=2 + 2 regs, d, >r dst
    let code = [
        mixed, 0x07, //  i
        0x03, 0x00, 0x01, 0x02, // I: len=3
        0x02, 0x04, 0x05, // R: len=2
        0x01, 0x00, // d
        0x09, // >r
    ];
    let op = crate::jitcode_runtime::decode_op_at(&code, 0).expect("must decode");
    assert_eq!(op.argcodes, "iIRd>r");
    assert_eq!(
        ref_var_list_operand_offset(&code, &op),
        Some(5),
        "the R list starts past the 4-byte I list, not at offset 1",
    );

    // An op with no Ref list at all has no offset to answer with, and the
    // caller must decline rather than read whatever sits at a guessed one.
    let int_only = *insns_opname_to_byte()
        .get("int_add/ii>i")
        .expect("int_add/ii>i must be in insns table");
    let code = [int_only, 0x01, 0x02, 0x03];
    let op = crate::jitcode_runtime::decode_op_at(&code, 0).expect("must decode");
    assert_eq!(ref_var_list_operand_offset(&code, &op), None);
}

#[test]
fn foriter_body_identity_names_the_jitcode_its_op_pc_indexes() {
    // The in-flight FOR_ITER coordinate pairs a `for_iter_next` residual's own
    // JitCode offset with a JitCode identity, and
    // `inflight_foriter_body_pc` resolves the pair through THAT jitcode's pc
    // tables.  Inside an inline sub-walk the offset is the callee's, so the
    // identity must be the callee's too — `fbw_mode.snapshot_sym` still names
    // the outer portal.
    //
    // A null `snapshot_sym` is the discriminating fixture: it is the one input
    // for which the portal arm can produce nothing at all, so a returned
    // identity can only have come from `inline_callee_consts`.  The callee arm
    // returns before the sym is read, so a live portal can never outrank it.
    let mut tc = fresh_trace_ctx();
    let descr_pool: Vec<DescrRef> = Vec::new();
    let session = std::cell::RefCell::new(WalkSession::default());
    let callee_index = 4242;
    let mut wc = WalkContext {
        frame_state: WalkFrameState::new(WalkFrameStateData {
            callee_shadow: None,
            concrete_registers_r: ([]).to_vec(),
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            ..Default::default()
        }),
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &RegisterBank::default(),
        registers_i: &RegisterBank::default(),
        registers_f: &RegisterBank::default(),

        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    assert!(
        wc.fbw_mode.snapshot_sym.is_null(),
        "fixture premise: no portal identity is available",
    );
    assert!(
        fbw_foriter_body_from_op_pc(&wc, 96).is_none(),
        "with neither a callee nor a portal there is no coordinate to record",
    );

    wc.inline_callee_consts = Some(InlineCalleeConsts {
        jitcode_index: callee_index,
    });
    wc.fbw_mode.inline_subwalk = true;
    match fbw_foriter_body_from_op_pc(&wc, 96) {
        Some(InflightForiterBody::Jit {
            jitcode_index,
            op_pc,
        }) => {
            assert_eq!(jitcode_index, callee_index, "identity must be the callee's");
            assert_eq!(op_pc, 96, "offset must be recorded verbatim");
        }
        other => panic!("expected the callee's Jit coordinate, got {other:?}"),
    }
}

#[test]
fn a_guard_after_the_subwalks_store_is_what_declines_the_pop_fold() {
    use super::specialize::subwalk_guard_follows_store;

    let obj = OpRef::input_arg_ref(0);
    let value = OpRef::ConstInt(7);

    // The shape `w_list_pop_end_inner`'s Integer arm records today: every guard
    // is ahead of `ll_list_int_set_len`, so a CALL-boundary resume re-executes
    // a pop that has not happened.
    let mut tc = TraceCtx::for_test_types(&[Type::Ref]);
    let start = tc.get_trace_position();
    tc.record_guard(majit_ir::OpCode::GuardNonnull, &[obj], 0);
    tc.record_op(majit_ir::OpCode::SetfieldGc, &[obj, value]);
    assert!(
        !subwalk_guard_follows_store(&tc, start),
        "guard-then-store is the sound order and must not decline"
    );

    // What removing the `w_int_new` short-circuit would produce: the boxing
    // body's own guards land after the length is already shrunk, so the same
    // resume pops a second time.
    let mut tc = TraceCtx::for_test_types(&[Type::Ref]);
    let start = tc.get_trace_position();
    tc.record_op(majit_ir::OpCode::SetfieldGc, &[obj, value]);
    tc.record_guard(majit_ir::OpCode::GuardNonnull, &[obj], 0);
    assert!(
        subwalk_guard_follows_store(&tc, start),
        "a guard past the body's first store is the double-apply shape"
    );

    // The window starts at the recorded position: the caller's own pre-descent
    // stores are not the body's, and a guard after them is not a finding.
    let mut tc = TraceCtx::for_test_types(&[Type::Ref]);
    tc.record_op(majit_ir::OpCode::SetfieldGc, &[obj, value]);
    let start = tc.get_trace_position();
    tc.record_guard(majit_ir::OpCode::GuardNonnull, &[obj], 0);
    assert!(
        !subwalk_guard_follows_store(&tc, start),
        "ops recorded before the sub-walk are outside the window"
    );

    // Array stores commit just as much as field stores.
    let mut tc = TraceCtx::for_test_types(&[Type::Ref]);
    let start = tc.get_trace_position();
    tc.record_op(majit_ir::OpCode::SetarrayitemGc, &[obj, value, value]);
    tc.record_guard(majit_ir::OpCode::GuardNonnull, &[obj], 0);
    assert!(
        subwalk_guard_follows_store(&tc, start),
        "SetarrayitemGc is a committed store too"
    );

    // A start past the end is a window that no longer exists — an empty read
    // there must not pass for "no guard after a store".
    let tc = TraceCtx::for_test_types(&[Type::Ref]);
    let mut past_end = tc.get_trace_position();
    past_end._count = (tc.num_inputargs() + tc.num_ops() + 1) as u32;
    assert!(
        subwalk_guard_follows_store(&tc, past_end),
        "an unreadable window declines"
    );
}

/// `mirror_vable_static_to_boxes` must complete the `_opimpl_setfield_vable`
/// pairing (pyjitpl.py) — the `virtualizable_boxes[index] = valuebox`
/// store followed by `synchronize_virtualizable()` — so the shadow never runs
/// ahead of the live frame.
///
/// Gated structurally rather than behaviourally on purpose.  The behavioural
/// consequence is `check_synchronized_virtualizable`'s assert, which is
/// `debug_assertions`-only, and on this tree no input reaches it through this
/// helper: with the `synchronize_virtualizable()` call deleted, the whole debug
/// suite passes 7599/0 and all 397 `pyre/bench/synth/*.py` fixtures run clean
/// under a debug `pyre-dynasm` (20 of the first 25 compile loops, and
/// `class_body_exec_hot_loop.py` reaches 4 loops / 1 bridge / 403 guard
/// failures, so the search was live).  A test that asserts the pairing itself
/// bites where a fixture cannot.
#[test]
fn mirroring_a_static_vable_field_flushes_it_to_the_live_frame() {
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    let slot_count = info.num_static_extra_boxes;

    // A frame-shaped block for the flush to land in.  `write_virtualizable_back` walks
    // every static field and the array pointer, so the buffer has to span the
    // whole struct and the array block needs its `[len][items]` header.
    let mut frame = Box::new(std::mem::MaybeUninit::<pyre_interpreter::PyFrame>::zeroed());
    let mut array_block =
        Box::new(std::mem::MaybeUninit::<pyre_object::FixedObjectArray>::zeroed());
    let frame_ptr = frame.as_mut_ptr() as *mut u8;
    let array_ptr = array_block.as_mut_ptr() as *mut u8;
    unsafe {
        let len_ptr = array_ptr.add(pyre_object::FIXED_OBJECT_ARRAY_TOKEN.len_offset) as *mut usize;
        *len_ptr = 0;
        let slot =
            frame_ptr.add(crate::frame_layout::PYFRAME_LOCALS_CELLS_STACK_OFFSET) as *mut *mut u8;
        *slot = array_ptr;
        let last_instr =
            frame_ptr.add(crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET) as *mut isize;
        *last_instr = 11;
    }

    let mut tc = TraceCtx::for_test_types(&[Type::Ref]);
    let vable = tc.const_ref(frame_ptr as i64);
    let initial_boxes = vec![tc.const_null(); slot_count];
    let initial_values = vec![Value::Int(0); slot_count];
    tc.install_virtualizable_info(info.clone());
    tc.init_virtualizable_boxes(
        &info,
        vable,
        Value::Ref(majit_ir::GcRef(frame_ptr as usize)),
        &initial_boxes,
        &initial_values,
        &[0],
    );
    // The synchronization target.  `init_virtualizable_boxes` seeds the boxes
    // and the identity but not this, and `synchronize_virtualizable` returns
    // early without it — an omission that would make the assert below pass for
    // the wrong reason if it were the other way round.
    tc.set_virtualizable_heap_ptr(frame_ptr);

    let read_last_instr = || unsafe {
        *(frame_ptr.add(crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET) as *const isize)
    };

    // Non-vacuity: the frame must NOT already hold the value we are about to
    // publish, or the assert below would hold with no flush at all.
    assert_eq!(
        read_last_instr(),
        11,
        "the fixture must start from a frame value distinct from the published one",
    );

    let published = tc.const_int(37);
    crate::trace_opcode::mirror_vable_static_to_boxes(
        &mut tc,
        "last_instr",
        published,
        Value::Int(37),
    );

    assert_eq!(
        read_last_instr(),
        37,
        "mirroring a static vable field must flush it to the live frame — \
         `_opimpl_setfield_vable` pairs the shadow store with \
         `synchronize_virtualizable()` (pyjitpl.py:1189-1194), and without the \
         flush the next `getfield_vable_*` fails `check_boxes` \
         (virtualizable.py:157)",
    );
    let index = info
        .static_field_index_by_name("last_instr")
        .expect("last_instr is a declared static vable field");
    assert_eq!(
        tc.virtualizable_entry_at(index).map(|entry| entry.1),
        Some(Value::Int(37)),
        "the shadow half must carry the published value too",
    );
}

/// A decline recorded on one thread must be visible to a reader on another.
/// `_thread` gives Python real OS threads and each traces on its own, so a
/// per-thread census reports only whichever thread happened to dump it and
/// silently loses every decline the others took.
#[test]
fn the_decline_census_counts_a_record_from_another_thread() {
    const PROBE: &str = "CensusCrossThreadProbe";
    let count_of = || {
        census_entries()
            .into_iter()
            .find(|(name, _)| *name == PROBE)
            .map_or(0, |(_, count)| count)
    };

    let before = count_of();
    std::thread::spawn(|| census_record(PROBE))
        .join()
        .expect("probe thread panicked");

    assert_eq!(
        count_of(),
        before + 1,
        "a decline recorded off-thread never reached the census"
    );
}

/// The three receivers a traceback walk hops through — `traceback`, `frame`,
/// `code` — refuse to be base classes and are not heap types, so `GuardClass`
/// on the payload already establishes the Python-visible class and
/// `walker_guard_exception_attr_slot` owes no second pin.  `list` and `set`
/// have a `_getusercls` typeptr, so their payload determines the class too.
/// A subclassable builtin without one is the control: `dict` keeps the pin
/// because a `dict` subclass shares the payload and retags `w_class`, and
/// `DICT_TYPE.user_subclass` is null.
#[test]
fn the_traceback_walk_receivers_pin_their_class_through_the_payload() {
    pyre_interpreter::typedef::init_typeobjects();

    for pytype in [
        &pyre_interpreter::pytraceback::PYTRACEBACK_TYPE as *const pyre_object::PyType,
        &pyre_interpreter::pyframe::FRAME_TYPE as *const pyre_object::PyType,
        &pyre_interpreter::pycode::CODE_TYPE as *const pyre_object::PyType,
    ] {
        let w_type = pyre_interpreter::typedef::gettypeobject(unsafe { &*pytype });
        assert!(
            unsafe { walker_payload_determines_w_class(pytype, w_type) },
            "{} carries no subclass and no heap retagging, so the payload determines its class",
            unsafe { pyre_object::w_type_get_name(w_type) },
        );
    }

    let list_type = &pyre_object::LIST_TYPE as *const pyre_object::PyType;
    assert!(
        unsafe {
            walker_payload_determines_w_class(
                list_type,
                pyre_object::pyobject::get_instantiate(&*list_type),
            )
        },
        "a list subclass instance carries LIST_USER_TYPE, so the list payload determines w_class",
    );

    let set_type = &pyre_object::setobject::SET_TYPE as *const pyre_object::PyType;
    assert!(
        unsafe {
            walker_payload_determines_w_class(
                set_type,
                pyre_object::pyobject::get_instantiate(&*set_type),
            )
        },
        "a set subclass instance carries SET_USER_TYPE, so the set payload determines w_class",
    );

    let frozenset_type = &pyre_object::setobject::FROZENSET_TYPE as *const pyre_object::PyType;
    assert!(
        unsafe {
            walker_payload_determines_w_class(
                frozenset_type,
                pyre_object::pyobject::get_instantiate(&*frozenset_type),
            )
        },
        "a frozenset subclass instance carries FROZENSET_USER_TYPE, so the frozenset payload determines w_class",
    );

    let dict_type = &pyre_object::DICT_TYPE as *const pyre_object::PyType;
    assert!(
        unsafe { (*dict_type).user_subclass.is_null() },
        "dict is the control only while its PyType has a null user_subclass",
    );
    assert!(
        !unsafe {
            walker_payload_determines_w_class(
                dict_type,
                pyre_object::pyobject::get_instantiate(&*dict_type),
            )
        },
        "dict is acceptable as a base class, so its payload does not determine w_class",
    );
}
