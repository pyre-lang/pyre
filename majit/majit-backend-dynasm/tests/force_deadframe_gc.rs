//! A forced frame's live `Ref` slots must survive a collection taken while the
//! run that owns the frame is still executing.
//!
//! `force` (`llmodel.py`) casts the force token to a jitframe, resolves
//! it, stamps `jf_descr` and hands that frame straight back — it does not copy
//! anything out. So the deadframe it mints BORROWS: the frame is still on the
//! JF shadow stack, still owns its `jf_forward` chain, and still carries the
//! `jf_gcmap` the call site stored (`aarch64/callbuilder.py push_gcmap` calls
//! `aarch64/assembler.py push_gcmap`, which writes the map into
//! `jf_gcmap` before the call; `pop_gcmap` at `assembler.py:1000-1003` is what
//! zeroes it again, and that only runs once the call has returned). Everything
//! such a deadframe's own lifetime does must therefore be nothing. Releasing
//! the chain would free memory the compiled code is running on, and nulling
//! `jf_gcmap` would stop the frame's trace after its five header words
//! (`jitframe.py:115-116` returns early on a null map), leaving every live
//! `Ref` slot of a still-executing frame unforwarded.
//!
//! All three tests below compile `force_token` -> `call_may_force` ->
//! `guard_not_forced`, with two `Ref` values live across the call so the
//! register allocator spills them into gcmap-covered frame slots. The called
//! helper forces the frame. The two probe objects are also held by registered
//! GC roots, so the collector moves them and names their new addresses, and
//! every read — through the borrowed deadframe and through the guard exit that
//! follows it — has to agree with those roots.
//!
//! The first two differ only in where the deadframe drops relative to the
//! collection, and that is the whole point:
//!
//! * `borrowed_deadframe_reads_the_forwarded_refs_across_a_collection` keeps
//!   it alive across the collection, covering the read path.
//! * `dropping_a_borrowed_deadframe_leaves_the_running_frame_traceable` drops
//!   it FIRST and collects afterwards, so the only thing standing between the
//!   frame's `Ref` slots and the collector is what that drop did to the frame.
//!   A minor collection promotes every surviving nursery object, and promoted
//!   objects do not move again, so a collection taken before the drop cannot
//!   witness a drop that damages the frame. Only this ordering can.
//!
//! The third, `owned_deadframe_root_slot_roots_the_returned_frames_refs`,
//! moves the collection past the point where the run RETURNS, and so covers
//! the other deadframe constructor. `execute_token` mints its result with
//! `JitFrameDeadFrame::new`, which takes an owner-root slot for the frame; by
//! then the compiled epilogue's `gen_footer_shadowstack` has popped the frame
//! off the JF shadow stack, so the reason the borrowed cases stayed traceable
//! is gone. What is left is that slot, walked by the collector's root phase
//! with the rest of the owner roots. That test walks the roots to show the
//! frame is named there and walks the JF shadow stack to show it is in nothing
//! else, and only then collects.
mod common;
use parking_lot::Mutex;
use std::cell::{Cell, UnsafeCell};
use std::sync::Arc;

use majit_backend::{Backend, DeadFrame, JitCellToken};
use majit_backend_dynasm::runner::DynasmBackend;
use majit_ir::forwarding::bound_operand_from_opref as rb;
use majit_ir::{
    CallDescr, DescrRef, EffectInfo, ExtraEffect, GcRef, InputArg, OopSpecIndex, Op, OpCode, OpRc,
    OpRef, Type, Value,
};

/// The owner-root slots and the published nursery state are shared beyond one
/// fixture, so a collection driven by one test would walk
/// another test's frames. Run them one at a time regardless of the harness's
/// parallel scheduling; poison-tolerant so one failure does not wedge the rest.
static SERIAL: Mutex<()> = Mutex::new(());

const PROBE_COUNT: usize = 2;
const PROBE_SIZE: usize = 16;
/// Written into each probe's payload at allocation and re-read through the
/// final slot value: an address that survived forwarding still points at the
/// object, not merely at a plausible-looking word.
const PROBE_MARKERS: [i64; PROBE_COUNT] = [0x7e57_0001, 0x7e57_0002];

thread_local! {
    // Test-only execution context, not backend state: compiled helpers run on
    // the test thread and have no Rust argument through which to receive these
    // root slots and observations. Each test is serialized above, and no value
    // in this block affects process-global identity or GC reachability outside
    // that test's run.
    /// The probe objects, registered with the collector as root slots. Their
    /// addresses have to stay fixed for the whole run, which a thread-local's
    /// storage gives; the collector rewrites the contents in place.
    static PROBES: UnsafeCell<[GcRef; PROBE_COUNT]> =
        const { UnsafeCell::new([GcRef(0); PROBE_COUNT]) };
    /// The backend the may-force helper forces through. The helper is reached
    /// from compiled code, which carries no context of its own.
    static BACKEND: Cell<*const DynasmBackend> = const { Cell::new(std::ptr::null()) };
    /// What the helper saw. Recorded rather than asserted on the spot: the
    /// helper runs on a JIT frame, and a panic there would unwind through
    /// compiled code.
    static OBSERVED: Cell<Option<Observation>> = const { Cell::new(None) };
}

fn probes_ptr() -> *mut GcRef {
    PROBES.with(|probes| probes.get().cast::<GcRef>())
}

fn probe_snapshot() -> [GcRef; PROBE_COUNT] {
    let base = probes_ptr();
    std::array::from_fn(|index| unsafe { *base.add(index) })
}

/// One run of the may-force helper.
#[derive(Clone, Copy, Debug)]
struct Observation {
    /// The frame's `Ref` slots read through the borrowed deadframe, before any
    /// collection the helper takes.
    before: [GcRef; PROBE_COUNT],
    /// The same slots read through the same deadframe after the collection, or
    /// `None` when the helper took no collection with the deadframe alive.
    after: Option<[GcRef; PROBE_COUNT]>,
    /// The registered roots when the helper read `before`.
    roots_before: [GcRef; PROBE_COUNT],
    /// The registered roots when the helper returned — equal to `roots_before`
    /// for a helper that takes no collection at all.
    roots_after: [GcRef; PROBE_COUNT],
    /// The frame `force` resolved to. Recorded so the owning case can show the
    /// frame it inspects afterwards is this one and not a `_check_frame_depth`
    /// replacement, which would make `frame_on_jf_stack` a fact about a
    /// different object.
    forced_frame: usize,
    /// Whether the JF shadow stack named the forced frame while the run was
    /// still in the call. It is the premise the borrowed cases rest on, and the
    /// control for the owning case's claim that the frame has LEFT the stack by
    /// the time the run returns.
    frame_on_jf_stack: bool,
}

fn active_backend() -> &'static DynasmBackend {
    let ptr = BACKEND.with(Cell::get);
    assert!(!ptr.is_null(), "no backend published for the helper");
    unsafe { &*ptr }
}

fn read_slots(backend: &DynasmBackend, frame: &DeadFrame) -> [GcRef; PROBE_COUNT] {
    std::array::from_fn(|index| backend.get_ref_value(frame, index))
}

/// The frame `deadframe` reads through, at its current address.
fn frame_addr_of(deadframe: &DeadFrame) -> usize {
    deadframe
        .as_jitframe()
        .expect("a collector with JITFRAME registered hands out nursery frames")
        .jf_gcref()
        .0
}

/// The jitframe addresses this thread's JF shadow stack currently holds.
///
/// `walk_jf_roots` is the walk the collector's root phase makes over this
/// thread's stack, so an address missing from here is an address the
/// shadow-stack arm of that phase cannot reach.
fn jf_shadow_stack_addrs() -> Vec<usize> {
    let mut addrs = Vec::new();
    majit_gc::shadow_stack::walk_jf_roots(|jf| addrs.push(jf.0));
    addrs
}

/// The addresses this thread's non-jitframe roots currently hold — the
/// shadow stack and the owner-root slots, the latter being where
/// `JitFrameDeadFrame::new` parks the frame it owns and `Drop` releases it.
fn owner_root_addrs() -> Vec<usize> {
    let mut addrs = Vec::new();
    majit_gc::shadow_stack::walk_roots(|root| addrs.push(root.0));
    addrs
}

/// Force, then collect with the deadframe still alive, then read through it.
extern "C" fn force_collect_then_read(force_token: i64) {
    let backend = active_backend();
    let Some(deadframe) = backend.force(GcRef(force_token as usize)) else {
        return;
    };
    let forced_frame = frame_addr_of(&deadframe);
    let frame_on_jf_stack = jf_shadow_stack_addrs().contains(&forced_frame);
    let before = read_slots(backend, &deadframe);
    let roots_before = probe_snapshot();
    majit_gc::collect_full();
    let after = read_slots(backend, &deadframe);
    let roots_after = probe_snapshot();
    drop(deadframe);
    OBSERVED.set(Some(Observation {
        before,
        after: Some(after),
        roots_before,
        roots_after,
        forced_frame,
        frame_on_jf_stack,
    }));
}

/// Force, read, drop the deadframe, and only then collect — so the collection
/// sees whatever state the drop left the still-executing frame in.
extern "C" fn force_drop_then_collect(force_token: i64) {
    let backend = active_backend();
    let Some(deadframe) = backend.force(GcRef(force_token as usize)) else {
        return;
    };
    let forced_frame = frame_addr_of(&deadframe);
    let frame_on_jf_stack = jf_shadow_stack_addrs().contains(&forced_frame);
    let before = read_slots(backend, &deadframe);
    let roots_before = probe_snapshot();
    drop(deadframe);
    majit_gc::collect_full();
    let roots_after = probe_snapshot();
    OBSERVED.set(Some(Observation {
        before,
        after: None,
        roots_before,
        roots_after,
        forced_frame,
        frame_on_jf_stack,
    }));
}

/// Force and read, but take no collection at all — the caller collects once the
/// run has returned. Nor may anything here reach the GC allocator: the
/// collection the owning test takes has to be the FIRST one the probes see, or
/// a minor collection would already have promoted them and their addresses
/// could no longer change.
extern "C" fn force_read_and_collect_nothing(force_token: i64) {
    let backend = active_backend();
    let Some(deadframe) = backend.force(GcRef(force_token as usize)) else {
        return;
    };
    let forced_frame = frame_addr_of(&deadframe);
    let frame_on_jf_stack = jf_shadow_stack_addrs().contains(&forced_frame);
    let before = read_slots(backend, &deadframe);
    let roots_before = probe_snapshot();
    drop(deadframe);
    OBSERVED.set(Some(Observation {
        before,
        after: None,
        roots_before,
        roots_after: roots_before,
        forced_frame,
        frame_on_jf_stack,
    }));
}

/// The `CALL_MAY_FORCE` descriptor: one `Ref` argument, the force token, and
/// no result. The function address itself is argument 0 of the operation and
/// is not part of the signature.
#[derive(Debug)]
struct MayForceCallDescr {
    arg_types: Vec<Type>,
}

impl majit_ir::Descr for MayForceCallDescr {
    fn index(&self) -> u32 {
        u32::MAX
    }

    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        Some(self)
    }
}

impl CallDescr for MayForceCallDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }

    fn result_type(&self) -> Type {
        Type::Void
    }

    fn result_size(&self) -> usize {
        0
    }

    fn get_extra_info(&self) -> &EffectInfo {
        static INFO: EffectInfo = EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
        &INFO
    }
}

/// A compiled `force_token` / `call_may_force` / `guard_not_forced` trace over
/// two live `Ref` values, with the probe objects allocated and rooted.
struct Fixture {
    backend: DynasmBackend,
    token: JitCellToken,
}

impl Fixture {
    fn build(helper: extern "C" fn(i64), token_number: u64) -> Fixture {
        let mut gc = majit_gc::collector::MiniMarkGC::new();
        let probe_tid = gc.register_type(majit_gc::TypeInfo::simple(PROBE_SIZE));
        // jitframe.py — the frame is a nursery object traced through
        // `jf_gcmap`; a collector with a type table must carry JITFRAME.
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
        let mut allocated = [GcRef(0); PROBE_COUNT];
        for (index, slot) in allocated.iter_mut().enumerate() {
            let obj = gc.alloc_with_type(probe_tid, PROBE_SIZE);
            assert!(!obj.is_null(), "probe {index} allocation failed");
            unsafe { *(obj.0 as *mut i64) = PROBE_MARKERS[index] };
            *slot = obj;
        }

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        backend.set_gc_allocator(Box::new(gc));

        // A second, independent hold on the same objects. It is what makes
        // them survive the collection and what names their post-collection
        // addresses; the frame slots are then checked against it.
        let base = probes_ptr();
        for (index, obj) in allocated.iter().enumerate() {
            unsafe {
                *base.add(index) = *obj;
                assert!(
                    majit_gc::gc_add_root(base.add(index)),
                    "the installed allocator published no root hook"
                );
            }
        }

        let inputargs = vec![
            InputArg::from_type_rc(Type::Ref, 0),
            InputArg::from_type_rc(Type::Ref, 1),
        ];
        let r0 = inputargs[0].opref();
        let r1 = inputargs[1].opref();

        let force_token = Op::new(OpCode::ForceToken, &[]);
        force_token.pos().set(OpRef::ref_op(2));

        let call = Op::new(
            OpCode::CallMayForceN,
            &[
                rb(OpRef::const_int(helper as usize as i64)),
                rb(OpRef::ref_op(2)),
            ],
        );
        call.pos().set(OpRef::void_op(3));
        call.setdescr(Arc::new(MayForceCallDescr {
            arg_types: vec![Type::Ref],
        }) as DescrRef);

        // Both probes are fail args here, so they are live across the call and
        // the call's gcmap has to cover the slots they spill to.
        let guard = Op::new(OpCode::GuardNotForced, &[]);
        guard.pos().set(OpRef::void_op(4));
        guard.set_fail_arg_types(vec![Type::Ref, Type::Ref]);
        guard.setfailargs(vec![rb(r0), rb(r1)].into());

        let finish = Op::new(OpCode::Finish, &[]);
        finish.pos().set(OpRef::void_op(5));
        finish.set_fail_arg_types(vec![Type::Ref, Type::Ref]);
        finish.setfailargs(vec![rb(r0), rb(r1)].into());

        let ops: Vec<OpRc> = vec![force_token, call, guard, finish]
            .into_iter()
            .map(OpRc::new)
            .collect();
        let token = JitCellToken::new(token_number as i64);
        common::rebind_trace_arg_boxes(&inputargs, &ops);
        backend
            .compile_loop(&inputargs, &ops, &token)
            .expect("compile_loop");

        Fixture { backend, token }
    }

    /// Enter the trace and return the guard exit's deadframe plus what the
    /// helper recorded on the way through.
    fn run(&self) -> (DeadFrame, Observation) {
        OBSERVED.set(None);
        BACKEND.with(|cell| cell.set(&self.backend as *const DynasmBackend));
        let args: Vec<Value> = probe_snapshot().iter().map(|r| Value::Ref(*r)).collect();
        let frame = self.backend.execute_token(&self.token, &args);
        let observed = OBSERVED
            .take()
            .expect("the may-force helper never recorded a run");
        (frame, observed)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let base = probes_ptr();
        for index in 0..PROBE_COUNT {
            majit_gc::gc_remove_root(unsafe { base.add(index) });
        }
        BACKEND.with(|cell| cell.set(std::ptr::null()));
        majit_backend_dynasm::runner::clear_gc_allocator();
    }
}

/// Assertions every ordering shares: the helper forced a real frame, the
/// collection actually moved the probes, and the guard exit reports the
/// post-collection addresses.
///
/// `roots_after` is passed rather than read off `observed` because the
/// collection that matters is not always the helper's: the owning case takes
/// its own once the run has returned, and it is that one the guard exit's
/// slots have to agree with.
fn check_common(
    fixture: &Fixture,
    frame: &DeadFrame,
    observed: &Observation,
    roots_after: [GcRef; PROBE_COUNT],
) {
    assert!(
        !fixture.backend.get_latest_descr(frame).is_finish(),
        "the helper forced the frame, so GUARD_NOT_FORCED had to fail: {observed:?}"
    );
    // `gen_shadowstack_header` pushes the entry frame before the trace body
    // runs, so a helper reached from inside that body must see it. This is also
    // the control for the owning case's later claim that the stack no longer
    // names the frame: without it an empty walk proves nothing.
    assert!(
        observed.frame_on_jf_stack,
        "the JF shadow stack did not name the forced frame mid-run, so the compiled prologue \
         never pushed it and no walk of that stack means anything here: {observed:?}"
    );
    assert_eq!(
        observed.before, observed.roots_before,
        "the borrowed deadframe read slots that were not the probes"
    );
    // Without a move, a slot that was never forwarded reads the same as one
    // that was, and every comparison below holds for the wrong reason.
    for index in 0..PROBE_COUNT {
        assert_ne!(
            observed.roots_before[index], roots_after[index],
            "probe {index} did not move, so the forwarding checks are vacuous: \
             {observed:?} roots_after={roots_after:?}"
        );
    }

    let exit = read_slots(&fixture.backend, frame);
    assert_eq!(
        exit, roots_after,
        "the guard exit reported pre-collection addresses for the forced frame's Ref slots"
    );
    for (index, slot) in exit.iter().enumerate() {
        assert_eq!(
            unsafe { *(slot.0 as *const i64) },
            PROBE_MARKERS[index],
            "the guard exit's fail arg {index} does not point at the probe object"
        );
    }
}

/// A collection taken while the borrowed deadframe is alive forwards the
/// running frame's `Ref` slots, and the deadframe — which reads the frame in
/// place rather than a copy — answers the forwarded values.
#[test]
fn borrowed_deadframe_reads_the_forwarded_refs_across_a_collection() {
    let _serial = SERIAL.lock();
    let fixture = Fixture::build(force_collect_then_read, 5001);
    let (frame, observed) = fixture.run();

    let after = observed
        .after
        .expect("this helper reads through the deadframe after collecting");
    assert_eq!(
        after, observed.roots_after,
        "the borrowed deadframe kept answering pre-collection addresses"
    );
    check_common(&fixture, &frame, &observed, observed.roots_after);
}

/// Dropping the borrowed deadframe must leave the still-executing frame
/// exactly as it found it. A collection taken after the drop is the only thing
/// that can see the difference, because it is the first one whose outcome
/// depends on the frame's `jf_gcmap` still being the map the call site stored.
#[test]
fn dropping_a_borrowed_deadframe_leaves_the_running_frame_traceable() {
    let _serial = SERIAL.lock();
    let fixture = Fixture::build(force_drop_then_collect, 5002);
    let (frame, observed) = fixture.run();

    assert!(
        observed.after.is_none(),
        "this helper drops the deadframe before collecting"
    );
    check_common(&fixture, &frame, &observed, observed.roots_after);
}

/// `execute_token` hands back a deadframe minted by `JitFrameDeadFrame::new`,
/// which takes an owner-root slot for the frame. By then
/// `gen_footer_shadowstack` has popped the frame off the JF shadow stack, so
/// that slot is the only root naming it — asserted here both ways round,
/// present among the owner roots and absent from the shadow stack, before
/// anything is collected.
///
/// The collection is taken here and not inside the helper, and that ordering is
/// the test: a minor collection promotes every surviving nursery object and
/// promoted objects do not move again, so a collection taken while the frame
/// was still on the shadow stack would fix the probe addresses and leave this
/// one with nothing left to forward.
#[test]
fn owned_deadframe_root_slot_roots_the_returned_frames_refs() {
    let _serial = SERIAL.lock();
    let fixture = Fixture::build(force_read_and_collect_nothing, 5003);
    let (frame, observed) = fixture.run();
    assert!(
        observed.after.is_none(),
        "this helper takes no collection of its own"
    );

    let tip = frame_addr_of(&frame);
    assert_eq!(
        tip, observed.forced_frame,
        "the returned frame is not the one the helper saw on the JF shadow stack, so the \
         absence check below would be about a different frame"
    );

    let live = owner_root_addrs();
    assert!(
        live.contains(&tip),
        "the returned deadframe's frame {tip:#x} is not among the owner roots ({live:#x?}), \
         so the collector's root phase never names it"
    );
    let on_stack = jf_shadow_stack_addrs();
    assert!(
        !on_stack.contains(&tip),
        "frame {tip:#x} is still on the JF shadow stack ({on_stack:#x?}), so the collection \
         below would root it through the stack and say nothing about the owner root"
    );

    majit_gc::collect_full();
    let roots_after = probe_snapshot();
    check_common(&fixture, &frame, &observed, roots_after);

    // The frame moved with the collection; the slot names the new address.
    let moved = frame_addr_of(&frame);
    let live = owner_root_addrs();
    assert!(
        live.contains(&moved),
        "after the collection the owner roots ({live:#x?}) no longer name the frame {moved:#x}"
    );
    drop(frame);
    let live = owner_root_addrs();
    assert!(
        !live.contains(&moved),
        "dropping the deadframe left frame {moved:#x} among the owner roots ({live:#x?}), \
         which would keep a dead frame alive"
    );
}
