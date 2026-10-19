//! The six `BC_*ARRAYITEM_VABLE_*` dispatch arms must decide
//! `_nonstandard_virtualizable` BEFORE promoting the index.
//!
//! `pyjitpl.py _opimpl_getarrayitem_vable` (and `:1236-1247
//! _opimpl_setarrayitem_vable`) open with the decision:
//!
//! ```python
//! def _opimpl_getarrayitem_vable(self, pc, ..., arrayindex, index, ...):
//!     if self.metainterp._nonstandard_virtualizable(pc, box, fdescr):
//!         arraybox = self.opimpl_getfield_gc_r(box, fdescr)
//!         return self.opimpl_getarrayitem_gc_i(arraybox, indexbox, adescr)
//!     self.metainterp.check_synchronized_virtualizable()
//!     index = self._get_arrayitem_vable_index(pc, arrayindex, indexbox)
//! ```
//!
//! The promote lives on the first line of `_get_arrayitem_vable_index`
//! (`pyjitpl.py`), which only the STANDARD leg reaches. The non-standard
//! leg goes to an ordinary `getfield_gc_r` + `getarrayitem_gc_*` with the index
//! box exactly as it stands.
//!
//! majit's dispatch arms hoist the promote out of `TraceCtx` to the call site
//! on purpose — the walker owns the `MIFrameStack`, so promoting here gives the
//! guard a full-framestack snapshot, and `implement_guard_value`'s register-bank
//! rewrite is something only the walker can do. The hazard the hoist creates is
//! that it is easy to hoist it one step too far, above the branch that selects
//! it, and a non-standard access with a non-constant index then mints a
//! `GUARD_VALUE` that upstream does not — over-specializing the trace on what
//! is an ordinary heap read.
//!
//! ## Two things had to be armed, because a `!` assertion cannot arm itself
//!
//! The subject here is which BOX a guard names, not how many guards there are:
//! the non-standard branch mints a `GUARD_VALUE` of its own for the Step 4
//! `PTR_EQ` promote inside `_nonstandard_virtualizable`. So "no guard named the
//! index" is also what a walk that aborted before reaching the arm would
//! report, and what a fixture whose vable box was never resolved would report.
//! Each case therefore asserts that the arm ran and minted its own guard before
//! asserting what was *not* promoted.
//!
//! That still only proves the arm ran. That the assertion can FAIL was checked
//! by inverting the production decision by hand — see the note on the test
//! itself for the exact inversion and what it produced.

use std::sync::Arc;

use majit_ir::{OpRef, Type, Value};
use majit_metainterp::jitcode::JitCodeBuilder;
use majit_metainterp::virtualizable::VirtualizableInfo;
use majit_metainterp::{
    ClosureRuntime, JitCode, JitCodeMachine, JitCodeSym, MIFrame, MIFrameStack,
};

/// The minimum a `JitCodeSym` has to answer for the dispatcher to run one
/// opcode. Every other member of the trait has a default.
struct MinimalSym;

impl JitCodeSym for MinimalSym {
    fn total_slots(&self) -> usize {
        0
    }

    fn loop_header_pc(&self) -> usize {
        0
    }
}

/// A virtualizable with one int array, and nothing else — the smallest shape
/// that makes `vable_array_descrs` resolve an `array_field_descrs[0]` and an
/// `array_descrs[0]` for the arm to hand to the heap ops.
fn one_int_array_vinfo() -> Arc<VirtualizableInfo> {
    const TOKEN_OFFSET: usize = 0;
    const ARRAY_FIELD_OFFSET: usize = 8;
    const LENGTH_OFFSET: usize = 0;
    const ITEMS_OFFSET: usize = 8;
    const ITEM_SIZE: usize = 8;

    let mut info = VirtualizableInfo::new(TOKEN_OFFSET);
    info.add_array_field(
        "items",
        Type::Int,
        ARRAY_FIELD_OFFSET,
        LENGTH_OFFSET,
        ITEMS_OFFSET,
        majit_ir::make_array_descr(ITEMS_OFFSET, ITEM_SIZE, Type::Int),
    );
    // `virtualizable.py finish()`. `_nonstandard_virtualizable` Step 3
    // emits the force (`pyjitpl.py emit_force_virtualizable`) before it
    // reaches the Step 4 `PTR_EQ`, and that emission reads `clear_vable_ptr`.
    // Without it the fixture panics inside the arm — which would look like the
    // arm never running.
    info.set_clear_vable(
        clear_vable_noop as *const (),
        VirtualizableInfo::make_clear_vable_descr(),
    );
    info.finalize_arc(majit_ir::descr::make_size_descr(64))
}

/// Stands in for `virtualizable.py clear_vable_token`. The emission under
/// test only bakes its ADDRESS into a `COND_CALL`; nothing in a one-step
/// dispatch calls it, so a body that clears nothing is the honest fixture.
extern "C" fn clear_vable_noop(_vable: *mut u8) {}

/// Which of the two arms under test to build a jitcode for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Get,
    Set,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Get => "BC_GETARRAYITEM_VABLE_I",
            Arm::Set => "BC_SETARRAYITEM_VABLE_I",
        }
    }
}

/// Registers the fixture's jitcode uses. Named rather than spelled inline
/// because the two arms read them in different orders.
const VABLE_REG: u16 = 0;
const INDEX_REG: u16 = 0;
const VALUE_REG: u16 = 1;
const DEST_REG: u16 = 1;
const ARRAY_IDX: u16 = 0;

fn build_jitcode(
    arm: Arm,
) -> (
    Arc<JitCode>,
    usize,
    majit_jitcode::codewriter::assembler::Assembler,
) {
    let mut asm = majit_jitcode::codewriter::assembler::Assembler::new();
    let mut builder = JitCodeBuilder::new();
    let live_ints: &[u8] = if arm == Arm::Set { &[0, 1] } else { &[0] };
    builder.live(&mut asm, live_ints, &[0], &[]);
    let pc = builder.current_pos();
    match arm {
        Arm::Get => {
            builder.vable_getarrayitem_int_with_base(DEST_REG, VABLE_REG, ARRAY_IDX, INDEX_REG)
        }
        Arm::Set => {
            builder.vable_setarrayitem_int_with_base(VABLE_REG, ARRAY_IDX, INDEX_REG, VALUE_REG)
        }
    }
    let jitcode = Arc::new(builder.finish());
    jitcode.set_index(0);
    (jitcode, pc, asm)
}

/// Whether the vable register holds a recorded box or a prebuilt constant.
/// Step 4's `PTR_EQ` runs through `execute_and_record`, so the two shapes
/// differ in whether the comparison survives into the trace.
#[derive(Clone, Copy)]
enum VableBox {
    Recorded,
    Constant,
}

/// Run one dispatch step of `arm` against a NON-standard virtualizable whose
/// index box is non-constant, and report `(guards minted, whether any guard
/// names the index box, whether Step 4's PTR_EQ was recorded)`.
fn run_arm(arm: Arm, vable: VableBox) -> (usize, bool, bool) {
    let info = one_int_array_vinfo();
    let (jitcode, pc, asm) = build_jitcode(arm);
    let mut staticdata = majit_metainterp::MetaInterpStaticData::new();
    staticdata.op_live = majit_metainterp::jitcode::insns::BC_LIVE as i32;
    staticdata.liveness_info.set(asm.all_liveness().to_vec());
    let mut meta = majit_metainterp::MetaInterp::<()>::new(200);
    meta.staticdata = Arc::new(staticdata);
    meta.finish_setup_descrs_for_jitdrivers();
    meta.force_start_tracing(0, (0, 0), None, None, &[Value::Ref(majit_ir::GcRef(2))]);
    let ctx = meta.trace_ctx().expect("force-started fixture trace");
    // This is an IR-shape fixture with symbolic pointer sentinels, as with
    // TraceCtx::for_test_types; do not ask a native CPU to dereference them.
    ctx.set_cpu(None);

    // The STANDARD virtualizable, i.e. `virtualizable_boxes[-1]`
    // (`pyjitpl.py:1195 virtualizable_boxes[-1]`).
    const ARRAY_LEN: usize = 3;
    let standard = ctx.const_ref(1);
    let slot_count = info.num_static_extra_boxes + ARRAY_LEN;
    let initial_boxes = vec![ctx.const_null(); slot_count];
    let initial_values = vec![Value::Ref(majit_ir::GcRef::NULL); slot_count];
    ctx.install_virtualizable_info(info.clone());
    ctx.init_virtualizable_boxes(
        &info,
        standard,
        Value::Ref(majit_ir::GcRef(1)),
        &initial_boxes,
        &initial_values,
        &[ARRAY_LEN],
    );

    // A DIFFERENT object in the vable register: `_nonstandard_virtualizable`
    // reaches its Step 4 `PTR_EQ`, reads unequal, and takes the non-standard
    // branch. This is the whole point of the fixture — the standard branch
    // promotes on purpose.
    //
    // `VableBox::Recorded` must NOT be a `ConstPtr`. Step 4 runs its `PTR_EQ`
    // through `execute_and_record`, so a constant on both sides folds the
    // comparison and no guard is minted — which would leave the `guards > 0`
    // line measuring nothing. A virtualizable in a register is a recorded box
    // in production anyway; the concrete pointer lives on the box
    // (`InputArg.getref_base` / `set_opref_concrete`), not from the
    // box being a ConstPtr.
    let other_vable = match vable {
        VableBox::Recorded => {
            let b = OpRef::input_arg_ref(0);
            ctx.set_opref_concrete(b, Value::Ref(majit_ir::GcRef(2)));
            b
        }
        VableBox::Constant => ctx.const_ref(2),
    };

    // A non-constant index carrying a recording-time concrete. Any other shape
    // makes the promote a no-op, so the test would pass without measuring it.
    let zero = ctx.const_int(0);
    let index = ctx.record_op(majit_ir::OpCode::IntAdd, &[zero, zero]);
    ctx.set_opref_concrete(index, Value::Int(0));
    assert!(
        !index.is_constant(),
        "the index box must be promotable, or nothing here is being measured"
    );
    let stored = ctx.const_int(7);

    let mut frame = MIFrame::new(jitcode, pc);
    // This fixture executes the array opcode once, not its preceding LIVE.
    // MIFrame::new initializes the bytecode cursor independently of `pc`.
    frame.code_cursor = pc;
    frame.ref_regs[VABLE_REG as usize] = Some(other_vable);
    frame.int_regs[INDEX_REG as usize] = Some(index);
    if arm == Arm::Set {
        frame.int_regs[VALUE_REG as usize] = Some(stored);
    }
    let mut frames = MIFrameStack::empty();
    frames.frames.push(frame);

    let guards_before = ctx.num_guards();
    let runtime = ClosureRuntime::new(|pc: usize| pc);
    let mut sym = MinimalSym;
    let mut machine = JitCodeMachine::<MinimalSym, _>::with_framestack(&mut frames, &[], &[]);
    machine.run_one_step(ctx, &mut sym, &runtime);
    drop(machine);

    ctx.ensure_ops_materialized();
    let promoted_index = ctx.ops().iter().any(|recorded| {
        recorded.opcode == majit_ir::OpCode::GuardValue
            && recorded
                .getarglist()
                .first()
                .is_some_and(|arg| arg.to_opref() == index)
    });
    let recorded_ptr_eq = ctx
        .ops()
        .iter()
        .any(|recorded| recorded.opcode == majit_ir::OpCode::PtrEq);
    (
        ctx.num_guards() - guards_before,
        promoted_index,
        recorded_ptr_eq,
    )
}

/// ## The negative control, and how it was taken
///
/// This assertion is a `!promoted_index`, which cannot arm itself. The `guards
/// > 0` line below is what says the arm ran at all; that it fires on the
/// pre-fix SHAPE was checked by hand rather than by a knob, because a
/// production flag whose only caller is a test is a second code path to keep
/// correct forever.
///
/// The inversion applied to each of the six arms in
/// `pyjitpl/dispatch.rs` was to move the promote back above the branch, i.e.
/// replace
///
/// ```text
/// let nonstandard = ctx.nonstandard_virtualizable(..);
/// let index = if nonstandard { index } else { self.implement_guard_value(..) };
/// ```
///
/// with the pre-fix
///
/// ```text
/// let index = self.implement_guard_value(..);
/// let nonstandard = ctx.nonstandard_virtualizable(..);
/// ```
///
/// Applied to all six arms, this test fails on the first case
/// (`BC_GETARRAYITEM_VABLE_I`) with the `promoted_index` message below;
/// restored, it passes. Anyone re-narrowing the decision has to reproduce that
/// by the same route.
#[test]
fn a_nonstandard_vable_array_access_does_not_promote_the_index() {
    for arm in [Arm::Get, Arm::Set] {
        let (guards, promoted_index, recorded_ptr_eq) = run_arm(arm, VableBox::Recorded);
        assert!(recorded_ptr_eq, "{}: Step 4's PTR_EQ", arm.name());
        assert!(
            guards > 0,
            "{}: the arm must still mint the Step 4 PTR_EQ promote of \
             `_nonstandard_virtualizable`. Zero guards means the walk never \
             reached the branch, and the index assertion below would then be \
             passing vacuously.",
            arm.name()
        );
        assert!(
            !promoted_index,
            "{}: the index box was promoted on the NON-standard branch. \
             `pyjitpl.py:1220` / `:1239` reach `getfield_gc_r` + \
             `get|setarrayitem_gc_*` with the index box untouched; the promote \
             belongs to `_get_arrayitem_vable_index` (`pyjitpl.py:1205`), which \
             only the standard leg enters.",
            arm.name()
        );
    }
}

/// The same walk with a prebuilt `ConstPtr` in the vable register. Step 3
/// already returned for `standard_box is box`, so two constants reaching Step 4
/// are *different* constants and cannot be equal at runtime either — the
/// `PTR_EQ` folds to `ConstInt(0)` and `implement_guard_value`'s Const arm then
/// declines the promote, leaving neither the comparison nor a `GUARD_VALUE` in
/// the trace.
#[test]
fn a_constant_vable_folds_the_step_four_ptr_eq_away() {
    for arm in [Arm::Get, Arm::Set] {
        let (guards, promoted_index, recorded_ptr_eq) = run_arm(arm, VableBox::Constant);
        assert!(
            !recorded_ptr_eq,
            "{}: two different ConstPtrs must fold, not record",
            arm.name()
        );
        assert_eq!(
            guards,
            0,
            "{}: a folded PTR_EQ has nothing to promote",
            arm.name()
        );
        assert!(!promoted_index, "{}", arm.name());
    }
}
