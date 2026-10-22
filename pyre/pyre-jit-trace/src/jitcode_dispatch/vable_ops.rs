//! Virtualizable field / array access via the metainterp.
//!
//! **Parity:** trace-side counterpart of `pyjitpl.py`'s virtualizable
//! `opimpl_*`, backed by `virtualizable.py` / `virtualref.py`
//! (implementation mirrors `majit-metainterp/virtualizable.rs` /
//! `virtualref.rs`).
//!
//! getfield / setfield / getarrayitem / setarrayitem / arraylen against
//! a virtualizable, routed through the metainterp vable bookkeeping.

use super::*;

/// Register-bank half of `pyjitpl.MetaInterp.replace_box`, owned by a live
/// walker frame. Resume snapshots (`InlineParentFrame.boxes`) are rewritten
/// immediately, as `replace_box` walks `framestack`.
///
/// All register banks share the frame's owned slots, so replacement follows
/// `MIFrame.replace_active_box_in_frame` without a retained mutable slice or
/// delayed replay. A paused Python caller stores those lists on
/// `InlineParentFrame`; the portal is `WalkSession.framestack[0]`; a
/// transparent helper uses `WalkSession.helper_live`.
fn replace_slots(slots: &mut [OpRef], oldbox: OpRef, newbox: OpRef) {
    for slot in slots {
        if *slot == oldbox {
            *slot = newbox;
        }
    }
}

fn replace_box_in_walk_frame<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    oldbox: OpRef,
    newbox: OpRef,
) {
    ctx.frame_state.replace_active_box(oldbox, newbox);
}

fn replace_live_regs(live: &LiveFrameRegs, oldbox: OpRef, newbox: OpRef) {
    live.replace_active_box(oldbox, newbox);
}

/// Bind this walk's live banks onto the MIFrame that is about to pause
/// for a child: the top inlined Python frame, or the portal.
pub(super) fn bind_paused_caller_regs(
    session: &std::cell::RefCell<WalkSession>,
    registers_r: &RegisterBank,
    registers_i: &RegisterBank,
    registers_f: &RegisterBank,
    frame_state: &WalkFrameState,
) {
    let live = LiveFrameRegs::new(registers_r, registers_i, registers_f, frame_state);
    let mut session = session.borrow_mut();
    if session.framestack.is_empty() {
        session.framestack.push(InlineFrame::portal());
    }
    session
        .framestack
        .last_mut()
        .expect("portal occupies framestack[0]")
        .live = Some(live);
}

pub(super) fn push_helper_live(
    session: &std::cell::RefCell<WalkSession>,
    registers_r: &RegisterBank,
    registers_i: &RegisterBank,
    registers_f: &RegisterBank,
    frame_state: &WalkFrameState,
) {
    session.borrow_mut().helper_live.push(LiveFrameRegs::new(
        registers_r,
        registers_i,
        registers_f,
        frame_state,
    ));
}

pub(super) fn pop_helper_live(session: &std::cell::RefCell<WalkSession>) {
    session.borrow_mut().helper_live.pop();
}

fn replace_box_in_paused_frames(session: &mut WalkSession, oldbox: OpRef, newbox: OpRef) {
    for live in &session.helper_live {
        replace_live_regs(live, oldbox, newbox);
    }
    for frame in &mut session.framestack {
        if let Some(live) = frame.live.as_ref() {
            replace_live_regs(live, oldbox, newbox);
        }
    }
    for frame in &mut session.framestack {
        let call_entry = match &mut frame.helper_call_entry {
            HelperCallEntry::Frame(entry) => Some(entry),
            HelperCallEntry::Unavailable | HelperCallEntry::CallerBoundary => None,
        };
        for parent in frame.parents.iter_mut().chain(call_entry) {
            replace_slots(&mut parent.boxes, oldbox, newbox);
            if let Some(state) = parent.frame_state.as_ref() {
                state.replace_active_box(oldbox, newbox);
            }
            let bank = match oldbox.ty() {
                Some(Type::Int) => parent.registers_i.as_ref(),
                Some(Type::Ref) => parent.registers_r.as_ref(),
                Some(Type::Float) => parent.registers_f.as_ref(),
                _ => None,
            };
            if let Some(bank) = bank {
                bank.replace_active_box(oldbox, newbox);
            }
            if let Some(blackhole) = parent.blackhole.as_mut() {
                for (_, value) in &mut blackhole.float_values {
                    replace_slots(std::slice::from_mut(value), oldbox, newbox);
                }
            }
        }
    }
}

/// Install the walker's framestack half of `MetaInterp.replace_box` so
/// `_nonstandard_virtualizable` can rewrite banks immediately.
pub(super) fn with_replace_frames<Sym: WalkSym, R>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    f: impl FnOnce(&mut WalkContext<'_, '_, Sym>) -> R,
) -> R {
    struct WalkerReplaceData {
        session: *const std::cell::RefCell<WalkSession>,
        registers_r: *const RegisterBank,
        registers_i: *const RegisterBank,
        registers_f: *const RegisterBank,
        frame_state: *const WalkFrameState,
    }
    unsafe fn walk(data: *mut (), oldbox: OpRef, newbox: OpRef) {
        let data = unsafe { &*data.cast::<WalkerReplaceData>() };
        let mut session = unsafe { (*data.session).borrow_mut() };
        replace_box_in_paused_frames(&mut session, oldbox, newbox);
        match oldbox.ty() {
            Some(Type::Int) => unsafe { (*data.registers_i).replace_active_box(oldbox, newbox) },
            Some(Type::Ref) => unsafe { (*data.registers_r).replace_active_box(oldbox, newbox) },
            Some(Type::Float) => unsafe { (*data.registers_f).replace_active_box(oldbox, newbox) },
            _ => {}
        }
        unsafe {
            (*data.frame_state).replace_active_box(oldbox, newbox);
        }
    }
    let mut data = WalkerReplaceData {
        session: ctx.session,
        registers_r: ctx.registers_r,
        registers_i: ctx.registers_i,
        registers_f: ctx.registers_f,
        frame_state: &ctx.frame_state,
    };
    // SAFETY: `data` is the walk-local ReplaceData, live until
    // `ClearReplaceFrames` drops (normal return or unwind).
    unsafe {
        ctx.trace_ctx
            .set_replace_frames(Some(walk), &raw mut data as *mut ());
    }
    // SAFETY: the guard is a local of this call and `ctx` is borrowed for
    // longer, so it is dropped while the `TraceCtx` it names is alive.
    let _clear = unsafe { majit_metainterp::ClearReplaceFrames::new(&mut *ctx.trace_ctx) };
    f(ctx)
}

/// `pyjitpl.py MetaInterp.replace_box`: promotion updates the recording
/// state and every live MIFrame, including callers suspended in descendants.
pub(super) fn walker_replace_box<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    oldbox: OpRef,
    newbox: OpRef,
) {
    ctx.trace_ctx.replace_box(oldbox, newbox);
    replace_box_in_all_walk_frames(ctx, oldbox, newbox);
}

fn replace_box_in_all_walk_frames<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    oldbox: OpRef,
    newbox: OpRef,
) {
    replace_box_in_paused_frames(&mut ctx.session.borrow_mut(), oldbox, newbox);
    let bank = match oldbox.ty() {
        Some(Type::Int) => ctx.registers_i,
        Some(Type::Ref) => ctx.registers_r,
        Some(Type::Float) => ctx.registers_f,
        _ => panic!("replace_active_box_in_frame requires a typed value: {oldbox:?}"),
    };
    bank.replace_active_box(oldbox, newbox);
    replace_box_in_walk_frame(ctx, oldbox, newbox);
}

#[cfg(test)]
mod frame_replacement_tests {
    use super::*;

    #[test]
    fn descendants_replace_paused_registers_and_resume_boxes() {
        let old = OpRef::input_arg_ref(0);
        let middle = OpRef::input_arg_ref(1);
        let standard = OpRef::input_arg_ref(2);
        let session = std::cell::RefCell::new(WalkSession::default());
        let parent_regs = RegisterBank::new([old]);
        let suspended_regs = RegisterBank::new([old]);
        let make_state = || {
            WalkFrameState::new(WalkFrameStateData {
                callee_shadow: Some(CalleeLocalsShadow {
                    frame_box: old,
                    ..Default::default()
                }),
                outer_active_boxes: vec![old],
                vstack_boxes: vec![old],
                vstack_last_ref: old,
                ..Default::default()
            })
        };
        let parent_state = make_state();
        let suspended_state = make_state();
        let active_state = make_state();
        push_helper_live(
            &session,
            &suspended_regs,
            &RegisterBank::default(),
            &RegisterBank::default(),
            &suspended_state,
        );
        session.borrow_mut().framestack.push(InlineFrame {
            is_portal: false,
            w_code: 1,
            recursion_greenkey: true,
            call_id: 1,
            debug_merge_point_py_pc: None,
            parents: vec![InlineParentFrame {
                jitcode_index: 0,
                call_jitcode_pc: Some(0),
                call_pc: Some(0),
                resume_pc: None,
                call_stack_overrides: Vec::new(),
                blackhole: None,
                resume_coord: ParentResumeCoord::Backxlat(0),
                resume_marker_jit_pc: None,
                boxes: vec![old],
                registers_r: Some(parent_regs.clone()),
                registers_i: None,
                registers_f: None,
                frame_state: Some(parent_state.clone()),
                caller_py_pc: None,
                result_slot: CallResultSlot::Unknown,
            }],
            entry_executed_effects: 0,
            entry_heap_writing_effects: 0,
            entry_journal_mark: fbw_effect_journal_mark(),
            helper_resume: HelperResume::Section,
            helper_call_entry: HelperCallEntry::Unavailable,
            helper_entry_array_len: None,
            entry_trace_index: 0,
            live: None,
        });
        replace_box_in_paused_frames(&mut session.borrow_mut(), old, middle);
        replace_box_in_paused_frames(&mut session.borrow_mut(), middle, standard);
        assert_eq!(parent_regs.get(0), Some(standard));
        assert_eq!(suspended_regs.get(0), Some(standard));
        // The active frame is rewritten directly, never on a later replay
        // where the trace could have reused one of these operation ids.
        assert_eq!(active_state.borrow().vstack_boxes, [old]);
        // Guard capture in the child already sees the rewritten caller.
        assert_eq!(
            session.borrow().framestack.last().unwrap().parents[0].boxes,
            vec![standard]
        );
        // Both frames have already changed before either continuation runs.
        assert_eq!(parent_state.borrow().vstack_boxes, [standard]);
        assert_eq!(suspended_state.borrow().vstack_boxes, [standard]);
        for (regs, state) in [
            (parent_regs, parent_state),
            (suspended_regs, suspended_state),
        ] {
            let mut trace = TraceCtx::for_test_types(&[Type::Ref; 3]);
            let ints = Vec::new();
            let floats = Vec::new();
            let mut concrete_i = Vec::new();
            let mut ctx = WalkContext {
                frame_state: state,
                inline_callee_consts: None,
                fbw_mode: FbwWalkMode::<crate::state::PyreSym>::default(),
                session: &session,
                registers_r: &regs,
                registers_i: &RegisterBank::new(ints.iter().copied()),
                registers_f: &RegisterBank::new(floats.iter().copied()),

                concrete_registers_i: &mut concrete_i,
                descr_refs: &[],
                raw_descrs: RawDescrPool::Global,
                is_authoritative_executor: false,
                trace_ctx: &mut trace,
                is_top_level: false,
                sub_jitcode_lookup: &|_| None,
                entry_py_pc: EntryPyPc::Py(0),
                outer_resume_marker_jit_pc: None,
                outer_jitcode_index: 0,

                pending_guard_snapshot_error: None,

                vstack_depth: 1,
                vstack_cur_pypc: 0,
                vstack_valid: true,

                vstack_reorder_ceiling: u32::MAX,

                vstack_handler_landing_py: None,
                live_before_jit_pc: usize::MAX,
                live_after_jit_pc: usize::MAX,
            };
            assert_eq!(ctx.registers_r.to_vec(), vec![standard]);
            assert_eq!(ctx.frame_state.borrow().outer_active_boxes, vec![standard]);
            assert_eq!(ctx.frame_state.borrow().vstack_boxes, vec![standard]);
            assert_eq!(
                ctx.frame_state
                    .borrow()
                    .callee_shadow
                    .as_ref()
                    .unwrap()
                    .frame_box,
                standard
            );

            // The traceback emitter is another vable writer, outside the
            // bytecode arms. An alias promoted there must rewrite registers
            // before the next writer can replace the single pending slot.
            // Explicit guard promotions use the same full-frame operation as
            // nonstandard virtualizable promotion, not just this Ref bank.
            walker_replace_box(&mut ctx, standard, middle);
            assert_eq!(ctx.frame_state.borrow().vstack_boxes, [middle]);
            assert_eq!(
                session.borrow().framestack.last().unwrap().parents[0].boxes,
                [middle]
            );
            walker_replace_box(&mut ctx, middle, standard);
            ctx.registers_r.set(0, old);
            ctx.frame_state.borrow_mut().vstack_boxes[0] = old;
            let info = crate::frame_layout::build_pyframe_virtualizable_info();
            let initial = ctx.trace_ctx.const_int(0);
            let pointer = Value::Ref(majit_ir::GcRef(0x1000));
            ctx.trace_ctx.set_opref_concrete(old, pointer);
            ctx.trace_ctx.install_virtualizable_info(info.clone());
            ctx.trace_ctx.set_virtualizable_boxes_with_info(
                vec![initial, standard],
                vec![Value::Int(0), pointer],
                info.as_ref(),
                &[],
            );
            let site = TracebackNodeSite {
                frame: old,
                w_code: 0,
                last_instruction: 7,
                w_code_pin: None,
            };
            let tail = ctx.trace_ctx.const_ref(0);
            let guards_before = ctx.trace_ctx.num_guards();
            // ValueError is a `_new_exception` class, so its canonical
            // instance is the slim `_getusercls` layout. The traceback
            // descr has to be that group's, the same choice
            // `record_fresh_application_traceback` makes from `ob_type`.
            let kind = pyre_object::interp_exceptions::ExcKind::ValueError;
            emit_traceback_node(
                &mut ctx,
                middle,
                kind,
                pyre_object::interp_exceptions::exc_kind_canonical_is_user_layout(kind),
                &site,
                tail,
                0,
            )
            .unwrap();
            assert!(ctx.trace_ctx.num_guards() > guards_before);
            assert_eq!(ctx.registers_r.to_vec(), vec![standard]);
            assert_eq!(ctx.frame_state.borrow().vstack_boxes, vec![standard]);
        }
    }

    #[test]
    fn replace_box_writes_parent_banks_on_framestack_without_a_mailbox() {
        let old = OpRef::input_arg_ref(0);
        let new = OpRef::input_arg_ref(1);
        let session = std::cell::RefCell::new(WalkSession::default());
        let parent_regs = RegisterBank::new([old]);
        let parent_state = WalkFrameState::new(WalkFrameStateData {
            vstack_boxes: vec![old],
            vstack_last_ref: old,
            ..Default::default()
        });
        session.borrow_mut().framestack.push(InlineFrame {
            is_portal: false,
            w_code: 1,
            recursion_greenkey: true,
            call_id: 1,
            debug_merge_point_py_pc: None,
            parents: vec![InlineParentFrame {
                jitcode_index: 0,
                call_jitcode_pc: Some(0),
                call_pc: Some(0),
                resume_pc: None,
                call_stack_overrides: Vec::new(),
                blackhole: None,
                resume_coord: ParentResumeCoord::Backxlat(0),
                resume_marker_jit_pc: None,
                boxes: vec![old],
                registers_r: Some(parent_regs.clone()),
                registers_i: None,
                registers_f: None,
                frame_state: Some(parent_state.clone()),
                caller_py_pc: None,
                result_slot: CallResultSlot::Unknown,
            }],
            entry_executed_effects: 0,
            entry_heap_writing_effects: 0,
            entry_journal_mark: fbw_effect_journal_mark(),
            helper_resume: HelperResume::Section,
            helper_call_entry: HelperCallEntry::Unavailable,
            helper_entry_array_len: None,
            entry_trace_index: 0,
            live: None,
        });
        replace_box_in_paused_frames(&mut session.borrow_mut(), old, new);
        assert_eq!(parent_regs.get(0), Some(new));
        assert_eq!(parent_state.borrow().vstack_boxes, [new]);
        assert_eq!(parent_state.borrow().vstack_last_ref, new);
        assert_eq!(
            session.borrow().framestack.last().unwrap().parents[0].boxes,
            vec![new]
        );
    }

    #[test]
    fn replace_box_writes_portal_live_regs_without_a_mailbox() {
        let old = OpRef::input_arg_ref(0);
        let new = OpRef::input_arg_ref(1);
        let session = std::cell::RefCell::new(WalkSession::default());
        let regs = RegisterBank::new([old]);
        let state = WalkFrameState::new(WalkFrameStateData {
            vstack_boxes: vec![old],
            ..Default::default()
        });
        bind_paused_caller_regs(
            &session,
            &regs,
            &RegisterBank::default(),
            &RegisterBank::default(),
            &state,
        );
        assert!(session.borrow().at_portal());
        assert!(session.borrow().framestack[0].is_portal);
        assert!(session.borrow().framestack[0].live.is_some());
        replace_box_in_paused_frames(&mut session.borrow_mut(), old, new);
        assert_eq!(regs.get(0), Some(new));
        assert_eq!(state.borrow().vstack_boxes, [new]);
    }

    #[test]
    fn replace_box_does_not_walk_last_exc_or_tmpreg() {
        // `pyjitpl.py MetaInterp.replace_box` walks framestack only.
        // `last_exc_value` is a MetaInterp concrete pointer, `tmpreg_*`
        // belong to the blackhole interpreter.
        let old = OpRef::input_arg_ref(0);
        let new = OpRef::input_arg_ref(1);
        let session = std::cell::RefCell::new(WalkSession::default());
        {
            let mut session = session.borrow_mut();
            session.last_exc_value = Some(old);
            session.tmpreg_r = old;
            session.tmpreg_i = OpRef::input_arg_int(0);
            session.tmpreg_f = OpRef::input_arg_float(0);
        }
        replace_box_in_paused_frames(&mut session.borrow_mut(), old, new);
        let session = session.borrow();
        assert_eq!(session.last_exc_value, Some(old));
        assert_eq!(session.tmpreg_r, old);
        assert_eq!(session.tmpreg_i, OpRef::input_arg_int(0));
        assert_eq!(session.tmpreg_f, OpRef::input_arg_float(0));
    }

    #[test]
    fn replace_box_writes_helper_live_regs_without_a_mailbox() {
        let old = OpRef::input_arg_ref(0);
        let new = OpRef::input_arg_ref(1);
        let session = std::cell::RefCell::new(WalkSession::default());
        let regs = RegisterBank::new([old]);
        let state = WalkFrameState::new(WalkFrameStateData {
            vstack_boxes: vec![old],
            ..Default::default()
        });
        push_helper_live(
            &session,
            &regs,
            &RegisterBank::default(),
            &RegisterBank::default(),
            &state,
        );
        assert_eq!(session.borrow().helper_live.len(), 1);
        replace_box_in_paused_frames(&mut session.borrow_mut(), old, new);
        assert_eq!(regs.get(0), Some(new));
        assert_eq!(state.borrow().vstack_boxes, [new]);
        pop_helper_live(&session);
        assert!(session.borrow().helper_live.is_empty());
    }

    #[test]
    fn paused_replacement_retains_and_updates_all_typed_banks() {
        let old = [
            OpRef::input_arg_ref(0),
            OpRef::input_arg_int(0),
            OpRef::input_arg_float(0),
        ];
        let new = [
            OpRef::input_arg_ref(1),
            OpRef::input_arg_int(1),
            OpRef::input_arg_float(1),
        ];
        let session = std::cell::RefCell::new(WalkSession::default());
        let banks = old.map(|value| RegisterBank::new([value, value]));
        let state = WalkFrameState::new(WalkFrameStateData::default());
        bind_paused_caller_regs(&session, &banks[0], &banks[1], &banks[2], &state);
        for i in 0..3 {
            replace_box_in_paused_frames(&mut session.borrow_mut(), old[i], new[i]);
            assert_eq!(banks[i].to_vec(), [new[i], new[i]]);
        }
        drop(banks);
        // The portal frame owns the storage, not a pointer into the dropped owner.
        let live = session.borrow().framestack[0]
            .live
            .as_ref()
            .expect("portal live")
            .clone();
        for (bank, (old, new)) in [&live.registers_r, &live.registers_i, &live.registers_f]
            .into_iter()
            .zip(old.into_iter().zip(new))
        {
            replace_box_in_paused_frames(&mut session.borrow_mut(), new, old);
            assert_eq!(bank.to_vec(), [old, old]);
        }
    }
}

/// Resolve the concrete half of a JitCode register operand.
///
/// RPython's MIFrame banks hold Box objects, so assigning a valuebox to
/// `virtualizable_boxes` preserves its concrete value. Pyre splits that Box
/// between an OpRef bank and a parallel concrete bank. Ref values prefer the
/// recorder Box payload because the active-trace root walker forwards it in
/// place across a collection. The frame-owned concrete shadow is now rooted
/// too; retaining this preference preserves the canonical Box payload when
/// both channels are present. Residual results do not necessarily have
/// such a payload, so they fall back to the authoritative walker's shadow.
/// Int/Float values are not GC addresses and keep the shadow-first order.
pub(super) fn vable_value_concrete<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
    bank: char,
    value: OpRef,
) -> Option<Value> {
    let concrete = match bank {
        'i' => read_int_reg_concrete(code, op, operand_offset, ctx),
        'r' => read_ref_reg_concrete(code, op, operand_offset, ctx),
        'f' => read_float_reg_concrete(code, op, operand_offset, ctx),
        _ => unreachable!("value bank must be 'i', 'r' or 'f'"),
    };
    let from_register = match (bank, concrete) {
        ('i', ConcreteValue::Int(value)) => Some(Value::Int(value)),
        ('i', ConcreteValue::Bool(value)) => Some(Value::Int(i64::from(value))),
        ('r', ConcreteValue::Ref(value)) => Some(Value::Ref(majit_ir::GcRef(value as usize))),
        ('f', ConcreteValue::Float(value)) => Some(Value::Float(value)),
        _ => None,
    };
    if bank == 'r' {
        // `pyframe.py popvalue` stores `None`. The assembler patches that
        // src to the constants suffix (`ConstPtr.NULL`). Pyre splits Box
        // into OpRef + concrete, so a leftover live color can still be the
        // encoded `value` while the register concrete is `Ref(0)`.
        // `lookup_opref_concrete` of that leftover would win and stamp a
        // live box against a NULL heap (`check_synchronized_virtualizable`).
        // The store is ConstPtr.NULL; prefer the register / const-null half.
        if value.as_const_ptr().is_some_and(|g| g.is_null())
            || matches!(from_register, Some(Value::Ref(r)) if r.is_null())
        {
            Some(Value::Ref(majit_ir::GcRef::NULL))
        } else {
            ctx.trace_ctx
                .lookup_opref_concrete(value)
                .or_else(|| ctx.trace_ctx.recover_ref_value(value, 8))
                .or(from_register)
        }
    } else {
        from_register.or_else(|| ctx.trace_ctx.concrete_of_opref(value))
    }
}

/// Resolve the concrete value after an operand-stack recovery may have
/// replaced the bytecode register's box.
///
/// The register shadow is authoritative only while `effective_value` is the
/// box encoded by the instruction. Once TOS recovery substitutes another box,
/// re-reading that original color can return a stale concrete value from a
/// prior loop iteration.
pub(super) fn vable_effective_value_concrete<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
    bank: char,
    encoded_value: OpRef,
    effective_value: OpRef,
) -> Option<Value> {
    if effective_value == encoded_value {
        vable_value_concrete(code, op, operand_offset, ctx, bank, effective_value)
    } else {
        ctx.trace_ctx.concrete_of_opref(effective_value)
    }
}

/// Pin a Ref store's concrete across a collecting call (`walker_promote_vable_array_index`
/// / `record_op` Trace-pool append / `set_ref`). The captured `Value` is not a
/// root; `push_roots` / `pin_root` keep it, and `live` re-reads the slot the
/// way `consume_boxes` keeps the decoded GCREF in a traced local.
struct VableStoreRoot {
    _scope: pyre_object::gc_roots::RootScope,
    slot: Option<usize>,
}

impl VableStoreRoot {
    fn capture(captured: Value) -> Self {
        let scope = pyre_object::gc_roots::push_roots();
        let slot = match captured {
            Value::Ref(r) if !r.is_null() && r != majit_ir::GcRef::NO_CONCRETE => {
                let base = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(r.as_usize() as pyre_object::PyObjectRef);
                Some(base)
            }
            _ => None,
        };
        Self {
            _scope: scope,
            slot,
        }
    }

    fn recapture(&mut self, captured: Value) {
        if let Value::Ref(r) = captured
            && !r.is_null()
            && r != majit_ir::GcRef::NO_CONCRETE
        {
            let base = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(r.as_usize() as pyre_object::PyObjectRef);
            self.slot = Some(base);
        }
    }

    fn live(&self, fallback: Value) -> Value {
        match self.slot {
            Some(idx) => {
                let p = pyre_object::gc_roots::shadow_stack_get(idx);
                if (p as usize) == 0 {
                    fallback
                } else {
                    Value::Ref(majit_ir::GcRef(p as usize))
                }
            }
            None => fallback,
        }
    }
}

/// A `Ref` whose OpRef is `NONE` has no box. `_opimpl_getarrayitem_vable`
/// returns the box, so that raw copy is not a shadow entry.
fn ref_copy_without_box(value: OpRef, live: Value) -> bool {
    value.is_none() && matches!(live, Value::Ref(_))
}

/// `publish_fresh_recipe_ref`: a real Ref is stamped on its box before any
/// shadow copy. Constants and `NONE` have no recorder slot; null and
/// `GcRef::NO_CONCRETE` are not objects.
fn stamp_real_ref_on_box(ctx: &mut TraceCtx, value: OpRef, live: Value) {
    if let Value::Ref(gc) = live
        && !value.is_none()
        && !value.is_constant()
        && !gc.is_null()
        && gc != majit_ir::GcRef::NO_CONCRETE
    {
        ctx.try_set_opref_concrete(value, live);
    }
}

/// Word published after `vable_setarrayitem_checked` / `store_live_frame_array_slot`.
///
/// A Ref's root is the [`VableStoreRoot`] pin, which a collection updates in
/// place. Int and Float are not addresses; the captured copy is the value.
fn live_after_vable_store(store_root: &VableStoreRoot, value_bank: char, captured: Value) -> Value {
    if value_bank == 'r' {
        store_root.live(captured)
    } else {
        captured
    }
}

/// Non-fold shadow publish. The box is stamped first. A `NONE` Ref is not
/// written: there is no box to forward, and clearing the slot would drop a
/// fold entry the root walker has already updated.
fn publish_nonfold_slot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    frame_reg: u16,
    slot: i64,
    value: OpRef,
    live: Value,
) {
    stamp_real_ref_on_box(ctx.trace_ctx, value, live);
    // The `borrow_mut` temporary has to live for the whole shadow update.
    if let Some(shadow) = ctx.frame_state.borrow_mut().callee_shadow.as_mut() {
        shadow.set_opref(slot, value);
        if !ref_copy_without_box(value, live) {
            shadow.set_concrete(frame_reg, slot, live);
        }
    }
}

/// `frame` when this walk is a generator resume writing its OWN suspended
/// frame — the one case where a walk-time store lands on a frame the walk did
/// not create.
///
/// Every other frame these stores reach was built by the walk that is writing
/// it, so a decline discards the frame and only `last_instr` (which the caller
/// reads back) needs an undo.  A resumed generator's frame is the object the
/// generator will next be resumed from, so a declined walk has to hand it back
/// byte for byte.
fn durable_resume_frame<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    frame: usize,
) -> Option<usize> {
    let resume = ctx.fbw_mode.generator_resume?;
    (frame != 0 && frame == resume.frame).then_some(frame)
}

/// Heapcache forward of a scalar vable field read off the inlined callee's
/// own seeded virtual frame. The constructor and `setfield_vable_*` install
/// the value under the parent-struct descr `vable_static_record_descr` uses;
/// a hit is the walker's half of `optimize_getfield_gc` on that virtual
/// (`virtualize.py`).
fn seeded_virtual_frame_getfield<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    obj: OpRef,
    descr: &majit_ir::DescrRef,
) -> Option<OpRef> {
    let seeded = ctx
        .frame_state
        .borrow()
        .callee_shadow
        .as_ref()
        .map(|shadow| shadow.frame_box);
    if seeded != Some(obj) {
        return None;
    }
    let field_index = ctx
        .trace_ctx
        .virtualizable_info()
        .and_then(|info| {
            info.static_field_by_descr(descr)
                .map(|idx| info.static_field_struct_descr(idx).index())
        })
        .unwrap_or_else(|| descr.index());
    ctx.trace_ctx.heapcache_getfield_cached(obj, field_index)
}

/// `getfield_vable_<i|r|f>/rd>X` handler. Operand layout `rd>X`:
/// 1B r-reg(vable_box) + 2B descr(field) + 1B X-dst.
///
/// RPython parity: `pyjitpl.py opimpl_getfield_vable_{i,r,f}`:
///
///   def opimpl_getfield_vable_i(self, box, fielddescr, pc):
///       if self._nonstandard_virtualizable(pc, box, fielddescr):
///           return self.opimpl_getfield_gc_i(box, fielddescr)
///       self.metainterp.check_synchronized_virtualizable()
///       index = self._get_virtualizable_field_index(fielddescr)
///       return self.metainterp.virtualizable_boxes[index]
///
/// The walker delegates to the orthodox `TraceCtx::vable_getfield_{int,
/// ref,float}` ports (`majit-metainterp/src/trace_ctx.rs`)
/// which already implement the full
/// `_nonstandard_virtualizable` check + heapcache-aware GETFIELD_GC
/// fallback + `virtualizable_boxes[index]` cache read.  Only the OpRef
/// component of the `(OpRef, Value)` tuple is meaningful here, since
/// register banks carry only OpRefs — the concrete `Value` is tracked
/// via the per-step concrete frame snapshot.  `dst_bank` selects the result bank
/// (`'i'`/`'r'`/`'f'`) the walker writes back into, mirroring
/// `getfield_gc_via_heapcache`'s shape.
pub(crate) fn getfield_vable_via_metainterp<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let obj = read_ref_reg_raw(code, op, 0, ctx)?;
    // Inside an inlined-callee sub-walk, a scalar getfield_vable_r
    // of the callee's namespace(idx5)/pycode(idx1) must resolve to the callee's
    // compile-time `InlineCalleeConsts` whether the callee frame is unseeded
    // (strict path, handled by the `obj.is_none()` branch below) OR a seeded
    // virtual frame (multiframe path). On the multiframe path the read hits the
    // seeded frame box but the heapcache forward MISSES — the codewriter's
    // per-fn vable descr identity differs from the seeding descr — so it would
    // record a non-const `GetfieldGcR`, leaving the LOAD_GLOBAL fold's namespace
    // operand non-concrete (the residual that the nested-unjournaled abort
    // declines, e.g. nbody `advance()`'s `len`). The consts route is
    // descr-identity-independent (`const_ref`) and only intercepts the two
    // static Ref fields (`try_resolve_inline_callee_static_field` returns `None`
    // for everything else), so it is safe to consult on the seeded path too.
    if let Some(resolved) = try_resolve_inline_callee_static_field(code, op, ctx, dst_bank)? {
        return Ok(resolved);
    }
    // RPython's `box` is always a live virtualizable-frame box. An
    // unseeded walker Ref register holds `OpRef::None` (`raw() ==
    // u32::MAX`); feeding it into the metainterp vable path would resize
    // the heapcache flag vector to 16 GiB. Bail to a trace abort instead.
    if obj.is_none() {
        // Path-1 (#68): an inlined callee reading a scalar field off its
        // OWN unseeded portal frame.  A resolvable static field (`w_globals`
        // namespace for a LOAD_GLOBAL, `pycode` promote-to-const) folds to
        // the callee constant; anything else is declined up-front by
        // `callee_fast_path_inlinable`, so an unresolved field here is a
        // genuine unseeded-box error.
        if let Some(resolved) = try_resolve_inline_callee_static_field(code, op, ctx, dst_bank)? {
            return Ok(resolved);
        }
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let descr = read_descr(code, op, 1, ctx)?;
    // R7 parity: RPython `opimpl_getfield_vable_{i,r,f}(box, fielddescr,
    // pc)` threads orgpc through `_nonstandard_virtualizable(pc, ...)`
    // (pyjitpl.py).  Pyre's walker has the matching
    // JitCode PC in `op.pc`; pass it through so the helper signature
    // stays line-by-line equivalent even if `begin_nonstandard_virtualizable`
    // currently ignores the pc at the leaf (`trace_ctx.rs let _ = pc;`).
    let pc = op.pc;
    // Concrete struct pointer for pyjitpl.py cache-hit sanity
    // check.  The walker keeps a parallel concrete Ref-bank shadow;
    // thread the same live pointer that RPython's `box.getref_base()`
    // would expose to `executor.execute(...)`.
    let vable_struct_ptr = match read_ref_reg_concrete(code, op, 0, ctx) {
        ConcreteValue::Ref(ptr) => ptr as i64,
        ConcreteValue::Null => 0,
        ConcreteValue::Int(_) | ConcreteValue::Float(_) | ConcreteValue::Bool(_) => 0,
    };
    // The cpu the read folds through has to be the one the MetaInterp stamps
    // with, or the trace and the optimizer could disagree about the constant;
    // `trace.rs` installs this same `shared()` handle via `set_cpu`.
    let cpu = crate::pyre_cpu::shared();
    // `_opimpl_getfield_vable` returns `virtualizable_boxes[index]` when
    // `standard_box is box`.  The jitcode `VableField` descr may lack a
    // vinfo backref, so `walker_nonstandard_virtualizable` would take
    // `finish_known_nonstandard` and mark the portal frame nonstandard.
    let (result, shadow_value) = with_replace_frames(ctx, |ctx| {
        let nonstandard = if vable_is_standard_virtualizable(ctx, obj) {
            false
        } else {
            walker_nonstandard_virtualizable(ctx, pc, obj, &descr)?
        };
        // Seeded virtual callee frame: fold a scalar field through the heapcache
        // `emit_new_pyframe_inline_with_params` and `setfield_vable_*` install
        // under the parent-struct descr (`vable_static_record_descr`). Twin of
        // `getarrayitem_vable_via_metainterp`'s per-slot OpRef shadow: no GC op,
        // so `last_instr` reads the preceding `setfield_vable_i_imm` and
        // `debugdata` reads the constructor store (PY_NULL, or a New
        // `FrameDebugData` when the callee's namespace is the rare override).
        // `opimpl_getfield_vable_*` (`pyjitpl.py`) reaches that heapcache read
        // through `_nonstandard_virtualizable`; a miss falls through to
        // `vable_getfield_*_checked`.
        if nonstandard && let Some(cached) = seeded_virtual_frame_getfield(ctx, obj, &descr) {
            return Ok((cached, ctx.trace_ctx.concrete_of_opref(cached)));
        }
        Ok(match dst_bank {
            'i' => ctx.trace_ctx.vable_getfield_int_checked(
                nonstandard,
                cpu.as_ref(),
                obj,
                vable_struct_ptr,
                descr.clone(),
            ),
            'r' => ctx.trace_ctx.vable_getfield_ref_checked(
                nonstandard,
                cpu.as_ref(),
                obj,
                vable_struct_ptr,
                descr.clone(),
            ),
            'f' => ctx.trace_ctx.vable_getfield_float_checked(
                nonstandard,
                cpu.as_ref(),
                obj,
                vable_struct_ptr,
                descr.clone(),
            ),
            _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
        })
    })?;
    // RPython `opimpl_getfield_vable_{i,r,f}` returns
    // `virtualizable_boxes[index]` (`pyjitpl.py`) — a Box whose
    // `_resint`/`_resref`/`_resfloat` is filled at construction time.
    // `box.getint()` returns the live value without any side-lookup.
    // Pyre splits OpRef↔concrete into a side table; mirror the Box.value
    // contract by stamping the read result's concrete into
    // `opref_concrete` so `concrete_of_opref(result)` honors the same
    // contract for downstream consumers (`goto_if_not/iL`,
    // `switch/id`, `int_*` arithmetic).  The non-standard heapcache
    // path inside `vable_getfield_int_checked` already does the same stamp;
    // the standard path returns the cached `(opref, value)` pair
    // without stamping.  `None` means no live concrete is available
    // for this slot — skip to match the heapcache path's gating.
    if let Some(shadow_value) = shadow_value {
        ctx.trace_ctx.set_opref_concrete(result, shadow_value);
    }
    let dst = code[op.pc + 4] as usize;
    // concrete_of_opref derivation: derive shadow concrete via `concrete_of_opref`.  The
    // `vable_getfield_*_checked` helpers in `TraceCtx` already populate the
    // concrete shadow for virtualizable-resident fields via the
    // `standard_virtualizable_box()`/`virtualizable_boxes` channel,
    // and feed `set_opref_concrete` on the GETFIELD_GC fallback for
    // non-vable structs — both surface through this lookup.
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'i' => {
            write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'r' => {
            write_vable_field_ref_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'f' => {
            let len = ctx.registers_f.len();
            let _ = ctx
                .registers_f
                .get(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            ctx.registers_f.set(dst, result);
        }
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `setfield_vable_<i|r|f>/r<v>d` handler. Operand layout:
/// 1B r-reg(vable_box) + 1B <v>-reg(value) + 2B descr(field).
/// No dst byte (set, not get).
///
/// RPython parity: `pyjitpl.py _opimpl_setfield_vable`:
///
///   def _opimpl_setfield_vable(self, box, valuebox, fielddescr, pc):
///       if self._nonstandard_virtualizable(pc, box, fielddescr):
///           return self._opimpl_setfield_gc_any(box, valuebox, fielddescr)
///       index = self._get_virtualizable_field_index(fielddescr)
///       self.metainterp.virtualizable_boxes[index] = valuebox
///       self.metainterp.synchronize_virtualizable()
///       # XXX only the index'th field needs to be synchronized, really
///
/// The walker delegates to `walker_nonstandard_virtualizable` then
/// `TraceCtx::vable_setfield_checked` (`majit-metainterp/src/trace_ctx.rs`)
/// which implements the SETFIELD_GC fallback +
/// `virtualizable_boxes[index] = valuebox` write + `synchronize_virtualizable`
/// mirror.  The concrete `Value` is read from the same register's parallel
/// concrete bank (matching the
/// `pyjitpl/dispatch.rs` shape `let (value, concrete) =
/// self.read_<bank>_reg(src); ctx.vable_setfield_checked(...)`).
///
/// `value_bank` selects the value register bank (`'i'`/`'r'`/`'f'`),
/// mirroring `setfield_gc_via_heapcache`'s parameter shape.
fn current_inline_vable_target<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    vable: OpRef,
) -> Option<usize> {
    let inline = current_inline_concrete_frame();
    if inline == 0 {
        return None;
    }
    match ctx
        .trace_ctx
        .lookup_opref_concrete(vable)
        .or_else(|| ctx.trace_ctx.recover_ref_value(vable, 8))
    {
        Some(Value::Ref(value)) if value.as_usize() == inline => Some(inline),
        _ => None,
    }
}

/// The concrete frame whose `locals_cells_stack_w` a `setarrayitem_vable` on
/// `vable` writes.  `_opimpl_setarrayitem_vable` sends a non-standard box
/// through `_opimpl_setarrayitem_gc_any` → `execute_setarrayitem_gc`, and
/// `execute_and_record` performs the store (`executor.py do_setarrayitem_gc`
/// → `cpu.bh_setarrayitem_gc_r`) while recording it, so the array holds the
/// value `heapcache.setarrayitem` publishes and a later
/// `_do_getarrayitem_gc_any` sanity load reads it back.
///
/// The walk's own frame is either the forward-inline callee published by
/// `InlineConcreteFrameGuard`, or, for a bridge-resume level, the frame
/// `setup_reconstructed_callee_frame` built and left in `callee_shadow`
/// without publishing it (see `drive_bridge_frame_subwalk`).  Both are
/// allocated by the walk; the strict-fold arm above already writes the
/// second.
fn own_frame_array_store_target<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    vable: OpRef,
) -> Option<usize> {
    if let Some(frame) = current_inline_vable_target(ctx, vable) {
        return Some(frame);
    }
    let shadow_frame = ctx
        .frame_state
        .borrow()
        .callee_shadow
        .as_ref()
        .map_or(0, |shadow| shadow.concrete_frame);
    if shadow_frame != 0
        && matches!(
            ctx.trace_ctx.lookup_opref_concrete(vable),
            Some(Value::Ref(value)) if value.as_usize() == shadow_frame
        )
    {
        return Some(shadow_frame);
    }
    // `pyframe.py pushvalue` writes `locals_cells_stack_w[depth]`
    // (`_opimpl_setarrayitem_vable` → `synchronize_virtualizable_at` →
    // `write_box_at`). Pyre's `outer_executor_owns_state` skips that
    // write, and Option B `popvalue`/`peekvalue` read the live slot, so
    // a portal push must still store the snapshot the walk flushes on
    // CRN. The identity box is the live frame; the snapshot is
    // `virtualizable_heap_ptr`.
    if vable_is_standard_virtualizable(ctx, vable)
        && let Some(ptr) = ctx.trace_ctx.virtualizable_heap_ptr()
    {
        let addr = ptr as usize;
        if addr != 0 {
            return Some(addr);
        }
    }
    None
}

pub(crate) fn setfield_vable_via_metainterp<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    value_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Strict fresh-frame fold: a scalar setfield to the current inline level's
    // own (unseeded) portal frame is a virtual-field write — `valuestackdepth`
    // / `last_instr` sync on a branchless leaf that resumes at the caller
    // boundary — so it folds to a no-op, emitting no SETFIELD_GC.  The
    // `fresh_virtualizable` OptVirtualize elision (jtransform.py).
    let fold_frame_reg = fbw_strict_fold_frame_reg(ctx);
    if fold_frame_reg != u16::MAX && code[op.pc + 1] as u16 == fold_frame_reg {
        let value = match value_bank {
            'i' => read_int_reg(code, op, 1, ctx)?,
            'r' => read_ref_reg(code, op, 1, ctx)?,
            'f' => read_float_reg(code, op, 1, ctx)?,
            _ => unreachable!("value_bank must be 'i', 'r' or 'f'"),
        };
        let descr = read_descr(code, op, 2, ctx)?;
        let field_index = ctx
            .trace_ctx
            .virtualizable_info()
            .and_then(|info| info.static_field_by_descr(&descr));
        let concrete = vable_value_concrete(code, op, 1, ctx, value_bank, value);
        let shadow_frame = ctx
            .frame_state
            .borrow()
            .callee_shadow
            .as_ref()
            .map(|shadow| shadow.concrete_frame);
        if let (Some(Value::Int(int_val)), Some(field_index), Some(frame)) =
            (concrete, field_index, shadow_frame)
        {
            if let Some(frame) = durable_resume_frame(ctx, frame) {
                fbw_arm_durable_frame_undo(frame);
            }
            crate::state::store_live_frame_static_int(frame, field_index, int_val);
        }
        let vable = read_ref_reg_raw(code, op, 0, ctx)?;
        // Same split as `setarrayitem_vable`: only the standard virtualizable
        // folds the write away. `_opimpl_setfield_vable` records `SETFIELD_GC`
        // for any other box.
        if vable_is_standard_virtualizable(ctx, vable) {
            // `_opimpl_setfield_vable` always writes
            // `virtualizable_boxes[index] = valuebox` (`pyjitpl.py`).  The
            // live-frame store is the heap half; without this box write a
            // later `getfield_vable` (the shared `dispatch_bytecode`
            // `last_instr` / `debugdata` block) reads a stale slot.
            if let (Some(field_index), Some(concrete)) = (field_index, concrete)
                && ctx.trace_ctx.virtualizable_box_at(field_index).is_some()
            {
                ctx.trace_ctx
                    .set_virtualizable_entry_at(field_index, value, concrete);
                ctx.trace_ctx.synchronize_virtualizable_at(field_index);
            }
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }
    let obj = read_ref_reg_raw(code, op, 0, ctx)?;
    // Same unseeded-register guard as `getfield_vable_via_metainterp`:
    // a `None` box would resize the heapcache flag vector to 16 GiB.
    if obj.is_none() {
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let value = match value_bank {
        'i' => read_int_reg(code, op, 1, ctx)?,
        'r' => read_ref_reg(code, op, 1, ctx)?,
        'f' => read_float_reg(code, op, 1, ctx)?,
        _ => unreachable!("value_bank must be 'i', 'r' or 'f'"),
    };
    let descr = read_descr(code, op, 2, ctx)?;
    let concrete = vable_value_concrete(code, op, 1, ctx, value_bank, value);
    let inline_field_index = ctx
        .trace_ctx
        .virtualizable_info()
        .and_then(|info| info.static_field_by_descr(&descr));
    // R7 parity: pyjitpl.py `_opimpl_setfield_vable(box,
    // valuebox, fielddescr, pc)` threads orgpc through
    // `_nonstandard_virtualizable(pc, ...)`; walker has `op.pc` for the
    // JitCode PC, pass through.
    let write = with_replace_frames(ctx, |ctx| {
        let nonstandard = walker_nonstandard_virtualizable(ctx, op.pc, obj, &descr)?;
        Ok(ctx
            .trace_ctx
            .vable_setfield_checked(nonstandard, obj, descr.clone(), value, concrete))
    })?;
    // `MIFrame` owns one red frame per inlined call.  The trace shadow remains
    // authoritative for optimization, while the matching concrete frame is
    // its blackhole-resume image; mirror only own-frame standard-vable writes,
    // never an outer/nonstandard virtualizable.
    if let (Some(frame), Some(Value::Int(value)), Some(field_index)) = (
        current_inline_vable_target(ctx, obj),
        concrete,
        inline_field_index,
    ) {
        if let Some(frame) = durable_resume_frame(ctx, frame) {
            fbw_arm_durable_frame_undo(frame);
        }
        crate::state::store_live_frame_static_int(frame, field_index, value);
    }
    let _ = write;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Flatten's shared `get_is_being_profiled` + `promote_greens` block:
/// `-live-`, `getfield debugdata`, `goto_if_not_ptr_iszero`, then the
/// profiled diamond and `{int,ref}_guard_value` before `switch next_instr`.
pub(crate) fn is_shared_pmg_block(code: &[u8], target: usize) -> bool {
    let Some(live) = crate::jitcode_runtime::decode_op_at(code, target) else {
        return false;
    };
    if live.opname != "live" {
        return false;
    }
    let Some(getfield) = crate::jitcode_runtime::decode_op_at(code, live.next_pc) else {
        return false;
    };
    if getfield.opname != "getfield_vable_r" {
        return false;
    }
    let mut pc = getfield.next_pc;
    if let Some(op) = crate::jitcode_runtime::decode_op_at(code, pc) {
        if op.opname == "live" {
            pc = op.next_pc;
        }
    }
    crate::jitcode_runtime::decode_op_at(code, pc)
        .is_some_and(|op| op.opname == "goto_if_not_ptr_iszero")
}

/// Freeze the entering opcode's banks when `pc` is flatten's shared
/// portal merge-greens copy.
///
/// `pmg_shared` does not store `last_instr` (`interp_jit.py
/// PyFrame.dispatch` greens only). Orgpc for a pmg-internal guard
/// therefore stays on the entering opcode's `live_before` py, not the
/// vable field.
pub(crate) fn begin_hold_if_shared_hook<Sym: WalkSym>(
    code: &[u8],
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
) {
    if !is_shared_pmg_block(code, pc) {
        return;
    }
    ctx.session.borrow_mut().begin_hold_opcode_live_before(
        ctx.registers_i,
        ctx.registers_r,
        ctx.registers_f,
        false, // pmg_shared does not store last_instr
        pc,
    );
}

/// Whether `vable` is the standard virtualizable identity, by Box or by
/// the concrete frame pointer `virtualizable_boxes[-1]` carries.
///
/// `_nonstandard_virtualizable` compares Box identity first
/// (`pyjitpl.py`); a portal frame register that is a copy of that
/// identity still names the same object, and the `dispatch_bytecode`
/// `last_instr` store has to see it.
fn vable_is_standard_virtualizable<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    vable: OpRef,
) -> bool {
    if ctx.trace_ctx.standard_virtualizable_box() == Some(vable) {
        return true;
    }
    let vable_ptr = match ctx
        .trace_ctx
        .lookup_opref_concrete(vable)
        .or_else(|| ctx.trace_ctx.recover_ref_value(vable, 8))
    {
        Some(Value::Ref(ptr)) => ptr.as_usize(),
        _ => return false,
    };
    if vable_ptr == 0 {
        return false;
    }
    if ctx.trace_ctx.standard_virtualizable_ptr() == Some(vable_ptr) {
        return true;
    }
    // Root portal seed points `virtualizable_heap_ptr` at the
    // `snapshot_for_tracing` copy while `virtualizable_boxes[-1]` is the
    // live frame (`standard_virtualizable_ptr`). A SETARRAYITEM whose
    // vable box is that snapshot is still `write_box_at` on the standard
    // virtualizable; treating it as nonstandard skips the box update and
    // Option B `popvalue` then reads a NULL slot.
    matches!(
        ctx.trace_ctx.virtualizable_heap_ptr(),
        Some(heap) if heap as usize == vable_ptr
    )
}

/// `setfield_vable_i_imm/rdd`: `_opimpl_setfield_vable` with a u16 immediate
/// instead of an int register. Same standard-vable box update, no recorded
/// op (`pyjitpl.py` `_opimpl_setfield_vable`).
pub(crate) fn setfield_vable_int_imm<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let imm = (code[op.pc + 2] as u16 | ((code[op.pc + 3] as u16) << 8)) as i64;
    let fold_frame_reg = fbw_strict_fold_frame_reg(ctx);
    if fold_frame_reg != u16::MAX && code[op.pc + 1] as u16 == fold_frame_reg {
        let descr = read_descr(code, op, 3, ctx)?;
        let field_index = ctx
            .trace_ctx
            .virtualizable_info()
            .and_then(|info| info.static_field_by_descr(&descr));
        let shadow_frame = ctx
            .frame_state
            .borrow()
            .callee_shadow
            .as_ref()
            .map(|shadow| shadow.concrete_frame);
        if let (Some(field_index), Some(frame)) = (field_index, shadow_frame) {
            if frame != 0 {
                if let Some(frame) = durable_resume_frame(ctx, frame) {
                    fbw_arm_durable_frame_undo(frame);
                }
                crate::state::store_live_frame_static_int(frame, field_index, imm);
            }
        }
        let obj = read_ref_reg_raw(code, op, 0, ctx)?;
        let is_std = vable_is_standard_virtualizable(ctx, obj);
        let has_box =
            field_index.is_some_and(|index| ctx.trace_ctx.virtualizable_box_at(index).is_some());
        // Same split as `setfield_vable_via_metainterp`: only the
        // standard virtualizable folds the write away.
        // `_opimpl_setfield_vable` always writes
        // `virtualizable_boxes[index] = valuebox`.
        if is_std {
            if let Some(field_index) = field_index
                && has_box
            {
                let imm_box = ctx.trace_ctx.const_int(imm);
                ctx.trace_ctx
                    .set_virtualizable_entry_at(field_index, imm_box, Value::Int(imm));
                ctx.trace_ctx.synchronize_virtualizable_at(field_index);
            }
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        // Identity mismatch: fall through to `vable_setfield_checked`
        // (records SETFIELD_GC for a nonstandard box).
    }
    let obj = read_ref_reg_raw(code, op, 0, ctx)?;
    if obj.is_none() {
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let descr = read_descr(code, op, 3, ctx)?;
    let concrete = Some(Value::Int(imm));
    let inline_field_index = ctx
        .trace_ctx
        .virtualizable_info()
        .and_then(|info| info.static_field_by_descr(&descr));
    let is_std = vable_is_standard_virtualizable(ctx, obj);
    let has_box =
        inline_field_index.is_some_and(|index| ctx.trace_ctx.virtualizable_box_at(index).is_some());
    // Root portal walks have `fold_frame_reg == MAX` (no callee_shadow).
    // `_opimpl_setfield_vable` still writes `virtualizable_boxes[index]
    // = valuebox` for the standard virtualizable, including when the
    // frame register is a copy of `virtualizable_boxes[-1]`.
    if is_std {
        if let Some(field_index) = inline_field_index
            && has_box
        {
            let imm_box = ctx.trace_ctx.const_int(imm);
            ctx.trace_ctx
                .set_virtualizable_entry_at(field_index, imm_box, Value::Int(imm));
            ctx.trace_ctx.synchronize_virtualizable_at(field_index);
        }
        // Portal SETFIELD of `valuestackdepth` / `last_instr` updates
        // `virtualizable_boxes` above; the heap write belongs on an inlined
        // callee's own frame, matching `setfield_vable_via_metainterp`.
        // `standard_virtualizable_ptr` is the live portal frame, and writing
        // it here advances live `valuestackdepth` while `last_instr` stays at
        // function entry. Function-entry CRN then re-enters `LOAD_GLOBAL`
        // with a full stack (`object_array` OOB). Flush writes vsd together
        // with `last_instr` onto the snapshot.
        if let (Some(frame), Some(field_index)) =
            (current_inline_vable_target(ctx, obj), inline_field_index)
        {
            if let Some(frame) = durable_resume_frame(ctx, frame) {
                fbw_arm_durable_frame_undo(frame);
            }
            crate::state::store_live_frame_static_int(frame, field_index, imm);
        }
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    let write = with_replace_frames(ctx, |ctx| {
        let nonstandard = walker_nonstandard_virtualizable(ctx, op.pc, obj, &descr)?;
        Ok(ctx.trace_ctx.vable_setfield_checked(
            nonstandard,
            obj,
            descr.clone(),
            OpRef::ConstInt(imm),
            concrete,
        ))
    })?;
    if let (Some(frame), Some(field_index)) =
        (current_inline_vable_target(ctx, obj), inline_field_index)
    {
        if let Some(frame) = durable_resume_frame(ctx, frame) {
            fbw_arm_durable_frame_undo(frame);
        }
        crate::state::store_live_frame_static_int(frame, field_index, imm);
    }
    let _ = write;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Resolve the `(fdescr, adescr)` virtualizable-array descr pair from a
/// `(VableArray, Array)` jitcode descr-pool pair.
///
/// RPython parity: `MIFrame.vable_array_index_pair_at`
/// (`blackhole.rs`) reads the two 2-byte descr-pool indices,
/// asserts they resolve to `(BhDescr::VableArray { index }, BhDescr::Array
/// { .. })`, and yields the `index`. `vable_array_descrs`
/// (`pyjitpl/dispatch.rs`) then maps that index through the
/// jitdriver's `VirtualizableInfo` to the canonical `array_field_descrs`
/// /`array_descrs` pair.  The pool entry is `VableArray { index }`, not a
/// Field descr; hydrating that variant (`BhDescr::VableArray` in
/// `make_descr_from_bh`) already returns `pyframe_array_field_descr(index)`,
/// which is the vinfo listed row. The lookup still maps the index onto
/// those descrs because `TraceCtx::vable_array_flat_index` keys on
/// `array_field_by_descr` (descr identity, `virtualizable.rs`).
///
/// `field_offset` / `array_offset` are byte offsets (from `op.pc + 1`) of
/// the two descr operands, matching the per-op argcode layout.
pub(crate) fn vable_array_descrs_from_jitcode<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    field_offset: usize,
    array_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<(DescrRef, DescrRef), DispatchError> {
    let read_pool_idx = |off: usize| {
        let lo = code[op.pc + 1 + off] as usize;
        let hi = code[op.pc + 1 + off + 1] as usize;
        lo | (hi << 8)
    };
    let field_idx = read_pool_idx(field_offset);
    let array_pool_idx = read_pool_idx(array_offset);
    // RPython `MIFrame.vable_array_index_pair_at` reads `self.descrs[idx]`
    // — pyre's single per-walk pool, selected by `ctx.raw_descrs`
    // (global `ALL_DESCRS` only for the build-time canonical jitcode a
    // specialization sub-walk inlines, per-`CodeObject` `exec.descrs`
    // otherwise).
    let array_field_index = match (
        ctx.raw_descrs.bh_descr_at(field_idx),
        ctx.raw_descrs.bh_descr_at(array_pool_idx),
    ) {
        (
            Some(majit_jitcode::jitcode::BhDescr::VableArray { index }),
            Some(majit_jitcode::jitcode::BhDescr::Array { .. }),
        ) => *index,
        _ => {
            if std::env::var("PYRE_FBW_INLINE_DIAG").is_ok() {
                eprintln!(
                    "[vable-arr-malformed] pc={} pool_len={:?} field_idx={field_idx} \
                     field={:?} array_idx={array_pool_idx} array={:?}",
                    op.pc,
                    ctx.raw_descrs.len(),
                    ctx.raw_descrs.bh_descr_at(field_idx),
                    ctx.raw_descrs.bh_descr_at(array_pool_idx),
                );
            }
            return Err(DispatchError::VableArrayDescrMalformed {
                pc: op.pc,
                field_idx,
                array_idx: array_pool_idx,
            });
        }
    };
    let info = ctx
        .trace_ctx
        .virtualizable_info()
        .ok_or(DispatchError::VableArrayMissingVirtualizableInfo { pc: op.pc })?;
    let fdescr = info
        .array_field_descrs()
        .get(array_field_index)
        .cloned()
        .ok_or(DispatchError::VableArrayIndexOutOfRange {
            pc: op.pc,
            index: array_field_index,
        })?;
    let adescr = info.array_descrs.get(array_field_index).cloned().ok_or(
        DispatchError::VableArrayIndexOutOfRange {
            pc: op.pc,
            index: array_field_index,
        },
    )?;
    Ok((fdescr, adescr))
}

/// `pyjitpl.py _get_arrayitem_vable_index` opens with
/// `indexbox = self.implement_guard_value(indexbox, pc)`: the flat array slot
/// is derived from the index's recording-time value, so the trace is only
/// sound if it also pins that value.
///
/// The promote is hoisted out of `TraceCtx::get_arrayitem_vable_index` to the
/// walker for the reason `pyjitpl/dispatch.rs` states at its own six
/// `BC_*ARRAYITEM_VABLE_*` sites: promoting at the call site gets the guard a
/// full-framestack, vable-carrying snapshot, where the callee can only build
/// the minimal one `TraceCtx::promote_int` produces without an `MIFrameStack`.
/// The hoist moves the promote to the caller, not ahead of the branch that
/// selects it: `_get_arrayitem_vable_index` is reached only from the standard
/// leg of `_opimpl_get|setarrayitem_vable` (pyjitpl.py, :1244), so both
/// call sites run `TraceCtx::nonstandard_virtualizable` first and skip this
/// promote on the non-standard leg.
///
/// Mirrors `guard_value_record` (`arith.rs`) step for step, including the
/// `replace_box` and register-bank rewrite. The callee's promote discards its
/// promoted box, so nothing downstream would otherwise see the pinned value.
/// `pyjitpl.py MIFrame.implement_guard_value`: record `GUARD_VALUE` and
/// capture the live framestack at the promote (`generate_guard` →
/// `capture_resumedata`).
pub(crate) fn walker_implement_guard_value<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    box_: OpRef,
    runtime_value: i64,
) -> Result<OpRef, DispatchError> {
    if box_.is_constant() {
        return Ok(box_);
    }
    let expected = ctx.trace_ctx.const_int(runtime_value);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[box_, expected], 0);
    // Full-body walks install `snapshot_sym` so `generate_guard` can
    // capture the live framestack. Unit fixtures leave it null; skip
    // the capture rather than abort the op.
    if !ctx.fbw_mode.snapshot_sym.is_null() {
        walker_capture_snapshot_for_last_guard(ctx, pc)?;
    }
    walker_replace_box(ctx, box_, expected);
    Ok(expected)
}

fn walker_promote_vable_array_index<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    index: OpRef,
    index_value: i64,
) -> Result<OpRef, DispatchError> {
    walker_implement_guard_value(ctx, pc, index, index_value)
}

/// `pyjitpl.py MIFrame._nonstandard_virtualizable`: promote the
/// `isstandard` `PTR_EQ` through [`walker_implement_guard_value`] so the
/// snapshot is complete when the guard is recorded.
pub(crate) fn walker_nonstandard_virtualizable<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    vable: OpRef,
    fielddescr: &majit_ir::DescrRef,
) -> Result<bool, DispatchError> {
    match ctx
        .trace_ctx
        .begin_nonstandard_virtualizable(pc, vable, fielddescr)
    {
        majit_metainterp::NonstandardVable::Decided(nonstandard) => Ok(nonstandard),
        majit_metainterp::NonstandardVable::PendingEq {
            eqbox,
            isstandard,
            vable_opref,
            standard_box,
        } => {
            let promoted = walker_implement_guard_value(ctx, pc, eqbox, isstandard)?;
            Ok(ctx.trace_ctx.commit_nonstandard_virtualizable(
                promoted,
                vable_opref,
                standard_box,
                fielddescr,
            ))
        }
    }
}

/// One live-frame slot read on the `Void` getarrayitem fallback.
enum LiveFrameSlot {
    /// The array word is a non-null reference.
    Present(Value),
    /// The frame was readable and the slot word is NULL.
    Null,
    /// No frame, the index is outside the local region, or the array
    /// could not be read.
    Unavailable,
}

/// `getarrayitem_vable_<i|r|f>/ridd>X` handler. Operand layout `ridd>X`:
/// 1B r-reg(vable) + 1B i-reg(index) + 2B fdescr(VableArray) + 2B
/// adescr(Array) + 1B X-dst.
///
/// RPython parity: `pyjitpl.py _opimpl_getarrayitem_vable`
/// (`opimpl_getarrayitem_vable_{i,r,f}`).  Delegates to
/// `TraceCtx::vable_getarrayitem_{int,ref,float}_checked`
/// (`trace_ctx.rs`) which implements the
/// `_nonstandard_virtualizable` GETFIELD_GC + GETARRAYITEM_GC fallback
/// and the standard-vable `virtualizable_boxes[index]` cache read.
/// Mirrors `getfield_vable_via_metainterp`'s concrete-stamp + dst-write
/// shape; the trait counterpart is `pyjitpl/dispatch.rs`.
pub(crate) fn getarrayitem_vable_via_metainterp<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Strict fresh-frame fold: when this op reads the current inline level's
    // own portal frame, resolve it register-to-register through the
    // per-slot OpRef shadow and emit NO GC op — the `fresh_virtualizable` case
    // (jtransform.py `is_virtualizable_getset` returns `False`).  The
    // param-store seed dominates every read of a branchless leaf's own frame,
    // so the slot is present; a genuinely-unbound read falls through to the
    // `VableBoxNotSeeded` abort below (never the 16 GiB metainterp path).
    // The per-slot OpRef shadow is `MIFrame.registers_r`. Each inlined
    // call is its own MIFrame (`pyjitpl.py` `newframe`); a callee
    // `getarrayitem_vable` reads that frame's boxes, never the portal's.
    // `frame_materialized` governs STORES (`folded_store_is_observable_local`);
    // a LOAD_FAST still answers from the same map the level's own bytecode
    // writes (`try_walker_specialize_builtin_locals_in_callee`).
    let fold_frame_reg = fbw_strict_fold_frame_reg(ctx);
    let names_callee_frame = fold_frame_reg != u16::MAX && code[op.pc + 1] as u16 == fold_frame_reg;
    if names_callee_frame {
        let index = read_int_reg(code, op, 1, ctx)?;
        if let Some(majit_ir::Value::Int(slot)) = ctx.trace_ctx.concrete_of_opref(index) {
            let result = ctx
                .frame_state
                .borrow()
                .callee_shadow
                .as_ref()
                .and_then(|shadow| shadow.opref.get(&slot).copied());
            if let Some(result) = result {
                // `OpRef::NONE` is an unbound hole: fall through.
                // `ConstPtr(0)` is this frame's value (`PyFrame.__init__`
                // `locals_cells_stack_w = [None] * size` / `PUSH_NULL`).
                if result != OpRef::NONE {
                    let dst = code[op.pc + 7] as usize;
                    let concrete = concrete_from_recorded_opref(ctx, result);
                    match dst_bank {
                        'i' => write_int_reg(ctx, op.pc, dst, result, concrete)?,
                        'r' => {
                            write_ref_reg(ctx, op.pc, dst, result, concrete)?;
                        }
                        'f' => {
                            let len = ctx.registers_f.len();
                            let _ = ctx.registers_f.get(dst).ok_or(
                                DispatchError::RegisterOutOfRange {
                                    pc: op.pc,
                                    reg: dst,
                                    len,
                                    bank: "f",
                                },
                            )?;
                            ctx.registers_f.set(dst, result);
                        }
                        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
                    }
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }
    let vable = read_ref_reg_raw(code, op, 0, ctx)?;
    // An unseeded walker Ref register holds `OpRef::None` (`raw() ==
    // u32::MAX`); feeding it into the metainterp vable path would resize
    // the heapcache flag vector to 16 GiB. Bail to a trace abort, mirroring
    // the scalar `getfield_vable_via_metainterp` guard.
    if vable.is_none() {
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let index = read_int_reg(code, op, 1, ctx)?;
    // pyjitpl.py `indexbox.getint()` — the array slot is chosen from
    // the concrete index. Fail loud if the walker can't resolve it.
    let index_value = match ctx.trace_ctx.concrete_of_opref(index) {
        Some(Value::Int(v)) => v,
        _ => {
            return Err(DispatchError::VableArrayIndexNotConcrete {
                pc: op.pc,
                value: index,
            });
        }
    };
    let (fdescr, adescr) = vable_array_descrs_from_jitcode(code, op, 2, 4, ctx)?;
    // Upstream decides standardness before promoting the index: an ordinary
    // heap access on the non-standard leg must retain the index box as-is.
    let nonstandard = with_replace_frames(ctx, |ctx| {
        walker_nonstandard_virtualizable(ctx, op.pc, vable, &fdescr)
    })?;
    let index = if nonstandard {
        index
    } else {
        walker_promote_vable_array_index(ctx, op.pc, index, index_value)?
    };
    let (result, shadow_value) = match dst_bank {
        'i' => ctx.trace_ctx.vable_getarrayitem_int_checked(
            nonstandard,
            op.pc,
            vable,
            index,
            index_value,
            fdescr,
            adescr,
        ),
        'r' => ctx.trace_ctx.vable_getarrayitem_ref_checked(
            nonstandard,
            op.pc,
            vable,
            index,
            index_value,
            fdescr,
            adescr,
        ),
        'f' => ctx.trace_ctx.vable_getarrayitem_float_checked(
            nonstandard,
            op.pc,
            vable,
            index,
            index_value,
            fdescr,
            adescr,
        ),
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    };
    let shadow_value = shadow_value.unwrap_or(Value::Void);
    // When the read missed every concrete channel (`Void`) but we are inside
    // an inlined callee, fall back to the per-frame concrete-locals shadow —
    // a loop-carried local read after a may-force op in the loop body has no
    // heapcache entry this pass, yet its recording-time concrete is known.
    let shadow_value = if matches!(shadow_value, Value::Void) {
        let frame_reg = code[op.pc + 1] as u16;
        let (entry, slot_opref, live_frame) = {
            let state = ctx.frame_state.borrow();
            let shadow = state.callee_shadow.as_ref();
            let entry = shadow
                .and_then(|shadow| shadow.concrete.get(&index_value).copied())
                .filter(|entry| entry.frame_reg == frame_reg);
            let slot_opref = shadow.and_then(|shadow| shadow.opref.get(&index_value).copied());
            // `do_residual_call` step 3 cleared `fold_frame_reg` after
            // `materialize_published_callee_locals` or after
            // `vrefs_after_residual_call` stopped this frame's vref. The
            // shadow OpRef is still the pre-call value
            // (`optimize_VIRTUAL_REF_FINISH` forced that virtual before
            // `direct_call_may_force`); the live array is what the residual
            // wrote, and this iteration must return that word. The vable
            // operand has to be this shadow's frame: a read of another
            // frame must not stamp this level's live slot.
            let live_frame = shadow.and_then(|shadow| {
                (shadow.frame_materialized
                    && shadow.fold_frame_reg == u16::MAX
                    && shadow.concrete_frame != 0
                    && shadow.frame_box == vable)
                    .then_some(shadow.concrete_frame)
            });
            (entry, slot_opref, live_frame)
        };
        // `_opimpl_getarrayitem_vable` returns
        // `metainterp.virtualizable_boxes[index]`: the slot's current box is
        // the value. A NULL word read from the live frame is that box. No
        // frame, an index outside the local region, or an unreadable array
        // leaves the shadow fallback in place.
        let live = match live_frame {
            Some(frame) => match active_frame_nlocals(ctx) {
                Some(nlocals) if index_value >= 0 && index_value < nlocals => {
                    match crate::state::concrete_stack_value(frame, index_value as usize) {
                        Some(raw) if raw.is_null() => LiveFrameSlot::Null,
                        Some(raw) => {
                            LiveFrameSlot::Present(Value::Ref(majit_ir::GcRef(raw as usize)))
                        }
                        None => LiveFrameSlot::Unavailable,
                    }
                }
                _ => LiveFrameSlot::Unavailable,
            },
            None => LiveFrameSlot::Unavailable,
        };
        match live {
            LiveFrameSlot::Present(live) => live,
            LiveFrameSlot::Null => Value::Ref(majit_ir::GcRef(0)),
            LiveFrameSlot::Unavailable => {
                // `_opimpl_getarrayitem_vable` returns
                // `metainterp.virtualizable_boxes[index]`. The box owns the
                // value; a moving GC updates it because the box stays
                // reachable. A `Ref` is therefore taken only from
                // `concrete_of_opref(slot_opref)`. `entry.value` is a raw
                // copy and is returned only for `Int` and `Float`. A `Ref`
                // with no box concrete is `Void`. The OpRef is consulted
                // only when a frame-matched `concrete` entry exists, so the
                // `frame_reg` filter still bounds the lookup.
                entry
                    .and_then(|_| {
                        slot_opref.and_then(|opref| ctx.trace_ctx.concrete_of_opref(opref))
                    })
                    .or_else(|| {
                        entry.and_then(|entry| match entry.value {
                            value @ (Value::Int(_) | Value::Float(_)) => Some(value),
                            _ => None,
                        })
                    })
                    .unwrap_or(Value::Void)
            }
        }
    } else {
        shadow_value
    };
    // Fold-frame slots live in `callee_shadow`, including `ConstPtr(0)`
    // from `PyFrame.__init__` / `PUSH_NULL`. Portal `virtualizable_boxes`
    // at the same numeric index is a different frame.
    let (result, shadow_value) = if dst_bank == 'r' && names_callee_frame && index_value >= 0 {
        let slot_opref = ctx
            .frame_state
            .borrow()
            .callee_shadow
            .as_ref()
            .and_then(|shadow| shadow.opref.get(&index_value).copied());
        match slot_opref {
            Some(shadow_op) if shadow_op != OpRef::NONE => {
                let conc = ctx
                    .trace_ctx
                    .concrete_of_opref(shadow_op)
                    .unwrap_or_else(|| {
                        if opref_is_null_const_ptr(shadow_op) {
                            Value::Ref(majit_ir::GcRef(0))
                        } else {
                            shadow_value
                        }
                    });
                (shadow_op, conc)
            }
            _ => (result, shadow_value),
        }
    } else {
        (result, shadow_value)
    };
    // Mirror `getfield_vable_via_metainterp`: stamp the read result's
    // concrete so `concrete_of_opref(result)` honors the Box.value
    // contract for downstream consumers; `Value::Void` = no live concrete.
    //
    // An indexed read hands back the STORED box, so the stamp is a write onto a
    // box that may already carry a live concrete. A frame whose array slot is
    // still an unmaterialized hole reads NULL while the real Ref lives in the
    // guard's register file, so stamping that NULL would clobber the live value
    // (last write wins) and a downstream residual would consume the hole. Skip
    // ONLY that clobber: a NULL read onto a box with no live Ref still stamps,
    // keeping NULL a real concrete for the slots that genuinely hold one.
    let clobbers_live_ref = matches!(shadow_value, Value::Ref(majit_ir::GcRef(0)))
        && matches!(
            ctx.trace_ctx.box_value(result),
            Some(Value::Ref(majit_ir::GcRef(addr))) if addr != 0
        );
    if !matches!(shadow_value, Value::Void) && !clobbers_live_ref {
        ctx.trace_ctx.set_opref_concrete(result, shadow_value);
    }
    let dst = code[op.pc + 7] as usize;
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'i' => write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?,
        'r' => {
            write_ref_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'f' => {
            let len = ctx.registers_f.len();
            let _ = ctx
                .registers_f
                .get(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            ctx.registers_f.set(dst, result);
        }
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `setarrayitem_vable_<i|r|f>/riXdd` handler. Operand layout `riXdd`:
/// 1B r-reg(vable) + 1B i-reg(index) + 1B X-reg(value) + 2B
/// fdescr(VableArray) + 2B adescr(Array). No dst byte.
///
/// A folded store lands in a slot whose value survives the trace: the inline
/// level materialized a real callee `PyFrame` and `slot` is in its LOCAL
/// region.  Such a slot is readable after the call through a traceback,
/// `f_locals` or `sys._getframe`, so the store must be recorded rather than
/// mirrored.  Slots at or above `nlocals` are operand-stack cells on a
/// virtual frame: those live in the same shadow (`MIFrame.registers_r`).
/// A materialized frame records the store so `f_locals` / traceback /
/// `popvalue` read the live array.
fn folded_store_is_observable_local<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    slot: i64,
) -> bool {
    if !callee_frame_is_materialized(ctx) {
        return false;
    }
    let Some(nlocals) = active_frame_nlocals(ctx) else {
        return false;
    };
    slot >= 0 && slot < nlocals
}

fn callee_frame_is_materialized<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>) -> bool {
    ctx.frame_state
        .borrow()
        .callee_shadow
        .as_ref()
        .is_some_and(|shadow| shadow.frame_materialized)
}

/// `CodeObject` of the frame a vable op names — the callee's inside an inline
/// sub-walk that owns a [`CalleeLocalsShadow`], the outer portal frame's
/// otherwise.
///
/// Shadow ownership, not `fbw_mode.inline_subwalk`, is the discriminant: the
/// flag is also set for a canonical-helper descent (`run_sub_jitcode_walk`)
/// and for a recursive root closure driven through the same entry, neither of
/// which resolves a callee shadow.  Those walk the frame the outer sym already
/// describes, so keying off the flag would leave them with no frame at all.
fn active_frame_code<'a, Sym: WalkSym>(
    ctx: &'a WalkContext<'_, '_, Sym>,
) -> Option<&'a pyre_interpreter::CodeObject> {
    // A resolved shadow is authoritative: falling back to the outer sym when
    // its `code_ptr` is unset would answer with the CALLER's frame, the exact
    // misattribution `active_frame_nlocals` exists to prevent.
    let code_ptr = if let Some(shadow) = ctx.frame_state.borrow().callee_shadow.as_ref() {
        shadow.code_ptr
    } else {
        let sym = ctx.fbw_mode.snapshot_sym;
        if sym.is_null() {
            return None;
        }
        // SAFETY: sym and its jitcode are live for the full-body walk; read-only.
        unsafe {
            let jitcode = (*sym).jitcode();
            if jitcode.is_null() {
                return None;
            }
            let jitcode = &*jitcode;
            jitcode.payload.code_ptr
        }
    };
    // SAFETY: the code object outlives the walk that resolved it; read-only.
    unsafe { code_ptr.as_ref() }
}

/// Local-region size of the frame a `getarrayitem_vable_*` /
/// `setarrayitem_vable_*` slot indexes.
///
/// The op names its own frame, so the split between the
/// `locals_cells_stack_w` prefix and the operand-stack region is that frame's
/// own count.  `fbw_mode.snapshot_sym` describes the outermost portal frame;
/// reading the count off it inside a sub-walk misreads every callee slot at or
/// above the CALLER's count as operand stack.  For `_find_ti(self, dt, i)`
/// (6 slots) inlined into `utcoffset(self, dt)` (2), that folded away the
/// `STORE_FAST` of every local from slot 2 up, and a guard resuming inside the
/// callee then read them back as NULL from the reconstructed frame — the next
/// call in the callee arrived one positional argument short.
fn active_frame_nlocals<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>) -> Option<i64> {
    if ctx.frame_state.borrow().callee_shadow.is_some() {
        return active_frame_code(ctx)
            .map(|code| crate::state::callee_layout_for_call_assembler(code).0 as i64);
    }
    // No callee shadow: the walk names the outer portal frame, and the sym is
    // the authority for it even where its jitcode carries no code object (a
    // fixture install), which the code-derived count could not answer.
    let sym = ctx.fbw_mode.snapshot_sym;
    if sym.is_null() {
        return None;
    }
    // SAFETY: pointer live for the full-body walk; read-only.
    Some(unsafe { (*sym).nlocals() } as i64)
}

/// Stamp `vstack_last_ref` for a Ref store into `locals_cells_stack_w`
/// at `index_value` (`pyframe.py pushvalue`).
///
/// The standard-vable strict-fold arm returns after
/// `vable_setarrayitem_checked` without reaching the SETARRAYITEM_GC tail,
/// so GET_ITER's iterator push never set `vstack_last_ref` and ResultToTos
/// left a NONE hole that later FOR_ITER last_ref (the yielded int) filled.
///
/// After every stack-slot SET the live mirror names the same box as
/// `virtualizable_boxes[NUM_VABLE_SCALARS + slot]` (`pyjitpl.py`
/// `_opimpl_setarrayitem_vable`). Capture reads `vstack_opcode_entry`,
/// cloned at the Python-opcode boundary, so a mid-opcode `popvalue` None
/// does not become the opcode-start re-pop (`unary_negative`).
///
/// The operand-stack mirror (`vstack_boxes`) is gated on `vstack_valid` and
/// `index >= nlocals`. `virtualizable_boxes` is written only by
/// `vable_setarrayitem_checked` (`pyjitpl.py _opimpl_setarrayitem_vable`).
/// A second `set_virtualizable_entry_at` here restamped a leftover live
/// Const over a ConstPtr.NULL heap write (`check_synchronized_virtualizable`).
fn stamp_vstack_from_array_store<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    index_value: i64,
    value: OpRef,
    value_bank: char,
) {
    if value_bank == 'r' && ctx.vstack_valid {
        if let Some(nlocals) = active_frame_nlocals(ctx)
            && index_value >= nlocals
        {
            let stack_slot = (index_value - nlocals) as usize;
            let mut st = ctx.frame_state.borrow_mut();
            if let Some((_, _, _, mask)) = st.vstack_reorder_saved.as_mut() {
                if mask.len() <= stack_slot {
                    mask.resize(stack_slot + 1, false);
                }
                mask[stack_slot] = true;
            }
            if st.vstack_boxes.len() <= stack_slot {
                st.vstack_boxes.resize(stack_slot + 1, OpRef::NONE);
            }
            st.vstack_boxes[stack_slot] = value;
            st.vstack_last_ref = value;
            drop(st);
        }
    }
}

/// `pyframe.py pushvalue` after `make_result_of_lastop`.
///
/// A CALL_ASSEMBLER fold records GUARD_NOT_FORCED / GUARD_NO_EXCEPTION
/// before the walk reaches the opcode's `setarrayitem_vable` of dest.
/// BINARY GET reads `virtualizable_boxes` / `callee_shadow`
/// (`_opimpl_getarrayitem_vable`), not only `vstack_boxes`. Stamp dest
/// at TOS the way `_opimpl_setarrayitem_vable` does. GUARD_NOT_FORCED
/// must already have captured: dest lives in FRAME until this push.
pub(crate) fn stamp_pushvalue_result<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    dest: OpRef,
    dest_bank: char,
    resume_depth: usize,
) {
    if dest_bank != 'r' || dest == OpRef::NONE {
        return;
    }
    let resume_depth = resume_depth.max(1);
    let Some(stack_base) = active_frame_nlocals(ctx) else {
        return;
    };
    let slot = stack_base + (resume_depth as i64 - 1);
    if slot < 0 {
        return;
    }
    let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
    let vable_index = nvs.saturating_add(slot as usize);
    let concrete = ctx
        .trace_ctx
        .concrete_of_opref(dest)
        .or_else(|| ctx.trace_ctx.lookup_opref_concrete(dest))
        .unwrap_or(Value::Void);
    if ctx.trace_ctx.virtualizable_box_at(vable_index).is_some() {
        ctx.trace_ctx
            .set_virtualizable_entry_at(vable_index, dest, concrete);
        ctx.trace_ctx.synchronize_virtualizable_at(vable_index);
        if matches!(concrete, Value::Ref(_)) {
            ctx.trace_ctx
                .write_virtualizable_heap_value_at(vable_index, concrete);
        }
    }
    let fold_frame_reg = fbw_strict_fold_frame_reg(ctx);
    if let Some(shadow) = ctx.frame_state.borrow_mut().callee_shadow.as_mut() {
        shadow.set_opref(slot, dest);
        if fold_frame_reg != u16::MAX {
            shadow.set_concrete(fold_frame_reg, slot, concrete);
        }
    }
    stamp_vstack_from_array_store(ctx, slot, dest, dest_bank);
    let vsd = stack_base + resume_depth as i64;
    let vsd_op = ctx.trace_ctx.const_int(vsd);
    crate::trace_opcode::mirror_vable_static_to_boxes(
        ctx.trace_ctx,
        "valuestackdepth",
        vsd_op,
        Value::Int(vsd),
    );
}

/// `pyframe.py popvalue` stores `None`. The assembler patches that src to
/// the constants suffix (`vable_setarrayitem_ref_const_value`). A leftover
/// live color with `concrete = Ref(0)` is still that store: the Box is
/// `ConstPtr(NULL)`, not the leftover producer (`try_set` is a no-op on
/// Const). A pending CallMayForce dest is unstamped and is not this path.
fn canonical_ref_array_store<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    value: OpRef,
    concrete: Value,
) -> (OpRef, Value) {
    if matches!(concrete, Value::Ref(r) if r.is_null()) {
        (
            ctx.trace_ctx.const_null(),
            Value::Ref(majit_ir::GcRef::NULL),
        )
    } else {
        (value, concrete)
    }
}

/// RPython parity: `pyjitpl.py _opimpl_setarrayitem_vable`.
/// Delegates to `TraceCtx::vable_setarrayitem_checked`
/// (`trace_ctx.rs`) which implements the `_nonstandard_virtualizable`
/// SETARRAYITEM_GC fallback + the standard-vable
/// `virtualizable_boxes[index] = valuebox` + `synchronize_virtualizable`.
/// Trait counterpart: `pyjitpl/dispatch.rs`.
pub(crate) fn setarrayitem_vable_via_metainterp<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    value_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Strict fresh-frame fold (twin of `getarrayitem_vable_via_metainterp`): a
    // write to the current inline level's own portal frame updates
    // the per-slot OpRef + concrete shadow and emits NO SETARRAYITEM_GC for
    // unmaterialized locals and operand-stack pushes. A materialized
    // LOCAL store falls through so `pyframe.py` / traceback / `f_locals`
    // read the live array (`folded_store_is_observable_local`).
    let fold_frame_reg = fbw_strict_fold_frame_reg(ctx);
    if fold_frame_reg != u16::MAX && code[op.pc + 1] as u16 == fold_frame_reg {
        let index = read_int_reg(code, op, 1, ctx)?;
        if let Some(majit_ir::Value::Int(slot)) = ctx.trace_ctx.concrete_of_opref(index) {
            if !folded_store_is_observable_local(ctx, slot) {
                let value = match value_bank {
                    'i' => read_int_reg(code, op, 2, ctx)?,
                    'r' => read_ref_reg(code, op, 2, ctx)?,
                    'f' => read_float_reg(code, op, 2, ctx)?,
                    _ => unreachable!("value_bank must be 'i', 'r' or 'f'"),
                };
                let captured = vable_value_concrete(code, op, 2, ctx, value_bank, value)
                    .unwrap_or(majit_ir::Value::Void);
                // `set_ref` can collect. Pin the captured GCREF first and store
                // the live word (`store_reconstructed_callee_array_image`).
                let store_root = VableStoreRoot::capture(captured);
                let (value, concrete) = if value_bank == 'r' {
                    canonical_ref_array_store(ctx, value, store_root.live(captured))
                } else {
                    (value, captured)
                };
                // Stamp the box before the shadow copy, and before
                // `store_live_frame_array_slot`. A non-standard vable falls
                // through and publishes again after those collecting calls; the
                // box already carries this word, so that arm re-reads the pin
                // instead of putting the pre-call copy back.
                stamp_real_ref_on_box(ctx.trace_ctx, value, concrete);
                if let Some(shadow) = ctx.frame_state.borrow_mut().callee_shadow.as_mut() {
                    shadow.set_opref(slot, value);
                    shadow.set_concrete(fold_frame_reg, slot, concrete);
                }
                // `locals_cells_stack_w` is a `FixedObjectArray` of boxed refs, so a
                // folded store should only ever carry one.  `store_live_frame_array_slot`
                // already ignores anything else, so this stays a debug assertion rather
                // than a release panic on a path the corpus does not exercise.
                debug_assert!(
                    matches!(concrete, majit_ir::Value::Ref(_) | majit_ir::Value::Void),
                    "folded locals_cells_stack_w store must carry a Ref-compatible value"
                );
                if matches!(concrete, majit_ir::Value::Ref(_)) {
                    // Drop the frame-state borrow before the store. `set_ref`
                    // can collect, and the collector walks this state.
                    let concrete_frame = ctx
                        .frame_state
                        .borrow()
                        .callee_shadow
                        .as_ref()
                        .map(|shadow| shadow.concrete_frame);
                    if let Some(concrete_frame) = concrete_frame {
                        if let Some(frame) = durable_resume_frame(ctx, concrete_frame) {
                            fbw_arm_durable_frame_undo(frame);
                            fbw_note_generator_stack_store(GeneratorStackStore {
                                frame,
                                slot: slot as usize,
                                value,
                            });
                        }
                        crate::state::store_live_frame_array_slot(
                            concrete_frame,
                            slot as usize,
                            store_root.live(concrete),
                        );
                    }
                }
                let vable = read_ref_reg_raw(code, op, 0, ctx)?;
                // `_opimpl_setarrayitem_vable` records `SETARRAYITEM_GC` for every
                // box that is not `virtualizable_boxes[-1]`. The strict fold is
                // that standard arm: no GC op, but still
                // `virtualizable_boxes[index] = valuebox` +
                // `synchronize_virtualizable_at` (`pyjitpl.py`).  Shadow +
                // live-frame stores above are the heap half; without the box
                // write, `jit_merge_point` force writes the stale array over
                // those slots and the next instruction's `getarrayitem_vable`
                // reads NULL (`comparison on null operand`).
                if vable_is_standard_virtualizable(ctx, vable) {
                    let (fdescr, adescr) = vable_array_descrs_from_jitcode(code, op, 3, 5, ctx)?;
                    match ctx.trace_ctx.vable_setarrayitem_checked(
                        false, op.pc, vable, index, slot, fdescr, adescr, value, concrete,
                    ) {
                        VableArrayStore::Stored(_) => {
                            stamp_vstack_from_array_store(ctx, slot, value, value_bank);
                            return Ok((DispatchOutcome::Continue, op.next_pc));
                        }
                        // Array length unknown: do not swallow the store.
                        // Fall through so `SETARRAYITEM_GC` still records it.
                        VableArrayStore::OutOfVable => {}
                    }
                }
            }
        }
    }
    let vable = read_ref_reg_raw(code, op, 0, ctx)?;
    // See `getarrayitem_vable_via_metainterp`: an unseeded `OpRef::None`
    // vable would resize the heapcache flag vector to 16 GiB; bail instead.
    if vable.is_none() {
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let index = read_int_reg(code, op, 1, ctx)?;
    let index_value = match ctx.trace_ctx.concrete_of_opref(index) {
        Some(Value::Int(v)) => v,
        _ => {
            return Err(DispatchError::VableArrayIndexNotConcrete {
                pc: op.pc,
                value: index,
            });
        }
    };
    let encoded_value = match value_bank {
        'i' => read_int_reg(code, op, 2, ctx)?,
        'r' => read_ref_reg(code, op, 2, ctx)?,
        'f' => read_float_reg(code, op, 2, ctx)?,
        _ => unreachable!("value_bank must be 'i', 'r' or 'f'"),
    };
    // `walker_promote_vable_array_index` records a guard and can append to a
    // nursery-backed Trace pool. Pin the value first so the store below
    // reloads a live GCREF (`write_box_at` / `vable_write_array_item_at`).
    // `try_set_opref_concrete` skips Const OpRefs and InputArg holes.
    let encoded_captured =
        vable_value_concrete(code, op, 2, ctx, value_bank, encoded_value).unwrap_or(Value::Void);
    let mut store_root = VableStoreRoot::capture(encoded_captured);
    // TOS recovery runs after `walker_promote_vable_array_index`. Pin that
    // box's concrete now; recapturing the stamp after the promote is a
    // from-space copy (`fwd=false free=true`). Only STORE_FAST (a pop into
    // a local) recovers TOS. A stack-slot ConstPtr.NULL store must not
    // recapture the previous TOS into the cleared cell.
    if encoded_value.is_none()
        && value_bank == 'r'
        && ctx.vstack_valid
        && let Some(nlocals) = active_frame_nlocals(ctx)
        && index_value >= 0
        && index_value < nlocals
        && let Some(&tos) = ctx.frame_state.borrow().vstack_boxes.last()
        && !tos.is_none()
    {
        if let Some(tos_c) = ctx
            .trace_ctx
            .lookup_opref_concrete(tos)
            .or_else(|| ctx.trace_ctx.concrete_of_opref(tos))
        {
            store_root.recapture(tos_c);
        }
    }
    let (fdescr, adescr) = vable_array_descrs_from_jitcode(code, op, 3, 5, ctx)?;
    // As in the read path, only the standard virtualizable leg promotes the
    // array index and needs the full walker-owned resume snapshot.
    let nonstandard = with_replace_frames(ctx, |ctx| {
        walker_nonstandard_virtualizable(ctx, op.pc, vable, &fdescr)
    })?;
    let index = if nonstandard {
        index
    } else {
        walker_promote_vable_array_index(ctx, op.pc, index, index_value)?
    };
    let mut value = encoded_value;
    // A STORE_FAST (`index < nlocals`) pops the operand-stack TOS and writes
    // it into a local slot.  When the popped value lives in an operand-stack
    // temp that spans a nested loop, the loop-header merge-point seeds no
    // operand-stack color (`trace.rs` loop-header entry), so the codewriter's
    // value register reads back `OpRef::NONE` even though the box is live.
    // The walk-level operand-stack mirror (`vstack_boxes`, the analog of
    // `MIFrame.registers_r`) is the authoritative kept-stack source and was
    // maintained across the nested loop; recover the TOS box from it.  Writing
    // the unbound `NONE` into the vable array otherwise leaves an untyped slot
    // that fails the guard-snapshot buildability precondition
    // (`GuardSnapshotVableUntyped`).  Gate on a live mirror + a Ref store into
    // the local region so a genuine null / non-mirrored slot keeps the legacy
    // read.
    if value.is_none()
        && value_bank == 'r'
        && ctx.vstack_valid
        && let Some(nlocals) = active_frame_nlocals(ctx)
        && index_value >= 0
        && index_value < nlocals
        && let Some(&tos) = ctx.frame_state.borrow().vstack_boxes.last()
        && !tos.is_none()
    {
        value = tos;
    }
    let captured =
        vable_effective_value_concrete(code, op, 2, ctx, value_bank, encoded_value, value)
            .unwrap_or(Value::Void);
    // Pre-call word passed into the recording call. The shadow is published
    // from a later [`VableStoreRoot::live`] read: this local is not a root.
    let (value, concrete) = if value_bank == 'r' {
        canonical_ref_array_store(ctx, value, store_root.live(captured))
    } else {
        (value, captured)
    };
    let _write = match ctx.trace_ctx.vable_setarrayitem_checked(
        nonstandard,
        op.pc,
        vable,
        index,
        index_value,
        fdescr,
        adescr,
        value,
        concrete,
    ) {
        VableArrayStore::Stored(write) => write,
        // The out-of-vable store recorded nothing, so there is no pre-store
        // entry to roll back. Whether it should abort the trace is tracked
        // separately.
        VableArrayStore::OutOfVable => None,
    };
    if index_value >= 0
        && let Some(frame) = own_frame_array_store_target(ctx, vable)
    {
        if let Some(frame) = durable_resume_frame(ctx, frame) {
            fbw_arm_durable_frame_undo(frame);
            fbw_note_generator_stack_store(GeneratorStackStore {
                frame,
                slot: index_value as usize,
                value,
            });
        }
        // Re-read after `vable_setarrayitem_checked`. The array store may
        // itself collect, so the shadow reads the pin again below.
        crate::state::store_live_frame_array_slot(
            frame,
            index_value as usize,
            live_after_vable_store(&store_root, value_bank, concrete),
        );
        // Only this level's own frame: a store to a foreign vable must not
        // bind in `locals()` through this shadow. The published word is the
        // pin after both calls. A `NONE` Ref has no box and is not stored.
        let live = live_after_vable_store(&store_root, value_bank, concrete);
        publish_nonfold_slot(ctx, code[op.pc + 1] as u16, index_value, value, live);
    } else if index_value >= 0 {
        // The heap array was not this level's concrete frame (virtual, or
        // the store targeted a copy of the frame box). `popvalue` still
        // has to recover the push, so keep the per-slot OpRef shadow.
        let own_box = ctx
            .frame_state
            .borrow()
            .callee_shadow
            .as_ref()
            .is_some_and(|shadow| !shadow.frame_box.is_none() && shadow.frame_box == vable);
        if own_box {
            let live = live_after_vable_store(&store_root, value_bank, concrete);
            publish_nonfold_slot(ctx, code[op.pc + 1] as u16, index_value, value, live);
        }
    }
    let _ = _write;
    // A Ref stored to the operand-stack region of the vable array is an
    // operand-stack push (`pyframe.pushvalue` lowers to
    // `setarrayitem_vable_r(locals_cells_stack_w, depth, w_obj)`). The
    // standard-vable fold returns earlier; this tail is the GC / nonstandard
    // store. `stamp_vstack_from_array_store` writes the same box into the
    // walk-level mirror so both stores agree.
    stamp_vstack_from_array_store(ctx, index_value, value, value_bank);
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `arraylen_vable/rdd>i` handler. Operand layout `rdd>i`: 1B r-reg(vable)
/// + 2B fdescr(VableArray) + 2B adescr(Array) + 1B i-dst.
///
/// RPython parity: `pyjitpl.py opimpl_arraylen_vable`.
/// Delegates to `TraceCtx::vable_arraylen_vable` (`trace_ctx.rs`)
/// which implements the `_nonstandard_virtualizable` GETFIELD_GC +
/// ARRAYLEN_GC fallback and the standard-vable
/// `ConstInt(get_array_length(...))` read.  Trait counterpart:
/// `pyjitpl/dispatch.rs`.
pub(crate) fn arraylen_vable_via_metainterp<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let vable = read_ref_reg_raw(code, op, 0, ctx)?;
    // See `getarrayitem_vable_via_metainterp`: an unseeded `OpRef::None`
    // vable would resize the heapcache flag vector to 16 GiB; bail instead.
    if vable.is_none() {
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let vable_struct_ptr = match read_ref_reg_concrete(code, op, 0, ctx) {
        ConcreteValue::Ref(ptr) => ptr as i64,
        ConcreteValue::Null
        | ConcreteValue::Int(_)
        | ConcreteValue::Float(_)
        | ConcreteValue::Bool(_) => 0,
    };
    let (fdescr, adescr) = vable_array_descrs_from_jitcode(code, op, 1, 3, ctx)?;
    let cpu = crate::pyre_cpu::shared();
    let result = with_replace_frames(ctx, |ctx| {
        let nonstandard = walker_nonstandard_virtualizable(ctx, op.pc, vable, &fdescr)?;
        Ok(ctx.trace_ctx.vable_arraylen_vable_checked(
            nonstandard,
            cpu.as_ref(),
            vable,
            vable_struct_ptr,
            fdescr.clone(),
            adescr,
        ))
    })?;
    let dst = code[op.pc + 6] as usize;
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

#[cfg(test)]
mod callee_ref_owner_tests {
    use super::*;

    fn no_subs(_idx: usize) -> Option<SubJitCodeBody> {
        None
    }

    fn shadow_slot<Sym: WalkSym>(wc: &WalkContext<'_, '_, Sym>, slot: i64) -> Option<Value> {
        wc.frame_state
            .borrow()
            .callee_shadow
            .as_ref()
            .and_then(|shadow| shadow.concrete.get(&slot).map(|entry| entry.value))
    }

    fn shadow_op<Sym: WalkSym>(wc: &WalkContext<'_, '_, Sym>, slot: i64) -> Option<OpRef> {
        wc.frame_state
            .borrow()
            .callee_shadow
            .as_ref()
            .and_then(|shadow| shadow.opref.get(&slot).copied())
    }

    fn with_ctx<R>(
        tc: &mut TraceCtx,
        shadow: Option<CalleeLocalsShadow>,
        regs_r: &[OpRef],
        regs_i: &[OpRef],
        concrete_r: Vec<ConcreteValue>,
        raw_pool: &[majit_metainterp::jitcode::RuntimeBhDescr],
        body: impl FnOnce(&mut WalkContext<'_, '_, crate::state::PyreSym>) -> R,
    ) -> R {
        let session = std::cell::RefCell::new(WalkSession::default());
        let regs_r = RegisterBank::new(regs_r.iter().copied());
        let regs_i = RegisterBank::new(regs_i.iter().copied());
        let regs_f = RegisterBank::default();
        let descr_pool: Vec<DescrRef> = Vec::new();
        let mut concrete_i = [];
        let mut wc = WalkContext {
            frame_state: WalkFrameState::new(WalkFrameStateData {
                callee_shadow: shadow,
                concrete_registers_r: concrete_r,
                outer_active_boxes: Vec::new(),
                vstack_boxes: Vec::new(),
                vstack_last_ref: OpRef::NONE,
                vstack_reorder_saved: None,
                ..Default::default()
            }),
            inline_callee_consts: None,
            fbw_mode: FbwWalkMode::default(),
            session: &session,
            registers_r: &regs_r,
            registers_i: &regs_i,
            registers_f: &regs_f,
            concrete_registers_i: &mut concrete_i,
            descr_refs: &descr_pool,
            raw_descrs: RawDescrPool::PerFn(raw_pool),
            is_authoritative_executor: false,
            trace_ctx: tc,
            is_top_level: true,
            sub_jitcode_lookup: &no_subs,
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
        body(&mut wc)
    }

    fn array_pool(
        info: &majit_metainterp::virtualizable::VirtualizableInfo,
    ) -> Vec<majit_metainterp::jitcode::RuntimeBhDescr> {
        let bh_array = majit_jitcode::jitcode::BhDescr::from_array_descr(
            info.array_descrs[0]
                .as_array_descr()
                .expect("the fixture's array descr"),
        );
        vec![
            majit_metainterp::jitcode::RuntimeBhDescr::Descr(Box::new(
                majit_jitcode::jitcode::BhDescr::VableArray { index: 0 },
            )),
            majit_metainterp::jitcode::RuntimeBhDescr::Descr(Box::new(bh_array)),
        ]
    }

    fn install_portal(
        tc: &mut TraceCtx,
    ) -> std::sync::Arc<majit_metainterp::virtualizable::VirtualizableInfo> {
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        let array_len = 3;
        let slot_count = info.num_static_extra_boxes + array_len;
        let standard = tc.const_ref(1);
        let hole = tc.record_op(majit_ir::OpCode::NewWithVtable, &[]);
        let initial_boxes = vec![hole; slot_count];
        tc.install_virtualizable_info(info.clone());
        tc.init_virtualizable_boxes(
            &info,
            standard,
            Value::Ref(majit_ir::GcRef(1)),
            &initial_boxes,
            &[],
            &[array_len],
        );
        info
    }

    fn one_local_frame() -> (pyre_interpreter::pyframe::FrameBox, usize) {
        let function =
            pyre_interpreter::compile_exec("def f(x):\n    return x\n").expect("compile");
        let code = function
            .constants
            .iter()
            .find_map(|constant| match constant {
                pyre_interpreter::ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("function code");
        let mut frame = pyre_interpreter::pyframe::PyFrame::new(code);
        let frame_ptr = (&mut *frame) as *mut pyre_interpreter::pyframe::PyFrame as usize;
        (frame, frame_ptr)
    }

    /// `read_retired_callee_local` and `null_live_vable_slot_reads_null_not_the_shadow`
    /// still seed `OpRef::const_ptr` of the same address as the shadow. Their
    /// assert is satisfied by `inline_const_to_value` and does not observe a
    /// box-less `Ref` copy. This test is that case: the shadow holds `Ref(X)`
    /// and the slot box has no concrete.
    #[test]
    fn unavailable_ref_comes_only_from_the_box() {
        let mut tc = TraceCtx::for_test_types(&[Type::Ref]);
        let info = install_portal(&mut tc);
        let raw_pool = array_pool(&info);
        let vable = OpRef::input_arg_ref(0);
        tc.set_opref_concrete(vable, Value::Ref(majit_ir::GcRef(2)));
        let index = tc.const_int(0);
        let slot_box = tc.record_op(majit_ir::OpCode::NewWithVtable, &[]);
        assert!(
            !slot_box.is_constant() && !slot_box.is_none(),
            "the slot box must be a real OpRef"
        );
        assert_eq!(tc.concrete_of_opref(slot_box), None);
        let stale = Value::Ref(majit_ir::GcRef(pyre_object::w_none() as usize));
        let mut shadow = CalleeLocalsShadow {
            fold_frame_reg: u16::MAX,
            concrete_frame: 0,
            frame_box: vable,
            frame_materialized: true,
            ..Default::default()
        };
        shadow.set_opref(0, slot_box);
        shadow.set_concrete(0, 0, stale);
        // `ridd>r`: r-reg(0) + i-reg(1) + fdescr(0) + adescr(1) + dst(2).
        let code: [u8; 8] = [0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x02];
        let op = DecodedOp {
            key: "getarrayitem_vable_r/ridd>r",
            opname: "getarrayitem_vable_r",
            argcodes: "ridd>r",
            pc: 0,
            next_pc: code.len(),
        };
        let regs_r = [vable, OpRef::NONE, OpRef::NONE];
        let regs_i = [index, index, OpRef::NONE];
        let read = with_ctx(
            &mut tc,
            Some(shadow),
            &regs_r,
            &regs_i,
            vec![ConcreteValue::Null; 3],
            &raw_pool,
            |wc| {
                let outcome =
                    getarrayitem_vable_via_metainterp(&code, &op, wc, 'r').map(|(out, _)| out);
                let dst = wc.registers_r.get(2);
                let concrete = dst.and_then(|opref| wc.trace_ctx.concrete_of_opref(opref));
                let kept = shadow_slot(wc, 0);
                (outcome, concrete, kept)
            },
        );
        assert_eq!(read.0, Ok(DispatchOutcome::Continue));
        assert_eq!(
            read.1, None,
            "a Ref copy is not the value when the box has no concrete"
        );
        assert_eq!(read.2, Some(stale), "the read does not rewrite the shadow");

        let forwarded = Value::Ref(majit_ir::GcRef(pyre_object::w_bool_from(true) as usize));
        tc.set_opref_concrete(slot_box, forwarded);
        let read = with_ctx(
            &mut tc,
            {
                let mut shadow = CalleeLocalsShadow {
                    fold_frame_reg: u16::MAX,
                    concrete_frame: 0,
                    frame_box: vable,
                    frame_materialized: true,
                    ..Default::default()
                };
                shadow.set_opref(0, slot_box);
                shadow.set_concrete(0, 0, stale);
                Some(shadow)
            },
            &regs_r,
            &regs_i,
            vec![ConcreteValue::Null; 3],
            &raw_pool,
            |wc| {
                let outcome =
                    getarrayitem_vable_via_metainterp(&code, &op, wc, 'r').map(|(out, _)| out);
                let concrete = wc
                    .registers_r
                    .get(2)
                    .and_then(|opref| wc.trace_ctx.concrete_of_opref(opref));
                (outcome, concrete, shadow_slot(wc, 0))
            },
        );
        assert_eq!(read.0, Ok(DispatchOutcome::Continue));
        assert_eq!(read.1, Some(forwarded));
        assert_eq!(read.2, Some(stale));

        // Int copies are not addresses. An unstamped slot box still returns
        // the shadow integer.
        let mut shadow = CalleeLocalsShadow {
            fold_frame_reg: u16::MAX,
            concrete_frame: 0,
            frame_box: vable,
            frame_materialized: true,
            ..Default::default()
        };
        shadow.set_opref(0, slot_box);
        shadow.set_concrete(0, 0, Value::Int(42));
        // The previous stamp would answer the box first. Clear it by using a
        // fresh unstamped hole.
        let int_box = tc.record_op(majit_ir::OpCode::NewWithVtable, &[]);
        assert_eq!(tc.concrete_of_opref(int_box), None);
        shadow.set_opref(0, int_box);
        let read = with_ctx(
            &mut tc,
            Some(shadow),
            &regs_r,
            &regs_i,
            vec![ConcreteValue::Null; 3],
            &raw_pool,
            |wc| {
                let outcome =
                    getarrayitem_vable_via_metainterp(&code, &op, wc, 'i').map(|(out, _)| out);
                let concrete = wc
                    .registers_i
                    .get(2)
                    .and_then(|opref| wc.trace_ctx.concrete_of_opref(opref));
                (outcome, concrete)
            },
        );
        assert_eq!(read.0, Ok(DispatchOutcome::Continue));
        assert_eq!(read.1, Some(Value::Int(42)));
    }

    #[test]
    fn fold_setarrayitem_stamps_the_unstamped_box() {
        let (_frame, frame_ptr) = one_local_frame();
        let mut tc = TraceCtx::for_test_types(&[Type::Ref, Type::Ref]);
        let info = install_portal(&mut tc);
        let standard = tc
            .standard_virtualizable_box()
            .expect("init_virtualizable_boxes pushes the identity");
        let raw_pool = array_pool(&info);
        let value = OpRef::input_arg_ref(1);
        assert_eq!(tc.concrete_of_opref(value), None);
        let none = Value::Ref(majit_ir::GcRef(pyre_object::w_none() as usize));
        let index = tc.const_int(0);
        let shadow = CalleeLocalsShadow {
            fold_frame_reg: 0,
            concrete_frame: frame_ptr,
            frame_box: standard,
            frame_materialized: false,
            ..Default::default()
        };
        // `riXdd`: opcode, vable r0, index i1, value r2, fdescr 0, adescr 1.
        let code: [u8; 8] = [0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x01, 0x00];
        let op = DecodedOp {
            key: "setarrayitem_vable_r/rirdd",
            opname: "setarrayitem_vable_r",
            argcodes: "rirdd",
            pc: 0,
            next_pc: code.len(),
        };
        let regs_r = [standard, OpRef::NONE, value];
        let regs_i = [index, index];
        let mut concrete_r = vec![ConcreteValue::Null; 3];
        concrete_r[2] = ConcreteValue::Ref(pyre_object::w_none());
        with_ctx(
            &mut tc,
            Some(shadow),
            &regs_r,
            &regs_i,
            concrete_r,
            &raw_pool,
            |wc| {
                let outcome =
                    setarrayitem_vable_via_metainterp(&code, &op, wc, 'r').map(|(out, _)| out);
                assert_eq!(outcome, Ok(DispatchOutcome::Continue));
                assert_eq!(shadow_slot(wc, 0), Some(none));
                assert_eq!(shadow_op(wc, 0), Some(value));
            },
        );
        assert_eq!(
            tc.concrete_of_opref(value),
            Some(none),
            "the fold arm stamps the box before set_concrete"
        );
    }

    #[test]
    fn nonfold_setarrayitem_stamps_the_box_with_the_live_ref() {
        let (_frame, frame_ptr) = one_local_frame();
        let mut tc = TraceCtx::for_test_types(&[Type::Ref, Type::Ref]);
        let info = install_portal(&mut tc);
        let raw_pool = array_pool(&info);
        let vable = OpRef::input_arg_ref(0);
        let value = OpRef::input_arg_ref(1);
        tc.set_opref_concrete(vable, Value::Ref(majit_ir::GcRef(frame_ptr)));
        assert_eq!(tc.concrete_of_opref(value), None);
        let none = Value::Ref(majit_ir::GcRef(pyre_object::w_none() as usize));
        let index = tc.const_int(0);
        let shadow = CalleeLocalsShadow {
            fold_frame_reg: u16::MAX,
            concrete_frame: frame_ptr,
            frame_box: vable,
            frame_materialized: false,
            ..Default::default()
        };
        let code: [u8; 8] = [0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x01, 0x00];
        let op = DecodedOp {
            key: "setarrayitem_vable_r/rirdd",
            opname: "setarrayitem_vable_r",
            argcodes: "rirdd",
            pc: 0,
            next_pc: code.len(),
        };
        let regs_r = [vable, OpRef::NONE, value];
        let regs_i = [index, index];
        let mut concrete_r = vec![ConcreteValue::Null; 3];
        concrete_r[2] = ConcreteValue::Ref(pyre_object::w_none());
        with_ctx(
            &mut tc,
            Some(shadow),
            &regs_r,
            &regs_i,
            concrete_r,
            &raw_pool,
            |wc| {
                let outcome =
                    setarrayitem_vable_via_metainterp(&code, &op, wc, 'r').map(|(out, _)| out);
                assert_eq!(outcome, Ok(DispatchOutcome::Continue), "{outcome:?}");
                assert_eq!(shadow_slot(wc, 0), Some(none));
                assert_eq!(shadow_op(wc, 0), Some(value));
            },
        );
        assert_eq!(tc.concrete_of_opref(value), Some(none));
        assert_eq!(
            crate::state::concrete_stack_value(frame_ptr, 0).map(|raw| raw as usize),
            Some(pyre_object::w_none() as usize)
        );
    }

    #[test]
    fn nonfold_setarrayitem_keeps_an_int_copy() {
        let (_frame, frame_ptr) = one_local_frame();
        let mut tc = TraceCtx::for_test_types(&[Type::Ref]);
        let info = install_portal(&mut tc);
        let raw_pool = array_pool(&info);
        let vable = OpRef::input_arg_ref(0);
        tc.set_opref_concrete(vable, Value::Ref(majit_ir::GcRef(frame_ptr)));
        let index = tc.const_int(0);
        let stored = tc.const_int(11);
        let shadow = CalleeLocalsShadow {
            fold_frame_reg: u16::MAX,
            concrete_frame: frame_ptr,
            frame_box: vable,
            frame_materialized: false,
            ..Default::default()
        };
        let code: [u8; 8] = [0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x01, 0x00];
        let op = DecodedOp {
            key: "setarrayitem_vable_i/riidd",
            opname: "setarrayitem_vable_i",
            argcodes: "riidd",
            pc: 0,
            next_pc: code.len(),
        };
        let regs_r = [vable];
        let regs_i = [index, index, stored];
        with_ctx(
            &mut tc,
            Some(shadow),
            &regs_r,
            &regs_i,
            vec![ConcreteValue::Null; 1],
            &raw_pool,
            |wc| {
                let outcome =
                    setarrayitem_vable_via_metainterp(&code, &op, wc, 'i').map(|(out, _)| out);
                assert_eq!(outcome, Ok(DispatchOutcome::Continue), "{outcome:?}");
                assert_eq!(shadow_slot(wc, 0), Some(Value::Int(11)));
                assert_eq!(shadow_op(wc, 0), Some(stored));
            },
        );
    }

    /// No existing harness moves a `VableStoreRoot` pin inside
    /// `vable_setarrayitem_checked` or `store_live_frame_array_slot`.
    /// `TraceCtx::for_test_types` records into a host buffer, and `set_ref`'s
    /// write barrier is not a collection point (`call_pure_r_all_const_folded_const_is_the_forwarded_box`).
    /// The collector updates the pin in place. This test does that update,
    /// then publishes through `live_after_vable_store` and
    /// `publish_nonfold_slot`, which is the non-fold writer's post-call commit.
    #[test]
    fn publish_nonfold_slot_uses_the_reread_pin_and_skips_a_ref_without_a_box() {
        let from = pyre_object::w_none();
        let to = pyre_object::w_bool_from(true);
        let from_v = Value::Ref(majit_ir::GcRef(from as usize));
        let to_v = Value::Ref(majit_ir::GcRef(to as usize));
        let root = VableStoreRoot::capture(from_v);
        let pre = root.live(from_v);
        assert_eq!(pre, from_v);
        let slot = root.slot.expect("a real Ref is pinned");
        pyre_object::gc_roots::shadow_stack_set(slot, to);
        let live = live_after_vable_store(&root, 'r', pre);
        assert_eq!(live, to_v);
        assert_ne!(live, pre);
        assert_eq!(
            live_after_vable_store(&root, 'i', Value::Int(4)),
            Value::Int(4)
        );

        let mut tc = TraceCtx::for_test_types(&[Type::Ref, Type::Ref]);
        let value = OpRef::input_arg_ref(1);
        tc.set_opref_concrete(value, from_v);
        let mut shadow = CalleeLocalsShadow::default();
        shadow.set_opref(0, OpRef::input_arg_ref(0));
        shadow.set_concrete(7, 0, Value::Int(5));
        let raw_pool: Vec<majit_metainterp::jitcode::RuntimeBhDescr> = Vec::new();
        with_ctx(
            &mut tc,
            Some(shadow),
            &[],
            &[],
            Vec::new(),
            &raw_pool,
            |wc| {
                publish_nonfold_slot(wc, 3, 1, value, live);
                assert_eq!(shadow_slot(wc, 1), Some(to_v));
                assert_eq!(shadow_op(wc, 1), Some(value));
                // `NONE` + `Ref` has no box. Do not store it, and do not
                // clear the integer already in the slot.
                publish_nonfold_slot(wc, 3, 0, OpRef::NONE, from_v);
                assert_eq!(shadow_op(wc, 0), None);
                assert_eq!(shadow_slot(wc, 0), Some(Value::Int(5)));
                publish_nonfold_slot(wc, 3, 2, OpRef::NONE, Value::Int(9));
                assert_eq!(shadow_slot(wc, 2), Some(Value::Int(9)));
                publish_nonfold_slot(wc, 3, 2, OpRef::NONE, Value::Float(1.5));
                assert_eq!(shadow_slot(wc, 2), Some(Value::Float(1.5)));
            },
        );
        assert_eq!(tc.concrete_of_opref(value), Some(to_v));
    }
}
