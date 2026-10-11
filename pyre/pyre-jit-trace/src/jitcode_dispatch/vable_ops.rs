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
        for parent in &mut frame.parents {
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
            }],
            entry_executed_effects: 0,
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
            }],
            entry_executed_effects: 0,
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
        ctx.trace_ctx
            .lookup_opref_concrete(value)
            .or_else(|| ctx.trace_ctx.recover_ref_value(value, 8))
            .or(from_register)
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

/// A push writes at the current `valuestackdepth`; a pop/clear decrements its
/// target index first and updates `valuestackdepth` afterward.  This ordering is
/// emitted by `emit_pushvalue_ref_const!` / `emit_popvalue_ref!` and lets the
/// generic vable-store handler distinguish a live NULL push from clearing a
/// dead slot without weakening flush validation. pyjitpl.py
/// `virtualizable_boxes[index] = valuebox` stores one Box carrying both identity
/// and concrete value, so it needs no separate marker; pyre keeps the halves
/// apart and marks the shadow slot beside the trace instead.
fn vable_store_is_stack_push<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>, index: i64) -> bool {
    let depth_index = ctx
        .trace_ctx
        .virtualizable_info()
        .and_then(|info| info.static_field_index_by_name("valuestackdepth"));
    depth_index
        .and_then(|idx| ctx.trace_ctx.virtualizable_entry_at(idx))
        .is_some_and(|(_, value)| matches!(value, Value::Int(depth) if depth == index))
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
    let (result, shadow_value) = with_replace_frames(ctx, |ctx| {
        let nonstandard = walker_nonstandard_virtualizable(ctx, pc, obj, &descr)?;
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
    let frame = ctx
        .frame_state
        .borrow()
        .callee_shadow
        .as_ref()
        .map_or(0, |shadow| shadow.concrete_frame);
    if frame == 0 {
        return None;
    }
    match ctx.trace_ctx.lookup_opref_concrete(vable) {
        Some(Value::Ref(value)) if value.as_usize() == frame => Some(frame),
        _ => None,
    }
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
        if let (Some(Value::Int(value)), Some(field_index), Some(shadow)) = (
            vable_value_concrete(code, op, 1, ctx, value_bank, value),
            ctx.trace_ctx
                .virtualizable_info()
                .and_then(|info| info.static_field_by_descr(&descr)),
            ctx.frame_state.borrow().callee_shadow.as_ref(),
        ) {
            if let Some(frame) = durable_resume_frame(ctx, shadow.concrete_frame) {
                fbw_arm_durable_frame_undo(frame);
            }
            crate::state::store_live_frame_static_int(shadow.concrete_frame, field_index, value);
        }
        let vable = read_ref_reg_raw(code, op, 0, ctx)?;
        // Same split as `setarrayitem_vable`: only the standard virtualizable
        // folds the write away. `_opimpl_setfield_vable` records `SETFIELD_GC`
        // for any other box.
        if ctx.trace_ctx.standard_virtualizable_box() == Some(vable) {
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

/// `setfield_vable_i_imm/rddd`: `_opimpl_setfield_vable` with a u32 immediate
/// instead of an int register. Same standard-vable box update, no recorded
/// op (`pyjitpl.py` `_opimpl_setfield_vable`).
pub(crate) fn setfield_vable_int_imm<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let lo = code[op.pc + 2] as u32 | ((code[op.pc + 3] as u32) << 8);
    let hi = code[op.pc + 4] as u32 | ((code[op.pc + 5] as u32) << 8);
    let imm = (lo | (hi << 16)) as i64;
    let fold_frame_reg = fbw_strict_fold_frame_reg(ctx);
    if fold_frame_reg != u16::MAX && code[op.pc + 1] as u16 == fold_frame_reg {
        let descr = read_descr(code, op, 5, ctx)?;
        if let (Some(field_index), Some(shadow)) = (
            ctx.trace_ctx
                .virtualizable_info()
                .and_then(|info| info.static_field_by_descr(&descr)),
            ctx.frame_state.borrow().callee_shadow.as_ref(),
        ) {
            if let Some(frame) = durable_resume_frame(ctx, shadow.concrete_frame) {
                fbw_arm_durable_frame_undo(frame);
            }
            crate::state::store_live_frame_static_int(shadow.concrete_frame, field_index, imm);
        }
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    let obj = read_ref_reg_raw(code, op, 0, ctx)?;
    if obj.is_none() {
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let descr = read_descr(code, op, 5, ctx)?;
    let concrete = Some(Value::Int(imm));
    let inline_field_index = ctx
        .trace_ctx
        .virtualizable_info()
        .and_then(|info| info.static_field_by_descr(&descr));
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
/// /`array_descrs` pair.  The walker must use the *vinfo* descrs (not the
/// raw jitcode-pool descrs) because `TraceCtx::vable_array_flat_index`
/// keys on `array_field_by_descr` (descr identity, `virtualizable.rs`).
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
fn walker_implement_guard_value<Sym: WalkSym>(
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
    // own (unseeded) portal frame, resolve it register-to-register through the
    // per-slot OpRef shadow and emit NO GC op — the `fresh_virtualizable` case
    // (jtransform.py `is_virtualizable_getset` returns `False`).  The
    // param-store seed dominates every read of a branchless leaf's own frame,
    // so the slot is present; a genuinely-unbound read falls through to the
    // `VableBoxNotSeeded` abort below (never the 16 GiB metainterp path).
    let fold_frame_reg = fbw_strict_fold_frame_reg(ctx);
    if fold_frame_reg != u16::MAX && code[op.pc + 1] as u16 == fold_frame_reg {
        let index = read_int_reg(code, op, 1, ctx)?;
        if let Some(majit_ir::Value::Int(slot)) = ctx.trace_ctx.concrete_of_opref(index) {
            let result = ctx
                .frame_state
                .borrow()
                .callee_shadow
                .as_ref()
                .and_then(|shadow| shadow.opref.get(&slot).copied());
            if let Some(result) = result {
                let dst = code[op.pc + 7] as usize;
                let concrete = concrete_from_recorded_opref(ctx, result);
                match dst_bank {
                    'i' => write_int_reg(ctx, op.pc, dst, result, concrete)?,
                    'r' => write_ref_reg(ctx, op.pc, dst, result, concrete)?,
                    'f' => {
                        let len = ctx.registers_f.len();
                        let _ =
                            ctx.registers_f
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
                return Ok((DispatchOutcome::Continue, op.next_pc));
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
        let entry = ctx
            .frame_state
            .borrow()
            .callee_shadow
            .as_ref()
            .and_then(|shadow| shadow.concrete.get(&index_value).copied())
            .filter(|entry| entry.frame_reg == code[op.pc + 1] as u16);
        let slot_opref = ctx
            .frame_state
            .borrow()
            .callee_shadow
            .as_ref()
            .and_then(|shadow| shadow.opref.get(&index_value).copied());
        // Prefer re-resolving the slot's SSA OpRef through the op-table (a
        // GC-forwarded, rooted value channel) over the raw concrete copy the
        // shadow holds: `CalleeLocalsShadow.concrete` is not visited by the
        // trace-ref walker, so a Ref copy dangles if a minor collection moved a
        // nursery object across the may-force residual. The op-table concrete IS
        // forwarded. Fall back to the stored copy when the OpRef carries no live
        // concrete (ints/floats are value copies of immutable payloads and are
        // always safe). Only consult the OpRef when a frame-matched `concrete`
        // entry exists, so the per-frame isolation the `frame_reg` filter gives
        // `concrete` also bounds the OpRef re-resolution.
        entry
            .and_then(|_| slot_opref.and_then(|opref| ctx.trace_ctx.concrete_of_opref(opref)))
            .or_else(|| entry.map(|entry| entry.value))
            .unwrap_or(Value::Void)
    } else {
        shadow_value
    };
    // Mirror `getfield_vable_via_metainterp`: stamp the read result's
    // concrete so `concrete_of_opref(result)` honors the Box.value
    // contract for downstream consumers; `Value::Void` = no live concrete.
    //
    // An indexed read hands back the STORED box, so the stamp is a write onto a
    // box that may already carry a live concrete. A frame whose array slot is
    // still an unmaterialized hole reads NULL while the real Ref lives in the
    // guard's register file, so stamping that NULL would clobber the live value
    // (last write wins) and fold a downstream may-force residual's Ref arg to
    // NULL -> `MayForceNullRefArgUnsupported`. Skip ONLY that clobber: a NULL
    // read onto a box with no live Ref still stamps, keeping NULL a real
    // concrete for the slots that genuinely hold one.
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
        'r' => write_ref_reg(ctx, op.pc, dst, result, concrete_for_shadow)?,
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
/// mirrored.  Slots at or above `nlocals` are operand-stack cells, which no
/// Python-level read can reach and which the resume snapshot reconstructs.
fn folded_store_is_observable_local<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    slot: i64,
) -> bool {
    {
        let state = ctx.frame_state.borrow();
        let Some(shadow) = state.callee_shadow.as_ref() else {
            return false;
        };
        if !shadow.frame_materialized {
            return false;
        }
    }
    let Some(nlocals) = active_frame_nlocals(ctx) else {
        return false;
    };
    slot >= 0 && slot < nlocals
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
    // write to the current inline level's own (unseeded) portal frame updates
    // the per-slot OpRef + concrete shadow and emits NO SETARRAYITEM_GC.  Both
    // local stores (`index < nlocals`) and operand-stack pushes (`index >=
    // nlocals`) on the fresh frame are pure mirror writes here — the value
    // lives in an SSA register; the vable array is folded away.
    let fold_frame_reg = fbw_strict_fold_frame_reg(ctx);
    if fold_frame_reg != u16::MAX && code[op.pc + 1] as u16 == fold_frame_reg {
        let index = read_int_reg(code, op, 1, ctx)?;
        if let Some(majit_ir::Value::Int(slot)) = ctx.trace_ctx.concrete_of_opref(index)
            && !folded_store_is_observable_local(ctx, slot)
        {
            let value = match value_bank {
                'i' => read_int_reg(code, op, 2, ctx)?,
                'r' => read_ref_reg(code, op, 2, ctx)?,
                'f' => read_float_reg(code, op, 2, ctx)?,
                _ => unreachable!("value_bank must be 'i', 'r' or 'f'"),
            };
            let concrete = vable_value_concrete(code, op, 2, ctx, value_bank, value)
                .unwrap_or(majit_ir::Value::Void);
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
                        concrete,
                    );
                }
            }
            let vable = read_ref_reg_raw(code, op, 0, ctx)?;
            // `_opimpl_setarrayitem_vable` records `SETARRAYITEM_GC` for every
            // box that is not `virtualizable_boxes[-1]`. The strict fold is
            // that standard arm (shadow only). An inlined callee frame is the
            // other arm: keep the live write above and fall through so the
            // same store is recorded on the array `getfield_gc` reads.
            if ctx.trace_ctx.standard_virtualizable_box() == Some(vable) {
                return Ok((DispatchOutcome::Continue, op.next_pc));
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
    let encoded_value = match value_bank {
        'i' => read_int_reg(code, op, 2, ctx)?,
        'r' => read_ref_reg(code, op, 2, ctx)?,
        'f' => read_float_reg(code, op, 2, ctx)?,
        _ => unreachable!("value_bank must be 'i', 'r' or 'f'"),
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
    let concrete =
        vable_effective_value_concrete(code, op, 2, ctx, value_bank, encoded_value, value)
            .unwrap_or(Value::Void);
    let is_stack_push = value_bank == 'r' && vable_store_is_stack_push(ctx, index_value);
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
        is_stack_push,
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
        crate::state::store_live_frame_array_slot(frame, index_value as usize, concrete);
        // Keep the inline concrete-locals shadow current so a later read of this
        // slot (after a may-force op clears the heapcache) recovers the concrete.
        // Seed BOTH maps: the read fallback prefers re-resolving the slot's OpRef
        // through the (GC-forwarded) op-table over the raw `concrete` copy, so the
        // OpRef must track the stored value on every write — otherwise a re-stored
        // slot would re-resolve a stale OpRef while `concrete` held the fresh value.
        // Only this level's own frame: a store to a foreign vable must not bind
        // in `locals()` through this shadow.
        if let Some(shadow) = ctx.frame_state.borrow_mut().callee_shadow.as_mut() {
            shadow.set_opref(index_value, value);
            shadow.set_concrete(code[op.pc + 1] as u16, index_value, concrete);
        }
    }
    let _ = _write;
    // A Ref stored to the operand-stack region of the vable array is an
    // operand-stack push (`pyframe.pushvalue` lowers to
    // `setarrayitem_vable_r(locals_cells_stack_w, depth, w_obj)`). Retain the
    // last-write TOS candidate for ordinary single-result opcodes. Method-form
    // LOAD_ATTR is the one two-result shape that also needs its exact lower
    // slot mirrored: it writes `[method, self]`, and retaining only `self`
    // made boundary reconciliation refill the method hole from the stale
    // pre-LOAD_ATTR receiver. A later guard then resumed CALL with that
    // receiver as its callee. Scope the positional write to that opcode; the
    // general boundary model remains authoritative for other push/pop shapes.
    // Local/cell stores remain excluded by the `index >= nlocals` gate.
    if value_bank == 'r' && ctx.vstack_valid {
        if let Some(nlocals) = active_frame_nlocals(ctx)
            && index_value >= nlocals
        {
            // `vstack_cur_pypc` is a pc in the frame the op names, so it must
            // be decoded against that frame's own bytecode.
            let method_load = active_frame_code(ctx).is_some_and(|code| {
                pyre_interpreter::decode_instruction_at(code, ctx.vstack_cur_pypc as usize)
                    .is_some_and(|(instr, op_arg)| match instr {
                        pyre_interpreter::Instruction::LoadAttr { namei } => {
                            namei.get(op_arg).is_method()
                        }
                        _ => false,
                    })
            });
            let stack_slot = (index_value - nlocals) as usize;
            if method_load {
                if ctx.frame_state.borrow().vstack_boxes.len() <= stack_slot {
                    ctx.frame_state
                        .borrow_mut()
                        .vstack_boxes
                        .resize(stack_slot + 1, OpRef::NONE);
                }
                ctx.frame_state.borrow_mut().vstack_boxes[stack_slot] = value;
            }
            // Mark the slot as execution-derived while an out-of-order region
            // is armed, so the boundary restore does not put its pc-derived
            // snapshot back over it.  Outside a region the mask does not exist
            // and this costs one `Option` test.
            //
            // This is a property of the STORE, not of the opcode that issued
            // it: the mask records which slot the walk demonstrably wrote, and
            // that is the same fact whatever the surrounding opcode is.  Only
            // the positional mirror write above is method-form specific, and
            // for the reason its own comment gives.  Scoping the mask to that
            // opcode too left every other push shape unprotected, so a window
            // whose only executed store came from, say, `LOAD_NAME` or
            // `BINARY_OP` reported an empty mask and the restore reinstated the
            // pre-window mirror wholesale over it.
            if let Some((_, _, _, mask)) =
                ctx.frame_state.borrow_mut().vstack_reorder_saved.as_mut()
            {
                if mask.len() <= stack_slot {
                    mask.resize(stack_slot + 1, false);
                }
                mask[stack_slot] = true;
            }
            ctx.frame_state.borrow_mut().vstack_last_ref = value;
        }
    }
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
