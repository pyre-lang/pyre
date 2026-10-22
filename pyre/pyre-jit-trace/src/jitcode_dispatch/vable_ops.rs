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
                result_slot: CallResultSlot::Unknown,
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
    // `finish_known_nonstandard` and mark the portal frame nonstandard —
    // the shared we_are_jitted `getfield debugdata` then poisons the
    // subsequent `getfield last_instr` into a heap load of the snapshot
    // ConstInt, and `switch` replays one opcode until TraceTooLong.
    let (result, shadow_value) = with_replace_frames(ctx, |ctx| {
        let nonstandard = if vable_is_standard_virtualizable(ctx, obj) {
            false
        } else {
            walker_nonstandard_virtualizable(ctx, pc, obj, &descr)?
        };
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

fn skip_lives(code: &[u8], mut pc: usize) -> usize {
    while let Some(op) = crate::jitcode_runtime::decode_op_at(code, pc) {
        if op.opname != "live" {
            return pc;
        }
        pc = op.next_pc;
    }
    pc
}

/// Flatten's shared we_are_jitted block: `getfield debugdata` /
/// `ptr_nonzero` / `goto_if_not` / residual / `switch last_instr`.
pub(crate) fn is_shared_waj_block(code: &[u8], target: usize) -> bool {
    let pc = skip_lives(code, target);
    let Some(getfield) = crate::jitcode_runtime::decode_op_at(code, pc) else {
        return false;
    };
    if getfield.key != "getfield_vable_r/rd>r" {
        return false;
    }
    let Some(ptr_nz) =
        crate::jitcode_runtime::decode_op_at(code, skip_lives(code, getfield.next_pc))
    else {
        return false;
    };
    if ptr_nz.key != "ptr_nonzero/r>i" {
        return false;
    }
    let Some(gin) = crate::jitcode_runtime::decode_op_at(code, skip_lives(code, ptr_nz.next_pc))
    else {
        return false;
    };
    gin.key == "goto_if_not/iL"
}

/// Flatten's shared `get_is_being_profiled` + `promote_greens` block:
/// `-live-`, `getfield debugdata`, `goto_if_not_ptr_iszero`, then the
/// profiled diamond and `{int,ref}_guard_value` before `switch next_instr`.
/// Distinct from [`is_shared_waj_block`]: that copy has no leading `-live-`
/// and uses `ptr_nonzero` + `goto_if_not`.
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
/// `dispatch_bytecode` copy or the factored portal merge-greens copy.
///
/// `waj_shared` stores `last_instr` (`pyopcode.py dispatch_bytecode`);
/// `pmg_shared` does not (`interp_jit.py PyFrame.dispatch` greens only).
/// Orgpc for a pmg-internal guard therefore stays on the entering
/// opcode's `live_before` py, not the vable field.
pub(crate) fn begin_hold_if_shared_hook<Sym: WalkSym>(
    code: &[u8],
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
) {
    let uses_vable_last_instr = if is_shared_waj_block(code, pc) {
        true
    } else if is_shared_pmg_block(code, pc) {
        false
    } else {
        return;
    };
    ctx.session.borrow_mut().begin_hold_opcode_live_before(
        ctx.registers_i,
        ctx.registers_r,
        ctx.registers_f,
        uses_vable_last_instr,
    );
}

/// Walker-inline the factored `dispatch_bytecode` we_are_jitted copy.
///
/// Upstream holds one
/// `if self.debugdata is not None and self.debugdata.w_f_trace is not None:
///      ec.bytecode_only_trace(self); next_instr = r_uint(self.last_instr)`
/// in the dispatch loop.  Pyre unrolls the loop, so flatten emits
/// `goto L_shared` per instruction and one out-of-line residual.  The
/// unarmed arm does not reread `last_instr` (`pyopcode.py
/// dispatch_bytecode`); resume at the opcode body that follows the
/// `goto`.  The armed arm falls through to the residual and the
/// `switch` (`next_instr = r_uint(self.last_instr)`).  Flatten's shared
/// block records both conjuncts; this inline walks the `w_f_trace` test
/// the same way as `debugdata` so the unarmed walk never enters the
/// switch.
pub(crate) fn try_inline_shared_waj_goto<Sym: WalkSym>(
    code: &[u8],
    goto_op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    target: usize,
) -> Option<Result<(DispatchOutcome, usize), DispatchError>> {
    if !is_shared_waj_block(code, target) {
        return None;
    }
    let getfield = crate::jitcode_runtime::decode_op_at(code, skip_lives(code, target))?;
    let (_, mut pc) = match getfield_vable_via_metainterp(code, &getfield, ctx, 'r') {
        Ok(pair) => pair,
        Err(err) => return Some(Err(err)),
    };
    pc = skip_lives(code, pc);
    let ptr_nz = crate::jitcode_runtime::decode_op_at(code, pc)?;
    if ptr_nz.key != "ptr_nonzero/r>i" {
        return None;
    }
    let (_, pc) = match super::arith::ptr_nullity_record(code, &ptr_nz, ctx, true) {
        Ok(pair) => pair,
        Err(err) => return Some(Err(err)),
    };
    let gin = crate::jitcode_runtime::decode_op_at(code, skip_lives(code, pc))?;
    if gin.key != "goto_if_not/iL" {
        return None;
    }
    let valuebox = match read_int_reg(code, &gin, 0, ctx) {
        Ok(value) => value,
        Err(err) => return Some(Err(err)),
    };
    let switchcase = match ctx.trace_ctx.concrete_of_opref(valuebox) {
        Some(Value::Int(v)) => v,
        _ => {
            return Some(Err(DispatchError::GotoIfNotValueNotConcrete {
                pc: gin.pc,
                value: valuebox,
            }));
        }
    };
    // Unarmed (`switchcase == 0`): jump to the opcode body after `goto
    // L_shared`.  Resume the armed arm at this `goto` so blackhole
    // re-enters the one `dispatch_bytecode` copy with this opcode's
    // live — the shared residual's `-live-` is the union of every body.
    if switchcase == 0 {
        return Some(super::goto_if_not_branch_on_with_other(
            code,
            &gin,
            ctx,
            valuebox,
            switchcase,
            goto_op.next_pc,
            goto_op.pc,
        ));
    }
    // Armed debugdata: fall through.  Snapshot, GuardNotForced, and
    // ABORT_ESCAPE resume at this `goto` so `last_instr` / fail_args /
    // blackhole stay on this opcode's live (`dispatch_bytecode`'s
    // in-loop copy).  Walking the shared `switch last_instr` instead
    // stuck on the merge-point ConstInt (TraceTooLong).  Capturing at
    // the residual published RETURN's `last_instr` (acc 576201 on the
    // armed-before-entry fixture).
    //
    // Flatten's shared block also tests `_d.w_f_trace is not None`
    // (`emit_shared_cond_residual`) between this `goto_if_not` and the
    // residual.  Walk that conjunct the same way as debugdata; unarmed
    // (`w_f_trace` is null) skips the residual and the switch.  Leaving
    // the extra ops unconsumed declined this inline (`residual_call_r_v`
    // no longer follows the first `goto_if_not`) and the generic walker
    // then took the switch — one merge-point ConstInt, `compile_simple_loop`
    // unrolling to GuardAlwaysFails.
    match super::goto_if_not_branch_on(code, &gin, ctx, valuebox, switchcase, goto_op.next_pc) {
        Ok((DispatchOutcome::Continue, _)) => {}
        other => return Some(other),
    }
    let mut residual_pc = skip_lives(code, gin.next_pc);
    let mut residual = crate::jitcode_runtime::decode_op_at(code, residual_pc)?;
    if residual.key == "getfield_gc_r/rd>r" || residual.key == "getfield_gc_r_pure/rd>r" {
        match getfield_gc_via_heapcache(code, &residual, ctx, OpCode::GetfieldGcR, 'r') {
            Ok((DispatchOutcome::Continue, _)) => {}
            other => return Some(other),
        }
        residual_pc = skip_lives(code, residual.next_pc);
        let ptr_nz = crate::jitcode_runtime::decode_op_at(code, residual_pc)?;
        if ptr_nz.key != "ptr_nonzero/r>i" {
            return None;
        }
        match super::arith::ptr_nullity_record(code, &ptr_nz, ctx, true) {
            Ok((DispatchOutcome::Continue, _)) => {}
            other => return Some(other),
        }
        let gin = crate::jitcode_runtime::decode_op_at(code, skip_lives(code, ptr_nz.next_pc))?;
        if gin.key != "goto_if_not/iL" {
            return None;
        }
        let valuebox = match read_int_reg(code, &gin, 0, ctx) {
            Ok(value) => value,
            Err(err) => return Some(Err(err)),
        };
        let switchcase = match ctx.trace_ctx.concrete_of_opref(valuebox) {
            Some(Value::Int(v)) => v,
            _ => {
                return Some(Err(DispatchError::GotoIfNotValueNotConcrete {
                    pc: gin.pc,
                    value: valuebox,
                }));
            }
        };
        if switchcase == 0 {
            return Some(super::goto_if_not_branch_on_with_other(
                code,
                &gin,
                ctx,
                valuebox,
                switchcase,
                goto_op.next_pc,
                goto_op.pc,
            ));
        }
        match super::goto_if_not_branch_on(code, &gin, ctx, valuebox, switchcase, goto_op.next_pc) {
            Ok((DispatchOutcome::Continue, _)) => {}
            other => return Some(other),
        }
        residual_pc = skip_lives(code, gin.next_pc);
        residual = crate::jitcode_runtime::decode_op_at(code, residual_pc)?;
    }
    if !residual.key.starts_with("residual_call_r_v") {
        return None;
    }
    match super::residual_call::dispatch_residual_call_iRd_kind_at(
        code,
        &residual,
        ctx,
        'v',
        goto_op.pc,
        goto_op.next_pc,
    ) {
        Ok((DispatchOutcome::Continue, _)) => {
            Some(Ok((DispatchOutcome::Continue, goto_op.next_pc)))
        }
        other => Some(other),
    }
}

/// Whether `vable` is the standard virtualizable identity, by Box or by
/// the concrete frame pointer `virtualizable_boxes[-1]` carries.
///
/// `_nonstandard_virtualizable` compares Box identity first
/// (`pyjitpl.py`); a portal frame register that is a copy of that
/// identity still names the same object, and the shared
/// `dispatch_bytecode` `last_instr` reread (`next_instr =
/// r_uint(self.last_instr)`) has to see the store.
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
        if own_box && let Some(shadow) = ctx.frame_state.borrow_mut().callee_shadow.as_mut() {
            shadow.set_opref(index_value, value);
            shadow.set_concrete(code[op.pc + 1] as u16, index_value, concrete);
        }
    }
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
