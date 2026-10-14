//! Shared frame-owned state for the walker's remaining semantic mirrors.
//!
//! `pyjitpl.py MIFrame` owns its live values; `MetaInterp.replace_box` updates
//! every frame synchronously. Until the semantic/color mirrors disappear,
//! they must share that ownership and GC lifetime instead of being copied
//! between a suspended SubWalkFrame and its active WalkContext.

use super::{CalleeLocalsShadow, ConcreteValue};
use majit_ir::{GcRef, OpRef, Value};
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

pub struct WalkFrameStateData {
    /// Present only for an inlined-callee sub-walk. Top-level and other walks
    /// have no callee shadow, preserving the former empty-stack no-op behavior.
    pub callee_shadow: Option<CalleeLocalsShadow>,
    /// Frozen `PyFrame` state at the outer Python opcode boundary —
    /// `sym.registers_r ∪ sym.registers_i.opref ∪ sym.registers_f.opref`
    /// captured at walk entry (the retired per-opcode arm entry did
    /// this; inline sub-walks seed it from the CALL-site capture; the
    /// full-body root leaves it empty and collects at guard capture),
    /// filtered by `OpRef::is_none()`.  This is what
    /// [`super::walker_capture_snapshot_for_last_guard`] passes as the
    /// snapshot frame's active boxes on the arm path.
    ///
    /// Sub-walks clone the parent's Vec — outer active-box count is
    /// small (a Python frame's live locals + stack tail) and walker
    /// nesting depth is shallow (2–3 levels), so the per-sub-walk
    /// clone cost is negligible.
    pub outer_active_boxes: Vec<OpRef>,
    /// PyPy-faithful kept-operand-stack snapshot: the walk-level
    /// symbolic operand stack, indexed by ABSOLUTE operand-stack depth
    /// (slot `s`, `s in 0..vstack_depth`).  The Python operand stack is
    /// all-Ref (`W_Root`), so a single `Vec<OpRef>` (Ref bank) suffices.
    /// This is the walker analog of PyPy's `MIFrame.registers_r`
    /// valuestack array snapshotted by `get_list_of_active_boxes`
    /// (`pyjitpl.py`) — the authoritative per-slot box source the
    /// `stack_sync` vable overlay reads at a branch guard instead of the
    /// unreliable `registers_r[stack_slot_color_map[s]]` static-color read.
    ///
    /// Maintained ONLY when `sym.owns_virtualizable_shadow()`; on any
    /// unmodeled stack effect the maintenance sets `vstack_valid = false`
    /// and `stack_sync` omits every operand slot, which resume
    /// re-materializes (zero regression).
    ///
    /// The mirror is the SOLE kept-stack source at a branch guard: the
    /// flat `stack_slot_color_map` static-color read it once fell back to
    /// is retired, so a slot the mirror does not cover is omitted rather
    /// than read from the flat map.  `PYRE_VSTACK_DIAG` logs the per-op
    /// reconcile trace.
    pub vstack_boxes: Vec<OpRef>,
    /// #73: the last Ref box written via [`super::write_ref_reg`] during the
    /// CURRENT Python opcode — the box a value-producing opcode lands on
    /// the operand-stack TOS.  Reset to `OpRef::NONE` at every opcode
    /// boundary; read by [`super::reconcile_vstack_at_boundary`] for the
    /// RESULT-TO-TOS class.
    pub vstack_last_ref: OpRef,
    /// The `(py_pc, depth, boxes)` the mirror held when `vstack_reorder_ceiling`
    /// was armed.  A layout excursion that returns to that exact coordinate has
    /// retired no Python opcode, so the operand stack it left is still the
    /// operand stack it comes back to and the saved boxes are restored verbatim
    /// — the shadow reseed cannot reconstruct them, because mid-expression the
    /// virtualizable's stack region holds the NULLs the in-flight opcode's
    /// `popvalue_maybe_none` wrote.  `None` outside a region.
    ///
    /// There is nothing to snapshot upstream: `pyjitpl.py`
    /// `MIFrame.run_one_step` steps a live frame whose `registers_r` survive
    /// the step.  This mirror is instead reconstructed from source pcs, and
    /// that reconstruction is exactly what an excursion can lose.
    pub vstack_reorder_saved: Option<(u32, usize, Vec<OpRef>, Vec<bool>)>,
    /// The exception a bridge resumes with, and after a walked
    /// `set_current_exception` the value that store published.  Read by the
    /// nullary bare-reraise folds, and by the PUSH_EXC_INFO `prev` save only
    /// under [`super::FbwWalkMode::current_exception_seed_from_walk_store`].
    pub current_exception_seed: Option<OpRef>,
    /// Concrete shadow paired with [`WalkFrameStateData::current_exception_seed`].
    pub current_exception_seed_concrete: pyre_object::PyObjectRef,
    /// Concrete shadow mirror for `registers_r`.
    ///
    /// Semantic-slot indexed, length equals `registers_r.len()`. At
    /// `dispatch_via_miframe` entry, populated by concatenating
    /// `PyreSym.concrete_locals` + `PyreSym.concrete_stack`; sub-walks
    /// allocate a fresh `Vec<ConcreteValue>` sized to the callee's
    /// `num_regs_r` and fill arg slots from the parent's slice at the
    /// arg byte indices.
    ///
    /// **Mutable invariant**: every walker handler that
    /// writes `registers_r[dst]` MUST also write `concrete_registers_r
    /// [dst]` in lock-step.  Use the [`super::write_ref_reg`] helper which
    /// enforces this contract.  Sites that don't know the result's
    /// concrete pass `ConcreteValue::Null` — downstream consumers
    /// (e.g. `raise/r` GUARD_CLASS gate) treat `Null` as "no info,
    /// skip the guard", same as slots the snapshot never populated.
    /// Copy-style handlers (`ref_copy/r>r`,
    /// `last_exc_value/>r`) propagate the source's concrete.
    ///
    /// The slice is mutable so the concrete shadow tracks the symbolic
    /// register in lock-step. If it were immutable, sibling handlers
    /// like `last_exc_value/>r` could rewrite the symbolic register
    /// without touching the concrete snapshot, so a follow-on `raise/r`
    /// would read a stale concrete and silently skip the GUARD_CLASS
    /// gate; the lock-step contract keeps walker-side GUARD_CLASS sound.
    ///
    /// **Companion bank** `super::WalkContext::concrete_registers_i` carries the same
    /// contract for the Int bank.  Production entries size and seed it:
    /// `dispatch_via_miframe` from `top_constants_i` and `argboxes_i`,
    /// and both inline-callee entries; only the test fixtures pass
    /// `&mut []`.
    ///
    /// `goto_if_not/iL` and `switch/id` read neither bank: both resolve
    /// their branch value through `TraceCtx::concrete_of_opref` and
    /// surface `GotoIfNotValueNotConcrete` rather than guess a
    /// direction.
    pub concrete_registers_r: Vec<ConcreteValue>,
    /// Inlined callee's `Function.w_func_globals_obj`.  Zero outside an
    /// inline sub-walk.  `walk_frame_state_roots` forwards it.
    pub inline_w_globals: usize,
    /// Inlined callee's `W_Code`.  Zero outside an inline sub-walk.  Same
    /// forwarding contract as [`Self::inline_w_globals`].
    pub inline_w_code: usize,
    /// Alternate resume for the item class check inside
    /// `list_iter_descr_next`. The walker records `GuardNonnull` and
    /// `GuardClass`; the strengthened `GuardNonnullClass` keeps the
    /// `GuardNonnull` resume. `Some((jit_pc, boxes))` is the `-live-`
    /// after the loop-header `jit_merge_point` and the active boxes
    /// collected at that same pc. `None` leaves every other guard on the
    /// preamble marker in [`Self::outer_active_boxes`].
    pub list_iter_class_guard_resume: Option<(usize, Vec<OpRef>)>,
}

impl Default for WalkFrameStateData {
    fn default() -> Self {
        Self {
            callee_shadow: None,
            outer_active_boxes: Vec::new(),
            vstack_boxes: Vec::new(),
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_saved: None,
            current_exception_seed: None,
            current_exception_seed_concrete: pyre_object::PY_NULL,
            concrete_registers_r: Vec::new(),
            inline_w_globals: 0,
            inline_w_code: 0,
            list_iter_class_guard_resume: None,
        }
    }
}

#[derive(Clone, Default)]
pub struct WalkFrameState(Rc<RefCell<WalkFrameStateData>>);

impl WalkFrameState {
    pub fn new(data: WalkFrameStateData) -> Self {
        Self(Rc::new(RefCell::new(data)))
    }

    /// A short field access only: release before executing/allocating guest
    /// objects. The collector's checked borrow detects violations.
    pub fn borrow(&self) -> Ref<'_, WalkFrameStateData> {
        self.0.borrow()
    }

    /// As with `borrow`, no borrow may span a collecting call.
    pub fn borrow_mut(&self) -> RefMut<'_, WalkFrameStateData> {
        self.0.borrow_mut()
    }

    /// `virtualizable.py write_boxes`: validate the sparse native shadow, then
    /// read each live value immediately before its write. The writer may box
    /// Int/Float values and collect; neither a borrow nor a copied Ref list may
    /// span that allocation. Ref values themselves require no boxing allocation.
    pub(crate) fn write_callee_locals(
        &self,
        nlocals: usize,
        frame_reg: u16,
        mut write: impl FnMut(usize, Value),
    ) -> bool {
        {
            let data = self.borrow();
            let Some(shadow) = data.callee_shadow.as_ref() else {
                return false;
            };
            if !super::callee_locals_region_is_publishable(shadow, nlocals, frame_reg) {
                return false;
            }
        }
        for slot in 0..nlocals {
            let value = {
                let data = self.borrow();
                data.callee_shadow
                    .as_ref()
                    .expect("callee shadow retired during writeback")
                    .concrete
                    .get(&(slot as i64))
                    .map(|entry| entry.value)
            };
            if let Some(value) = value {
                write(slot, value);
            }
        }
        true
    }

    pub(crate) fn replace_active_box(&self, oldbox: OpRef, newbox: OpRef) {
        let mut data = self.borrow_mut();
        let replace = |slot: &mut OpRef| {
            if *slot == oldbox {
                *slot = newbox;
            }
        };
        for slot in &mut data.outer_active_boxes {
            replace(slot);
        }
        if let Some((_, boxes)) = &mut data.list_iter_class_guard_resume {
            for slot in boxes {
                replace(slot);
            }
        }
        for slot in &mut data.vstack_boxes {
            replace(slot);
        }
        replace(&mut data.vstack_last_ref);
        if let Some((_, _, boxes, _)) = &mut data.vstack_reorder_saved {
            for slot in boxes {
                replace(slot);
            }
        }
        if let Some(shadow) = &mut data.callee_shadow {
            replace(&mut shadow.frame_box);
            for slot in shadow.opref.values_mut() {
                replace(slot);
            }
        }
        if let Some(slot) = &mut data.current_exception_seed {
            replace(slot);
        }
    }

    #[cfg(test)]
    pub(crate) fn root(&self) -> WalkFrameStateRoot {
        self.root_with(majit_metainterp::FrontendResrefOwner::NONE)
    }

    pub(crate) fn root_with(
        &self,
        owner: majit_metainterp::FrontendResrefOwner,
    ) -> WalkFrameStateRoot {
        let payload = super::FrontendBoxArea::new(Rc::as_ptr(&self.0), owner);
        let area = unsafe {
            majit_gc::shadow_stack::MutatorExtraAreaGuard::new(
                walk_frame_state_roots,
                std::ptr::from_ref(payload.as_ref()).cast(),
                "miframe_state",
            )
        };
        WalkFrameStateRoot {
            _area: area,
            _payload: payload,
            _owner: self.clone(),
        }
    }
}

pub(crate) struct WalkFrameStateRoot {
    _area: majit_gc::shadow_stack::MutatorExtraAreaGuard,
    _payload: Box<super::FrontendBoxArea>,
    _owner: WalkFrameState,
}

unsafe fn walk_frame_state_roots(data: *const (), visitor: &mut dyn FnMut(&mut GcRef)) {
    let area = unsafe { &*(data as *const super::FrontendBoxArea) };
    let owner = area.owner;
    let cell = unsafe { &*(area.storage as *const RefCell<WalkFrameStateData>) };
    let mut data = cell
        .try_borrow_mut()
        .expect("walk frame state borrow held across collection");
    let walk_frame_box = |op: &mut OpRef, visitor: &mut dyn FnMut(&mut GcRef)| {
        super::walk_frontend_box(op, owner, visitor);
    };
    for value in &mut data.outer_active_boxes {
        walk_frame_box(value, visitor);
    }
    if let Some((_, boxes)) = &mut data.list_iter_class_guard_resume {
        for value in boxes {
            walk_frame_box(value, visitor);
        }
    }
    for value in &mut data.vstack_boxes {
        walk_frame_box(value, visitor);
    }
    walk_frame_box(&mut data.vstack_last_ref, visitor);
    if let Some((_, _, boxes, _)) = &mut data.vstack_reorder_saved {
        for value in boxes {
            walk_frame_box(value, visitor);
        }
    }
    if let Some(shadow) = &mut data.callee_shadow {
        walk_frame_box(&mut shadow.frame_box, visitor);
        for value in shadow.opref.values_mut() {
            walk_frame_box(value, visitor);
        }
        for entry in shadow.concrete.values_mut() {
            if let Value::Ref(root) = &mut entry.value {
                visitor(root);
            }
        }
        // FrameBox frames are currently nonmoving, but the owning reference
        // is still a GC edge (call_jit.rs walk_jit_callee_frame_roots_area).
        let mut frame = GcRef(shadow.concrete_frame);
        visitor(&mut frame);
        shadow.concrete_frame = frame.0;
        // code_ptr is a Rust CodeObject address, not a PyObject GC pointer.
        // The JitCode code-root publication traces that code's GC fields.
    }
    if let Some(value) = &mut data.current_exception_seed {
        walk_frame_box(value, visitor);
    }
    let mut exception = GcRef(data.current_exception_seed_concrete as usize);
    visitor(&mut exception);
    data.current_exception_seed_concrete = exception.0 as pyre_object::PyObjectRef;
    if !data.current_exception_seed_concrete.is_null() {
        // Native exception carriers also need their young child slots walked.
        unsafe {
            pyre_interpreter::eval::walk_raw_exception_roots(
                data.current_exception_seed_concrete,
                visitor,
            )
        };
    }
    for value in &mut data.concrete_registers_r {
        if let ConcreteValue::Ref(ptr) = value {
            let mut root = GcRef(*ptr as usize);
            visitor(&mut root);
            *ptr = root.0 as pyre_object::PyObjectRef;
        }
    }
    if data.inline_w_globals != 0 {
        let mut root = GcRef(data.inline_w_globals);
        visitor(&mut root);
        data.inline_w_globals = root.0;
    }
    if data.inline_w_code != 0 && data.inline_w_code != usize::MAX {
        let mut root = GcRef(data.inline_w_code);
        visitor(&mut root);
        data.inline_w_code = root.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with_refs(word: usize) -> WalkFrameState {
        let op = OpRef::const_ptr(GcRef(word));
        let mut shadow = CalleeLocalsShadow {
            frame_box: op,
            concrete_frame: word,
            ..Default::default()
        };
        shadow.set_opref(0, op);
        shadow.set_concrete(1, 0, Value::Ref(GcRef(word)));
        WalkFrameState::new(WalkFrameStateData {
            callee_shadow: Some(shadow),
            outer_active_boxes: vec![op],
            vstack_boxes: vec![op],
            vstack_last_ref: op,
            vstack_reorder_saved: Some((0, 1, vec![op], vec![true])),
            current_exception_seed: Some(op),
            // Only the symbolic seed may use a sentinel: concrete exceptions
            // have their real children traversed by the callback.
            current_exception_seed_concrete: pyre_object::PY_NULL,
            concrete_registers_r: vec![
                ConcreteValue::Ref(word as pyre_object::PyObjectRef),
                ConcreteValue::Int(word as i64),
            ],
            inline_w_globals: word,
            inline_w_code: word,
            list_iter_class_guard_resume: Some((7, vec![op])),
            ..Default::default()
        })
    }

    fn assert_refs(state: &WalkFrameState, word: usize) {
        let state = state.borrow();
        let op = OpRef::const_ptr(GcRef(word));
        assert_eq!(state.outer_active_boxes, [op]);
        assert_eq!(state.vstack_boxes, [op]);
        assert_eq!(state.vstack_last_ref, op);
        assert_eq!(state.vstack_reorder_saved.as_ref().unwrap().2, [op]);
        assert_eq!(state.current_exception_seed, Some(op));
        assert!(
            matches!(state.concrete_registers_r[0], ConcreteValue::Ref(p) if p as usize == word)
        );
        let shadow = state.callee_shadow.as_ref().unwrap();
        assert_eq!(shadow.frame_box, op);
        assert_eq!(shadow.concrete_frame, word);
        assert_eq!(shadow.opref.get(&0), Some(&op));
        assert_eq!(
            shadow.concrete.get(&0).unwrap().value,
            Value::Ref(GcRef(word))
        );
        assert_eq!(state.inline_w_globals, word);
        assert_eq!(state.inline_w_code, word);
        assert_eq!(
            state
                .list_iter_class_guard_resume
                .as_ref()
                .map(|(pc, boxes)| (*pc, boxes.as_slice())),
            Some((7, [op].as_slice()))
        );
    }

    #[test]
    fn nested_scoped_roots_forward_every_frame_mirror_and_retire_independently() {
        const BASE: usize = 0x98_0000_0000;
        let _restore = [
            crate::RestoreConstPtr::intern(BASE + 0x1000),
            crate::RestoreConstPtr::intern(BASE + 0x2000),
        ];
        let _runtime = crate::trace_ctx_for_test(0);
        let _stw = majit_gc::gc_sync::quiesce_mutators();
        let outer = state_with_refs(BASE + 0x1000);
        let inner = state_with_refs(BASE + 0x2000);
        let paused = outer.clone();
        let outer_root = outer.root();
        let inner_root = inner.root();
        drop(outer);
        majit_gc::shadow_stack::walk_my_extra_areas(|root| {
            if root.0 == BASE + 0x1000 || root.0 == BASE + 0x2000 {
                root.0 += 0x80;
            }
        });
        assert_refs(&paused, BASE + 0x1080);
        assert_refs(&inner, BASE + 0x2080);
        assert!(matches!(
            paused.borrow().concrete_registers_r[1],
            ConcreteValue::Int(n) if n == (BASE + 0x1000) as i64
        ));
        drop(outer_root);
        majit_gc::shadow_stack::walk_my_extra_areas(|root| {
            if root.0 == BASE + 0x1080 || root.0 == BASE + 0x2080 {
                root.0 += 0x80;
            }
        });
        assert_refs(&paused, BASE + 0x1080);
        assert_refs(&inner, BASE + 0x2100);
        drop(inner_root);
    }

    #[test]
    fn callee_writeback_releases_borrows_and_reads_forwarded_later_slots() {
        const BASE: usize = 0x99_0000_0000;
        let _runtime = crate::trace_ctx_for_test(0);
        let _stw = majit_gc::gc_sync::quiesce_mutators();
        let mut shadow = CalleeLocalsShadow::default();
        shadow.set_concrete(1, 0, Value::Int(1000));
        shadow.set_concrete(1, 1, Value::Ref(GcRef(BASE + 0x1000)));
        shadow.set_concrete(1, 2, Value::Float(2.5));
        shadow.set_concrete(1, 3, Value::Ref(GcRef(BASE + 0x2000)));
        let state = WalkFrameState::new(WalkFrameStateData {
            callee_shadow: Some(shadow),
            ..WalkFrameStateData::default()
        });
        let _root = state.root();
        let published = state.clone();
        let mut written = Vec::new();
        assert!(published.write_callee_locals(5, 1, |slot, value| {
            // Boxing a scalar can collect before a later Ref is loaded. Walk
            // the real registered owner, so a retained borrow panics here and
            // an unrooted shadow copy returns a stale later value.
            if matches!(value, Value::Int(_) | Value::Float(_)) {
                majit_gc::shadow_stack::walk_my_extra_areas(|root| {
                    if (BASE + 0x1000..BASE + 0x3000).contains(&root.0) {
                        root.0 += 0x80;
                    }
                });
            }
            written.push((slot, value));
        }));
        assert_eq!(
            written,
            vec![
                (0, Value::Int(1000)),
                (1, Value::Ref(GcRef(BASE + 0x1080))),
                (2, Value::Float(2.5)),
                (3, Value::Ref(GcRef(BASE + 0x2100))),
            ]
        );
    }

    #[test]
    fn callee_writeback_validates_all_slots_before_writing() {
        let mut shadow = CalleeLocalsShadow::default();
        shadow.set_concrete(1, 0, Value::Int(1000));
        shadow.set_opref(1, OpRef::ref_op(7));
        let state = WalkFrameState::new(WalkFrameStateData {
            callee_shadow: Some(shadow),
            ..WalkFrameStateData::default()
        });
        assert!(!state.write_callee_locals(2, 1, |_, _| panic!("partial write")));
        assert!(!state.write_callee_locals(1, 2, |_, _| panic!("wrong frame")));
        assert!(!WalkFrameState::default().write_callee_locals(1, 1, |_, _| panic!("no shadow")));
    }

    #[test]
    fn replacement_is_immediate_before_a_collection_or_resume() {
        let state = state_with_refs(0x1000);
        let paused = state.clone();
        let old = OpRef::const_ptr(GcRef(0x1000));
        let new = OpRef::input_arg_ref(0);
        paused.replace_active_box(old, new);
        let state = state.borrow();
        assert_eq!(state.outer_active_boxes, [new]);
        assert_eq!(state.vstack_boxes, [new]);
        assert_eq!(state.vstack_last_ref, new);
        assert_eq!(state.vstack_reorder_saved.as_ref().unwrap().2, [new]);
        assert_eq!(state.current_exception_seed, Some(new));
        assert_eq!(
            state
                .list_iter_class_guard_resume
                .as_ref()
                .map(|(pc, boxes)| (*pc, boxes.as_slice())),
            Some((7, [new].as_slice()))
        );
        let shadow = state.callee_shadow.as_ref().unwrap();
        assert_eq!(shadow.frame_box, new);
        assert_eq!(shadow.opref.get(&0), Some(&new));
    }

    #[test]
    #[should_panic(expected = "walk frame state borrow held across collection")]
    fn collecting_with_a_live_field_borrow_is_rejected() {
        let state = WalkFrameState::default();
        let _borrow = state.borrow();
        let payload = super::super::FrontendBoxArea::new(
            Rc::as_ptr(&state.0),
            majit_metainterp::FrontendResrefOwner::NONE,
        );
        // Direct invocation keeps the expected panic outside the global root
        // traversal; no registry borrow is poisoned by this diagnostic.
        unsafe { walk_frame_state_roots(std::ptr::from_ref(payload.as_ref()).cast(), &mut |_| {}) };
    }

    #[test]
    fn standing_exception_seed_keeps_its_young_children() {
        use pyre_object::interp_exceptions::{ExcKind, W_ExceptionExtended, w_exception_new_empty};
        pyre_interpreter::typedef::init_typeobjects();
        let _pins = pyre_object::gc_roots::push_roots();
        let exception =
            pyre_object::gc_roots::pin_root(w_exception_new_empty(ExcKind::UnicodeTranslateError));
        let child = pyre_object::gc_roots::pin_root(pyre_object::unicodeobject::w_str_new("child"));
        let replacement =
            pyre_object::gc_roots::pin_root(pyre_object::unicodeobject::w_str_new("forwarded"));
        let _stw = majit_gc::gc_sync::quiesce_mutators();
        unsafe {
            (*(exception as *mut W_ExceptionExtended)).w_object = child;
        }
        let state = WalkFrameState::new(WalkFrameStateData {
            current_exception_seed_concrete: exception,
            ..Default::default()
        });
        let _root = state.root();
        let mut seen = false;
        majit_gc::shadow_stack::walk_my_extra_areas(|root| {
            seen |= root.0 == exception as usize;
            if root.0 == child as usize {
                root.0 = replacement as usize;
            }
        });
        assert!(seen);
        assert_eq!(
            unsafe { (*(exception as *const W_ExceptionExtended)).w_object },
            replacement
        );
    }
}
