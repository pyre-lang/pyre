//! `WalkerFrameOps` trait — abstraction over the small surface of
//! `MIFrame` methods a trace-recording helper needs, so one helper body
//! can run against either `MIFrame` or the walker `WalkContext` and emit
//! the same guard-and-record shape from both.
//!
//! Its only remaining consumer is `state.rs`'s
//! `trace_unbox_int_with_resume{,_descr}`, which today is reached from a
//! unit test alone; the strategy-aware STORE_SUBSCR helpers this trait was
//! introduced for have been deleted.
//!
//! ## Trait shape — `self`-only signatures, `ctx` reached via accessor
//!
//! Earlier draft passed `ctx: &mut TraceCtx` as an explicit method
//! argument alongside `&mut self`.  That shape forces the caller to
//! split-borrow `self.trace_ctx` from `self`, which is illegal whenever
//! the same `TraceCtx` is reached transitively through both `self` and
//! the explicit arg (always true for `MIFrame`'s `self.ctx: *mut TraceCtx`
//! and `WalkContext`'s `self.trace_ctx: &'frame mut TraceCtx`).
//!
//! The current shape — `&mut self` only, `ctx` reached through
//! `self.ctx_mut()` / `self.ctx()` accessor methods — lets each impl
//! choose how to materialise the borrow (`MIFrame` does `unsafe { &mut
//! *self.ctx }`; `WalkContext` returns its `trace_ctx` field).  The
//! accessor is scoped to a single statement at each callsite so the
//! borrow ends before the next `self`-mut-call, satisfying the borrow
//! checker without nested-borrow gymnastics.
//!
//! ## Why a trait (not free functions)
//!
//! Five of the six methods are pure compositions over `ctx_mut` +
//! `generate_guard` and live as default impls below.  The lone
//! load-bearing method is `generate_guard`, which on `MIFrame` walks
//! `get_list_of_active_boxes`, `build_framestack_snapshot`, etc.  The
//! walker has the same semantic responsibility (capture a multi-frame
//! resume snapshot before recording the guard op) but reaches it
//! through `walker_capture_snapshot_for_last_guard` and the
//! dispatch-time `WalkContext` register banks instead of `MIFrame`'s
//! state.  Two distinct implementations are unavoidable; a trait makes
//! the two impls interchangeable at the `generated_*` call sites.
//!
//! ## Trait scope
//!
//! Only the methods reached by the STORE_SUBSCR specialization closure
//! are members.  Other `MIFrame` methods (`guard_nonnull`,
//! `trace_dynamic_list_index`, ...) stay where they are until a walker
//! specialization needs the same surface.
//!
//! `MIFrame` and `WalkContext` both delegate `generate_guard` to their
//! existing infrastructure (`MIFrame::generate_guard` and
//! `walker_capture_snapshot_for_last_guard`, respectively).  The
//! generated strategy helpers are generic over this trait, so the
//! residual-call walker specialization can reuse the retained MIFrame IR
//! shape.

use majit_ir::{OpCode, OpRef, Type};
use majit_metainterp::TraceCtx;
use pyre_object::PyType;

/// Surface used by the strategy-aware STORE_SUBSCR specialization.  See
/// module doc for the rationale.
pub trait WalkerFrameOps {
    /// Mut access to the embedded `TraceCtx`.  Each impl materialises
    /// the borrow from its own internal storage (`MIFrame`'s
    /// `self.ctx: *mut TraceCtx` raw pointer deref; `WalkContext`'s
    /// `self.trace_ctx: &'frame mut TraceCtx` field).
    fn ctx_mut(&mut self) -> &mut TraceCtx;

    /// Immutable counterpart of [`ctx_mut`].  Used by `value_type`'s
    /// `get_opref_type` lookup which doesn't need write access.
    fn ctx(&self) -> &TraceCtx;

    /// `pyjitpl.py get_list_of_active_boxes` `Box.type` parity — return the OpRef's
    /// intrinsic type (Const kind, recorded result_type, or PtrInfo
    /// virtualized).  Default impl reads `ctx.get_opref_type`; impls
    /// override only if they carry a faster cached lookup.
    fn value_type(&self, value: OpRef) -> Type {
        if value.is_none() {
            return Type::Ref;
        }
        self.ctx().get_opref_type(value).unwrap_or(Type::Ref)
    }

    /// `pyjitpl.py` `generate_guard` parity — flush a pending
    /// quasi-immut guard, capture multi-frame resume snapshot, then
    /// record the guard op with its snapshot.  The single load-bearing
    /// method; impls diverge between `MIFrame` register state and
    /// `WalkContext` (walker register banks + dispatch
    /// snapshot helper).
    fn generate_guard(&mut self, opcode: OpCode, args: &[OpRef]);

    /// `pyjitpl.py` `implement_guard_value` parity — pick the
    /// const factory (`const_ref` for Type::Ref, `const_int`
    /// otherwise), record `GUARD_VALUE`, then update the heapcache's
    /// box replacement so downstream reads see the proved constant.
    fn implement_guard_value(&mut self, value: OpRef, expected: i64) {
        let ty = self.value_type(value);
        let expected_ref = match ty {
            Type::Ref => self.ctx_mut().const_ref(expected),
            _ => self.ctx_mut().const_int(expected),
        };
        self.generate_guard(OpCode::GuardValue, &[value, expected_ref]);
        // pyjitpl.py `replace_box` parity.
        self.ctx_mut()
            .heap_cache_mut()
            .replace_box(value, expected_ref);
    }

    /// `pyjitpl.py` `opimpl_guard_class` parity.  Skips the
    /// guard when the heapcache already knows the class or the OpRef
    /// is constant (the runtime type is already pinned).  Otherwise
    /// records `GUARD_CLASS` with `expected_type_const` and updates the
    /// heapcache's class+nullity record.  The obj is non-null by
    /// construction (a value-stack operand, the same invariant under
    /// which the codewriter emits guard_class at
    /// jtransform.py handle_getfield_typeptr); the optimizer
    /// strengthens a preceding `GUARD_NONNULL` into `GUARD_NONNULL_CLASS`
    /// (rewrite.py:408-444).
    fn guard_class(&mut self, obj: OpRef, expected_type: *const PyType) {
        if self.ctx().heap_cache().is_class_known(obj) {
            return;
        }
        if obj.is_constant() {
            // The const-arm `flush_guard_not_invalidated` in
            // `MIFrame::guard_class` (trace_opcode.rs) is skipped
            // here.  It only fires when a prior quasi-immut field read
            // set `pending_guard_not_invalidated_pc`; the
            // `store_subscr_value` precondition (concrete obj/key/value
            // are direct stack reads, not quasi-immut loads) excludes
            // that path on the walker leg.
            self.ctx_mut().heap_cache_mut().class_now_known(obj);
            return;
        }
        let expected_type_const = self.ctx_mut().const_int(expected_type as usize as i64);
        self.generate_guard(OpCode::GuardClass, &[obj, expected_type_const]);
        // heapcache.py `class_now_known` parity.
        self.ctx_mut().heap_cache_mut().class_now_known(obj);
    }

    /// `intobject.py` `int_intval`-pattern guard — class-guard the obj
    /// as `INT_TYPE`, read its `intval` field, then `implement_guard_value`
    /// the unboxed payload against `expected`.
    fn guard_int_object_value(&mut self, int_obj: OpRef, expected: i64) {
        if pyre_object::tagged_int::CAN_BE_TAGGED {
            if let Some(majit_ir::Value::Ref(r)) = self.ctx().concrete_of_opref(int_obj) {
                if r != majit_ir::GcRef::NO_CONCRETE {
                    let o = r.as_usize() as pyre_object::PyObjectRef;
                    if !o.is_null() {
                        if pyre_object::tagged_int::is_tagged_int(o) {
                            let lowbit =
                                crate::helpers::emit_tag_lowbit_test(self.ctx_mut(), int_obj, true);
                            self.generate_guard(OpCode::GuardTrue, &[lowbit]);
                            let untagged = crate::helpers::emit_untag_int(
                                self.ctx_mut(),
                                int_obj,
                                pyre_object::tagged_int::untag_int(o),
                            );
                            self.implement_guard_value(untagged, expected);
                            return;
                        } else {
                            let lowbit = crate::helpers::emit_tag_lowbit_test(
                                self.ctx_mut(),
                                int_obj,
                                false,
                            );
                            self.generate_guard(OpCode::GuardFalse, &[lowbit]);
                        }
                    }
                }
            }
        }
        self.guard_class(int_obj, &pyre_object::pyobject::INT_TYPE as *const PyType);
        let actual_value = crate::state::opimpl_getfield_gc_i(
            self.ctx_mut(),
            int_obj,
            crate::descr::int_intval_descr(),
        );
        self.implement_guard_value(actual_value, expected);
    }

    /// `listobject.py` strategy field guard — `getfield_gc_i(strategy)`
    /// then `implement_guard_value`.  Skips runtime W_ListObject layout
    /// reasoning by reading the strategy id directly from its descr.
    fn guard_list_strategy(&mut self, obj: OpRef, expected: i64) {
        let strategy = crate::state::opimpl_getfield_gc_i(
            self.ctx_mut(),
            obj,
            crate::descr::list_strategy_descr(),
        );
        self.implement_guard_value(strategy, expected);
    }

    /// `baseobjspace` subclass-override guard — a builtin SUBCLASS instance
    /// shares the payload `ob_type` (so it passes `guard_class`) but retags
    /// `w_class` and may override the dunder; guard the exact canonical
    /// `w_class` so such an instance side-exits to the generic residual
    /// (which honours the override) rather than a direct backing-storage
    /// access.  Skipped when `obj` is a freshly-allocated unescaped virtual
    /// (cannot be an external subclass instance) or the expected typeobj is
    /// null.  Walker-native counterpart: `walker_guard_exact_w_class`.
    fn guard_exact_w_class(&mut self, obj: OpRef, expected_typeobj: pyre_object::PyObjectRef) {
        if expected_typeobj.is_null() || self.ctx().heap_cache().is_unescaped(obj) {
            return;
        }
        let actual =
            crate::state::opimpl_getfield_gc_r(self.ctx_mut(), obj, crate::descr::w_class_descr());
        self.implement_guard_value(actual, expected_typeobj as i64);
    }
}

// `MIFrame` impl — delegates `generate_guard` to the existing
// `MIFrame::generate_guard` method so the retained helper keeps its current
// `flush_guard_not_invalidated` / `orgpc` plumbing untouched.
// The `ctx_mut` / `ctx` accessors materialise the
// borrow from `self.ctx: *mut TraceCtx` via unsafe deref.  The deref is
// sound because `MIFrame`'s lifetime invariant
// (`pyjitpl.py MIFrame.metainterp`) guarantees `self.ctx` always
// points at the live `TraceCtx` owned by the enclosing dispatch frame.
impl WalkerFrameOps for crate::state::MIFrame {
    fn ctx_mut(&mut self) -> &mut TraceCtx {
        unsafe { &mut *self.ctx }
    }

    fn ctx(&self) -> &TraceCtx {
        unsafe { &*self.ctx }
    }

    fn generate_guard(&mut self, opcode: OpCode, args: &[OpRef]) {
        // Re-borrow `ctx` from the raw pointer (rather than
        // `self.ctx_mut()` which holds `&mut self`) so the
        // `&mut MIFrame` borrow Rust's `MIFrame::generate_guard` needs
        // is disjoint from the `&mut TraceCtx` arg it also needs.
        // `self.ctx` is `*mut TraceCtx` so dereferencing it produces a
        // borrow whose lifetime is independent of `self`'s.
        let ctx_ptr: *mut TraceCtx = self.ctx;
        let ctx_ref: &mut TraceCtx = unsafe { &mut *ctx_ptr };
        crate::state::MIFrame::generate_guard(self, ctx_ref, opcode, args)
    }
}

// `WalkContext` impl — `generate_guard` delegates to the existing
// `walker_capture_snapshot_for_last_guard` snapshot helper after
// recording the guard op via `ctx.record_guard`.  The walker has no
// quasi-immut field reads in its STORE_SUBSCR path (concrete obj/key/value
// are direct register reads, not field loads), so the
// `flush_guard_not_invalidated` step that `MIFrame::generate_guard`
// performs is intentionally omitted here.
impl<'frame, 'static_a: 'frame, Sym: crate::state::WalkSym> WalkerFrameOps
    for crate::jitcode_dispatch::WalkContext<'frame, 'static_a, Sym>
{
    fn ctx_mut(&mut self) -> &mut TraceCtx {
        self.trace_ctx
    }

    fn ctx(&self) -> &TraceCtx {
        self.trace_ctx
    }

    fn generate_guard(&mut self, opcode: OpCode, args: &[OpRef]) {
        // `pyjitpl.py generate_guard` const-skip parity.
        if let Some(&first) = args.first() {
            if first.is_constant() {
                return;
            }
        }
        // Fill virtualizable holes before the guard is recorded so
        // `capture_resumedata` numbers only boxes that already exist
        // (`pyjitpl.py MetaInterp.generate_guard`).
        self.trace_ctx
            .fill_virtualizable_boxes_to_declared_layout(None);
        self.trace_ctx.record_guard(opcode, args, 0);
        // `walker_capture_snapshot_for_last_guard` returns a typed abort
        // (`GuardSnapshotVableUntyped`) for the multi-frame vable case the
        // full-body walk routes through `DispatchError`. The `()` trait
        // signature (shared with the retained `MIFrame` impl) has no
        // error channel, so latch the first failure on the context; the
        // STORE_SUBSCR specialization's dispatcher call site drains it
        // and aborts the walk — a guard without a resume snapshot must
        // not reach the compile pipeline.
        if let Err(e) = crate::jitcode_dispatch::walker_capture_snapshot_for_last_guard(self, 0) {
            if self.pending_guard_snapshot_error.is_none() {
                self.pending_guard_snapshot_error = Some(e);
            }
        }
    }
}
