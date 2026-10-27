use pyre_object::PyObjectRef;
use pyre_object::quasiimmut::{
    QuasiImmut, QuasiImmutField, QuasiImmutLock, null_mutate_lock, null_mutate_word,
};
use std::cell::{Cell, RefCell};
use std::sync::OnceLock;
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::PyFrame;

pub type ForceFrameFn = unsafe extern "C" fn(*mut PyFrame);

/// Process-wide hook installed by JIT init (`eval.rs force_pyframe`).
/// A runtime load, not a `OnceLock`: the unset word must stay a runtime
/// read so translated `force_frame` is not folded to a permanent no-op.
static FORCE_FRAME_HOOK: std::sync::atomic::AtomicPtr<()> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

pub fn register_force_frame_hook(f: ForceFrameFn) {
    FORCE_FRAME_HOOK.store(f as *mut (), Ordering::Release);
}

/// Restores the previous process-wide hook when dropped. Tests that
/// overwrite `FORCE_FRAME_HOOK` must hold this so a panic still clears it.
pub struct ForceFrameHookGuard {
    prev: *mut (),
}

impl Drop for ForceFrameHookGuard {
    fn drop(&mut self) {
        FORCE_FRAME_HOOK.store(self.prev, Ordering::Release);
    }
}

/// Install `f` and restore the previous hook when the guard drops.
#[must_use]
pub fn install_force_frame_hook(f: ForceFrameFn) -> ForceFrameHookGuard {
    ForceFrameHookGuard {
        prev: FORCE_FRAME_HOOK.swap(f as *mut (), Ordering::AcqRel),
    }
}

/// `rvirtualizable.py hook_access_field` → `jit_force_virtualizable`:
/// materialize a frame's virtualizable fields before something outside the JIT
/// reads them.
///
/// Upstream emits this per redirected FIELD ACCESS, and
/// `jtransform.rewrite_op_jit_force_virtualizable` deletes it again in every
/// graph the codewriter looks inside — only the graphs the JIT cannot see keep
/// the call (`rvirtualizable.replace_force_virtualizable_with_call`).  So the
/// frame-chain walk itself never forces upstream; the consumer that reads a
/// field does.  Call sites here are therefore the consumers, not the walkers:
/// a walk that only follows `f_backref` and tests `hide()` reads nothing the
/// JIT keeps virtual, and forcing there escapes the traced virtualizable for
/// every frame-walking helper — `pyjitpl.vable_after_residual_call` then
/// aborts the whole trace with `ABORT_ESCAPE`.
///
/// Residual by policy: upstream `hook_access_field` rewrites the field
/// access at translation time. The 17 `force_frame` subjects stay residual
/// until the front emits `jit_force_virtualizable` at those sites.
#[inline]
#[majit_macros::dont_look_inside]
pub fn force_frame(frame: *mut PyFrame) {
    let p = FORCE_FRAME_HOOK.load(Ordering::Acquire);
    if !p.is_null() {
        let f: ForceFrameFn = unsafe { std::mem::transmute(p) };
        unsafe { f(frame) };
    }
}

/// Force a frame whose fastlocals application code is about to read.
///
/// RPython needs no such call because the force is injected for it, not
/// because none happens: `rvirtualizable.py hook_access_field` genops
/// `jit_force_virtualizable` on every redirected FIELD access,
/// `virtualizable.py:288-292` rewrites those into a
/// `force_virtualizable_if_necessary` call across `translator.graphs`, and
/// `jtransform.py rewrite_op_jit_force_virtualizable` drops it again
/// only in the graphs the codewriter looks inside.  So `pyframe.py:539
/// fast2locals` always finds a materialized `locals_cells_stack_w` with no
/// hand-placed hook anywhere.
///
/// That injection does not exist here, and not because a piece of the rtyper
/// is unported — `FieldListAccessor` and `_parse_field_list` both are.
/// Upstream discovers the redirected set from `classdesc.get_param('_virtualizable_')`,
/// and no `ClassDesc` in this tree ever carries that key: pyre's virtualizable
/// is the hand-written Rust `PyFrame`, and its field set is declared out of
/// band in `pyre-jit-trace/src/virtualizable_spec.rs`.  So `rclass.rs
/// buildinstancerepr`'s virtualizable arm is unreachable rather than blocked,
/// and it is kept fail-closed for the day a producer appears.  Explicit calls
/// like this one stand in for the injection — load-bearing, not redundant.
/// Deleting them was measured: `f_lineno` starts reporting the `def` line,
/// `f_locals` grows an entry, and the two read together segfault, with the
/// whole synthetic corpus green throughout.
///
/// [`PyExecutionContext::gettopframe_nohidden`] does not substitute.  It forces
/// the VREF of the frame it starts from and then walks `f_backref` unforced, and
/// the vref force alone does not materialize `locals_cells_stack_w`: measured,
/// `locals()` on a compiled frame reached that way reports the values the frame
/// last wrote out, under correct keys.  So the top frame needs this call as much
/// as one handed out some other way (a traceback's `tb_frame`) does.
///
/// # Safety
/// `frame` must be a live `PyFrame` (or null).
pub fn force_frame_before_locals_read(frame: *mut PyFrame) {
    if !frame.is_null() {
        force_frame(frame);
    }
}

/// Stand-in for `rvirtualizable.py hook_access_field`'s
/// `llops.genop('jit_force_virtualizable', [vinst, cname, cflags])`.
///
/// `jtransform.py rewrite_op_jit_force_virtualizable` deletes this call in
/// every graph the codewriter looks inside (`return []`).  Residual calls
/// still run the body, which is the `replace_force_virtualizable_with_call`
/// half: interpreter-visible graphs keep `force_virtualizable_if_necessary`.
/// `#[inline(never)]` keeps the callee name on the Call so the rewrite can
/// see it.
///
/// # Reach: gateways, not redirected-array readers
///
/// Manual frame getset gateways publish their bodies through
/// `gateway::BUILTIN_WRAPPER_DESCRIPTORS`, the Rust counterpart of
/// `BuiltinCode.func`'s SomePBC family.  The untranslated interpreter retains
/// the force, while `jtransform` deletes it from every gateway graph admitted
/// by the codewriter — the same two populations `hook_access_field` creates
/// upstream.
///
/// Which gateways carry it starts from `hook_access_field`'s own condition,
/// `if self.my_redirected_fields.get(cname.value)` — and the redirected set is
/// the one `pyre-jit-trace` `virtualizable_gen.rs` declares, not
/// `interp_jit.py`'s string list: `last_instr`, `pycode`, `valuestackdepth`,
/// `debugdata` and the `locals_cells_stack_w` array (`w_globals` is skipped;
/// no `PyFrame` field answers to it).  `pyframe.rs`
/// `__majit_wrap_descr_typecheck_fget_f_back` and `..._fget_f_builtins` read
/// `f_backref` and `w_builtin`, so they carry none.
///
/// Residual `typeobject.py ensure_module_attr` (inside
/// `type_create_new_type`) calls [`force_virtualizable_if_necessary`]
/// before `get_w_globals`: during tracing that is the
/// `TOKEN_TRACING_RESCALL` → `TOKEN_NONE` escape marker, not a
/// question of whether the heap slot can disagree with the shadow.
/// `get_w_globals` / `getdebug` / `getcode` omit it: pyre residualizes
/// LOAD_GLOBAL / STORE_NAME / MAKE_FUNCTION through those reads while
/// `pyopcode.py` looks those opcodes inside, so a native force on the
/// accessor would abort loops pypy compiles. The `f_code` gateway
/// still omits the escape-flush marker; that placement is at its own
/// definition.
///
/// Upstream reaches the rewrite because `hook_access_field` puts the marker
/// at every redirected FIELD access, which lands it inside graphs the
/// codewriter does look into.  A hand-placed marker only reaches it if the
/// function holding it is one of those graphs, so the candidates are the
/// functions that READ the redirected array and that the codewriter looks
/// inside.  Measured against `jit_metadata.json`, that is two of them:
/// `pyframe.rs` `PyFrame::fast2locals` and `PyFrame::frame_locals_proxy_snapshot`,
/// both `@jit.unroll_safe` so that `contains_loop` does not keep the
/// codewriter out.  `FrameLocalsProxy::locals_plus_value` reads the same
/// array and became a third candidate when it took the same hint.
///
/// Placing it at those two was tried and does not pay.  The forces are
/// deleted in the traced copies, but the fixtures that reach `f_locals`
/// reach it through a residual call chain, and the residual copy escapes:
/// five gate rows lose compilations (`getframe_bridge_force_*_declined`
/// `loops_aborted 0 -> 20` with `bridges_compiled 1 -> 0`,
/// `getframe_root_loop_force_blackhole_crn*` `loops_compiled 1 -> 0`,
/// `raise_reg_unbound_jitstress` `9 -> 7`) and NOT ONE output changes.
/// Nothing changes because pyre already materializes the array for a
/// residual read by another route — `pyjitpl.py
/// vable_and_vrefs_before_residual_call` stores the boxes back before the
/// call — which is why `flocals_read_through_bound_proxy` passes without any
/// force.  The `f_locals` path is also already forced one frame up, by the
/// `__majit_wrap_descr_typecheck_fget_getdictscope` gateway, so a marker at the readers
/// is redundant where it would matter and new only where it costs.
///
/// So the redirected-array readers carry no force.
///
/// Non-gateway force sites call [`force_frame`] or
/// [`force_frame_before_locals_read`] directly, and only one of them sits in
/// a graph that has a jitcode — `pyframe.rs`
/// `FrameLocalsProxy::force_locals`. It wants the force to survive there:
/// it carries the `vable_token` test itself. In the rest
/// the codewriter never looks inside, so a direct call and a deleted marker
/// come to the same thing.
///
/// Residual redirected-field reads that pypy also residualizes use
/// [`force_virtualizable_if_necessary`] instead: that helper *does* open
/// with the token test, matching the function
/// `replace_force_virtualizable_with_call` installs, and is the one
/// `type_create_new_type`'s `ensure_module_attr` arm calls.
///
/// # No `vable_token` test here
///
/// `virtualizable.py force_virtualizable_if_necessary` opens with
/// `if virtualizable.vable_token:`, and this function deliberately does not.
/// The test is not missing — it is downstream, inside the backend hook
/// [`force_frame`] reaches (`pyre-jit` `force_pyframe`, which calls
/// `force_virtualizable_if_necessary` with the token read). What runs BEFORE
/// that hook tests anything is `flush_active_frame_escape`, and that ordering
/// is pyre's escape signal rather than an oversight.
///
/// Repeating the test at this call site was measured and is wrong: it gates
/// the escape flush too, so a walk that must abort no longer does.  Eight
/// gate rows moved on it — seven `getframe_*_declined` fixtures lost the
/// decline they exist to pin (`loops_aborted 5 -> 0` with
/// `guard_failures 0 -> 14480`), and
/// `exception_reused_object_tb_not_doubled` started reporting a second
/// traceback shape whose `f_locals` is missing the `except`-bound name,
/// because the uncommitted walk kept the node it had already attached.
#[inline(never)]
pub fn jit_force_virtualizable(frame: *mut PyFrame) {
    force_frame_before_locals_read(frame);
}

/// `virtualizable.py force_virtualizable_if_necessary`:
///
/// ```text
/// def force_virtualizable_if_necessary(virtualizable):
///     if virtualizable.vable_token:
///         force_now(virtualizable)
/// ```
///
/// `rvirtualizable.py hook_access_field` genops `jit_force_virtualizable` on
/// every redirected field access; `replace_force_virtualizable_with_call`
/// rewrites the residual copies into this helper; `jtransform.py
/// rewrite_op_jit_force_virtualizable` deletes the Call in graphs the
/// codewriter looks inside. Native residual execution has no rewritten
/// graph, so `type_create_new_type`'s `ensure_module_attr` arm — the
/// residual graph that keeps the Call — calls this directly.
///
/// `force_now` on `TOKEN_TRACING_RESCALL` only stores `TOKEN_NONE` — the
/// values are already correct during tracing, and that store is the escape
/// marker `tracing_after_residual_call` reads. An Active token materializes
/// through [`force_frame_before_locals_read`]. TOKEN_NONE is a no-op, so
/// interpreted code that is not in a residual-from-trace does not enter
/// the backend hook.
///
/// `#[inline(never)]` keeps the callee name on the Call so
/// `rewrite_op_jit_force_virtualizable` can delete it from looked-inside
/// graphs. Distinct from [`jit_force_virtualizable`]: that stand-in is the
/// gateway/escape-flush path and must not test the token first.
#[inline(never)]
pub fn force_virtualizable_if_necessary(frame: *mut PyFrame) {
    if frame.is_null() {
        return;
    }
    let token = unsafe { (*frame).vable_token };
    if token == 0 {
        return;
    }
    if token == majit_metainterp::virtualref::token_tracing_rescall() as usize {
        unsafe { (*frame).vable_token = 0 };
        return;
    }
    force_frame_before_locals_read(frame);
}

/// `_jit_vref.py` `vref()` = `jit_force_virtual`.
///
/// `topframeref` and every frame's `f_backref` hold a `jit.virtual_ref` — at
/// interpreter level (`_jit_vref.py VRefRepr.lowleveltype = OBJECTPTR`) that
/// is just the frame pointer itself, so forcing is the identity.  Once the JIT
/// virtualizes an inlined callee, the slot instead holds a `JitVirtualRef`
/// whose `virtualref.py force_virtual_if_necessary` materializes the real
/// frame.  The JIT installs that materialization via
/// [`register_force_vref_hook`]; interp-level code never stores a
/// `JitVirtualRef`, so `is_virtual_ref` is always false and this stays the
/// identity fast path.
pub type ForceVRefFn = unsafe extern "C" fn(*mut PyFrame) -> *mut PyFrame;

/// Process-wide hook installed by JIT init (`eval.rs force_pyframe_vref`).
/// A runtime load, not a `OnceLock`: the unset word must stay a runtime
/// read so a translated vref-arm `force_vref` is not folded to a
/// permanent panic.  A later register overwrites; null is unset.
static FORCE_VREF_HOOK: std::sync::atomic::AtomicPtr<()> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

pub fn register_force_vref_hook(f: ForceVRefFn) {
    FORCE_VREF_HOOK.store(f as *mut (), Ordering::Release);
}

/// The frame a chain slot NAMES, read WITHOUT forcing —
/// `virtualref.py force_virtual`'s trailing `return vref.forced`.
///
/// Exact for the whole recording walk: `virtual_ref_during_tracing` writes
/// `forced = real_object` at allocation and only `continue_tracing` ever
/// rewrites it.  Null when the vref is still virtual with nothing
/// materialized — a live compiled frame — so callers must read that as
/// "names no reachable frame", never as a match.
///
/// For identity tests only, never for handing a frame to application code.
/// Forcing would be wrong here, not merely expensive: a live vref carries
/// `TOKEN_TRACING_RESCALL` across a residual, `force_virtual` clears it, and
/// that cleared token is the one marker `tracing_after_residual_call` reads as
/// "the callee forced this vref".  A reader that forced would report its own
/// read as a callee escape.
#[inline]
pub fn vref_referent(ptr: *mut PyFrame) -> *mut PyFrame {
    if unsafe { majit_metainterp::virtualref::ptr_is_virtual_ref(ptr as majit_gc::header::GCREF) } {
        unsafe {
            majit_metainterp::virtualref::vref_forced(ptr as majit_gc::header::GCREF)
                as *mut PyFrame
        }
    } else {
        ptr
    }
}

/// An activation's pending result held in root slots across a call that can
/// collect: the exit value, or the `PyError` handle. The name and object
/// context live on the GC object the handle points at.
///
/// `execute_frame`'s `finally` and `ExecutionContext.leave` keep the exit value
/// and the propagating `OperationError` in RPython locals, which the GC
/// transform roots across every collecting call.  Here they are Rust copies,
/// so the forcing half of `leave` would hand back a pre-collection address.
pub struct PinnedResult {
    roots: pyre_object::gc_roots::RootScope,
    base: usize,
}

impl PinnedResult {
    pub fn pin(result: &mut crate::PyResult) -> Self {
        let roots = pyre_object::gc_roots::push_roots();
        let base = match result {
            Ok(w_value) => roots.pin_roots(&[*w_value]),
            Err(err) => err.pin(&roots),
        };
        Self { roots, base }
    }

    pub fn reload(&self, result: &mut crate::PyResult) {
        match result {
            Ok(w_value) => *w_value = self.roots.get(self.base),
            Err(err) => err.reload(&self.roots, self.base),
        }
    }
}

/// Force a vref stored in the frame chain (`topframeref` / `f_backref`).
/// `virtualref.py force_virtual_if_necessary`: `if inst.typeptr !=
/// jit_virtual_ref_vtable: return inst` (the pointer already *is* the frame)
/// else materialize via `force_virtual`.
///
/// `support.py _ll_1_jit_force_virtual` owns `@oopspec("jit.force_virtual(inst)")`.
/// `jtransform.py _handle_jit_call` turns that into a residual with
/// `OS_JIT_FORCE_VIRTUAL` + `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`;
/// `pyjitpl.py do_residual_call` then runs `_do_jit_force_virtual`.
/// Keep `dont_look_inside`: the typeptr test is `vref.virtual`, which
/// `rewrite_op_jit_is_virtual` refuses from jit-visible code, and
/// `ptr_is_virtual_ref` lives in `majit-metainterp`, which is not lowered
/// to LLBC. `inline(never)` keeps the call so Charon sees the oopspec
/// residual instead of inlining the typeptr test into the caller.
///
/// The `FORCE_VREF_HOOK` arm is an indirect call through a runtime fn
/// pointer, so that half is unwalkable in any case.
#[inline(never)]
#[majit_macros::oopspec("jit.force_virtual(inst)")]
#[majit_macros::dont_look_inside]
pub fn force_vref(inst: *mut PyFrame) -> *mut PyFrame {
    if unsafe { majit_metainterp::virtualref::ptr_is_virtual_ref(inst as majit_gc::header::GCREF) }
    {
        // Only the tracer stores a `JitVirtualRef` here, and it registers the
        // hook when the driver comes up, so an unset hook means the slot was
        // misidentified.  Returning `inst` would hand a vref out as a frame.
        let p = FORCE_VREF_HOOK.load(Ordering::Acquire);
        assert!(
            !p.is_null(),
            "frame-chain vref with no force hook: the JIT never came up"
        );
        let f: ForceVRefFn = unsafe { std::mem::transmute(p) };
        unsafe { f(inst) }
    } else {
        inst
    }
}

/// Return the live `PyFrame` as the Python-visible `frame` object passed
/// to a trace / profile callback.
///
/// `pyframe.py:_trace` hands the callback the real frame
/// (`jit.hint(frame, access_directly=False)`).  Now that `PyFrame` is a
/// W_Root (`FRAME_TYPE` typedef), pyre does the same: the callback's
/// `frame.f_lineno = N` / `frame.f_trace = local` land directly on the
/// live frame's getsets, so no `sys.namespace` wrapper and no
/// writeback pass is needed.  The frame is the current executing frame
/// (on the `CURRENT_FRAME` chain and thus GC-rooted) for the whole
/// callback, so returning it is safe; mark it escaped so the JIT keeps
/// it materialised while the callback holds a reference
/// (`pyframe.py mark_as_escaped`).
fn wrap_trace_frame(frame: *mut PyFrame) -> PyObjectRef {
    if frame.is_null() {
        return pyre_object::w_none();
    }
    unsafe { (*frame).mark_as_escaped() };
    frame as PyObjectRef
}

/// pypy/interpreter/executioncontext.py app_profile_call.
///
/// Module-level low-level trampoline used by `setprofile` to bridge
/// `setllprofile` (which stores a function pointer) to the user
/// callable held in `w_profilefuncarg`.  RPython:
///
/// ```python
/// def app_profile_call(space, w_callable, frame, event, w_arg):
///     frame = jit.hint(frame, access_directly=False)
///     space.call_function(w_callable, frame, space.newtext(event), w_arg)
/// ```
///
/// `jit.hint(frame, access_directly=False)` is a JIT-only annotation
/// that demotes the green frame to a regular w_object before the user
/// callable observes it.  Pyre's internal `PyFrame` is not itself a
/// `PyObject`, so the callback sees a frame wrapper carrying the same
/// Python-visible fields instead of a raw struct pointer.
pub fn app_profile_call(
    _space: PyObjectRef,
    w_callable: PyObjectRef,
    frame: *mut PyFrame,
    event: &str,
    w_arg: PyObjectRef,
) -> Result<(), crate::PyError> {
    // executioncontext.py — `frame = jit.hint(frame, access_directly=False)`:
    // from here on, frame is just a normal w_object.
    let frame = majit_metainterp::jit::hint_no_access_directly(frame);
    let _roots = pyre_object::gc_roots::push_roots();
    let callable_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_callable);
    let frame_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(wrap_trace_frame(frame));
    let event_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new(event));
    let arg_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_arg);
    crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(callable_slot),
        &[
            pyre_object::gc_roots::shadow_stack_get(frame_slot),
            pyre_object::gc_roots::shadow_stack_get(event_slot),
            pyre_object::gc_roots::shadow_stack_get(arg_slot),
        ],
    )
    .map(|_| ())
}

/// pypy/interpreter/executioncontext.py:320 `self.profilefunc` — the
/// low-level callback stored by `setllprofile`.  In RPython this is a
/// raw function pointer; pyre mirrors the type with a `fn(...)` so the
/// trampoline (`app_profile_call`) is the actual stored value rather
/// than the user callable, matching upstream's two-slot design
/// (`profilefunc` + `w_profilefuncarg`).
pub type ProfileFunc = fn(
    space: PyObjectRef,
    w_callable: PyObjectRef,
    frame: *mut PyFrame,
    event: &str,
    w_arg: PyObjectRef,
) -> Result<(), crate::PyError>;

pub const TICK_COUNTER_STEP: usize = 100;

#[derive(Debug, Default)]
pub struct WRootFinalizerQueue;

static W_ROOT_FINALIZER_QUEUE: WRootFinalizerQueue = WRootFinalizerQueue;

impl WRootFinalizerQueue {
    pub fn finalizer_trigger(&mut self) {
        finalizer_queue_trigger();
    }

    fn fq_tag() -> pyre_object::gc_hook::GcFinalizerQueueTag {
        pyre_object::gc_hook::GcFinalizerQueueTag {
            fq: std::ptr::addr_of!(W_ROOT_FINALIZER_QUEUE) as usize,
            trigger: finalizer_queue_trigger,
        }
    }

    /// framework.py `get_finalizer_queue_index`. `tag.fq` is the address of
    /// this queue's static instance. The index lives on the process-wide
    /// table, not on a collector instance.
    fn fq_index() -> usize {
        pyre_object::gc_hook::try_gc_get_finalizer_queue_index(Self::fq_tag())
    }

    /// `pypy/interpreter/executioncontext.py:640` —
    /// `self.space.finalizer_queue.next_dead()`.
    ///
    /// Returns the next `w_obj` whose finalizer should run, or `None`
    /// when the death queue is empty.
    ///
    pub fn next_dead(&mut self) -> Option<PyObjectRef> {
        let obj = pyre_object::gc_hook::try_gc_finalizer_next_dead(Self::fq_index());
        (!obj.is_null()).then_some(obj)
    }
}

/// How many times the collector has run the death-queue trigger below.
///
/// The collector skips a queue whose deque is empty
/// (`majit-gc` `Collector::execute_finalizer_triggers`), so a collection bumps
/// this exactly when that collection put something finalizable on the queue.
/// Reading it either side of a collection is therefore the collector's own
/// answer to "did that sweep find anything to finalize", which is why the
/// drain it feeds keeps its upstream shape and reports nothing.
static FINALIZER_TRIGGER_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Read [`FINALIZER_TRIGGER_COUNT`].
pub fn finalizer_trigger_count() -> usize {
    FINALIZER_TRIGGER_COUNT.load(Ordering::Relaxed)
}

fn finalizer_queue_trigger() {
    FINALIZER_TRIGGER_COUNT.fetch_add(1, Ordering::Relaxed);
    let action = space_user_del_action();
    if !action.is_null() {
        unsafe { (*action).fire() };
    }
}

// `baseobjspace.py:450` / `executioncontext.py:724`: both the action flag and
// the user-finalizer action belong to the object space, not to an individual
// execution context.  Pyre does not yet expose a typed Rust `ObjSpace`, so the
// process-global runtime instance is its storage owner.  ECs carry a thin
// reference to the flag, while finalizer users resolve the action here just as
// PyPy reaches `self.space.user_del_action`.
static SPACE_USER_DEL_ACTION: OnceLock<usize> = OnceLock::new();

fn install_space_user_del_action(
    space: PyObjectRef,
    actionflag: &mut (dyn ActionFlagOps + 'static),
) -> *mut UserDelAction {
    *SPACE_USER_DEL_ACTION
        .get_or_init(|| Box::into_raw(UserDelAction::new(space, actionflag)) as usize)
        as *mut UserDelAction
}

#[inline(never)]
#[majit_macros::dont_look_inside]
pub fn space_user_del_action() -> *mut UserDelAction {
    SPACE_USER_DEL_ACTION.get().copied().unwrap_or(0) as *mut UserDelAction
}

/// Trace the object-space-owned finalizer action exactly once as a non-stack
/// root.  In translated PyPy this happens through the object-space graph; the
/// leaked Rust allocation is outside that graph, so its managed slots must be
/// forwarded explicitly.
pub(crate) fn walk_space_user_del_action_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let action = space_user_del_action();
    if action.is_null() {
        return;
    }
    let action = unsafe { &mut *action };
    let mut forward = |slot: &mut PyObjectRef| {
        if !slot.is_null() {
            visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
        }
    };
    forward(&mut action.base.space);
    if let Some(pending) = action.pending_with_disabled_del.as_mut() {
        for obj in pending {
            forward(obj);
        }
    }
}

/// `baseobjspace.py W_Root.register_finalizer` — the single entry
/// point through which a construction site asks for `_finalize_` to run once
/// `obj` becomes unreachable.
///
/// Upstream's contract is "call this at most once"; it holds structurally
/// there, because every requester sits in the object's own constructor. pyre
/// follows that ownership too: `WeakrefLifeline.enable_callbacks` registers
/// the lifeline itself, while instance/generator/PickleBuffer construction
/// registers its own object. The collector additionally keeps its duplicate
/// guard as a fail-safe for repeated construction hooks.
// `rgc.py FinalizerQueue.register_finalizer` carries `@jit.dont_look_inside`.
// Its collector hook is runtime state, so the translated trace keeps one
// residual call, as for [`may_ignore_finalizer`].
#[majit_macros::dont_look_inside]
pub fn register_finalizer(obj: PyObjectRef) {
    pyre_object::gc_hook::try_gc_register_finalizer(WRootFinalizerQueue::fq_index(), obj);
}

/// objspace.py:486-487 `allocate_instance`:
/// `if w_subtype.hasuserdel: self.finalizer_queue.register_finalizer(instance)`.
pub fn maybe_register_user_finalizer(obj: PyObjectRef) {
    let Some(w_type) = crate::typedef::r#type(obj) else {
        return;
    };
    if unsafe { pyre_object::w_type_get_hasuserdel(w_type.as_ptr()) } {
        register_finalizer(obj);
    }
}

/// `rgc.may_ignore_finalizer` — "Optimization hint: says that it is valid for
/// any finalizer for 'obj' to be ignored, depending on the GC."  The object
/// stays on the queue it was registered with; what changes is that the queue
/// passes over it when the collector comes to deliver.
///
/// Only ever a hint, so a build with no collector behind it drops the call and
/// the finalizer runs — the same answer a collector without the flag gives.
///
/// `W_Root.may_unregister_rpython_finalizer` — the same hint behind a
/// `hasuserdel` guard — has no pyre caller: every module that reaches it
/// upstream (`_socket`, `zlib`, `select.epoll`/`kqueue`, `_cffi_backend`)
/// registers no finalizer here, so there is nothing for it to withhold.  `_io`
/// wants the other guard and carries its own
/// (`_io::maybe_unregister_rpython_finalizer_io`).
// `rgc.py`'s `may_ignore_finalizer` carries `@jit.dont_look_inside`.  Its
// collector hook is runtime state, so the translated trace must retain one
// residual call instead of reading the process-global hook while translating.
#[majit_macros::dont_look_inside]
pub fn may_ignore_finalizer(obj: PyObjectRef) {
    pyre_object::gc_hook::try_gc_ignore_finalizer(obj);
}

/// Shared execution context for all frames in one interpreter run.
///
/// Holds the builtin module dict used by module-level frames.
#[derive(Debug, Clone)]
pub enum PendingLoopExit {
    Done(crate::PyResult),
    ContinueRunningNormally,
    ExitFrameWithException(crate::PyError),
}

/// Clone of the `w_tracefunc?` hidden word pair that starts empty.  A copied
/// execution context must not keep the source's compiled-loop watchers.
///
/// `repr(C)` two-word wrapper: the field offset of
/// `ExecutionContext.mutate_w_tracefunc` is the mutate word
/// (`quasiimmut.py get_mutate_field_name` of pyre `w_tracefunc`).
#[repr(C)]
struct WTracefuncWatchers {
    mutate_w_tracefunc: AtomicPtr<QuasiImmut>,
    mutate_w_tracefunc_lock: AtomicPtr<QuasiImmutLock>,
}

pyre_object::quasi_immut_words!(
    WTracefuncWatchers,
    mutate_w_tracefunc,
    mutate_w_tracefunc_lock
);

impl WTracefuncWatchers {
    #[inline]
    fn field(&self) -> QuasiImmutField<'_> {
        QuasiImmutField::new(&self.mutate_w_tracefunc, &self.mutate_w_tracefunc_lock)
    }
}

/// Twin of [`WTracefuncWatchers`] for `profilefunc?`. The field offset of
/// `ExecutionContext.mutate_profilefunc` is the mutate word
/// (`quasiimmut.py get_mutate_field_name` of pyre `profilefunc`).
#[repr(C)]
struct ProfilefuncWatchers {
    mutate_profilefunc: AtomicPtr<QuasiImmut>,
    mutate_profilefunc_lock: AtomicPtr<QuasiImmutLock>,
}

pyre_object::quasi_immut_words!(
    ProfilefuncWatchers,
    mutate_profilefunc,
    mutate_profilefunc_lock
);

impl ProfilefuncWatchers {
    #[inline]
    fn field(&self) -> QuasiImmutField<'_> {
        QuasiImmutField::new(&self.mutate_profilefunc, &self.mutate_profilefunc_lock)
    }
}

const _: () =
    assert!(std::mem::size_of::<WTracefuncWatchers>() == 2 * std::mem::size_of::<usize>());
const _: () =
    assert!(std::mem::size_of::<ProfilefuncWatchers>() == 2 * std::mem::size_of::<usize>());

impl Default for WTracefuncWatchers {
    fn default() -> Self {
        Self {
            mutate_w_tracefunc: null_mutate_word(),
            mutate_w_tracefunc_lock: null_mutate_lock(),
        }
    }
}

impl Clone for WTracefuncWatchers {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl Drop for WTracefuncWatchers {
    fn drop(&mut self) {
        // SAFETY: `&mut self` is exclusive and drop glue runs once.
        unsafe { self.field().reclaim() };
    }
}

impl Default for ProfilefuncWatchers {
    fn default() -> Self {
        Self {
            mutate_profilefunc: null_mutate_word(),
            mutate_profilefunc_lock: null_mutate_lock(),
        }
    }
}

impl Clone for ProfilefuncWatchers {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl Drop for ProfilefuncWatchers {
    fn drop(&mut self) {
        // SAFETY: `&mut self` is exclusive and drop glue runs once.
        unsafe { self.field().reclaim() };
    }
}

/// `executioncontext.py ExecutionContext._immutable_fields_` lists
/// `w_tracefunc?` and `profilefunc?`. The struct is `repr(Rust)`;
/// `EC_DESCR_GROUP` ranks fields by `offset_of!`, not declaration order.
#[majit_macros::jit_immutable_fields("w_tracefunc?", "profilefunc?")]
#[derive(Clone)]
pub struct ExecutionContext {
    pub space: PyObjectRef,
    pub topframeref: *mut PyFrame,
    pub w_tracefunc: PyObjectRef,
    /// Hidden `mutate_w_tracefunc` (`quasiimmut.py get_mutate_field_name` of
    /// pyre `w_tracefunc`; `executioncontext.py _immutable_fields_ =
    /// ['w_tracefunc?']`). [`Clone`] yields an empty field so a copied EC
    /// does not inherit the source's loop watchers. The wrapper is
    /// `repr(C)`, so this offset is the mutate word.
    mutate_w_tracefunc: WTracefuncWatchers,
    pub is_tracing: i32,
    pub compiler: PyObjectRef,
    /// pypy/interpreter/executioncontext.py:320 — function pointer to
    /// the low-level profiling trampoline (e.g. `app_profile_call`).
    /// `None` means profiling is disabled.  The user callable lives in
    /// `w_profilefuncarg`; the trampoline forwards to it.
    pub profilefunc: Option<ProfileFunc>,
    /// Hidden `mutate_profilefunc` (`quasiimmut.py get_mutate_field_name` of
    /// pyre `profilefunc`; `executioncontext.py _immutable_fields_ =
    /// ['profilefunc?']`). The wrapper is `repr(C)`, so this offset is
    /// the mutate word.
    mutate_profilefunc: ProfilefuncWatchers,
    pub w_profilefuncarg: PyObjectRef,
    pub thread_disappeared: bool,
    pub w_async_exception_type: PyObjectRef,
    /// `threadlocals.py ExecutionContext._signals_enabled` — only the
    /// execution context owned by `OSThreadLocals._mainthreadident` may run
    /// app-level signal handlers.  A fork from another thread promotes that
    /// thread's EC in `thread::after_fork_child`.
    pub signals_enabled: i32,
    /// Last process-wide all-threads hook generations applied by this EC.
    /// Only the owning OS thread writes these fields.
    pub trace_all_generation: usize,
    pub profile_all_generation: usize,
    /// `os_local.py:ExecutionContext._thread_local_objs`: weak references to
    /// `_thread._local` objects which own a dictionary for this EC.
    pub thread_local_refs: Vec<PyObjectRef>,
    /// Cached access to process-owned `space.actionflag`.  Every EC points to
    /// the same flag, matching `baseobjspace.py:450`.
    pub actionflag: SpaceActionFlag,
    /// `pypy/objspace/std/dictmultiobject.py:60-69
    /// allocate_and_init_instance(module=True)` parity — the builtins
    /// module's `w_dict` is a `W_ModuleDictObject` backed by
    /// `ModuleDictStrategy` (`celldict.py`).  Populated once at
    /// construction time; pinned with `pin_root` so the strategy
    /// storage survives the EC's lifetime.
    builtins_module: PyObjectRef,
    // `space.builtin.w_dict` is a single W_ModuleDictObject in PyPy
    // (`pypy/interpreter/baseobjspace.py:642`).  Pyre used to keep a
    // parallel `builtins: DictStorage` snapshot here, but that snapshot froze the builtin set at
    // EC construction time — runtime mutations to `__builtins__`
    // weren't visible to new frames.  Reading the live storage from
    // `builtins_module` each call removes the double-storage gap.
    /// Cached dict wrapper over `self.builtins` — pyframe.py
    /// `space.builtin` returns the same object every call.
    builtin_dict_cache: std::cell::Cell<PyObjectRef>,
    /// `pypy/interpreter/baseobjspace.py` `space.check_signal_action` —
    /// the `CheckSignalAction` registered when the `signal` module loads.
    /// `checksignals` calls its `perform` directly.  Stored as a trait
    /// pointer into the process object-space singleton owned by
    /// `module::_signal` (`install_signal_handling`); `None` until installed.
    pub check_signal_action: Option<*mut dyn AsyncActionOps>,
    /// `executioncontext.py sys_exc_operror` — the active exception for
    /// `sys.exc_info()` / bare `raise`, saved/restored across handler
    /// regions by PUSH_EXC_INFO / POP_EXCEPT.  Single source of truth
    /// (replaces the former `eval::CURRENT_EXCEPTION` thread-local), so
    /// the JIT can read/write it as a GETFIELD_GC/SETFIELD_GC slot on the
    /// per-thread EC pointer and the optimizer can dead-store-eliminate a
    /// balanced save/restore.  GC-rooted via `walk_pyframe_roots`.
    pub sys_exc_value: PyObjectRef,
    /// `warmspot.py rewrite_can_enter_jits` leaves compiled-loop exits outside
    /// the portal's jitcode and propagates them out of band.  This is pyre's
    /// execution-context-owned carrier for that pending exit.
    pending_loop_exit: Option<PendingLoopExit>,
    /// Per-execution-context coroutine origin tracking depth. CPython keeps
    /// the corresponding setting in the thread state; pyre keeps all such
    /// execution state on the PyPy-style ExecutionContext.
    pub coroutine_origin_tracking_depth: i64,
    /// `executioncontext.py:55` — linked-list head for running generators and
    /// coroutines.  Each node owns the suspended exception state of its
    /// caller through `GeneratorOrCoroutine.previous_gen_or_coroutine`.
    pub current_gen_or_coroutine: PyObjectRef,
    /// `executioncontext.py:53-54` — PEP 525 hooks, owned by the execution
    /// context (thread-specific), never by a process-global side table.
    pub w_asyncgen_firstiter_fn: PyObjectRef,
    pub w_asyncgen_finalizer_fn: PyObjectRef,
    /// `executioncontext.py:52` — the current PEP 567 Context object for this
    /// execution context, reached from `lib_pypy/_contextvars.py` through
    /// `__pypy__.get_contextvar_context()` and swapped temporarily inside
    /// `Context.run`.  Execution-context-owned, not a process global or an
    /// independent Rust TLS side table.
    pub contextvar_context: PyObjectRef,
    /// `objspace.py` `ec._py_repr` — the identity dict of objects currently
    /// being `repr()`'d (`__pypy__.objects_in_repr()`), lazily created.
    /// Recursive `__repr__` (e.g. `OrderedDict`) consults it to emit `...`
    /// instead of recursing.  Execution-context-owned like `contextvar_context`.
    pub py_repr: PyObjectRef,
    /// Mid-repr object set for `Py_ReprEnter` / the display cycle guard.
    /// Upstream keeps this on the execution context (`objspace.py
    /// get_objects_in_repr` / `cpyext Py_ReprEnter`), not a thread-local.
    pub repr_active: RefCell<Vec<PyObjectRef>>,
    /// Placeholders currently being resolved (`_PyThreadStateImpl.lazy_imports`).
    pub lazy_imports_resolving: RefCell<Vec<PyObjectRef>>,
    /// Frame-close census for `__del__` before the next opcode. Owned by
    /// the execution context that is finishing the generator, not TLS.
    pub pending_close_finalizer: Cell<bool>,
    /// Number of user Python frames currently executing bytecode on this
    /// context.  Bumped once at every `eval_loop` / `eval_loop_jit` entry and
    /// dropped when that activation returns, so the module-level frame, an
    /// `exec`ed body and a resumed generator each cost one unit exactly like a
    /// called function does.  `stack_check()` compares it against
    /// `sys.getrecursionlimit()`.
    ///
    /// Context-owned rather than a Rust TLS side table because a compiled
    /// trace already holds this context as a value -- it reads and writes
    /// `topframeref` at every inlined call -- so the activation seam charges
    /// the unit with the same GETFIELD_GC/SETFIELD_GC pair it already emits
    /// there instead of residualizing a call to reach a thread-local.
    pub py_recursion_depth: usize,
    /// The innermost frame whose activation has already spent its
    /// [`py_recursion_depth`](Self::py_recursion_depth) unit, as a bare
    /// address.  A frame is executed through nested entry points -- the JIT
    /// wrapper may run it as compiled code, hand it to the JIT eval loop, or
    /// decline and re-enter the plain evaluator for the very same frame -- and
    /// only the outermost of those pays.  Any frame reached from here is a
    /// different, simultaneously-live frame, so its address cannot collide
    /// with the one recorded.
    ///
    /// Never dereferenced and never a GC root: `eval.rs`'s `visit_ec_slots`
    /// forwards the context's reference slots one named field at a time, and
    /// this one is an identity token that may name a frame already freed, so
    /// forwarding it would be wrong rather than merely unnecessary.
    pub accounted_activation: usize,
    /// Left by `generator_invoke_execute_frame`'s finally once
    /// the frame body returned and `running` is cleared; read by the jd2 abort
    /// fallback to tell 'the resume never returned' from 'the body already ran'.
    /// Per thread: each mutator resumes its own generators. Recorded only
    /// while a jd2 walk has armed it, so the plain interpreter never holds
    /// the last resumed generator or its result here.
    pub genentry_note_armed: bool,
    pub genentry_note_finished: bool,
    pub genentry_note_is_err: bool,
    pub genentry_note_gen: PyObjectRef,
    pub genentry_note_word: PyObjectRef,
    /// This thread's `SHADOW_STACK` cell, resolved once.
    ///
    /// `shadowstack.py` `gc_enter_roots_frame` resolves the root stack once
    /// per function. The cell address is stable for the thread, so the
    /// dispatch loop loads this word instead of entering `LocalKey::with`
    /// on every opcode. Zero means not yet resolved. Not a GC root: it
    /// names the thread-local stack, not a Python object. A worker EC must
    /// not keep the parent's cell.
    pub root_stack_slot: Cell<usize>,
}

pub type PyExecutionContext = ExecutionContext;

/// `ExecutionContext.get_builtin()` cache read. This concrete accessor keeps
/// the rtyper from having to bind generic `std::cell::Cell<T>::get` while
/// preserving the exact `Cell<PyObjectRef>` storage used by the PyPy-parity
/// builtin-module cache.
#[majit_macros::dont_look_inside]
pub fn execution_context_builtin_cache_get(ec: &ExecutionContext) -> PyObjectRef {
    ec.builtin_dict_cache.get()
}

/// Byte offset of `sys_exc_value` within `ExecutionContext`, for the JIT's
/// GETFIELD_GC/SETFIELD_GC lowering of PUSH_EXC_INFO / POP_EXCEPT.
pub const EC_SYS_EXC_VALUE_OFFSET: usize = std::mem::offset_of!(ExecutionContext, sys_exc_value);

/// Byte offset of `current_gen_or_coroutine` within `ExecutionContext`, for
/// the JIT lowering of `executioncontext.py sys_exc_info`'s empty-slot arm:
/// when `sys_exc_operror` is None the function still has to prove the
/// generator chain is empty before answering None, otherwise it walks
/// `_get_topmost_exception`.
pub const EC_CURRENT_GEN_OR_COROUTINE_OFFSET: usize =
    std::mem::offset_of!(ExecutionContext, current_gen_or_coroutine);

/// Byte offset of `topframeref` within `ExecutionContext`, for the JIT's
/// GETFIELD_GC/SETFIELD_GC lowering of [`ExecutionContext::enter`] /
/// [`ExecutionContext::leave`] at an inlined call.  The traced sequence is
/// `executioncontext.py` — read the slot into the callee's `f_backref`,
/// then store the callee's `jit.virtual_ref` back into it.
pub const EC_TOPFRAMEREF_OFFSET: usize = std::mem::offset_of!(ExecutionContext, topframeref);

/// Byte offset of `w_tracefunc` within `ExecutionContext`, for the JIT's
/// GETFIELD_GC_R lowering of `gettrace()` (`executioncontext.py`, a
/// `jit.promote` of this slot).  A trace recorded while the slot is NULL pins
/// that with a guard, so installing a global trace function leaves compiled
/// code instead of running on past the events it now owes.
pub const EC_W_TRACEFUNC_OFFSET: usize = std::mem::offset_of!(ExecutionContext, w_tracefunc);

/// Byte offset of `profilefunc` within `ExecutionContext`, for the JIT's
/// GETFIELD_GC_I lowering of the `profilefunc?` pin (`executioncontext.py`
/// `_immutable_fields_`).  A compiled loop recorded with the slot empty
/// must GNI-fail when `setllprofile` installs a profiler, the same way
/// `w_tracefunc?` fails after `settrace`.
pub const EC_PROFILEFUNC_OFFSET: usize = std::mem::offset_of!(ExecutionContext, profilefunc);

/// Byte offset of `mutate_w_tracefunc` within `ExecutionContext`. The
/// wrapper is `repr(C)`, so this is the mutate word.
pub const EC_MUTATE_W_TRACEFUNC_OFFSET: usize =
    std::mem::offset_of!(ExecutionContext, mutate_w_tracefunc);

/// Byte offset of `mutate_profilefunc` within `ExecutionContext`.
pub const EC_MUTATE_PROFILEFUNC_OFFSET: usize =
    std::mem::offset_of!(ExecutionContext, mutate_profilefunc);

/// `quasiimmut.py get_current_qmut_instance` for the EC `w_tracefunc?` slot.
///
/// # Safety
/// `ec` must be null or a live `ExecutionContext`.
pub unsafe fn ec_current_w_tracefunc_qmut(
    ec: *const ExecutionContext,
) -> Option<std::sync::Arc<pyre_object::quasiimmut::QuasiImmut>> {
    if ec.is_null() {
        return None;
    }
    Some(unsafe { (*ec).current_w_tracefunc_qmut() })
}

/// `quasiimmut.py get_current_qmut_instance` for the EC `profilefunc?` slot.
///
/// # Safety
/// `ec` must be null or a live `ExecutionContext`.
pub unsafe fn ec_current_profilefunc_qmut(
    ec: *const ExecutionContext,
) -> Option<std::sync::Arc<pyre_object::quasiimmut::QuasiImmut>> {
    if ec.is_null() {
        return None;
    }
    Some(unsafe { (*ec).current_profilefunc_qmut() })
}

/// Byte offset of `py_recursion_depth` within `ExecutionContext`, for the
/// JIT's GETFIELD_GC_I/SETFIELD_GC lowering of the activation seam that
/// `pyframe.py` (`execute_frame.insert_stack_check_here`) puts at the same
/// place in the plain evaluator.
pub const EC_PY_RECURSION_DEPTH_OFFSET: usize =
    std::mem::offset_of!(ExecutionContext, py_recursion_depth);

/// Byte offset of `accounted_activation` within `ExecutionContext`, read and
/// restored by the same seam so a frame that reaches a portal door after the
/// charge does not pay a second unit.
pub const EC_ACCOUNTED_ACTIVATION_OFFSET: usize =
    std::mem::offset_of!(ExecutionContext, accounted_activation);

/// Size of `ExecutionContext`, for the JIT's StructPtrInfo SizeDescr
/// describing the (non-GC) EC struct.  The EC is never JIT-allocated;
/// this size only backs the field-tracking SizeDescr.
pub const EC_SIZE: usize = std::mem::size_of::<ExecutionContext>();

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    #[inline]
    pub fn new() -> Self {
        let builtins_module = crate::builtins::new_builtin_module_dict();
        let builtins_module = pyre_object::gc_roots::pin_root(builtins_module);
        Self {
            space: pyre_object::PY_NULL,
            topframeref: std::ptr::null_mut(),
            w_tracefunc: pyre_object::PY_NULL,
            mutate_w_tracefunc: WTracefuncWatchers::default(),
            is_tracing: 0,
            compiler: pyre_object::PY_NULL,
            profilefunc: None,
            mutate_profilefunc: ProfilefuncWatchers::default(),
            w_profilefuncarg: pyre_object::PY_NULL,
            thread_disappeared: false,
            w_async_exception_type: pyre_object::PY_NULL,
            genentry_note_armed: false,
            genentry_note_finished: false,
            genentry_note_is_err: false,
            genentry_note_gen: pyre_object::PY_NULL,
            genentry_note_word: pyre_object::PY_NULL,
            signals_enabled: 0,
            trace_all_generation: 0,
            profile_all_generation: 0,
            thread_local_refs: Vec::new(),
            actionflag: SpaceActionFlag::new(),
            builtins_module,
            builtin_dict_cache: std::cell::Cell::new(pyre_object::PY_NULL),
            check_signal_action: None,
            sys_exc_value: pyre_object::PY_NULL,
            pending_loop_exit: None,
            coroutine_origin_tracking_depth: 0,
            current_gen_or_coroutine: pyre_object::PY_NULL,
            w_asyncgen_firstiter_fn: pyre_object::PY_NULL,
            w_asyncgen_finalizer_fn: pyre_object::PY_NULL,
            contextvar_context: pyre_object::PY_NULL,
            py_repr: pyre_object::PY_NULL,
            repr_active: RefCell::new(Vec::new()),
            lazy_imports_resolving: RefCell::new(Vec::new()),
            pending_close_finalizer: Cell::new(false),
            py_recursion_depth: 0,
            accounted_activation: 0,
            root_stack_slot: Cell::new(0),
        }
    }

    /// `OSThreadLocals.enter_thread()` / `ExecutionContext(space)` parity.
    ///
    /// Interpreter-wide objects (the object space, action flag, finalizer
    /// action, and builtins module) remain shared, while frame, exception,
    /// tracing, profiling, and generator state starts fresh for each OS thread.
    pub fn clone_for_thread(&self) -> Self {
        let mut ec = self.clone();
        ec.topframeref = std::ptr::null_mut();
        // The activation accounting is per-context and the worker runs none of
        // the spawning thread's frames.  Carrying the depth over would spend
        // part of `sys.getrecursionlimit()` before the thread's first frame;
        // carrying the address over would let a frame minted here read itself
        // as already accounted, the way an address reused by this thread's
        // allocator would, and never pay its unit.
        ec.py_recursion_depth = 0;
        ec.accounted_activation = 0;
        // The cached cell belongs to the spawning thread's TLS.
        ec.root_stack_slot.set(0);
        ec.w_tracefunc = pyre_object::PY_NULL;
        ec.is_tracing = 0;
        ec.store_profilefunc(None, pyre_object::PY_NULL);
        ec.thread_disappeared = false;
        ec.w_async_exception_type = pyre_object::PY_NULL;
        ec.genentry_note_armed = false;
        ec.genentry_note_finished = false;
        ec.genentry_note_is_err = false;
        ec.genentry_note_gen = pyre_object::PY_NULL;
        ec.genentry_note_word = pyre_object::PY_NULL;
        // threadlocals.py — a fresh worker EC starts disabled.  `_set_ec`
        // enables only the process main thread, and reinit_threads promotes
        // the surviving EC after a fork.
        ec.signals_enabled = 0;
        ec.trace_all_generation = 0;
        ec.profile_all_generation = 0;
        ec.thread_local_refs.clear();
        ec.actionflag = SpaceActionFlag::new();
        // The cached Module wrapper is execution-context-owned and movable.
        // A new thread lazily builds its own wrapper over the shared
        // builtins_module, matching a fresh PyPy ExecutionContext.
        ec.builtin_dict_cache = std::cell::Cell::new(pyre_object::PY_NULL);
        ec.sys_exc_value = pyre_object::PY_NULL;
        ec.pending_loop_exit = None;
        // executioncontext.py — a fresh ExecutionContext starts at 0.
        ec.coroutine_origin_tracking_depth = 0;
        ec.current_gen_or_coroutine = pyre_object::PY_NULL;
        ec.w_asyncgen_firstiter_fn = pyre_object::PY_NULL;
        ec.w_asyncgen_finalizer_fn = pyre_object::PY_NULL;
        ec.contextvar_context = pyre_object::PY_NULL;
        ec.py_repr = pyre_object::PY_NULL;
        ec.repr_active = RefCell::new(Vec::new());
        ec.lazy_imports_resolving = RefCell::new(Vec::new());
        ec.pending_close_finalizer = Cell::new(false);
        ec
    }

    /// Expose the two builtin-owner slots to the translated shadow-stack root
    /// walker.  PyPy reaches both through the process-owned object space;
    /// pyre stores movable GC refs directly on the EC and must forward those
    /// fields in place.
    pub(crate) fn walk_builtin_roots(&mut self, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        visitor(unsafe {
            &mut *(&mut self.builtins_module as *mut PyObjectRef as *mut majit_ir::GcRef)
        });
        let cache = self.builtin_dict_cache.as_ptr();
        visitor(unsafe { &mut *(cache as *mut majit_ir::GcRef) });
        for wref in &mut self.thread_local_refs {
            visitor(unsafe { &mut *(wref as *mut PyObjectRef as *mut majit_ir::GcRef) });
        }
        visitor(unsafe {
            &mut *(&mut self.contextvar_context as *mut PyObjectRef as *mut majit_ir::GcRef)
        });
        visitor(unsafe {
            &mut *(&mut self.genentry_note_gen as *mut PyObjectRef as *mut majit_ir::GcRef)
        });
        visitor(unsafe {
            &mut *(&mut self.genentry_note_word as *mut PyObjectRef as *mut majit_ir::GcRef)
        });
        visitor(unsafe { &mut *(&mut self.py_repr as *mut PyObjectRef as *mut majit_ir::GcRef) });
        for entry in self.repr_active.get_mut().iter_mut() {
            visitor(unsafe { &mut *(entry as *mut PyObjectRef as *mut majit_ir::GcRef) });
        }
        for entry in self.lazy_imports_resolving.get_mut().iter_mut() {
            visitor(unsafe { &mut *(entry as *mut PyObjectRef as *mut majit_ir::GcRef) });
        }
        if let Some(pending) = self.pending_loop_exit.as_mut() {
            match pending {
                PendingLoopExit::Done(Ok(value)) => {
                    visitor(unsafe { &mut *(value as *mut PyObjectRef as *mut majit_ir::GcRef) })
                }
                PendingLoopExit::Done(Err(err)) | PendingLoopExit::ExitFrameWithException(err) => {
                    err.walk_gc_refs(visitor)
                }
                PendingLoopExit::ContinueRunningNormally => {}
            }
        }
    }

    pub fn set_pending_loop_exit(&mut self, exit: PendingLoopExit) {
        self.pending_loop_exit = Some(exit);
    }

    pub fn take_pending_loop_exit(&mut self) -> Option<PendingLoopExit> {
        self.pending_loop_exit.take()
    }

    pub fn __init__(&mut self, space: PyObjectRef) {
        self.space = space;
        self.compiler = pyre_object::w_none();
    }

    pub fn install_user_del_action(&mut self) {
        let actionflag = self.actionflag.shared_mut();
        install_space_user_del_action(self.space, actionflag);
        #[cfg(all(
            feature = "cpyext",
            not(feature = "sandbox"),
            any(target_os = "macos", target_os = "linux")
        ))]
        {
            let space = self.space;
            SPACE_CPYEXT_DEALLOC_ACTION
                .get_or_init(|| Box::into_raw(PyObjDeallocAction::new(space, actionflag)) as usize);
        }
        if let Some(hooks) = crate::importing::optional_module_hooks() {
            (hooks.gc_initialize)(self.space, actionflag);
        }
        pyre_object::gc_hook::register_maybe_finalizer_hook(maybe_register_user_finalizer);
    }

    #[inline]
    pub fn _mark_thread_disappeared(_space: PyObjectRef) {
        let _ = _space;
    }

    #[inline]
    pub fn gettopframe(&self) -> *mut PyFrame {
        // `executioncontext.py:_gettopframe` → `self.topframeref()` (vref
        // force), then the virtualizable force materializes the fastlocals.
        let frame = force_vref(self.topframeref);
        // The force reaches the JIT's virtualizable writeback through a backend
        // hook whose callee this crate cannot follow, so it is judged as able to
        // collect and the frame is handed back out of the anchor.
        let anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame) };
        force_frame(frame);
        anchor.live()
    }

    /// `self.topframeref()` without the virtualizable force that
    /// [`Self::gettopframe`] adds — "raw" is about `force_frame`, not about the
    /// vref.  The vref force is not optional: `executioncontext.py/72/446/451`
    /// all read the slot *with* the parens, and the only unforced reads upstream
    /// are `:88`/`:96`, which move the vref along rather than dereference it.
    /// A `JitVirtualRef` handed out here would be dereferenced at `PyFrame`
    /// field offsets by every caller.
    #[inline]
    pub fn gettopframe_raw(&self) -> *mut PyFrame {
        force_vref(self.topframeref)
    }

    /// `executioncontext.py gettopframe_nohidden` — follow `f_backref` past every
    /// `hidden_applevel` frame.  No force: the walk reads only the vref links
    /// and `hide()`, and `hide()` goes through `pyframe_get_pycode`, a plain
    /// field read.  Consumers that hand the frame to app code call
    /// [`force_frame`] themselves.
    #[majit_macros::unroll_safe]
    pub fn gettopframe_nohidden(&self) -> *mut PyFrame {
        let mut frame = force_vref(self.topframeref);
        while !frame.is_null() {
            // SAFETY: frame pointers are owned by interpreter call stack and can be
            // null-checked before dereference.
            unsafe {
                let current = &*frame;
                if !current.hide() {
                    return frame;
                }
                frame = current.get_f_back();
            }
        }
        frame
    }

    /// `executioncontext.py getnextframe_nohidden` — the same walk from an
    /// arbitrary frame.
    /// Force-free for the reason given on [`gettopframe_nohidden`].
    #[majit_macros::unroll_safe]
    pub fn getnextframe_nohidden(mut frame: *mut PyFrame) -> *mut PyFrame {
        while !frame.is_null() {
            // SAFETY: caller provides a valid frame chain or null.
            unsafe {
                let current = &*frame;
                let next = current.get_f_back();
                if next.is_null() {
                    return std::ptr::null_mut();
                }
                if !(&*next).hide() {
                    return next;
                }
                frame = next;
            }
        }
        frame
    }

    pub fn enter(&mut self, frame: *mut PyFrame) {
        // pypy/interpreter/executioncontext.py enter parity.
        //
        // `if self.space.reverse_debugging: self._revdb_enter(frame)` folds
        // away: the space carries `reverse_debugging = False`
        // (baseobjspace.py) unless `config.translation.reverse_debugger`
        // sets it (baseobjspace.py:444), and reverse debugging is not ported.
        // Same fold `bytecode_only_trace` and `side_effects_ok` already carry.
        //
        // `f_backref` is a traced `Type::Ref` field, so the store carries the
        // generational barrier the JIT's `SetfieldGc` lowering emits for it.
        // `frame` can be a non-moving old-generation frame while `topframeref`
        // names a young one, and without the remembered-set entry the next
        // minor collection leaves `f_backref` pointing at the pre-copy nursery
        // address.  Same obligation as the inline-call push and the resumed
        // chain's relink.
        // A caller that already linked the frame -- `install_current_frame`
        // sets both ends of this pair -- would make the store below name the
        // frame itself, and `walk_pyframe_roots` follows `f_backref` with no
        // cycle guard. `topframeref` may already be a `JitVirtualRef` whose
        // address is not the frame, so compare the referent. A `debug_assert`
        // is absent from the release binary that runs the witnesses.
        //
        // `executioncontext.py` `enter` has no such refusal because upstream
        // never reaches it twice for one frame: `interp_jit.py` puts
        // `jit_merge_point` on `PyFrame.dispatch`, so the force and deopt
        // legs of CALL_ASSEMBLER (`assembler_call_helper`) re-run only
        // `dispatch`, below `enter`. pyre's portal is one level up, at
        // `execute_frame`, so those legs re-enter through the portal helper
        // and reach `enter` a second time for a frame that is already top.
        // `install_current_frame` refuses the same case. The structural fix
        // is moving the portal down to `dispatch`; until then this refusal
        // keeps the re-entry from making `f_backref` a self loop.
        if std::ptr::eq(vref_referent(self.topframeref), frame) {
            return;
        }
        pyre_object::gc_hook::try_gc_write_barrier(frame as pyre_object::gc_hook::GCREF);
        majit_gc::bh_probe_note_store(frame as usize, crate::pyframe::PYFRAME_F_BACKREF_OFFSET, 1);
        unsafe {
            (*frame).f_backref = self.topframeref;
        }
        // `self.topframeref = jit.virtual_ref(frame)`.  At interp level
        // `virtual_ref(frame)` is the frame pointer itself
        // (`_jit_vref.py lowleveltype = OBJECTPTR`, no allocation); the
        // JIT rewrites it to a `JitVirtualRef` when it virtualizes the frame.
        self.topframeref = frame;
    }

    #[allow(clippy::too_many_arguments)]
    /// Hands back the exit value at its live address: the profile callback is
    /// application-level Python, so a movable value moves under it.
    /// `executioncontext.py leave`'s profile arm — `_trace(frame,
    /// 'leaveframe', w_exitvalue)`, which is where `setprofile`'s `return`
    /// event comes from.  `return_trace` carries only the `gettrace()` one.
    ///
    /// Split out of [`Self::leave`] because the JIT portal owes this event
    /// without the `topframeref` and vref surgery `leave` wraps it in: the
    /// arms that run a frame compiled never reach `leave` at all
    /// (`eval::eval_with_jit_inner`).
    ///
    /// Returns the exit value at its live address — the callback is
    /// application-level Python and can move it.
    pub fn leaveframe_trace(
        &mut self,
        frame: *mut PyFrame,
        w_exitvalue: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        if self.profilefunc.is_none() {
            return Ok(w_exitvalue);
        }
        // The exit value rides a root slot for the callback's duration.
        // Nothing runs Python without a profiler installed, so the slot is
        // taken on this arm rather than for every frame that leaves.
        let _roots = pyre_object::gc_roots::push_roots();
        let exit_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_exitvalue);
        let result = self._trace(
            frame,
            "leaveframe",
            pyre_object::gc_roots::shadow_stack_get(exit_slot),
            None,
        );
        result.map(|()| pyre_object::gc_roots::shadow_stack_get(exit_slot))
    }

    pub fn leave(
        &mut self,
        frame: *mut PyFrame,
        w_exitvalue: PyObjectRef,
        got_exception: bool,
    ) -> Result<PyObjectRef, crate::PyError> {
        // pypy/interpreter/executioncontext.py leave parity.
        // The original wraps `_trace('leaveframe', …)` in try/finally so
        // the topframeref restore and the vref dance always run.  We
        // capture the trace result and propagate it after the cleanup
        // block below.
        // `_trace` runs the profile callback, which is application-level
        // Python; the chain surgery below reads the frame's own fields.
        let anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame) };
        let mut trace_result = self.leaveframe_trace(frame, w_exitvalue);
        let frame = anchor.live();
        let frame_vref = self.topframeref;
        // At interp level a vref in the chain *is* the frame pointer, so the
        // slot `force_vref` is about to be handed goes stale across the force
        // `get_f_back` performs just above it.
        let vref_anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame_vref) };
        unsafe {
            // `self.topframeref = frame.f_backref` (no parens — move the raw
            // caller vref back, do not force it).
            self.topframeref = (*frame).f_backref;
            if (*frame).escaped() || got_exception {
                // Both forces below can materialize frames, so they can
                // collect, and the traced exit value is a local of this
                // `finally` across them.
                let pinned = PinnedResult::pin(&mut trace_result);
                // `f_back = frame.f_backref()` — forced (get_f_back forces).
                let f_back = (*frame).get_f_back();
                if !f_back.is_null() {
                    (*f_back).mark_as_escaped();
                }
                // `frame_vref()` — force the leaving frame's own vref so it
                // stays reachable after the JIT frame is gone (identity at
                // interp level).
                force_vref(vref_anchor.live());
                pinned.reload(&mut trace_result);
            }
        }
        // `jit.virtual_ref_finish(frame_vref, frame)` — keepalives only at
        // interp level (both operands are already live in Rust); the JIT
        // records VIRTUAL_REF_FINISH during tracing.
        // `if self.space.reverse_debugging: self._revdb_leave(got_exception)`
        // (executioncontext.py) folds away with the arm in `enter`.
        trace_result
    }

    /// executioncontext.py — `c_call_trace(self, frame, w_func, args=None)`.
    pub fn c_call_trace(
        &mut self,
        frame: *mut PyFrame,
        w_func: PyObjectRef,
        args: Option<&crate::argument::Arguments>,
    ) -> Result<(), crate::PyError> {
        self._c_call_return_trace(frame, w_func, args, "c_call")
    }

    /// executioncontext.py — `c_return_trace(self, frame, w_func, args=None)`.
    pub fn c_return_trace(
        &mut self,
        frame: *mut PyFrame,
        w_func: PyObjectRef,
        args: Option<&crate::argument::Arguments>,
    ) -> Result<(), crate::PyError> {
        self._c_call_return_trace(frame, w_func, args, "c_return")
    }

    /// executioncontext.py — `_c_call_return_trace`.
    pub fn _c_call_return_trace(
        &mut self,
        frame: *mut PyFrame,
        mut w_func: PyObjectRef,
        args: Option<&crate::argument::Arguments>,
        event: &str,
    ) -> Result<(), crate::PyError> {
        if self.profilefunc.is_none() {
            if !frame.is_null() {
                unsafe {
                    (*frame).getorcreatedebug(-1).is_being_profiled = false;
                }
            }
            return Ok(());
        }
        // executioncontext.py:128-134 FunctionWithFixedCode method-call
        // rebinding.  PyPy:
        //   if isinstance(w_func, FunctionWithFixedCode) and args is not None:
        //       w_firstarg = args.firstarg()
        //       if w_firstarg is not None:
        //           w_func = descr_function_get(self.space, w_func,
        //                                       w_firstarg, self.space.type(w_firstarg))
        // BuiltinFunction (function.py, the module-level
        // builtin sibling) is intentionally excluded — function.py
        // splits the two so that "builtin function binds
        // differently" (no descriptor rebinding).  Pyre's
        // `is_function_with_fixed_code` (FUNCTION_TYPE && !can_change_code)
        // is the line-by-line port of the isinstance check.
        if let Some(args) = args {
            unsafe {
                if crate::is_function_with_fixed_code(w_func)
                    && let Some(w_firstarg) = args.firstarg()
                    && !w_firstarg.is_null()
                {
                    let w_type = crate::typedef::r#type(w_firstarg)
                        .map_or(pyre_object::PY_NULL, |p| p.as_ptr());
                    w_func = crate::descr_function_get(w_func, w_firstarg, w_type);
                }
            }
        }
        self._trace(frame, event, w_func, None)
    }

    pub fn c_exception_trace(
        &mut self,
        frame: *mut PyFrame,
        _w_exc: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        if self.profilefunc.is_none() {
            if !frame.is_null() {
                unsafe {
                    (*frame).getorcreatedebug(-1).is_being_profiled = false;
                }
            }
            return Ok(());
        }
        self._trace(frame, "c_exception", _w_exc, None)
    }

    pub fn call_trace(&mut self, frame: *mut PyFrame) -> Result<(), crate::PyError> {
        if !self.gettrace().is_null() || self.profilefunc.is_some() {
            // The callback is Python, and the profiling flag below is a write
            // into the frame's own debug block.
            let anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame) };
            self._trace(frame, "call", pyre_object::w_none(), None)?;
            let frame = anchor.live();
            if self.profilefunc.is_some() && !frame.is_null() {
                unsafe {
                    (*frame).getorcreatedebug(-1).is_being_profiled = true;
                }
            }
        }
        Ok(())
    }

    /// Hands back the returned value at its live address: the callback is
    /// application-level Python, so a movable value moves under it.
    pub fn return_trace(
        &mut self,
        frame: *mut PyFrame,
        w_retval: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        if self.gettrace().is_null() {
            return Ok(w_retval);
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let retval_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_retval);
        // The callback is handed the value off its slot too: publication ends
        // in a forwarding query, which is a safepoint, and it rewrites the
        // slot rather than this frame's copy.
        self._trace(
            frame,
            "return",
            pyre_object::gc_roots::shadow_stack_get(retval_slot),
            None,
        )?;
        Ok(pyre_object::gc_roots::shadow_stack_get(retval_slot))
    }

    #[inline(always)]
    pub fn bytecode_trace(
        &mut self,
        mut frame: *mut PyFrame,
        decr_by: usize,
    ) -> Result<(), crate::PyError> {
        // executioncontext.py bytecode_trace:
        //   def bytecode_trace(self, frame, decr_by=TICK_COUNTER_STEP):
        //       self.bytecode_only_trace(frame)
        //       actionflag = self.space.actionflag
        //       if actionflag.decrement_ticker(decr_by) < 0:
        //           actionflag.action_dispatcher(self, frame)
        //
        // bytecode_only_trace runs first; if it raises (tracefunc
        // callback exception), the ticker decrement + slow-path
        // action_dispatcher do NOT run.  Use `?` so a tracer error
        // short-circuits before touching actionflag.
        frame = self.bytecode_only_trace(frame)?;
        if space_decrement_ticker(self, decr_by as isize) < 0 {
            // executioncontext.py — `actionflag.action_dispatcher`.
            // Routed through a residual (dont_look_inside) boundary so the
            // tracer never sees the action machinery's trait-object
            // virtual calls + `Result<(), PyError>` propagation, which the
            // JIT codewriter cannot model.  The slow path runs only when
            // the ticker is negative (a signal / fired action), so the
            // residual call adds nothing to the no-signal hot path.
            let self_ptr = self as *mut ExecutionContext;
            if perform_pending_actions(self_ptr, frame) != 0
                && let Some(err) = crate::call::take_call_error()
            {
                return Err(err);
            }
        }
        Ok(())
    }

    /// Run pending async actions (signal delivery, finalizers).  Split
    /// out so the JIT warm-up loop can take the slow path inline once the
    /// ticker goes negative without re-entering `bytecode_trace`'s tracer
    /// gate.  Mirrors the `actionflag.action_dispatcher(self, frame)` call
    /// in `bytecode_trace` (executioncontext.py).
    pub fn perform_actions(&mut self, frame: *mut PyFrame) -> Result<(), crate::PyError> {
        let self_ptr = self as *mut ExecutionContext;
        self.actionflag.action_dispatcher(self_ptr, frame)
    }

    /// pypy/interpreter/executioncontext.py `bytecode_only_trace`.
    ///
    /// ```python
    /// def bytecode_only_trace(self, frame):
    ///     if self.space.reverse_debugging:
    ///         self._revdb_potential_stop_point(frame)
    ///     if (frame.get_w_f_trace() is None or self.is_tracing or
    ///         self.gettrace() is None):
    ///         return
    ///     self.run_trace_func(frame)
    /// ```
    ///
    /// reverse_debugging is not implemented in pyre, so the
    /// `_revdb_potential_stop_point` arm reduces to a no-op.  All
    /// three short-circuit conditions stay — including
    /// `frame.get_w_f_trace() is None`, which avoids the trailing
    /// `instr_prev_plus_one` write inside `run_trace_func` when the
    /// per-frame trace is unset (so a later `frame.f_trace = cb`
    /// observes the unmodified instr_prev_plus_one and can fire its
    /// first `line` event correctly).
    #[inline(always)]
    /// Answers the frame, because the tracer arm below runs Python and a frame
    /// a compiled trace built moves there.  Returning it puts the reload on the
    /// arm that can collect instead of on every caller: this is the per-opcode
    /// path, and an anchor taken before the early-outs would be a shadow-stack
    /// push and pop for every bytecode with no tracer installed.
    pub fn bytecode_only_trace(
        &mut self,
        frame: *mut PyFrame,
    ) -> Result<*mut PyFrame, crate::PyError> {
        // PyPy has no object-space-null guard here: `space` is the interpreter
        // owner, not an optional tracing flag. Pyre represents that owner with
        // PY_NULL in this runtime, so testing it suppressed every line event
        // while still allowing call/return events through `_trace`.
        if frame.is_null() {
            return Ok(frame);
        }
        let f_trace_is_none = unsafe { (*frame).get_w_f_trace().is_null() };
        if f_trace_is_none || self.is_tracing != 0 || self.w_tracefunc.is_null() {
            return Ok(frame);
        }
        let anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame) };
        self.run_trace_func(frame)?;
        Ok(anchor.live())
    }

    pub fn _run_finalizers_now(&mut self) {
        let action = space_user_del_action();
        if !action.is_null() {
            unsafe { (*action)._run_finalizers() };
        }
    }

    /// Consume a failed-attribute finalization request after the live red
    /// frame has cleared its own dispatch flag.  This boundary has no live
    /// opcode temporaries outside the published PyFrame roots, so a moving
    /// collection is admissible.  Nursery-born `hasuserdel` instances are
    /// invisible to `try_gc_collect_oldgen`; `incminimark.collect(2)` is
    /// the same full pass `gc.collect` uses to discover them.
    pub fn run_failed_attr_finalizers(&self) {
        let action = space_user_del_action();
        if !action.is_null() {
            pyre_object::gc_hook::try_gc_collect(2);
            unsafe { (*action)._run_finalizers() };
        }
    }

    /// pypy/interpreter/executioncontext.py `run_trace_func`.
    ///
    /// ```python
    /// @jit.unroll_safe
    /// def run_trace_func(self, frame):
    ///     code = frame.getcode() # promote the frame!
    ///     d = frame.getorcreatedebug()
    ///     lastline = d.f_lineno
    ///     lineno = frame.pycode._get_lineno_for_pc_tracing(frame.last_instr)
    ///     if lineno != -1:
    ///         d.f_lineno = lineno
    ///     if d.f_trace_lines and lineno != -1:
    ///         # when we are at a start of a line, or executing a backwards jump,
    ///         # produce a line event
    ///         if lastline != lineno or frame.last_instr < d.instr_prev_plus_one:
    ///             self._trace(frame, 'line', self.space.w_None)
    ///     if d.f_trace_opcodes:
    ///         self._trace(frame, 'opcode', self.space.w_None)
    ///     d.instr_prev_plus_one = frame.last_instr + 1
    /// ```
    /// `executioncontext.py run_trace_func` is `@jit.unroll_safe`.
    /// This body has no loop, so `look_inside_graph` already admits it;
    /// the hint matches the upstream decorator.
    #[majit_macros::unroll_safe]
    pub fn run_trace_func(&mut self, frame: *mut PyFrame) -> Result<(), crate::PyError> {
        if frame.is_null() {
            return Ok(());
        }
        // Reached only with a tracer installed, so the anchor is already on the
        // slow path.  `getorcreatedebug` allocates a nursery `FrameDebugData`
        // and the events below run Python, so take it before either.
        let anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame) };
        let last_instr = unsafe { (*anchor.live()).last_instr };
        // `frame.pycode._get_lineno_for_pc_tracing(frame.last_instr)`, not
        // `get_last_lineno`: the two differ on exactly the instructions that
        // carry a line but may not start one, and `-1` is what both guards
        // below are written against.
        let lineno = unsafe { (*anchor.live()).get_lineno_for_pc_tracing() };
        let (want_line, want_opcode) = unsafe {
            // `getorcreatedebug()` with no argument: reached only through
            // `bytecode_only_trace`, which returns early unless
            // `get_w_f_trace()` is non-null, so the debug block already exists
            // and the seed is dead.  Passing `lineno` here would seed
            // `f_lineno` with the line whose novelty decides the event.
            let d = (*anchor.live()).getorcreatedebug(-1);
            let lastline = d.f_lineno;
            // Persistent, so the next invocation compares against this line —
            // but only for an instruction that names one, or a `RESUME` would
            // overwrite the resumed line with the `def` line it carries.
            if lineno != -1 {
                d.f_lineno = lineno;
            }
            let want_line = d.f_trace_lines
                && lineno != -1
                && (lastline != lineno || last_instr < d.instr_prev_plus_one);
            let want_opcode = d.f_trace_opcodes;
            (want_line, want_opcode)
        };
        if want_line {
            self._trace(anchor.live(), "line", pyre_object::w_none(), None)?;
        }
        if want_opcode {
            self._trace(anchor.live(), "opcode", pyre_object::w_none(), None)?;
        }
        let frame = anchor.live();
        // `d.instr_prev_plus_one = frame.last_instr + 1` — the next-PC sentinel
        // that makes a backward jump re-fire the line event even when it stays
        // on the same source line.
        unsafe {
            (*frame).getorcreatedebug(-1).instr_prev_plus_one = last_instr + 1;
        }
        Ok(())
    }

    /// pypy/interpreter/executioncontext.py:202-208
    /// `bytecode_trace_after_exception`.
    ///
    /// ```python
    /// def bytecode_trace_after_exception(self, frame):
    ///     "Like bytecode_trace(), but without increasing the ticker."
    ///     actionflag = self.space.actionflag
    ///     self.bytecode_only_trace(frame)
    ///     if actionflag.get_ticker() < 0:
    ///         actionflag.action_dispatcher(self, frame)     # slow path
    /// ```
    ///
    /// pyre's action_dispatcher slow path is not yet wired (the
    /// pre-existing stub only decremented the ticker once); the
    /// `bytecode_only_trace` call mirrors upstream so a tracer error
    /// surfaces through this helper's `Result`.
    pub fn bytecode_trace_after_exception(
        &mut self,
        frame: *mut PyFrame,
    ) -> Result<(), crate::PyError> {
        let frame = self.bytecode_only_trace(frame)?;
        if self.actionflag.get_ticker() < 0 {
            // executioncontext.py — `if actionflag.get_ticker()
            // < 0: actionflag.action_dispatcher(self, frame)`.  Routed
            // through the same residual boundary as `bytecode_trace` so a
            // signal delivered during exception handling propagates.
            let self_ptr = self as *mut ExecutionContext;
            if perform_pending_actions(self_ptr, frame) != 0
                && let Some(err) = crate::call::take_call_error()
            {
                return Err(err);
            }
        }
        Ok(())
    }

    /// pypy/interpreter/executioncontext.py exception_trace.
    ///
    /// ```python
    /// def exception_trace(self, frame, operationerr):
    ///     if self.gettrace() is not None:
    ///         self._trace(frame, 'exception', None, operationerr)
    /// ```
    ///
    /// `_trace` normalizes that same live carrier in place, so a tracing
    /// callback and the subsequent handler search observe one exception value
    /// rather than a reconstructed `(type, value, traceback)` snapshot.
    pub fn exception_trace(
        &mut self,
        frame: *mut PyFrame,
        operr: &mut crate::error::OperationError,
    ) -> Result<(), crate::PyError> {
        if !self.gettrace().is_null() {
            self._trace(frame, "exception", pyre_object::PY_NULL, Some(operr))?;
        }
        Ok(())
    }

    /// `executioncontext.py sys_exc_info` — the topmost handled
    /// exception, or PY_NULL when nothing is being handled.
    pub fn sys_exc_info(&self) -> PyObjectRef {
        if !self.sys_exc_value.is_null() {
            return self.sys_exc_value;
        }
        if !self.current_gen_or_coroutine.is_null() {
            return self._get_topmost_exception();
        }
        pyre_object::PY_NULL
    }

    /// `executioncontext.py _get_topmost_exception` — the first
    /// exception parked on the suspended generator/coroutine chain.
    pub fn _get_topmost_exception(&self) -> PyObjectRef {
        let mut generator = self.current_gen_or_coroutine;
        while !generator.is_null() {
            let saved =
                unsafe { pyre_object::generator::w_generator_get_saved_exc_value(generator) };
            if !saved.is_null() {
                return saved;
            }
            generator = unsafe { pyre_object::generator::w_generator_get_previous(generator) };
        }
        pyre_object::PY_NULL
    }

    pub fn current_exception(&self) -> PyObjectRef {
        self.sys_exc_value
    }

    pub fn set_sys_exc_info(&mut self, operror: PyObjectRef) {
        self.sys_exc_value = operror;
    }

    pub fn clear_sys_exc_info(&mut self) {
        self.sys_exc_value = pyre_object::PY_NULL;
    }

    /// `executioncontext.py push_gen_or_coroutine` — exchange the caller's active handler
    /// exception with the exception suspended on `gen`, then link `gen` as
    /// the current generator/coroutine.
    pub fn push_gen_or_coroutine(&mut self, generator: PyObjectRef) {
        let current_exc = self.sys_exc_value;
        let saved_exc =
            unsafe { pyre_object::generator::w_generator_get_saved_exc_value(generator) };
        self.sys_exc_value = saved_exc;
        unsafe {
            pyre_object::generator::w_generator_set_saved_exc_value(generator, current_exc);
            pyre_object::generator::w_generator_set_previous(
                generator,
                self.current_gen_or_coroutine,
            );
        }
        self.current_gen_or_coroutine = generator;
    }

    /// `executioncontext.py pop_gen_or_coroutine` — unlink `gen`, park the exception state
    /// produced by its frame on the generator, and restore its caller's state.
    pub fn pop_gen_or_coroutine(&mut self, generator: PyObjectRef) {
        debug_assert_eq!(self.current_gen_or_coroutine, generator);
        let caller_exc =
            unsafe { pyre_object::generator::w_generator_get_saved_exc_value(generator) };
        self.current_gen_or_coroutine =
            unsafe { pyre_object::generator::w_generator_get_previous(generator) };
        unsafe {
            pyre_object::generator::w_generator_set_previous(generator, pyre_object::PY_NULL);
            pyre_object::generator::w_generator_set_saved_exc_value(generator, self.sys_exc_value);
        }
        self.sys_exc_value = caller_exc;
    }

    #[majit_macros::dont_look_inside]
    pub fn settrace(&mut self, w_func: PyObjectRef) {
        if w_func.is_null() || w_func == pyre_object::w_none() {
            if self.mutate_w_tracefunc.field().is_installed() {
                self.mutate_w_tracefunc.field().invalidate();
            }
            self.w_tracefunc = pyre_object::PY_NULL;
            return;
        }
        // executioncontext.py settrace: force, then the `w_tracefunc?`
        // store, then raise the JIT trace_limit. `force_all_frames` can
        // collect while materializing a virtualizable; pin `w_func`
        // across that.
        let _roots = pyre_object::gc_roots::push_roots();
        let func_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_func);
        self.force_all_frames(false);
        if self.mutate_w_tracefunc.field().is_installed() {
            self.mutate_w_tracefunc.field().invalidate();
        }
        self.w_tracefunc = pyre_object::gc_roots::shadow_stack_get(func_slot);
        crate::call::set_jit_param("trace_limit", 10000);
    }

    pub fn gettrace(&self) -> PyObjectRef {
        // executioncontext.py gettrace: `return jit.promote(self.w_tracefunc)`.
        majit_metainterp::jit::promote(self.w_tracefunc)
    }

    /// `quasiimmut.py get_current_qmut_instance` for `w_tracefunc?`.
    pub fn current_w_tracefunc_qmut(&self) -> std::sync::Arc<pyre_object::quasiimmut::QuasiImmut> {
        self.mutate_w_tracefunc.field().get_current_qmut_instance()
    }

    /// `quasiimmut.py get_current_qmut_instance` for `profilefunc?`.
    pub fn current_profilefunc_qmut(&self) -> std::sync::Arc<pyre_object::quasiimmut::QuasiImmut> {
        self.mutate_profilefunc.field().get_current_qmut_instance()
    }

    fn store_profilefunc(&mut self, func: Option<ProfileFunc>, w_arg: PyObjectRef) {
        if self.mutate_profilefunc.field().is_installed() {
            self.mutate_profilefunc.field().invalidate();
        }
        self.profilefunc = func;
        self.w_profilefuncarg = w_arg;
    }

    /// `executioncontext.py setprofile`.
    pub fn setprofile(&mut self, w_func: PyObjectRef) -> Result<(), crate::PyError> {
        if w_func.is_null() || w_func == pyre_object::w_none() {
            self.store_profilefunc(None, pyre_object::PY_NULL);
            Ok(())
        } else {
            self.setllprofile(Some(app_profile_call), w_func)
        }
    }

    /// `executioncontext.py getprofile`.
    pub fn getprofile(&self) -> PyObjectRef {
        self.w_profilefuncarg
    }

    /// `executioncontext.py setllprofile`.
    pub fn setllprofile(
        &mut self,
        func: Option<ProfileFunc>,
        w_arg: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        let w_arg = if func.is_some() {
            // executioncontext.py setllprofile: `if w_arg is None: raise
            // ValueError("Cannot call setllprofile with real None")`.
            // The check is against RPython-level None (== null in pyre);
            // Python-level `w_none()` (`space.w_None`) is a valid user
            // argument that flows through unchanged.
            if w_arg.is_null() {
                return Err(crate::PyError::value_error(
                    "Cannot call setllprofile with real None",
                ));
            }
            let _roots = pyre_object::gc_roots::push_roots();
            let arg_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_arg);
            self.force_all_frames(true);
            pyre_object::gc_roots::shadow_stack_get(arg_slot)
        } else {
            w_arg
        };
        self.store_profilefunc(func, w_arg);
        Ok(())
    }

    /// `executioncontext.py force_all_frames` — "force" every frame in the
    /// sense of the JIT, so one that is running in assembler fails its next
    /// `GUARD_NOT_FORCED` and falls back to interpreted execution, where the
    /// freshly installed trace / profile callback is honoured.
    ///
    /// Upstream gets that effect from the walk itself: `f_backref` holds a
    /// `jit.virtual_ref`, so `getnextframe_nohidden`'s `frame.f_backref()` is a
    /// `jit_force_virtual`, and `virtualref.force_virtual` runs
    /// `ResumeGuardForcedDescr.force_now` on a token that still names a live
    /// JIT frame.  Pyre's walk calls [`force_vref`] at the same points, and for
    /// an inlined callee that is now the whole story: `walker_ec_enter`
    /// (`jitcode_dispatch/inline_call.rs`) records the `Self::enter` body
    /// verbatim, so `opimpl_virtual_ref` mints a real `JitVirtualRef` and the
    /// `topframeref` store publishes it into the chain this walk reads.
    ///
    /// What the walk still cannot reach is the frame the portal itself runs.
    /// Tracing starts at that frame's own merge point, so its [`Self::enter`]
    /// was executed by the interpreter rather than recorded, and `topframeref`
    /// names the frame directly instead of a vref — the identity force again,
    /// on precisely the frame "running in assembler" that this consumer exists
    /// for.  Hence the direct call.
    pub fn force_all_frames(&mut self, is_being_profiled: bool) {
        let mut frame = self.gettopframe_nohidden();
        while !frame.is_null() {
            let anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame) };
            force_frame(frame);
            frame = anchor.live();
            if is_being_profiled {
                unsafe {
                    (*frame).getorcreatedebug(-1).is_being_profiled = true;
                }
            }
            frame = Self::getnextframe_nohidden(frame);
        }
    }

    pub fn call_tracing(&mut self, w_func: PyObjectRef, w_args: PyObjectRef) -> PyObjectRef {
        let was_tracing = self.is_tracing;
        self.is_tracing = 0;
        let result = crate::baseobjspace::call(w_func, w_args, None);
        self.is_tracing = was_tracing;
        result
    }

    /// pypy/interpreter/executioncontext.py _trace.
    ///
    /// Two-arm dispatch: the tracing arm fires every event for the
    /// frame's `w_f_trace` (or the global `w_tracefunc` on `'call'`),
    /// the profiling arm fires only on `call`/`return`/`c_call`/
    /// `c_return`/`c_exception` and routes through the
    /// `profilefunc` low-level trampoline (`app_profile_call` for
    /// `setprofile`-installed callbacks).
    ///
    /// `operr` is the live interpreter-level exception carrier — the argument
    /// `executioncontext.py _trace` reads through `operr.w_type`,
    /// `operr.normalize_exception(space)` and `operr.get_w_traceback(space)`.
    /// The normalization below memoizes the application exception on that same
    /// value; the class and traceback are then read from the normalized
    /// `W_BaseException`, where pyre stores their authoritative state.
    pub fn _trace(
        &mut self,
        frame: *mut PyFrame,
        event: &str,
        mut w_arg: PyObjectRef,
        mut operr: Option<&mut crate::error::OperationError>,
    ) -> Result<(), crate::PyError> {
        // executioncontext.py:347 if self.is_tracing or frame.hide():
        if self.is_tracing != 0 {
            return Ok(());
        }
        if frame.is_null() {
            return Ok(());
        }
        if unsafe { (*frame).hide() } {
            return Ok(());
        }

        // Everything below this point can collect: `normalize_exception`
        // materialises the exception, `fast2locals` builds the locals mapping,
        // `wrap_trace_frame` allocates the frame object handed out, and the
        // callback is application-level Python.  A frame a compiled trace built
        // is a nursery allocation, so `frame` names the abandoned copy after any
        // of them and the `getorcreatedebug` / `locals2fast` writes below would
        // land there.  Anchor once and re-read where the frame is next used.
        let frame_anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame) };

        let mut space = self.space;

        // executioncontext.py:353-356 Tracing cases
        let w_callback = if event == "call" {
            self.gettrace()
        } else {
            unsafe { (*frame).get_w_f_trace() }
        };

        if !w_callback.is_null() && event != "leaveframe" {
            // The callback is a nursery function; `normalize_exception`,
            // `getorcreatedebug`, and `fast2locals` all allocate before it is
            // pinned for the call below.  Hold it (and `w_arg`) across those.
            let _callback_roots = pyre_object::gc_roots::push_roots();
            // Pin the carrier before the other `pin_root` safepoints.
            // `expand_pop_roots` reloads it before `normalize_exception`.
            let operr_slot = operr.as_mut().map(|operr| operr.pin(&_callback_roots));
            let callback_live_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_callback);
            let w_arg_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_arg);
            let space_slot = pyre_object::gc_roots::shadow_stack_len();
            space = pyre_object::gc_roots::pin_root(space);
            // `executioncontext.py _trace` exception-event branch:
            //   if operr is not None:
            //       w_value = operr.normalize_exception(space)
            //       w_arg = space.newtuple([operr.w_type, w_value,
            //                               operr.get_w_traceback(space)])
            //
            // Normalization mutates the caller's live carrier in place.
            // Pyre's normalized value owns the authoritative class and
            // traceback slots, so read both from it instead of rebuilding a
            // second interpreter-level exception object.
            let w_trace_arg = if let Some(operr) = operr {
                if let Some(slot) = operr_slot {
                    operr.reload(&_callback_roots, slot);
                }
                let w_value = operr.normalize_exception(space)?;
                let w_type = crate::typedef::r#type(w_value)
                    .map_or_else(pyre_object::w_none, |p| p.as_ptr());
                // `operr.get_w_traceback(space)` — the slot read, its escape
                // mark, and the `w_None` for an empty chain.  The normalization
                // above put the same instance on the carrier, so the carrier
                // reads it back.
                let w_traceback =
                    operr.get_w_traceback(pyre_object::gc_roots::shadow_stack_get(space_slot));
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(w_type);
                fields.push(w_value);
                fields.push(w_traceback);
                pyre_object::tupleobject::w_tuple_new(fields.take())
            } else {
                pyre_object::gc_roots::shadow_stack_get(w_arg_slot)
            };
            let _ = pyre_object::gc_roots::pin_root(w_trace_arg);
            let arg_live_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

            let frame = frame_anchor.live();
            let lineno = unsafe { (*frame).get_last_lineno() };
            let init_lineno = if unsafe { (*frame).last_instr } >= 1 {
                lineno
            } else {
                -1
            };
            let had_locals = unsafe {
                let d = (*frame).getorcreatedebug(init_lineno);
                !d.w_locals.is_null()
            };
            if had_locals {
                unsafe { (*frame).fast2locals()? };
            }
            let frame = frame_anchor.live();
            let (prev_line_tracing, old_lineno) = unsafe {
                let d = (*frame).getorcreatedebug(init_lineno);
                (d.is_in_line_tracing, d.f_lineno)
            };

            // executioncontext.py:376 self.is_tracing += 1
            self.is_tracing += 1;
            let mut call_result = (|| {
                unsafe {
                    let d = (*frame).getorcreatedebug(init_lineno);
                    if event == "line" {
                        d.is_in_line_tracing = true;
                    }
                    d.f_lineno = lineno;
                }
                // executioncontext.py — `frame = jit.hint(frame,
                // access_directly=False)`: from here on, frame is just a
                // normal w_object.
                let frame = majit_metainterp::jit::hint_no_access_directly(frame);
                // executioncontext.py:382-385 space.call_function(w_callback, frame, w_event, w_arg)
                let _trace_roots = pyre_object::gc_roots::push_roots();
                let callback_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(pyre_object::gc_roots::shadow_stack_get(
                    callback_live_slot,
                ));
                let frame_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(wrap_trace_frame(frame_anchor.live()));
                let event_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new(event));
                let arg_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(pyre_object::gc_roots::shadow_stack_get(
                    arg_live_slot,
                ));
                let call_result = crate::call::call_function_impl_result(
                    pyre_object::gc_roots::shadow_stack_get(callback_slot),
                    &[
                        pyre_object::gc_roots::shadow_stack_get(frame_slot),
                        pyre_object::gc_roots::shadow_stack_get(event_slot),
                        pyre_object::gc_roots::shadow_stack_get(arg_slot),
                    ],
                );
                // The callback received the live frame, so its
                // `frame.f_trace = local` / `frame.f_lineno = N` setattrs
                // already landed on the frame's getsets — no writeback pass.
                let w_result = call_result?;
                let result_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(w_result);
                let frame = frame_anchor.live();
                let w_result = pyre_object::gc_roots::shadow_stack_get(result_slot);
                if w_result != pyre_object::w_none() {
                    unsafe {
                        (*frame).getorcreatedebug(init_lineno).w_f_trace = w_result;
                    }
                }
                Ok::<(), crate::PyError>(())
            })();

            if call_result.is_err() {
                self.settrace(pyre_object::w_none());
                unsafe {
                    (*frame_anchor.live())
                        .getorcreatedebug(init_lineno)
                        .w_f_trace = pyre_object::PY_NULL;
                }
            }
            let frame = frame_anchor.live();

            unsafe {
                let d = (*frame).getorcreatedebug(init_lineno);
                if d.f_lineno == lineno {
                    // executioncontext.py:397-404 — for generator/coroutine
                    // resumptions (`event == 'call'` while `last_instr >= 0`)
                    // keep `d.f_lineno` at the yield line so the instruction
                    // right after `YIELD_VALUE` (still on the same source
                    // line) does not fire a spurious line event; skip the
                    // restore in that case.
                    if event != "call" || (*frame).last_instr < 0 {
                        d.f_lineno = old_lineno;
                    }
                }
                d.is_in_line_tracing = prev_line_tracing;
            }
            self.is_tracing -= 1;
            // executioncontext.py:401-402 reads `d.w_locals is not None`
            // again inside finally — the callback may have installed or
            // cleared w_locals.  Re-read here instead of caching
            // `had_locals` from before the call.
            let post_had_locals = unsafe {
                let d = (*frame).getorcreatedebug(init_lineno);
                !d.w_locals.is_null()
            };
            if post_had_locals {
                call_result = match call_result {
                    Ok(()) => {
                        unsafe { (*frame).locals2fast(false)? };
                        Ok(())
                    }
                    Err(error) => {
                        let mut error = error;
                        let held = pyre_object::gc_roots::push_roots();
                        let error_slot = error.pin(&held);
                        unsafe { (*frame).locals2fast(false)? };
                        error.reload(&held, error_slot);
                        Err(error)
                    }
                };
            }
            space = pyre_object::gc_roots::shadow_stack_get(space_slot);
            w_arg = pyre_object::gc_roots::shadow_stack_get(w_arg_slot);
            // executioncontext.py:392-395 — re-raise the callback's
            // exception after restoring trace bookkeeping. Caller chain
            // (call_trace/return_trace/bytecode_trace/exception_trace
            // → eval_frame_plain) propagates the error up so the
            // tracefunc behaves like an inline raise from the executing
            // frame.
            call_result?;
        }

        // executioncontext.py:404-428 Profile cases
        if let Some(profilefunc) = self.profilefunc {
            // executioncontext.py:406-411 — only call/return/c_call/
            // c_return/c_exception events fire the profile callback.
            let event = if event == "leaveframe" {
                "return"
            } else if event == "call"
                || event == "c_call"
                || event == "c_return"
                || event == "c_exception"
            {
                event
            } else {
                return Ok(());
            };

            // executioncontext.py:416-417 assert self.is_tracing == 0; self.is_tracing += 1
            assert_eq!(self.is_tracing, 0);
            self.is_tracing += 1;
            // executioncontext.py:420-425 self.profilefunc(...), clearing
            // profile slots on exceptions.
            let profile_result = profilefunc(
                space,
                self.w_profilefuncarg,
                frame_anchor.live(),
                event,
                w_arg,
            );
            if let Err(err) = profile_result {
                // executioncontext.py:421-425 — clear profile slots and
                // re-raise so the caller observes the failure (matches
                // the bare `raise` after the `except:` block).
                self.store_profilefunc(None, pyre_object::PY_NULL);
                self.is_tracing -= 1;
                return Err(err);
            }
            // executioncontext.py:427-428 self.is_tracing -= 1
            self.is_tracing -= 1;
        }
        Ok(())
    }

    /// executioncontext.py `checksignals`:
    /// ```python
    /// if self.space.check_signal_action is not None:
    ///     self.space.check_signal_action.perform(self, None)
    /// ```
    /// Called from the EINTR retry paths (`raise_signal`, `pthread_kill`,
    /// `pthread_sigmask`) to deliver a signal that may have arrived
    /// mid-syscall.  Propagates a handler exception (e.g.
    /// `KeyboardInterrupt`) to the caller.
    pub fn checksignals(&mut self) -> Result<(), crate::PyError> {
        if let Some(action) = self.check_signal_action {
            let self_ptr = self as *mut ExecutionContext;
            unsafe {
                perform_async_action(action, self_ptr, std::ptr::null_mut())?;
            }
        }
        Ok(())
    }

    pub fn _revdb_enter(&mut self, _frame: *mut PyFrame) {
        let _ = _frame;
    }

    pub fn _revdb_leave(&mut self, _got_exception: bool) {
        let _ = _got_exception;
    }

    pub fn _revdb_potential_stop_point(&mut self, _frame: *mut PyFrame) {
        let _ = _frame;
    }

    #[allow(unreachable_code)]
    pub fn _freeze_(&self) {
        if !self.topframeref.is_null() {}
    }

    /// Celldict globals for a fresh module (`__main__`, imported source
    /// modules). PyPy `imp.importing.exec_code_module` adds only the
    /// `__builtins__` module to this namespace; builtin names remain a
    /// LOAD_GLOBAL fallback and are not observable module globals.
    pub fn fresh_module_globals(&self) -> PyObjectRef {
        let dict = pyre_object::dictmultiobject::w_module_dict_new();
        // Root the fresh dict while installing `__builtins__`: allocating its
        // module-dict cell may trigger a minor collection before the caller
        // has stored the new namespace anywhere else.
        let _root = pyre_object::gc_roots::push_roots();
        let dict = pyre_object::gc_roots::pin_root(dict);
        unsafe {
            let w_builtin = self.get_builtin();
            if !w_builtin.is_null() {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    dict,
                    "__builtins__",
                    w_builtin,
                );
            }
        }
        dict
    }

    /// `pypy/module/__builtin__/moduledef.py:Module.__init__` — `space.builtin`
    /// is a `Module` (not a dict).  Lazily build a `Module` whose
    /// backing dict IS `self.builtins_module`, so subsequent
    /// `module.getdict(space)` access (`pyframe.py fget_f_builtins`)
    /// surfaces the same storage as a dict view.  The cache field stores
    /// the Module identity so identity-sensitive callers (PyPy
    /// `pick_builtin` `if w_builtin is space.builtin: return space.builtin`)
    /// observe the same object every call.
    /// The dict half of [`Self::get_builtin`] — `interp->builtins`, the
    /// object 3.14 plants as `__builtins__` in a fresh `exec`/`eval`
    /// namespace and in every imported module (only `__main__` gets the
    /// module itself).
    pub fn get_builtin_dict(&self) -> PyObjectRef {
        self.builtins_module
    }

    pub fn get_builtin(&self) -> PyObjectRef {
        let cached = execution_context_builtin_cache_get(self);
        if !cached.is_null() {
            return cached;
        }
        // `pypy/interpreter/module.py:Module.__init__` — `space.builtin`
        // is a `Module` whose `w_dict` is the `W_ModuleDictObject`
        // allocated by `allocate_and_init_instance(module=True)`
        // (`dictmultiobject.py`).  Pyre's
        // `new_builtin_module_dict` already populated the
        // W_ModuleDictObject in `self.builtins_module`; wrap it
        // through `w_module_new_aliasing_dict` without a raw storage
        // pointer (the strategy storage IS the canonical store).
        // Enumerating a dict materialises key objects and can collect.  Keep
        // only owned key strings across that operation; raw values must be
        // looked up again after the module allocation.
        let entries = unsafe { pyre_object::w_dict_str_entries(self.builtins_module) };
        let mut keys = Vec::with_capacity(entries.len());
        for (key, _) in entries {
            keys.push(key);
        }
        let module = pyre_object::w_module_new_aliasing_dict(
            pyre_object::w_str_new("builtins"),
            self.builtins_module,
        );
        // function.py BuiltinFunction.w_moduleobj. The defining
        // module object does not exist while `new_builtin_module_dict` fills
        // the namespace, so bind each builtin's `__self__` now.
        for key in keys {
            if let Some(value) =
                unsafe { pyre_object::w_dict_getitem_str(self.builtins_module, &key) }
            {
                unsafe { crate::function::builtin_function_set_module_obj(value, module) };
            }
        }
        self.builtin_dict_cache.set(module);
        // `baseobjspace.py:650` installs a self-reference here
        // (`self.setitem(self.builtin.w_dict, '__builtins__', w_builtin)`).
        // 3.14 does not: `__builtins__` is planted by the frame/import
        // machinery into *other* modules' globals, so `builtins` itself has
        // no such key and `import builtins; builtins.__builtins__` raises
        // AttributeError.
        module
    }

    /// Direct lookup into the builtins storage, bypassing the dict-object
    /// wrapper (whose internal hash table does not see entries inserted
    /// through `install_default_builtins` on the underlying DictStorage).
    /// Used by `LOAD_GLOBAL` / `LOAD_NAME` to reach builtins like `print`
    /// when the frame's `w_globals` lacks the name (pypy/interpreter/
    /// pyopcode.py LOAD_GLOBAL builtin fallback).
    pub fn lookup_builtin(&self, name: &str) -> Option<PyObjectRef> {
        // `pypy/interpreter/pyopcode.py LOAD_GLOBAL` builtin
        // fallback reads through the builtin Module's W_DictObject
        // (`space.getitem(space.builtin.w_dict, name)`).  Pyre's
        // `builtins_module` IS that W_ModuleDictObject; the
        // dispatching `w_dict_getitem_str` routes through
        // `ModuleDictStrategy::getitem_str` (`celldict.py`).
        unsafe { pyre_object::w_dict_getitem_str(self.builtins_module, name) }
    }
}

/// Residual entry point for the bytecode-trace slow path: run pending
/// async actions (signal delivery, finalizers).  Marked
/// Arm the eval breaker's async bit (`interp_signal.py:34-36`
/// `SignalActionFlag.reset_ticker` publishing a negative ticker).
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`): the
/// bit lives in a process-global word an OS signal handler also writes, so
/// the tracer cannot model the read-modify-write and residualises it, the
/// [`disarm_async_eval_breaker`] twin.
#[majit_macros::dont_look_inside]
pub fn arm_async_eval_breaker() {
    majit_ir::eval_breaker_word::set_async();
}

/// Clear the eval breaker's async bit — see [`arm_async_eval_breaker`].
#[majit_macros::dont_look_inside]
pub fn disarm_async_eval_breaker() {
    majit_ir::eval_breaker_word::clear_async();
}

fn has_pending_signal_action() -> bool {
    crate::module::signal::signalstate::has_pending_signals()
}

/// `dont_look_inside` so the tracer treats it as an opaque call and never
/// follows the action machinery's trait-object virtual dispatch +
/// `Result<(), PyError>` propagation (which the JIT codewriter cannot
/// model).  Returns 0 on success and 1 when an action raised, stashing
/// the error in `PENDING_CALL_ERROR` for the caller to re-raise — the
/// same cross-residual error convention as `call_function_impl`.
#[majit_macros::dont_look_inside]
pub extern "C" fn perform_pending_actions(ec: *mut ExecutionContext, frame: *mut PyFrame) -> i64 {
    if ec.is_null() {
        return 0;
    }
    match unsafe { (*ec).perform_actions(frame) } {
        Ok(()) => 0,
        Err(err) => {
            crate::call::set_call_error(err);
            1
        }
    }
}

/// Invoke one action, ending the action object's exclusive borrow before a
/// requested GIL hand-off.  `GILReleaseAction` is process-owned, so the next
/// thread can dispatch this same object as soon as it acquires the GIL.  A
/// direct `perform()` implementation that yields would leave the first
/// thread's `&mut dyn AsyncActionOps` live across that second mutable borrow.
///
/// # Safety
///
/// `action` and `ec` must point to live objects for the duration of the
/// action call.  The caller must hold the GIL on entry.
unsafe fn perform_async_action(
    action: *mut dyn AsyncActionOps,
    ec: *mut ExecutionContext,
    frame: *mut PyFrame,
) -> Result<(), crate::PyError> {
    let control = unsafe { (*action).perform(&mut *ec, frame)? };
    if control == AsyncActionControl::YieldGil {
        majit_gc::rgil::yield_thread();
    }
    Ok(())
}

#[derive(Clone)]
pub struct AbstractActionFlag {
    _periodic_actions: Vec<*mut dyn AsyncActionOps>,
    _nonperiodic_actions: Vec<*mut dyn AsyncActionOps>,
    pub(crate) _fired_bitmask: usize,
    has_bytecode_counter: bool,
    pub checkinterval_scaled: usize,
}

impl Default for AbstractActionFlag {
    fn default() -> Self {
        Self::new()
    }
}

impl AbstractActionFlag {
    pub fn new() -> Self {
        Self {
            _periodic_actions: Vec::new(),
            _nonperiodic_actions: Vec::new(),
            _fired_bitmask: 0,
            has_bytecode_counter: false,
            checkinterval_scaled: 10000 * TICK_COUNTER_STEP,
        }
    }

    /// pypy/interpreter/executioncontext.py:531-556
    /// `_rebuild_action_dispatcher`.
    ///
    /// TODO(justified): PyPy rebuilds an inner
    /// closure on every `register_*_action` call so that
    /// `periodic_actions = unrolling_iterable(self._periodic_actions)`
    /// captures the current periodic-action list at codegen time;
    /// RPython's JIT then unrolls the periodic loop at trace time.
    /// `unrolling_iterable` is an RPython-only JIT hint with no
    /// runtime semantic effect — the closure body is identical to a
    /// straightforward `for action in self._periodic_actions: …`.
    ///
    /// Pyre's `action_dispatcher` (`ActionFlagOps` trait default,
    /// in this module) iterates `_periodic_actions`
    /// directly each call.  That body is `@jit.dont_look_inside`, so
    /// the tracer never unrolls the periodic loop.
    /// `unrolling_iterable` only froze the closure's captured list;
    /// rebuilding it changes nothing the tracer records.
    pub fn _rebuild_action_dispatcher(&mut self) {}
}

/// `pypy/interpreter/executioncontext.py` — `AbstractActionFlag`
/// + `ActionFlag` virtual-dispatch interface.
///
/// PyPy's `AbstractActionFlag.fire` (line 482), `setcheckinterval`
/// (line 522), and `action_dispatcher` (line 531) all call
/// `self.reset_ticker(...)`, which Python resolves through the
/// concrete subclass override (`ActionFlag.reset_ticker` at line 574
/// writes `self._ticker = value`; `SignalActionFlag.reset_ticker`
/// writes a C global).  Rust composition cannot virtual-dispatch
/// from a base-struct inherent method, so this trait carries the
/// virtual surface (`reset_ticker` / `get_ticker` /
/// `decrement_ticker`) plus an accessor pair (`abstract_flag` /
/// `abstract_flag_mut`) that lets the default `fire` /
/// `setcheckinterval` / `action_dispatcher` implementations reach
/// the shared `_periodic_actions` / `_nonperiodic_actions` /
/// `_fired_bitmask` / `checkinterval_scaled` state.  Concrete
/// subclasses (`ActionFlag`) implement only the required methods;
/// the default bodies stay a line-for-line port of upstream.
pub trait ActionFlagOps {
    /// pypy/interpreter/executioncontext.py `ActionFlag.reset_ticker`.
    fn reset_ticker(&mut self, value: isize);

    /// pypy/interpreter/executioncontext.py `ActionFlag.get_ticker`.
    fn get_ticker(&self) -> isize;

    /// pypy/interpreter/executioncontext.py `ActionFlag.decrement_ticker`.
    fn decrement_ticker(&mut self, by: isize) -> isize;

    /// interp_signal.py `SignalActionFlag.rearm_ticker` —
    /// `p.c_value = -1`. `_raise_in_thread` calls this after writing the
    /// EC-owned async-exception slot so `action_dispatcher` runs.
    fn rearm_ticker(&mut self) {
        self.reset_ticker(-1);
    }

    /// Composition accessor: shared `AbstractActionFlag` state.
    fn abstract_flag(&self) -> &AbstractActionFlag;
    /// Composition accessor: shared `AbstractActionFlag` state.
    fn abstract_flag_mut(&mut self) -> &mut AbstractActionFlag;

    /// pypy/interpreter/executioncontext.py:493-510
    /// `register_periodic_action`.
    ///
    /// ```python
    /// if use_bytecode_counter:
    ///     self._periodic_actions.append(action)
    ///     self.has_bytecode_counter = True
    /// else:
    ///     self._periodic_actions.insert(0, action)
    /// self._rebuild_action_dispatcher()
    /// ```
    ///
    /// PyPy comment (line 503-504): "hack to put the release-the-GIL
    /// one at the end of the list, and the report-the-signals one at
    /// the start of the list."  When `use_bytecode_counter` is False
    /// (signal handling), the action is prepended so it runs first.
    ///
    /// Lives on the trait (PyPy hangs `register_periodic_action` on
    /// `AbstractActionFlag` itself, never overridden by `ActionFlag`)
    /// so `*mut dyn ActionFlagOps` callers — the type pyre uses for
    /// `space.actionflag` to admit a future `SignalActionFlag`
    /// analogue — can reach it without downcasting.
    ///
    /// PyPy line 502 asserts `isinstance(action, PeriodicAsyncAction)`.
    /// Pyre enforces the same constraint at compile time by typing the
    /// argument as `*mut dyn PeriodicAsyncActionOps` — non-periodic
    /// `AsyncAction` subclasses do not impl that subtrait, so passing
    /// one is a type error rather than a runtime assert failure.
    fn register_periodic_action(
        &mut self,
        action: *mut dyn PeriodicAsyncActionOps,
        use_bytecode_counter: bool,
    ) {
        // Trait upcast (stable since Rust 1.86) erases the periodic
        // marker so storage stays as the polymorphic
        // `*mut dyn AsyncActionOps` that `action_dispatcher` reads.
        let action_ops: *mut dyn AsyncActionOps = action;
        let flag = self.abstract_flag_mut();
        if use_bytecode_counter {
            flag._periodic_actions.push(action_ops);
            flag.has_bytecode_counter = true;
        } else {
            flag._periodic_actions.insert(0, action_ops);
        }
        flag._rebuild_action_dispatcher();
    }

    /// pypy/interpreter/executioncontext.py:512-517
    /// `register_nonperiodic_action`.  Trait default for the same
    /// reason as `register_periodic_action` above.
    fn register_nonperiodic_action(&mut self, action: *mut dyn AsyncActionOps) -> isize {
        let flag = self.abstract_flag_mut();
        flag._nonperiodic_actions.push(action);
        assert!(flag._nonperiodic_actions.len() < 32);
        flag._rebuild_action_dispatcher();
        (flag._nonperiodic_actions.len() - 1) as isize
    }

    /// pypy/interpreter/executioncontext.py `getcheckinterval`.
    fn getcheckinterval(&self) -> usize {
        self.abstract_flag().checkinterval_scaled / TICK_COUNTER_STEP
    }

    /// pypy/interpreter/executioncontext.py `fire`.
    ///
    /// ```python
    /// def fire(self, action):
    ///     "Request for the action to be run before the next opcode."
    ///     assert action._action_index >= 0
    ///     mask = r_uint(1) << action._action_index
    ///     if not self._fired_bitmask & mask:
    ///         self._fired_bitmask |= mask
    ///         # set the ticker to -1 in order to force action_dispatcher()
    ///         # to run at the next possible bytecode
    ///         self.reset_ticker(-1)
    /// ```
    ///
    /// `@jit.dont_look_inside` (`AbstractActionFlag.fire`). GC callbacks
    /// write `_fired_bitmask` at allocations the tracer cannot see, so
    /// this body stays opaque. The marker is a body-local const:
    /// `#[dont_look_inside]` emits a generic prebuild method, and that
    /// method would make object-safe `ActionFlagOps` unusable as
    /// `dyn ActionFlagOps`.
    fn fire(&mut self, action: *mut dyn AsyncActionOps) {
        #[doc(hidden)]
        #[allow(non_upper_case_globals, dead_code)]
        const _jit_look_inside_fire: bool = false;
        if action.is_null() {
            return;
        }
        let base = unsafe { (*action).async_action() };
        // executioncontext.py — `assert action._action_index >= 0`.
        debug_assert!(
            base._action_index >= 0,
            "periodic actions must not call fire"
        );
        let bitmask = base.bitmask;
        let already_fired = self.abstract_flag()._fired_bitmask & bitmask != 0;
        if !already_fired {
            self.abstract_flag_mut()._fired_bitmask |= bitmask;
            self.reset_ticker(-1);
        }
    }

    /// pypy/interpreter/executioncontext.py `setcheckinterval`.
    ///
    /// ```python
    /// def setcheckinterval(self, interval):
    ///     MAX = sys.maxint // TICK_COUNTER_STEP
    ///     if interval < 1:
    ///         interval = 1
    ///     elif interval > MAX:
    ///         interval = MAX
    ///     self.checkinterval_scaled = interval * TICK_COUNTER_STEP
    ///     self.reset_ticker(-1)
    /// ```
    fn setcheckinterval(&mut self, interval: usize) {
        let max = usize::MAX / TICK_COUNTER_STEP;
        let interval = interval.max(1).min(max);
        self.abstract_flag_mut().checkinterval_scaled = interval * TICK_COUNTER_STEP;
        self.reset_ticker(-1);
    }

    /// pypy/interpreter/executioncontext.py `action_dispatcher`.
    ///
    /// ```python
    /// def action_dispatcher(ec, frame):
    ///     # periodic actions (first reset the bytecode counter)
    ///     self.reset_ticker(self.checkinterval_scaled)
    ///     for action in periodic_actions:
    ///         action.perform(ec, frame)
    ///     # nonperiodic actions
    ///     if self._fired_bitmask:
    ///         for i in range(len(self._nonperiodic_actions)):
    ///             mask = r_uint(1) << i
    ///             if self._fired_bitmask & mask:
    ///                 action = self._nonperiodic_actions[i]
    ///                 self._fired_bitmask &= ~mask
    ///                 action.perform(ec, frame)
    ///         if self._fired_bitmask:
    ///             self.reset_ticker(-1)
    /// ```
    ///
    /// A periodic/nonperiodic `perform` may raise (e.g.
    /// `CheckSignalAction` delivering `KeyboardInterrupt`); the error
    /// propagates out via `?`, aborting the remaining actions, exactly
    /// like PyPy's `action.perform(...)` raising an `OperationError`.
    ///
    /// `@jit.dont_look_inside` (`_rebuild_action_dispatcher`'s
    /// `action_dispatcher`; also `@objectmodel.dont_inline`). The
    /// bitmask is written from GC callbacks, so the tracer must not
    /// record this body. The marker is a body-local const:
    /// `#[dont_look_inside]` emits a generic prebuild method, and that
    /// method would make object-safe `ActionFlagOps` unusable as
    /// `dyn ActionFlagOps`.
    fn action_dispatcher(
        &mut self,
        ec: *mut ExecutionContext,
        frame: *mut PyFrame,
    ) -> Result<(), crate::PyError> {
        #[doc(hidden)]
        #[allow(non_upper_case_globals, dead_code)]
        const _jit_look_inside_action_dispatcher: bool = false;
        // Each `perform` may deliver a signal, which runs the handler at
        // app level; the next action in the loop is handed the same frame.
        let anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame) };
        // Breaker bits are armed together with `fire_action_ticker`. Handle
        // them before `reset_ticker` so a sticky request is not cleared by
        // the positive reload, and so `MemoryError` reaches the caller's
        // exception path (`pyopcode.py handle_bytecode`).
        service_eval_breaker(ec, anchor.live())?;
        // executioncontext.py — `self.reset_ticker(self.checkinterval_scaled)`.
        let interval = self.abstract_flag().checkinterval_scaled as isize;
        self.reset_ticker(interval);
        // Interpreter GC poll, the `PeriodicAsyncAction` that runs at this
        // wrap. Urgent `EB_GC` also enters through `fire_action_ticker`.
        // `has_bytecode_counter` is false until a periodic action (the GIL
        // release, or this poll) registers, and `decrement_ticker` then does
        // not count, so the ticker never wraps on its own. Registering the
        // poll sets that flag; the period is `getcheckinterval` (10000)
        // bytecodes, against `gc_interp::POLL_INTERVAL` (1024).
        pyre_object::gc_interp::on_ticker_wrap();
        if majit_ir::eval_breaker_word::take_memory_error() {
            majit_gc::gc_sync::safepoint_poll();
            return Err(crate::PyError::memory_error(""));
        }
        // executioncontext.py:539-540 — periodic actions iter.
        // Snapshot pointers before iteration; perform() may register
        // additional actions and mutate `_periodic_actions`.
        let periodic: Vec<*mut dyn AsyncActionOps> = self.abstract_flag()._periodic_actions.clone();
        for action_ptr in periodic {
            if action_ptr.is_null() || ec.is_null() {
                continue;
            }
            unsafe {
                perform_async_action(action_ptr, ec, anchor.live())?;
            }
        }
        // executioncontext.py:543-556 — nonperiodic bit-mask scan.
        // Clear each bit before perform() so a fire() during perform()
        // re-arms cleanly (NB at executioncontext.py).
        if self.abstract_flag()._fired_bitmask != 0 {
            let nactions = self.abstract_flag()._nonperiodic_actions.len();
            for i in 0..nactions {
                let mask: usize = 1usize << i;
                if self.abstract_flag()._fired_bitmask & mask != 0 {
                    let action_ptr = self.abstract_flag()._nonperiodic_actions[i];
                    self.abstract_flag_mut()._fired_bitmask &= !mask;
                    if !action_ptr.is_null() && !ec.is_null() {
                        unsafe {
                            perform_async_action(action_ptr, ec, anchor.live())?;
                        }
                    }
                }
            }
            // executioncontext.py:560-561 — if a higher-index action
            // re-fired an earlier bit during iteration, force the next
            // bytecode to re-enter dispatch.
            if self.abstract_flag()._fired_bitmask != 0 {
                self.reset_ticker(-1);
            }
        }
        // A request that is still armed (STW not finished, signal not
        // consumed, memory error owed to another thread) must make the next
        // bytecode re-enter. `EB_GC_INTERP` is excluded: it stays set for
        // the process lifetime and is not itself an event. `EB_FINALIZING`
        // also stays set, and the thread running teardown is the one it
        // never parks, so only another thread re-arms for it.
        //
        // A lost `-1` from `decrement_ticker`'s plain store is recovered
        // here, at the next wrap, so delivery slips by at most one ticker
        // period (`signals.c pypysig_counter` has the same race).
        let mut pending = majit_ir::eval_breaker_word::EB_ASYNC
            | majit_ir::eval_breaker_word::EB_STW
            | majit_ir::eval_breaker_word::EB_GC
            | majit_ir::eval_breaker_word::EB_MEMORY_ERROR;
        if !crate::module::thread::current_is_finalizing_thread() {
            pending |= majit_ir::eval_breaker_word::EB_FINALIZING;
        }
        let still = majit_ir::eval_breaker_word::load() & pending;
        if still != 0 {
            self.reset_ticker(-1);
        }
        Ok(())
    }
}

/// Events the per-bytecode head used to poll, now run from
/// `action_dispatcher` after `fire_action_ticker` forced the ticker negative.
pub(crate) fn service_eval_breaker(
    ec: *mut ExecutionContext,
    frame: *mut PyFrame,
) -> Result<(), crate::PyError> {
    let _ = frame;
    if majit_ir::eval_breaker_word::load() & majit_ir::eval_breaker_word::EB_STW != 0 {
        majit_gc::gc_sync::safepoint_poll();
    }
    // After the rendezvous: `set_finalizing` publishes the flag inside a
    // stop-the-world, so a thread resuming from that safepoint must park
    // before the actions below run a finalizer on it.
    if majit_ir::eval_breaker_word::load() & majit_ir::eval_breaker_word::EB_FINALIZING != 0 {
        crate::module::thread::park_if_finalizing();
    }
    if majit_ir::eval_breaker_word::take_memory_error() {
        // The collection that armed the error may have requested STW after
        // the poll above.
        majit_gc::gc_sync::safepoint_poll();
        return Err(crate::PyError::memory_error(""));
    }
    if !ec.is_null() && !crate::module::thread::all_thread_hooks_current(unsafe { &*ec }) {
        crate::module::thread::apply_all_thread_hooks(unsafe { &mut *ec })?;
    }
    // A bit armed without a store into this EC's ticker (the published cell
    // can lag registration) is copied here, off the per-bytecode path.
    if !ec.is_null()
        && majit_ir::eval_breaker_word::load() & majit_ir::eval_breaker_word::EB_ASYNC != 0
    {
        unsafe { (*ec).actionflag.sync_async_ticker() };
    }
    Ok(())
}

/// pypy/interpreter/executioncontext.py `ActionFlag` merged with
/// pypy/module/signal/interp_signal.py `SignalActionFlag`.
///
/// PyPy starts with a plain `ActionFlag` (its ticker is a Python field)
/// and, when the `signal` module loads, rebinds `space.actionflag` to a
/// `SignalActionFlag` whose ticker is the C `pypysig_counter` cell.  pyre
/// merges the two. `_ticker` is an `AtomicIsize` (`pypysig_counter`): the
/// OS handler and `fire_action_ticker` store `-1` into it, and the
/// interpreter reads it from `decrement_ticker`.
pub struct ActionFlag {
    base: AbstractActionFlag,
    /// `executioncontext.py ActionFlag._ticker`, stored atomically so an OS
    /// signal handler and a collector can store `-1` (`pypysig_counter`).
    _ticker: std::sync::atomic::AtomicIsize,
}

impl Clone for ActionFlag {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            _ticker: std::sync::atomic::AtomicIsize::new(
                self._ticker.load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

impl Default for ActionFlag {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionFlag {
    pub fn new() -> Self {
        Self {
            base: AbstractActionFlag::new(),
            _ticker: std::sync::atomic::AtomicIsize::new(0),
        }
    }

    pub fn perform_frame_action(&mut self, ec: &mut ExecutionContext, frame: *mut PyFrame) {
        let _ = (ec, frame);
    }

    /// Address used to identify this as the signal-driving ticker.  The OS
    /// handler never dereferences it; it is stable for the process lifetime
    /// because the owning process `ActionFlag` allocation never moves.
    pub fn ticker_addr(&mut self) -> *mut isize {
        self._ticker.as_ptr()
    }

    /// Transfer an asynchronous eval-breaker request into the registered
    /// ticker. The OS handler may already have stored `-1`; this is the
    /// GIL-side copy for a bit armed without that store.
    pub(crate) fn sync_async_ticker(&mut self) {
        if self.is_registered_ticker() {
            self._ticker.store(-1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Count bytecodes on this ticker so `action_dispatcher` (and the GC
    /// poll beside its periodic actions) runs every `getcheckinterval`
    /// dispatches. `decrement_ticker` does not subtract while
    /// `has_bytecode_counter` is false, so without this the poll never
    /// wraps unless a thread or signal action already set the flag.
    /// Does not add another periodic-action object: the poll is the
    /// `on_ticker_wrap` call in `action_dispatcher`.
    pub fn enable_gc_bytecode_counter(&mut self) {
        if self.base.has_bytecode_counter {
            return;
        }
        self.base.has_bytecode_counter = true;
        if self._ticker.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            let interval = self.base.checkinterval_scaled as isize;
            self.reset_ticker(interval);
        }
    }

    /// True when `self._ticker` is the signal-registered ticker cell — the
    /// single cell compiled-loop back-edges poll. Only that ticker drives the
    /// shared async bit; per-EC flags that are not the registered breaker
    /// source must not touch the shared word, or one context's dispatch clear
    /// would drop another context's pending async.
    ///
    /// The cell identity is compared as `usize` rather than via
    /// `ptr::eq`: this runs inside the traced eval loop (`decrement_ticker`),
    /// and a raw-pointer equality on `*const isize` can lower to an
    /// `int_eq/ir>i` kind shape that has no blackhole handler; casting both
    /// addresses to `usize` keeps the comparison an int/int equality.
    fn is_registered_ticker(&self) -> bool {
        let here = self._ticker.as_ptr() as usize;
        let registered = crate::module::signal::signalstate::registered_ticker_ptr() as usize;
        here == registered
    }
}

/// pypy/interpreter/executioncontext.py — `class ActionFlag(AbstractActionFlag)`.
impl ActionFlagOps for ActionFlag {
    fn abstract_flag(&self) -> &AbstractActionFlag {
        &self.base
    }

    fn abstract_flag_mut(&mut self) -> &mut AbstractActionFlag {
        &mut self.base
    }

    /// interp_signal.py `SignalActionFlag.get_ticker` — `p.c_value`.
    /// The safe-checkpoint synchronization above is the Rust equivalent of
    /// the signal handler making this cell negative.
    fn get_ticker(&self) -> isize {
        self._ticker.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// interp_signal.py `SignalActionFlag.reset_ticker` —
    /// `p.c_value = value`.
    fn reset_ticker(&mut self, value: isize) {
        self._ticker
            .store(value, std::sync::atomic::Ordering::Relaxed);
        if self.is_registered_ticker() {
            if value < 0 {
                arm_async_eval_breaker();
            } else {
                let async_work_pending = crate::module::thread::has_pending_async_exception()
                    || has_pending_signal_action();
                if !async_work_pending {
                    disarm_async_eval_breaker();
                }
                // A signal/async exception published between the first
                // pending check and the clear above must re-arm the bit.  If
                // it arrives after this second check, its publisher runs
                // after the clear and leaves the bit armed itself.
                if crate::module::thread::has_pending_async_exception()
                    || has_pending_signal_action()
                {
                    arm_async_eval_breaker();
                }
            }
        }
    }

    /// interp_signal.py `SignalActionFlag.decrement_ticker`.
    ///
    /// ```python
    /// def decrement_ticker(self, by):
    ///     p = pypysig_getaddr_occurred()
    ///     value = p.c_value
    ///     if self.has_bytecode_counter:    # this 'if' is constant-folded
    ///         if jit.isconstant(by) and by == 0:
    ///             pass     # normally constant-folded too
    ///         else:
    ///             value -= by
    ///             p.c_value = value
    ///     return value
    /// ```
    ///
    /// `CheckSignalAction` registers with `use_bytecode_counter=False`,
    /// so `has_bytecode_counter` stays false and the ticker is never
    /// decremented here — it only goes negative when the OS handler calls
    /// `signal_pushback` (or `fire` requests a non-periodic action).
    fn decrement_ticker(&mut self, by: isize) -> isize {
        // `signals.c pypysig_counter`: a relaxed load and a relaxed store.
        // A concurrent `fire_action_ticker` store of `-1` can land between
        // them and be overwritten. That is the race `pypysig_pushback` has.
        // The armer also sets its breaker bit, and `action_dispatcher`
        // calls `reset_ticker(-1)` while any of those bits is still set, so
        // a lost `-1` delays delivery by at most one ticker period, or until
        // `CheckSignalAction::after_thread_switch` re-arms from the pending
        // signal bits when a blocked thread resumes.
        let value = self._ticker.load(std::sync::atomic::Ordering::Relaxed);
        if self.base.has_bytecode_counter {
            if by == 0 {
                return value;
            }
            let next = value.wrapping_sub(by);
            self._ticker
                .store(next, std::sync::atomic::Ordering::Relaxed);
            if next < 0 && self.is_registered_ticker() {
                arm_async_eval_breaker();
            }
            return next;
        }
        value
    }

    fn rearm_ticker(&mut self) {
        self.reset_ticker(-1);
        // Compiled loops poll the eval-breaker word. On wasm the signal
        // module never registers the ticker cell, so reset_ticker alone
        // would leave that word unarmed.
        arm_async_eval_breaker();
    }
}

/// Stable reference to the process-owned `space.actionflag`.
///
/// PyPy stores one `ActionFlag` on `ObjSpace` and every execution context
/// reaches it through `self.space.actionflag` (`executioncontext.py`).
/// Pyre's opaque `PyObjectRef` space has no typed Rust fields yet, so the
/// process runtime owns the equivalent allocation and each EC carries this
/// thin reference.  The GIL serializes interpreter access just as it does for
/// PyPy's plain Python fields; the OS signal handler is the only asynchronous
/// writer and already uses the registered ticker address.
#[derive(Clone, Copy)]
pub struct SpaceActionFlag {
    ptr: *mut ActionFlag,
}

static SPACE_ACTIONFLAG: OnceLock<usize> = OnceLock::new();

impl Default for SpaceActionFlag {
    fn default() -> Self {
        Self::new()
    }
}

impl SpaceActionFlag {
    pub fn new() -> Self {
        let ptr = *SPACE_ACTIONFLAG.get_or_init(|| {
            let ptr = Box::into_raw(Box::new(ActionFlag::new()));
            // The leaked process flag is the ticker cell armers store into.
            majit_ir::eval_breaker_word::publish_action_ticker(unsafe { &raw mut (*ptr)._ticker });
            ptr as usize
        }) as *mut ActionFlag;
        Self { ptr }
    }

    fn inner(&self) -> &ActionFlag {
        // SAFETY: SPACE_ACTIONFLAG owns one leaked process-lifetime value.
        // Interpreter calls are serialized by the GIL; the signal handler
        // touches only process-global atomics, never this allocation.
        unsafe { &*self.ptr }
    }

    fn inner_mut(&mut self) -> &mut ActionFlag {
        // SAFETY: see `inner`; callers hold the GIL while mutating the flag.
        unsafe { &mut *self.ptr }
    }

    /// The stable object-space allocation used when an action caches its
    /// `space.actionflag` back-reference.  Returning the shared allocation,
    /// rather than this EC's thin wrapper, keeps the pointer valid even if an
    /// embedding drops the EC that first registered the action.
    pub fn shared_mut(&mut self) -> &'static mut ActionFlag {
        // SAFETY: SPACE_ACTIONFLAG owns one leaked process-lifetime value.
        // Registration and action dispatch are serialized by the GIL.
        unsafe { &mut *self.ptr }
    }

    pub fn ticker_addr(&mut self) -> *mut isize {
        self.inner_mut().ticker_addr()
    }

    pub(crate) fn sync_async_ticker(&mut self) {
        self.inner_mut().sync_async_ticker()
    }

    pub fn enable_gc_bytecode_counter(&mut self) {
        self.inner_mut().enable_gc_bytecode_counter();
    }
}

/// Arm the bytecode countdown for the interpreter GC poll. The process
/// flag is the ticker every EC shares (`executioncontext.py` `space.actionflag`).
/// Setup at activation entry, never part of a trace.
#[majit_macros::dont_look_inside]
pub fn enable_gc_bytecode_counter() {
    SpaceActionFlag::new().enable_gc_bytecode_counter();
}

impl ActionFlagOps for SpaceActionFlag {
    fn abstract_flag(&self) -> &AbstractActionFlag {
        self.inner().abstract_flag()
    }

    fn abstract_flag_mut(&mut self) -> &mut AbstractActionFlag {
        self.inner_mut().abstract_flag_mut()
    }

    fn get_ticker(&self) -> isize {
        self.inner().get_ticker()
    }

    fn reset_ticker(&mut self, value: isize) {
        self.inner_mut().reset_ticker(value);
    }

    fn decrement_ticker(&mut self, by: isize) -> isize {
        self.inner_mut().decrement_ticker(by)
    }

    fn rearm_ticker(&mut self) {
        self.inner_mut().rearm_ticker();
    }
}

/// `dont_look_inside`: the ticker lives behind `SpaceActionFlag`'s interior
/// pointer, which the codewriter refuses to lower (`rewrite_op_getsubstruct`
/// bakes an abort before such a call). The residual takes the execution
/// context itself, so no interior address appears in the caller's graph.
/// Upstream traces carry the ticker as raw field operations instead; lowering
/// this gate to those ops is a later port.
#[majit_macros::dont_look_inside]
pub fn space_decrement_ticker(ec: &mut ExecutionContext, by: isize) -> isize {
    let mut actionflag = ec.actionflag;
    actionflag.decrement_ticker(by)
}

/// One-word residual-call ABI for [`space_decrement_ticker`].
///
/// The residual lowering types a value-returning call as `(i64, i64) -> i64`;
/// the Rust signature's reference and `isize` are narrower than a word on
/// wasm32, where `call_indirect` type-checks its callee.
pub extern "C" fn space_decrement_ticker_jit_abi(ec: *mut ExecutionContext, by: i64) -> i64 {
    // SAFETY: the residual's first slot is the execution context the walked
    // graph read it from; it outlives the call.
    let ec = unsafe { &mut *(ec) };
    space_decrement_ticker(ec, by as isize) as i64
}

pub struct AsyncAction {
    pub space: PyObjectRef,
    _action_index: isize,
    /// pypy/interpreter/executioncontext.py:600 `self.bitmask = 1 << index`.
    /// Set by `AbstractActionFlag::register_nonperiodic_action` and
    /// consumed by `fire` / `action_dispatcher`'s bit test.
    pub bitmask: usize,
    /// pypy/interpreter/executioncontext.py `AsyncAction.fire`
    /// uses `self.space.actionflag` — a constant lookup once the action
    /// is constructed because PyPy's `space.actionflag` is set once at
    /// `pypy/interpreter/baseobjspace.py` and never replaced.
    /// Pyre's EC stores a thin reference to the process-owned flag, so we cache
    /// the registering reference here (`AsyncAction::new` /
    /// `UserDelAction::new` / `AsyncActionOps::register_periodic_action`).
    /// The process-owned flag outlives every action. `fire()` dereferences
    /// this slot directly,
    /// matching PyPy's `self.space.actionflag.fire(self)` 1:1 without
    /// the TLS detour.  `null` means the action has not been registered
    /// yet — calling `fire` in that state is a programmer error
    /// (PyPy's docstring at line 605 reads "The action must have been
    /// registered at space initalization time.").
    ///
    /// Stored as `*mut dyn ActionFlagOps` so that `fire()` can dispatch
    /// through the `ActionFlagOps::fire` trait default (which lives on
    /// the trait, mirroring PyPy's polymorphic `actionflag.fire` call)
    /// AND so a future `SignalActionFlag` analogue (PyPy's signal-aware
    /// `AbstractActionFlag` subclass) can be installed without touching
    /// `AsyncAction`.  PyPy types `space.actionflag` as the abstract
    /// base, never as the concrete subclass.
    pub actionflag: *mut dyn ActionFlagOps,
}

impl Default for AsyncAction {
    fn default() -> Self {
        Self {
            space: pyre_object::PY_NULL,
            _action_index: -1,
            bitmask: 0,
            // Fat pointer: null data + ActionFlag vtable.  `fire()` /
            // `register_periodic_action` tests `is_null()` on the data
            // pointer before any deref, so the placeholder vtable never
            // gets reached on an unregistered action.
            actionflag: std::ptr::null_mut::<ActionFlag>(),
        }
    }
}

impl AsyncAction {
    /// pypy/interpreter/executioncontext.py `AsyncAction.__init__`.
    ///
    /// ```python
    /// def __init__(self, space):
    ///     self.space = space
    ///     if not isinstance(self, PeriodicAsyncAction):
    ///         self._action_index = self.space.actionflag.register_nonperiodic_action(self)
    ///     else:
    ///         self._action_index = -1
    /// ```
    ///
    /// Returns `Box<Self>` because Rust cannot hand out a stable
    /// pointer to a value before the constructor returns; PyPy's
    /// heap-identity makes `register_nonperiodic_action(self)` safe
    /// in-place.  Caller owns the Box and must keep it alive as long
    /// as the actionflag holds the registered pointer.
    pub fn new(space: PyObjectRef, actionflag: &mut (dyn ActionFlagOps + 'static)) -> Box<Self> {
        let mut action = Box::new(Self {
            space,
            _action_index: -1,
            bitmask: 0,
            actionflag: actionflag as *mut dyn ActionFlagOps,
        });
        let action_ptr: *mut dyn AsyncActionOps = &mut *action;
        let index = actionflag.register_nonperiodic_action(action_ptr);
        action._action_index = index;
        action.bitmask = 1usize << (index as usize);
        action
    }

    /// pypy/interpreter/executioncontext.py:597-600 `else` branch:
    /// `self._action_index = -1`.  Used by PeriodicAsyncAction
    /// subclasses whose `__init__` skips `register_nonperiodic_action`.
    /// The base `AsyncAction` is returned unboxed because
    /// `PeriodicAsyncAction::new` wraps it in a `Box<PeriodicAsyncAction>`
    /// — registration with `register_periodic_action` then captures the
    /// outer Box's stable pointer.
    pub fn new_periodic_base(space: PyObjectRef) -> Self {
        Self {
            space,
            _action_index: -1,
            bitmask: 0,
            // PeriodicAsyncAction subclasses receive their actionflag
            // back-reference at `register_periodic_action` time (the
            // trait default below) rather than at construction —
            // pyjitpl-style mirror of PyPy's two-step `__init__` +
            // `space.actionflag.register_periodic_action(...)` flow.
            //
            // Fat null pointer constructed via ActionFlag upcast — see
            // the `Default` impl above for the same pattern.
            actionflag: std::ptr::null_mut::<ActionFlag>(),
        }
    }
}

/// pypy/interpreter/executioncontext.py — `AsyncAction` virtual surface.
///
/// PyPy resolves `action.perform(ec, frame)` through Python class
/// lookup (line 608-609 — base body is "To be overridden.").  Rust
/// composition cannot virtual-dispatch from a base-struct inherent
/// method, so this trait carries the virtual `perform` entry plus an
/// accessor pair (`async_action` / `async_action_mut`) for the shared
/// `_action_index` / `bitmask` / `space` state held on `AsyncAction`.
/// Concrete actions impl the trait; the action lists store
/// `*mut dyn AsyncActionOps` so the dispatcher reaches the override
/// through the embedded vtable, and `fire` / `action_dispatcher` can
/// reach the base bitmask through the accessor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsyncActionControl {
    Continue,
    /// Complete the action call first, then hand the GIL to a waiter.
    YieldGil,
}

pub trait AsyncActionOps {
    /// pypy/interpreter/executioncontext.py `AsyncAction.perform`:
    /// `def perform(self, executioncontext, frame): "To be overridden."`
    ///
    /// Returns `Result` so an overriding action (e.g. `CheckSignalAction`)
    /// can raise — PyPy's `perform` propagates an `OperationError` up
    /// through `action_dispatcher` to the eval loop.  The control value lets
    /// the process-owned GIL action defer its hand-off until this method's
    /// exclusive receiver has ended.
    fn perform(
        &mut self,
        executioncontext: &mut ExecutionContext,
        frame: *mut PyFrame,
    ) -> Result<AsyncActionControl, crate::PyError>;

    /// Composition accessor: shared `AsyncAction` state.
    fn async_action(&self) -> &AsyncAction;
    /// Composition accessor: shared `AsyncAction` state.
    fn async_action_mut(&mut self) -> &mut AsyncAction;

    /// `AsyncAction.__init__`'s non-periodic registration, for a concrete
    /// action embedded in another stable-address owner.  PyPy can register
    /// `self` during ordinary object construction; Rust must wait until the
    /// outer owner has reached its final address, then initialize the same
    /// `_action_index`/`bitmask`/`space.actionflag` fields in place.
    fn register_nonperiodic_action(
        &mut self,
        space: PyObjectRef,
        actionflag: &mut (dyn ActionFlagOps + 'static),
    ) where
        Self: Sized + 'static,
    {
        // `AsyncAction.__init__` runs once per action upstream. Here the call
        // is a separate step, so a second one would leave the list entry made
        // by the first pointing at an action whose `bitmask` has moved on.
        debug_assert_eq!(
            self.async_action()._action_index,
            -1,
            "register_nonperiodic_action called twice for one action"
        );
        let action_ptr: *mut dyn AsyncActionOps = self;
        let index = actionflag.register_nonperiodic_action(action_ptr);
        let base = self.async_action_mut();
        base.space = space;
        base.actionflag = actionflag as *mut dyn ActionFlagOps;
        base._action_index = index;
        base.bitmask = 1usize << (index as usize);
    }

    /// pypy/interpreter/executioncontext.py `AsyncAction.fire`:
    ///
    /// ```python
    /// @rgc.no_collect
    /// def fire(self):
    ///     "Request for the action to be run before the next opcode."
    ///     self.space.actionflag.fire(self)
    /// ```
    ///
    /// Pyre stores `actionflag` on `ExecutionContext` rather than on
    /// `space` (TODO — pyre's space is an opaque
    /// `PyObjectRef`).  To keep PyPy's `self.space.actionflag.fire(self)`
    /// directness, the actionflag back-reference is cached on the base
    /// `AsyncAction` at registration time (see
    /// `AsyncAction::new` / `UserDelAction::new` / the trait default
    /// `register_periodic_action` above).  `fire()` dereferences that
    /// back-ref unconditionally; the pointer is stable for the process
    /// lifetime because pyrex owns the EC for the entire run.  The
    /// `Self: Sized + 'static` bound captures the concrete subclass
    /// vtable in the `*mut dyn AsyncActionOps` fat pointer so
    /// `actionflag.fire`'s `(*action).async_action()` reads through the
    /// override-aware vtable rather than the bare-base one.
    ///
    /// PyPy's docstring at line 605 reads "The action must have been
    /// registered at space initalization time."  Pyre asserts the same:
    /// `self.actionflag.is_null()` would mean the action skipped both
    /// `register_nonperiodic_action` and `register_periodic_action`,
    /// which is a programmer error.
    fn fire(&mut self)
    where
        Self: Sized + 'static,
    {
        let actionflag = self.async_action().actionflag;
        debug_assert!(
            !actionflag.is_null(),
            "AsyncAction::fire called before registration (executioncontext.py:605)"
        );
        let action_ptr: *mut dyn AsyncActionOps = self;
        // executioncontext.py — `self.space.actionflag.fire(self)`.
        // The trait dispatch reaches the `ActionFlagOps::fire` default
        // body at line 1454 (which encodes executioncontext.py
        // `AbstractActionFlag.fire`).
        unsafe { (*actionflag).fire(action_ptr) };
    }
}

/// pypy/interpreter/executioncontext.py — bare `AsyncAction`.
/// Default `perform` is the "To be overridden." no-op; concrete
/// subclasses override.
impl AsyncActionOps for AsyncAction {
    fn perform(
        &mut self,
        _executioncontext: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<AsyncActionControl, crate::PyError> {
        Ok(AsyncActionControl::Continue)
    }

    fn async_action(&self) -> &AsyncAction {
        self
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        self
    }
}

/// pypy/interpreter/executioncontext.py:612-615 `class
/// PeriodicAsyncAction(AsyncAction)`.  Marker subtrait that narrows
/// `space.actionflag.register_periodic_action(self, ...)` so only
/// types descending from `PeriodicAsyncAction` (the trait analogue of
/// PyPy's class hierarchy) can be registered as periodic.  PyPy
/// guards the same constraint at runtime with
/// `assert isinstance(action, PeriodicAsyncAction)` in
/// `executioncontext.py`; pyre lifts the assertion to the type
/// system.
pub trait PeriodicAsyncActionOps: AsyncActionOps {
    /// pypy/interpreter/executioncontext.py AsyncAction `else` branch
    /// helper: subclasses of `PeriodicAsyncAction` call
    /// `space.actionflag.register_periodic_action(self, ...)` after
    /// `__init__`.  Lifted into this subtrait so the
    /// `*mut dyn PeriodicAsyncActionOps` cast captures the concrete
    /// subclass vtable (which carries the override of `perform`);
    /// `Self: Sized` keeps trait-object dispatch off this method.
    fn register_periodic_action(
        &mut self,
        actionflag: &mut (dyn ActionFlagOps + 'static),
        use_bytecode_counter: bool,
    ) where
        Self: Sized + 'static,
    {
        // executioncontext.py:603-606 — cache the actionflag back-ref
        // on the base AsyncAction so `fire()` can reach it without a
        // TLS lookup.  PyPy's `self.space.actionflag` is a constant
        // lookup once `space.actionflag` is set at
        // `pypy/interpreter/baseobjspace.py` and never replaced;
        // storing the resolved pointer here is the same caching the
        // PyPy implementation does implicitly.
        self.async_action_mut().actionflag = actionflag as *mut dyn ActionFlagOps;
        let action_ptr: *mut dyn PeriodicAsyncActionOps = self;
        actionflag.register_periodic_action(action_ptr, use_bytecode_counter);
    }
}

pub struct PeriodicAsyncAction {
    pub base: AsyncAction,
}

impl PeriodicAsyncAction {
    /// pypy/interpreter/executioncontext.py `AsyncAction.__init__`
    /// `else` arm (`isinstance(self, PeriodicAsyncAction)`): only sets
    /// `_action_index = -1`, leaves registration to subclass-specific
    /// `space.actionflag.register_periodic_action(self, use_bytecode_counter)`.
    ///
    /// Returns `Box<Self>` because `register_periodic_action` (called
    /// after construction by the subclass via `register`) takes a
    /// stable `*mut dyn AsyncActionOps`.
    pub fn new(space: PyObjectRef) -> Box<Self> {
        Box::new(Self {
            base: AsyncAction::new_periodic_base(space),
        })
    }

    // pypy/interpreter/executioncontext.py — PeriodicAsyncAction
    // subclasses call `space.actionflag.register_periodic_action(self,
    // use_bytecode_counter)` after their own `__init__`.  In pyre this
    // is the trait default `AsyncActionOps::register_periodic_action`
    // — it must be reached through the concrete subclass's vtable so
    // the registered fat pointer's `perform` slot points to the
    // override, not `PeriodicAsyncAction::perform` (which is a no-op
    // per `executioncontext.py`).  Never call
    // `PeriodicAsyncAction(...).register_periodic_action(...)`
    // directly without a concrete subclass.
}

/// pypy/interpreter/executioncontext.py:612-615 — `class
/// PeriodicAsyncAction(AsyncAction)`: empty body, inherits the
/// "To be overridden." `perform` from `AsyncAction`.  Concrete
/// subclasses (CheckSignalAction etc.) override.
impl AsyncActionOps for PeriodicAsyncAction {
    fn perform(
        &mut self,
        _executioncontext: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<AsyncActionControl, crate::PyError> {
        Ok(AsyncActionControl::Continue)
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base
    }
}

/// pypy/interpreter/executioncontext.py — `PeriodicAsyncAction`
/// admits `space.actionflag.register_periodic_action(self, ...)`.
impl PeriodicAsyncActionOps for PeriodicAsyncAction {}

pub struct UserDelAction {
    pub base: AsyncAction,
    pub finalizers_lock_count: usize,
    pub enabled_at_app_level: bool,
    pub pending_with_disabled_del: Option<Vec<PyObjectRef>>,
    /// `pypy/interpreter/executioncontext.py:640` —
    /// `self.space.finalizer_queue` access target.
    ///
    /// TODO: PyPy reads the queue via `self.space`
    /// (typed `ObjSpace`).  Pyre's `space` is an opaque `PyObjectRef`,
    /// so the queue is held as a UserDelAction field instead.  When
    /// GC integration lands a typed space surface this
    /// can be folded back into `space.finalizer_queue`.
    pub finalizer_queue: WRootFinalizerQueue,
}

impl UserDelAction {
    /// pypy/interpreter/executioncontext.py `UserDelAction.__init__`.
    ///
    /// ```python
    /// class UserDelAction(AsyncAction):
    ///     def __init__(self, space):
    ///         AsyncAction.__init__(self, space)
    ///         self.finalizers_lock_count = 0
    ///         self.enabled_at_app_level = True
    ///         self.pending_with_disabled_del = None
    /// ```
    ///
    /// `AsyncAction.__init__` (executioncontext.py) registers
    /// the live instance as a nonperiodic action because UserDelAction
    /// is NOT a PeriodicAsyncAction subclass.  Pyre boxes the
    /// UserDelAction so registration captures a stable
    /// `*mut dyn AsyncActionOps` whose vtable dispatches into
    /// `UserDelAction::perform` (executioncontext.py).
    pub fn new(space: PyObjectRef, actionflag: &mut (dyn ActionFlagOps + 'static)) -> Box<Self> {
        let mut action = Box::new(Self {
            base: AsyncAction {
                space,
                _action_index: -1,
                bitmask: 0,
                // executioncontext.py:603-606 — cache actionflag
                // back-ref for `fire()` (mirror of PyPy's
                // `self.space.actionflag` constant lookup).
                actionflag: actionflag as *mut dyn ActionFlagOps,
            },
            finalizers_lock_count: 0,
            enabled_at_app_level: true,
            pending_with_disabled_del: None,
            finalizer_queue: WRootFinalizerQueue,
        });
        let action_ptr: *mut dyn AsyncActionOps = &mut *action;
        let index = actionflag.register_nonperiodic_action(action_ptr);
        action.base._action_index = index;
        action.base.bitmask = 1usize << (index as usize);
        action
    }

    /// `UserDelAction._run_finalizers`:
    /// ```python
    /// @jit.dont_look_inside
    /// def _run_finalizers(self):
    ///     # called by perform() when we have to "perform" this action,
    ///     # and also directly at the end of gc.collect).
    ///     while True:
    ///         w_obj = self.space.finalizer_queue.next_dead()
    ///         if w_obj is None:
    ///             break
    ///         self._call_finalizer(w_obj)
    /// ```
    ///
    /// `self.space.finalizer_queue` is read via the local
    /// `self.finalizer_queue` field (see struct doc for
    /// TODO).
    ///
    /// The hint is what keeps the body opaque.  `look_inside_graph` already
    /// rejects this graph for its loop, so carrying it changes no jitcode
    /// today — it only drops the name from the `unsafe_loopy_graphs`
    /// diagnostic — but that rejection is a property of the loop rather than
    /// a decision about the body, and the body is one to keep out of a trace:
    /// it drains a collector queue and runs app-level finalizers.
    #[majit_macros::dont_look_inside]
    pub fn _run_finalizers(&mut self) {
        loop {
            let w_obj = self.finalizer_queue.next_dead();
            match w_obj {
                None => break,
                Some(w) => {
                    // The deque entry was the object's only root; pin it across the __del__ call.
                    let _roots = pyre_object::gc_roots::push_roots();
                    let w = pyre_object::gc_roots::pin_root(w);
                    self._call_finalizer(w);
                }
            }
        }
    }

    pub fn gc_disabled(&mut self, w_obj: PyObjectRef) -> bool {
        let _ = w_obj;
        if let Some(list) = self.pending_with_disabled_del.as_mut() {
            list.push(w_obj);
            true
        } else {
            false
        }
    }

    /// Defer this finalizer under `gc.disable()`, and otherwise record that it
    /// ran.  Returns whether the caller should go on and run it.
    ///
    /// The queue the object arrives on is the collector's, so the run is
    /// recorded where `finalize_garbage` sets `_PyGC_SET_FINALIZED` — before
    /// the call, not after it the way the refcount path's
    /// `PyObject_CallFinalizer` does.  `gc.is_finalized` reads it back for an
    /// object its own finalizer resurrected.  A deferred finalizer has not run
    /// and is not recorded.
    ///
    /// Every branch of [`Self::_call_finalizer`] that runs a `tp_finalize`
    /// equivalent comes through here: the async-generator awaitables, the
    /// generators and coroutines, and `__del__`.  The remaining branches
    /// release a buffer export, which `tp_dealloc` does and `tp_finalize`
    /// does not.
    pub fn begin_finalizer(&mut self, w_obj: PyObjectRef) -> bool {
        if self.gc_disabled(w_obj) {
            return false;
        }
        majit_gc::gc_mark_finalizer_run(w_obj as usize);
        true
    }

    pub fn _call_finalizer(&mut self, w_obj: PyObjectRef) {
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_obj);
        let current = || pyre_object::gc_roots::shadow_stack_get(root_base);
        crate::module::_weakref::interp__weakref::finalize_weakrefs(current());
        if let Some(pb) = crate::module::__pypy__::W_PickleBuffer::from_obj(current()) {
            pb.release_export();
            return;
        }
        if let Some(iter) = crate::module::r#struct::unpack_iter::W_UnpackIter::from_obj(current())
        {
            iter.release_export();
            return;
        }
        #[cfg(all(windows, feature = "host_env"))]
        if let Some(w_hkey) = crate::module::_winreg::interp_winreg::W_HKEY::from_obj(current()) {
            w_hkey._finalize_();
            return;
        }
        if unsafe { pyre_object::interp_sre::is_sre_scanner(current()) } {
            unsafe {
                crate::module::_sre::interp_sre::sre_scanner_release_export(current());
            }
            return;
        }
        if unsafe { pyre_object::memoryview::is_w_memoryview(current()) } {
            unsafe { crate::builtins::memoryview_finalize(current()) };
            return;
        }
        if let Some(hooks) = crate::importing::optional_module_hooks()
            && let Some(calls_python) = (hooks.cffi_finalizer_kind)(current())
        {
            if calls_python && !self.begin_finalizer(current()) {
                return;
            }
            (hooks.run_cffi_finalize)(current());
            return;
        }
        if pyre_object::generator::AsyncGenASend::from_obj(current()).is_some()
            || pyre_object::generator::AsyncGenAThrow::from_obj(current()).is_some()
        {
            // executioncontext.py gc_disabled — a finalizer that would call a
            // user-defined app-level function defers under `gc.disable()`.
            // This one emits a RuntimeWarning, so it reaches the overridable
            // `warnings` machinery, exactly like the generator branch below.
            if !self.begin_finalizer(current()) {
                return;
            }
            crate::baseobjspace::async_gen_awaitable_finalize(current());
            return;
        }
        if unsafe { pyre_object::generator::is_generator_or_coroutine(current()) } {
            if !self.begin_finalizer(current()) {
                return;
            }
            if let Err(error) = crate::baseobjspace::generator_finalize(current()) {
                let mut error = error;
                let err_slot = error.pin(&_roots);
                // CPython 3.14 `_PyGen_Finalize` reports close failures with
                // `PyErr_FormatUnraisable`, whose hook object is None and
                // whose message names the generator itself.
                let w_exc = error.to_exc_object();
                let frame = crate::eval::current_frame();
                if !frame.is_null() {
                    unsafe {
                        crate::pytraceback::record_application_traceback(
                            w_exc,
                            frame,
                            (*frame).last_instr as i64,
                        )
                    };
                }
                let repr = unsafe { crate::display::py_repr_wtf8(current()) }
                    .unwrap_or_else(|_| rustpython_wtf8::Wtf8Buf::from_string("<?>".to_owned()));
                let where_desc = crate::display::wtf8_format!("while closing generator ", repr);
                error.reload(&_roots, err_slot);
                error.clone().write_unraisable_with_traceback(
                    self.base.space,
                    &where_desc,
                    pyre_object::w_none(),
                    true,
                    rustpython_wtf8::Wtf8::new(""),
                );
                crate::eval::set_in_flight_exception(pyre_object::PY_NULL);
            }
            return;
        }
        let Some(w_type) = crate::typedef::r#type(current()) else {
            return;
        };
        // A heap type is born nursery
        // (`try_gc_alloc_collecting_rooted`) and `lookup_in_type` /
        // `begin_finalizer` / the `__del__` call all collect, so the type sits
        // on the same shadow stack as the receiver rather than in a raw local.
        let _ = pyre_object::gc_roots::pin_root(w_type.as_ptr());
        let type_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_type = || pyre_object::gc_roots::shadow_stack_get(type_slot);
        let Some(w_del) = (unsafe { crate::baseobjspace::lookup_in_type(w_type(), "__del__") })
        else {
            return;
        };
        if !self.begin_finalizer(current()) {
            return;
        }
        // `__del__` runs arbitrary Python, so it can collect and move the
        // callable that the error report names; publish it and read it back
        // the way `current()` does for the receiver.
        let _ = pyre_object::gc_roots::pin_root(w_del);
        let del_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let del = || pyre_object::gc_roots::shadow_stack_get(del_slot);
        // pyre's combined helper cannot distinguish get-vs-call errors;
        // report through the call arm (executioncontext.py:680-690).
        if let Err(error) =
            unsafe { crate::baseobjspace::get_and_call_function(del(), current(), w_type(), &[]) }
        {
            let mut error = error;
            let err_slot = error.pin(&_roots);
            // PyPy executioncontext.py:680-690 passes an empty `where` and
            // the `__del__` descriptor to `write_unraisable`.  Python 3.14's
            // `_PyErr_FormatUnraisable` gives this finalizer case a more
            // specific hook message; 3.14 is pyre's compatibility target
            // when its observable result differs from PyPy's.
            let del_repr = unsafe { crate::display::py_repr_wtf8(del()) }
                .unwrap_or_else(|_| rustpython_wtf8::Wtf8Buf::from_string("<?>".to_string()));
            let where_desc = crate::display::wtf8_format!("while calling deallocator ", del_repr);
            error.reload(&_roots, err_slot);
            error.clone().write_unraisable_with_traceback(
                self.base.space,
                &where_desc,
                pyre_object::w_none(),
                true,
                rustpython_wtf8::Wtf8::new(""),
            );
        }
    }
}

/// pypy/interpreter/executioncontext.py PeriodicAsyncAction — `class
/// UserDelAction(AsyncAction)`.  `perform` (line 632-633) calls
/// `self._run_finalizers()`.
impl AsyncActionOps for UserDelAction {
    fn perform(
        &mut self,
        _executioncontext: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<AsyncActionControl, crate::PyError> {
        self._run_finalizers();
        Ok(AsyncActionControl::Continue)
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base
    }
}

/// `pypy/module/cpyext/state.py PyObjDeallocAction` — run the
/// deallocators of the mirrors the collector has queued.
///
/// Separate from [`UserDelAction`] because it is a different queue with a
/// different trigger: the collector fires this one from
/// `incminimark.py rrc_invoke_callback` when a raw-refcount link has died, and the
/// deallocator it runs is a C function rather than an app-level `__del__`.
#[cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]
pub struct PyObjDeallocAction {
    pub base: AsyncAction,
}

#[cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]
const _: () = {
    impl PyObjDeallocAction {
        fn new(
            space: PyObjectRef,
            actionflag: &mut (dyn ActionFlagOps + 'static),
        ) -> Box<PyObjDeallocAction> {
            let mut action = Box::new(PyObjDeallocAction {
                base: AsyncAction {
                    space,
                    _action_index: -1,
                    bitmask: 0,
                    actionflag: actionflag as *mut dyn ActionFlagOps,
                },
            });
            let action_ptr: *mut dyn AsyncActionOps = &mut *action;
            let index = actionflag.register_nonperiodic_action(action_ptr);
            action.base._action_index = index;
            action.base.bitmask = 1usize << (index as usize);
            action
        }
    }

    /// `state.py` — `perform` is `_rawrefcount_perform(self.space)`.
    impl AsyncActionOps for PyObjDeallocAction {
        fn perform(
            &mut self,
            _executioncontext: &mut ExecutionContext,
            _frame: *mut PyFrame,
        ) -> Result<AsyncActionControl, crate::PyError> {
            crate::cpyext::pyobject::drain_dead();
            Ok(AsyncActionControl::Continue)
        }

        fn async_action(&self) -> &AsyncAction {
            &self.base
        }

        fn async_action_mut(&mut self) -> &mut AsyncAction {
            &mut self.base
        }
    }
};

#[cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]
static SPACE_CPYEXT_DEALLOC_ACTION: OnceLock<usize> = OnceLock::new();

/// Schedule the drain.  Called from inside a collection, so it may do nothing
/// but set the action flag's bit.
///
/// A no-op before the space has installed its actions: nothing can have created
/// a mirror yet, so nothing can be queued.
#[cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]
pub fn fire_cpyext_dealloc_action() {
    let action = SPACE_CPYEXT_DEALLOC_ACTION.get().copied().unwrap_or(0) as *mut PyObjDeallocAction;
    if !action.is_null() {
        unsafe { (*action).base.fire() };
    }
}

pub fn report_error(
    space: PyObjectRef,
    error: &crate::PyError,
    where_desc: &rustpython_wtf8::Wtf8,
    w_obj: PyObjectRef,
) {
    let mut error = error.clone();
    error.write_unraisable(space, where_desc, w_obj);
}

pub fn make_finalizer_queue<WRoot>(w_root: WRoot, _space: PyObjectRef) -> WRootFinalizerQueue {
    let _ = w_root;
    WRootFinalizerQueue
}

#[cfg(test)]
mod tests {
    use super::{force_frame, force_frame_before_locals_read};
    use crate::PyFrame;
    use std::sync::atomic::{AtomicPtr, Ordering};

    static SEEN: AtomicPtr<PyFrame> = AtomicPtr::new(std::ptr::null_mut());

    unsafe extern "C" fn record(frame: *mut PyFrame) {
        SEEN.store(frame, Ordering::SeqCst);
    }

    #[test]
    fn registered_force_frame_hook_is_invoked_with_the_frame_pointer() {
        let _lock = crate::stack_check::test_support::stack_and_jit_test_guard();
        force_frame_before_locals_read(std::ptr::null_mut());

        SEEN.store(std::ptr::null_mut(), Ordering::SeqCst);
        let _hook = super::install_force_frame_hook(record);
        let dummy = 0x0000_0000_DEAD_BEEF as *mut PyFrame;
        force_frame(dummy);
        assert_eq!(
            SEEN.load(Ordering::SeqCst),
            dummy,
            "register_force_frame_hook + force_frame must dispatch the same pointer"
        );
    }

    #[test]
    fn force_vref_identity_dispatch_and_unset_panic() {
        let _lock = crate::stack_check::test_support::stack_and_jit_test_guard();
        use super::{FORCE_VREF_HOOK, force_vref, register_force_vref_hook};
        use majit_metainterp::virtualref::{ObjectHeader, VirtualRefInfo};

        struct RestorePrev(*mut ());
        impl Drop for RestorePrev {
            fn drop(&mut self) {
                FORCE_VREF_HOOK.store(self.0, Ordering::Release);
            }
        }
        let _restore = RestorePrev(FORCE_VREF_HOOK.swap(std::ptr::null_mut(), Ordering::AcqRel));

        static SEEN_VREF: AtomicPtr<PyFrame> = AtomicPtr::new(std::ptr::null_mut());
        unsafe extern "C" fn record_vref(frame: *mut PyFrame) -> *mut PyFrame {
            SEEN_VREF.store(frame, Ordering::SeqCst);
            0x0000_0000_FEED_FACE as *mut PyFrame
        }

        assert!(
            std::ptr::eq(force_vref(std::ptr::null_mut()), std::ptr::null_mut()),
            "null is identity and must not require a hook"
        );
        let mut header = ObjectHeader { typeptr: 1 };
        let dummy = &mut header as *mut ObjectHeader as *mut PyFrame;
        assert!(
            std::ptr::eq(force_vref(dummy), dummy),
            "a non-vref is identity and must not call the hook"
        );
        assert!(SEEN_VREF.load(Ordering::SeqCst).is_null());

        let info = VirtualRefInfo::new();
        let vref =
            info.virtual_ref_during_tracing(0x1000 as majit_gc::header::GCREF) as *mut PyFrame;
        assert!(
            unsafe {
                majit_metainterp::virtualref::ptr_is_virtual_ref(vref as majit_gc::header::GCREF)
            },
            "virtual_ref_during_tracing must produce a JitVirtualRef"
        );
        register_force_vref_hook(record_vref);
        let got = force_vref(vref);
        assert!(
            std::ptr::eq(SEEN_VREF.load(Ordering::SeqCst), vref),
            "force_vref must invoke the hook with the vref"
        );
        assert!(
            std::ptr::eq(got, 0x0000_0000_FEED_FACE as *mut PyFrame),
            "force_vref must return the hook's result"
        );

        FORCE_VREF_HOOK.store(std::ptr::null_mut(), Ordering::Release);
        let panicked = std::panic::catch_unwind(|| force_vref(vref)).is_err();
        assert!(
            panicked,
            "a vref with no hook must panic, not return the vref as a frame"
        );
    }
}
