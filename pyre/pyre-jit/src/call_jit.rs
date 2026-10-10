//! JIT-specific call infrastructure — force/bridge callbacks, callee
//! frame creation helpers, frame pool.
//!
//! Separated from pyre-interpreter/src/call.rs so pyre-interpreter stays JIT-free.

use pyre_interpreter::{locals_w, locals_w_mut};
use std::borrow::Cow;
use std::cell::UnsafeCell;
use std::sync::Once;

/// Whether `MAJIT_PROBE_LIVENESS` is set, cached at first access.
#[allow(dead_code)]
fn majit_probe_liveness_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("MAJIT_PROBE_LIVENESS").is_some());
    *ENABLED
}

/// Whether `PYRE_PROBE_BH_STARTUP=1` is set, cached at first access.
#[allow(dead_code)]
fn pyre_probe_bh_startup_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("PYRE_PROBE_BH_STARTUP").ok().as_deref() == Some("1")
    });
    *ENABLED
}

use pyre_interpreter::bytecode::{Instruction, OpArgState};
use pyre_interpreter::{
    PyResult, function_get_closure, function_get_defaults, function_get_globals_obj,
    function_get_name, is_function, register_jit_exc_clearer, register_jit_exc_raiser,
    register_jit_function_caller,
};
use pyre_object::intobject::w_int_get_value;
use pyre_object::intobject::w_int_new;
use pyre_object::pyobject::is_int;
use pyre_object::{PY_NULL, PyObjectRef};

use majit_ir::GcRef;
use pyre_interpreter::pyframe::PyFrame;
use pyre_jit_trace::trace::trace_bytecode;

// Force cache removed: CallAssemblerI + bridge handles recursion
// natively without memoization.

thread_local! {
    /// Callee PyFrame address whose CALL_ASSEMBLER bridge walk committed
    /// and adopted its end-of-walk state (raise_continue_running_normally
    /// analogue). The CA slow path calls the bridge hook and then the
    /// blackhole hook back-to-back; the blackhole consumes this to
    /// complete the callee from the adopted state instead of re-running
    /// the guard-state resume over already-applied effects.
    static CA_WALK_ADOPTED_FRAME: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    /// Callee PyFrame address whose CALL_ASSEMBLER bridge walk terminated
    /// with a kept finish-concrete stash (the walked callee ran to its
    /// return/uncaught raise concretely).  The CA bridge hook returns a
    /// bare bool and cannot carry the concrete result itself, so it sets
    /// this handshake and the back-to-back blackhole hook consumes the
    /// stash to complete the callee — the retrace-reaches-finishframe
    /// `DoneWithThisFrame`/`ExitFrameWithExceptionRef` the assembler
    /// caller catches (pyjitpl.py, jitexc.py) — instead of
    /// re-running the resumed region over already-applied effects.
    static CA_WALK_FINISHED_FRAME: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static CA_WALK_RESUME_FRAME: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static SELF_RECURSIVE_DISPATCH_CACHE: UnsafeCell<Option<(u64, Option<u64>)>> =
        const { UnsafeCell::new(None) };
}

struct FrameRoot {
    depth: usize,
}

impl FrameRoot {
    fn new(frame: &mut PyFrame) -> Self {
        Self {
            depth: majit_gc::shadow_stack::push(majit_ir::GcRef(frame as *mut PyFrame as usize)),
        }
    }

    fn frame(&mut self) -> &mut PyFrame {
        let frame = majit_gc::shadow_stack::get(self.depth).0 as *mut PyFrame;
        unsafe { &mut *frame }
    }
}

impl Drop for FrameRoot {
    fn drop(&mut self) {
        majit_gc::shadow_stack::try_pop_to(self.depth);
    }
}

// warmspot.py:449 portal result_type == REF: FINISH always boxes via
// wrapint, so the force/resume paths always receive a boxed Ref.
// FinishProtocol and normalize_direct_finish_result removed — they
// were dead code since result_type is always Type::Ref.

#[inline]
pub(crate) fn recursive_force_cache_safe(callable: PyObjectRef) -> bool {
    unsafe {
        if pyre_interpreter::is_builtin_code(
            pyre_interpreter::function_get_code(callable) as pyre_object::PyObjectRef
        ) {
            return false;
        }
        if !function_get_closure(callable).is_null() {
            return false;
        }
        let code =
            &*(pyre_interpreter::get_pycode(callable) as *const pyre_interpreter::CodeObject);
        let func_name = function_get_name(callable);
        let mut arg_state = OpArgState::default();
        let mut saw_self_reference = false;

        for code_unit in code.instructions.iter().copied() {
            let (instruction, op_arg) = arg_state.get(code_unit);
            match instruction {
                Instruction::LoadName { namei } => {
                    let idx = namei.get(op_arg) as usize;
                    if code.names[idx].as_str() != func_name {
                        return false;
                    }
                    saw_self_reference = true;
                }
                Instruction::LoadGlobal { namei } => {
                    let raw = namei.get(op_arg) as usize;
                    let name_idx = raw >> 1;
                    if code.names[name_idx].as_str() != func_name {
                        return false;
                    }
                    saw_self_reference = true;
                }
                Instruction::StoreName { .. }
                | Instruction::StoreGlobal { .. }
                | Instruction::LoadDeref { .. }
                | Instruction::StoreDeref { .. }
                | Instruction::DeleteDeref { .. }
                | Instruction::ImportName { .. }
                | Instruction::ImportFrom { .. }
                | Instruction::DeleteName { .. }
                | Instruction::DeleteGlobal { .. }
                | Instruction::MakeCell { .. }
                | Instruction::CopyFreeVars { .. } => return false,
                _ => {}
            }
        }

        if !saw_self_reference {
            return false;
        }
    }

    true
}

fn self_recursive_dispatch(green_key: u64) -> Option<u64> {
    SELF_RECURSIVE_DISPATCH_CACHE.with(|cell| unsafe {
        let slot = &mut *cell.get();
        if let Some((cached_key, token_num)) = *slot {
            if cached_key == green_key && token_num.is_some() {
                return token_num;
            }
        }

        let (driver, _) = crate::eval::driver_pair();
        let token_num = driver.get_loop_token(green_key).map(|token| token.number);
        if token_num.is_some() {
            *slot = Some((green_key, token_num));
        }
        token_num
    })
}

// Force cache implementation removed — CallAssemblerI + bridge
// handles recursive dispatch natively.

// ── JitFrame layout descriptors published to the backends ─────────

fn jitframe_layout_descrs() -> majit_gc::rewrite::JitFrameDescrs {
    use majit_backend::jitframe::*;
    majit_gc::rewrite::JitFrameDescrs {
        jitframe_tid: crate::jit::descr::JITFRAME_GC_TYPE_ID,
        jitframe_fixed_size: JITFRAME_FIXED_SIZE,
        jf_frame_info_ofs: JF_FRAME_INFO_OFS,
        jf_descr_ofs: JF_DESCR_OFS,
        jf_force_descr_ofs: JF_FORCE_DESCR_OFS,
        jf_savedata_ofs: JF_SAVEDATA_OFS,
        jf_guard_exc_ofs: JF_GUARD_EXC_OFS,
        jf_forward_ofs: JF_FORWARD_OFS,
        jf_frame_ofs: JF_FRAME_OFS,
        // RPython llmodel.py unpack_arraydescr + rewrite.py:680-684 consume  allow-line-citation
        // unpack_arraydescr()/lendescr offsets as jitframe-base-relative
        // addresses, not offsets relative to jf_frame itself.
        jf_frame_baseitemofs: FIRST_ITEM_OFFSET,
        jf_frame_lengthofs: JF_FRAME_OFS + LENGTHOFS,
        sign_size: SIGN_SIZE,
        jf_frame_itemsize: SIGN_SIZE,
    }
}

#[cfg(test)]
mod tests {
    use super::jitframe_layout_descrs;
    use majit_backend::jitframe::{FIRST_ITEM_OFFSET, JF_FRAME_OFS};
    use pyre_interpreter::pyframe::PyFrame;

    #[test]
    fn jitframe_layout_descrs_uses_frame_relative_offsets() {
        let descrs = jitframe_layout_descrs();
        assert_eq!(descrs.jf_frame_baseitemofs, FIRST_ITEM_OFFSET);
        assert_eq!(descrs.jf_frame_lengthofs, JF_FRAME_OFS);
    }

    #[test]
    fn blackhole_leave_rejects_a_red_outside_the_open_frame_scope() {
        std::thread::spawn(|| {
            let mut ec = pyre_interpreter::PyExecutionContext::default();
            pyre_interpreter::call::set_last_exec_ctx(&mut ec);
            let mut bh = majit_metainterp::blackhole::BlackholeInterpreter::default();
            // A malformed recovered red must never be dereferenced merely
            // because it is nonzero. There is no matching execution scope.
            bh.virtualizable_ptr = 2;
            super::leave_resumed_blackhole_frame(&bh, false);
            super::leave_resumed_blackhole_frame(&bh, true);
            assert!(ec.topframeref.is_null());
            pyre_interpreter::call::set_last_exec_ctx(std::ptr::null_mut());
        })
        .join()
        .unwrap();
    }

    /// Stack `PyFrame` whose Drop-owned pointers stay null.
    fn stack_pyframe() -> PyFrame {
        unsafe { std::mem::zeroed::<PyFrame>() }
    }

    #[test]
    fn blackhole_leave_marks_an_on_chain_parent_without_rewriting_top() {
        std::thread::spawn(|| {
            let mut ec = pyre_interpreter::PyExecutionContext::default();
            pyre_interpreter::call::set_last_exec_ctx(&mut ec);
            let mut parent = stack_pyframe();
            let mut child = stack_pyframe();
            child.f_backref = &mut parent;
            ec.topframeref = &mut child;
            let mut bh = majit_metainterp::blackhole::BlackholeInterpreter::default();
            bh.virtualizable_ptr = std::ptr::from_mut(&mut parent) as i64;
            super::leave_resumed_blackhole_frame(&bh, false);
            assert!(parent.frame_finished_execution());
            assert!(!child.frame_finished_execution());
            assert!(std::ptr::eq(ec.topframeref, std::ptr::from_mut(&mut child)));
            pyre_interpreter::call::set_last_exec_ctx(std::ptr::null_mut());
        })
        .join()
        .unwrap();
    }

    #[test]
    fn blackhole_leave_marks_a_parent_beyond_1024_frames() {
        std::thread::spawn(|| {
            let mut ec = pyre_interpreter::PyExecutionContext::default();
            // Allocate the complete chain before publishing any pointers.
            let mut frames: Vec<PyFrame> = (0..1026).map(|_| stack_pyframe()).collect();
            let base = frames.as_mut_ptr();
            for index in 1..frames.len() {
                // Keep accesses derived from the published allocation pointer;
                // do not reborrow the Vec's entire mutable slice between links.
                unsafe { (*base.add(index)).f_backref = base.add(index - 1) };
            }
            let top = unsafe { base.add(frames.len() - 1) };
            ec.topframeref = top;
            pyre_interpreter::call::set_last_exec_ctx(&mut ec);
            let mut bh = majit_metainterp::blackhole::BlackholeInterpreter::default();
            bh.virtualizable_ptr = base as i64;
            super::leave_resumed_blackhole_frame(&bh, false);
            // Clear the TLS anchor before an assertion can unwind the frame owner.
            pyre_interpreter::call::set_last_exec_ctx(std::ptr::null_mut());
            assert!(frames[0].frame_finished_execution());
            assert!(
                frames[1..]
                    .iter()
                    .all(|frame| !frame.frame_finished_execution())
            );
            assert!(std::ptr::eq(ec.topframeref, top));
        })
        .join()
        .unwrap();
    }

    #[test]
    fn blackhole_leave_closes_the_open_scope() {
        std::thread::spawn(|| {
            let mut ec = pyre_interpreter::PyExecutionContext::default();
            pyre_interpreter::call::set_last_exec_ctx(&mut ec);
            let mut parent = stack_pyframe();
            let mut child = stack_pyframe();
            child.f_backref = &mut parent;
            ec.topframeref = &mut child;
            let mut bh = majit_metainterp::blackhole::BlackholeInterpreter::default();
            bh.virtualizable_ptr = std::ptr::from_mut(&mut child) as i64;
            super::leave_resumed_blackhole_frame(&bh, false);
            assert!(child.frame_finished_execution());
            assert!(!parent.frame_finished_execution());
            assert!(std::ptr::eq(
                ec.topframeref,
                std::ptr::from_mut(&mut parent)
            ));
            pyre_interpreter::call::set_last_exec_ctx(std::ptr::null_mut());
        })
        .join()
        .unwrap();
    }
}

#[cfg(feature = "cranelift")]
pub fn jitframe_layout_info() -> majit_backend_cranelift::JitFrameLayoutInfo {
    majit_backend_cranelift::JitFrameLayoutInfo {
        jitframe_descrs: Some(jitframe_layout_descrs()),
    }
}

#[cfg(feature = "dynasm")]
pub fn jitframe_layout_info_dynasm() -> majit_backend_dynasm::JitFrameLayoutInfo {
    majit_backend_dynasm::JitFrameLayoutInfo {
        jitframe_descrs: Some(jitframe_layout_descrs()),
    }
}

// ── Callee frame allocation ───────────────────────────────────────

thread_local! {
    /// Callee frames the JIT created and has not finished running.
    /// Their only root between `jit_create_callee_frame_*` and
    /// `jit_drop_callee_frame` is this list: compiled code holds the
    /// frame in a register / jitframe slot, and the frame sits on no
    /// `CURRENT_FRAME` chain. Walked by `walk_jit_callee_frame_roots`.
    static LIVE_CALLEE_FRAMES: UnsafeCell<Vec<*mut PyFrame>> = const { UnsafeCell::new(Vec::new()) };

    static JIT_CALLEE_FRAME_ROOT_AREA: JitCalleeFrameRootArea = JitCalleeFrameRootArea {
        live_frames: LIVE_CALLEE_FRAMES.with(|cell| cell as *const _),
    };
}

struct JitCalleeFrameRootArea {
    live_frames: *const UnsafeCell<Vec<*mut PyFrame>>,
}

/// Create a callee frame for a JIT call, exactly as the interpreter
/// creates every other frame: `baseobjspace.py createframe` on a
/// `pyframe.py class PyFrame(W_Root)`, a normal GC object whose
/// lifetime is its reachability. Nothing recycles the block, so a frame
/// the program retains past the call — a traceback node, an `f_back`
/// chain, a `sys._getframe` result — stays valid for as long as Python
/// can reach it.
fn alloc_callee_frame(
    code: *const (),
    args: &[PyObjectRef],
    w_globals: PyObjectRef,
    execution_context: *const pyre_interpreter::PyExecutionContext,
) -> *mut PyFrame {
    let ptr = pyre_interpreter::pyframe::FrameBox::new(
        PyFrame::new_for_call_with_closure_and_globals_obj(
            code,
            args,
            w_globals,
            execution_context,
            PY_NULL,
            pyre_interpreter::pyframe::FrameLocalsArrayAllocation::OldGenGc,
        ),
    )
    .into_raw();
    LIVE_CALLEE_FRAMES.with(|cell| unsafe { &mut *cell.get() }.push(ptr));
    ptr
}

/// Drop the JIT's root on a callee frame that finished running. The
/// frame itself is not freed — `executioncontext.py leave` frees
/// nothing either; the collector reclaims it once nothing reaches it.
fn unroot_callee_frame(ptr: *mut PyFrame) {
    // This list is what makes a minor collection scan the frame's slots.
    // Once it stops, a young ref stored since the last minor — argument
    // boxing, for one — is reachable only through the old-gen locals
    // array's own items, which a minor does not walk. The frame outlives the call whenever the program
    // retained it (a traceback node, an `f_back` chain), so re-arm the
    // remembered set here, at the moment root-walking stops: the same barrier
    // `pyframe.py __init__` arms at creation, for the same reason.
    unsafe {
        pyre_interpreter::pyframe::remember_frame_locals_array((*ptr).locals_cells_stack_w);
    }
    if pyre_object::gc_hook::try_gc_owns_object(ptr as pyre_object::gc_hook::GCREF) {
        pyre_object::gc_hook::try_gc_write_barrier(ptr as pyre_object::gc_hook::GCREF);
    }
    LIVE_CALLEE_FRAMES.with(|cell| {
        let frames = unsafe { &mut *cell.get() };
        // Calls return in LIFO order, so the match is normally the last
        // entry.
        if let Some(pos) = frames.iter().rposition(|p| *p == ptr) {
            frames.swap_remove(pos);
        }
    });
}

/// Visit the GC-ref slots of one live callee frame: every
/// `locals_cells_stack_w` item plus the ref-bearing statics — the same
/// field set `walk_pyframe_roots` (pyre-interpreter::eval) visits for
/// interpreter frames on the `CURRENT_FRAME` chain.
///
/// # Safety
/// `frame` must point at a fully-initialized live `PyFrame`.
unsafe fn visit_callee_frame_roots(frame: *mut PyFrame, visitor: &mut dyn FnMut(&mut GcRef)) {
    let frame = unsafe { &mut *frame };
    for slot in locals_w_mut!(frame).as_mut_slice() {
        visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut GcRef) });
    }
    // Strong edge to the `f_generator_wref` WEAKREF box, the same slot
    // `walk_pyframe_roots` and `pyframe_object_custom_trace` forward; the
    // collector invalidates its `weakptr`.
    visitor(unsafe { &mut *(&mut frame.f_generator_wref as *mut PyObjectRef as *mut GcRef) });
    visitor(unsafe { &mut *(&mut frame.w_yielding_from as *mut PyObjectRef as *mut GcRef) });
    if !frame.debugdata.is_null() {
        let data = unsafe { &mut *frame.debugdata };
        visitor(unsafe { &mut *(&mut data.w_globals as *mut PyObjectRef as *mut GcRef) });
    }
}

/// Extra GC root walker for the callee frames a JIT call is running.
/// They sit on no `CURRENT_FRAME`/`f_backref` chain while compiled code
/// runs, so `walk_pyframe_roots` reaches neither the frames themselves
/// nor their locals: a major collection would sweep the block compiled
/// code is executing on, and a young object stored into a frame slot
/// (argument boxing, back-edge CALL_ASSEMBLER writeback) would be
/// invisible to a minor collection, leaving the slot pointing at
/// evacuated nursery memory. Registered via `register_extra_root_walker`,
/// mirroring the framework.py `root_walker.walk_roots` seam the collector
/// already uses for the other host-side root sources.
pub fn walk_jit_callee_frame_roots(visitor: &mut dyn FnMut(&mut GcRef)) {
    let data = capture_jit_callee_frame_root_area();
    unsafe { walk_jit_callee_frame_roots_area(data, visitor) };
}

pub fn capture_jit_callee_frame_root_area() -> *const () {
    JIT_CALLEE_FRAME_ROOT_AREA.with(|area| area as *const _ as *const ())
}

/// # Safety
/// `data` must come from [`capture_jit_callee_frame_root_area`], and the
/// owning thread must be quiesced.
pub unsafe fn walk_jit_callee_frame_roots_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut GcRef),
) {
    let area = unsafe { &*(data as *const JitCalleeFrameRootArea) };
    let live_frames = unsafe { &mut *(*area.live_frames).get() };
    for slot in live_frames.iter_mut() {
        // Mark the frame object itself — this list is its only root
        // until the call returns. The slot is handed over as a mutable
        // `GcRef` so a relocation would be written back, as upstream's
        // slot-address roots are (`shadowstack.py:43-46`); it stays
        // unwritten only because `FrameBox::new` allocates frames
        // non-moving.
        visitor(unsafe { &mut *(slot as *mut *mut PyFrame as *mut GcRef) });
        unsafe { visit_callee_frame_roots(*slot, visitor) };
    }
}

// ── JIT call callbacks ───────────────────────────────────────────

extern "C" fn jit_call_user_function_from_frame(
    frame_ptr: *const PyFrame,
    callable: PyObjectRef,
    args_ptr: *const PyObjectRef,
    nargs: i64,
) -> i64 {
    let frame = unsafe { &*frame_ptr };
    let args = unsafe { std::slice::from_raw_parts(args_ptr, nargs as usize) };
    // Depth tracked by ExecutionContext::py_recursion_depth (eval-loop entry).
    match pyre_interpreter::call::call_user_function(frame, callable, args) {
        Ok(result) => result as i64,
        Err(mut err) => {
            // llmodel.py _store_exception: write the exception
            // to the backend's `_exception_emulator` tp/val cells. The
            // matching GUARD_NO_EXCEPTION in the trace then reads
            // pos_exception()/pos_exc_value() and fails, and resume
            // data hands control to the except block. Do NOT stash the
            // PyError through a side channel — that would let the
            // interpreter-side eval loop surface it before the guard
            // machinery sees it, bypassing try/except.
            let exc_obj = err.to_exc_object();
            if exc_obj != pyre_object::PY_NULL {
                store_jit_exception(exc_obj as i64);
            }
            0 // garbage — GUARD_NO_EXCEPTION will fire
        }
    }
}

/// llmodel.py _store_exception: publish a materialised exception
/// object into the active backend's `_store_exception` cells so the
/// GuardNoException after the call detects it. One arm per backend; the wasm
/// backend keeps the pending exception in shared linear memory.
pub(crate) fn store_jit_exception(value: i64) {
    #[cfg(feature = "cranelift")]
    majit_backend_cranelift::jit_exc_raise(value);
    #[cfg(feature = "dynasm")]
    majit_backend_dynasm::jit_exc_raise(value);
    #[cfg(target_arch = "wasm32")]
    majit_backend_wasm::jit_exc_raise(value);
    #[cfg(not(any(feature = "cranelift", feature = "dynasm", target_arch = "wasm32")))]
    let _ = value;
}

/// Backend bridge for pyre-interpreter's residual-call helpers, which
/// cannot reference the backend crates directly. Publishes a materialised
/// exception object into the active backend's _store_exception cells so
/// the GuardNoException after the call detects it.
extern "C" fn jit_exc_raise_shim(value: i64) {
    store_jit_exception(value);
}

extern "C" fn jit_exc_clear_shim() {
    clear_residual_call_exception();
}

/// Publish a raise from a may-force residual helper to BOTH executors.
///
/// `bh_call_fn`/`bh_call_fn_N` is bound as the may-force CALL target
/// (`cpu.rs` `call_fn`, bound `CallFlavor::MayForce` by `codewriter.rs`'s
/// `register_helper_fn_pointers`), so the same helper runs under the blackhole
/// interpreter AND inside a compiled trace.  The blackhole reads the raise from
/// `BH_LAST_EXC_VALUE`; a compiled trace's `GUARD_NO_EXCEPTION` reads it from
/// the backend `_store_exception`
/// cells (`jit_exc_raise`).  Writing only `BH_LAST_EXC_VALUE` leaves
/// `GUARD_NO_EXCEPTION` reading a stale 0, so the guard wrongly passes and the
/// helper's NULL result flows to the consumer — keep both states in sync.
///
/// Both cells this writes — `BH_LAST_EXC_VALUE` and the backend `JIT_EXC_VALUE`
/// (via `store_jit_exception`) — are GC-rooted by `walk_parked_exception_roots`
/// and by the per-mutator `PyFrameRootArea`, so this writer needs no rooting of
/// its own.
pub(crate) fn publish_residual_call_exception(obj: PyObjectRef) {
    // Every residual raise enters the two exception channels here, so this is
    // the one place that can hold the line on what they may carry — and the
    // one place where the producer is still on the stack.  Both consumers
    // (`GUARD_NO_EXCEPTION` in compiled code, `_exit_frame_with_exception` in
    // the blackhole) end up reading the value's `ExcKind` tag through a match
    // with no wildcard arm; see `exit_frame_exception_ref`.
    if !obj.is_null()
        && unsafe { pyre_object::interp_exceptions::w_exception_kind_checked(obj) }.is_none()
    {
        reject_non_exception_channel_value(obj, "publish_residual_call_exception", String::new);
    }
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(obj as i64));
    store_jit_exception(obj as i64);
}

/// The caller's residual-call exception pair, taken out of its cells for the
/// span of a nested JIT re-entry by [`park_residual_call_exception`].
///
/// A cell is the only root its exception has (`walk_bh_last_exc_value` /
/// `walk_jit_exc_value`), so a value taken out of one is pinned on the shadow
/// stack for the span and read back relocated: an RPython local holding a GC
/// object is a shadow-stack root, a Rust local is not.
struct ParkedResidualException {
    /// `None` when both cells were empty — the ordinary case, since a
    /// residual call is reached with no exception in flight.
    scope: Option<pyre_object::gc_roots::RootScope>,
    save: usize,
    bh_pinned: bool,
    backend_pinned: bool,
}

/// Take the caller's `BH_LAST_EXC_VALUE` / backend `_store_exception` pair out
/// of their cells and leave both empty.
///
/// Upstream never needs this: the raise a residual call reports lives in
/// `metainterp.last_exc_value`, a field of the MetaInterp instance that owns
/// the call, and `llmodel.py _store_exception` is read back by the same
/// `bh_call_*` that armed it. Pyre keeps one cell per thread, so a nested
/// execution inside the call aliases the caller's.
fn park_residual_call_exception() -> ParkedResidualException {
    let bh = majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
    let backend = crate::eval::jit_exc_value_peek_backend();
    if bh == 0 && backend == 0 {
        return ParkedResidualException {
            scope: None,
            save: 0,
            bh_pinned: false,
            backend_pinned: false,
        };
    }
    let scope = pyre_object::gc_roots::push_roots();
    let save = pyre_object::gc_roots::shadow_stack_len();
    let bh_pinned = bh != 0;
    let backend_pinned = backend != 0;
    // Publish both cells before any normalize: `pin_root` queries after the
    // first write and a collection there would move the still-unrooted one.
    let mut parked = [pyre_object::PY_NULL; 2];
    let mut n = 0;
    if bh_pinned {
        parked[n] = bh as pyre_object::PyObjectRef;
        n += 1;
    }
    if backend_pinned {
        parked[n] = backend as pyre_object::PyObjectRef;
        n += 1;
    }
    if n > 0 {
        let base = scope.publish(&parked[..n]);
        scope.normalize(base, n);
    }
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    drain_backend_jit_exc();
    ParkedResidualException {
        scope: Some(scope),
        save,
        bh_pinned,
        backend_pinned,
    }
}

/// Restore what [`park_residual_call_exception`] took, discarding whatever the
/// nested execution left in the cells: a raise the nested callee already
/// handled is not this helper's to report, and the caller's
/// `GUARD_NO_EXCEPTION` would read it as a spurious pending exception — the
/// failure [`drain_backend_jit_exc`] names for the walker's snapshot side.
fn unpark_residual_call_exception(parked: ParkedResidualException) {
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    drain_backend_jit_exc();
    if parked.scope.is_none() {
        return;
    }
    let mut index = parked.save;
    if parked.bh_pinned {
        let bh = pyre_object::gc_roots::shadow_stack_get(index) as i64;
        index += 1;
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(bh));
    }
    if parked.backend_pinned {
        store_jit_exception(pyre_object::gc_roots::shadow_stack_get(index) as i64);
    }
}

/// [`park_residual_call_exception`] as an RAII scope, for a region with more
/// than one exit.
///
/// A whole tracing session needs this for the same reason a single residual
/// call does. `pyjitpl.py execute_raised` records a residual's raise into
/// `metainterp.last_exc_value` — an instance field, so upstream's dies with the
/// MetaInterp that owned the session. Pyre's stand-in is one cell per thread,
/// so a session that ends with a raise still recorded (a walk that executes
/// residuals concretely ends on the loop-exit `StopIteration`) leaves that
/// value visible to whatever runs next, and the next guard failure's blackhole
/// resume reads it as its own pending raise.
///
/// Declare the guard before anything else in the scope so it drops last, after
/// every inner [`pyre_object::gc_roots::RootScope`] has unwound and the shadow
/// stack is back to the depth [`park_residual_call_exception`] recorded.
pub(crate) struct ResidualExceptionScope {
    parked: Option<ParkedResidualException>,
    dbg: bool,
}

impl ResidualExceptionScope {
    pub(crate) fn park(dbg: bool) -> Self {
        Self {
            parked: Some(park_residual_call_exception()),
            dbg,
        }
    }
}

impl Drop for ResidualExceptionScope {
    fn drop(&mut self) {
        if self.dbg {
            let bh = majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
            let backend = crate::eval::jit_exc_value_peek_backend();
            if bh != 0 || backend != 0 {
                eprintln!("[residual-exc-scope] discarding bh={bh:#x} backend={backend:#x}");
            }
        }
        if let Some(parked) = self.parked.take() {
            unpark_residual_call_exception(parked);
        }
    }
}

/// Drain the backend `_store_exception` cells (`jit_exc_clear`) without
/// touching `BH_LAST_EXC_VALUE`.
///
/// `pyjitpl.py execute_raised` records a raise from a residual call
/// into `metainterp.last_exc_value` only — the backend `_store_exception`
/// cells (`llmodel.py` `_store_exception`) are written exclusively by
/// compiled / blackhole execution, never by metainterp tracing.  Pyre's
/// authoritative full-body walk shares the `bh_*` residual helpers with
/// the blackhole, so a raising helper publishes to BOTH cells
/// ([`publish_residual_call_exception`]).  The walker's `execute_raised`
/// analogue (`jitcode_dispatch.rs` residual-executor Err arm) keeps the
/// raise in `BH_LAST_EXC_VALUE` (= `last_exc_value`) but must drain the
/// backend cells so the recording leaves them pristine; otherwise an
/// aborted trace's snapshot-side raise leaks into the live frame's
/// re-run, where compiled `GUARD_NO_EXCEPTION` reads it as a spurious
/// pending exception.
pub(crate) fn drain_backend_jit_exc() {
    #[cfg(feature = "cranelift")]
    if majit_backend_cranelift::jit_exc_is_pending() {
        majit_backend_cranelift::jit_exc_clear();
    }
    #[cfg(feature = "dynasm")]
    if majit_backend_dynasm::jit_exc_is_pending() {
        majit_backend_dynasm::jit_exc_clear();
    }
    #[cfg(target_arch = "wasm32")]
    if majit_backend_wasm::jit_exc_value_peek() != 0 {
        majit_backend_wasm::jit_exc_clear();
    }
}

/// Clear the two pyre carriers for a residual-call exception after a compiled
/// frame has handled it and returned normally.
///
/// RPython keeps `MetaInterp.last_exc_value` on the metainterp that owns the
/// call and `handle_possible_exception()` clears it when control reaches the
/// handler.  Pyre's residual helpers publish to both this TLS stand-in and the
/// backend `_store_exception` cells, so a successful compiled portal exit must
/// leave both empty.  A caller exception that legitimately spans a nested JIT
/// entry is removed and restored by `park_residual_call_exception`, outside
/// this boundary.
pub(crate) fn clear_residual_call_exception() {
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    drain_backend_jit_exc();
}

/// `quasiimmut.py make_invalidation_function._invalidate_now` for the hidden
/// mutate field at `field_addr`: unlink the instance and flip every loop flag
/// it recorded.
///
/// This is the host half of `majit_metainterp::quasiimmut`'s
/// `do_force_quasi_immutable`.  Upstream reads a `GCREF` out of the field and
/// invalidates what it points at; pyre embeds the state in the owner, so
/// unlinking and sweeping is one call at the field's own address.
pub(crate) extern "C" fn force_quasi_immutable(field_addr: i64) {
    if field_addr == 0 {
        return;
    }
    // Safety: the only caller is `do_force_quasi_immutable`, which derives the
    // address from a live struct pointer plus the mutate field descr's offset;
    // that field is a `QuasiImmutField` embedded in the owner.
    unsafe {
        pyre_object::quasiimmut::sweep_quasi_immut_field(
            field_addr as *const pyre_object::quasiimmut::QuasiImmutField,
        )
    };
}

pub(crate) extern "C" fn record_caught_blackhole_traceback(
    exc_value: i64,
    frame_value: i64,
    jitcode_index: i64,
    opcode_position: i64,
) {
    let Ok(jitcode_index) = i32::try_from(jitcode_index) else {
        return;
    };
    let Ok(opcode_position) = i32::try_from(opcode_position) else {
        return;
    };
    let frame_ptr = frame_value as *mut PyFrame;
    if frame_ptr.is_null() || exc_value == 0 {
        return;
    }
    // Both operands are raw i64 copies of GC refs. The blackhole roots the
    // interpreter's `virtualizable_ptr` / `exception_last_value` slots, not
    // these argument copies; a wasm local is invisible to `walk_roots`, and
    // an inlined-callee `PyFrame` is a nursery object. Pin the pair before
    // any work that can collect (`decode_instruction_at`, traceback attach).
    let frame_anchor = unsafe { pyre_interpreter::eval::FrameAnchor::from_raw(frame_ptr) };
    let _roots = pyre_object::gc_roots::push_roots();
    let exc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exc_value as pyre_object::PyObjectRef);
    let forwards_existing = pyre_jit_trace::state::jitcode_pc_raise_keeps_existing_traceback(
        jitcode_index,
        opcode_position,
    );
    // An explicit reraise preserves even a cleared or shortened traceback.
    // An iterator exception can instead arrive from an inlined callee whose
    // traceback does not yet contain this frame.
    if forwards_existing {
        let py_pc = pyre_jit_trace::py_coord::exact_py_pc_for_jitcode_pc_public(
            jitcode_index,
            opcode_position,
        )
        .or_else(|| {
            pyre_jit_trace::py_coord::containing_py_pc_for_jitcode_pc_public(
                jitcode_index,
                opcode_position,
            )
        });
        let explicit_reraise = pyre_jit_trace::state::raw_code_for_jitcode_index(jitcode_index)
            .zip(py_pc)
            .and_then(|(code, pc)| unsafe {
                pyre_interpreter::decode_instruction_at(&*code, pc as usize)
            })
            .is_some_and(|(instruction, _)| {
                matches!(
                    instruction,
                    Instruction::RaiseVarargs { .. } | Instruction::Reraise { .. }
                )
            });
        if explicit_reraise {
            return;
        }
        let exc_value = pyre_object::gc_roots::shadow_stack_get(exc_slot) as i64;
        let frame_ptr = frame_anchor.live();
        let (owns_head, head_lasti) = unsafe {
            let head =
                pyre_object::interp_exceptions::w_exception_get_traceback(exc_value as PyObjectRef);
            let owns = !head.is_null()
                && pyre_interpreter::pytraceback::is_pytraceback(head)
                && pyre_interpreter::pytraceback::w_pytraceback_get_frame(head) == frame_ptr;
            let lasti =
                owns.then(|| pyre_interpreter::pytraceback::w_pytraceback_get_lasti(head) / 2);
            (owns, lasti)
        };
        if owns_head {
            // A forwarding raise keeps the existing node, but PyPy's frame
            // still carries the instruction recorded by that node.  Repair a
            // resume/escape rollback that ran between the original attach and
            // this duplicate recorder before returning early.
            let last_instr = head_lasti.expect("owned traceback head has a lasti");
            unsafe { (*frame_ptr).last_instr = last_instr as isize };
            pyre_jit_trace::jitcode_dispatch::finalize_published_last_instr(
                frame_ptr as usize,
                last_instr as isize,
            );
            return;
        }
    }
    let resolved = pyre_jit_trace::py_coord::containing_py_pc_for_jitcode_pc_public(
        jitcode_index,
        opcode_position,
    );
    let exact =
        pyre_jit_trace::py_coord::exact_py_pc_for_jitcode_pc_public(jitcode_index, opcode_position);
    let exc_value = pyre_object::gc_roots::shadow_stack_get(exc_slot) as i64;
    let frame_ptr = frame_anchor.live();
    let last_instruction = exact
        .or(resolved)
        .map_or(unsafe { (*frame_ptr).last_instr as i64 }, i64::from);
    // `dispatch_bytecode` (pyopcode.py) stamps `self.last_instr` before both
    // `handle_operation_error` and its traceback attach.  Do the same before
    // the duplicate-node check below: a first recorder can be followed by an
    // escape rollback that restores only the old resume coordinate, and the
    // second recorder must still repair the frame even when it need not add a
    // second node.
    unsafe {
        (*frame_ptr).last_instr = last_instruction as isize;
        pyre_jit_trace::jitcode_dispatch::finalize_published_last_instr(
            frame_ptr as usize,
            last_instruction as isize,
        );
    }
    // One catch can be reached by two recorders: this hook, emitted into the
    // loop trace by the in-trace handler-entry arm, and the IR-virtual node the
    // bridge handler-entry arm builds when the exception edge deopts into a
    // bridge.  Both then run for the same delivery and the frame gets two
    // adjacent nodes, so the handler reads its own name twice.  Upstream never
    // has the pair — `pyopcode.py handle_operation_error` is the single attach
    // point — so drop the second: `record_application_traceback` prepends, and
    // this exact node already exists exactly when the chain HEAD carries this
    // frame at this instruction.  A genuine repeat (the same frame raising
    // again at the same offset) always attaches to a fresh exception, and a
    // bare reraise, which does legitimately re-record one frame, is skipped
    // above.
    unsafe {
        let head =
            pyre_object::interp_exceptions::w_exception_get_traceback(exc_value as PyObjectRef);
        if !head.is_null()
            && pyre_interpreter::pytraceback::is_pytraceback(head)
            && pyre_interpreter::pytraceback::w_pytraceback_get_frame(head) == frame_ptr
            && pyre_interpreter::pytraceback::w_pytraceback_get_lasti(head) == last_instruction * 2
        {
            return;
        }
    }
    unsafe {
        pyre_interpreter::pytraceback::record_application_traceback(
            exc_value as PyObjectRef,
            frame_ptr,
            last_instruction,
        );
    }
}

/// `pypy/interpreter/error.py record_context` for an exception whose
/// handler is part of the trace, split so the trace records the store itself.
///
/// The implicit `__context__` is derived by `handle_exception_with_context`,
/// and a raise the walk routes straight to a handler in the same trace never
/// surfaces an error the interpreter dispatches — the same gap
/// [`record_inline_traceback_for_recording`] closes for the traceback node.
/// Without this, the compiled iterations of a loop raising under a live handler
/// leave `__context__` null while the interpreted ones set it.
///
/// Chains through `chain_context`, so the active exception is selected, a
/// non-exception or self-referential pair is refused, and a chain that would
/// close a cycle is broken by nulling the offending link — all exactly as the
/// interpreter does it — and then answers the slot's resulting value.  The
/// trace's own `SetfieldGc` therefore stores what is already there; `0` means
/// the slot stays null.
///
/// The cycle break writes the `__context__` of an exception other than
/// `w_exc`, which the emitted store does not describe, so the call is declared
/// `cannot_raise_effect_info()` — a heap mutator that cannot raise — rather
/// than as having no heap effect.
pub(crate) extern "C" fn resolve_exception_context(exc_value: i64) -> i64 {
    if exc_value == 0 {
        return 0;
    }
    let w_exc = exc_value as PyObjectRef;
    if !unsafe { pyre_object::is_exception(w_exc) } {
        return 0;
    }
    let existing = unsafe { pyre_object::interp_exceptions::w_exception_get_context(w_exc) };
    if !existing.is_null() {
        return existing as i64;
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let w_exc = pyre_object::gc_roots::pin_root(w_exc);
    let active = pyre_interpreter::eval::get_sys_exception();
    pyre_interpreter::error::chain_context(w_exc, active);
    unsafe { pyre_object::interp_exceptions::w_exception_get_context(w_exc) as i64 }
}

pub(crate) extern "C" fn record_inline_traceback_for_recording(
    exc_value: i64,
    w_code_value: i64,
    w_globals_value: i64,
    jitcode_index: i64,
    opcode_position: i64,
) {
    let Ok(jitcode_index) = i32::try_from(jitcode_index) else {
        return;
    };
    let Ok(opcode_position) = i32::try_from(opcode_position) else {
        return;
    };
    // Inline frames follow the same rule as concrete blackhole frames: a raise
    // that forwards an already-propagating exception preserves the traceback
    // attached by the original raising instruction.
    if pyre_jit_trace::state::jitcode_pc_raise_keeps_existing_traceback(
        jitcode_index,
        opcode_position,
    ) {
        return;
    }
    if exc_value == 0 || w_code_value == 0 {
        return;
    }
    let Some(last_instruction) = pyre_jit_trace::py_coord::containing_py_pc_for_jitcode_pc_public(
        jitcode_index,
        opcode_position,
    )
    .map(i64::from) else {
        return;
    };
    let w_exc = exc_value as PyObjectRef;
    let w_code = w_code_value as PyObjectRef;
    let w_globals = w_globals_value as PyObjectRef;
    let _roots = pyre_object::gc_roots::push_roots();
    let base = _roots.pin_roots(&[w_exc, w_code, w_globals]);
    let w_exc = _roots.get(base);
    let w_code = _roots.get(base + 1);
    let w_globals = _roots.get(base + 2);
    // `record_application_traceback` requires the traceback's own frame
    // identity. The recording walker cannot force the optimizer's virtual
    // locals, so materialize a traceback-only frame from the promoted callee
    // metadata instead of substituting the caller's frame.
    let Ok(mut frame) = pyre_interpreter::createframe_obj(
        w_code as *const (),
        w_globals,
        pyre_interpreter::call::getexecutioncontext(),
        None,
    ) else {
        return;
    };
    frame.last_instr = last_instruction as isize;
    // `pyopcode.py handle_operation_error` marks a propagating frame
    // finished before it leaves that frame.  This object materializes that
    // already-unwound inline callee, so preserve the same lifecycle state;
    // traceback.clear_frames() must not mistake it for a live activation.
    frame.set_frame_finished_execution(true);
    let frame_ptr = frame.into_raw();
    let _ = pyre_object::gc_roots::pin_root(frame_ptr as PyObjectRef);
    unsafe {
        pyre_interpreter::pytraceback::record_application_traceback(
            w_exc,
            frame_ptr,
            last_instruction,
        );
    }
}

/// `pytraceback.py record_application_traceback` for a frame the resume
/// crossed and discarded — an inlined callee the exception-edge bridge unwound
/// clear out of on its way to the live frame's own handler.
///
/// `pyopcode.py:148` runs BEFORE the `:152` exception-table lookup, so a frame
/// that only propagates contributes a node exactly like one that catches.  The
/// levels named here never reach a recorder that could speak for them: the
/// route exists because the unwind discards them, and by the time the walk
/// starts they are gone.
///
/// The node's frame is fabricated from the code object, as
/// [`record_inline_traceback_for_recording`] does and for the same reason —
/// the level's own `PyFrame` stayed virtual in the compiled trace and the
/// resume never materialized it, so there is no other object to name.  Unlike
/// that case there is also no seeded frame it could be confused with: nothing
/// else in this process can reach the discarded level either.
///
/// `py_pc` is the level's resume coordinate, which is `next_instr`-style — the
/// same coordinate `exc_table_offset` converts with `saturating_sub(1)` and
/// `set_last_instr_from_next_instr` converts for the live frame.  Everything
/// below wants the instruction that RAN: the opcode decode that answers the
/// bare-reraise question, `frame.last_instr`, and `record_application_traceback`'s
/// `tb_lasti`.  Convert once, up front.
pub(crate) extern "C" fn record_discarded_level_traceback(
    exc_value: i64,
    w_code_value: i64,
    py_pc: i64,
) {
    if exc_value == 0 || w_code_value == 0 || py_pc < 0 {
        return;
    }
    let last_instruction = py_pc.saturating_sub(1);
    let w_code = w_code_value as PyObjectRef;
    let raw_code =
        unsafe { pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject };
    if raw_code.is_null() {
        return;
    }
    // Same rule the other two recorders follow: a raise that forwards an
    // already-propagating exception keeps the traceback it arrived with.
    if pyre_jit_trace::state::raise_at_py_pc_keeps_existing_traceback(
        unsafe { &*raw_code },
        last_instruction as usize,
    ) {
        return;
    }
    let w_globals = unsafe { pyre_interpreter::w_code_get_w_globals(w_code) };
    let w_exc = exc_value as PyObjectRef;
    let _roots = pyre_object::gc_roots::push_roots();
    let base = _roots.pin_roots(&[w_exc, w_code, w_globals]);
    let w_exc = _roots.get(base);
    let w_code = _roots.get(base + 1);
    let w_globals = _roots.get(base + 2);
    let Ok(mut frame) = pyre_interpreter::createframe_obj(
        w_code as *const (),
        w_globals,
        pyre_interpreter::call::getexecutioncontext(),
        None,
    ) else {
        return;
    };
    frame.last_instr = last_instruction as isize;
    // This discarded level has already propagated out during the inlined
    // unwind, exactly the `pyopcode.py:184` finished-frame transition.
    frame.set_frame_finished_execution(true);
    let frame_ptr = frame.into_raw();
    let _ = pyre_object::gc_roots::pin_root(frame_ptr as PyObjectRef);
    unsafe {
        pyre_interpreter::pytraceback::record_application_traceback(
            w_exc,
            frame_ptr,
            last_instruction,
        );
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_force_callee_frame(frame_ptr: i64) -> i64 {
    // `assembler_call_helper` (warmspot.py) resumes the callee
    // frame the rewritten CALL_ASSEMBLER passed as arg 0.
    run_frame_through_portal(frame_ptr, PortalEntry::Resume)
}

#[derive(Clone, Copy)]
enum PortalEntry {
    /// `ll_portal_runner`: compiled code owns the EC vref bracket and this
    /// door supplies the activation's hook/recursion bracket.
    TracedActivation,
    /// `portal_ptr`: continue the body of an activation already in progress.
    Resume,
}

/// warmspot.py `ll_portal_runner` core — run the frame the JIT handed
/// in through the portal (`maybe_compile_and_run` + interpreter main loop;
/// ContinueRunningNormally re-enters the JIT via the portal,
/// warmspot.py).
///
/// The pointer is the callee `PyFrame` the trace itself built —
/// `emit_new_pyframe_inline_with_params` emits `NewWithVtable` +
/// `SetfieldGc` for `pycode` / `w_globals` / `execution_context` and the
/// locals array, and the GC rewriter stores that pointer into the callee
/// jitframe's first slot, which is what every force site reads back. It is
/// the `frame` red of `jd.portal_calldescr`, and `ll_portal_runner` forwards
/// its reds unchanged (`warmspot.py` `portal_ptr(*args)`).
///
/// Rebuilding a frame here instead would drop the callee's arguments (its
/// locals array) and its `last_instr`, and would run the interpreter on a
/// frame that dies with the C stack.
fn run_frame_through_portal(frame_ptr: i64, entry: PortalEntry) -> i64 {
    // `jit_drop_callee_frame` tolerates a tagged word on the same slot because
    // it only unroots; there is nothing sensible to run here for one. The
    // portal's result is a Ref whose NULL spelling means "exception stored",
    // so returning NULL for a non-frame would manufacture an exception-less
    // error rather than deopt cleanly.
    debug_assert!(
        frame_ptr != 0 && frame_ptr & 1 == 0,
        "CALL_ASSEMBLER arg 0 must be the callee PyFrame"
    );
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    // `warmspot.py` has two distinct functions: `ll_portal_runner` begins the
    // portal call and `portal_ptr` resumes its body from CRN/force handoff.
    // Keep that decision at the caller that knows which function it is
    // invoking; CURRENT_FRAME is an extra pyre TLS root, not activation
    // identity, and is neither complete nor stable enough to choose the door.
    let outcome = match entry {
        PortalEntry::TracedActivation => crate::eval::portal_traced_activation_result(frame),
        PortalEntry::Resume => crate::eval::portal_body_result(frame),
    };
    let result = match outcome {
        Ok(r) => r,
        Err(mut err) => {
            // `ll_portal_runner` raises into RPython's single ll exception
            // state, and every one of its executors reads that one state:
            // compiled `CALL_ASSEMBLER` through `pos_exception()` /
            // `pos_exc_value()` (`llmodel.py`), `bhimpl_recursive_call_*`
            // through the propagated RPython exception (`blackhole.py`), and
            // the trace-time execution `do_recursive_call` performs through
            // `metainterp.execute_raised` (`executor.py _do_call`).  Pyre spells
            // that one state as two carriers, so this raise owes both:
            // `execute_varargs_call` (majit-metainterp `executor.rs`) reads
            // only `BH_LAST_EXC_VALUE`, and with the backend cells alone it
            // sees no raise, records `GUARD_NO_EXCEPTION` over it, writes the
            // NULL result into the destination slot and leaves the backend
            // cells armed for an unrelated guard.
            publish_residual_call_exception(err.to_exc_object());
            pyre_object::PY_NULL
        }
    };

    // warmspot.py:449 result_type=REF: always boxed Ref
    result as i64
}

/// warmspot.py `ll_portal_runner` — the raw-address portal entry whose
/// address is stored in `jd.portal_runner_adr` (warmspot.py) and
/// called by the synthetic callback loop that `compile_tmp_callback`
/// (compile.py:1125-1132) builds for a not-yet-compiled recursive callee.
///
/// The call uses the `jd.portal_calldescr` ABI —
/// `JitDriverStaticData::build_portal_calldescr` lays the args out in `vars`
/// declaration order: greens `[next_instr, is_being_profiled, pycode]` then
/// reds `[frame, ec]`. The red frame owns activation identity, while the green
/// `next_instr` remains the control coordinate forwarded to the portal.
#[majit_macros::jit_may_force]
pub extern "C" fn ll_portal_runner_shim(
    next_instr: i64,
    _is_being_profiled: i64,
    _pycode: i64,
    frame_ptr: i64,
    _ec: i64,
) -> i64 {
    // The callback loop `compile_tmp_callback` builds (compile.py)
    // passes the callee `frame` red that `emit_new_pyframe_inline_with_params`
    // constructed — a proper `PyFrame` with `locals_cells_stack_w` already
    // populated (NewWithVtable + SetfieldGc). PyPy `ll_portal_runner` forwards
    // every portal argument to `portal_ptr(*args)`. Preserve that exact
    // coordinate: a tracing MIFrame may be ahead of the heap frame's stale
    // `last_instr`, and using the latter resumes the right activation at the
    // wrong bytecode.
    if frame_ptr != 0 {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        if let Some(pc) = crate::eval::green_pc_position(next_instr) {
            frame.set_last_instr_from_next_instr(pc);
        }
    }
    run_frame_through_portal(frame_ptr, PortalEntry::TracedActivation)
}

/// warmspot.py — assembler_call_helper.
///
/// Called when CALL_ASSEMBLER guard-fails (not a finish exit).
/// Receives a JitFrame pointer, reconstructs a PyFrame from the
/// jf_frame slots, and resumes execution in the interpreter.
///
/// This is the JitFrame-aware counterpart to `jit_force_callee_frame`
/// (which operates on PyFrame directly). When the GC rewriter wires
/// nursery JitFrame allocation, this function replaces the force path.
#[allow(dead_code)]
#[majit_macros::jit_may_force]
pub extern "C" fn assembler_call_helper(jitframe_ptr: i64, _virtualizable_ref: i64) -> i64 {
    use majit_backend::jitframe::JitFrame;

    let jf = jitframe_ptr as *mut JitFrame;

    // warmspot.py:1022 — fail_descr = cpu.get_latest_descr(deadframe)
    // compile.py handle_fail: dispatches on fail_descr to either
    // _trace_and_compile_from_bridge or resume_in_blackhole.
    // Bridge compilation is driven by must_compile() in jitdriver.
    // This force path always resumes in the interpreter (blackhole).
    let _descr = unsafe { majit_backend::llmodel::get_latest_descr(jf) };

    // For now, reconstruct a PyFrame and run it in the interpreter.
    // This is the "blackhole" path — RPython resume.py parity.
    //
    // Step 1: read the raw int arg from jf_frame[0]
    let raw_local0 = unsafe { majit_backend::llmodel::get_int_value_direct(jf, 0) } as i64;

    // Step 2: create a PyFrame and run it
    // The caller_frame is in inputs[0] which was the JitFrame's first
    // virtualizable input. For now, fall back to the existing force path.
    jit_force_self_recursive_call_raw_1(jitframe_ptr, raw_local0)
}

/// RPython: FieldDescr.offset is resolved at rtyper time. In pyre, Rust struct
/// layout determines field offsets. This resolver maps (owner_type, field_name)
/// to byte offsets for BhDescr::Field resolution in the blackhole.
#[allow(dead_code)]
fn resolve_field_offset(owner: &str, field_name: &str) -> usize {
    use pyre_interpreter::pyframe::PyFrame;
    match field_name {
        "code" | "pycode" => std::mem::offset_of!(PyFrame, pycode),
        "locals_cells_stack_w" => std::mem::offset_of!(PyFrame, locals_cells_stack_w),
        "valuestackdepth" => std::mem::offset_of!(PyFrame, valuestackdepth),
        "next_instr" | "f_lasti" | "last_instr" => std::mem::offset_of!(PyFrame, last_instr),
        "vable_token" => std::mem::offset_of!(PyFrame, vable_token),
        // #171 codewriter descr-bridge (blackhole side): the dotted nested
        // `int_items.{len,block}` leaves `_handle_list_call`
        // emits resolve to offset 0 in the codewriter assembler because
        // `W_ListObject` is a runtime Rust type absent from its struct
        // layouts. Resolve them to the real `IntArray` offsets so a
        // blackholed codewriter list body addresses the typed storage, not
        // the list header (mirrors the JIT-side bridge in
        // `pyre-jit-trace/descr.rs make_descr_from_bh`).
        "int_items.len" => {
            std::mem::offset_of!(pyre_object::W_ListObject, int_items)
                + pyre_object::INT_ARRAY_LEN_OFFSET
        }
        "int_items.block" => {
            std::mem::offset_of!(pyre_object::W_ListObject, int_items)
                + pyre_object::INT_ARRAY_BLOCK_OFFSET
        }
        "float_items.len" => {
            std::mem::offset_of!(pyre_object::W_ListObject, float_items)
                + pyre_object::FLOAT_ARRAY_LEN_OFFSET
        }
        "float_items.block" => {
            std::mem::offset_of!(pyre_object::W_ListObject, float_items)
                + pyre_object::FLOAT_ARRAY_BLOCK_OFFSET
        }
        _ => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][blackhole] WARNING: unresolved field offset owner={:?} name={:?}",
                    owner, field_name
                );
            }
            0
        }
    }
}

/// C-ABI portal runner matching the `mainjitcode_calldescr` arg layout
/// (`"iirrr"`: `next_instr:i, is_being_profiled:i, pycode:r, frame:r, ec:r`).
///
/// `warmspot.py ll_portal_runner(*args)`, in the shape its callers dispatch.
/// The five scalars are the `portalfunc_ARGS` tuple `warmspot.py`
/// `handle_jitexception` rebuilds, already extracted from the merged banks
/// `blackhole.py bhimpl_recursive_call_r` passes as `greens_* + reds_*`:
/// greens `[next_instr:i, is_being_profiled:i, pycode:r]` then reds
/// `[frame:r, ec:r]`.  The merged-bank form that does the extraction itself is
/// `eval.rs pyre_portal_runner`, which serves the `ContinueRunningNormally`
/// leg rather than a call.
///
/// This is the function `get_portal_runner` hands to `bh_call_r` for
/// `bhimpl_recursive_call_*` (blackhole.py), and the one
/// `CallControl::get_jitcode_calldescr` publishes. `bh_call_r` invokes it
/// through `collect_call_args` + `bh_call_i_dispatch`, i.e. as a raw C function
/// whose five arguments arrive in integer registers per the calldescr — so the
/// runner must take those five scalars directly. (A Rust `fn(&[i64], …)` would
/// receive slice fat pointers there instead, corrupting every argument, which
/// is why no `fn(&[i64], …)` spelling of it may be published beside a descr.)
pub extern "C" fn bh_portal_runner_c(
    next_instr: i64,
    _is_being_profiled: i64,
    pycode: i64,
    frame_ptr: i64,
    ec: i64,
) -> i64 {
    let pycode = pycode as PyObjectRef;
    let frame_ptr = frame_ptr as *mut PyFrame;
    let ec = ec as *const pyre_interpreter::PyExecutionContext;

    if frame_ptr.is_null() {
        return pyre_object::PY_NULL as i64;
    }
    let frame = unsafe { &mut *frame_ptr };
    // warmspot.py:976 calls `portal_ptr(*args)`; it does not copy the green
    // `pycode` back onto the red frame. The frame owns its pycode while `ec`
    // is the independent second red for this activation. In particular,
    // a stale CRN green must never collapse this frame onto another code
    // object's identity.
    if majit_metainterp::majit_log_enabled()
        && !pycode.is_null()
        && frame.pycode != pycode as *const ()
    {
        eprintln!(
            "[blackhole-resume] CRN/frame identity mismatch: green_pycode={pycode:p} frame_pycode={:p} red_ec={ec:p}",
            frame.pycode,
        );
    }
    if let Some(pc) = crate::eval::green_pc_position(next_instr) {
        frame.set_last_instr_from_next_instr(pc);
        // The blackhole wrote the failing guard's recorded operand depth into the
        // frame; resuming at the merge-point `next_instr` (a different pc) would
        // carry that over-count and overflow the frame at its peak stack use.
        // Re-derive the depth from the resume pc — the same correction the
        // CALL_ASSEMBLER CRN arm applies to the same kind of green `next_instr`.
        crate::eval::correct_resume_vsd(frame, pc);
    }
    // The bracket is owed, but not for the reason a reading of
    // `bhimpl_recursive_call_r` suggests.  pyre's codewriter emits no
    // `recursive_call`, so the live door here is `bhimpl_jit_merge_point`'s
    // recursive-portal arm (blackhole.py), taken when `nextblackholeinterp`
    // is not None — a frame that already has a caller in the resumed chain,
    // i.e. an inlined callee, mid-body at its own loop header.  That IS a
    // resume, and upstream reaches it through `portal_ptr` = `dispatch`, one
    // level below `execute_frame`.  What makes it an activation for pyre is
    // where the frame came from: `emit_new_pyframe_inline_with_params` built
    // it inside compiled code, and `walker_ec_enter` records only `enter`'s
    // frame-chain half, never `call_trace` — so this is the frame's first and
    // only bracket, not a second one.  The bottom level, which is the frame
    // the interpreter itself bracketed, never arrives here: it raises
    // `ContinueRunningNormally` and takes the unbracketed entry instead.
    //
    // `ec` is a red arg and the frame no longer carries one, so publish it for
    // the duration of the call: `execute_frame` reads its execution context
    // through `space.getexecutioncontext()`, which is what the bracket below
    // will consult.
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    if !ec.is_null() {
        pyre_interpreter::call::set_last_exec_ctx(ec);
    }
    let result = crate::eval::portal_activation_result(frame);
    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
    match result {
        Ok(result) => result as i64,
        Err(mut err) => {
            // The other portal door, and it owes the same two carriers for the
            // same reason: `warmspot.py handle_jitexception` raises for real,
            // so the state a portal raise arms is caller-blind.  This address
            // serves both `bhimpl_recursive_call_*` (`blackhole.py`), which
            // reads `BH_LAST_EXC_VALUE`, and — through `get_jitcode_calldescr`
            // — the `fnaddr` every JitCode carries, which upstream is a call
            // target for `bhimpl_inline_call_*` (`blackhole.py`) and for a
            // trace's own residual call, and those read the backend cells.
            // Publishing both is also what the catch side already assumes:
            // `route_to_catch` drains `BH_LAST_EXC_VALUE` and the backend cells
            // together.
            publish_residual_call_exception(err.to_exc_object());
            pyre_object::PY_NULL as i64
        }
    }
}

/// jitexc.py JitException hierarchy — structural parity with RPython.
///
/// `_run_forever` must exit via exactly one of these variants.
#[allow(dead_code)] // not all variants are constructed yet
pub enum JitException {
    /// jitexc.py ContinueRunningNormally(gi, gr, gf, ri, rr, rf):
    /// blackhole reached the merge point → restart the portal. The six
    /// vectors carry the green/red int/ref/float live-in arguments used
    /// by `maybe_compile_and_run` to resume execution.
    ContinueRunningNormally {
        green_int: Vec<i64>,
        green_ref: Vec<PyObjectRef>,
        green_float: Vec<f64>,
        red_int: Vec<i64>,
        red_ref: Vec<PyObjectRef>,
        red_float: Vec<f64>,
    },
    /// jitexc.py DoneWithThisFrameVoid.
    DoneWithThisFrameVoid,
    /// jitexc.py DoneWithThisFrameInt(result).
    DoneWithThisFrameInt(i64),
    /// jitexc.py DoneWithThisFrameRef(result).
    DoneWithThisFrameRef(PyObjectRef),
    /// jitexc.py DoneWithThisFrameFloat(result).
    DoneWithThisFrameFloat(f64),
    /// jitexc.py ExitFrameWithExceptionRef(value): blackhole ran out
    /// of caller frames while propagating a Python exception.
    ExitFrameWithExceptionRef(pyre_interpreter::error::PyError),
}

/// RPython jitexc.py parity: typed exception channels for blackhole result.
///
/// Each variant matches an RPython JitException subclass 1:1.
/// The value is carried in its native type — no boxing into PyObjectRef.
pub enum BlackholeResult {
    /// jitexc.py ContinueRunningNormally(gi, gr, gf, ri, rr, rf)
    ContinueRunningNormally {
        green_int: Vec<i64>,
        green_ref: Vec<PyObjectRef>,
        green_float: Vec<f64>,
        red_int: Vec<i64>,
        red_ref: Vec<PyObjectRef>,
        red_float: Vec<f64>,
    },
    /// jitexc.py DoneWithThisFrameVoid
    DoneWithThisFrameVoid,
    /// jitexc.py DoneWithThisFrameInt(result: Signed)
    DoneWithThisFrameInt(i64),
    /// jitexc.py DoneWithThisFrameRef(result: GCREF)
    DoneWithThisFrameRef(PyObjectRef),
    /// jitexc.py DoneWithThisFrameFloat(result: FLOATSTORAGE)
    DoneWithThisFrameFloat(f64),
    /// jitexc.py ExitFrameWithExceptionRef(value: GCREF)
    ExitFrameWithExceptionRef(pyre_interpreter::error::PyError),
    /// pyre-only: the blackhole stopped on a pyre `abort_permanent` marker.
    /// This is a designed bail whose resume coordinate the codewriter already
    /// stamped into the frame; the frame is resumable as-is and the compiled
    /// loop must not be invalidated.
    BailToInterpreter,
}

impl From<JitException> for BlackholeResult {
    /// jitexc.py parity: each JitException variant maps to BlackholeResult
    /// with its payload preserved — ContinueRunningNormally carries the
    /// 6 green/red lists through.
    fn from(exc: JitException) -> Self {
        match exc {
            JitException::ContinueRunningNormally {
                green_int,
                green_ref,
                green_float,
                red_int,
                red_ref,
                red_float,
            } => BlackholeResult::ContinueRunningNormally {
                green_int,
                green_ref,
                green_float,
                red_int,
                red_ref,
                red_float,
            },
            JitException::DoneWithThisFrameVoid => BlackholeResult::DoneWithThisFrameVoid,
            JitException::DoneWithThisFrameInt(v) => BlackholeResult::DoneWithThisFrameInt(v),
            JitException::DoneWithThisFrameRef(r) => BlackholeResult::DoneWithThisFrameRef(r),
            JitException::DoneWithThisFrameFloat(f) => BlackholeResult::DoneWithThisFrameFloat(f),
            JitException::ExitFrameWithExceptionRef(err) => {
                BlackholeResult::ExitFrameWithExceptionRef(err)
            }
        }
    }
}

impl BlackholeResult {
    /// warmspot.py:985-1005: convert typed DoneWithThisFrame* result to PyResult.
    ///
    /// This is the warmspot boundary where the typed JIT exception value
    /// is converted back into a Python-level result. RPython's warmspot
    /// does this implicitly via result_kind dispatch; pyre boxes here.
    pub fn to_pyresult(&self) -> Option<PyResult> {
        match self {
            BlackholeResult::DoneWithThisFrameVoid => Some(Ok(pyre_object::PY_NULL)),
            BlackholeResult::DoneWithThisFrameInt(v) => {
                Some(Ok(pyre_object::intobject::w_int_new(*v) as PyObjectRef))
            }
            BlackholeResult::DoneWithThisFrameRef(r) => Some(Ok(*r)),
            BlackholeResult::DoneWithThisFrameFloat(f) => {
                Some(Ok(pyre_object::floatobject::w_float_new(*f) as PyObjectRef))
            }
            // warmspot.py handle_jitexception: raise the exception. `Clone` aliases the
            // `OperationError` handle. The source stays in `bh_result` and is
            // dropped without a later mutation, so the returned alias is the
            // only carrier that unwinds. Do not propagate both.
            BlackholeResult::ExitFrameWithExceptionRef(err) => Some(Err(err.clone())),
            _ => None,
        }
    }

    /// [`to_pyresult`] as the hand-off it is: the blackhole activation is over
    /// and its result now belongs to the Rust caller, so the residual-call
    /// exception carriers are cleared with it.
    ///
    /// `metainterp.last_exc_value` (`pyjitpl.py execute_raised`) and the
    /// `llmodel.py _store_exception` cells belong to the activation that raised
    /// into them: upstream drops the first with the MetaInterp instance and
    /// reads the second back in the same `bh_call_*` that armed it.  Pyre keeps
    /// one of each per thread, so a raise this activation already turned into
    /// its own `PyResult` — or one a nested frame absorbed before finishing —
    /// stays armed for whatever the caller does next, where a compiled
    /// `GUARD_NO_EXCEPTION` or a blackhole residual reads it as its own pending
    /// raise.  `ContinueRunningNormally` and `BailToInterpreter` are not
    /// hand-offs and keep the carriers, which is why this clears only where
    /// [`to_pyresult`] answers `Some`.
    ///
    /// [`to_pyresult`]: Self::to_pyresult
    pub fn take_pyresult(&self) -> Option<PyResult> {
        let result = self.to_pyresult()?;
        clear_residual_call_exception();
        Some(result)
    }
}

/// resume.py rebuild_from_numbering / read_jitcode_pos_pc output.
/// Each decoded frame section from rd_numb.
pub struct ResumedFrame {
    /// resume.py:1050 jitcode_pos → jitcodes[jitcode_pos].
    /// PyCode pointer — same level as frame.pycode / getcode(func).
    pub code: *const (),
    /// Python bytecode PC the resume data carries (from `frame.pc =
    /// orgpc` at trace time).  pyre's tracer records Python bytecode
    /// PCs because it interprets Python bytecode (not JitCode); `py_pc`
    /// is the raw, pre-adjustment value (before any Cache / ExtendedArg /
    /// NotTaken backtracking).
    pub py_pc: usize,
    /// Raw frame.pc from rd_numb (= orgpc from snapshot).
    /// Some(pc): snapshot guard — orgpc known, liveness-based filling.
    ///   pc=0 is valid (function start / loop header at bytecode 0).
    /// None: no-snapshot guard (rd_numb pc=-1), positional fallback.
    /// CHAIN virtualizable pointer (same value on every section).
    /// RPython parity: there is ONE virtualizable per jitdriver_sd for the
    /// whole blackhole chain; inner sections do not own a separate PyFrame.
    /// Carried on every `ResumedFrame` only because pyre pre-decodes the
    /// rd_numb stream into a `Vec<ResumedFrame>` instead of streaming it
    /// like RPython's `blackhole_from_resumedata` — the value MUST be
    /// identical across sections (enforced by `build_resumed_frames`).
    pub frame_ptr: *mut PyFrame,
    /// valuestackdepth extracted from vable_values (snapshot).
    pub vsd: usize,
    /// interp_jit.py:31 w_globals — namespace pointer from vable_values.
    /// virtualizable.py write_from_resume_data_partial:
    /// ALL static fields come from resume data, not from the heap.
    pub namespace: *const (),
    /// resume.py consume_one_section: resolved values.
    /// Structure: [live_registers...] — no scalar inputarg header.
    /// RPython parity: vable values come from snapshot, not fail_args.
    pub values: Vec<majit_ir::Value>,
}

/// resume.py decode_ref / getvirtual_ptr parity.
///
/// Re-box optimizer-unboxed values back to PyObjectRef for the
/// blackhole's ref register file. RPython's decode_ref dispatches
/// on TAGVIRTUAL/TAGCONST/TAGBOX/TAGSMALLINT; pyre's deadframe
/// already contains typed Values, so we just box Int/Float to
/// W_IntObject/W_FloatObject.
#[allow(dead_code)]
fn materialize_virtual(val: &majit_ir::Value) -> i64 {
    use majit_ir::Value;
    match val {
        Value::Ref(r) => r.as_usize() as i64,
        Value::Int(v) => pyre_object::intobject::w_int_new(*v) as i64,
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v) as i64,
        Value::Void => 0i64,
    }
}

/// resume.py _callback_i → next_int() → write_an_int.
/// RPython trusts type discipline — no cross-type coercion.
#[allow(dead_code)]
fn materialize_virtual_int(val: &majit_ir::Value) -> i64 {
    match val {
        majit_ir::Value::Int(v) => *v,
        other => panic!("materialize_virtual_int: expected Int, got {:?}", other),
    }
}

/// resume.py _callback_f → next_float() → write_a_float.
/// RPython trusts type discipline — no cross-type coercion.
#[allow(dead_code)]
fn materialize_virtual_float(val: &majit_ir::Value) -> i64 {
    match val {
        majit_ir::Value::Float(v) => v.to_bits() as i64,
        other => panic!("materialize_virtual_float: expected Float, got {:?}", other),
    }
}

/// Fused recursive call with boxed arg.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_recursive_call_1(
    caller_frame: i64,
    callable: i64,
    boxed_arg: PyObjectRef,
) -> i64 {
    // result_type=REF: no RawInt unbox needed — arg is already boxed Ref
    if majit_metainterp::majit_log_enabled() {
        let caller = unsafe { &*(caller_frame as *const PyFrame) };
        let caller_arg0 = if locals_w!(caller).len() > 0
            && !locals_w!(caller)[0].is_null()
            && unsafe { is_int(locals_w!(caller)[0]) }
        {
            Some(unsafe { w_int_get_value(locals_w!(caller)[0]) })
        } else {
            None
        };
        let callee_arg0 = if !boxed_arg.is_null() && unsafe { is_int(boxed_arg) } {
            Some(unsafe { w_int_get_value(boxed_arg) })
        } else {
            None
        };
        eprintln!(
            "[jit][force-recursive-boxed] enter caller_arg0={:?} callee_arg0={:?}",
            caller_arg0, callee_arg0
        );
    }
    let frame_ptr = create_callee_frame_impl(caller_frame, callable, &[boxed_arg]);
    let result = jit_force_callee_frame(frame_ptr);
    jit_drop_callee_frame(frame_ptr);
    if majit_metainterp::majit_log_enabled() {
        let caller = unsafe { &*(caller_frame as *const PyFrame) };
        let caller_arg0 = if locals_w!(caller).len() > 0
            && !locals_w!(caller)[0].is_null()
            && unsafe { is_int(locals_w!(caller)[0]) }
        {
            Some(unsafe { w_int_get_value(locals_w!(caller)[0]) })
        } else {
            None
        };
        eprintln!(
            "[jit][force-recursive-boxed] exit caller_arg0={:?}",
            caller_arg0
        );
    }
    result
}

/// Fused recursive call with RAW INT arg, boxed result.
///
/// This keeps the trace-side argument in raw-int form even before the callee
/// has stabilized on a raw-int finish protocol. It is a closer match to
/// RPython's recursive portal argument flow than boxing the argument in the
/// trace before every helper-boundary call.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_recursive_call_argraw_boxed_1(
    caller_frame: i64,
    callable: i64,
    raw_int_arg: i64,
) -> i64 {
    // result_type=REF: box the int arg, dispatch as boxed Ref
    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    jit_force_recursive_call_1(caller_frame, callable, boxed)
}

/// Self-recursive single-arg boxed helper.
///
/// Keeps the boxed helper path off the generic callable redispatch and
/// blackhole fallback route. This mirrors the specialized raw helper:
/// the callee frame is created directly from the caller's code/globals.
/// RPython warmspot.py ll_portal_runner portal_runner parity.
///
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_self_recursive_call_1(
    caller_frame: i64,
    boxed_arg: PyObjectRef,
) -> PyObjectRef {
    if caller_frame == 0 {
        return boxed_arg;
    }
    // result_type=REF: arg is already boxed Ref
    let frame_ptr = create_self_recursive_callee_frame_impl_1_boxed(caller_frame, boxed_arg);
    // blackhole.py bhimpl_recursive_call_r: calls
    // cpu.bh_call_r(portal_runner_adr, ...) which re-enters JIT.
    // warmspot.py ll_portal_runner: maybe_compile_and_run + portal_ptr.
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        crate::eval::portal_runner(frame)
    };
    jit_drop_callee_frame(frame_ptr);
    result
}

/// Self-recursive single-arg helper with raw-int arg and boxed result.
///
/// Mirrors `jit_force_self_recursive_call_1`, but keeps the trace-side
/// argument unboxed so recursive helper-boundary calls do not allocate a
/// temporary `W_Int` in the trace.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_self_recursive_call_argraw_boxed_1(
    caller_frame: i64,
    raw_int_arg: i64,
) -> i64 {
    // result_type=REF: box the int arg, dispatch as boxed Ref
    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    jit_force_self_recursive_call_1(caller_frame, boxed) as i64
}

/// Fully fused recursive call with RAW INT arg — no boxing in trace at all.
///
/// Eliminates ALL per-recursive-call overhead from trace:
///   Before: CallI(box) + CallMayForce(force_1, frame, callable, boxed)
///   After:  CallMayForce(force_raw_1, frame, callable, raw_int)
///
/// Boxing happens inside this function, not in the trace.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_recursive_call_raw_1(
    caller_frame: i64,
    callable_ref: PyObjectRef,
    raw_int_arg: i64,
) -> i64 {
    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    let frame_ptr = create_callee_frame_impl_1_boxed(caller_frame, callable_ref, boxed);
    // blackhole.py bhimpl_recursive_call_r: a recursive call
    // from compiled assembler is `cpu.bh_call_r(portal_runner_adr, ...)`
    // — i.e. it always re-enters through the portal runner. The portal
    // runner (warmspot.py:944-953) calls `maybe_compile_and_run` and
    // then `portal_ptr(*args)`, so the JIT-vs-interpreter decision is
    // made there. There is no "try blackhole first, then fallback to
    // portal_runner" path in RPython.
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        // warmspot.py:449 result_type=REF: always boxed Ref
        crate::eval::portal_runner(frame) as i64
    };
    jit_drop_callee_frame(frame_ptr);
    result
}

/// Specialized raw-int recursive helper for closure-free self-recursion.
///
/// Unlike `jit_force_recursive_call_raw_1`, this does not need to rediscover
/// the callee's code/globals from a function object on every call. The caller
/// frame already carries the exact recursive target:
/// - `caller.pycode` is the callee code object
/// - `caller.get_w_globals()` resolves its code-global/common namespace
/// - the current activation supplies the shared execution context
///
/// Trace-time recursive CALL_ASSEMBLER handles the optimized path. The
/// concrete helper should mirror RPython's force_fn behavior: execute the
/// callee's own frame without JIT on that frame, but let nested portal
/// calls re-enter compiled code through the normal portal runner path.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_self_recursive_call_raw_1(caller_frame: i64, raw_int_arg: i64) -> i64 {
    if majit_metainterp::majit_log_enabled() && raw_int_arg <= 4 {
        eprintln!("[jit][force-self-recursive] enter arg={}", raw_int_arg);
    }
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let w_code = caller.pycode;
    let green_key = crate::eval::make_green_key(w_code, 0, caller.get_is_being_profiled());
    let _token_num = self_recursive_dispatch(green_key);

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    let frame_ptr = create_self_recursive_callee_frame_impl_1_boxed(caller_frame, boxed);
    // blackhole.py bhimpl_recursive_call_r: calls
    // cpu.bh_call_r(portal_runner_adr, ...) which invokes
    // warmspot.py ll_portal_runner. portal_runner re-enters
    // the JIT through maybe_compile_and_run + portal_ptr.
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        // warmspot.py:449 result_type=REF: always boxed Ref
        crate::eval::portal_runner(frame) as i64
    };
    jit_drop_callee_frame(frame_ptr);
    if majit_metainterp::majit_log_enabled() && raw_int_arg <= 4 {
        eprintln!(
            "[jit][force-self-recursive] exit arg={} result={}",
            raw_int_arg, result
        );
    }
    result
}

/// Unbox a Ref (PyObjectRef to boxed int) to a raw i64 value.
/// Used by call_assembler_guard_failure's FALLBACK path when the first
/// local is a Ref type (boxed int) instead of raw Int.
fn unbox_int_for_force(raw: i64) -> i64 {
    let obj = raw as pyre_object::PyObjectRef;
    if !obj.is_null() && unsafe { is_int(obj) } {
        unsafe { w_int_get_value(obj) }
    } else {
        raw
    }
}

/// resume.py VStrPlainInfo.allocate / resume.py
/// VUniPlainInfo.allocate parity — materialize a Plain string/unicode
/// virtual via the frontend backend's bh_newstr / bh_strsetitem (and
/// unicode variants). Registered into Cranelift's guard-exit recovery
/// path so `rebuild_state_after_failure` hands bridge-input refs a real
/// string pointer instead of NULL (compiler.rs).
#[allow(dead_code)]
fn materialize_str_plain_for_cranelift(is_unicode: bool, chars: &[i64]) -> i64 {
    use majit_backend::Backend;
    let (driver, _) = crate::eval::driver_pair();
    let backend = driver.meta_interp().backend();
    let length = chars.len() as i64;
    let string = if is_unicode {
        backend.bh_newunicode(length)
    } else {
        backend.bh_newstr(length)
    };
    for (i, c) in chars.iter().enumerate() {
        if is_unicode {
            backend.bh_unicodesetitem(string, i as i64, *c);
        } else {
            backend.bh_strsetitem(string, i as i64, *c);
        }
    }
    string
}

/// resume.py string_concat / slice_string and the unicode
/// counterparts — materialize Concat / Slice string virtuals via
/// cpu.bh_call_r(funcptr, args_i, args_r, args_f, calldescr).
#[allow(dead_code)]
fn materialize_str_call_for_cranelift(
    _is_unicode: bool,
    func: i64,
    calldescr: &majit_ir::DescrRef,
    args_i: &[i64],
    args_r: &[i64],
) -> i64 {
    use majit_backend::Backend;
    let (driver, _) = crate::eval::driver_pair();
    let backend = driver.meta_interp().backend();
    let cd = calldescr
        .as_call_descr()
        .expect("materialize_str_call: calldescr must downcast to CallDescr");
    let bh_calldescr = majit_jitcode::jitcode::BhCallDescr::from_call_descr(cd);
    let result = backend.bh_call_r(
        func,
        if args_i.is_empty() {
            None
        } else {
            Some(args_i)
        },
        if args_r.is_empty() {
            None
        } else {
            Some(args_r)
        },
        None,
        &bh_calldescr,
    );
    result.0 as i64
}

pub fn install_jit_call_bridge() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        // warmstate.py equal_whatever ll_streq / ll_strhash registration —
        // pyre's `#[jit_interp]` macro emits the canonical `*const
        // &'static str` slot ABI for STR/UNICODE greens
        // (`majit-macros::jit_interp::emit_green_repr`).  Each frontend
        // owns its own `rstr.STR` / `rstr.UNICODE` decoder; pyre
        // registers its `default_str_eq` / `default_str_hash` /
        // `default_unicode_hash` here so `equal_whatever(GreenType::Str,
        // ..)` / `hash_whatever(GreenType::Str, ..)` route to content-
        // aware comparison/hashing for every `greens=[name: str]`
        // JitCell.  Lives on the pyre side (not in metainterp's
        // `JitDriver::new`) so non-pyre frontends supply their own slot
        // ABI without inheriting pyre's via a process-global default.
        majit_ir::value::set_str_resolver(
            majit_ir::value::default_str_eq,
            majit_ir::value::default_str_hash,
        );
        majit_ir::value::set_unicode_resolver(
            majit_ir::value::default_str_eq,
            majit_ir::value::default_unicode_hash,
        );
        // Residual trampolines rebuild `&str` / `&Wtf8` from one `Ref` word,
        // a pointer to an rstr `STR`. The reader is this frontend's layout.
        majit_ir::helper_fnaddr::set_rstr_payload_reader(
            pyre_object::unicodeobject::rstr_payload_word,
        );
        // Same frontend-owns-its-object-model split for the `w_class`
        // header identity `OptVirtualize` folds `new_with_vtable` reads
        // to: `SimpleSizeDescr` carries pyre's `ob_type` in its `vtable`
        // slot but knows nothing about `PyType`, so pyre registers the
        // `get_instantiate` decoder here.
        majit_ir::descr::set_w_class_obj_resolver(pyre_jit_trace::descr::w_class_obj_for_vtable);
        register_jit_function_caller(jit_call_user_function_from_frame);
        register_jit_exc_raiser(jit_exc_raise_shim);
        register_jit_exc_clearer(jit_exc_clear_shim);
        // compile.py `memory_error = MemoryError()` parity — give
        // the backend malloc helpers a way to set `JIT_EXC_VALUE` to
        // pyre's lazy `W_BaseException(MemoryError, "")` singleton
        // before propagating NULL on OOM.  Backend-shared (mirrors
        // RPython where the same `memory_error` instance is reachable
        // from both the x86 and aarch64 backends).
        majit_backend::register_memory_error_provider(|| {
            pyre_object::interp_exceptions::memory_error_singleton() as i64
        });
        // rpython/translator/c/src/stack.h:42-43 LL_stack_criticalcode_start
        // /stop hooks — wrap blackhole_from_resumedata,
        // handle_async_forcing, and handle_guard_failure_in_trace so
        // StackOverflow doesn't interrupt those critical sections.
        // The pyre helpers are `extern "C" fn()`; thin wrappers adapt
        // them to the Rust `fn()` signature register_criticalcode_hooks
        // expects.
        fn criticalcode_start_adapter() {
            pyre_interpreter::stack_check::pyre_stack_criticalcode_start();
        }
        fn criticalcode_stop_adapter() {
            pyre_interpreter::stack_check::pyre_stack_criticalcode_stop();
        }
        majit_metainterp::register_criticalcode_hooks(
            criticalcode_start_adapter,
            criticalcode_stop_adapter,
        );
        // rpython/rlib/rstack.py stack_almost_full hook — lets
        // compile.py:702-703 and warmstate.py:430 query the recursion-
        // limit-driven PYRE_STACKTOOBIG budget instead of the OS thread
        // stack.
        fn stack_almost_full_adapter() -> bool {
            pyre_interpreter::stack_check::stack_almost_full()
        }
        majit_metainterp::register_stack_almost_full_hook(stack_almost_full_adapter);
        #[cfg(feature = "cranelift")]
        {
            majit_backend_cranelift::register_call_assembler_force(jit_force_callee_frame);
            majit_backend_cranelift::register_call_assembler_bridge(jit_ca_handle_guard_failure);
            majit_backend_cranelift::register_call_assembler_blackhole(
                jit_blackhole_resume_from_guard,
            );
            majit_backend_cranelift::register_jitframe_layout(jitframe_layout_info());
            majit_backend_cranelift::register_call_assembler_unbox_int(unbox_int_for_force);
            // resume.py VStr/VUni.allocate parity — Cranelift
            // backend's materialize_virtual_recursive invokes these
            // callbacks so that bridge-input refs and call_assembler
            // blackhole inputs (compiler.rs) receive materialized string
            // pointers, not NULL.
            majit_backend_cranelift::register_materialize_str_plain(
                materialize_str_plain_for_cranelift,
            );
            majit_backend_cranelift::register_materialize_str_call(
                materialize_str_call_for_cranelift,
            );
            // rpython/jit/backend/llsupport/llmodel.py insert_stack_check
            // parity. Cranelift consumes these addresses to emit the same
            // load/sub/cmp fast path as dynasm, using a stack-slot address as
            // its current-stack approximation and calling slowpath only on miss.
            // `pyre_stack_check_slowpath_for_backend` raises into pos_exception
            // (`llmodel.py` `_store_exception`). `_build_stack_check_slowpath`
            // copies pos_exc_value into jf_guard_exc and jumps to
            // propagate_exception_path.
            majit_backend_cranelift::register_stack_check_addresses(
                pyre_interpreter::stack_check::pyre_stack_get_end_adr(),
                pyre_interpreter::stack_check::pyre_stack_get_length_adr(),
                pyre_interpreter::stack_check::pyre_stack_check_slowpath_for_backend as *const ()
                    as usize,
            );
            majit_backend_cranelift::register_prologue_probe_addr(
                pyre_interpreter::stack_check::pyre_stack_check_for_jit_prologue as *const ()
                    as usize,
            );
        }
        #[cfg(feature = "dynasm")]
        {
            majit_backend_dynasm::register_call_assembler_force(jit_force_callee_frame);
            majit_backend_dynasm::register_call_assembler_bridge(jit_ca_handle_guard_failure);
            majit_backend_dynasm::register_call_assembler_blackhole(
                jit_blackhole_resume_from_guard,
            );
            majit_backend_dynasm::register_jitframe_layout(jitframe_layout_info_dynasm());
            majit_backend_dynasm::register_call_assembler_unbox_int(unbox_int_for_force);
            // rpython/jit/backend/llsupport/llmodel.py insert_stack_check
            // parity. The backend inlines MOV [endaddr]; SUB rsp; CMP [lengthaddr]
            // in every JIT prologue and calls slowpath_addr on miss.
            // `pyre_stack_check_slowpath_for_backend` raises into pos_exception
            // (`llmodel.py` `_store_exception`). `_build_stack_check_slowpath`
            // copies pos_exc_value into jf_guard_exc and jumps to
            // propagate_exception_path.
            majit_backend_dynasm::register_stack_check_addresses(
                pyre_interpreter::stack_check::pyre_stack_get_end_adr(),
                pyre_interpreter::stack_check::pyre_stack_get_length_adr(),
                pyre_interpreter::stack_check::pyre_stack_check_slowpath_for_backend as *const ()
                    as usize,
            );
        }
    });
}

/// `DoneWithThisFrame` concrete return from an assembler-called frame.
/// A void return stashes `Null`, i.e. Python `None`.
fn ca_finish_return_word(cv: pyre_jit_trace::state::ConcreteValue) -> i64 {
    let result = match cv {
        pyre_jit_trace::state::ConcreteValue::Null => pyre_object::w_none(),
        other => other.to_pyobj(),
    };
    result as i64
}

/// `blackhole.py` `_exit_frame_with_exception` / `jitexc.py`
/// `ExitFrameWithExceptionRef`: publish the uncaught exception and
/// return the CALL_ASSEMBLER null result so the caller's
/// `GUARD_NO_EXCEPTION` fires.
fn ca_finish_raise_word(cv: pyre_jit_trace::state::ConcreteValue) -> i64 {
    let pyre_jit_trace::state::ConcreteValue::Ref(exc_ref) = cv else {
        unreachable!("FinishConcrete::Raise must hold a concrete Ref")
    };
    debug_assert!(!exc_ref.is_null());
    publish_residual_call_exception(exc_ref);
    0
}

/// compile.py handle_fail → resume_in_blackhole parity.
///
/// RPython: guard failure always resumes via jitcode-level blackhole
/// (blackhole_from_resumedata → _run_forever). There is no IR-level
/// blackhole in RPython.
///
/// When rd_numb is available, uses ResumeDataDirectReader for exact
/// frame decoding (resume.py parity).
/// Complete a CALL_ASSEMBLER callee whose bridge walk already carried it past
/// the guard, so the guard-state resume must not re-run the region.
///
/// `callee_frame` is the callee `PyFrame*` the guard failed in — pyre's CA
/// virtualizable layout puts it at `fail_values[0]` (the same layout
/// `jit_ca_handle_guard_failure` reads `raw_values[0]` under).
/// Returns `Some(result)` once the walk completed the callee, i.e. the
/// `DoneWithThisFrame` / `ExitFrameWithExceptionRef` the assembler caller
/// catches (pyjitpl.py opimpl_raise, jitexc.py), and `None` when the caller still
/// has to resume the guard state.
///
/// Shared by both CALL_ASSEMBLER completions — the native back-to-back
/// blackhole hook below and the wasm in-guest deopt helper
/// (`wasm_ca_resume_deopt`) — so the two cannot drift apart on which walk
/// outcomes are replay-safe.
fn ca_complete_after_bridge_walk(
    ca_finished_frame: usize,
    ca_adopted_frame: usize,
    callee_frame: usize,
) -> Option<i64> {
    // The CA bridge walk ran the callee to its finishframe concretely and
    // kept the finish-concrete stash (`CA_WALK_FINISHED_FRAME` handshake set
    // by `trace_and_compile_from_bridge`); a guard-state resume would re-run
    // the resumed region over the already-applied effects.  Complete the
    // callee with the stashed concrete instead.
    if ca_finished_frame != 0 && ca_finished_frame == callee_frame {
        match pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_take() {
            Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Return(cv)) => {
                return Some(ca_finish_return_word(cv));
            }
            Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Raise(cv)) => {
                return Some(ca_finish_raise_word(cv));
            }
            // The epilogue reset the stash between the hook calls; fall
            // through to the guard-state resume.
            None => {}
        }
    } else {
        // A kept stash without a matching frame must not leak into a later
        // top-level portal `fbw_finish_concrete_take`.
        pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_reset();
    }

    // raise_continue_running_normally parity (pyjitpl.py +
    // warmspot.py:970-983): the CA bridge walk committed its end-of-walk
    // state into the live callee frame and the frame adopted it. The walk
    // already executed the resumed region's effects concretely, so a
    // guard-state resume would re-apply them over the advanced heap
    // (double-execution / stale-value return). Complete the callee from its
    // adopted state via the portal runner instead — the same
    // portal_ptr(*args) completion the ContinueRunningNormally arm of
    // handle_blackhole_result performs.
    if ca_adopted_frame != 0 && ca_adopted_frame == callee_frame {
        let frame = unsafe { &mut *(ca_adopted_frame as *mut PyFrame) };
        return match crate::eval::portal_body_result(frame) {
            Ok(result) => Some(result as i64),
            Err(mut err) => {
                let exc_obj = err.to_exc_object();
                if exc_obj != pyre_object::PY_NULL {
                    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
                    store_jit_exception(exc_obj as i64);
                }
                Some(0)
            }
        };
    }

    None
}

fn jit_blackhole_resume_from_guard(
    descr_addr: usize,
    deadframe: *mut majit_backend::jitframe::JitFrame,
) -> Option<i64> {
    let ca_adopted_frame = CA_WALK_ADOPTED_FRAME.with(|c| c.replace(0));
    let ca_finished_frame = CA_WALK_FINISHED_FRAME.with(|c| c.replace(0));
    let ca_resume_frame = CA_WALK_RESUME_FRAME.with(|c| c.replace(0));

    // Drain the exception `park_jit_pending_error` parked
    // (`unpackiterable_driver`). The prologue stack check is not a
    // producer of that slot: `stack_check_slowpath` publishes through
    // `jit_publish_exception` into `pos_exception`, and
    // `_build_stack_check_slowpath` copies `pos_exc_value` into
    // `jf_guard_exc` before `propagate_exception_path`. Do this before
    // resume setup so a parked driver error is not overwritten by the
    // blackhole walk.
    if let Err(mut exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
        // This callback returns as the result of CALL_ASSEMBLER, whose caller
        // immediately executes GUARD_NO_EXCEPTION.  Publish into the same two
        // exception cells as every raising residual call; merely stashing the
        // PyError for the outer eval loop would let the null call result reach
        // bytecode consumers before that boundary.
        let exc_obj = exc.to_exc_object();
        if exc_obj != pyre_object::PY_NULL {
            publish_residual_call_exception(exc_obj);
        }
        pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_reset();
        return None;
    }

    if deadframe.is_null() {
        pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_reset();
        return None;
    }
    // `llmodel.py grab_exc_value`: read `jf_guard_exc`. The slot stays put;
    // `jitframe_trace` is the root across this resume.
    let guard_exc = unsafe { (*deadframe).jf_guard_exc as i64 };

    // compile.py `resume_in_blackhole(descr, deadframe)`:
    // recover the failed descr from `descr_addr` (history.py:125
    // `cpu.get_latest_descr` is the C-ABI carrier) and derive
    // (trace_id, fail_index) from descr identity.  The recovery is
    // infallible for live JIT code — `Backend::fail_descr_arc_from_addr`
    // panics if the raw value is not a live `FailDescrCell` pointer,
    // matching RPython's `cpu.get_latest_descr(deadframe)`
    // (warmspot.py assembler_call_helper) which has no failure mode.
    use majit_backend::Backend;
    let (driver, _) = crate::eval::driver_pair();
    let backend = driver.meta_interp().backend();
    let descr_arc = backend.fail_descr_arc_from_addr(descr_addr);
    let descr_fd = descr_arc
        .as_fail_descr()
        .expect("fail_descr_arc_from_addr returned non-FailDescr");
    let trace_id = descr_fd.trace_id();
    let fail_index = descr_fd.fail_index_per_trace();
    let n_fail_args = descr_fd.fail_arg_types().len();
    // `resume_in_blackhole` reads fail args off the deadframe
    // (`cpu.grab_exc_value` / `get_latest_descr` path in `llmodel.py`).
    let fail_args = majit_backend::FailArgSource::from_jitframe(deadframe, descr_fd, n_fail_args);
    let fail0 = if ca_resume_frame != 0 {
        ca_resume_frame as i64
    } else if fail_args.len() > 0 {
        fail_args.get(0)
    } else {
        0
    };

    if let Some(result) =
        ca_complete_after_bridge_walk(ca_finished_frame, ca_adopted_frame, fail0 as usize)
    {
        return Some(result);
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[blackhole-resume] trace={} fail_idx={} nvals={}",
            trace_id, fail_index, n_fail_args,
        );
    }

    // Path 1: rd_numb-based resume (resume.py blackhole_from_resumedata)
    // When rd_numb is present, use ResumeDataDirectReader to decode
    // frame sections precisely, matching RPython blackhole_from_resumedata.
    //
    // `resume_in_blackhole(metainterp_sd, jitdriver_sd, self, deadframe)`
    // is handed the failing descr and reads everything off it:
    // `ResumeGuardDescr.get_resumestorage()` for the `rd_*` pool (shared
    // through `Arc<ResumeStorage>` so blackhole resume reads the same
    // `rd_consts` the GC root walker updates) and the guard's own
    // `fail_arg_types` for the deadframe slot types. resume.py parity:
    // deadframe_types tells decode_ref() whether a TAGBOX slot holds a raw
    // int (needs boxing) or a GcRef (use as-is). Without it, unboxed ints
    // are treated as pointers → SIGSEGV. Both come off the one descr so
    // the storage and the types cannot describe different deadframes.
    let exit_layout = majit_metainterp::exit_layout_for_descr(&descr_arc);
    if let Some(storage) = exit_layout.storage.clone() {
        let deadframe_types = exit_layout.exit_types.to_vec();
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[blackhole-resume] rd_numb len={} rd_consts len={} raw_deadframe len={}",
                storage.rd_numb().expect("rd_numb").len(),
                storage.rd_consts().unwrap_or(&[]).len(),
                fail_args.len(),
            );
        }
        // resume.py:922 storage.rd_consts: the decoder borrows the shared
        // pool; TAGCONST Ref entries stay visible to `walk_rd_consts_refs`.
        // resume.py _prepare_pendingfields(storage.rd_pendingfields):
        // deferred field writes must be replayed before consume_vref_and_vable.
        // blackhole.py `current_exc = _prepare_resume_from_failure(
        // guard_opnum, deadframe)`. `guard_exc` is `cpu.grab_exc_value`
        // (`llmodel.py`) read off this deadframe above, so a
        // GUARD_NO_EXCEPTION / GUARD_EXCEPTION / GUARD_NOT_FORCED failure
        // inside a CALL_ASSEMBLER-entered callee delivers its pending
        // exception to the blackhole resume instead of resuming the
        // no-exception continuation with a NULL result.
        // compile.py `ResumeGuardForcedDescr.handle_fail` fishes the
        // cache `handle_async_forcing` saved via `cpu.get_savedata_ref(deadframe)`.
        let savedata = crate::eval::savedata_from_jitframe(deadframe);
        let mut savedata_slot = [savedata.map_or(0, majit_ir::GcRef::as_usize) as i64];
        let _savedata_root = unsafe {
            majit_metainterp::resume::DeadFrameRefRoots::enter(&mut savedata_slot, |_| {
                savedata.is_some()
            })
        };
        let all_virtuals = if descr_arc.is_guard_forced() && savedata.is_some() {
            majit_metainterp::AllVirtuals::show(majit_ir::GcRef(savedata_slot[0] as usize))
        } else {
            None
        };
        let result = blackhole_resume_via_rd_numb(
            storage.rd_numb().expect("rd_numb"),
            storage.rd_consts().unwrap_or(&[]),
            fail_args,
            storage.rd_pendingfields(),
            storage.rd_virtuals(),
            Some(deadframe_types.as_slice()),
            guard_exc,
            false, // CALL_ASSEMBLER portal is jd0 (virtualizable)
            all_virtuals,
            None, // `raw_deadframe` is rooted only by the copy made inside
        );
        return handle_blackhole_result(result);
    }

    // RPython compile.py:701-716 parity: every guard must have rd_numb
    // from capture_resumedata + store_final_boxes_in_guard (resume.py:397).
    // Hitting this path means a guard was compiled without snapshot data.
    //
    // `PropagateExceptionDescr.handle_fail` (compile.py) still has to
    // re-raise: the failure stub stored the exception in `jf_guard_exc`
    // and this function already received it as `guard_exc`. Dropping it
    // and returning no result makes the CALL_ASSEMBLER caller observe a
    // NULL return with no exception set.
    if guard_exc != 0 {
        publish_residual_call_exception(guard_exc as PyObjectRef);
        return Some(0);
    }
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[blackhole-resume] no rd_numb for trace={} fail={} (force_fn fallback)",
            trace_id, fail_index,
        );
    }
    None
}

/// RAII guard registering each `Ref`-typed slot of the resume `deadframe`
/// copy as a GC root for the duration of `blackhole_from_resumedata`.
///
/// `deadframe` is an off-heap copy of the guard-failure values
/// (`raw_exit_values` over the result's exit slots).  The collector is moving,
/// so a minor collection
/// triggered while `blackhole_from_resumedata` lazily materializes virtuals
/// (`getvirtual_ptr` → allocator) relocates the boxed objects and leaves the
/// copy holding from-space pointers; `resume.rs::decode_ref` then reads
/// a stale pointer for a box-sourced slot and the blackhole dereferences
/// freed memory.  Resume *constants* are already forwarded by
/// `rd_consts_root_walker_area`, but the box-sourced slots here are not.
/// Registering each `Ref` element slot makes the root walker forward it in
/// place.
///
/// Only `Ref`-typed slots are registered: `Int`/`Float` slots hold raw
/// scalars (`decode_ref` boxes those lazily via `box_int`/`box_float`), and a
/// scalar that happened to alias a managed address must never be treated as a
/// root.  `deadframe_types[idx]` is parallel to `deadframe[idx]` (resume.rs
/// `decode_ref` keys the same index), so the type gate is exact.
#[allow(dead_code)] // resume.py decode_ref deadframe roots
struct ResumeDeadframeRoots {
    slots: Vec<*mut pyre_object::gc_hook::GCREF>,
    frame_roots: Vec<pyre_interpreter::pyframe::FrameLocalsRoot>,
}

#[allow(dead_code)] // resume.py decode_ref deadframe roots
impl ResumeDeadframeRoots {
    fn register_pyframe_locals_slot(
        value: i64,
        frame_roots: &mut Vec<pyre_interpreter::pyframe::FrameLocalsRoot>,
    ) {
        let ptr = value as pyre_object::gc_hook::GCREF;
        if ptr.is_null()
            || !pyre_object::gc_hook::try_gc_owns_object(ptr)
            || majit_gc::gc_is_nursery_object(ptr as usize)
        {
            return;
        }
        let current = pyre_object::gc_hook::try_gc_current_object_address(ptr);
        if current.is_null() || !pyre_object::gc_hook::try_gc_owns_object(current) {
            return;
        }
        let type_id = unsafe { (*majit_gc::header::header_of(current as usize)).type_id() };
        if type_id != pyre_interpreter::pyframe::PYFRAME_GC_TYPE_ID {
            return;
        }
        frame_roots.push(pyre_interpreter::pyframe::FrameLocalsRoot::new(
            current as *mut PyFrame,
        ));
    }

    fn register(deadframe: &mut [i64], deadframe_types: Option<&[majit_ir::Type]>) -> Self {
        let mut slots = Vec::new();
        let mut frame_roots = Vec::new();
        if let Some(types) = deadframe_types {
            for (idx, ty) in types.iter().enumerate() {
                if !matches!(ty, majit_ir::Type::Ref) {
                    continue;
                }
                let Some(cell) = deadframe.get_mut(idx) else {
                    break;
                };
                // Null / `NULLREF` slots carry no object to forward.
                if *cell == 0 {
                    continue;
                }
                let slot = cell as *mut i64 as *mut pyre_object::gc_hook::GCREF;
                if unsafe { pyre_object::gc_hook::try_gc_add_root(slot) } {
                    slots.push(slot);
                }
                Self::register_pyframe_locals_slot(*cell, &mut frame_roots);
            }
        }
        Self { slots, frame_roots }
    }
}

impl Drop for ResumeDeadframeRoots {
    fn drop(&mut self) {
        for &slot in &self.slots {
            pyre_object::gc_hook::try_gc_remove_root(slot);
        }
        self.frame_roots.clear();
    }
}

/// RAII guard rooting a single bare `Ref`-carrying `i64` slot across an
/// allocating region. Mirrors the dynasm CA helper's `gc_add_root` over the
/// grabbed guard exception (majit-backend-dynasm `ca_helper`): once the slot is
/// registered the collector forwards `*slot` in place if it moves the object,
/// so the bare carrier stays valid across a hook that may allocate. `register`
/// is a no-op for a null slot (non-exception guards leave `grab_exc_value` 0).
#[cfg(target_arch = "wasm32")]
struct BareRefRoot {
    slot: Option<*mut pyre_object::gc_hook::GCREF>,
}

#[cfg(target_arch = "wasm32")]
impl BareRefRoot {
    fn register(value: &mut i64) -> Self {
        if *value == 0 {
            return Self { slot: None };
        }
        let slot = value as *mut i64 as *mut pyre_object::gc_hook::GCREF;
        let registered = unsafe { pyre_object::gc_hook::try_gc_add_root(slot) };
        Self {
            slot: registered.then_some(slot),
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl Drop for BareRefRoot {
    fn drop(&mut self) {
        if let Some(slot) = self.slot {
            pyre_object::gc_hook::try_gc_remove_root(slot);
        }
    }
}

/// `blackhole.py _exit_frame_with_exception`:
///
/// ```python
/// e = cast_opaque_ptr(GCREF, e)
/// raise ExitFrameWithExceptionRef(e)
/// ```
///
/// Upstream hands the value on as an opaque GCREF and every later
/// classification runs through a real class lookup.  Pyre instead reads a
/// `#[repr(u8)]` `ExcKind` tag out of the object and feeds it to
/// `kind_from_exc`, a wildcard-free match the compiler may lower to a
/// bounds-check-free jump table — so a word that is not a `W_BaseException`
/// becomes an indirect branch to an address derived from whatever byte sits
/// at the tag offset.  `w_exception_kind_checked` restores the class check in
/// front of that read; a value that fails it names itself here instead.
fn exit_frame_exception_ref(
    exc_value: i64,
    site: &str,
    context: impl FnOnce() -> String,
) -> pyre_interpreter::PyError {
    let obj = exc_value as PyObjectRef;
    if unsafe { pyre_object::interp_exceptions::w_exception_kind_checked(obj) }.is_some() {
        return unsafe { pyre_interpreter::PyError::from_exc_object(obj) };
    }
    if obj.is_null() {
        return pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::RuntimeError,
            "blackhole exception (null exc_value)",
        );
    }
    reject_non_exception_channel_value(obj, site, context)
}

/// Report a value that reached an exception channel without being a
/// `W_BaseException`, and abort.
///
/// Never returns: the caller is about to classify the value by its `ExcKind`
/// tag, and there is no correct classification for a value that has no tag.
/// Name lookups through `ob_type` / `w_class` are intentionally omitted —
/// an aligned word is not a live type, and following one is how this
/// reporter SIGSEGVed after already printing the header dump.
fn reject_non_exception_channel_value(
    obj: PyObjectRef,
    site: &str,
    context: impl FnOnce() -> String,
) -> ! {
    // Nothing has been read out of `obj` yet, and this function only runs for a
    // value already known not to be a `W_BaseException` — a small integer or an
    // address at the end of a mapping reaches here too, and every read below
    // can fault. Emit the pointer first so that a fault still leaves evidence.
    //
    // Every line goes through the interpreter's own fd-2 seam, not `eprintln!`:
    // the wasm guest has no process stderr, so an `eprintln!` here is a silent
    // sink and this reporter — whose only job is producing evidence before the
    // abort — would leave nothing but the panic string on the one backend where
    // the value is hardest to reconstruct.
    pyre_interpreter::host_seam::emit_stderr(
        format!("[jit][BUG] {site}: not a W_BaseException: obj={obj:p}\n").as_bytes(),
    );
    // Then the raw header, before touching anything reached *through* it: if
    // `ob_type` is itself garbage the name lookup below faults, and then this
    // line is all the evidence there is.
    let tag = unsafe { pyre_object::interp_exceptions::w_exception_kind_byte(obj) };
    let words = unsafe { std::slice::from_raw_parts(obj as *const u64, 4) };
    pyre_interpreter::host_seam::emit_stderr(
        format!(
            "[jit][BUG] {site}: obj={obj:p} tag_byte={tag} \
             words=[{:#018x} {:#018x} {:#018x} {:#018x}]\n",
            words[0], words[1], words[2], words[3],
        )
        .as_bytes(),
    );
    // Second line, and only now: the context may probe words of unproven
    // provenance, so the header above must already be out.
    pyre_interpreter::host_seam::emit_stderr(
        format!("[jit][BUG] {site}: context: {}\n", context()).as_bytes(),
    );
    // Do not follow `ob_type` / `w_class`. Alignment is not a type proof —
    // a reused nursery word that happens to be 8-aligned still faults
    // `type_name_of` / `w_type_get_name`.
    panic!(
        "{site}: exception channel value is not a W_BaseException \
         (obj={obj:p} tag_byte={tag} \
         words=[{:#018x} {:#018x} {:#018x} {:#018x}])",
        words[0], words[1], words[2], words[3]
    );
}

/// `executioncontext.py ExecutionContext.leave`'s frame-chain half, for a
/// frame a blackhole resumed into and has now finished.
///
/// The profile-hook half (`if self.profilefunc: self._trace(frame,
/// 'leaveframe', w_exitvalue)`) stays with the interpreter's own
/// [`pyre_interpreter::PyExecutionContext::leave`], for the reason
/// `walker_ec_leave` states: `is_being_profiled` is a portal-driver green, so a
/// trace recorded with profiling off is only ever entered with profiling off,
/// and the blackhole resuming out of it inherits that key.
fn leave_resumed_blackhole_frame(
    frame: &majit_metainterp::blackhole::BlackholeInterpreter,
    got_exception: bool,
) {
    let recovered = frame.virtualizable_ptr as *mut PyFrame;
    if recovered.is_null() {
        return;
    }
    let ec =
        pyre_interpreter::call::getexecutioncontext() as *mut pyre_interpreter::PyExecutionContext;
    if ec.is_null() {
        return;
    }
    let open = unsafe { pyre_interpreter::executioncontext::vref_referent((*ec).topframeref) };
    if std::ptr::eq(open, recovered) {
        // `leave(frame)` is given the current top. The recovered red is that
        // frame, so run the chain half in full.
        leave_compiled_frame_chain(recovered, got_exception);
        return;
    }
    // Not the open scope. `leave` must not rewrite `topframeref` or it
    // would drop an inlined callee still sitting there. The blackhole's
    // own frame still needs the `dispatch_bytecode` / `handle_operation_error`
    // finish mark if it is a live chain member (the portal under that
    // callee). A garbage red is not on the chain and is not dereferenced.
    if frame_is_on_unforced_ec_chain(ec, recovered) {
        unsafe {
            (*recovered).set_frame_finished_execution(true);
        }
    }
}

/// Walk `topframeref` / `f_backref` without forcing, the way
/// `executioncontext.py` moves those slots (no parens) rather than
/// `frame.f_backref()`.
fn frame_is_on_unforced_ec_chain(
    ec: *mut pyre_interpreter::PyExecutionContext,
    frame_ptr: *mut PyFrame,
) -> bool {
    let mut cur = unsafe { pyre_interpreter::executioncontext::vref_referent((*ec).topframeref) };
    // ExecutionContext.enter/leave (executioncontext.py) maintain a chain
    // ending at null, not a fixed-depth window. A recovered ancestor may be
    // arbitrarily far below the open scope when the recursion limit is raised.
    while !cur.is_null() {
        if std::ptr::eq(cur, frame_ptr) {
            return true;
        }
        cur = unsafe { pyre_interpreter::executioncontext::vref_referent((*cur).f_backref) };
    }
    false
}

/// `executioncontext.py ExecutionContext.leave`'s frame-chain half for a frame
/// resumed by the blackhole interpreter after compiled code has gone away.
///
/// It closes a scope whose `enter` ran somewhere the interpreter's `leave`
/// never reaches, once the compiled frame is gone.  That is what makes this a
/// narrower recovery operation than `leave` itself:
///
/// * The identity guard. `leave` reads `frame_vref` off `topframeref` and
///   forces it; that word only names this frame while its scope is the open
///   one. A walk that materialised an inlined callee can leave a different vref
///   there, and the caller has no other way to find out.
/// * No force OF THIS FRAME'S OWN VREF. The guard has already established that
///   `frame_vref` resolves to this frame, so `leave`'s `frame_vref()` has
///   nothing left to materialize, and forcing a vref that was finished with the
///   NULL form is exactly what `force_pyframe_vref` refuses — measured as
///   `InvalidVirtualRef: frame-chain vref forced after its frame died` on
///   `exception_escape_inlined_midframe_tb_node` and
///   `exception_try_call_inlined_callee_raise`.
///
/// That measurement scoped only the `frame_vref()` half.  This bullet used to
/// carry a third one extending it to the caller — "reached through
/// `vref_referent` rather than `get_f_back`, which forces, for the same
/// reason" — and the reason does not carry: see below.  Both fixtures named
/// above pass with the caller force restored.
///
/// The caller, then, is reached with the force — `f_back =
/// frame.f_backref()`, parens included, i.e. [`PyFrame::get_f_back`].  That
/// force is NOT covered by the argument above: the identity guard establishes
/// something about `frame_vref`, and nothing about `f_backref`.  Reading the
/// caller with the non-forcing `vref_referent` answers NULL whenever the caller
/// is a still-virtual inlined frame, and NULL is then taken for "no caller", so
/// the caller is neither materialised nor marked escaped.  `force_pyframe_vref`
/// (`eval.rs`) states the invariant that breaks: a leaving frame that escaped
/// "marks the caller escaped so its own leave does the same", and the state that
/// reaches its refusal "requires that propagation to have been skipped".  With
/// the propagation skipped, `walker_ec_leave`'s unescaped arm records the NULL
/// form of `VIRTUAL_REF_FINISH`, which leaves the caller's vref at
/// `virtual_token == TOKEN_NONE` with `forced == NULL` — the invalid state
/// `virtualref.py` names `InvalidVirtualRef`.  Reading the escaped callee's
/// `f_back` afterwards then aborts the process, a non-unwinding panic:
/// measured on a residual callee whose frame escapes into a traceback and which
/// returns NORMALLY under an inlined caller.
///
fn leave_compiled_frame_chain(frame_ptr: *mut PyFrame, got_exception: bool) {
    let ec =
        pyre_interpreter::call::getexecutioncontext() as *mut pyre_interpreter::PyExecutionContext;
    if ec.is_null() {
        return;
    }
    let frame_vref = unsafe { (*ec).topframeref };
    if !std::ptr::eq(
        pyre_interpreter::executioncontext::vref_referent(frame_vref),
        frame_ptr,
    ) {
        return;
    }
    // executioncontext.py `leave`: a guard-failure blackhole resumes
    // inside a frame whose `enter` already ran in compiled code. Advancing to
    // `nextblackholeinterp` therefore has to close that still-open scope. The
    // resume-data frame chain itself is not the application frame chain, and
    // releasing a BlackholeInterpreter does not restore `topframeref`.
    unsafe {
        // `PyFrame.dispatch_bytecode` / `handle_operation_error` marks the leaving
        // frame finished. Only dereference the recovered red after proving
        // that it names this execution context's still-open frame scope.
        (*frame_ptr).set_frame_finished_execution(true);
        (*ec).topframeref = (*frame_ptr).f_backref;
        if (*frame_ptr).escaped() || got_exception {
            // `f_back = frame.f_backref()` — with the parens, so an inlined
            // caller that is still virtual is materialised here rather than
            // read as absent.  The mark is what makes the caller's own leave
            // record the non-NULL `VIRTUAL_REF_FINISH`; skipping it strands the
            // caller's vref in the state `force_pyframe_vref` aborts on.
            let f_back = (*frame_ptr).get_f_back();
            if !f_back.is_null() {
                (*f_back).mark_as_escaped();
            }
        }
    }
}

/// The escape-propagation arm of `ExecutionContext.leave`, for the portal
/// activation whose chain restore is emitted by the compiled trace.
///
/// `portal_traced_activation_result` must not call
/// [`leave_compiled_frame_chain`]: `record_ec_leave_frame_chain` owns the
/// `topframeref` restore after CALL_ASSEMBLER returns.  It still owes this
/// independent half of the same upstream `finally`, however.  A frame that
/// escaped (or raised) forces and marks its caller so a later `f_back` walk
/// cannot reach a caller virtual reference that was finished unforced.
pub(crate) fn propagate_portal_frame_escape(frame: *mut PyFrame, got_exception: bool) {
    if frame.is_null() {
        return;
    }
    unsafe {
        if (*frame).escaped() || got_exception {
            // `ExecutionContext.leave`: `f_back = frame.f_backref()`; the
            // call forces a still-virtual caller before marking it escaped.
            let f_back = (*frame).get_f_back();
            if !f_back.is_null() {
                (*f_back).mark_as_escaped();
            }
        }
    }
}

/// resume.py blackhole_from_resumedata parity:
/// Decode rd_numb via ResumeDataDirectReader, build blackhole chain,
/// run _run_forever.
pub fn blackhole_resume_via_rd_numb<'df>(
    rd_numb: &[u8],
    rd_consts: &[majit_ir::Const],
    deadframe: majit_backend::FailArgSource<'df>,
    rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    deadframe_types: Option<&[majit_ir::Type]>,
    guard_exc: i64,
    // compile.py `vinfo = self.jitdriver_sd.virtualizable_info`: a novable
    // jitdriver (jd1 `unpackiterable_driver`) has `vinfo=None`, so its resume
    // data carries no vable section. Decoding with the (jd0) vinfo would try to
    // consume a phantom vable and dereference garbage; a novable resume passes
    // `None` for both the vinfo and the per-frame virtualizable handle.
    novable: bool,
    // compile.py — the cache `handle_async_forcing` already
    // materialized, when this resume is the GUARD_NOT_FORCED that follows a
    // force. `None` for every other guard.
    all_virtuals: Option<(Vec<i64>, Vec<i64>)>,
    // The caller's own rooting of the `deadframe` slice it passed, when it has
    // one. It covers the same window as `deadframe_roots` below and must end at
    // the same point, so the caller hands ownership over instead of holding it
    // across this call: a caller-side scope would stay registered for the whole
    // forward run.
    caller_deadframe_roots: Option<majit_metainterp::resume::DeadFrameRefRoots>,
) -> BlackholeResult {
    // Same window as `handle_fail`, for every blackhole resume including the
    // CALL_ASSEMBLER caller: the decode below rebuilds virtuals through the
    // blackhole allocator while the grabbed exception is still only a bare
    // pointer with no deadframe root behind it.
    let guard_exc_root = majit_metainterp::blackhole::GuardExcRoot::park(guard_exc);

    use majit_metainterp::resume;

    // Thread-local BH pool (RPython BlackholeInterpBuilder). Each access
    // is scoped to a single call so that bh.run() (which may re-enter
    // blackhole_resume_via_rd_numb) cannot create overlapping &mut refs.
    thread_local! {
        static BH_BUILDER_RD: std::cell::UnsafeCell<majit_metainterp::blackhole::BlackholeInterpBuilder> =
            std::cell::UnsafeCell::new(pyre_jit_trace::jitcode_runtime::build_pyre_production_bh_builder());
    }
    let sync_bh_builder_control_opcodes =
        |builder: &mut majit_metainterp::blackhole::BlackholeInterpBuilder| {
            let (op_live, op_catch_exception, op_rvmprof_code) =
                pyre_jit_trace::state::blackhole_control_opcodes();
            builder.setup_cached_control_opcodes(op_live, op_catch_exception, op_rvmprof_code);
        };
    let release_bh_rd = |bh: Box<majit_metainterp::blackhole::BlackholeInterpreter>| {
        BH_BUILDER_RD.with(|cell| unsafe { (&mut *cell.get()).release_interp(bh) });
    };

    // `resume.py` `blackhole_from_resumedata` indexes `jitcodes[jitcode_pos]`,
    // the list `blackhole.py` `resume_in_blackhole` passes
    // (`MetaInterpStaticData.jitcodes` from `CodeWriter.make_jitcodes`).
    // `novable` only means this stream has no vable section.
    let resolve_jitcode = |jitcode_index: i32, pc: i32| -> Option<resume::ResolvedJitCode> {
        if pc < 0 {
            return None;
        }
        let op_live = pyre_jit_trace::state::blackhole_control_opcodes().0 as u8;
        let pyjitcode = pyre_jit_trace::state::pyjitcode_for_jitcode_index(jitcode_index)?;
        if pyjitcode.has_abort_opcode() {
            return None;
        }
        // A published resume frame carries a decodable JitCode `-live-`
        // coordinate. An unrepresentable frame declines this blackhole path.
        let resolved_pc = if pyjitcode.jitcode.can_decode_live_vars(pc as usize, op_live) {
            pc as usize
        } else {
            return None;
        };
        Some(
            resume::ResolvedJitCode::new(pyjitcode.jitcode.clone(), resolved_pc)
                .with_virtualizable_stack_base(pyjitcode.metadata.stack_base),
        )
    };

    // resume.py `cpu.get_int_value(deadframe, i)`: values stay in
    // `jf_frame[]`. `jitframe_trace` walks the Ref slots. A host copy
    // is not a deadframe.

    // resume.py _prepare_virtuals: convert RdVirtualInfo → VirtualInfo
    // for lazy materialization in getvirtual_ptr/getvirtual_int.
    let count = deadframe.len() as i32;
    let rd_virtuals_converted: Option<Vec<resume::VirtualInfo>> = rd_virtuals.map(|rd_virts| {
        let num_virtuals = rd_virts.len();
        rd_virts
            .iter()
            .map(|rd| resume::rd_virtual_to_virtual_info(rd, rd_consts, count, num_virtuals))
            .collect()
    });
    let rd_virtuals_slice = rd_virtuals_converted.as_deref();

    // resume.py blackhole_from_resumedata:
    // ResumeDataDirectReader decodes rd_numb, builds BH chain.
    // compile.py:990 parity: vinfo = self.jitdriver_sd.virtualizable_info —
    // read the active driver's cached Arc instead of rebuilding a fresh
    // VirtualizableInfo, so a single VirtualizableInfo identity is shared
    // with tracing, setup_bridge_sym, and the guard-failure recovery
    // consumers. resume.py vrefinfo = metainterp_sd.virtualref_info —
    // hand the metainterp's own VRefInfo through so consume_virtualref_info
    // can decode JIT_VIRTUAL_REF handles. resume.py ginfo is currently
    // unused in pyre (no greenfield_info installed on the driver).
    let (driver, driver_vinfo) = crate::eval::driver_pair();
    let vinfo_dyn: &dyn resume::VirtualizableInfo = driver_vinfo.as_ref();
    // A novable driver's resume data has no vable section; pass no vinfo so the
    // decoder skips `consume_vable_info` entirely.
    let vinfo_arg: Option<&dyn resume::VirtualizableInfo> =
        if novable { None } else { Some(vinfo_dyn) };
    let vrefinfo_dyn: &dyn resume::VRefInfo = driver.meta_interp().virtualref_info();
    let allocator = crate::eval::PyreBlackholeAllocator;
    // resume.py `self.metainterp_sd.liveness_info` — the one pool, for
    // every driver. A novable drain frame's `-live-` markers carry offsets
    // baked at extraction into the build-time byte stream; those bytes are the
    // prefix of this buffer (`Assembler::resuming_build_time_liveness`), so a
    // baked offset and a runtime-interned one address the same space and no
    // per-driver pick is needed. Snapshot once per call so the slice outlives
    // ResumeDataDirectReader.
    let all_liveness = pyre_jit_trace::state::liveness_info_snapshot();
    // Scope the &mut to chain construction; the run() loop below uses
    // release_bh_rd to drop and re-acquire the borrow.
    let bh = BH_BUILDER_RD.with(|cell| unsafe {
        let builder = &mut *cell.get();
        builder.set_cpu(driver.meta_interp().blackhole_cpu());
        sync_bh_builder_control_opcodes(builder);
        resume::blackhole_from_resumedata(
            builder,
            &resolve_jitcode,
            rd_numb,
            rd_consts,
            &all_liveness,
            deadframe.clone(),
            deadframe_types,        // deadframe_types: decode_ref boxes TAGBOX ints
            rd_virtuals_slice,      // rd_virtuals
            rd_guard_pendingfields, // rd_guard_pendingfields
            Some(vrefinfo_dyn), // resume.py blackhole_from_resumedata metainterp_sd.virtualref_info
            vinfo_arg, // resume.py blackhole_from_resumedata self.jitdriver_sd.virtualizable_info
            None,      // resume.py blackhole_from_resumedata greenfield_info unused in pyre
            all_virtuals, // resume.py blackhole_from_resumedata GUARD_NOT_FORCED cache
            &allocator,
        )
    });

    let (mut bh, virtualizable_ptr) = bh;

    // resume.py: virtualizable_ptr was read by consume_vable_info
    // from the vable section. Set on the blackhole for vable bytecodes.
    // A novable resume runs no vable opcodes, so leave both the pointer and
    // the vinfo handle unset (null) instead of borrowing jd0's.
    if !novable {
        if virtualizable_ptr != 0 {
            bh.virtualizable_ptr = virtualizable_ptr;
        } else if deadframe.len() > 0 {
            // Fallback for guards without vable section.
            bh.virtualizable_ptr = deadframe.get(0);
        }
        bh.virtualizable_info = crate::eval::get_virtualizable_info();
    }
    // Last read of `deadframe`.  `blackhole_from_resumedata` has copied every
    // live value into the blackhole register banks, which the collector reaches
    // through their own root source (`walk_bh_regs`), so the off-heap copy is
    // dead from here on.  Release its roots BEFORE the forward run: `bh.run()`
    // executes the whole remainder of the resumed frame — for a module-level
    // frame that is the rest of the program — and every `Ref` cell left
    // registered pins the object it happened to hold at the guard, whether or
    // not the resumed code still uses it.  A `for v in gen(): break` leaves the
    // abandoned generator in such a cell, so it is never collected and its
    // `finally` never runs.  `blackhole.py resume_in_blackhole` ends
    // `deadframe`'s live range at `_prepare_resume_from_failure`, before
    // `_run_forever`, for the same reason.
    drop(caller_deadframe_roots);
    // resume.py builds the caller chain (`nextblackholeinterp`)
    // but does not set the virtualizable-info handle on each frame.  pyre
    // stores the vinfo per-`BlackholeInterpreter` (RPython reads it from
    // the field descriptor `blackhole.py fielddescr.get_vinfo()`, a
    // global), so every caller frame that runs a vable opcode after the
    // innermost frame returns to it needs the same handle.  Propagate it
    // down the whole chain, mirroring the forward-exec inheritance
    // `self.virtualizable_info = parent.virtualizable_info` performed when
    // a frame enters a callee.
    // blackhole.py get_portal_runner parity:
    //   jitdriver_sd = self.builder.metainterp_sd.jitdrivers_sd[jdindex]
    //   fnptr        = adr2int(jitdriver_sd.portal_runner_adr)
    //   calldescr    = jitdriver_sd.mainjitcode.calldescr
    //
    // The table keeps every driver at its own `jdindex`: a resumed jd1 or jd2
    // jitcode still carries that index, so a one-slot table would leave its
    // `handle_jitexc_from_bh` unreachable. Only jd0's slot is replaced.
    let jitdrivers_sd: std::sync::Arc<[majit_metainterp::blackhole::BhJitDriverSd]> = {
        let mut table = driver.meta_interp().staticdata.bh_jitdrivers_sd().to_vec();
        let jd0 = majit_metainterp::blackhole::BhJitDriverSd {
            result_type: majit_metainterp::blackhole::BhReturnType::Ref,
            portal_runner_ptr: Some(bh_portal_runner_c),
            handle_jitexc_from_bh: driver
                .meta_interp()
                .staticdata
                .jitdrivers_sd
                .first()
                .and_then(|jd| jd.handle_jitexc_from_bh),
            mainjitcode_calldescr: {
                // `get_portal_runner` returns this descr paired with
                // `bh_portal_runner_c`, and the blackhole recursive-portal resume
                // (`bhimpl_recursive_call_r` -> `bh_call_r`) verifies the merged
                // green+red argument banks against it — so it must describe the
                // portal runner, not the traced function.  `bh_portal_runner_c`
                // consumes the warmspot `portalfunc_ARGS` (`warmspot.py:972-975`):
                // greens `[next_instr:i, is_being_profiled:i, pycode:r]` + reds
                // `[frame:r, ec:r]` (`interp_jit.py:67-68`) = 2 int + 3 ref.
                // `bh.jitcode.calldescr` is the traced function's own frame-based
                // residual-call descr and does not describe the runner.
                // Fresh descr so `descr.py CallDescr.create_call_stub` sees
                // `"iirrr"` / `'r'`. Cloning `bh.jitcode.calldescr` and
                // rewriting those two fields would keep a stub selected for
                // the traced function (`BhCallDescr::clone` copies a resolved
                // `call_stub`). `from_arg_classes` leaves the stub unresolved
                // and keeps extra_info.
                majit_metainterp::blackhole::BhCallDescr::from_arg_classes(
                    "iirrr".to_string(),
                    'r',
                    bh.jitcode.calldescr.extra_info.clone(),
                )
            },
            // `get_portal_runner` reaches this table by INDEX
            // (`jitdrivers_sd[jdindex]`), so the driver never has to be found
            // by its portal jitcode here.  Leaving it unset keeps
            // `portal_jd_for` from claiming a frame for a synthetic driver
            // assembled around whichever jitcode happened to be innermost when
            // the chain was built.
        };
        match table.first_mut() {
            Some(slot) => *slot = jd0,
            None => table.push(jd0),
        }
        std::sync::Arc::from(table)
    };
    {
        let vinfo = bh.virtualizable_info;
        let mut current = Some(&mut *bh);
        while let Some(frame) = current {
            frame.virtualizable_info = vinfo;
            frame.record_caught_exception = Some(record_caught_blackhole_traceback);
            // Upstream resolves `jitdrivers_sd[jdindex]` through
            // `self.builder.metainterp_sd` (blackhole.py:1079, :1096), so every
            // frame of a chain shares one table.  `BlackholeInterpBuilder`
            // carries that snapshot and `acquire_interp` hands it out, but this
            // path derives the entry from the innermost frame's own jitcode and
            // so cannot seed the builder before `blackhole_from_resumedata`
            // acquires the chain.  Publish it to every frame instead: any frame
            // above the bottom one reaches `bhimpl_jit_merge_point`'s
            // recursive-portal branch, which indexes the table directly.
            frame.jitdrivers_sd = std::sync::Arc::clone(&jitdrivers_sd);
            current = frame.nextblackholeinterp.as_deref_mut();
        }
    }

    // Portal red-arg registers (`pypy/module/pypyjit/interp_jit.py:67
    // reds = ['frame', 'ec']`) are filled per-frame by
    // `consume_one_section` from each section's `-live-` op (resume.py
    // :1381 / `_prepare_next_section` resume.py). RPython's vable
    // bytecodes use that live frame red directly. Pyre's blackhole vable
    // handlers instead address `BlackholeInterpreter::virtualizable_ptr`,
    // so bind that Pyre-only cache from EACH frame's own red register.
    {
        let multi_frame = bh.nextblackholeinterp.is_some();
        let mut current = Some(&mut *bh);
        let mut frame_index = 0usize;
        while let Some(frame) = current {
            if majit_metainterp::bh_debug_enabled() {
                let raw_code =
                    pyre_jit_trace::state::raw_code_for_jitcode_index(frame.jitcode.index() as i32);
                let qualname = raw_code
                    .map(|raw| unsafe { &*raw }.qualname.as_str())
                    .unwrap_or("<null>");
                let py_pc = pyre_jit_trace::py_coord::containing_py_pc_for_jitcode_pc_public(
                    frame.jitcode.index() as i32,
                    frame.position as i32,
                );
                eprintln!(
                    "[bh-chain] frame={frame_index} jitcode={} qualname={qualname} position={} py_pc={py_pc:?}",
                    frame.jitcode.index(),
                    frame.position,
                );
            }
            // The outer/root frame's standard virtualizable identity comes
            // from `consume_vable_info` above. Preserve it for a single-frame
            // resume. In a multi-frame snapshot that one vable section names
            // only the outer portal frame, so even the innermost blackhole
            // must instead bind its own frame red.
            if (multi_frame || frame.virtualizable_ptr == 0)
                && let Some(jitcode_index) = frame.jitcode.try_index()
            {
                let (frame_reg, _) =
                    pyre_jit_trace::state::portal_red_regs_at(jitcode_index as i32);
                // #1311: the admission below is a bare `!= 0`, and a word that
                // is neither zero nor a frame has reached it — one dump held an
                // aligned heap address that was not a `PyFrame`, another the
                // tagged word `2`, and `record_application_traceback` then
                // dereferenced it.  Which frame of the chain contributes the bad
                // red, and whether the register is the wrong one for this
                // jitcode or was simply never filled, are questions the whole
                // bank answers and the admitted value alone does not, so report
                // it beside the index read from it.  A report and not a test:
                // rejecting the word here would exchange the fault for a
                // silently missing traceback node while the wrong word kept
                // flowing.
                if majit_metainterp::bh_debug_enabled() {
                    eprintln!(
                        "[bh-chain] frame={frame_index} jitcode={jitcode_index} \
                         frame_reg={frame_reg} registers_r={:x?}",
                        frame.registers_r,
                    );
                }
                if frame_reg != u16::MAX {
                    if let Some(&frame_ptr) = frame.registers_r.get(frame_reg as usize) {
                        if frame_ptr != 0 {
                            frame.virtualizable_ptr = frame_ptr;
                        }
                    }
                }
            }
            current = frame.nextblackholeinterp.as_deref_mut();
            frame_index += 1;
        }
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!("[blackhole-resume] rd_numb path, chain built, running _run_forever",);
    }

    // #326: no rollback snapshot is taken across `bh.run()`.  The blackhole
    // commits every STORE_FAST / operand push to the virtualizable heap frame
    // as it runs forward, and an abort used to restore that frame to the
    // guard snapshot so the interpreter's re-run from the guard resume PC
    // "applied each side effect exactly once".  That reasoning only covered
    // frame-local state: a `print`, a heap store or a user frame the blackhole
    // already executed has no undo, so rewinding re-applied every one of them
    // (`x = [1,2]; <hot loop>; x.append(3); class C: pass` left `[1,2,3,3]`).
    // The only abort that can reach the blackhole at runtime sits on a Python
    // opcode boundary — `abort` is gated out of blackhole dispatch by
    // `PyJitCode::has_abort_opcode`, leaving `abort_permanent` (a whole
    // unsupported opcode) and the two declined-call rejections (whose call
    // never ran) — so the frame the blackhole leaves behind is a valid resume
    // point and the interpreter continues from it.  This is
    // `convert_and_run_from_pyjitpl`'s shape (`blackhole.py`): each
    // frame resumes at its own current pc, never rewound.

    // blackhole.py resume_in_blackhole:
    //   current_exc = _prepare_resume_from_failure(guard_opnum, deadframe)
    //   _run_forever(blackholeinterp, current_exc)
    // `_resume_mainloop` prologue (blackhole.py): when the guard
    // failure carries a pending exception (GUARD_NO_EXCEPTION /
    // GUARD_EXCEPTION / GUARD_NOT_FORCED → `cpu.grab_exc_value(deadframe)`),
    // hand it to the resumed frame before running any bytecode so an
    // exception guard unwinds to its `catch_exception` handler instead of
    // resuming the no-exception path. `_run_forever` re-offers the exception
    // to each caller in turn (resume_mainloop returns the unhandled exc, the
    // loop advances to nextblackholeinterp), so walk the chain here.
    if guard_exc != 0 {
        loop {
            // The traceback records below allocate their nodes through the
            // collecting nursery hook and can move the exception; the park
            // holds its forwarded address, `guard_exc` the pre-move copy.
            let guard_exc = guard_exc_root.get();
            if bh.handle_exception_in_frame(guard_exc) {
                // Handler found in this frame; `position` now points at it.
                // Fall through to the run loop to execute the handler.
                break;
            }
            // blackhole.py no handler here → propagate to the caller.
            let next = bh.nextblackholeinterp.take();
            let frame_ptr = bh.virtualizable_ptr as *mut PyFrame;
            let jitcode_index = bh.jitcode.try_index().map(|v| v as i32);
            let last_opcode_position = bh.last_opcode_position;
            leave_resumed_blackhole_frame(&bh, true);
            release_bh_rd(bh);
            match next {
                Some(caller) => {
                    // `_run_forever` records a node at every frame boundary it
                    // propagates through, so a frame the chain merely passes
                    // over still appears in the traceback.  Only the bottommost
                    // frame was recorded here, which drops the node of every
                    // level between the raise and the outermost frame — for an
                    // inlined callee that is the whole of its presence, since
                    // no interpreter frame ever runs for it.  The helper
                    // resolves the raise coordinate and honours a bare reraise,
                    // as it does for the sibling propagation loop.
                    if !frame_ptr.is_null() {
                        // Re-read past `leave_resumed_blackhole_frame`.
                        let guard_exc = guard_exc_root.get();
                        match jitcode_index {
                            Some(jitcode_index) => record_caught_blackhole_traceback(
                                guard_exc,
                                frame_ptr as i64,
                                i64::from(jitcode_index),
                                last_opcode_position as i64,
                            ),
                            None => unsafe {
                                pyre_interpreter::pytraceback::record_application_traceback(
                                    guard_exc as PyObjectRef,
                                    frame_ptr,
                                    (*frame_ptr).last_instr as i64,
                                )
                            },
                        }
                    }
                    bh = caller;
                }
                None => {
                    // blackhole.py:1629 bottommost frame, unhandled →
                    // raise ExitFrameWithExceptionRef(exc).
                    let mut err = exit_frame_exception_ref(
                        guard_exc_root.get(),
                        "guard_exc propagation",
                        || {
                            format!(
                                "jitcode={jitcode_index:?} \
                                 last_opcode_position={last_opcode_position}"
                            )
                        },
                    );
                    if !frame_ptr.is_null() {
                        match jitcode_index {
                            // Every indexed frame recorded during blackhole
                            // propagation goes through the guarded recorder;
                            // `pyopcode.py handle_operation_error` is
                            // upstream's single traceback attach point.
                            Some(jitcode_index) => record_caught_blackhole_traceback(
                                err.exc_object as i64,
                                frame_ptr as i64,
                                i64::from(jitcode_index),
                                last_opcode_position as i64,
                            ),
                            None => unsafe {
                                pyre_interpreter::pytraceback::record_application_traceback(
                                    err.exc_object,
                                    frame_ptr,
                                    (*frame_ptr).last_instr as i64,
                                );
                            },
                        }
                        // `err` was built from the parked exception, which
                        // the record above can move.
                        err.set_exc_object(guard_exc_root.get() as PyObjectRef);
                    }
                    // `guard_exc` is now owned by the typed
                    // `ExitFrameWithExceptionRef` result. A can-raise
                    // residual helper published the same value to both the
                    // backend exception cell and pyre's blackhole adapter;
                    // the guard supplied the former as `guard_exc`, so the
                    // latter must be consumed at this terminal handoff.
                    // RPython has only the deadframe/exception-result
                    // transfer here, not a second TLS owner.
                    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
                    return BlackholeResult::ExitFrameWithExceptionRef(err);
                }
            }
        }
    }

    // blackhole.py _run_forever parity.
    loop {
        // `_resume_mainloop` can hand an unhandled exception to a suspended
        // caller by setting `got_exception` below. Propagate that state before
        // dispatching the caller: its current position is the normal post-call
        // continuation and the call result is deliberately garbage on raise.
        // Running even one opcode first consumes that garbage as a successful
        // return and replaces the real exception with a secondary TypeError.
        let run_outcome = (!bh.got_exception).then(|| bh.run());
        // A frame that ran out of bytecode never reached a `*_return` handler,
        // so `return_type` and `tmpreg_*` still hold whatever the previous
        // occupant of this pooled interpreter left there — and the tail of this
        // loop publishes exactly those two to the caller.  `blackhole.py`'s
        // `dispatch_loop` cannot reach that state at all: it is `while True:`
        // over `ord(code[position])`, so reading past the end raises.
        if matches!(
            run_outcome,
            Some(majit_metainterp::blackhole::BhRunOutcome::EndOfCode)
        ) {
            let name = bh.jitcode.name().to_string();
            let index = bh.jitcode.try_index();
            release_bh_rd(bh);
            panic!(
                "blackhole_resume_via_rd_numb: jitcode {name:?} index {index:?} ended \
                 without a return opcode, so there is no result to pass to the caller"
            );
        }
        if let Some(majit_metainterp::blackhole::BhRunOutcome::ContinueRunningNormally(args)) =
            run_outcome
        {
            // blackhole.py:1068: raise ContinueRunningNormally(*args)
            //
            // The blackhole reached a merge point with no pending exception
            // (an unhandled raise would have propagated through the exception
            // path above, not reached `run()`'s ContinueRunningNormally).  A
            // residual call that raised AND was caught in-frame
            // (`check_residual_call_exception_after` → `route_to_catch`) cleared
            // only `BH_LAST_EXC_VALUE`; the backend `_store_exception` cells
            // (`store_jit_exception` writes BOTH — see
            // `publish_residual_call_exception`) keep the consumed exception.
            // Re-entering compiled code via this ContinueRunningNormally would
            // then have its first `GUARD_NO_EXCEPTION` read the stale cell as a
            // spurious pending exception and deopt at a coordinate with no
            // handler (the loop header), escaping the original try-block.  Drain
            // the backend cells here so the re-entry starts pristine, mirroring
            // the walker's `execute_raised` drain (`drain_backend_jit_exc`).
            drain_backend_jit_exc();
            let frame_ptr = bh.virtualizable_ptr as *mut PyFrame;

            let mut red_ref: Vec<PyObjectRef> =
                args.red_ref.iter().map(|&v| v as PyObjectRef).collect();
            if red_ref.is_empty() {
                red_ref.push(frame_ptr as PyObjectRef);
            }
            return BlackholeResult::ContinueRunningNormally {
                green_int: args.green_int,
                green_ref: args.green_ref.iter().map(|&v| v as PyObjectRef).collect(),
                green_float: args
                    .green_float
                    .iter()
                    .map(|&v| f64::from_bits(v as u64))
                    .collect(),
                red_int: args.red_int,
                red_ref,
                red_float: args
                    .red_float
                    .iter()
                    .map(|&v| f64::from_bits(v as u64))
                    .collect(),
            };
        }
        if !bh.got_exception && bh.aborted {
            if bh.abort_permanent_bail {
                release_bh_rd(bh);
                return BlackholeResult::BailToInterpreter;
            }
            let position = bh.position;
            let last_opcode_position = bh.last_opcode_position;
            release_bh_rd(bh);
            panic!(
                "blackhole_resume_via_rd_numb: canonical blackhole aborted at \
                 position {position} (last opcode {last_opcode_position})"
            );
        }
        if bh.got_exception {
            // `run()` brackets `exception_last_value` with `push_bh_regs` and
            // pops that bracket on return, so from here the propagating
            // exception lives only in a Rust local — and this block
            // allocates: `record_application_traceback` builds a traceback
            // node, and so does the caller's `handle_exception_in_frame` →
            // `route_to_catch`.  The exception is a nursery object
            // (`w_exception_new_empty` → `alloc_exception_nursery`,
            // `malloc_fixedsize` → `collect_and_reserve`), so that allocation
            // moves it.  An unrooted local keeps the pre-move address, which
            // is what hands `exit_frame_exception_ref` a mapped object of the
            // wrong class.
            // `blackhole.py _run_forever` keeps every interp in the chain
            // — and with it `exception_last_value` — transitively traced for
            // its whole life; root the value across the equivalent window.
            let _exc_roots = pyre_object::gc_roots::push_roots();
            let exc_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(bh.exception_last_value as PyObjectRef);
            let exc_value = pyre_object::gc_roots::shadow_stack_get(exc_slot) as i64;
            let next = bh.nextblackholeinterp.take();
            let frame_ptr = bh.virtualizable_ptr as *mut PyFrame;
            let jitcode_index = bh.jitcode.try_index().map(|v| v as i32);
            let last_opcode_position = bh.last_opcode_position;
            let bh_entry_position = bh.entry_position;
            let bh_opcode_at = bh.jitcode.code.get(last_opcode_position).copied();
            let bh_jitcode_name = bh.jitcode.name().to_string();
            // For `BC_RAISE` (`raise/r`) the following byte is the register the
            // exception was read out of, so record which slot of which bank
            // produced the value.
            let bh_raise_reg = bh.jitcode.code.get(last_opcode_position + 1).copied();
            let bh_regs_r_len = bh.registers_r.len();
            // A copy, not a probe: a ref register may hold a raw non-object
            // word (a force token, an erased unboxed buffer), so anything that
            // dereferences these belongs on the failure path only.
            //
            // The copy has to be taken here because `release_bh_rd` below ends
            // the borrow before the diagnostic closure runs, but it is only
            // ever read when `exit_frame_exception_ref` rejects the value, so
            // the same screen decides whether to pay for it. A move between
            // here and the closure does not change the answer: the class of the
            // propagating value is what is being screened.
            let bh_regs_r = if unsafe {
                pyre_object::interp_exceptions::w_exception_kind_checked(exc_value as PyObjectRef)
            }
            .is_none()
            {
                bh.registers_r.clone()
            } else {
                Vec::new()
            };
            let bh_code_window: Vec<u8> = bh
                .jitcode
                .code
                .get(last_opcode_position.saturating_sub(10)..=last_opcode_position + 1)
                .unwrap_or(&[])
                .to_vec();
            let keeps_existing_traceback = bh_opcode_at
                .is_some_and(|opcode| opcode == majit_metainterp::jitcode::insns::BC_RERAISE)
                || jitcode_index.is_some_and(|index| {
                    pyre_jit_trace::state::jitcode_pc_raise_keeps_existing_traceback(
                        index,
                        last_opcode_position as i32,
                    )
                });
            // blackhole.py _run_forever propagates an unhandled exception one
            // BlackholeInterpreter frame at a time. Record each exiting
            // frame before advancing to nextblackholeinterp, matching
            // pytraceback.py record_application_traceback at every Python
            // frame boundary. A forwarding raise preserves the existing chain.
            if !keeps_existing_traceback && !frame_ptr.is_null() {
                let exc_value = pyre_object::gc_roots::shadow_stack_get(exc_slot) as i64;
                if let Some(jitcode_index) = jitcode_index {
                    record_caught_blackhole_traceback(
                        exc_value,
                        frame_ptr as i64,
                        i64::from(jitcode_index),
                        last_opcode_position as i64,
                    );
                } else {
                    unsafe {
                        pyre_interpreter::pytraceback::record_application_traceback(
                            exc_value as PyObjectRef,
                            frame_ptr,
                            (*frame_ptr).last_instr as i64,
                        );
                    }
                }
            }
            leave_resumed_blackhole_frame(&bh, true);
            release_bh_rd(bh);
            // Re-read through the pin: the records above are allocation
            // points, so the pinned slot — not the stale local — is the live
            // value from here on.
            let exc_value = pyre_object::gc_roots::shadow_stack_get(exc_slot) as i64;
            let Some(mut caller_bh) = next else {
                let err =
                    exit_frame_exception_ref(exc_value, "blackhole exception propagation", || {
                        format!(
                            "jitcode={jitcode_index:?} name={bh_jitcode_name} \
                             entry_position={bh_entry_position} \
                             last_opcode_position={last_opcode_position} opcode={:?} \
                             operand_reg={bh_raise_reg:?} registers_r.len={bh_regs_r_len} \
                             regs_holding_exception={:?} \
                             code[-10..]={bh_code_window:?} \
                             keeps_existing_traceback={keeps_existing_traceback} \
                             guard_exc={guard_exc:x} py_pc={:?} entry_py_pc={:?} \
                             deadframe_types={deadframe_types:?} deadframe={deadframe:x?} \
                             registers_r={bh_regs_r:x?}",
                            bh_opcode_at,
                            // Which ref registers hold a real
                            // `W_BaseException`: a hit in a slot other than
                            // `operand_reg` means the operand index is wrong,
                            // no hit anywhere means the bank never received
                            // the exception.
                            bh_regs_r
                                .iter()
                                .enumerate()
                                .filter(|(_, v)| unsafe {
                                    pyre_object::interp_exceptions::w_exception_kind_checked(
                                        **v as PyObjectRef,
                                    )
                                    .is_some()
                                })
                                .map(|(i, _)| i)
                                .collect::<Vec<_>>(),
                            jitcode_index.and_then(|index| {
                                pyre_jit_trace::py_coord::containing_py_pc_for_jitcode_pc_public(
                                    index,
                                    last_opcode_position as i32,
                                )
                            }),
                            jitcode_index.and_then(|index| {
                                pyre_jit_trace::py_coord::containing_py_pc_for_jitcode_pc_public(
                                    index,
                                    bh_entry_position as i32,
                                )
                            }),
                        )
                    });
                // `attach_tb` deliberately stays true on the way out.  It
                // suppresses the traceback record of the re-raising frame
                // only, and this frame's decision was already applied above by
                // skipping its own record; `handle_exception` clears the flag
                // for that same reason once a frame is done with it.  Clearing
                // it here hands the suppression one frame further instead: the
                // compiled frame never runs `handle_exception` itself, so the
                // interpreter caller is the first frame to read the flag and
                // would drop its own node.
                //
                // A residual helper can publish a raise to both exception
                // channels.  Blackhole propagation has consumed its own
                // channel into `ExitFrameWithExceptionRef`; keep the backend
                // cell in sync before the outer interpreter receives it.
                drain_backend_jit_exc();
                return BlackholeResult::ExitFrameWithExceptionRef(err);
            };
            caller_bh.last_opcode_position = caller_bh.position;
            if caller_bh.handle_exception_in_frame(exc_value) {
                bh = caller_bh;
                continue;
            }
            // `handle_exception_in_frame` records a traceback node on the way
            // through, so take the pinned value again for the store.
            caller_bh.exception_last_value =
                pyre_object::gc_roots::shadow_stack_get(exc_slot) as i64;
            caller_bh.got_exception = true;
            bh = caller_bh;
            continue;
        }

        // blackhole.py:1632-1644: pass return value to caller by _return_type.
        use majit_metainterp::blackhole::BhReturnType;
        let rt = bh.return_type;
        let next = bh.nextblackholeinterp.take();
        let caller = next;
        if caller.is_none() {
            // blackhole.py _done_with_this_frame
            // The terminal level has no caller branch below to close its
            // execution scope. Root the result across ExecutionContext.leave,
            // whose caller-vref force can collect.
            let roots = pyre_object::gc_roots::push_roots();
            if matches!(rt, BhReturnType::Ref) {
                let _ = roots.pin_root(bh.get_tmpreg_r() as PyObjectRef);
            }
            leave_resumed_blackhole_frame(&bh, false);
            let result = match rt {
                BhReturnType::Void => BlackholeResult::DoneWithThisFrameVoid,
                BhReturnType::Int => BlackholeResult::DoneWithThisFrameInt(bh.get_tmpreg_i()),
                BhReturnType::Ref => BlackholeResult::DoneWithThisFrameRef(roots.get(roots.base())),
                BhReturnType::Float => BlackholeResult::DoneWithThisFrameFloat(f64::from_bits(
                    bh.get_tmpreg_f() as u64,
                )),
            };
            release_bh_rd(bh);
            return result;
        }
        let mut caller_bh = caller.unwrap();
        // blackhole.py:1637-1644: dispatch by _return_type
        match rt {
            BhReturnType::Int => caller_bh.setup_return_value_i(bh.get_tmpreg_i()),
            BhReturnType::Ref => caller_bh.setup_return_value_r(bh.get_tmpreg_r()),
            BhReturnType::Float => caller_bh.setup_return_value_f(bh.get_tmpreg_f()),
            BhReturnType::Void => {}
        }
        leave_resumed_blackhole_frame(&bh, false);
        release_bh_rd(bh);
        bh = caller_bh;
    }
}

/// warmspot.py handle_jitexception parity.
///
/// RPython captures result_kind in closure (warmspot.py). For pyre,
/// portal result_type == REF (warmspot.py), so ALL CALL_ASSEMBLER
/// ops use _R. The result is always a Ref (PyObjectRef).
fn handle_blackhole_result(bh_result: BlackholeResult) -> Option<i64> {
    match bh_result {
        // warmspot.py:985-987: DoneWithThisFrameVoid → return None
        BlackholeResult::DoneWithThisFrameVoid => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] DoneWithThisFrameVoid");
            }
            Some(0)
        }
        // warmspot.py:988-990: DoneWithThisFrameInt → box to Ref.
        // Portal result_type == REF, so blackhole should normally raise
        // DoneWithThisFrameRef. This path handles edge cases.
        BlackholeResult::DoneWithThisFrameInt(v) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[blackhole-resume] DoneWithThisFrameInt({}) → box to Ref",
                    v
                );
            }
            Some(w_int_new(v) as i64)
        }
        // warmspot.py:991-993: DoneWithThisFrameRef → return e.result
        BlackholeResult::DoneWithThisFrameRef(r) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] DoneWithThisFrameRef({:?})", r);
            }
            Some(r as i64)
        }
        // warmspot.py:994-996: DoneWithThisFrameFloat → return e.result
        BlackholeResult::DoneWithThisFrameFloat(f) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] DoneWithThisFrameFloat({})", f);
            }
            Some(f.to_bits() as i64)
        }
        // warmspot.py:998-1005: ExitFrameWithExceptionRef → raise value.
        BlackholeResult::ExitFrameWithExceptionRef(mut err) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] ExitFrameWithExceptionRef → raise");
            }
            let exc_obj = err.to_exc_object();
            if exc_obj != pyre_object::PY_NULL {
                // Symmetric with the `ContinueRunningNormally` arm's
                // `portal_body_result` error fall-through below and with
                // `lib.rs::jit_exc_raise` — every backend's blackhole resume
                // publishes the pending exception, not just cranelift.
                //
                // Both carriers, because `handle_jitexception` reaches this arm
                // by `raise value` (warmspot.py) into RPython's single
                // ll exception state, which is caller-blind: the compiled
                // `CALL_ASSEMBLER` reads it through the backend cells and the
                // walker's trace-time `execute_varargs_call` reads it through
                // `BH_LAST_EXC_VALUE`.  `clear_residual_call_exception` empties
                // both, so arming both cannot leave one behind.
                publish_residual_call_exception(exc_obj);
            }
            Some(0) // garbage return — GUARD_NO_EXCEPTION will fire
        }
        // warmspot.py:970-983: ContinueRunningNormally → portal_ptr(*args).
        BlackholeResult::ContinueRunningNormally {
            green_int,
            green_ref,
            green_float,
            red_int,
            red_ref,
            red_float,
        } => {
            // warmspot.py:972-975: portalfunc_ARGS extraction.
            // Build merged arg lists: all_i = gi + ri, all_r = gr + rr, all_f = gf + rf.
            // warmstate.py unspecialize_value: Ref→GCREF(i64), Float→FLOATSTORAGE(i64).
            let mut all_i = green_int;
            all_i.extend(&red_int);
            let mut all_r: Vec<i64> = green_ref.iter().map(|r| *r as i64).collect();
            all_r.extend(red_ref.iter().map(|r| *r as i64));
            let mut all_f: Vec<i64> = green_float.iter().map(|f| f.to_bits() as i64).collect();
            all_f.extend(red_float.iter().map(|f| f.to_bits() as i64));
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[blackhole-resume] ContinueRunningNormally → portal_ptr(all_i={:?}, all_r=[{:#x?}])",
                    all_i, all_r,
                );
            }
            // warmspot.py:976-1005: portal_ptr(*args), and if it raises a
            // regular exception propagate it like ExitFrameWithExceptionRef
            // instead of collapsing it to a null Ref.
            let next_instr = match all_i.first().copied() {
                None => Some(0),
                Some(pc) => crate::eval::green_pc_position(pc),
            };
            let pycode = all_r.first().copied().unwrap_or(0) as PyObjectRef;
            let frame_ptr = all_r.get(1).copied().unwrap_or(0) as *mut PyFrame;
            let ec =
                all_r.get(2).copied().unwrap_or(0) as *const pyre_interpreter::PyExecutionContext;
            if frame_ptr.is_null() {
                return Some(pyre_object::PY_NULL as i64);
            }
            let frame = unsafe { &mut *frame_ptr };
            // The red frame is the activation identity.  PyPy forwards the
            // CRN arguments to `portal_ptr` without mutating that frame; pyre's
            // frame-only portal entry likewise reads the pycode / execution
            // context is the separate red argument. Do not overwrite pycode
            // from a possibly stale green snapshot.
            if majit_metainterp::majit_log_enabled()
                && !pycode.is_null()
                && frame.pycode != pycode as *const ()
            {
                eprintln!(
                    "[blackhole-resume] CALL_ASSEMBLER CRN/frame identity mismatch: green_pycode={pycode:p} frame_pycode={:p} red_ec={ec:p}",
                    frame.pycode,
                );
            }
            if let Some(next_instr) = next_instr {
                frame.set_last_instr_from_next_instr(next_instr);
                // The blackhole wrote the failing guard's recorded operand depth
                // into the frame; resuming at the merge-point `next_instr` (a
                // different pc) would carry that over-count and overflow the frame
                // at its peak stack use.  Re-derive the depth from the resume pc —
                // the CALL_ASSEMBLER-path mirror of the eval.rs CRN handoff.
                crate::eval::correct_resume_vsd(frame, next_instr);
            }
            let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
            if !ec.is_null() {
                pyre_interpreter::call::set_last_exec_ctx(ec);
            }
            let result = crate::eval::portal_body_result(frame);
            pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
            match result {
                Ok(result) => Some(result as i64),
                Err(mut err) => {
                    let exc_obj = err.to_exc_object();
                    if exc_obj != pyre_object::PY_NULL {
                        majit_metainterp::blackhole::BH_LAST_EXC_VALUE
                            .with(|c| c.set(exc_obj as i64));
                        store_jit_exception(exc_obj as i64);
                    }
                    Some(0)
                }
            }
        }
        BlackholeResult::BailToInterpreter => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] BailToInterpreter");
            }
            None
        }
    }
}

/// Derive the (`green_key`, `trace_id`, `fail_index`) bridge-source identity
/// strictly from the failing guard's descr Arc.
///
/// `pyjitpl.py handle_guard_failure(self, resumedescr, deadframe)`
/// reads identity from `resumedescr` directly: `resumedescr.rd_loop_token
/// .loop_token_wref()` (line 2897) yields the owning `JitCellToken`,
/// `resumedescr.get_resumestorage()` (line 2893) yields the `ResumeGuardDescr`
/// carrying the per-trace `fail_index`, and `record_loop_or_bridge`
/// (`compile.py`) stamps `trace_id` onto each `ResumeDescr`.
/// Pyre's `descr_owning_jct(arc).green_key`, `arc.fail_index_per_trace()`
/// and `arc.trace_id()` are the line-by-line equivalents — together they
/// constitute the canonical bridge-source identity.
///
/// Returns `None` when the owning loop's `JitCellToken` weakref is dead
/// (the loop was evicted by the memory manager): RPython raises
/// `compile.giveup()` from `_trace_and_compile_from_bridge`
/// (`compile.py:725-729`) in the same case, falling through to
/// `resume_in_blackhole`.  Pyre's caller signals the same intent by
/// returning `false` to drop into blackhole resume.
// dont_look_inside: bridge-compile identity machinery the tracer must not enter.
#[majit_macros::dont_look_inside]
pub(crate) fn bridge_source_identity_from_descr(
    descr_arc: &std::sync::Arc<dyn majit_ir::Descr>,
) -> Option<(u64, u64, u32)> {
    let descr_fd = descr_arc.as_fail_descr()?;
    let green_key = majit_backend::descr_owning_jct(descr_fd)?.green_key();
    let trace_id = descr_fd.trace_id();
    let fail_index = descr_fd.fail_index_per_trace();
    Some((green_key, trace_id, fail_index))
}

/// Outcome of `trace_and_compile_from_bridge`.
///
/// `pyjitpl.py handle_guard_failure` never returns — it raises
/// `ContinueRunningNormally` (bridge attached, resume in compiled code),
/// switches to the blackhole, or raises `DoneWithThisFrame` when
/// `interpret()` runs the resumed frames forward to a `Finish`.  Pyre
/// reifies those three exits so the caller can act on each.
pub enum BridgeResolution {
    /// Bridge attached — caller re-enters compiled code
    /// (`ContinueRunningNormally`).  The legacy `true` return.
    CompiledContinue,
    /// Resume the region in the blackhole interpreter
    /// (`SwitchToBlackhole`).  The legacy `false` return.
    ResumeBlackhole,
    /// The single-frame bridge walk ran the resumed frame forward to a
    /// `Finish` (`Terminate`) and captured its concrete return value —
    /// `pyjitpl.py` `interpret()` raising `DoneWithThisFrame` from the
    /// post-walk state.  The caller returns this value directly instead of
    /// rewinding the live frame and re-running the region (#177).
    Finished(pyre_jit_trace::state::ConcreteValue),
    /// Uncaught exception leaving the resumed frames. Either the walk's
    /// `FinishConcrete::Raise`, or `prepare_resume_from_failure`
    /// returning `ExitFrameWithExceptionRef` after
    /// `finishframe_exception` drained the framestack
    /// (`jitexc.py` `ExitFrameWithExceptionRef`). The caller propagates
    /// it and does not resume the frame.
    FinishedException(pyre_jit_trace::state::ConcreteValue),
}

/// Map the legacy bool bridge outcome (`true` = continue in compiled code,
/// `false` = resume in the blackhole) onto [`BridgeResolution`].
fn bridge_resolution_from_bool(compiled_continue: bool) -> BridgeResolution {
    if compiled_continue {
        BridgeResolution::CompiledContinue
    } else {
        BridgeResolution::ResumeBlackhole
    }
}

/// `MAJIT_BRIDGE_BAIL=<stage>` (diagnostic): resume in the blackhole from a
/// chosen point inside the guard-failure entry, so a wrong answer that appears
/// only with bridges on can be attributed to one prefix of what the attempt
/// does.  `MAJIT_NO_BRIDGE` and `MAJIT_MAX_BRIDGES` name WHICH bridge; this
/// names which STEP of it.  1 = before `decode_and_restore_guard_failure`, so
/// nothing the attempt owns has run; 2 = after it, so the resume data has been
/// decoded, its pending fields replayed and the virtualizable synchronised;
/// 3 = after the frame's `last_instr` is repointed, so only the walk is
/// missing.  Stage 1 is the control -- it has to reproduce `MAJIT_NO_BRIDGE=1`
/// exactly, and a stage that does not is measuring the build rather than the
/// step.
fn bridge_bail_stage() -> u32 {
    static STAGE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *STAGE.get_or_init(|| {
        majit_metainterp::jit_env::env_var("MAJIT_BRIDGE_BAIL")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Frontend compiled-loop metadata for the loop a failing guard belongs to.
///
/// `pyjitpl.py handle_guard_failure` / `compile.py _trace_and_compile_from_bridge`
/// read `rd_loop_token.outermost_jitdriver_sd`. `get_compiled_meta` keys
/// `compiled_loops` by leftover `active_jitdriver_sd`.
pub(crate) fn compiled_meta_for_bridge_source<M: Clone>(
    meta_interp: &majit_metainterp::MetaInterp<M>,
    descr_arc: &std::sync::Arc<dyn majit_ir::Descr>,
    green_key: u64,
) -> Option<std::sync::Arc<M>> {
    let jd_no = descr_arc
        .as_fail_descr()
        .and_then(majit_backend::descr_owning_jct)
        .and_then(|token| token.outermost_jitdriver_index)
        .unwrap_or(0);
    meta_interp
        .get_compiled_meta_on_driver(jd_no, green_key)
        .cloned()
}

/// compile.py (_trace_and_compile_from_bridge):
/// Called when a guard failure reaches the trace_eagerness threshold.
/// Traces the alternative path from the guard failure point and compiles
/// a bridge.
///
/// pyjitpl.py handle_guard_failure:
///   initialize_state_from_guard_failure(resumedescr, deadframe)
///   prepare_resume_from_failure(deadframe, inputargs, resumedescr, excdata)
///   self.interpret()
///
/// The tracing loop mirrors pyjitpl.py interpret(): execute bytecodes
/// from the guard failure PC until a Finish (return) or CloseLoop
/// (back-edge to loop header) is reached.
/// compile.py _trace_and_compile_from_bridge parity.
///
/// Returns [`BridgeResolution`]: `CompiledContinue` when the bridge was
/// compiled and attached, `ResumeBlackhole` on a trace abort / start
/// failure so the caller falls through to resume_in_blackhole (RPython
/// pyjitpl.py SwitchToBlackhole → run_blackhole_interp_to_cancel_tracing),
/// or `Finished(cv)` when a single-frame walk ran forward to a `Finish`
/// (`interpret()` raising `DoneWithThisFrame`).
///
/// `allow_finish_direct_return` gates the walk's `Finished` shortcut:
/// only the general guard path (`eval::handle_fail`) consumes a concrete
/// walk result directly. The CALL_ASSEMBLER callback passes `false`, and
/// a walk that reaches `Terminate` with a finish-concrete stash returns
/// `ResumeBlackhole` after publishing `CA_WALK_FINISHED_FRAME`.
/// `prepare_resume_from_failure`'s `ExitFrameWithExceptionRef` is not
/// that shortcut: it returns `FinishedException` on every caller, before
/// the walk, and the caller propagates `jitexc.ExitFrameWithExceptionRef`.
// dont_look_inside: bridge-compile machinery the tracer must not enter.
#[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
#[majit_macros::dont_look_inside]
pub fn trace_and_compile_from_bridge(
    // pyjitpl.py `handle_guard_failure(self, resumedescr, deadframe)`
    // threads `resumedescr` (the descr) as the canonical identity source
    // through the entire bridge tracer.  Pyre's backend FailDescr Arc
    // plays the same role: `descr_owning_jct(arc).green_key` (mirroring
    // `pyjitpl.py MetaInterp.handle_guard_failure resumedescr.rd_loop_token.loop_token_wref()`),
    // `arc.fail_index_per_trace()` (mirroring `compile.py:854
    // ResumeGuardDescr._attrs_`), and `arc.trace_id()` (mirroring the
    // `compile.py record_loop_or_bridge` stamp) are the line-by-
    // line readers.  Both production callers — the general guard path
    // (eval.rs `handle_fail`) and the CALL_ASSEMBLER bridge entry
    // (call_jit.rs `jit_ca_handle_guard_failure`) — have an Arc available
    // before reaching this function (both backends recover it from
    // `FailDescrCell` thin pointers via
    // `Backend::fail_descr_arc_from_addr`), so this function takes only
    // the descr Arc and derives `(green_key, trace_id, fail_index)`
    // itself.
    descr_arc: &std::sync::Arc<dyn majit_ir::Descr>,
    frame: &mut PyFrame,
    raw_values: &[i64],
    exit_layout: &majit_metainterp::CompiledExitLayout,
    // `cpu.grab_exc_value(deadframe)` (llmodel.py): the pending
    // exception this guard failure carries, or 0. Threaded so the bridge
    // tracer can decline a pending-exception resume at a non-exception
    // guard (the `pending_exc` decline below).
    guard_exc: i64,
    // Whether the caller can consume a walk `Finished(cv)` directly (the
    // general guard path can; the CALL_ASSEMBLER callback cannot, and
    // leaves that stash for `ca_complete_after_bridge_walk`). Does not
    // gate `prepare_resume_from_failure`'s `ExitFrameWithExceptionRef`,
    // which returns `FinishedException` before the walk.
    allow_finish_direct_return: bool,
) -> BridgeResolution {
    /// Publish the live callee frame's address to the handshake cells the CA
    /// slow path's back-to-back blackhole hook reads
    /// (`jit_blackhole_resume_from_guard`). Always re-read through
    /// `FrameRoot::frame`, i.e. off the shadow stack, and always at a return
    /// site: the walk that decides these flags runs before the bridge is
    /// optimised and compiled, and that work allocates, so an address copied
    /// while the walk was still running can name a frame the collector has
    /// since moved. The blackhole compares the adopted cell against the
    /// deadframe's slot 0, which IS forwarded across the same region, so a
    /// stale copy here misses the handshake and lets the guard-state resume
    /// re-run a region the walk already executed.
    fn stamp_ca_walk_frames(
        bridge_frame_root: &mut FrameRoot,
        stamp_resume_frame: bool,
        stamp_adopted_frame: bool,
    ) {
        if !stamp_resume_frame && !stamp_adopted_frame {
            return;
        }
        let frame = bridge_frame_root.frame() as *mut PyFrame as usize;
        if stamp_resume_frame {
            CA_WALK_RESUME_FRAME.with(|c| c.set(frame));
        }
        if stamp_adopted_frame {
            CA_WALK_ADOPTED_FRAME.with(|c| c.set(frame));
        }
    }

    {
        let (driver, _) = crate::eval::driver_pair();
        driver.clear_last_compiled_artifact_token();
    }

    use crate::eval::build_jit_state;
    use crate::jit::state::PyreEnv;

    // compile.py `ResumeGuardForcedDescr.handle_fail`: "Failures of
    // a GUARD_NOT_FORCED are never compiled, but always just blackholed."
    // A bridge walked from one force flavor (pure force, live call result)
    // would also be entered for forced-and-raised failures whose call result
    // is NULL — its result-class guard then dereferences NULL.
    if descr_arc.is_guard_forced() {
        return BridgeResolution::ResumeBlackhole;
    }
    // compile.py `must_compile() and not stack_almost_full()`: this is
    // pyre's own guard-failure entry, so the diagnostic that suppresses bridge
    // recording has to be read here as well as in the jitdriver's paths.
    if majit_metainterp::no_bridge_enabled() {
        return BridgeResolution::ResumeBlackhole;
    }
    let Some((green_key, trace_id, fail_index)) = bridge_source_identity_from_descr(descr_arc)
    else {
        // compile.py `_trace_and_compile_from_bridge` raises
        // `compile.giveup()` when `loop_token` is None (memmgr-evicted).
        // Pyre signals the same outcome by returning `false`, dropping
        // the caller into `resume_in_blackhole` (pyjitpl.py:711).
        return BridgeResolution::ResumeBlackhole;
    };
    // `MAJIT_MAX_BRIDGES` counts bridges actually attempted, so it is spent
    // after the bails above rather than alongside `no_bridge_enabled`.
    if !majit_metainterp::bridge_fuel_take() {
        return BridgeResolution::ResumeBlackhole;
    }
    if bridge_bail_stage() == 1 {
        return BridgeResolution::ResumeBlackhole;
    }

    let info = {
        let (_, info) = crate::eval::driver_pair();
        info.clone()
    };

    // pyjitpl.py handle_guard_failure / compile.py _trace_and_compile_from_bridge:
    // frontend meta is the compiled_loops row on
    // `rd_loop_token.outermost_jitdriver_sd`, not leftover `active_jitdriver_sd`.
    // RPython then calls initialize_state_from_guard_failure(resumedescr, deadframe)
    // which internally calls rebuild_from_resumedata (resume.py).
    // This restores the complete frame stack INSIDE the bridge function.
    let meta = {
        let (driver, _) = crate::eval::driver_pair();
        compiled_meta_for_bridge_source(driver.meta_interp(), descr_arc, green_key)
    };
    let mut jit_state_local = build_jit_state(frame, &info);
    // `num_resume_frames > 1` marks a multi-frame (inlined-callee) guard:
    // the guard fired inside a callee inlined into the trace, so the resume
    // pc is the INNERMOST frame's bytecode pc, which does not address the
    // live (outer) frame `eval_loop_jit` runs. Such a resume cannot be
    // completed by interpreting the live frame forward — see the
    // blackhole routing at the handoff below.
    let decoded_resume = meta.as_ref().and_then(|meta| {
        crate::eval::decode_and_restore_guard_failure(
            &mut jit_state_local,
            meta,
            raw_values,
            exit_layout,
        )
    });
    if bridge_bail_stage() == 2 {
        return BridgeResolution::ResumeBlackhole;
    }
    let Some(crate::eval::DecodedGuardFailure {
        resume_pc,
        num_resume_frames,
        coords: resume_coords,
        direct_virtuals,
        ..
    }) = decoded_resume
    else {
        return BridgeResolution::ResumeBlackhole;
    };
    let is_multiframe_resume = num_resume_frames > 1;
    // `resume_pc` addresses the INNERMOST resumed section, so for a
    // multi-frame guard it is a pc in an inlined callee's own code object.
    // It only addresses THIS frame while that section belongs to this
    // frame's code — the same condition the paired valuestackdepth
    // correction already applies before handing the coordinate back.
    // The routing below is meant to keep such a resume off the live frame,
    // but every arm after this point can still return `ResumeBlackhole`, and
    // the blackhole itself refuses a jitcode carrying an abort opcode or a pc
    // whose liveness it cannot decode. On that path the interpreter resumes
    // this frame, so a foreign pc left in `last_instr` would step a code
    // object the frame does not run. Keep the vable-derived position instead.
    let frame_code = jit_state_local.pycode_as_usize();
    let foreign_innermost =
        is_multiframe_resume && resume_coords.last().map(|&(code, _)| code) != Some(frame_code);
    if foreign_innermost {
        pyre_jit_trace::jitcode_dispatch::census_record_frame_shape_decline(
            frame_code,
            "Resume::ForeignInnermostLastInstr",
        );
    } else {
        frame.set_last_instr_from_next_instr(resume_pc);
    }
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame) };
    let env = PyreEnv;
    let mut jit_state = build_jit_state(frame, &info);

    // A resume_pc on LOAD_CONST + RETURN_VALUE (or `n<=0` RETURN) is still
    // a live `handle_guard_failure` walk. RPython's `interpret()` records
    // through that return and `compile_done_with_this_frame` attaches a
    // DoneWithThisFrame FINISH bridge (`compile.compile_trace` with
    // `ends_with_jump=False`). Blackhole-only is the SwitchToBlackhole
    // fallback, not the success path.
    // RPython rebuild_from_resumedata (pyjitpl.py)
    // restores the complete frame stack before bridge tracing.
    // Bridge tracing sees the full frame layout — no truncation.
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][bridge-trace] start key={} trace={} fail={} resume_pc={}",
            green_key, trace_id, fail_index, resume_pc
        );
    }

    if bridge_bail_stage() == 3 {
        return BridgeResolution::ResumeBlackhole;
    }
    // compile.py:714: start_retrace_from_guard + set bridge_info.
    let started = {
        let (driver, _) = crate::eval::driver_pair();
        driver.start_bridge_tracing(
            descr_arc,
            &mut jit_state,
            &env,
            raw_values,
            resume_pc,
            // `decode_and_restore_guard_failure` has already walked this
            // guard's resume data applying every write — `replay_pending_fields`
            // for the deferred stores, `ResumeVableMode::GuardFailureSync` for
            // the virtualizable — so the replay owes recording only, seeded
            // with the objects that reader allocated.
            false,
            &direct_virtuals,
        )
    };
    if !started {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] start_bridge_tracing failed key={} trace={} fail={}",
                green_key, trace_id, fail_index
            );
        }
        return BridgeResolution::ResumeBlackhole;
    }
    // resume.py `rebuild_from_resumedata` ran once, inside
    // `start_bridge_tracing` (`rebuild_state_after_failure`); the portal's
    // `jd.mainjitcode` seats the frames (`call.py grab_initial_jitcodes`).
    // `pyjitpl.py _handle_guard_failure` calls `prepare_resume_from_failure`
    // once, after `rebuild_from_resumedata`. The same call covers the path
    // with no portal jitcode: there is no framestack to rebuild, and the
    // exception resume still has to run before the walk.
    // `finishframe_exception` drained the framestack.
    // `compile_exit_frame_with_exception` either attached the
    // exit-with-exception bridge or raised `SwitchToBlackhole` (that
    // raise already ran `aborted_tracing`). `_handle_guard_failure`
    // then calls `interpret`, which always raises
    // `jitexc.ExitFrameWithExceptionRef` — the frame is not resumed.
    // A still-open trace is the case whose cleanup stopped at
    // `aborted_tracing`; `abort_trace` finishes it, and the exception
    // still leaves.
    let resume = {
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp_mut().prepare_resume_from_failure()
    };
    if let majit_metainterp::PrepareResumeFromFailure::ExitFrameWithExceptionRef(exc) = resume {
        let (driver, _) = crate::eval::driver_pair();
        if driver.is_tracing() {
            driver.meta_interp_mut().abort_trace(false);
        }
        return BridgeResolution::FinishedException(pyre_jit_trace::state::ConcreteValue::Ref(
            exc.0 as PyObjectRef,
        ));
    }
    // `_prepare_exception_resumption` (pyjitpl.py) +
    // `prepare_resume_from_failure` (pyjitpl.py) parity: for exception
    // guard bridges (GUARD_EXCEPTION / GUARD_NO_EXCEPTION), SAVE_EXC_CLASS +
    // SAVE_EXCEPTION at trace start, then RESTORE_EXCEPTION before the guard.
    // The walker owns that whole sequence — it emits it at the bridge-entry
    // frame state, which is where the guard can capture resume data — so this
    // site only consumes the flag.
    let last_bridge_is_exception_guard = {
        let (driver, _) = crate::eval::driver_pair();
        driver.last_bridge_is_exception_guard
    };
    if last_bridge_is_exception_guard {
        let (driver, _) = crate::eval::driver_pair();
        driver.last_bridge_is_exception_guard = false;
    }

    // A pending exception at a NON-exception guard means a may-force
    // residual call both raised and forced the frame, so GUARD_NOT_FORCED
    // failed before GUARD_NO_EXCEPTION ran. That guard's resume_pc is the
    // no-exception semantic fallthrough (the next opcode, e.g.
    // RETURN_VALUE), so a normal-path bridge walk would trace the return
    // of the NULL call result — emitting `Finish(NONE)` with no
    // return-value box — and run the post-call residual ops concretely on
    // a NULL operand. Decline the bridge and resume in the blackhole,
    // which propagates the pending exception to its `catch_exception`
    // handler exactly once. Skipping the walk also prevents the walk's
    // concrete side effects from double-applying the handler's mutations
    // against the post-blackhole replay.
    //
    // GUARD_NOT_FORCED is not an exception guard, so it does not stash the
    // raised value into `jf_guard_exc` (`guard_exc` is 0 here); the live
    // signal is the backend `pos_exception` cell the compiled
    // GUARD_NO_EXCEPTION reads. A stale cell only over-declines (the
    // blackhole resume with `guard_exc == 0` runs the no-exception
    // continuation, identical to the prior panic-then-fallback path), so
    // this never changes a result — only whether a bridge is attached.
    let pending_exc = guard_exc != 0 || {
        #[cfg(feature = "cranelift")]
        {
            majit_backend_cranelift::jit_exc_class_raw() != 0
        }
        #[cfg(all(feature = "dynasm", not(feature = "cranelift")))]
        {
            majit_backend_dynasm::jit_exc_class_raw() != 0
        }
        #[cfg(not(any(feature = "cranelift", feature = "dynasm")))]
        {
            false
        }
    };
    // A pending exception at an EXCEPTION guard (GUARD_NO_EXCEPTION /
    // GUARD_EXCEPTION) whose raising op sits inside an in-frame try is the
    // bridge-side analogue of the same fallthrough-not-catch resume gap.
    // pyre encodes the guard's resume_pc as the no-exception semantic
    // fallthrough (the next opcode after the call), NOT the `except`
    // handler.  `finishframe_exception` (`pyjitpl.py`) looks up
    // `catch_exception` on the resumed framestack innermost-first and
    // jumps there; the walker's `find_catch_for_exc_resume` is that
    // lookup.  Route any catching frame — live or inlined — onto that
    // handler.  An uncaught raise still blackholes this failure
    // (`compile_exit_frame_with_exception` is the remaining close).
    /// The exception-table byte offset a `next_instr`-style resume coordinate
    /// maps to, mirroring the live-frame form `frame.last_instr * 2` (where
    /// `last_instr` is `next_instr - 1`).
    fn exc_table_offset(py_pc: usize) -> u32 {
        (py_pc.saturating_sub(1) as u32) * 2
    }

    /// Index of the resumed frame that catches, searched INNERMOST-first by
    /// asking each frame's own exception table about its own resume pc.
    /// `coords` is outermost-first, so `Some(0)` means the raise unwinds clear
    /// out of every inlined callee into the live frame; `None` means it
    /// escapes them all.
    fn resumed_catch_level(coords: &[(usize, usize)]) -> Option<usize> {
        for (level, &(w_code, py_pc)) in coords.iter().enumerate().rev() {
            if w_code == 0 {
                return None;
            }
            let raw = unsafe {
                pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                    as *const pyre_interpreter::CodeObject
            };
            if raw.is_null() {
                return None;
            }
            let code = unsafe { &*raw };
            if pyre_interpreter::pycode::lookup_exceptiontable(
                &code.exceptiontable,
                exc_table_offset(py_pc),
            )
            .is_some()
            {
                return Some(level);
            }
        }
        None
    }

    let catch_level = if pending_exc && last_bridge_is_exception_guard {
        if is_multiframe_resume {
            // `resume_pc` (and thus `frame.last_instr`, set above) addresses the
            // INNERMOST inlined frame while `code` is the live OUTER frame, so
            // the single-frame lookup below would consult the wrong code object.
            // Ask each resumed frame about its OWN pc instead.
            // `finishframe_exception` walks `framestack` innermost-first and
            // jumps to whichever frame's `catch_exception` answers; a catch
            // inside an inlined callee is a real bridge, not a decline.
            resumed_catch_level(&resume_coords)
        } else {
            let off = if frame.last_instr < 0 {
                0u32
            } else {
                (frame.last_instr as u32) * 2
            };
            pyre_interpreter::pycode::lookup_exceptiontable(&code.exceptiontable, off)
                .map(|_| 0usize)
        }
    } else {
        None
    };
    let caught_in_frame = catch_level == Some(0);
    // `finishframe_exception` (`pyjitpl.py`) jumps to the first catching
    // frame, live or inlined.  Route whenever any resumed frame catches.
    // An unwind that lands in the live frame discards the callee frames
    // (no inlined framestack left).  A catch inside a callee keeps the
    // carrier walk so the handler is recorded there.
    let unwind_to_live_frame = is_multiframe_resume
        && caught_in_frame
        && resume_coords
            .first()
            .is_some_and(|&(outer_w_code, _)| outer_w_code == frame.pycode as usize);
    let route_exc_edge = catch_level.is_some();
    // Discard inlined levels only when the raise unwinds *to* the live
    // frame.  A catch inside a callee still needs those levels as the
    // carrier recipes.
    pyre_jit_trace::jitcode_dispatch::set_exc_edge_discarded_levels(if unwind_to_live_frame {
        resume_coords.get(1..).unwrap_or(&[])
    } else {
        &[]
    });
    if route_exc_edge && guard_exc != 0 {
        // Publish the grabbed exception (`cpu.grab_exc_value` result) so the
        // walker's `seed_standing_exception_for_walk` threads it into
        // `sym.last_exc_box`, which the handler's `last_exc_value/>r` reads.
        // Re-read `GUARD_EXC_VALUE`: `guard_exc` is the pre-move copy, and
        // resume rebuild above allocates.
        let guard_exc = majit_metainterp::blackhole::GUARD_EXC_VALUE.with(|c| c.get());
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(guard_exc));
    } else if !pending_exc {
        // No standing exception at this bridge's source guard (e.g. a
        // GUARD_NOT_FORCED that fails every iteration because the may-force
        // residual forced the frame).  A prior residual-call raise may have
        // left a stale non-null value in `BH_LAST_EXC_VALUE`; the walker's
        // exc-edge routing seeds `sym.last_exc_value` from it and would route
        // this exception-free bridge into the `except` handler (recording the
        // `v = -1` handler body as the bridge body → the handler path runs
        // every iteration).  Clear it so the walk resumes on the real
        // no-exception continuation.
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    }
    let exception_resume_prepared = {
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp().exception_resume_was_prepared()
    };
    if pending_exc && !route_exc_edge && !exception_resume_prepared {
        // Uncaught: `finishframe_exception` would
        // `compile_exit_frame_with_exception`.  That Finish(exc) close is
        // not wired on this walk yet, so this failure blackholes.
        // `AbstractResumeGuardDescr.done_compiling` clears `ST_BUSY_FLAG`;
        // `jitcounter.tick` already reset the counter, so the next failure
        // ticks again.
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] uncaught pending exc → blackhole key={} trace={} fail={} resume_pc={}",
                green_key, trace_id, fail_index, resume_pc
            );
        }
        let (driver, _) = crate::eval::driver_pair();
        if driver.is_tracing() {
            driver.meta_interp_mut().abort_trace(false);
        }
        return BridgeResolution::ResumeBlackhole;
    }

    // The unwind discards every inlined callee, so the state the walk must
    // start from is the LIVE frame at its own call site — not the innermost
    // callee coordinate `decode_and_restore_guard_failure` installed for the
    // blackhole (its `resume_pc` override plus the matching innermost
    // `depth_based_vsd_for_wcode` correction).  Re-point the frame to the
    // outermost section before anything below reads it: `PyreJitState` holds
    // only the frame pointer, so the `jit_state` built above picks this up,
    // and `snapshot_for_tracing` then captures the coordinate the post-bridge
    // interpreter should resume at.  Only reachable where this function used
    // to decline outright, so no existing resume changes shape.
    let resume_pc = match resume_coords.first() {
        Some(&(outer_w_code, outer_py_pc)) if unwind_to_live_frame => {
            frame.set_last_instr_from_next_instr(outer_py_pc);
            if let Some(vsd) =
                pyre_jit_trace::state::depth_based_vsd_for_wcode(outer_w_code, outer_py_pc)
            {
                jit_state.set_valuestackdepth(vsd);
                jit_state.clear_stack_above(vsd);
            }
            outer_py_pc
        }
        _ => resume_pc,
    };

    // The live frame is a virtualizable GC object held by raw bridge-trace
    // locals while retracing can collect. RPython keeps the virtualizable
    // object GC-visible during retracing
    // (`rpython/jit/metainterp/pyjitpl.py:2839-2841`); pyre roots the frame
    // word so PyFrame's custom trace can forward `locals_cells_stack_w`.
    let mut bridge_frame_root = FrameRoot::new(frame);

    // pyjitpl.py interpret(): after start_retrace_from_guard, RPython
    // runs a single interpret() over the resumed frame state until the
    // bridge closes or aborts. `trace_bytecode` is the pyre equivalent of
    // that whole-loop interpreter; calling it once preserves the concrete PC
    // updates across branches/back-edges. Re-invoking it in a synthetic
    // `pc + 1` loop diverges from RPython and corrupts nested-loop bridges.
    // The bridge tracer must leave the real frame exactly at the guard
    // resume state so that a `BridgeCompiled` outcome re-enters
    // `eval_loop_jit` there — the `ContinueRunningNormally` arm does NOT
    // reconstruct the frame, it runs the interpreter forward from the live
    // frame's `last_instr` / value stack as-is. The full-body walker runs from
    // a private `snapshot_for_tracing` copy, but may-force residual calls use
    // the shared execution context during the walk, which advances the live frame's
    // `last_instr` AND its locals (e.g. a loop counter) to the walked
    // opcode's concrete state.  Snapshot the resume state here and restore it
    // after the walk so the post-bridge interpreter resumes at the guard
    // point rather than mid-body or past a dropped loop iteration (a
    // value-stack underflow / off-by-one-iteration result otherwise).
    let resume_state = frame.snapshot_for_tracing();
    let mut adopted_walk_end_state = false;
    // Arm the bridge `Terminate` no-replay shortcut for this walk.  The walk
    // epilogue (`run_perfn_walk` in trace.rs) reads this flag: only when armed
    // does a bridge `Terminate` walk keep its finish-concrete stash + commit
    // the store journal, so the three decisions (epilogue predicate, journal
    // commit, the `!allow_finish_direct_return` consume-vs-rewind branch
    // below) stay in agreement and a committed journal never strands into a
    // blackhole re-run.
    //
    // `pyjitpl.py` `interpret()` "should always raise": a resumed trace
    // that runs its frames forward to a return raises `DoneWithThisFrame`
    // from the POST-walk state, whatever the resume's frame count — there is
    // no rewind-to-the-deadframe re-run upstream.  A multi-frame resume walks
    // the inlined callees to `SubReturn` and only the LIVE frame's return
    // raises `Terminate`, so its finish-concrete is that frame's result, the
    // same value the blackhole rebuild would produce.  Replaying instead
    // re-executes every residual the walk already ran concretely and
    // double-applies its callee-internal side effects (`Counter.bump()`'s
    // `self.pos += 1` counted twice in
    // `bench/synth/nested_callee_chain_mutation_abort.py`).
    //
    // The CALL_ASSEMBLER callback (`allow_finish_direct_return == false`)
    // cannot carry the concrete itself, so it hands a kept stash to its
    // back-to-back blackhole hook through `CA_WALK_FINISHED_FRAME`.  That hook
    // does not rebuild a framestack — it returns the stashed value as the CA
    // callee's result once `ca_finished_frame == fail_values[0]` proves the
    // walk's live frame IS the frame whose guard failed — so it takes the same
    // any-frame-count arming.  Only the guard-state blackhole fallback below it
    // rebuilds frames, and a kept stash never reaches that path.
    pyre_jit_trace::jitcode_dispatch::fbw_bridge_noreplay_arm(true);
    let outcome = {
        let (driver, _) = crate::eval::driver_pair();
        driver.jit_merge_point_keyed(
            green_key,
            resume_pc,
            &mut jit_state,
            &env,
            || {},
            |meta, sym| {
                // pyjitpl.py:1571-1578: a merge point that only appends to
                // `current_merge_points` restores `self.pc` and goes on
                // interpreting, so the driver re-enters this walk instead of
                // handing the guest back.  Each entry starts from the frame
                // as the previous segment left it, so the snapshot is taken
                // here rather than once outside — `FrameBox` owns a shadow
                // stack root guard and cannot be cloned.
                let trace_frame = bridge_frame_root.frame().snapshot_for_tracing();
                // `sym.set_live_vable_frame_addr` records this word as the
                // virtualizable the walk writes back through, so it has to be
                // the frame's address as of this entry: the driver re-enters
                // the closure per merge point and `snapshot_for_tracing`
                // allocates, either of which can have moved the frame since the
                // root was taken.  Read it back out of the root, the same way
                // `trace_frame` above is.
                let live_frame_addr = bridge_frame_root.frame() as *const PyFrame as usize;
                let (action, executed, walk_end_flushed) = trace_bytecode(
                    meta,
                    sym,
                    code,
                    resume_pc,
                    trace_frame,
                    live_frame_addr,
                    false,
                );
                // pyjitpl.py raise_continue_running_normally:
                // a bridge walk that closed at a merge point and committed
                // its end-of-walk state into the trace snapshot
                // (`flush_walk_end_state_to_frame`) hands the LIVE frame
                // that end state — the walked region's residual calls
                // executed concretely, so resuming at the guard would
                // re-apply every side effect.  Uncommitted → fall through
                // to the guard-state restore below (legacy replay).
                if walk_end_flushed {
                    let frame = bridge_frame_root.frame();
                    frame.restore_resume_state_from(&executed);
                    adopted_walk_end_state = true;
                }
                action
            },
        )
    };
    // Disarm so the flag cannot leak into a later (non-bridge) walk on this
    // thread; the epilogue has already consumed it.
    pyre_jit_trace::jitcode_dispatch::fbw_bridge_noreplay_arm(false);

    // #177 bridge `Terminate` no-replay: consume any finish-concrete the walk
    // kept.  A stash survives the epilogue only when `bridge_noreplay_armed`
    // held AND the walk reached `Terminate` with a materialized concrete and
    // no unjournaled effect — the epilogue then committed the store journal,
    // so the walked region's eager side effects (including callee-internal
    // `SetfieldGc`s like `self.pos += 1`) stand exactly once.  Rewinding the
    // live frame to the guard pc and re-running the region — via the
    // `ContinueRunningNormally` re-entry — would apply them a second time.
    // Hand the concrete result forward as `DoneWithThisFrame` instead,
    // mirroring the top-level portal (eval.rs `maybe_compile_and_run`) and
    // `pyjitpl.py` `interpret()` raising `DoneWithThisFrame` from the
    // post-walk state.  Always take (not peek) so a kept stash cannot leak
    // into a later top-level portal `fbw_finish_concrete_take`.
    //
    // CALL_ASSEMBLER callback: this caller returns a bare bool and cannot
    // carry the concrete result itself.  Leave the stash in its GC-rooted
    // cell, record the callee frame in `CA_WALK_FINISHED_FRAME`, and let the
    // CA slow path's back-to-back blackhole hook
    // (`jit_blackhole_resume_from_guard`) take the stash and complete the
    // callee with it — the finishframe `DoneWithThisFrame` /
    // `ExitFrameWithExceptionRef` the assembler caller catches
    // (pyjitpl.py, jitexc.py).  `Terminate` is the LIVE frame's
    // return at any resume frame count, and the live frame here IS the CA
    // callee, so the hook's `ca_finished_frame == fail_values[0]` handshake
    // matches for a multi-frame resume exactly as for a single-frame one.
    if !allow_finish_direct_return {
        if pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_peek().is_some() {
            let finished_frame = bridge_frame_root.frame() as *mut PyFrame as usize;
            CA_WALK_FINISHED_FRAME.with(|c| c.set(finished_frame));
            // This return site predates the four below, so it needs the
            // adopted stamp too: the finished cell only answers the hook's
            // first question, and a walk that both committed its end state and
            // kept a finish-concrete stash still has to answer the second one
            // if the first handshake does not match.
            stamp_ca_walk_frames(&mut bridge_frame_root, false, adopted_walk_end_state);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-trace] ca-finish-noreplay at resume_pc={} key={}",
                    resume_pc, green_key
                );
            }
            return BridgeResolution::ResumeBlackhole;
        }
    }
    match pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_take() {
        Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Return(cv)) => {
            // `Terminate` is the LIVE frame's function return, about to be
            // popped by its caller with `cv`; any inlined callee the resume
            // reconstructed already returned inside the walk.  No rewind, no
            // blackhole.
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-trace] finish-noreplay at resume_pc={} key={}",
                    resume_pc, green_key
                );
            }
            return BridgeResolution::Finished(cv);
        }
        Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Raise(cv)) => {
            // jitexc.py:44: hand the walk's uncaught exception to the
            // guard's portal as ExitFrameWithExceptionRef.
            return BridgeResolution::FinishedException(cv);
        }
        None => {}
    }
    if !adopted_walk_end_state {
        let frame = bridge_frame_root.frame();
        frame.restore_resume_state_from(&resume_state);
    }
    // A multi-frame (inlined-callee) guard restored on the non-flush path
    // leaves the live OUTER frame at the INNERMOST frame's resume pc with the
    // inlined call's result unmaterialized — the inlined callee frame exists
    // only in the bridge/blackhole reconstruction, never in the live frame's
    // operand stack, so the live stack carries a stale NULL where the callee
    // result belongs. `eval_loop_jit` cannot resume the live frame there
    // (it would store the NULL through the post-call ops). The freshly
    // compiled bridge stays attached for subsequent guard failures; complete
    // THIS iteration through the blackhole, which rebuilds the full inlined
    // framestack and runs it — exactly as a cold multi-frame guard does.
    // Signalled by returning `false` (→ `handle_fail` ResumeInBlackhole) even
    // though the bridge compiled. The walk took the non-flush path, so its
    // store journal was rolled back; a residual it executed concretely is not
    // in that journal, so the blackhole re-run applies such an effect twice —
    // the `Terminate` no-replay shortcut above is what keeps a returning walk
    // off this path.
    // The non-flush path applies to single-frame guards too: without the
    // committed walk-end state the live frame's last_instr / operand stack
    // were never advanced through the guard's resume region (only the
    // blackhole syncs them), so "continue running normally" resumes at the
    // enclosing loop header and silently skips the region between the guard
    // and the header — the deopt iteration's remaining inner-loop work.
    // Complete this iteration through the blackhole exactly like the
    // multi-frame case above; the compiled bridge stays attached for
    // subsequent failures.
    let resume_via_blackhole = !adopted_walk_end_state;

    // merge_point handles Finish/CloseLoop via bridge_info.
    if outcome.is_some() {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] compiled at resume_pc={} key={} blackhole_current={}",
                resume_pc, green_key, resume_via_blackhole
            );
        }
        stamp_ca_walk_frames(
            &mut bridge_frame_root,
            !allow_finish_direct_return && resume_via_blackhole,
            !allow_finish_direct_return && adopted_walk_end_state,
        );
        return bridge_resolution_from_bool(!resume_via_blackhole);
    }

    // pyjitpl.py:2982-2983 / 3095-3099 parity:
    // compile_trace() "raises in case it works". In pyre the bridge can
    // already be attached during this step even if jit_merge_point_keyed()
    // did not surface DetailedDriverRunOutcome::Jump yet. Stop tracing as
    // soon as the backend metadata shows that the bridge is attached.
    let compiled = {
        let (driver, _) = crate::eval::driver_pair();
        driver
            .meta_interp()
            .bridge_was_compiled(green_key, trace_id, fail_index)
    };
    if compiled {
        let (driver, _) = crate::eval::driver_pair();
        if driver.is_tracing() {
            driver.meta_interp_mut().abort_trace(false);
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] compiled at resume_pc={} key={} (attached) blackhole_current={}",
                resume_pc, green_key, resume_via_blackhole
            );
        }
        stamp_ca_walk_frames(
            &mut bridge_frame_root,
            !allow_finish_direct_return && resume_via_blackhole,
            !allow_finish_direct_return && adopted_walk_end_state,
        );
        return bridge_resolution_from_bool(!resume_via_blackhole);
    }

    // If the driver is no longer tracing, the bridge was compiled
    // (or aborted) inside merge_point. Check whether a bridge was
    // actually attached to distinguish success from abort.
    //
    // pyjitpl.py raise_continue_running_normally: a trace that
    // reached a merge point exits with ContinueRunningNormally from the
    // CURRENT state — never through the blackhole-from-guard path. A
    // committed walk already executed the region and the live frame
    // adopted its end state above, so the caller must continue running
    // normally even when the trace closed as a new loop at an inner
    // header instead of attaching a bridge to this guard; resuming in
    // the blackhole would re-apply every walked side effect from the
    // guard-time values.
    let tracing_active = {
        let (driver, _) = crate::eval::driver_pair();
        driver.is_tracing()
    };
    if !tracing_active {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] trace ended at resume_pc={} key={} compiled={} adopted={}",
                resume_pc, green_key, compiled, adopted_walk_end_state
            );
        }
        let continue_compiled = compiled || adopted_walk_end_state;
        stamp_ca_walk_frames(
            &mut bridge_frame_root,
            !allow_finish_direct_return && !continue_compiled,
            !allow_finish_direct_return && adopted_walk_end_state,
        );
        return bridge_resolution_from_bool(continue_compiled);
    }

    // Trace did not converge into a bridge. Abort like RPython's
    // run_blackhole_interp_to_cancel_tracing fallback path.
    if tracing_active {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] abort: no-bridge key={} trace={} fail={} resume_pc={}",
                green_key, trace_id, fail_index, resume_pc
            );
        }
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp_mut().abort_trace(false);
    }
    // A committed walk has already executed the region into the live
    // frame; the blackhole replay must not run even on this abort path.
    stamp_ca_walk_frames(
        &mut bridge_frame_root,
        !allow_finish_direct_return && !adopted_walk_end_state,
        !allow_finish_direct_return && adopted_walk_end_state,
    );
    bridge_resolution_from_bool(adopted_walk_end_state)
}

/// compile.py handle_fail for call_assembler guard failures.
/// Checks must_compile (jitcounter.tick), and if threshold reached,
/// traces the alternate path via trace_and_compile_from_bridge.
///
/// pyjitpl.py `handle_guard_failure(self, resumedescr, deadframe)`
/// — descr identity is the only argument crossing the C-ABI boundary;
/// the receiver derives `(green_key, trace_id, fail_index)` from the
/// recovered Arc, mirroring `compile.py:706-708 _trace_and_compile_
/// from_bridge` which walks `resumedescr.rd_loop_token.loop_token_wref()`
/// for the owning JCT.
fn jit_ca_handle_guard_failure(
    deadframe: *mut majit_backend::jitframe::JitFrame,
    descr_addr: usize,
    guard_value_operand: i64,
    guard_value_operand_present: bool,
) -> Option<i64> {
    // `compile.py AbstractResumeGuardDescr.handle_fail` has no live-value
    // precondition: it runs `must_compile` for every reported failure. A
    // GUARD_VALUE whose fail-argument vector is empty still has everything the
    // hotness decision needs, because the operand it hashes travels separately
    // in `guard_value_operand` (`compile.py make_a_counter_per_value` keys the
    // counter on the failing value, not on a fail argument). Bailing on
    // `num_values == 0` would leave such a guard without a counter tick, and so
    // without a bridge, forever. Only a null pointer with a non-zero length is
    // unusable — that pairing cannot be turned into a slice at all.
    if deadframe.is_null() {
        return None;
    }
    // `enter_profiler_tracing` is not re-entrant (pyjitpl.py handle_guard_failure — RPython's
    // `handle_guard_failure` unwinds to the top-level `execute_token` before any
    // tracing decision, so a guard never fires while another trace is open).
    // pyre's CALL_ASSEMBLER guard callback runs synchronously from the backend
    // trampoline, so it CAN fire mid-trace: the self-recursion fold replays its
    // CALL_ASSEMBLER concretely while the outer trace is still recording, and a
    // deopt in that callee reaches here with the outer trace's profiler event
    // still open.  Starting a nested bridge trace would panic in
    // `start_retrace_from_guard`'s `enter_profiler_tracing`.  Bail to the
    // blackhole resume (return false) so the callee completes in the interpreter
    // and the outer trace keeps recording; the deferred bridge compiles later
    // when this guard fails outside a trace.
    {
        let (driver, _) = crate::eval::driver_pair();
        if driver.is_tracing() {
            return None;
        }
    }

    // compile.py _trace_and_compile_from_bridge.  Native CA code
    // crosses the backend boundary with only the raw descr pointer; recover
    // the backend FailDescr Arc before any guard-failure routing so identity
    // is read from the descr just like PyPy reads `resumedescr`.  The
    // recovery is infallible for live JIT code —
    // `Backend::fail_descr_arc_from_addr` panics if the raw value is not
    // a live `FailDescrCell` pointer, matching RPython's
    // `cpu.get_latest_descr(deadframe)` (warmspot.py) which has no
    // failure mode.  `trace_and_compile_from_bridge` itself takes only
    // the descr Arc.
    let descr_arc: std::sync::Arc<dyn majit_ir::Descr> = {
        use majit_backend::Backend;
        let (driver, _) = crate::eval::driver_pair();
        driver
            .meta_interp()
            .backend()
            .fail_descr_arc_from_addr(descr_addr)
    };
    let Some((source_green_key, source_trace_id, source_fail_index)) =
        bridge_source_identity_from_descr(&descr_arc)
    else {
        return None;
    };
    let descr_fd = descr_arc
        .as_fail_descr()
        .expect("fail_descr_arc_from_addr returned non-FailDescr");
    let n_fail_args = descr_fd.fail_arg_types().len();
    let fail0 = unsafe { majit_backend::get_int_value(deadframe, descr_fd, 0) };
    let guard_value_operand = guard_value_operand_present.then_some(guard_value_operand);
    // pyjitpl.py: the virtualizable stays a GC root for the whole
    // handle_fail. `must_compile` and `get_compiled_exit_layout` allocate,
    // so root failarg 0 before either runs — a raw `&mut PyFrame` taken
    // here would dangle by the time `trace_and_compile_from_bridge`
    // builds its `FrameRoot`.
    if fail0 == 0 {
        return None;
    }
    let mut frame_root = FrameRoot::new(unsafe { &mut *(fail0 as *mut PyFrame) });

    // compile.py handle_fail traces exception guards the same as any other
    // (`must_compile` then `_trace_and_compile_from_bridge`). The exception
    // lives on the deadframe (`cpu.grab_exc_value`); read it here so
    // `_prepare_exception_resumption` / `prepare_resume_from_failure` see
    // the same value the general `handle_fail` path threads as `guard_exc`.
    let guard_exc = unsafe { (*deadframe).jf_guard_exc as i64 };
    let _guard_exc_root = majit_metainterp::blackhole::GuardExcRoot::park(guard_exc);

    // compile.py must_compile: jitcounter.tick(guard_hash, increment)
    let (must_compile, _) = {
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp_mut().must_compile_with_values(
            &descr_arc,
            &[],
            guard_value_operand,
            source_green_key,
        )
    };
    // compile.py: must_compile() and not stack_almost_full()
    if !must_compile || majit_metainterp::MetaInterp::<()>::stack_almost_full() {
        return None;
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][ca-bridge] must_compile fired: key={} trace={} fail={}",
            source_green_key, source_trace_id, source_fail_index,
        );
    }

    // `AbstractResumeGuardDescr.handle_fail`: the layout is the failing
    // descr's own; a bridge guard has no frontend record.
    let exit_layout = majit_metainterp::exit_layout_for_descr(&descr_arc);

    // compile.py try/finally: `start_compiling()` before
    // bridge, `done_compiling()` on every unwind path.  RAII guard
    // dispatches both via `descr.as_fail_descr()` (instance-method
    // dispatch per `compile.py`); drop pairs `done_compiling`
    // with the matching `start_compiling` even on panic.
    let resolution = {
        let _guard = crate::eval::GuardCompilingScope::new(&descr_arc);
        // `allow_finish_direct_return = false`: a Finish walk still
        // hands its stash to the blackhole hook via `CA_WALK_FINISHED_FRAME`
        // (`ResumeBlackhole`). A JUMP attach returns `CompiledContinue`
        // and re-enters the portal. `ExitFrameWithExceptionRef` from
        // `prepare_resume_from_failure` returns `FinishedException`
        // before that stash exists.
        let mut raw_values: Vec<i64> = (0..n_fail_args)
            .map(|i| unsafe { majit_backend::get_int_value(deadframe, descr_fd, i) })
            .collect();
        // The copy is not a JITFRAME: `jitframe_trace` cannot update it.
        // Root Ref slots for the same window `handle_fail` already covers
        // (`DeadFrameRefRoots` / `compute_gcmap`).
        let _deadframe_roots = unsafe {
            majit_metainterp::resume::DeadFrameRefRoots::enter(&mut raw_values, |index| {
                // Slot 0 is the virtualizable (PyFrame). It is a GC
                // object even when `exit_types` has not yet classified
                // it; a CA bridge walk that forces `f_locals` reads it
                // as `live_vable_frame_addr`.
                index == 0 || exit_layout.is_traced_ref_slot(index)
            })
        };
        trace_and_compile_from_bridge(
            &descr_arc,
            frame_root.frame(),
            &raw_values,
            &exit_layout,
            guard_exc,
            false,
        )
    };
    // compile.py record_loop_or_bridge registers every bridge's dependencies.
    crate::eval::register_quasi_immutable_deps(source_green_key);

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][ca-bridge] compiled={} key={} trace={} fail={}",
            matches!(resolution, BridgeResolution::CompiledContinue),
            source_green_key,
            source_trace_id,
            source_fail_index,
        );
    }

    match resolution {
        BridgeResolution::CompiledContinue => {
            // compile.py handle_fail must_compile arm raises
            // ContinueRunningNormally; handle_jitexception re-enters portal_ptr.
            // The bridge walk and compile allocate, so a minor collection may
            // have moved a nursery frame since `fail0` was read. The root holds
            // the forwarded address.
            let frame = frame_root.frame() as *mut PyFrame as i64;
            Some(run_frame_through_portal(frame, PortalEntry::Resume))
        }
        BridgeResolution::ResumeBlackhole => None,
        // `blackhole.py` `_exit_frame_with_exception` /
        // `jitexc.py` `ExitFrameWithExceptionRef`: the exception leaves
        // the assembler-called frame. Same delivery as
        // `ca_complete_after_bridge_walk`'s `FinishConcrete::Raise`.
        BridgeResolution::FinishedException(cv) => Some(ca_finish_raise_word(cv)),
        // `interpret()` raising `DoneWithThisFrame`. The walk's own
        // finish still takes the `CA_WALK_FINISHED_FRAME` stash; this
        // arm is the direct return when that handshake did not fire.
        BridgeResolution::Finished(cv) => Some(ca_finish_return_word(cv)),
    }
}

/// Concrete completion of the assembler-called frame that did not go
/// through the `CA_WALK_FINISHED_FRAME` stash: `prepare_resume_from_failure`
/// returned `ExitFrameWithExceptionRef` before the walk, or the walk
/// finished the frame (`interpret()` raising `DoneWithThisFrame`).
#[allow(dead_code)]
enum CaFrameCompletion {
    Returned(pyre_jit_trace::state::ConcreteValue),
    Raised(pyre_jit_trace::state::ConcreteValue),
}

/// Feed a CALL_ASSEMBLER callee guard through the normal bridge-hotness path.
/// The caller has already recovered the live descriptor and exit values; the
/// rest is the same must-compile/bridge attachment sequence used by the native
/// CALL_ASSEMBLER guard callback above.
///
/// A bridge that does not compile takes the same resume as a guard whose
/// counter has not fired (`AbstractResumeGuardDescr.handle_fail`'s blackhole
/// arm). `done_compiling` clears `ST_BUSY_FLAG` and `jitcounter.tick` already
/// reset the counter, so a later failure ticks again.
#[allow(dead_code)]
fn try_compile_ca_bridge(
    descr_arc: &std::sync::Arc<dyn majit_ir::Descr>,
    raw_values: &[i64],
    guard_value_operand: Option<i64>,
    // `cpu.grab_exc_value(deadframe)` (llmodel.py). `jit_ca_handle_guard_failure`
    // re-reads `jf_guard_exc` and threads that word; the wasm deopt has already
    // read that slot (`grab_exc_value` on the live jitframe).
    // `prepare_resume_from_failure` (pyjitpl.py) calls `execute_ll_raised` then
    // `handle_possible_exception`, so the bridge enters the handler with the
    // exception live. `0` is the no-exception resume.
    guard_exc: i64,
) -> Option<CaFrameCompletion> {
    if raw_values.is_empty() {
        return None;
    }
    let Some((source_green_key, _, _)) = bridge_source_identity_from_descr(descr_arc) else {
        return None;
    };
    let (must_compile, owning_key) = {
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp_mut().must_compile_with_values(
            descr_arc,
            raw_values,
            guard_value_operand,
            source_green_key,
        )
    };
    if !must_compile || majit_metainterp::MetaInterp::<()>::stack_almost_full() {
        return None;
    }
    // `AbstractResumeGuardDescr.handle_fail`: the layout is the failing
    // descr's own; a bridge guard has no frontend record.
    let exit_layout = majit_metainterp::exit_layout_for_descr(&descr_arc);
    let frame_ptr = raw_values[0] as *mut PyFrame;
    if frame_ptr.is_null() {
        return None;
    }
    let frame = unsafe { &mut *frame_ptr };
    let _guard = crate::eval::GuardCompilingScope::new(descr_arc);
    let resolution =
        trace_and_compile_from_bridge(descr_arc, frame, raw_values, &exit_layout, guard_exc, false);
    // `blackhole.py` `_exit_frame_with_exception` /
    // `jitexc.py` `ExitFrameWithExceptionRef` leaves the assembler-called
    // frame. The walk's own `FinishConcrete::Raise` still stashes and
    // returns `ResumeBlackhole`; this is the pre-walk escape.
    let completion = match resolution {
        BridgeResolution::FinishedException(cv) => Some(CaFrameCompletion::Raised(cv)),
        BridgeResolution::Finished(cv) => Some(CaFrameCompletion::Returned(cv)),
        BridgeResolution::CompiledContinue | BridgeResolution::ResumeBlackhole => None,
    };
    // The wasm CALL_ASSEMBLER path likewise bypasses handle_fail's dependency drain.
    crate::eval::register_quasi_immutable_deps(owning_key);
    completion
}

/// Queue a deferred inline merge, called from the out-of-line bridge that
/// stands in for it once that bridge has been entered often enough to pay for
/// rebuilding its owner (`majit_backend_wasm::INLINE_TRIP_THRESHOLD`). Its
/// table slot is published through `set_inline_trip_helper_slot` during
/// `JIT_DRIVER` init, the way the CA deopt helper's is.
///
/// ⛔ This runs with the bridge on the wasm stack, inside the
/// `run_compiled` call that `execute_assembler` holds the driver mutably
/// across, so it must not reach the driver: it only records the id, and
/// `execute_assembler` installs the merge once the trace has returned.
///
/// Returns 0 because the trace drops the result.
#[cfg(target_arch = "wasm32")]
pub extern "C" fn wasm_jit_inline_trip(pending_slot: i64) -> i64 {
    majit_backend_wasm::record_inline_trip(pending_slot);
    0
}

/// Host completion for an in-guest self-recursive CALL_ASSEMBLER callee that
/// deopted (`PYRE_WASM_CA`). The wasm CA arm `call_indirect`s this through the
/// shared `__indirect_function_table` (its slot published via
/// `majit_backend_wasm::set_ca_deopt_helper_slot` during `JIT_DRIVER` init)
/// when an in-guest callee frame returns a non-finish `fail_index`.
///
/// It is the wasm analog of the dynasm/cranelift CA fast path's deopt leg
/// (`call_assembler_fast_path_heap` → `jit_blackhole_resume_from_guard`): build
/// a `DeadFrame` from the callee's already-run frame, then **blackhole-resume it
/// AT the guard PC** — so the callee's pre-guard work is not re-executed (unlike
/// a force / entry re-run) — and return the call result, a boxed Ref. A callee
/// that actually finished (a base case, or a chained-bridge finish the arm's
/// static fast-path set did not recognise) short-circuits to its output slot.
///
/// `frame_ptr` is the deopted callee frame; `compiled_ptr` is the source
/// loop's `CompiledWasmLoop`, both baked into the trace by `compile_bridge`.
#[cfg(target_arch = "wasm32")]
pub extern "C" fn wasm_ca_resume_deopt(frame_ptr: i64, compiled_ptr: i64) -> i64 {
    use majit_backend::Backend;
    let frame =
        majit_backend_wasm::dead_frame_from_ran_frame(compiled_ptr as usize, frame_ptr as usize);

    // Decode the exit through the same Backend accessors the host's outermost
    // deopt handling uses (`run_compiled_detailed_with_values`); extract owned
    // values so the driver borrow is released before the blackhole re-enters it.
    enum Outcome {
        Finished(i64),
        FinishedException(i64),
        Deopt {
            descr_arc: std::sync::Arc<dyn majit_ir::Descr>,
            green_key: u64,
            exit_layout: majit_metainterp::CompiledExitLayout,
            raw_values: Vec<i64>,
            guard_value_operand: Option<i64>,
            guard_exc: i64,
            savedata: Option<majit_ir::GcRef>,
        },
    }
    let outcome = {
        let (driver, _) = crate::eval::driver_pair();
        let mi = driver.meta_interp();
        let backend = mi.backend();
        let descr_arc = backend.get_latest_descr_arc(&frame);
        let descr = descr_arc
            .as_fail_descr()
            .expect("CA deopt: get_latest_descr_arc returned a non-FailDescr Descr");
        if descr.is_finish() {
            let result = backend.get_ref_value(&frame, 0).as_usize() as i64;
            // compile.py ExitFrameWithExceptionDescrRef parity: a FINISH
            // descr is either DoneWithThisFrame (return the value) or
            // ExitFrameWithException (the callee raised; slot 0 holds the
            // exception). The self-recursive callee re-raised — propagate it
            // through the exception channel like the outer Finished arm
            // (`eval.rs`'s `handle_fail`) instead of banking it as the
            // recursive-call return, which would surface as `int + <exc>`.
            if descr.is_exit_frame_with_exception() {
                Outcome::FinishedException(result)
            } else {
                Outcome::Finished(result)
            }
        } else if let Some(value) = majit_backend::propagate_exception_handle_fail(
            descr,
            backend.grab_exc_value(&frame).0 as i64,
        ) {
            // `compile.py` `PropagateExceptionDescr.handle_fail`. Same
            // reader as the host exit: `grab_exc_value`, then
            // `memory_error` when the cell is null. Publish it the way
            // the ExitFrameWithException arm does.
            Outcome::FinishedException(value)
        } else {
            let green_key = majit_backend::descr_owning_jct(descr)
                .map(|jct| jct.green_key())
                .unwrap_or(0);
            let exit_layout = majit_metainterp::exit_layout_for_descr(&descr_arc);
            // compile.py `AbstractResumeGuardDescr.must_compile` reads a
            // GUARD_VALUE's actual operand directly from the deadframe before
            // hashing `(descr, value)`.  It need not be one of the guard's
            // fail arguments: wasm codegen parks that case in the trailing
            // `counter_value_spill` slot specifically for this read.  Keep it
            // while the in-guest CA frame is live, just as
            // `jit_ca_handle_guard_failure`'s native callback does; falling
            // back to `raw_values[index]` below turns an out-of-vector slot
            // into zero and merges every failing value into one hot counter.
            let guard_value_operand = majit_backend::guard_value_counter_slot(descr)
                .map(|slot| backend.get_value_direct(&frame, slot));
            let raw_values: Vec<i64> = descr
                .fail_arg_types()
                .iter()
                .enumerate()
                .map(|(i, &tp)| match tp {
                    majit_ir::Type::Int => backend.get_int_value(&frame, i),
                    majit_ir::Type::Ref => backend.get_ref_value(&frame, i).as_usize() as i64,
                    majit_ir::Type::Float => backend.get_float_value(&frame, i).to_bits() as i64,
                    majit_ir::Type::Void => 0,
                })
                .collect();
            let guard_exc = backend.grab_exc_value(&frame).0 as i64;
            let savedata = backend.get_savedata_ref(&frame);
            Outcome::Deopt {
                descr_arc,
                green_key,
                exit_layout,
                raw_values,
                guard_value_operand,
                guard_exc,
                savedata,
            }
        }
    };

    match outcome {
        Outcome::Finished(r) => r,
        // warmspot.py:998-1005 ExitFrameWithExceptionRef: the callee raised.
        // Publish the exception so the caller's GUARD_NO_EXCEPTION (after the
        // CALL_ASSEMBLER) fires, and return garbage — parity with
        // `handle_blackhole_result`'s ExitFrameWithExceptionRef arm.
        Outcome::FinishedException(exc) => {
            if exc != 0 {
                publish_residual_call_exception(exc as PyObjectRef);
            }
            0
        }
        Outcome::Deopt {
            descr_arc,
            green_key,
            exit_layout,
            mut raw_values,
            guard_value_operand,
            mut guard_exc,
            savedata,
        } => {
            // This `guard_exc` is a copy of `grab_exc_value` (`llmodel.py`).
            // Park it in `GUARD_EXC_VALUE` so `_prepare_exception_resumption`
            // reads the forwarded address, and keep the local rooted for the
            // blackhole resume after the bridge attempt.
            let _parked_guard_exc = majit_metainterp::blackhole::GuardExcRoot::park(guard_exc);
            let _guard_exc_root = BareRefRoot::register(&mut guard_exc);
            let _deadframe_roots = unsafe {
                majit_metainterp::resume::DeadFrameRefRoots::enter(&mut raw_values, |index| {
                    exit_layout.is_traced_ref_slot(index)
                })
            };
            // compile.py ResumeGuardForcedDescr.handle_fail reads
            // `cpu.get_savedata_ref(deadframe)` after the bridge attempt.
            // `dead_frame_from_ran_frame` already copied `jf_savedata`;
            // root that copy across the same window.
            let mut savedata_slot = [savedata.map_or(0, majit_ir::GcRef::as_usize) as i64];
            let _savedata_root = unsafe {
                majit_metainterp::resume::DeadFrameRefRoots::enter(&mut savedata_slot, |_| {
                    savedata.is_some()
                })
            };
            if let Some(completion) =
                try_compile_ca_bridge(&descr_arc, &raw_values, guard_value_operand, guard_exc)
            {
                // `prepare_resume_from_failure` returned
                // `ExitFrameWithExceptionRef` before the walk.
                // `blackhole.py` `_exit_frame_with_exception` /
                // `jitexc.py` `ExitFrameWithExceptionRef`: deliver it
                // ahead of the guard-state blackhole, which would resume
                // the frame the exception already left.
                CA_WALK_FINISHED_FRAME.with(|c| c.set(0));
                CA_WALK_ADOPTED_FRAME.with(|c| c.set(0));
                CA_WALK_RESUME_FRAME.with(|c| c.set(0));
                pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_reset();
                return match completion {
                    CaFrameCompletion::Raised(cv) => ca_finish_raise_word(cv),
                    CaFrameCompletion::Returned(cv) => ca_finish_return_word(cv),
                };
            }
            // The walk above may have carried the callee past the guard —
            // running the resumed region to the callee's return, or committing
            // its end-of-walk state into the live frame — executing every
            // residual in it concretely.  Take the same completion the native
            // CA hook takes; the guard-state resume below would re-run that
            // region and apply its side effects a second time (#177).  This is
            // the wasm side of the `CA_WALK_*` handshake: nothing else on this
            // target reads those cells, so an unconsumed completion is silently
            // dropped into a replay.
            let ca_finished_frame = CA_WALK_FINISHED_FRAME.with(|c| c.replace(0));
            let ca_adopted_frame = CA_WALK_ADOPTED_FRAME.with(|c| c.replace(0));
            let callee_frame = raw_values.first().copied().unwrap_or(0) as usize;
            if let Some(result) =
                ca_complete_after_bridge_walk(ca_finished_frame, ca_adopted_frame, callee_frame)
            {
                return result;
            }
            // compile.py ResumeGuardForcedDescr.handle_fail reads
            // `cpu.get_savedata_ref(deadframe)` after the bridge attempt.
            // `dead_frame_from_ran_frame` already copied `jf_savedata`;
            // root that copy across the same window.
            let mut savedata_slot = [savedata.map_or(0, majit_ir::GcRef::as_usize) as i64];
            let _savedata_root = unsafe {
                majit_metainterp::resume::DeadFrameRefRoots::enter(&mut savedata_slot, |_| {
                    savedata.is_some()
                })
            };
            let savedata = savedata.map(|_| majit_ir::GcRef(savedata_slot[0] as usize));
            let bh = crate::eval::resume_in_blackhole_from_exit_layout(
                &mut raw_values,
                &exit_layout,
                guard_exc,
                false,
                savedata,
            );
            handle_blackhole_result(bh).unwrap_or(0)
        }
    }
}

// ── Callee frame creation for call_assembler ─────────────────────

/// Public wrapper for trace-through inlining.
pub fn create_callee_frame_impl_pub(caller_frame: i64, callable: i64, args: &[PyObjectRef]) -> i64 {
    create_callee_frame_impl(caller_frame, callable, args)
}

fn fill_positional_defaults_for_jit_call<'a>(
    callable: PyObjectRef,
    w_code: PyObjectRef,
    args: &'a [PyObjectRef],
) -> Cow<'a, [PyObjectRef]> {
    let defaults = unsafe { function_get_defaults(callable) };
    if defaults.is_null() {
        return Cow::Borrowed(args);
    }

    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject)
    };
    let nparams = code.arg_count as usize;
    if args.len() >= nparams {
        return Cow::Borrowed(args);
    }

    let ndefaults = if unsafe { pyre_object::is_tuple(defaults) } {
        unsafe { pyre_object::w_tuple_len(defaults) }
    } else {
        0
    };
    if ndefaults == 0 {
        return Cow::Borrowed(args);
    }
    let first_default = nparams.saturating_sub(ndefaults);
    if args.len() < first_default {
        // function.py:_flat_pycall_defaults is entered only after argument
        // matching proves that all required positional parameters are present.
        // Do not synthesize PY_NULL for missing required args here; callers
        // that reach this helper without enough args must keep the original
        // frame shape and let the normal call/resume path handle the error.
        return Cow::Borrowed(args);
    }

    let defaults_to_load = nparams - first_default;
    let default_start = ndefaults - defaults_to_load;
    let mut full = Vec::with_capacity(nparams);
    full.extend_from_slice(args);
    for i in args.len()..nparams {
        if i >= first_default {
            let default_idx = default_start + (i - first_default);
            let Some(default) =
                (unsafe { pyre_object::w_tuple_getitem(defaults, default_idx as i64) })
            else {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][defaults] tuple access failed default_idx={default_idx} defaults={defaults:p}"
                    );
                }
                return Cow::Borrowed(args);
            };
            full.push(default);
        } else {
            full.push(PY_NULL);
        }
    }
    Cow::Owned(full)
}

/// Locals `createframe` stores, matched the way `funccall` matches them.
///
/// `def f(*args)` has one local, and that local is the packed tuple.
/// Copying the raw positional into it makes `len(args)` raise
/// `TypeError` on the int. A mismatch still uses the defaults-only
/// slice: the assembler entry has no error return, and the previous
/// frame shape is what that path already ran.
fn scope_args_for_jit_callee(
    callable: PyObjectRef,
    w_code: PyObjectRef,
    args: &[PyObjectRef],
) -> Vec<PyObjectRef> {
    let roots = pyre_object::gc_roots::push_roots();
    let mut live = Vec::with_capacity(2 + args.len());
    live.push(callable);
    live.push(w_code);
    live.extend_from_slice(args);
    let base = roots.pin_roots(&live);
    let callable = roots.get(base);
    let w_code = roots.get(base + 1);
    let mut positional = Vec::with_capacity(args.len());
    for i in 0..args.len() {
        positional.push(roots.get(base + 2 + i));
    }
    let code_ref = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject)
    };
    if let Ok(filled) =
        pyre_interpreter::call::fill_user_function_args(callable, code_ref, &positional)
    {
        return filled;
    }
    let callable = roots.get(base);
    let w_code = roots.get(base + 1);
    positional.clear();
    for i in 0..args.len() {
        positional.push(roots.get(base + 2 + i));
    }
    fill_positional_defaults_for_jit_call(callable, w_code, &positional).into_owned()
}

/// Self-recursive entries have the caller's code and no function object,
/// so defaults stay unbound here. `pack_varargs` still builds the
/// `*args` / `**kwargs` locals `createframe` writes.
fn scope_args_for_jit_code(code: *const (), args: &[PyObjectRef]) -> Vec<PyObjectRef> {
    let code_ref = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(code as PyObjectRef)
            as *const pyre_interpreter::CodeObject)
    };
    pyre_interpreter::call::pack_varargs(code_ref, args.to_vec())
}

fn create_callee_frame_impl_1_boxed(
    caller_frame: i64,
    callable: PyObjectRef,
    boxed_arg: PyObjectRef,
) -> i64 {
    create_callee_frame_impl(caller_frame, callable as i64, &[boxed_arg])
}

fn create_self_recursive_callee_frame_impl_1_boxed(
    caller_frame: i64,
    boxed_arg: PyObjectRef,
) -> i64 {
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let func_code = caller.pycode;
    let w_globals = caller.get_w_globals();
    // Read before the call: `alloc_callee_frame` resolves `__builtins__`, which
    // can run a user `__getitem__` and so collect.  What this line reports is
    // the operand the trace passed; `locals` below is the frame's own.
    let passed_arg = boxed_arg as usize;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[func_code as PyObjectRef, w_globals, boxed_arg]);
    let execution_context = pyre_interpreter::call::getexecutioncontext();
    let func_code = roots.get(base) as *const ();
    let w_globals = roots.get(base + 1);
    let boxed_arg = roots.get(base + 2);
    let frame_ptr =
        alloc_matched_self_recursive_frame(func_code, w_globals, &[boxed_arg], execution_context);
    if majit_metainterp::majit_log_enabled() {
        let f = unsafe { &*frame_ptr };
        eprintln!(
            "[jit][ca-frame] ptr={frame_ptr:p} locals=0x{:x} vsd={} boxed_arg=0x{passed_arg:x}",
            f.locals_cells_stack_w as usize, f.valuestackdepth,
        );
    }
    frame_ptr as i64
}

fn create_callee_frame_impl(_caller_frame: i64, callable: i64, args: &[PyObjectRef]) -> i64 {
    // `getexecutioncontext` can allocate the thread's first context. Pin
    // first, or that allocation forwards `callable` and the positionals
    // before the match below sees them.
    let roots = pyre_object::gc_roots::push_roots();
    let mut live = Vec::with_capacity(1 + args.len());
    live.push(callable as PyObjectRef);
    live.extend_from_slice(args);
    let base = roots.pin_roots(&live);
    let execution_context = pyre_interpreter::call::getexecutioncontext();
    let callable = roots.get(base);
    let mut positional = Vec::with_capacity(args.len());
    for i in 0..args.len() {
        positional.push(roots.get(base + 1 + i));
    }
    create_callee_frame_in_ctx(execution_context, callable, &positional)
}

/// [`create_callee_frame_impl`] with the execution context passed directly.
///
/// The caller frame contributes nothing else here — `function.py funccall`
/// builds the callee frame out of the *callee's* code, globals and closure,
/// and takes the execution context from the space.  A residual helper that
/// already holds `space.getexecutioncontext()` therefore has no reason to
/// resolve a caller frame for this.
fn create_callee_frame_in_ctx(
    execution_context: *const pyre_interpreter::PyExecutionContext,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> i64 {
    // `fill_user_function_args` allocates the `*args` tuple. The code and
    // globals `createframe` stores are reloaded from these pins: the words
    // captured before that allocation are not live across it.
    let roots = pyre_object::gc_roots::push_roots();
    let mut live = Vec::with_capacity(1 + args.len());
    live.push(callable);
    live.extend_from_slice(args);
    let base = roots.pin_roots(&live);
    let callable = roots.get(base);
    let mut positional = Vec::with_capacity(args.len());
    for i in 0..args.len() {
        positional.push(roots.get(base + 1 + i));
    }
    let w_code = unsafe { pyre_interpreter::getcode(callable) };
    let filled = scope_args_for_jit_callee(callable, w_code, &positional);
    let callable = roots.get(base);
    let w_code = unsafe { pyre_interpreter::getcode(callable) };
    let w_globals = unsafe { function_get_globals_obj(callable) };
    alloc_callee_frame(w_code as *const (), &filled, w_globals, execution_context) as i64
}

/// Self-recursive entries carry code and globals as raw words. Pin them
/// across `pack_varargs` and reread, same as [`create_callee_frame_in_ctx`].
fn alloc_matched_self_recursive_frame(
    func_code: *const (),
    w_globals: PyObjectRef,
    args: &[PyObjectRef],
    execution_context: *const pyre_interpreter::PyExecutionContext,
) -> *mut PyFrame {
    let roots = pyre_object::gc_roots::push_roots();
    let mut live = Vec::with_capacity(2 + args.len());
    live.push(func_code as PyObjectRef);
    live.push(w_globals);
    live.extend_from_slice(args);
    let base = roots.pin_roots(&live);
    let mut positional = Vec::with_capacity(args.len());
    for i in 0..args.len() {
        positional.push(roots.get(base + 2 + i));
    }
    let func_code = roots.get(base) as *const ();
    let filled = scope_args_for_jit_code(func_code, &positional);
    let func_code = roots.get(base) as *const ();
    let w_globals = roots.get(base + 1);
    alloc_callee_frame(func_code, &filled, w_globals, execution_context)
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_0(caller_frame: i64, callable: i64) -> i64 {
    create_callee_frame_impl(caller_frame, callable, &[])
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_1(
    caller_frame: i64,
    callable: PyObjectRef,
    arg0: PyObjectRef,
) -> i64 {
    create_callee_frame_impl_1_boxed(caller_frame, callable, arg0)
}

/// Self-recursive single-arg variant.
///
/// This skips rediscovering code/globals from a function object and reuses the
/// caller frame's code/namespace/execution_context directly, which matches the
/// existing self-recursive raw helper path more closely.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_self_recursive_callee_frame_1(
    caller_frame: i64,
    arg0: PyObjectRef,
) -> i64 {
    debug_assert!(
        caller_frame != 0,
        "jit_create_self_recursive_callee_frame_1: caller_frame is null"
    );
    if caller_frame == 0 {
        // Invariant violation: fall back to heap allocation with a
        // minimal frame. RPython never aborts the process for JIT
        // invariant failures — it falls back to tracing abort or
        // blackhole resume.
        return 0;
    }
    create_self_recursive_callee_frame_impl_1_boxed(caller_frame, arg0)
}

/// Self-recursive raw-int variant: creates the frame WITHOUT boxing
/// the argument. The raw int is passed directly to compiled code via
/// CallAssemblerI inputargs. Boxing only happens on guard failure
/// (in force_fn / jit_force_self_recursive_call_raw_1).
///
/// RPython parity: compiled code uses jitframe slots, not PyFrame
/// locals. Frame locals are only needed for interpreter fallback.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_self_recursive_callee_frame_1_raw_int(
    caller_frame: *const PyFrame,
    raw_int_arg: i64,
) -> i64 {
    let caller = unsafe { &*(caller_frame) };
    let func_code = caller.pycode;
    let w_globals = caller.get_w_globals();
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[func_code as PyObjectRef, w_globals]);
    let boxed_at = roots.pin_roots(&[pyre_object::intobject::w_int_new(raw_int_arg)]);
    let execution_context = pyre_interpreter::call::getexecutioncontext();
    let func_code = roots.get(base) as *const ();
    let w_globals = roots.get(base + 1);
    let boxed = roots.get(boxed_at);

    let frame_ptr =
        alloc_matched_self_recursive_frame(func_code, w_globals, &[boxed], execution_context);
    if majit_metainterp::majit_log_enabled() {
        let f = unsafe { &*frame_ptr };
        eprintln!(
            "[jit][ca-frame-raw] ptr={frame_ptr:p} locals=0x{:x} local0=0x{:x} vsd={} raw_arg={}",
            f.locals_cells_stack_w as usize,
            locals_w!(f)[0] as usize,
            f.valuestackdepth,
            raw_int_arg,
        );
    }
    frame_ptr as i64
}

/// Raw-int variant: accepts a raw int and boxes it internally.
/// Eliminates trace_box_int CallI from the trace (boxing folded into frame creation).
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_1_raw_int(
    caller_frame: i64,
    callable: PyObjectRef,
    raw_int_arg: i64,
) -> i64 {
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[callable]);
    let boxed_at = roots.pin_roots(&[pyre_object::intobject::w_int_new(raw_int_arg)]);
    let callable = roots.get(base);
    let boxed = roots.get(boxed_at);
    create_callee_frame_impl_1_boxed(caller_frame, callable, boxed)
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_2(
    caller_frame: i64,
    callable: i64,
    arg0: PyObjectRef,
    arg1: PyObjectRef,
) -> i64 {
    create_callee_frame_impl(caller_frame, callable, &[arg0, arg1])
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_3(
    caller_frame: i64,
    callable: i64,
    arg0: PyObjectRef,
    arg1: PyObjectRef,
    arg2: PyObjectRef,
) -> i64 {
    create_callee_frame_impl(caller_frame, callable, &[arg0, arg1, arg2])
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_4(
    caller_frame: i64,
    callable: i64,
    arg0: PyObjectRef,
    arg1: PyObjectRef,
    arg2: PyObjectRef,
    arg3: PyObjectRef,
) -> i64 {
    create_callee_frame_impl(caller_frame, callable, &[arg0, arg1, arg2, arg3])
}

pub fn callee_frame_helper(nargs: usize) -> Option<*const ()> {
    match nargs {
        0 => Some(jit_create_callee_frame_0 as *const ()),
        1 => Some(jit_create_callee_frame_1 as *const ()),
        2 => Some(jit_create_callee_frame_2 as *const ()),
        3 => Some(jit_create_callee_frame_3 as *const ()),
        4 => Some(jit_create_callee_frame_4 as *const ()),
        _ => None,
    }
}

/// Force callee and return BOXED result for the retired inline-call path.
/// warmspot.py:449 result_type=REF: jit_force_callee_frame already
/// returns boxed Ref, so this is just a pass-through.
#[majit_macros::jit_may_force]
pub extern "C" fn jit_force_callee_frame_boxed(frame_ptr: i64) -> i64 {
    jit_force_callee_frame(frame_ptr)
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_drop_callee_frame(frame_ptr: i64) {
    if frame_ptr & 1 != 0 {
        return;
    }
    let ptr = frame_ptr as *mut PyFrame;
    if majit_metainterp::majit_log_enabled() {
        eprintln!("[jit][ca-drop] ptr={ptr:p}");
    }
    unroot_callee_frame(ptr);
}

// Blackhole helper functions
//
// RPython blackhole.py: bhimpl_recursive_call_i, bhimpl_residual_call_*
//
// These are called by the BlackholeInterpreter through JitCode.fn_ptrs.
// Residual calls execute without accidental JIT re-entry; recursive portal
// calls are routed explicitly through the jitdriver's portal runner.

fn bh_call_self_recursive_portal(
    ec: *const pyre_interpreter::PyExecutionContext,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> Option<i64> {
    // "Is the caller running this same code object?" is an identity test on
    // the chain slot, which is what `vref_referent` exists for: it returns
    // what `topframeref` NAMES, without forcing.  `gettopframe_raw` forces,
    // and a force while the trace records clears the
    // `TOKEN_TRACING_RESCALL` marker (`virtualref.py:161-167`) that
    // `tracing_after_residual_call` reads back as "the callee escaped this
    // vref" — the probe would report its own read as an escape, and the
    // recorded `VIRTUAL_REF_FINISH` then materializes the caller frame on
    // every execution of the compiled trace.
    //
    // A null referent names no reachable frame (a caller that is still
    // virtual), so there is nothing to compare: decline, and the generic
    // residual below runs the call.
    let parent_frame_ptr =
        unsafe { pyre_interpreter::executioncontext::vref_referent((*ec).topframeref) };
    if parent_frame_ptr.is_null() {
        return None;
    }
    let parent_frame = unsafe { &*parent_frame_ptr };
    let callable_code = unsafe { pyre_interpreter::getcode(callable) };
    if parent_frame.pycode != callable_code as *const () {
        return None;
    }
    if !recursive_force_cache_safe(callable) {
        return None;
    }

    // blackhole.py get_portal_runner bhimpl_recursive_call_* reaches the
    // jitdriver's portal runner.  This branch narrows pyre's generic
    // Python CALL helper back to that shape for self-recursive portal
    // calls; non-recursive residual calls below remain opaque plain calls.
    let frame_ptr = create_callee_frame_in_ctx(ec, callable, args);
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        // The frame was built on the line above, so this entry is the callee's
        // activation and owes `execute_frame`'s hook bracket.
        crate::eval::portal_activation_result(frame)
    };
    jit_drop_callee_frame(frame_ptr);
    Some(match result {
        Ok(result) => result as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj);
            0
        }
    })
}

/// RPython: bhimpl_recursive_call_i — call a Python function in blackhole mode.
///
/// The blackhole pops callable and args into registers before calling this.
/// blackhole.py bhimpl_residual_call parity: variable-arity call helper.
///
/// Convention: residual_call_r_r dispatches with
/// args=[callable, null_or_self, arg0, ..., argN].  RPython
/// `bhimpl_residual_call_r_r` (blackhole.py) carries no frame —
/// `cpu.bh_call_r(func, None, args_r, ...)` — and `bh_call_fn_impl` carries
/// none either: it takes `space.getexecutioncontext()` and threads that
/// through the dispatch, so no branch resolves a caller frame out of
/// `topframeref`.  `null_or_self` is the CALL opcode's self slot
/// (`eval.rs`'s `PyFrame::call`): non-null means a method receiver to prepend as
/// arg0, NULL means a plain call.
///
/// For nargs=0: fn(callable, null_or_self) → 2 args
/// For nargs=1: fn(callable, null_or_self, arg0) → 3 args
/// For nargs=2: fn(callable, null_or_self, arg0, arg1) → 4 args
/// etc.
pub extern "C" fn bh_call_fn(
    callable: PyObjectRef,
    null_or_self: PyObjectRef,
    arg0: PyObjectRef,
) -> i64 {
    bh_call_fn_impl(callable, null_or_self, &[arg0])
}

pub extern "C" fn bh_call_fn_0(callable: PyObjectRef, null_or_self: PyObjectRef) -> i64 {
    bh_call_fn_impl(callable, null_or_self, &[])
}

pub extern "C" fn bh_call_fn_2(
    callable: PyObjectRef,
    null_or_self: PyObjectRef,
    arg0: PyObjectRef,
    arg1: PyObjectRef,
) -> i64 {
    bh_call_fn_impl(callable, null_or_self, &[arg0, arg1])
}

pub extern "C" fn bh_call_fn_3(
    callable: PyObjectRef,
    null_or_self: PyObjectRef,
    a0: PyObjectRef,
    a1: PyObjectRef,
    a2: PyObjectRef,
) -> i64 {
    bh_call_fn_impl(callable, null_or_self, &[a0, a1, a2])
}

pub extern "C" fn bh_call_fn_4(
    callable: PyObjectRef,
    null_or_self: PyObjectRef,
    a0: PyObjectRef,
    a1: PyObjectRef,
    a2: PyObjectRef,
    a3: PyObjectRef,
) -> i64 {
    bh_call_fn_impl(callable, null_or_self, &[a0, a1, a2, a3])
}

pub extern "C" fn bh_call_fn_5(
    callable: PyObjectRef,
    null_or_self: PyObjectRef,
    a0: PyObjectRef,
    a1: PyObjectRef,
    a2: PyObjectRef,
    a3: PyObjectRef,
    a4: PyObjectRef,
) -> i64 {
    bh_call_fn_impl(callable, null_or_self, &[a0, a1, a2, a3, a4])
}

pub extern "C" fn bh_call_fn_6(
    callable: PyObjectRef,
    null_or_self: PyObjectRef,
    a0: PyObjectRef,
    a1: PyObjectRef,
    a2: PyObjectRef,
    a3: PyObjectRef,
    a4: PyObjectRef,
    a5: PyObjectRef,
) -> i64 {
    bh_call_fn_impl(callable, null_or_self, &[a0, a1, a2, a3, a4, a5])
}

pub extern "C" fn bh_call_fn_7(
    callable: PyObjectRef,
    null_or_self: PyObjectRef,
    a0: PyObjectRef,
    a1: PyObjectRef,
    a2: PyObjectRef,
    a3: PyObjectRef,
    a4: PyObjectRef,
    a5: PyObjectRef,
    a6: PyObjectRef,
) -> i64 {
    bh_call_fn_impl(callable, null_or_self, &[a0, a1, a2, a3, a4, a5, a6])
}

pub extern "C" fn bh_call_fn_8(
    callable: PyObjectRef,
    null_or_self: PyObjectRef,
    a0: PyObjectRef,
    a1: PyObjectRef,
    a2: PyObjectRef,
    a3: PyObjectRef,
    a4: PyObjectRef,
    a5: PyObjectRef,
    a6: PyObjectRef,
    a7: PyObjectRef,
) -> i64 {
    bh_call_fn_impl(callable, null_or_self, &[a0, a1, a2, a3, a4, a5, a6, a7])
}

/// Per-arity `bh_call_fn_<n>` thunks for nargs 9..=14, sharing the
/// `(callable, null_or_self, arg0..arg{n-1})` ABI of the explicit
/// `bh_call_fn_0..8` above.  nargs=14 (16 i64 params) is the ceiling
/// the backend dispatch table (`call_stub.rs::dispatch_arity_body!`,
/// `MAX_HOST_CALL_ARITY` = 16) supports; CALL with nargs > 14 falls
/// through to `emit_abort_permanent!`.
macro_rules! bh_call_fn_arity {
    ($name:ident; $($arg:ident),+ $(,)?) => {
        pub extern "C" fn $name(
            callable: PyObjectRef,
            null_or_self: PyObjectRef,
            $($arg: PyObjectRef),+
        ) -> i64 {
            bh_call_fn_impl(callable, null_or_self, &[$($arg),+])
        }
    };
}
bh_call_fn_arity!(bh_call_fn_9; a0, a1, a2, a3, a4, a5, a6, a7, a8);
bh_call_fn_arity!(bh_call_fn_10; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
bh_call_fn_arity!(bh_call_fn_11; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
bh_call_fn_arity!(bh_call_fn_12; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
bh_call_fn_arity!(bh_call_fn_13; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
bh_call_fn_arity!(bh_call_fn_14; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);

/// CALL_KW residual shared body (`call_kw(callable, self_or_null,
/// positional, kwnames)` HLOp → per-arity `residual_call_r_r`).  Carries no
/// frame, like [`bh_call_fn_impl`]: `pyopcode.py CALL_FUNCTION_KW`
/// settles the C-profile question against its own frame and then calls the
/// frameless `space.call_args(w_function, args)`, which is what a residual
/// reaches.  Keyword resolution + the dispatched call run under
/// `force_plain_eval` (blackhole.py `bhimpl_residual_call_*` is an
/// opaque CPU call — no JIT re-entry; `call_kw`'s
/// `call_user_function_resolved` fast path routes through the JIT-aware
/// `get_eval_fn()`, so the guard is what keeps it on `eval_frame_plain`).
/// `positional` is `arg0..argN-1` in positional order (keyword tail
/// included); `kwnames` is the constant kwnames tuple.  MayForce: keyword
/// binding and the dispatched call may run Python.
fn bh_call_kw_impl(
    callable: PyObjectRef,
    null_or_self: PyObjectRef,
    kwnames: PyObjectRef,
    positional: &[PyObjectRef],
) -> i64 {
    let ec = pyre_interpreter::call::getexecutioncontext();
    assert!(
        !ec.is_null(),
        "bh_call_kw_impl requires a pinned execution context from \
         getexecutioncontext(); the eval loop must pin the execution context \
         before any residual call"
    );
    // `pyopcode.py CALL_FUNCTION_KW`'s C-profile diversion — see
    // [`residual_call_c_profile_frame`].  `call_kw` is `call_kw_in_ctx` with
    // the frame that settles it, plus the caller `FrameLocalsRoot` the
    // interpreter's own CALL_KW installs.
    let profile_frame = residual_call_c_profile_frame(ec, callable);
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    pyre_interpreter::call::set_last_exec_ctx(ec);
    let result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        if profile_frame.is_null() {
            pyre_interpreter::call::call_kw_in_ctx(ec, callable, null_or_self, positional, kwnames)
        } else {
            pyre_interpreter::call::call_kw(
                unsafe { &mut *profile_frame },
                callable,
                null_or_self,
                positional,
                kwnames,
            )
        }
    };
    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
    match result {
        Ok(result) => result as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            0
        }
    }
}

/// Per-arity `bh_call_kw_<n>` thunks for the CALL_KW residual, ABI
/// `(callable, null_or_self, kwnames, arg0..arg{n-1})` = 3 + n i64 params.
/// The backend dispatch tops out at `MAX_HOST_CALL_ARITY` = 16 i64 args, so
/// the kwnames slot leaves room for nargs 0..=13; CALL_KW with nargs > 13
/// falls through to `emit_abort_permanent!`.
macro_rules! bh_call_kw_arity {
    ($name:ident; $($arg:ident),* $(,)?) => {
        pub extern "C" fn $name(
            callable: PyObjectRef,
            null_or_self: PyObjectRef,
            kwnames: PyObjectRef,
            $($arg: PyObjectRef),*
        ) -> i64 {
            bh_call_kw_impl(callable, null_or_self, kwnames, &[$($arg),*])
        }
    };
}
bh_call_kw_arity!(bh_call_kw_0;);
bh_call_kw_arity!(bh_call_kw_1; a0);
bh_call_kw_arity!(bh_call_kw_2; a0, a1);
bh_call_kw_arity!(bh_call_kw_3; a0, a1, a2);
bh_call_kw_arity!(bh_call_kw_4; a0, a1, a2, a3);
bh_call_kw_arity!(bh_call_kw_5; a0, a1, a2, a3, a4);
bh_call_kw_arity!(bh_call_kw_6; a0, a1, a2, a3, a4, a5);
bh_call_kw_arity!(bh_call_kw_7; a0, a1, a2, a3, a4, a5, a6);
bh_call_kw_arity!(bh_call_kw_8; a0, a1, a2, a3, a4, a5, a6, a7);
bh_call_kw_arity!(bh_call_kw_9; a0, a1, a2, a3, a4, a5, a6, a7, a8);
bh_call_kw_arity!(bh_call_kw_10; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
bh_call_kw_arity!(bh_call_kw_11; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
bh_call_kw_arity!(bh_call_kw_12; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
bh_call_kw_arity!(bh_call_kw_13; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);

/// blackhole.py:1224 bhimpl_residual_call: cpu.bh_call_r.
/// RPython: cpu.bh_call_r (llmodel.py) invokes calldescr.call_stub_r
/// directly — a plain function-pointer call, no portal_runner indirection.
/// Only bhimpl_recursive_call_* (blackhole.py get_portal_runner) uses the portal
/// runner to re-enter JIT.
///
/// RPython `bhimpl_residual_call_r_r` (blackhole.py) carries no
/// frame: `cpu.bh_call_r(func, None, args_r, ...)`.  The user-function
/// dispatch below reads `execution_context` and the recursive-portal probe
/// walks the caller chain, so the parent frame is resolved here from the
/// execution context's top frame (`space.getexecutioncontext()
/// .gettopframe()`), matching the upstream frame-less ABI.
/// Whether the `PYRE_BH_NULL_ARG` residual-argument diagnostic is armed.
fn bh_null_arg_diag() -> bool {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ARMED.get_or_init(|| std::env::var_os("PYRE_BH_NULL_ARG").is_some())
}

/// The frame a CALL-family residual owes `c_call` / `c_return` to, or null.
///
/// `baseobjspace.py call_valuestack` diverts to `call_args_and_c_profile` when
/// the executing frame is profiled AND the callable wraps a builtin code;
/// `pyopcode.py` CALL_FUNCTION_KW and CALL_FUNCTION_EX take the same diversion
/// with their own frame.  `eval.rs`'s own CALL asks it at the same point, on
/// the RAW callable before the `_Method` unwrap, which is where `w_func` is
/// read upstream — `function.py is_builtin_code` does that unwrap itself.
///
/// A residual of one of those three helpers IS the bytecode's own dispatch:
/// the codewriter emits one only for the CALL family.  So the frame the events
/// belong to is this context's top frame, and the distinction `call.rs
/// c_profile_frame` draws — the bytecode's dispatch reports, one made from
/// inside `descr_call` does not — is settled here by construction rather than
/// by a test.  Without this the residuals reach the builtin through the
/// frameless `space.call_args`, where the reporting arm does not exist.
///
/// `profilefunc` is read before the frame because `gettopframe_raw` is
/// `force_vref`: asking the frame first would materialize the virtualizable on
/// every residual call a compiled loop makes.  It cannot change the answer,
/// since `executioncontext.py call_trace` sets `is_being_profiled` only while
/// `profilefunc is not None`.  It can leave a frame whose profiler was
/// uninstalled mid-loop carrying a stale flag one entry longer than the plain
/// interpreter would, which costs that frame the profiled green key until its
/// next interpreted call clears it, and reports nothing either way.
///
/// Reading the top frame is only correct while no walker-inlined callee can be
/// executing under a profiler, because an inlined callee has no frame on
/// `topframeref` / `f_backref` — a builtin residual firing inside one would
/// report the inliner's frame. Two independent guards close that: at record
/// time `inline_call.rs` declines on `ec_hook_installed()`, so a trace recorded
/// under a hook inlines nothing; at execute time `is_being_profiled` is a
/// portal green, so a profiled frame cannot enter a trace compiled under the
/// unprofiled key, and a frame that predates the hook keeps the flag false and
/// is declined by the test below — which is what `call_valuestack` does
/// upstream, gating on the same per-frame flag.
/// `c_call_reports_the_callee_frame_not_the_inliners` pins both arms against
/// cpython and pypy3. Relaxing the inline decline breaks the first guard.
fn residual_call_c_profile_frame(
    ec: *const pyre_interpreter::PyExecutionContext,
    callable: PyObjectRef,
) -> *mut pyre_interpreter::PyFrame {
    if ec.is_null() || unsafe { (*ec).profilefunc.is_none() } {
        return std::ptr::null_mut();
    }
    if !pyre_interpreter::function::is_builtin_code(callable) {
        return std::ptr::null_mut();
    }
    let frame = unsafe { (*ec).gettopframe_raw() };
    if frame.is_null() || !unsafe { (*frame).get_is_being_profiled() } {
        return std::ptr::null_mut();
    }
    frame
}

fn bh_call_fn_impl(callable: PyObjectRef, null_or_self: PyObjectRef, args: &[PyObjectRef]) -> i64 {
    // RPython's blackhole residual-call thunk is itself translated, so its
    // incoming GCREF arguments are shadowstack roots for the whole helper
    // call.  This Rust ABI boundary is outside that transform: the backend
    // gcmap keeps the caller objects alive, but a moving collection cannot
    // rewrite these copied native parameters.  Root them immediately and
    // dispatch only values reloaded from the forwarded slots.
    //
    // Both runs are published before the one normalize, so no argument is
    // still unpublished when a foreign collector can first run, and the
    // arguments land contiguously after the two leading slots.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = _roots.publish(&[callable, null_or_self]);
    _roots.publish(args);
    _roots.normalize(root_base, 2 + args.len());
    // `eval.rs`'s `PyFrame::call` — a non-null null_or_self is the method receiver
    // (load_method_fast_path pushes `[w_descr, w_obj]`); the call proceeds
    // as `callable(null_or_self, *args)`.
    let reload_args = || {
        let null_or_self = _roots.get(root_base + 1);
        let mut values = Vec::with_capacity(args.len() + usize::from(!null_or_self.is_null()));
        if !null_or_self.is_null() {
            values.push(null_or_self);
        }
        values.extend((0..args.len()).map(|i| _roots.get(root_base + 2 + i)));
        values
    };
    // `space.getexecutioncontext()` (`call.rs`'s `getexecutioncontext` →
    // TLS-pinned EC the eval loop stamps on entry).  This is the whole of the
    // caller-side state the residual needs: `bhimpl_residual_call_r_r` takes no
    // frame, and every
    // frame the dispatch below builds takes its execution context from the
    // space rather than from a caller frame.  A null here means the EC was
    // never pinned before a residual call, which is a wiring bug, so
    // fail-fast rather than building callee frames against a null context.
    let ec = pyre_interpreter::call::getexecutioncontext();
    assert!(
        !ec.is_null(),
        "bh_call_fn_impl requires a pinned execution context from \
         getexecutioncontext(); the eval loop must pin the execution context \
         before any residual call"
    );
    if _roots.get(root_base).is_null() {
        let mut err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            "call on null callable".to_string(),
        );
        publish_residual_call_exception(err.to_exc_object());
        return 0;
    }
    // Diagnostic (`PYRE_BH_NULL_ARG`): only `null_or_self` carries a checked
    // `PY_NULL` sentinel here (`walker_abort_if_mayforce_null_ref_arg` exempts
    // exactly that index).  A NULL in a real argument slot means the operand
    // stack the blackhole read was never published, so the callee dereferences
    // it — report the Python coordinate before that happens instead of leaving
    // a SIGSEGV inside the callee as the only evidence.
    if bh_null_arg_diag() {
        // The Python coordinate comes off the caller frame, which the dispatch
        // itself never resolves — force it here, inside the diagnostic, so an
        // unarmed run leaves the frame virtual.
        // `descr_repr` fsdecodes the code filename, and a filename that fails
        // to decode reaches a registered decode error handler — Python, and so
        // a collection.  The next iteration reads the frame again, so hold it
        // on the shadow stack and re-read rather than carry the pointer.
        let frame_anchor =
            unsafe { pyre_interpreter::eval::FrameAnchor::from_raw((*ec).gettopframe_raw()) };
        for i in 0..args.len() {
            let parent_frame_ptr = frame_anchor.live();
            if !parent_frame_ptr.is_null() && _roots.get(root_base + 2 + i).is_null() {
                let frame = unsafe { &*parent_frame_ptr };
                // A NULL here with a live fastlocal is an unbound blackhole
                // register (the resume never seeded the slot); a NULL fastlocal
                // means the frame itself lost the binding.
                let locals_w = locals_w!(frame);
                let shown = locals_w.len().min(8);
                let locals: Vec<Option<usize>> = (0..shown)
                    .map(|i| {
                        let v = locals_w[i];
                        (!v.is_null()).then_some(v as usize)
                    })
                    .collect();
                eprintln!(
                    "[bh-null-arg] arg[{i}]/{} NULL vsd={} base={} locals={locals:?} at {}",
                    args.len(),
                    frame.valuestackdepth,
                    frame.stack_base(),
                    frame.descr_repr(),
                );
            }
        }
    }
    // `call_valuestack`'s C-profile diversion, ahead of every fast path below:
    // each of them reaches the builtin through an entry that takes no frame
    // (`builtin_code_call`, `builtin_code_call_positional`), so the arm that
    // reports would never run.  [`residual_call_c_profile_frame`] carries the
    // whole rule; `call_callable` is the same door the plain interpreter's
    // CALL takes for this shape, receiver-prefixed argument vector included.
    let profile_frame = residual_call_c_profile_frame(ec, _roots.get(root_base));
    if !profile_frame.is_null() {
        let call_args = reload_args();
        let callable = _roots.get(root_base);
        // The two hooks are application-level Python and can enter compiled
        // code, which writes the very cells this helper publishes to — the
        // same nesting the user-function arm below parks across, and for the
        // same reason: a raise the hook handled internally would otherwise be
        // read by this call's `GUARD_NO_EXCEPTION` as a pending exception.
        let parked = park_residual_call_exception();
        let result = pyre_interpreter::call::call_callable(
            unsafe { &mut *profile_frame },
            callable,
            &call_args,
        );
        unpark_residual_call_exception(parked);
        return match result {
            Ok(result) => result as i64,
            Err(mut err) => {
                publish_residual_call_exception(err.to_exc_object());
                0
            }
        };
    }
    // A materialized bound Method normally reaches the generic callable
    // dispatcher, which unwraps it, allocates a receiver-prefixed argument
    // vector, and dispatches the underlying Function a second time.  The
    // method fields are immutable, so an exact fixed-arity BuiltinCode can
    // take the same gateway call directly.  Keep every signature-dependent
    // shape on the generic path: only a natural arity in the fixed range with
    // an exact argument count is equivalent to the positional fast path.
    let rooted_callable = _roots.get(root_base);
    let rooted_null_or_self = _roots.get(root_base + 1);
    let bound_builtin = unsafe {
        if !rooted_null_or_self.is_null() && pyre_interpreter::is_function(rooted_callable) {
            Some((rooted_callable, rooted_null_or_self))
        } else if rooted_null_or_self.is_null() && pyre_object::is_method(rooted_callable) {
            let function = pyre_object::w_method_get_func(rooted_callable);
            let receiver = pyre_object::w_method_get_self(rooted_callable);
            (!function.is_null() && !receiver.is_null()).then_some((function, receiver))
        } else {
            None
        }
    };
    if let Some((function, receiver)) = bound_builtin {
        let code = unsafe { pyre_interpreter::getcode(function) } as PyObjectRef;
        let positional_count = args.len() + 1;
        let exact_fixed_arity = !code.is_null()
            && unsafe { pyre_interpreter::is_builtin_code(code) }
            && positional_count <= 4
            && unsafe { pyre_interpreter::builtin_code_get_fast_natural_arity(code) as usize }
                == positional_count;
        if exact_fixed_arity {
            let extra = _roots.publish(&[code, receiver]);
            _roots.normalize(extra, 2);
            let code_slot = extra;
            let receiver_slot = extra + 1;
            let mut call_args = [pyre_object::PY_NULL; 4];
            call_args[0] = _roots.get(receiver_slot);
            for (index, slot) in call_args[1..positional_count].iter_mut().enumerate() {
                *slot = _roots.get(root_base + 2 + index);
            }
            let code = _roots.get(code_slot);
            return match unsafe {
                pyre_interpreter::builtin_code_call(code, &call_args[..positional_count])
            } {
                Ok(result) if !result.is_null() => result as i64,
                Ok(_) => 0,
                Err(mut err) => {
                    publish_residual_call_exception(err.to_exc_object());
                    0
                }
            };
        }
    }
    // llmodel.py bh_call_r — calldescr.call_stub_r is callable-type-agnostic.
    // Hot path: Function callables dispatched directly here (builtin or user
    // code), matching call_user_function_plain so eval_frame_plain is used
    // and JIT is not re-entered from the blackhole.
    // Cold path: type/method/staticmethod/classmethod/callable-instance are
    // delegated to call_function_impl_result under ForcePlainEvalGuard, which
    // mirrors baseobjspace.py dispatch without re-entering the JIT.
    let callable = _roots.get(root_base);
    if unsafe { is_function(callable) } {
        let code = unsafe { pyre_interpreter::getcode(callable) };
        if unsafe { pyre_interpreter::is_builtin_code(code as pyre_object::PyObjectRef) } {
            // `reload_args` allocates a vector per residual builtin call.  A
            // call with no bound receiver and at most four positionals reads
            // them straight out of the shadow slots instead — the same
            // allocation-free shape the bound-receiver arm above takes.  The
            // slice contents are identical either way, so the dispatch below
            // is unchanged.
            let mut inline_args = [pyre_object::PY_NULL; 4];
            let spilled_args;
            let call_args: &[pyre_object::PyObjectRef] =
                if args.len() <= inline_args.len() && _roots.get(root_base + 1).is_null() {
                    for (index, slot) in inline_args[..args.len()].iter_mut().enumerate() {
                        *slot = _roots.get(root_base + 2 + index);
                    }
                    &inline_args[..args.len()]
                } else {
                    spilled_args = reload_args();
                    &spilled_args
                };
            // `call_args` are raw positionals; a HOPELESS-arity Signature
            // (`*args`, optional positional) needs `_match_signature` binding
            // before the body reads its slots.  `builtin_code_call` never binds,
            // so route through the positional entry that does.
            return match pyre_interpreter::call::builtin_code_call_positional(
                code as pyre_object::PyObjectRef,
                call_args,
            ) {
                Ok(result) if !result.is_null() => result as i64,
                Ok(_) => 0,
                Err(mut err) => {
                    publish_residual_call_exception(err.to_exc_object());
                    0
                }
            };
        }
        let call_args = reload_args();
        if let Some(result) = bh_call_self_recursive_portal(ec, callable, &call_args) {
            return result;
        }
        let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
        // `set_last_exec_ctx` mirrors what the portal runner does on frame
        // re-entry so user functions invoked from the residual path observe
        // the caller's execution context.  That is `ec` itself: a frame's
        // `execution_context` is the context it was built under, and the top
        // frame of this EC was built under this EC.
        pyre_interpreter::call::set_last_exec_ctx(ec);
        // `blackhole.py bhimpl_residual_call_r_i bhimpl_residual_call_*` is opaque to the TRACE,
        // not to the JIT: `cpu.bh_call_*(func, ...)` runs the translated
        // callee, and a callee whose graph reaches a `jit_merge_point`
        // (`execute_frame` does) enters the JIT normally.  Pinning the callee
        // and its whole subtree to `eval_frame_plain` here left a hot inner
        // loop interpreted for as long as its caller's loop was compiled.
        // `bhimpl_recursive_call_*` (`blackhole.py get_portal_runner`) is not the
        // only way back to the portal — it is the form the codewriter emits
        // when the callee is *statically* the portal graph.
        //
        // Re-entrant TRACING stays blocked where upstream blocks it, on the
        // green key (`warmstate.py` JC_TRACING, mirrored by
        // `maybe_compile_and_run`'s `driver.is_tracing()`).
        //
        // Letting the callee re-enter the JIT also lets a nested compiled /
        // blackhole execution run inside this helper, and that nesting writes
        // the very cells this helper publishes to: `BH_LAST_EXC_VALUE` and the
        // backend `_store_exception` pair. Upstream cannot alias them — the
        // raise lives in `metainterp.last_exc_value`, a field of the
        // MetaInterp instance that owns the call, and `llmodel.py
        // _store_exception` is read back by the same `bh_call_*` that armed
        // it. Park the caller's pair across the nested run so this helper's
        // own outcome is the only thing it publishes; a nested raise the
        // callee handled internally would otherwise be read by the caller's
        // `GUARD_NO_EXCEPTION` as a spurious pending exception (the failure
        // `drain_backend_jit_exc` names for the walker's snapshot side).
        let parked = park_residual_call_exception();
        let result = pyre_interpreter::call::call_user_function_with_ctx(ec, callable, &call_args);
        unpark_residual_call_exception(parked);
        pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
        return match result {
            Ok(result) => result as i64,
            Err(mut err) => {
                publish_residual_call_exception(err.to_exc_object());
                0
            }
        };
    }
    // Cold path: type/method/staticmethod/classmethod/callable-instance.
    // Ensure LAST_EXEC_CTX reflects the calling context before delegating to
    // `call_function_impl_result`. `type_descr_call_impl` →
    // `call_user_function_with_args` reads LAST_EXEC_CTX as the fallback
    // execution context for `__new__`/`__init__` (call.rs);
    // without this pin it would use whatever frame last entered
    // `eval_frame_*`, which is not guaranteed to be the blackhole caller.
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    pyre_interpreter::call::set_last_exec_ctx(ec);
    let _plain_guard = pyre_interpreter::call::force_plain_eval();
    let callable = _roots.get(root_base);
    let call_args = reload_args();
    let result = pyre_interpreter::call::call_function_impl_result(callable, &call_args);
    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
    match result {
        Ok(result) => result as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            0
        }
    }
}

/// CALL_FUNCTION_EX residual (`call_function_ex(callable, self_or_null,
/// starargs, kwargs_or_null)` HLOp → `residual_call_r_r`).  Unpacks the
/// `*` iterable and merges the `**` mapping through the shared
/// `call::call_function_ex`, then dispatches.  Carries no frame, like
/// [`bh_call_fn_impl`]: `pyopcode.py CALL_FUNCTION_EX` settles the
/// C-profile question against its own frame and then calls the frameless
/// `space.call_args(w_function, args)`, which is what a residual reaches.
/// The nested Python call runs under `force_plain_eval` (blackhole.py bhimpl_residual_call_r_i
/// `bhimpl_residual_call_*` is an opaque CPU call — no JIT re-entry).
/// MayForce: unpacking an arbitrary iterable / mapping and the dispatched
/// call may run Python.
pub extern "C" fn bh_call_function_ex_fn(
    callable: PyObjectRef,
    self_or_null: PyObjectRef,
    starargs: PyObjectRef,
    kwargs_or_null: PyObjectRef,
) -> PyObjectRef {
    let ec = pyre_interpreter::call::getexecutioncontext();
    assert!(
        !ec.is_null(),
        "bh_call_function_ex_fn requires a pinned execution context from \
         getexecutioncontext(); the eval loop must pin the execution context \
         before any residual call"
    );
    // `pyopcode.py CALL_FUNCTION_EX`'s C-profile diversion — see
    // [`residual_call_c_profile_frame`].
    let profile_frame = residual_call_c_profile_frame(ec, callable);
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    pyre_interpreter::call::set_last_exec_ctx(ec);
    let result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        if profile_frame.is_null() {
            pyre_interpreter::call::call_function_ex_in_ctx(
                ec,
                callable,
                self_or_null,
                starargs,
                kwargs_or_null,
            )
        } else {
            pyre_interpreter::call::call_function_ex(
                unsafe { &mut *profile_frame },
                callable,
                self_or_null,
                starargs,
                kwargs_or_null,
            )
        }
    };
    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
    match result {
        Ok(result) => result,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// `_load_global` residual (pyopcode.py).  Resolves the namespace via
/// the executing frame's `get_w_globals_storage()` when the live frame OWNS this
/// `w_code` (`frame.pycode == w_code`) — honoring an `exec(code, ns)`
/// frame-specific namespace — and falls back to the callee's own promoted
/// `w_code` globals otherwise (an inlined / chained callee's frame register
/// aliases an outer frame, so it is not this code's frame).  The
/// `namespace_ptr` operand is ignored for the same aliasing reason; it
/// survives only as the cell-fold recogniser's hint.  The live frame is also
/// passed so `self.get_builtin()` works in compiled residual-call paths as
/// well as blackhole.  namei is the raw oparg from LOAD_GLOBAL:
/// name_idx = namei >> 1.
pub extern "C" fn bh_load_global_fn(
    namespace_ptr: PyObjectRef,
    w_code_ptr: PyObjectRef,
    frame_ptr: *mut PyFrame,
    namei: i64,
) -> PyObjectRef {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr) as *const pyre_interpreter::CodeObject)
    };
    let raw = namei as usize;
    let idx = raw >> 1;

    if idx >= code.names.len() {
        return PY_NULL;
    }

    let varname = code.names[idx].as_ref();
    let _ = namespace_ptr;
    // The builtins leg reads the frame again after the globals lookup, and
    // `finditem_str` reaches a user `__getitem__` on a non-dict mapping.  A
    // frame the JIT built lives in the nursery, so hold it on the shadow stack
    // and re-read at each use instead of carrying the operand across.
    let frame_anchor = unsafe { pyre_interpreter::eval::FrameAnchor::from_raw(frame_ptr) };
    let parent_frame_ptr = frame_anchor.live();
    // pypy/interpreter/pyopcode.py `_load_global`:
    //   w_value = self.space.finditem_str(self.get_w_globals_storage(), varname)
    //   if w_value is None:
    //       w_value = self.get_builtin().getdictvalue(self.space, varname)
    //       if w_value is None:
    //           self._load_global_failed(w_varname)
    //
    // `self.get_w_globals()` (pyframe.py) is the executing frame's own
    // globals object: the per-frame globals an `exec(code, ns)` installs,
    // falling back to the code's bound globals when there is no override.
    // Use it whenever the live frame OWNS this `w_code`
    // (`frame.pycode == w_code`), so a compiled `LOAD_GLOBAL` resolves
    // against the executing frame's globals — not the code's original module
    // dict — exactly as the interpreter does.
    //
    // A frame that does NOT own this `w_code` is an aliased OUTER frame on a
    // chained blackhole / inlined-callee resume (the same aliasing makes the
    // `namespace_ptr` operand unusable, hence ignored above).  Resolve from
    // the callee's own promoted `w_code` constant: reading `w_globals`
    // from it yields the current module dict object the const-folding
    // `frontend_global_flow_value` resolved statically, but live, so a
    // relocated dict (a growing `memo`) is followed instead of dangling.
    let w_globals = if !parent_frame_ptr.is_null()
        && unsafe { (*parent_frame_ptr).pycode } as usize == w_code_ptr as usize
    {
        unsafe { (*parent_frame_ptr).get_w_globals() }
    } else {
        unsafe { pyre_interpreter::w_code_get_w_globals(w_code_ptr as pyre_object::PyObjectRef) }
    };
    if !w_globals.is_null() {
        match pyre_interpreter::baseobjspace::finditem_str(w_globals, varname) {
            Ok(Some(w_value)) => return w_value,
            Ok(None) => {}
            Err(mut err) => {
                let exc_obj = err.to_exc_object();
                publish_residual_call_exception(exc_obj);
                return PY_NULL;
            }
        }
    }

    // Residual helper adaptation: `self` is the live portal frame passed as
    // an explicit Ref argument, so `self.get_builtin()` maps to
    // PyFrame::get_builtin() without relying on blackhole-only TLS.
    let parent_frame_ptr = frame_anchor.live();
    if !parent_frame_ptr.is_null() {
        let w_builtin = unsafe { (*parent_frame_ptr).get_builtin() };
        if !w_builtin.is_null() && unsafe { pyre_object::is_module(w_builtin) } {
            let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
            if !w_dict.is_null() {
                match pyre_interpreter::baseobjspace::finditem_str(w_dict, varname) {
                    Ok(Some(w_value)) => return w_value,
                    Ok(None) => {}
                    Err(mut err) => {
                        let exc_obj = err.to_exc_object();
                        publish_residual_call_exception(exc_obj);
                        return PY_NULL;
                    }
                }
            }
        }
    }

    // pyopcode.py `_load_global_failed`: raise NameError.
    let mut err = pyre_interpreter::PyError::new(
        pyre_interpreter::PyErrorKind::NameError,
        format!("name '{}' is not defined", varname),
    );
    let exc_obj = err.to_exc_object();
    publish_residual_call_exception(exc_obj);
    PY_NULL
}

/// LOAD_FROM_DICT_OR_GLOBALS residual (`load_from_dict_or_globals` HLOp →
/// `residual_call_ir_r`).  Mirrors `eval.rs::load_from_dict_or_globals`:
/// try `getitem(dict, name)` on the popped mapping first, then fall
/// back to the live frame's globals, else NameError.  `namei` is a direct
/// `code.names` index (no LOAD_GLOBAL push-null low-bit shift).
///
/// GC-safety: the globals are read from the LIVE frame when it owns this
/// `w_code` (`frame.pycode == w_code`), matching `bh_load_global_fn`, so a
/// relocated module dict is followed rather than a const-folded dangling
/// pointer.  A user `__getattr__`/`__getitem__` on the mapping may run
/// Python (`MayForce`).
pub extern "C" fn bh_load_from_dict_or_globals_fn(
    dict: PyObjectRef,
    w_code_ptr: PyObjectRef,
    frame_ptr: *mut PyFrame,
    namei: i64,
) -> PyObjectRef {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr) as *const pyre_interpreter::CodeObject)
    };
    let idx = namei as usize;
    if idx >= code.names.len() {
        return PY_NULL;
    }
    let varname = code.names[idx].as_ref();

    // CPython/PyPy LOAD_FROM_DICT_OR_GLOBALS uses mapping subscription, not
    // attribute lookup.  Preserve __getitem__/__missing__ and propagate every
    // exception except the KeyError that selects the globals fallback.
    // `pycode.py:127-129 co_names_w[idx]` — this residual already resolved the
    // name through `code.names[idx]`, so it names the key from the same index
    // instead of minting an immortal `w_str` on every execution.
    let key = unsafe {
        pyre_interpreter::pycode::w_code_getname_w_or_new(
            w_code_ptr as pyre_object::PyObjectRef,
            idx,
            varname,
        )
    };
    match pyre_interpreter::baseobjspace::getitem(dict, key) {
        Ok(val) => return val,
        Err(err) if matches!(err.kind, pyre_interpreter::PyErrorKind::KeyError) => {}
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            return PY_NULL;
        }
    }

    // Fall back to the live frame's globals (GC-safe when the frame owns
    // this w_code; else the promoted w_code's own globals).
    //
    // The globals lookup below reaches a user `__getitem__` on a non-dict
    // mapping, and the builtins leg after it reads both the frame and the
    // globals again.  The frame is held on the shadow stack and the globals
    // re-derived from it — a dict is one of the two kinds a minor collection
    // relocates, and the promoted `w_code` it otherwise comes off does not
    // move — so neither is carried across the lookup.
    let frame_anchor = unsafe { pyre_interpreter::eval::FrameAnchor::from_raw(frame_ptr) };
    let live_globals = |frame: *mut PyFrame| unsafe {
        if !frame.is_null() && (*frame).pycode as usize == w_code_ptr as usize {
            (*frame).get_w_globals()
        } else {
            pyre_interpreter::w_code_get_w_globals(w_code_ptr as pyre_object::PyObjectRef)
        }
    };
    let w_globals = live_globals(frame_anchor.live());
    if !w_globals.is_null() {
        match pyre_interpreter::baseobjspace::finditem_str(w_globals, varname) {
            Ok(Some(val)) => return val,
            Ok(None) => {}
            Err(mut err) => {
                publish_residual_call_exception(err.to_exc_object());
                return PY_NULL;
            }
        }
    }

    // `load_from_dict_or_globals` (eval.rs) finishes through
    // `load_global_value`, whose second leg is
    // `self.get_builtin().getdictvalue(space, varname)` (`pyopcode.py:979`).
    // This residual stopped after the globals leg, so a builtin name reached
    // here as a bogus `NameError`.
    //
    // A null frame is an anticipated input — the globals leg above already
    // treats it as one — so the builtin namespace is resolved from the globals
    // through `pick_builtin_obj_checked`, the same resolver whose result
    // `frame.get_builtin()` caches. Reading `globals["__builtins__"]` raw is
    // not equivalent: `exec`/`eval` plant the builtins *dict* rather than the
    // module (3.14 `PyEval_GetBuiltins`), and the `is_module` test below would
    // reject that spelling and report a bogus `NameError` for every builtin
    // name seen by a function created under an exec'd namespace.
    // `try_walker_load_global_cell_fold` declines the fold in that case and
    // leaves the name to this residual, so the handling has to live here.
    let parent_frame_ptr = frame_anchor.live();
    let w_builtin = if !parent_frame_ptr.is_null() {
        unsafe { (*parent_frame_ptr).get_builtin() }
    } else {
        match pyre_interpreter::baseobjspace::pick_builtin_obj_checked(
            live_globals(parent_frame_ptr),
            pyre_interpreter::call::take_last_exec_ctx(),
        ) {
            Ok(w_builtin) => w_builtin,
            Err(mut err) => {
                publish_residual_call_exception(err.to_exc_object());
                return PY_NULL;
            }
        }
    };
    if !w_builtin.is_null() && unsafe { pyre_object::is_module(w_builtin) } {
        let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
        if !w_dict.is_null() {
            match pyre_interpreter::baseobjspace::finditem_str(w_dict, varname) {
                Ok(Some(val)) => return val,
                Ok(None) => {}
                Err(mut err) => {
                    publish_residual_call_exception(err.to_exc_object());
                    return PY_NULL;
                }
            }
        }
    }

    let mut err = pyre_interpreter::PyError::oefmt_name_error_text(
        format!("name '{varname}' is not defined"),
        varname,
    );
    publish_residual_call_exception(err.to_exc_object());
    PY_NULL
}

/// `LOAD_ATTR` / method-form `LOAD_ATTR` residual for the standalone
/// (blackhole / deopt) per-CodeObject jitcode.  pyre's codewriter cannot
/// rtype `getattr` into `getfield_gc` (`rclass.py rtype_getattr`
/// rewrites `getattr` → `getfield_gc` after rtyping; pyre has no rtyper),
/// so the per-CodeObject jitcode lowers `LOAD_ATTR` to this residual call
/// rather than `abort_permanent`. The full-body walker traces the emitted
/// residual call.
///
/// Mirrors the interpreter `eval.rs` `load_attr` / `load_method` getattr
/// fallback (`baseobjspace::getattr_str`): returns the (possibly bound)
/// attribute.  For the method form the codewriter pushes the result plus
/// a `PY_NULL` self-slot; the bound method already carries `self`, so the
/// following `CALL` (`eval.rs`'s `PyFrame::call`) invokes it with
/// `null_or_self == NULL`, producing the same call as the unbound
/// `[w_descr, self]` fast path.  On `AttributeError` it sets
/// `BH_LAST_EXC_VALUE` and returns 0,
/// matching `bh_load_global_fn`'s NameError path.  `w_name` is the
/// interned immortal str constant the flatten driver lowers the getattr
/// HLOp's name operand to (`flatten_constant_operand` → `box_str_constant`),
/// the same interned-name ABI as `_pure_lookup_where_with_method_cache`.
pub extern "C" fn bh_getattr_fn(obj: PyObjectRef, w_name: PyObjectRef) -> PyObjectRef {
    // A lone surrogate is not a `&str` key. `getattr` is the existing
    // object-name path, which already declines to the surrogate lookup.
    let res = match unsafe { pyre_object::unicodeobject::w_str_get_value_opt(w_name) } {
        Some(name) => pyre_interpreter::baseobjspace::getattr_str(obj, name),
        None => pyre_interpreter::baseobjspace::getattr(obj, w_name),
    };
    match res {
        Ok(w_value) => w_value,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// Resolve `getattr(obj, code.names[name_idx])` for blackhole LOAD_ATTR
/// resume — the attribute / unbound-method half of LOOKUP_METHOD.  Mirrors
/// `PyFrame::load_method`'s `getattr` call so the blackhole reproduces the
/// interpreter's attribute lookup exactly (descriptor `__get__`, super
/// proxies, weakref forcing).  On error stashes the exception in
/// `BH_LAST_EXC_VALUE` and returns 0 (the trailing `-live-` lets the
/// blackhole route it through the except handler), matching
/// [`bh_load_global_fn`].
pub extern "C" fn bh_load_attr_fn(
    obj: PyObjectRef,
    w_code_ptr: PyObjectRef,
    name_idx: i64,
) -> PyObjectRef {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr) as *const pyre_interpreter::CodeObject)
    };
    // `name_idx` is a `co_names` index baked into the residual call by the
    // codewriter from the originating LOAD_ATTR oparg, so it is in range for
    // the code object loaded out of the resume frame.  The bound is a codegen
    // invariant rather than a runtime-reachable error (a negative index would
    // wrap to a huge `usize` and trip the same check), so assert it in debug
    // and degrade to a null result in release.
    let idx = name_idx as usize;
    debug_assert!(
        idx < code.names.len(),
        "bh_load_attr_fn name_idx {idx} out of range ({} names) — codegen invariant",
        code.names.len()
    );
    if idx >= code.names.len() {
        return PY_NULL;
    }
    let name = code.names[idx].as_ref();
    match pyre_interpreter::baseobjspace::getattr_str(obj, name) {
        Ok(attr) => attr,
        Err(mut err) => {
            // Publish the raise into BOTH the blackhole `BH_LAST_EXC_VALUE` and
            // the backend `_store_exception` cells (`publish_residual_call_exception`).
            // The LOAD_ATTR residual runs under the blackhole interpreter AND
            // inside a compiled trace; writing only `BH_LAST_EXC_VALUE` leaves a
            // compiled trace's `GUARD_NO_EXCEPTION` reading a stale 0, so the
            // guard wrongly passes and the NULL result flows to the consumer (a
            // raising property getter in a compiled loop is silently swallowed).
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj);
            PY_NULL
        }
    }
}

/// STORE_ATTR residual (`store_attr` HLOp → `residual_call_ir_v`).  The
/// symmetric counterpart of [`bh_load_attr_fn`]: resolves the attribute
/// name from the resume frame's code object via `name_idx` and runs the
/// generic `setattr_str` (may invoke user `__setattr__` → `MayForce`).
/// Void result, so always returns 0; an exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException`.
pub extern "C" fn bh_store_attr_fn(
    obj: PyObjectRef,
    value: PyObjectRef,
    w_code_ptr: PyObjectRef,
    name_idx: i64,
) -> i64 {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr) as *const pyre_interpreter::CodeObject)
    };
    // Same `co_names`-index codegen invariant as `bh_load_attr_fn`.
    let idx = name_idx as usize;
    debug_assert!(
        idx < code.names.len(),
        "bh_store_attr_fn name_idx {idx} out of range ({} names) — codegen invariant",
        code.names.len()
    );
    if idx >= code.names.len() {
        return 0;
    }
    let name = code.names[idx].as_ref();
    if let Err(mut err) = pyre_interpreter::baseobjspace::setattr_str(obj, name, value) {
        let exc_obj = err.to_exc_object();
        publish_residual_call_exception(exc_obj);
    }
    0
}

/// DELETE_ATTR residual (`delete_attr` HLOp → `residual_call_ir_v`).
/// Resolves the `co_names` name through the jitcode's own code object
/// (same invariant as `bh_store_attr_fn`) and runs `del obj.name` through
/// `baseobjspace::delattr_str`.  A user `__delattr__` may run Python
/// (`MayForce`); on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call
/// returns 0.
pub extern "C" fn bh_delete_attr_fn(
    obj: PyObjectRef,
    w_code_ptr: PyObjectRef,
    name_idx: i64,
) -> i64 {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr) as *const pyre_interpreter::CodeObject)
    };
    let idx = name_idx as usize;
    debug_assert!(
        idx < code.names.len(),
        "bh_delete_attr_fn name_idx {idx} out of range ({} names) — codegen invariant",
        code.names.len()
    );
    if idx >= code.names.len() {
        return 0;
    }
    let name = code.names[idx].as_ref();
    if let Err(mut err) = pyre_interpreter::baseobjspace::delattr_str(obj, name) {
        let exc_obj = err.to_exc_object();
        publish_residual_call_exception(exc_obj);
    }
    0
}

/// IMPORT_FROM residual (`import_from` HLOp → `residual_call_ir_r`).
/// Resolves the attribute name from the jitcode's own code object via
/// `name_idx` (same `co_names` invariant as `bh_load_attr_fn`) and runs
/// `importing::import_from(module, name)` — first `getattr` on the module,
/// then a `sys.modules` lookup for `<__name__>.<name>`.  Both can run a
/// user `__getattribute__` / `__getattr__` → `MayForce`.  `module` is the
/// peeked TOS (IMPORT_FROM does not pop it).  On error the exception is
/// published through `BH_LAST_EXC_VALUE` for the trailing `GuardNoException`
/// and the call returns 0.
pub extern "C" fn bh_import_from_fn(
    module: PyObjectRef,
    w_code: PyObjectRef,
    name_idx: i64,
) -> PyObjectRef {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject)
    };
    let idx = name_idx as usize;
    debug_assert!(
        idx < code.names.len(),
        "bh_import_from_fn name_idx {idx} out of range ({} names) — codegen invariant",
        code.names.len()
    );
    if idx >= code.names.len() {
        return PY_NULL;
    }
    let name = code.names[idx].as_ref();
    match pyre_interpreter::importing::import_from(module, name) {
        Ok(attr) => attr,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj);
            PY_NULL
        }
    }
}

/// LOAD_SUPER_ATTR residual (`load_super_attr` HLOp → `residual_call_ir_r`).
///
/// The codewriter emits this HLOp for the ZERO-argument oparg form only, so
/// `is_two_arg` arrives 0; the two-argument form is lowered to `simple_call`
/// plus `getattr` instead, which is the same two steps spelled as two ops.
/// The two-argument arm below is kept because it is what the operand means,
/// and because dropping it would leave the helper's shape disagreeing with the
/// HLOp's.
///
/// Resolves the attribute name from the jitcode's code object via `name_idx`
/// (same `co_names` invariant as `bh_load_attr_fn`), calls the actual global
/// `super` value with zero or two arguments, and runs `getattr` (both calls may
/// run Python → `MayForce`).  The zero-argument builtin path reads the
/// explicitly threaded red `frame`, matching PyPy's traced
/// `ec.gettopframe()` dependency. Returns the raw resolved attribute; the
/// `is_method` form
/// post-processes it through [`bh_super_attr_unwrap_fn`].  On error the
/// exception is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_load_super_attr_fn(
    global_super: PyObjectRef,
    self_obj: PyObjectRef,
    cls: PyObjectRef,
    frame: *mut pyre_interpreter::PyFrame,
    w_code_ptr: PyObjectRef,
    name_idx: i64,
    is_two_arg: i64,
) -> PyObjectRef {
    let idx = name_idx as usize;
    let result = pyre_interpreter::eval::load_super_attr_value(
        global_super,
        self_obj,
        cls,
        frame,
        w_code_ptr,
        idx,
        is_two_arg != 0,
    );
    match result {
        Ok(result) => result,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj);
            PY_NULL
        }
    }
}

/// LOAD_SUPER_ATTR method-form unwrap (`super_attr_unwrap` HLOp →
/// `residual_call_ir_r`).  Pure function of the raw attribute resolved by
/// [`bh_load_super_attr_fn`] — `which == 0` yields the func slot, `which ==
/// 1` the self slot.  When the attribute is a bound method, unwraps it to
/// `(func, receiver)`; otherwise (staticmethod / classmethod) yields
/// `(result, NULL)`.  Infallible and idempotent, so safe under the walk /
/// replay double-execution seam.
pub extern "C" fn bh_super_attr_unwrap_fn(result: PyObjectRef, which: i64) -> PyObjectRef {
    if unsafe { pyre_object::is_method(result) } {
        if which == 0 {
            unsafe { pyre_object::w_method_get_func(result) }
        } else {
            unsafe { pyre_object::w_method_get_self(result) }
        }
    } else if which == 0 {
        result
    } else {
        pyre_object::PY_NULL
    }
}

/// BINARY_SLICE residual (`binary_slice` HLOp → `residual_call_r_r`).
/// Computes `obj[start:stop]` through the shared
/// `runtime_ops::binary_slice_values` (the same code the interpreter's
/// `binary_slice` runs).  A `__getitem__` on a user object may run Python
/// (`MayForce`); on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call
/// returns 0, matching [`bh_load_attr_fn`].
pub extern "C" fn bh_binary_slice_fn(
    obj: PyObjectRef,
    start: PyObjectRef,
    stop: PyObjectRef,
) -> PyObjectRef {
    match pyre_interpreter::runtime_ops::binary_slice_values(obj, start, stop) {
        Ok(result) => result,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj);
            PY_NULL
        }
    }
}

/// STORE_SLICE residual (`store_slice` HLOp → `residual_call_r_v`).
/// Runs `obj[start:stop] = value` through the shared
/// `runtime_ops::store_slice_values` (the same code the interpreter's
/// `store_slice` runs — builds a `slice(start, stop, None)` and dispatches
/// `setitem`).  A user `__setitem__` or slice-bound `__index__` may run
/// Python and force virtualizables (`MayForce`).  Void result, so always
/// returns 0; on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException`, matching
/// `bh_delete_subscr_fn`.
pub extern "C" fn bh_store_slice_fn(
    obj: PyObjectRef,
    start: PyObjectRef,
    stop: PyObjectRef,
    value: PyObjectRef,
) -> i64 {
    if let Err(mut err) = pyre_interpreter::runtime_ops::store_slice_values(obj, start, stop, value)
    {
        let exc_obj = err.to_exc_object();
        publish_residual_call_exception(exc_obj);
    }
    0
}

/// DELETE_SUBSCR residual (`delete_subscr` HLOp → `residual_call_r_v`).
/// Runs `del obj[index]` through the shared `baseobjspace::delitem` (the
/// same code the interpreter's `delete_subscript` runs).  A `__delitem__`
/// on a user object may run Python (`MayForce`).  Void result, so always
/// returns 0; on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException`, matching
/// `bh_store_subscr_fn`.
pub extern "C" fn bh_delete_subscr_fn(obj: PyObjectRef, index: PyObjectRef) -> i64 {
    // Diagnostic (#24 wasm-Linux miscompile): the delete target and index Refs
    // must be non-null at the residual boundary. A null `items` on the wasm
    // resume path would silently mis-route — `is_list(null)` is false without a
    // deref, so the slice branch is skipped and the operand reads guest offset 0.
    // Trap loudly to pin a null `items`/`index` at the residual entry.
    assert!(
        !obj.is_null() && !index.is_null(),
        "bh_delete_subscr_fn: null residual operand (#24 wasm): obj={obj:p} index={index:p}"
    );
    if let Err(mut err) = pyre_interpreter::baseobjspace::delitem(obj, index) {
        let exc_obj = err.to_exc_object();
        publish_residual_call_exception(exc_obj);
    }
    0
}

/// LIST_EXTEND residual (`list_extend` HLOp → `residual_call_r_v`).
/// Runs `list.extend(iterable)` through the shared
/// `opcode_ops::list_extend_value` (the same code the interpreter's
/// `list_extend` runs); `list` is peeked and mutated in place.  A
/// non-iterable operand or a user iterator running Python can raise
/// (`MayForce`).  Void result, so always returns 0; on error the
/// exception is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException`, matching `bh_delete_subscr_fn`.
pub extern "C" fn bh_list_extend_fn(list: PyObjectRef, iterable: PyObjectRef) -> i64 {
    if let Err(mut err) = pyre_interpreter::opcode_ops::list_extend_value(list, iterable) {
        let exc_obj = err.to_exc_object();
        publish_residual_call_exception(exc_obj);
    }
    0
}

/// SET_ADD residual (`set_add` HLOp → `residual_call_r_v`).  Runs
/// `set.add(value)` (or `list.append`) through the shared
/// `opcode_ops::set_add_value`; `set` is peeked and mutated in place.
/// A user `__hash__`/`__eq__` can run Python (`MayForce`).  Void result.
pub extern "C" fn bh_set_add_fn(set: PyObjectRef, value: PyObjectRef) -> i64 {
    if let Err(mut err) = pyre_interpreter::opcode_ops::set_add_value(set, value) {
        publish_residual_call_exception(err.to_exc_object());
    }
    0
}

/// SET_UPDATE residual (`set_update` HLOp → `residual_call_r_v`).  Runs
/// `set.update(iterable)` (or `list.extend`) through the shared
/// `opcode_ops::set_update_value`; `set` is peeked and mutated in place.
/// A user iterator / `__hash__` can run Python (`MayForce`).  Void result.
pub extern "C" fn bh_set_update_fn(set: PyObjectRef, iterable: PyObjectRef) -> i64 {
    if let Err(mut err) = pyre_interpreter::opcode_ops::set_update_value(set, iterable) {
        publish_residual_call_exception(err.to_exc_object());
    }
    0
}

/// DICT_UPDATE residual (`dict_update` HLOp → `residual_call_r_v`).  Runs
/// `PyDict_Update` and the handler's own non-mapping rewrite through the
/// shared `opcode_ops::dict_update_value`; `dict` is peeked and mutated in
/// place.  A `keys()`/`__getitem__`/`__hash__` can run Python
/// (`MayForce`).  Void result.
pub extern "C" fn bh_dict_update_fn(dict: PyObjectRef, source: PyObjectRef) -> i64 {
    if let Err(mut err) = pyre_interpreter::opcode_ops::dict_update_value(dict, source) {
        publish_residual_call_exception(err.to_exc_object());
    }
    0
}

/// MAP_ADD residual (`map_add` HLOp → `residual_call_r_v`).  Runs
/// `dict[key] = value` through the shared `opcode_ops::map_add_value`;
/// `dict` is peeked and mutated in place.  A user key `__hash__`/`__eq__`
/// can run Python (`MayForce`).  Void result.
pub extern "C" fn bh_map_add_fn(dict: PyObjectRef, key: PyObjectRef, value: PyObjectRef) -> i64 {
    if let Err(mut err) = pyre_interpreter::opcode_ops::map_add_value(dict, key, value) {
        publish_residual_call_exception(err.to_exc_object());
    }
    0
}

/// DICT_MERGE residual (`dict_merge` HLOp → `residual_call_r_v`).  Runs
/// `dict.update(source)` with duplicate-key checks through the shared
/// `opcode_ops::dict_merge_value`; `dict` is peeked and mutated in place.
/// `w_callable` is the peeked callable used only for error-message
/// prefixes.  A `keys()`/`__getitem__`/`__hash__` can run Python
/// (`MayForce`).  Void result.
pub extern "C" fn bh_dict_merge_fn(
    dict: PyObjectRef,
    source: PyObjectRef,
    w_callable: PyObjectRef,
) -> i64 {
    if let Err(mut err) = pyre_interpreter::opcode_ops::dict_merge_value(dict, source, w_callable) {
        publish_residual_call_exception(err.to_exc_object());
    }
    0
}

/// Compute the LOOKUP_METHOD `null_or_self` for blackhole LOAD_ATTR resume,
/// given the already-resolved `attr` from [`bh_load_attr_fn`].  Delegates to
/// the shared `compute_load_method_bound`, a pure MRO inspection that never
/// re-invokes a descriptor, so it cannot raise (returns `PY_NULL` when no
/// receiver should be prepended).
pub extern "C" fn bh_load_method_self_fn(
    obj: PyObjectRef,
    attr: PyObjectRef,
    w_code_ptr: PyObjectRef,
    name_idx: i64,
) -> PyObjectRef {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr) as *const pyre_interpreter::CodeObject)
    };
    // Same `co_names`-index codegen invariant as `bh_load_attr_fn`; the
    // release-path `PY_NULL` is also the legitimate "no self prepended"
    // result, so the debug assert is what catches an invalid index.
    let idx = name_idx as usize;
    debug_assert!(
        idx < code.names.len(),
        "bh_load_method_self_fn name_idx {idx} out of range ({} names) — codegen invariant",
        code.names.len()
    );
    if idx >= code.names.len() {
        return pyre_object::PY_NULL;
    }
    let name = code.names[idx].as_ref();
    pyre_interpreter::eval::compute_load_method_bound(obj, attr, name)
}

#[inline]
fn load_special_method_name(method_kind: i64) -> &'static str {
    match method_kind {
        0 => "__enter__",
        1 => "__exit__",
        2 => "__aenter__",
        _ => "__aexit__",
    }
}

/// Resolve the LOAD_SPECIAL method named by `method_kind`.  Mirrors the
/// interpreter's `load_special` through the shared `load_special_resolve`:
/// `_PyObject_LookupSpecial` looks the method up on the type only, bypassing
/// the instance dict and any `__getattribute__` override, so a traced
/// `with`/`async with` calls the same method the interpreter would.  Returns
/// the bound method, so LOAD_SPECIAL's `null_or_self` half is the constant NULL
/// the codewriter pushes without a call.
pub extern "C" fn bh_load_special_fn(obj: PyObjectRef, method_kind: i64) -> PyObjectRef {
    let name = load_special_method_name(method_kind);
    match pyre_interpreter::baseobjspace::load_special_resolve(obj, name) {
        Ok(bound) => bound,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// WITH_EXCEPT_START residual.  The generated JitCode carries the five
/// handler-stack values, and this helper performs the same `__exit__` call as
/// `PyFrame::with_except_start` from the three semantic operands.
pub extern "C" fn bh_with_except_start_fn(
    exit_func: PyObjectRef,
    exit_self: PyObjectRef,
    val: PyObjectRef,
) -> PyObjectRef {
    let result = pyre_interpreter::eval::with_except_start_values(exit_func, exit_self, val);
    if result.is_null() {
        let mut err = pyre_interpreter::call::take_call_error()
            .unwrap_or_else(|| pyre_interpreter::PyError::type_error("__exit__ failed"));
        publish_residual_call_exception(err.to_exc_object());
    }
    result
}

/// `LOAD_NAME` residual for the standalone (blackhole / deopt)
/// per-CodeObject jitcode.  `pyopcode.py LOAD_NAME` — when
/// `getorcreatedebug().w_locals is not get_w_globals_storage()` the lookup
/// tries `finditem_str(w_locals, varname)` first, then falls through
/// to the `LOAD_GLOBAL` globals → builtins chain.  Delegates to the
/// interpreter trait impl (`eval.rs load_name_checked_value`) so the
/// blackhole re-execution and the interpreter share one lookup order.
/// The full-body walker traces the emitted residual call. `w_name` is the
/// interned immortal str constant the flatten driver lowers the
/// `load_name` HLOp's name operand to (`box_str_constant`); `namei`
/// feeds the `pycode._globals_caches[nameindex]` global cache
/// (`celldict.py:292`).  On error it sets `BH_LAST_EXC_VALUE` and
/// returns 0, matching `bh_load_global_fn`'s NameError path.
pub extern "C" fn bh_load_name_fn(
    frame_ptr: *mut PyFrame,
    w_name: PyObjectRef,
    namei: i64,
) -> PyObjectRef {
    use pyre_interpreter::pyopcode::NamespaceOpcodeHandler;
    assert!(
        !frame_ptr.is_null(),
        "bh_load_name_fn requires a non-null PyFrame; every LOAD_NAME emit \
         site must thread portal_frame_reg as the leading ref operand"
    );
    let frame = unsafe { &mut *frame_ptr };
    // `&str` namespace tables cannot hold a lone surrogate, so this is the
    // same miss `load_name_checked_value` reports for an unknown name.
    let Some(name) = (unsafe { pyre_object::unicodeobject::w_str_get_value_opt(w_name) }) else {
        let text = unsafe { pyre_object::unicodeobject::w_str_get_wtf8(w_name) };
        let mut msg = rustpython_wtf8::Wtf8Buf::from("name '");
        msg.push_wtf8(text);
        msg.push_str("' is not defined");
        let mut err = pyre_interpreter::PyError::new(pyre_interpreter::PyErrorKind::NameError, msg);
        err.set_w_name_context(w_name);
        publish_residual_call_exception(err.to_exc_object());
        return PY_NULL;
    };
    match frame.load_name_checked_value(name, namei as usize) {
        Ok(w_value) => w_value,
        Err(mut err) => {
            // Publish into BOTH the blackhole `BH_LAST_EXC_VALUE` and the
            // backend `_store_exception` cells: an unfoldable LOAD_NAME
            // (non-module locals) stays residual in the full-body-walk
            // compiled trace, where the following `GUARD_NO_EXCEPTION` reads
            // the backend cells — writing only `BH_LAST_EXC_VALUE` lets the
            // guard pass on a stale 0 and a raising LOAD_NAME is silently
            // swallowed.
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// `STORE_NAME` residual for the standalone (blackhole / deopt)
/// per-CodeObject jitcode.  `pyopcode.py STORE_NAME` —
/// `setitem_str(getorcreatedebug().w_locals, varname, w_newvalue)`.
/// Delegates to the interpreter trait impl (`eval.rs
/// store_name_value`), which routes non-dict mapping locals through
/// `space.setitem` and module/class namespaces through the dict
/// strategy.  Same blackhole-only execution contract and `w_name` ABI
/// as `bh_load_name_fn`.  `STORE_NAME` carries no nameindex-keyed
/// cache upstream and this residual is handed the wrapped name rather than an
/// index, so it enters through `store_name_value_w`, which names the key from
/// that `w_name` instead of resolving a `co_names_w` slot.
/// Returns 1 on success; on error it sets `BH_LAST_EXC_VALUE` and
/// returns 0, matching `bh_store_subscr_fn`.
pub extern "C" fn bh_store_name_fn(
    frame_ptr: *mut PyFrame,
    w_name: PyObjectRef,
    value: PyObjectRef,
) -> i64 {
    use pyre_interpreter::pyopcode::NamespaceOpcodeHandler;
    assert!(
        !frame_ptr.is_null(),
        "bh_store_name_fn requires a non-null PyFrame; every STORE_NAME emit \
         site must thread portal_frame_reg as the leading ref operand"
    );
    let frame = unsafe { &mut *frame_ptr };
    match unsafe { pyre_interpreter::eval::store_name_value_w(frame, w_name, value) } {
        Ok(()) => 1,
        Err(mut err) => {
            // Publish into both exception cells: STORE_NAME lowers into the
            // full-body-walk compiled trace, whose `GUARD_NO_EXCEPTION` reads
            // the backend `_store_exception` cells; writing only
            // `BH_LAST_EXC_VALUE` would let a raising `__setitem__` on a
            // mapping-locals namespace be silently swallowed.
            publish_residual_call_exception(err.to_exc_object());
            0
        }
    }
}

/// `STORE_GLOBAL` residual for the standalone (blackhole / deopt)
/// per-CodeObject jitcode.  `pyopcode.py STORE_GLOBAL` writes the
/// value directly into `w_globals`, bypassing `w_locals`.  Delegates to
/// the interpreter trait impl (`eval.rs store_global_value`), which
/// routes through `w_dict_setitem_str` on the eagerly-resolved
/// `w_globals` (or the back-mirror dict storage when null).  Same
/// blackhole-only execution contract and `w_name` ABI as
/// `bh_store_name_fn`.  `STORE_GLOBAL` carries no nameindex-keyed cache
/// and this residual is handed the wrapped name rather than an index, so it
/// enters through `store_global_value_w`.  Returns 1 on
/// success; on error it sets `BH_LAST_EXC_VALUE` and returns 0.
pub extern "C" fn bh_store_global_fn(
    frame_ptr: *mut PyFrame,
    w_name: PyObjectRef,
    value: PyObjectRef,
) -> i64 {
    use pyre_interpreter::pyopcode::NamespaceOpcodeHandler;
    assert!(
        !frame_ptr.is_null(),
        "bh_store_global_fn requires a non-null PyFrame; every STORE_GLOBAL emit \
         site must thread portal_frame_reg as the leading ref operand"
    );
    let frame = unsafe { &mut *frame_ptr };
    match unsafe { pyre_interpreter::eval::store_global_value_w(frame, w_name, value) } {
        Ok(()) => 1,
        Err(mut err) => {
            // Publish into both exception cells, matching the STORE_NAME arm.
            // `store_global_value` is infallible today (this arm is dead), but
            // STORE_GLOBAL lowers into the compiled trace with a following
            // `GUARD_NO_EXCEPTION` that reads the backend cells, so should a
            // fallible path appear the raise must not be swallowed.
            publish_residual_call_exception(err.to_exc_object());
            0
        }
    }
}

/// DELETE_NAME residual using the frame receiver and interned-name ABI.
/// pyopcode.py DELETE_NAME deletes from `w_locals` and raises `NameError`
/// when the binding is absent.  The trace carries no `co_names` index, so it
/// enters through `delete_name_w`, which deletes through the key object the
/// trace already holds.
pub extern "C" fn bh_delete_name_fn(frame_ptr: *mut PyFrame, w_name: PyObjectRef) -> i64 {
    assert!(
        !frame_ptr.is_null(),
        "bh_delete_name_fn requires a non-null PyFrame; every DELETE_NAME emit \
         site must thread portal_frame_reg as the leading ref operand"
    );
    let frame = unsafe { &mut *frame_ptr };
    match unsafe { pyre_interpreter::eval::delete_name_w(frame, w_name) } {
        Ok(()) => 1,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            0
        }
    }
}

/// `LOAD_LOCALS` residual using the frame receiver.
/// `pyopcode.py` — `pushvalue(getorcreatedebug().w_locals)`, which
/// must hand back the frame's own mapping so a metaclass `__prepare__` result
/// keeps its type. Infallible, so unlike the name residuals there is no
/// exception-publishing arm.
pub extern "C" fn bh_load_locals_fn(frame_ptr: *mut PyFrame) -> PyObjectRef {
    assert!(
        !frame_ptr.is_null(),
        "bh_load_locals_fn requires a non-null PyFrame; every LOAD_LOCALS emit \
         site must thread portal_frame_reg as its ref operand"
    );
    let frame = unsafe { &mut *frame_ptr };
    frame.get_or_create_w_locals()
}

/// `LOAD_BUILD_CLASS` residual using the frame receiver.
/// `pyopcode.py` — `get_builtin().getdictvalue('__build_class__')`,
/// a NameError when the selected builtin mapping has no entry. Delegates to
/// `eval.rs load_build_class_value` so the interpreter and the residual share
/// one lookup, the same contract `bh_load_name_fn` documents. On error it
/// publishes through both exception cells and returns 0.
pub extern "C" fn bh_load_build_class_fn(frame_ptr: *mut PyFrame) -> PyObjectRef {
    use pyre_interpreter::pyopcode::OpcodeStepExecutor;
    assert!(
        !frame_ptr.is_null(),
        "bh_load_build_class_fn requires a non-null PyFrame; every \
         LOAD_BUILD_CLASS emit site must thread portal_frame_reg as its ref operand"
    );
    let frame = unsafe { &mut *frame_ptr };
    match frame.load_build_class_value() {
        Ok(w_value) => w_value,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// IMPORT_NAME's builtin lookup, split from the subsequent Python call just
/// like PyPy's `pyopcode.py:IMPORT_NAME` (`get_builtin().__import__`, then
/// `space.call_function`).  Keeping this lookup as a small residual lets the
/// ordinary `CallFn` path descend through a gateway `BuiltinCode.func` rather
/// than hiding the whole importer behind one opaque residual.
pub extern "C" fn bh_load_import_fn(frame: *mut PyFrame) -> PyObjectRef {
    debug_assert!(
        !frame.is_null(),
        "bh_load_import_fn requires a non-null PyFrame"
    );
    if frame.is_null() {
        let mut err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::SystemError,
            "IMPORT_NAME received a null frame",
        );
        publish_residual_call_exception(err.to_exc_object());
        return PY_NULL;
    }
    match pyre_interpreter::importing::lookup_dunder_import(unsafe { &*frame }) {
        Ok(w_import) => w_import,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// IMPORT_NAME's locals argument, split from the call for the same reason as
/// [`bh_load_import_fn`].  Infallible — it peeks the debug slot rather than
/// creating one — so like `bh_load_locals_fn` it has no exception-publishing
/// arm.
///
/// The asymmetry with [`bh_load_import_fn`] is deliberate rather than an
/// oversight. That one resolves `__import__` and genuinely raises ImportError
/// when the name is absent, so it carries the publish-and-return-0 machinery.
/// This one reads a slot that is either populated or absent, and answers
/// `None` for absent, so it has no failure to report. A null frame is a wiring
/// bug in an emit site, not a runtime condition, and takes the same assert
/// `bh_load_locals_fn` uses for the same reason -- publishing an exception for
/// it would convert a miswired emit site into a `SystemError` raised at some
/// unrelated Python line.
pub extern "C" fn bh_load_import_locals_fn(frame_ptr: *mut PyFrame) -> PyObjectRef {
    assert!(
        !frame_ptr.is_null(),
        "bh_load_import_locals_fn requires a non-null PyFrame; every IMPORT_NAME \
         emit site must thread portal_frame_reg as its ref operand"
    );
    let frame = unsafe { &*frame_ptr };
    pyre_interpreter::importing::import_locals(frame)
}

/// IMPORT_NAME and MAKE_FUNCTION's globals argument — `pyopcode.py`'s
/// `self.get_w_globals()`.
///
/// `w_globals` stopped being a `PyFrame` field, so the namespace is no longer
/// readable as a virtualizable slot: `get_w_globals` answers from
/// `debugdata.w_globals` when the frame carries a payload and from
/// `promote(pycode).w_globals` when it does not. Asking the live frame is what
/// keeps an inlined callee on its own namespace.
pub extern "C" fn bh_load_import_globals_fn(frame_ptr: *mut PyFrame) -> PyObjectRef {
    assert!(
        !frame_ptr.is_null(),
        "bh_load_import_globals_fn requires a non-null PyFrame; every IMPORT_NAME \
         and MAKE_FUNCTION emit site must thread portal_frame_reg as its ref operand"
    );
    let frame = unsafe { &*frame_ptr };
    frame.get_w_globals()
}

/// DELETE_GLOBAL residual using the frame receiver and interned-name ABI.
/// pyopcode.py DELETE_GLOBAL deletes directly from `w_globals`.
pub extern "C" fn bh_delete_global_fn(frame_ptr: *mut PyFrame, w_name: PyObjectRef) -> i64 {
    use pyre_interpreter::pyopcode::OpcodeStepExecutor;
    assert!(
        !frame_ptr.is_null(),
        "bh_delete_global_fn requires a non-null PyFrame; every DELETE_GLOBAL emit \
         site must thread portal_frame_reg as the leading ref operand"
    );
    let frame = unsafe { &mut *frame_ptr };
    // Same miss as `delete_global` for a name the `&str` dict cannot hold.
    let Some(name) = (unsafe { pyre_object::unicodeobject::w_str_get_value_opt(w_name) }) else {
        let text = unsafe { pyre_object::unicodeobject::w_str_get_wtf8(w_name) };
        let mut msg = rustpython_wtf8::Wtf8Buf::from("'");
        msg.push_wtf8(text);
        msg.push_str("'");
        let mut err = pyre_interpreter::PyError::key_error(msg);
        publish_residual_call_exception(err.to_exc_object());
        return 0;
    };
    match frame.delete_global(name) {
        Ok(()) => 1,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            0
        }
    }
}

/// Load a constant from the code object.
/// jtransform.py parity: code comes from getfield_vable_r(frame, pycode).
pub extern "C" fn bh_load_const_fn(w_code_ptr: PyObjectRef, consti: i64) -> PyObjectRef {
    // `getconstant_w(index) -> co_consts_w[index]`: read the one shared object
    // off the virtualizable `pycode`, exactly as the interpreter does.
    let w_const = unsafe { pyre_interpreter::pycode::w_code_const(w_code_ptr, consti as usize) };
    w_const
}

/// Box a raw integer into a PyObject (w_int_new wrapper).
pub extern "C" fn bh_box_int_fn(value: i64) -> PyObjectRef {
    w_int_new(value)
}

/// `eval.rs`'s `raise_varargs` (RAISE_VARARGS) normalization for
/// blackhole/JitCode.
///
/// JitCode's `raise/r` bytecode carries only the final exception object, so
/// callers normalize `raise Type` and `raise X from Y` through this helper
/// before emitting `raise/r`.
///
/// `(frame: Ref, exc: Ref, cause: Ref) → Ref`.  The frame pointer is
/// emitted explicitly by `Instruction::RaiseVarargs` (codewriter.rs)
/// via `portal_frame_reg`, mirroring `bh_load_global_fn`'s frame-as-arg
/// ABI. `pyopcode.py RAISE_VARARGS` runs inside an opcode
/// dispatch where `frame` and the current execution context are always
/// valid, so `frame_ptr == 0` here signals a wiring bug — fail fast
/// at entry rather than degrade silently to a `RuntimeError`.
pub extern "C" fn bh_normalize_raise_varargs_with_frame(
    parent_frame_ptr: *const PyFrame,
    exc: PyObjectRef,
    raw_cause: PyObjectRef,
) -> PyObjectRef {
    assert!(
        !parent_frame_ptr.is_null(),
        "bh_normalize_raise_varargs_with_frame requires a non-null parent \
         PyFrame; every RAISE_VARARGS emit site must thread portal_frame_reg \
         as the leading ref operand"
    );
    let mut exc = exc;

    // pyopcode.py:704-722 — cause and exc normalization share
    // `self.space.getexecutioncontext()`. Pin the current activation for the
    // whole body so the cause-class-call and exc-class-call observe the same
    // thread state.
    let frame_ctx = pyre_interpreter::call::getexecutioncontext();
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    if !frame_ctx.is_null() {
        pyre_interpreter::call::set_last_exec_ctx(frame_ctx);
    }

    let mut cause = if raw_cause.is_null() {
        None
    } else {
        // pyopcode.py:706-707 — cause class-call must mirror the exc
        // class-call (pyopcode.py:711-713) on blackhole re-entry.
        // Force both onto the plain interpreter path so the constructor
        // cannot re-enter the tracer.
        //
        // That class call runs Python.  `exc` arrives as a bare operand out of
        // compiled code and holds no root of its own, so bracket the call the
        // way `framework.py get_livevars_for_roots` does — around the
        // operation, not the function — and read the forwarded value back.
        let roots = pyre_object::gc_roots::push_roots();
        let exc_slot = roots.publish(&[exc]);
        roots.normalize(exc_slot, 1);
        let result = {
            let _plain_guard = pyre_interpreter::call::force_plain_eval();
            pyre_interpreter::eval::normalize_raise_cause(raw_cause)
        };
        exc = roots.get(exc_slot);
        match result {
            Ok(cause) => Some(cause),
            Err(mut err) => {
                pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
                return err.to_exc_object();
            }
        }
    };

    let mut final_exc: PyObjectRef = unsafe {
        if pyre_object::is_exception(exc) {
            exc
        } else if pyre_interpreter::baseobjspace::exception_is_valid_obj_as_class_w(exc) {
            // pyopcode.py RAISE_VARARGS — `space.call_function(w_type)` does
            // not depend on frame-owned context state.
            //
            // It runs the exception's `__init__`.  `exc` is still unrooted and
            // the cause normalized above is a fresh nursery object, so both
            // ride the bracket across the call.
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.publish(&[
                exc,
                cause.as_ref().map_or(std::ptr::null_mut(), |c| c.w_cause),
            ]);
            roots.normalize(base, 2);
            let result = {
                let _plain_guard = pyre_interpreter::call::force_plain_eval();
                pyre_interpreter::call::call_function_impl_result(exc, &[])
            };
            exc = roots.get(base);
            if let Some(c) = cause.as_mut() {
                c.w_cause = roots.get(base + 1);
            }
            match result {
                Ok(obj) if pyre_object::is_exception(obj) => obj,
                Ok(obj) => pyre_interpreter::error::exception_from_call_type_error(exc, obj)
                    .to_exc_object(),
                Err(mut err) => err.to_exc_object(),
            }
        } else {
            pyre_interpreter::PyError::type_error("exceptions must derive from BaseException")
                .to_exc_object()
        }
    };

    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);

    if let Err(mut err) = pyre_interpreter::eval::attach_raise_cause(final_exc, cause) {
        final_exc = err.to_exc_object();
    }
    final_exc
}

/// Truthiness check: PyObjectRef → raw 0 or 1.
pub extern "C" fn bh_truth_fn(obj: PyObjectRef) -> i64 {
    if obj.is_null() {
        return 0;
    }
    match pyre_interpreter::opcode_ops::truth_value(obj) {
        Ok(truth) => truth as i64,
        Err(mut err) => {
            // A raising `__bool__` / `__len__` publishes for the trailing
            // GuardNoException, then returns 0.
            publish_residual_call_exception(err.to_exc_object());
            0
        }
    }
}

/// BUILD_TUPLE: `space.newtuple(list_w)` (`objspace.py:332` →
/// `tupleobject.py` wraptuple) consuming a length-prefixed
/// `GcTypedArray` of refs — the forced `popvalues` list
/// (`pyframe.py`).  Length travels inside the array (offset-0
/// prefix), so there is no arity cap.  Allocation-only; the items are
/// pre-existing heap refs, no user code runs.
pub extern "C" fn bh_newtuple_from_array(
    arr: *const pyre_object::object_array::GcTypedArray,
) -> PyObjectRef {
    let len = pyre_object::object_array::gcarray_len(arr);
    let mut items: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for i in 0..len {
        items.push(pyre_object::object_array::getarrayitem_ref(arr, i));
    }
    pyre_interpreter::runtime_ops::build_tuple_from_refs(&items)
}

/// BUILD_LIST arbitrary-arity residual: `space.newlist(items_w)`.  The
/// arity-`> 3` form of BUILD_LIST that the fixed `build_list_fn` (three
/// item slots) cannot cover; the items travel inside a length-prefixed
/// forced array exactly like [`bh_newtuple_from_array`].  Allocation-only
/// (`build_list_from_refs` = `w_list_new`), so `CallFlavor::Plain` with
/// no trailing `GuardNoException`.
pub extern "C" fn bh_newlist_from_array(
    arr: *const pyre_object::object_array::GcTypedArray,
) -> PyObjectRef {
    let len = pyre_object::object_array::gcarray_len(arr);
    let mut items: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for i in 0..len {
        items.push(pyre_object::object_array::getarrayitem_ref(arr, i));
    }
    pyre_interpreter::runtime_ops::build_list_from_refs(&items)
}

/// BUILD_MAP residual — the dict counterpart of [`bh_newtuple_from_array`].
/// The length-prefixed array holds the interleaved `[k0, v0, k1, v1, ...]`
/// pairs the codewriter unrolled via `setarrayitem_gc_r`;
/// `build_map_from_refs` consumes them in `chunks_exact(2)`.  Keys are
/// hashed (may run user `__hash__` / `__eq__`) and an unhashable key raises
/// (`MayForce`, fallible); on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call
/// returns 0.
pub extern "C" fn bh_build_map_from_array(
    arr: *const pyre_object::object_array::GcTypedArray,
) -> PyObjectRef {
    let len = pyre_object::object_array::gcarray_len(arr);
    let mut items: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for i in 0..len {
        items.push(pyre_object::object_array::getarrayitem_ref(arr, i));
    }
    match pyre_interpreter::runtime_ops::build_map_from_refs(&items) {
        Ok(dict) => dict,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj);
            PY_NULL
        }
    }
}

/// BUILD_SET residual — the set counterpart of [`bh_build_map_from_array`].
/// The length-prefixed array holds the `count` set elements the codewriter
/// unrolled via `setarrayitem_gc_r`.  Element hashing may run user `__hash__`
/// and a non-hashable element raises (`MayForce`, fallible); on error the
/// exception is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_build_set_from_array(
    arr: *const pyre_object::object_array::GcTypedArray,
) -> PyObjectRef {
    let len = pyre_object::object_array::gcarray_len(arr);
    let mut items: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for i in 0..len {
        items.push(pyre_object::object_array::getarrayitem_ref(arr, i));
    }
    match pyre_interpreter::runtime_ops::build_set_from_refs(&items) {
        Ok(set) => set,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj);
            PY_NULL
        }
    }
}

/// BUILD_STRING residual (`build_string_from_array` HLOp →
/// `residual_call_r_r`).  Concatenates the forced fragment array into a
/// single `str` through the shared `runtime_ops::build_string_from_refs`.
/// A non-`str` fragment raises `TypeError` (`CanRaise`, no user code); on
/// error the exception is published through `BH_LAST_EXC_VALUE` for the
/// trailing `GuardNoException` and the call returns 0.
pub extern "C" fn bh_build_string_from_array(
    arr: *const pyre_object::object_array::GcTypedArray,
) -> PyObjectRef {
    let len = pyre_object::object_array::gcarray_len(arr);
    let mut items: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for i in 0..len {
        items.push(pyre_object::object_array::getarrayitem_ref(arr, i));
    }
    match pyre_interpreter::runtime_ops::build_string_from_refs(&items) {
        Ok(s) => s,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// FORMAT_SIMPLE residual (`format_simple` HLOp → `residual_call_r_r`).
/// Formats `value` with the empty spec (`f"{x}"` → `str(value)`) through the
/// shared `runtime_ops::format_value`.  A user `__format__` may run Python
/// (`MayForce`); on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call
/// returns 0.
pub extern "C" fn bh_format_simple_fn(value: PyObjectRef) -> PyObjectRef {
    match pyre_interpreter::runtime_ops::format_value(value, pyre_object::PY_NULL) {
        Ok(s) => s,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj);
            PY_NULL
        }
    }
}

/// CONVERT_VALUE residual (`convert_value` HLOp → `residual_call_ir_r`).
/// Converts `value` per `conv` (`0=Str/3=None → str`, `1=Repr → repr`,
/// `2=Ascii → ascii`) through the shared `runtime_ops::convert_value`.  A
/// user `__str__` / `__repr__` may run Python (`MayForce`); on error the
/// exception is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_convert_value_fn(value: PyObjectRef, conv: i64) -> PyObjectRef {
    match pyre_interpreter::runtime_ops::convert_value(value, conv) {
        Ok(s) => s,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj);
            PY_NULL
        }
    }
}

/// FORMAT_WITH_SPEC residual (`format_with_spec` HLOp → `residual_call_r_r`).
/// Formats `value` with `spec` (`f"{x:.2f}"`) through the shared
/// `runtime_ops::format_value`.  A user `__format__` may run Python
/// (`MayForce`); on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call
/// returns 0.
pub extern "C" fn bh_format_with_spec_fn(value: PyObjectRef, spec: PyObjectRef) -> PyObjectRef {
    match pyre_interpreter::runtime_ops::format_value(value, spec) {
        Ok(s) => s,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj);
            PY_NULL
        }
    }
}

/// LOAD_DEREF residual (`load_deref_value` HLOp → `residual_call_ir_r`).
/// `cell` is the slot read from `locals_cells_stack_w`: a cell object whose
/// contents are the free/cell variable's value, or the raw value for a slot
/// that is not a cell.  Mirrors `load_deref` — dereference the cell and
/// raise if the result is empty.  `code` + `deref_idx` resolve the variable
/// name for the unbound-variable `NameError` (`deref_unbound_error`).  Runs no
/// user code but reads mutable heap (`CallFlavor::Plain`); on the
/// unbound-variable error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call
/// returns 0.
pub extern "C" fn bh_load_deref_value_fn(
    slot: PyObjectRef,
    w_code_ptr: PyObjectRef,
    deref_idx: i64,
) -> PyObjectRef {
    let value = if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
        unsafe { pyre_object::w_cell_get(slot) }
    } else {
        slot
    };
    if value == pyre_object::PY_NULL {
        let code = unsafe {
            &*(pyre_interpreter::w_code_get_ptr(w_code_ptr) as *const pyre_interpreter::CodeObject)
        };
        let exc_obj = pyre_interpreter::pyframe::deref_unbound_error(code, deref_idx as usize)
            .to_exc_object();
        publish_residual_call_exception(exc_obj);
        return PY_NULL;
    }
    value
}

/// STORE_DEREF residual (`store_deref_value` HLOp → `residual_call_r_r`).
/// `cell` is the slot read from `locals_cells_stack_w`; `value` is the
/// popped stack operand.  Mirrors `store_deref`: when the slot holds a
/// cell, mutate the cell's contents in place (`w_cell_set`, incminimark
/// write barrier inside) and return the unchanged cell so the caller
/// re-stores the same pointer into the slot; otherwise return the raw
/// `value` so the caller writes it into the slot directly.  Runs no user
/// code and never raises (`CallFlavor::PlainCannotRaise`).
pub extern "C" fn bh_store_deref_value_fn(cell: PyObjectRef, value: PyObjectRef) -> PyObjectRef {
    if !cell.is_null() && unsafe { pyre_object::is_cell(cell) } {
        unsafe { pyre_object::w_cell_set(cell, value) };
        cell
    } else {
        value
    }
}

/// MAKE_CELL residual (`make_cell_value` HLOp → `residual_call_ir_r`).
/// `current` is the slot read from `locals_cells_stack_w`.  Mirrors
/// `make_cell`: wrap the value in a fresh cell when the slot does not
/// already hold one (`initialize_frame_scopes` installs cells for pure
/// cellvars, so only an argument slot promoted to a cellvar still holds a
/// raw value here), and return the cell the caller stores back into the
/// slot.  A slot already holding a cell is returned unchanged so a
/// never-reassigned cellvar does not become a cell wrapping a cell.
/// `code` + `slot` name the cell family (`pycode.py` `PyCode._initialize`)
/// the new cell joins
/// (`pyframe.py` `PyFrame.initialize_frame_scopes`).
/// Allocates (may trigger a minor GC) but runs no user code and never
/// raises (`CallFlavor::Plain`).
pub extern "C" fn bh_make_cell_fn(
    current: PyObjectRef,
    code: PyObjectRef,
    slot: i64,
) -> PyObjectRef {
    if current.is_null() || !unsafe { pyre_object::is_cell(current) } {
        let family = unsafe { pyre_interpreter::pycode::w_code_cell_family(code, slot as usize) };
        pyre_object::w_cell_new(current, family)
    } else {
        current
    }
}

/// UNARY_NEGATIVE residual (`unary_negative` HLOp → `residual_call_r_r`).
/// Computes `-value` through `opcode_ops::unary_negative_value` (`neg`); a
/// user `__neg__` may run Python (`MayForce`).  On error the exception
/// is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_unary_negative_fn(value: PyObjectRef) -> PyObjectRef {
    match pyre_interpreter::opcode_ops::unary_negative_value(value) {
        Ok(result) => result,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj);
            PY_NULL
        }
    }
}

/// GET_ITER residual (`residual_call_r_r`).  Computes `iter(obj)` through
/// `baseobjspace::iter`; a user `__iter__` may run Python (`MayForce`).  On
/// error the exception is published through `BH_LAST_EXC_VALUE` for the
/// trailing `GuardNoException` and the call returns 0.
pub extern "C" fn bh_get_iter_fn(obj: PyObjectRef) -> PyObjectRef {
    match pyre_interpreter::baseobjspace::iter(obj) {
        Ok(result) => result,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// Residual callee for the jitted `jump_absolute` armed path:
/// `ec.bytecode_trace(frame, decr_by)`.
///
/// Void result (`residual_call_r_v`), matching `bh_delete_subscr_fn`:
/// always returns 0; an exception is published through `BH_LAST_EXC_VALUE`
/// for the trailing `GuardNoException`.
///
/// pyre-adaptation: the walked guard fails on `word >= JIT_BREAKER_FLOOR`,
/// so it also fires for the pyre-only breaker bits (`EB_STW`,
/// `EB_FINALIZING`, `EB_GC`, `EB_MEMORY_ERROR`).  This helper consumes only
/// what `bytecode_trace` consumes and leaves those bits armed for the
/// bytecode-boundary service in `eval_loop_jit`: that service collects
/// through `gc_interp::safepoint`, which is safe only at the interpreter's
/// dispatch safepoint, not from a residual call inside machine code.
pub extern "C" fn bh_bytecode_trace_jitted_slow(
    ec: *mut pyre_interpreter::PyExecutionContext,
    frame_ptr: *mut PyFrame,
) -> i64 {
    if ec.is_null() {
        return 0;
    }
    let decr_by = if pyre_interpreter::module::thread::gil::threads_initialized() {
        crate::eval::_get_adapted_tick_counter()
    } else {
        0
    };
    match unsafe { (*ec).bytecode_trace(frame_ptr, decr_by) } {
        Ok(()) => 0,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            0
        }
    }
}

/// UNARY_INVERT residual (`unary_invert` HLOp → `residual_call_r_r`).
/// Computes `~value` through `opcode_ops::unary_invert_value` (`invert`); a
/// user `__invert__` may run Python (`MayForce`).  On error the exception
/// is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_unary_invert_fn(value: PyObjectRef) -> PyObjectRef {
    match pyre_interpreter::opcode_ops::unary_invert_value(value) {
        Ok(result) => result,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj);
            PY_NULL
        }
    }
}

/// UNARY_POSITIVE residual (`pos` HLOp → `residual_call_r_r`).  Computes
/// `+value` through `opcode_ops::unary_positive_value` (`pos`); a user
/// `__pos__` may run Python (`MayForce`).  On error the exception is
/// published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_unary_positive_fn(value: PyObjectRef) -> PyObjectRef {
    match pyre_interpreter::opcode_ops::unary_positive_value(value) {
        Ok(result) => result,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj);
            PY_NULL
        }
    }
}

/// CALL_INTRINSIC_1 ListToTuple residual (`list_to_tuple` HLOp →
/// `residual_call_r_r`).  Converts a list to a tuple through the shared
/// `opcode_ops::list_to_tuple_value`; allocates a fresh tuple, and a
/// non-list operand raises TypeError (`MayForce`).  On error the
/// exception is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_list_to_tuple_fn(value: PyObjectRef) -> PyObjectRef {
    match pyre_interpreter::opcode_ops::list_to_tuple_value(value) {
        Ok(result) => result,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// GET_LEN residual (`get_len` HLOp → `residual_call_r_r`).  Pushes
/// `len(subject)` without consuming the subject.  Runs the type's
/// `__len__` (`MayForce`).
pub extern "C" fn bh_get_len_fn(subject: PyObjectRef) -> PyObjectRef {
    match pyre_interpreter::baseobjspace::len(subject) {
        Ok(result) => result,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// MATCH_SEQUENCE residual (`match_sequence` HLOp → `residual_call_r_r`).
/// Reads the subject type's PATMA marker; runs no user code and never
/// raises (`Plain`).
pub extern "C" fn bh_match_sequence_fn(subject: PyObjectRef) -> PyObjectRef {
    pyre_interpreter::opcode_ops::match_sequence_value(subject)
}

/// MATCH_MAPPING residual (`match_mapping` HLOp → `residual_call_r_r`).
/// Mirrors [`bh_match_sequence_fn`].
pub extern "C" fn bh_match_mapping_fn(subject: PyObjectRef) -> PyObjectRef {
    pyre_interpreter::opcode_ops::match_mapping_value(subject)
}

/// MATCH_KEYS residual (`match_keys` HLOp → `residual_call_r_r`).  Looks
/// each pattern key up in the subject via `get`, so it runs user code
/// (`MayForce`) and can raise on a duplicate key.
pub extern "C" fn bh_match_keys_fn(subject: PyObjectRef, keys: PyObjectRef) -> PyObjectRef {
    match pyre_interpreter::opcode_ops::match_keys_value(subject, keys) {
        Ok(result) => result,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// MATCH_CLASS residual (`match_class` HLOp → `residual_call_ir_r`).
/// `count` is the number of positional sub-patterns.  Runs `isinstance`
/// and attribute lookups, so user code (`MayForce`).
pub extern "C" fn bh_match_class_fn(
    subject: PyObjectRef,
    cls: PyObjectRef,
    kwd_attrs: PyObjectRef,
    count: i64,
) -> PyObjectRef {
    match pyre_interpreter::opcode_ops::match_class_value(subject, cls, kwd_attrs, count as usize) {
        Ok(result) => result,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// LOAD_COMMON_CONSTANT residual (`load_common_constant` HLOp →
/// `residual_call_ir_r`).  `disc` is the `CommonConstant` discriminant
/// (0-6).  Resolves the pushed object through the shared
/// `opcode_ops::load_common_constant_value`, matching the interpreter:
/// immortal type/exception classes for the class variants, a freshly
/// built builtin function for `all`/`any` (hence `MayForce` — it
/// allocates).  Runs no user code and never raises; an out-of-range
/// discriminant (corrupt bytecode) returns PY_NULL.
pub extern "C" fn bh_load_common_constant_fn(disc: i64) -> PyObjectRef {
    match pyre_interpreter::bytecode::CommonConstant::try_from(disc as u32) {
        Ok(cc) => pyre_interpreter::opcode_ops::load_common_constant_value(cc),
        Err(_) => pyre_object::PY_NULL,
    }
}
/// UNARY_NOT residual (`unary_not` HLOp → `residual_call_r_r`).  Returns
/// `not value` as a bool object via `opcode_ops::truth_value`.  A user
/// `__bool__` / `__len__` may run Python (`MayForce`), matching the
/// interpreter's UNARY_NOT truth path; a raising `__bool__` publishes
/// through `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the
/// call returns 0.
pub extern "C" fn bh_unary_not_fn(value: PyObjectRef) -> PyObjectRef {
    match pyre_interpreter::opcode_ops::truth_value(value) {
        Ok(truth) => pyre_object::w_bool_from(!truth),
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// LOAD_FAST_CHECK residual (`load_fast_check` HLOp → `residual_call_ir_r`).
/// The local slot is read from the vable exactly like LOAD_FAST and handed in
/// as `value` (possibly `PY_NULL` for an unbound local).  Returns `value`
/// unchanged when bound; on an unbound local raises `UnboundLocalError`,
/// resolving the variable name from the resume frame's code object via the
/// `co_varnames` index baked in by the codewriter.  Reads only that immutable
/// `co_varnames`, writes nothing and runs no user code
/// (`CallFlavor::CanRaise`); the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call returns
/// 0.
pub extern "C" fn bh_load_fast_check_fn(
    value: PyObjectRef,
    w_code_ptr: PyObjectRef,
    name_idx: i64,
) -> PyObjectRef {
    if !value.is_null() {
        return value;
    }
    let exc_obj = bh_unbound_local_error_fn(w_code_ptr, name_idx);
    publish_residual_call_exception(exc_obj);
    PY_NULL
}

/// Construct the value raised by DELETE_FAST when its local is unbound.
/// Unlike `bh_load_fast_check_fn`, this returns the exception object without
/// publishing it through the residual-exception channel. It allocates but
/// runs no user code and never raises (`CallFlavor::PlainCannotRaise`).
pub extern "C" fn bh_unbound_local_error_fn(w_code_ptr: PyObjectRef, name_idx: i64) -> PyObjectRef {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr) as *const pyre_interpreter::CodeObject)
    };
    // `name_idx` is a `co_varnames` index baked into the residual call by the
    // codewriter from the originating LOAD_FAST_CHECK oparg.  An out-of-range
    // index is a codegen invariant rather than a runtime-reachable error;
    // mirror `execute_load_fast_check`'s `idx < code_varnames_len(code)` guard,
    // falling back to the "<cell>" label rather than panicking in release.
    let idx = name_idx as usize;
    let name = if idx < code.varnames.len() {
        code.varnames[idx].as_ref()
    } else {
        "<cell>"
    };
    pyre_interpreter::PyError::unbound_local_error(format!(
        "cannot access local variable '{name}' where it is not associated with a value"
    ))
    .to_exc_object()
}

#[cfg(test)]
mod tests_bh_newtuple_from_array {
    use super::bh_newtuple_from_array;
    use pyre_object::object_array::{ArrayKind, allocate_array, setarrayitem_ref};

    #[test]
    fn builds_tuple_of_any_arity_from_ref_array() {
        // The length travels inside the length-prefixed array, so any
        // arity fits without fixed argument slots.
        let arr = allocate_array(5, ArrayKind::Ref, true);
        let items: Vec<pyre_object::PyObjectRef> = (0..5)
            .map(|i| pyre_object::w_int_new(i as i64 * 7))
            .collect();
        for (i, &w) in items.iter().enumerate() {
            setarrayitem_ref(arr, i, w);
        }
        let tup = bh_newtuple_from_array(arr as *const pyre_object::object_array::GcTypedArray);
        unsafe {
            assert_eq!(pyre_object::w_tuple_len(tup), 5);
            for (i, &w) in items.iter().enumerate() {
                let got =
                    pyre_object::w_tuple_getitem(tup, i as i64).expect("tuple item within range");
                assert_eq!(got, w);
            }
        }
    }
}

/// BUILD_SLICE: `space.newslice(w_start, w_end, w_step)`.
/// `argc` is 2 or 3; for argc=2 the CPython/PyPy opcode semantics use None
/// for `w_step` (`pypy/interpreter/pyopcode.py`).
pub extern "C" fn bh_build_slice_fn(
    argc: i64,
    start: PyObjectRef,
    stop: PyObjectRef,
    step: PyObjectRef,
) -> PyObjectRef {
    let step = if argc == 2 {
        pyre_object::w_none()
    } else {
        step
    };
    pyre_object::w_slice_new(start, stop, step)
}

/// UNPACK_SEQUENCE: validate that `seq` has exactly `count` elements and
/// return a tuple of those items, raising ValueError/TypeError on a length
/// mismatch or non-sequence the same way the interpreter does. The portal
/// reads the items back out with `bh_unpack_item_fn`; producing the
/// validated tuple once keeps the iteration-protocol fallback single-pass.
pub extern "C" fn bh_unpack_sequence_fn(count: i64, seq: PyObjectRef) -> PyObjectRef {
    match pyre_interpreter::runtime_ops::unpack_sequence_exact(seq, count as usize) {
        Ok(items) => pyre_interpreter::runtime_ops::build_tuple_from_refs(&items),
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// Read item `index` out of the validated tuple produced by
/// `bh_unpack_sequence_fn`.
pub extern "C" fn bh_unpack_item_fn(index: i64, seq: PyObjectRef) -> PyObjectRef {
    match pyre_interpreter::runtime_ops::sequence_getitem(seq, index as usize) {
        Ok(item) => item,
        Err(mut err) => {
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE
                .with(|c| c.set(err.to_exc_object() as i64));
            PY_NULL
        }
    }
}

/// UNPACK_EX: split `seq` for `a, *b, c = seq` into `before` head items, a
/// starred middle list, and `after` tail items, returning the
/// `before + 1 + after` slots in TOS order as a tuple (raising ValueError on
/// too few values, or any iteration error from a non-sequence source). The
/// portal reads each slot back out with `bh_unpack_item_fn`, mirroring
/// `bh_unpack_sequence_fn`.
pub extern "C" fn bh_unpack_ex_fn(before: i64, after: i64, seq: PyObjectRef) -> PyObjectRef {
    match pyre_interpreter::runtime_ops::unpack_ex_slots(before as usize, after as usize, seq) {
        Ok(slots) => pyre_interpreter::runtime_ops::build_tuple_from_refs(&slots),
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object());
            PY_NULL
        }
    }
}

/// Read the current (per-thread) exception saved in
/// `pyre_interpreter::eval::CURRENT_EXCEPTION`: the value a catch-covered
/// bare `raise` re-raises.
pub extern "C" fn bh_get_current_exception() -> PyObjectRef {
    pyre_interpreter::eval::get_current_exception()
}

/// The value `pyopcode.py PUSH_EXC_INFO` saves below the caught exception:
/// the exception being handled, or `None` when there is none.
pub extern "C" fn bh_current_exception_or_none() -> PyObjectRef {
    pyre_interpreter::eval::current_exception_or_none()
}

/// `eval.rs`'s `raise_varargs(0)` — the value a bare `raise` re-raises.
///
/// Returns the active exception when `sys_exc_value` holds a live
/// `BaseException`; otherwise (null / `None` / non-exception) returns a fresh
/// `RuntimeError("No active exception to reraise")` instance.  The codewriter
/// emits this for a bare `RAISE_VARARGS(0)` whose FrameState carries no
/// `last_exception` pair — where the runtime current-exception may be absent —
/// so the following `raise/r` always receives a non-null value
/// (`blackhole.py:1002` asserts non-null).  Unlike raw `get_current_exception`,
/// this can allocate (the `RuntimeError`), so it is registered `Plain`, not
/// `PlainCannotRaiseNoHeap`.
pub extern "C" fn bh_reraise_varargs_zero() -> PyObjectRef {
    let exc = pyre_interpreter::eval::get_current_exception();
    unsafe {
        if !exc.is_null() && pyre_object::is_exception(exc) {
            exc
        } else {
            pyre_interpreter::PyError::runtime_error("No active exception to reraise")
                .to_exc_object()
        }
    }
}

/// Store `exc` into the per-thread `CURRENT_EXCEPTION` slot. Matches
/// the write at `pyopcode.py POP_EXCEPT` (restore of saved
/// sys_exc_info, where `None` clears it) and at `pyopcode.py
/// PUSH_EXC_INFO` (new raised exception becomes current).
pub extern "C" fn bh_set_current_exception(exc: PyObjectRef) {
    pyre_interpreter::eval::restore_exc_info(exc);
}

/// Complete the `PUSH_EXC_INFO` ownership transfer after the caught exception
/// has become the execution context's current exception.  The interpreter's
/// `push_exc_info` clears this propagation carrier at the same point; leaving
/// it populated roots the handled exception (and its traceback/frame) after
/// the handler returns.
pub extern "C" fn bh_clear_in_flight_exception() {
    pyre_interpreter::eval::set_in_flight_exception(pyre_object::PY_NULL);
}

/// On-demand resume callback (pyre-jit side).  Registered into cranelift
/// via `register_resumedata_deopt` (eval.rs:init_callbacks) and called
/// from the `recovery_layout_ref()` consumer sites that have migrated off
/// the pre-baked `ExitRecoveryLayout` cache.
///
/// PyPy parity target: `pyjitpl.py rebuild_state_after_failure
/// MetaInterp.rebuild_state_after_failure(resumedescr, deadframe)`
/// drives `resume.rebuild_from_resumedata` to materialise virtuals +
/// replay pending fields from `rd_numb` / `rd_consts` / `rd_virtuals` /
/// `rd_pendingfields` carried on the `ResumeGuardDescr`.
///
/// `outputs` enters carrying the failing-guard fail_args (the JITed
/// exit code stored them).  We treat that as the `deadframe` input to
/// the decoder and replace `*outputs` with the per-section
/// concatenation that the recovery_layout walker produced before
/// (innermost-first per `compiler.rs`'s `recovery.frames` reverse walk).
///
/// Implementation lives in `call_jit.rs` rather than `eval.rs` because
/// `pyre-jit-trace`'s build.rs reads `eval.rs` through the JIT
/// translator's RPython subset, which rejects the trait-object
/// downcast pattern used here.
#[cfg(feature = "cranelift")]
pub fn cranelift_resumedata_deopt(
    descr_addr: usize,
    outputs: &mut Vec<i64>,
    types: &[majit_ir::Type],
    _bridge_num_inputs: usize,
) -> bool {
    use majit_backend::Backend;
    use majit_ir::FailDescr;
    use majit_metainterp::resume;

    if std::env::var_os("PYRE_DEOPT_PROBE").is_some() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static ENTRIES: AtomicU64 = AtomicU64::new(0);
        let n = ENTRIES.fetch_add(1, Ordering::Relaxed);
        eprintln!("[deopt-probe] entry #{n} descr_addr={descr_addr:#x}");
    }

    // 1. Recover descr Arc.
    let (driver, driver_vinfo) = crate::eval::driver_pair();
    let backend = driver.meta_interp().backend();
    let descr = backend.fail_descr_arc_from_addr(descr_addr);

    // 2. Unwrap to ResumeGuardDescr. Forced/exc wrappers return themselves
    //    from `as_any` so subtype downcasts work; the resume payload lives
    //    on the inner base (`resume_guard_descr`). Synthetic FINISH /
    //    ExitFrameWithException / external-JUMP descrs have no `rd_*`
    //    payload upstream (compile.py:624-662) — they short-circuit
    //    here.  Callers fall back to the recovery_layout walker for
    //    these until the synthetic construction path is restructured.
    let Some(rgd) = majit_metainterp::resume_guard_descr(&descr) else {
        return false;
    };

    // A novable driver's resume data has no virtualizable section. Derive the
    // owning driver from the guard descr, matching the blackhole path's
    // caller-supplied distinction, rather than guessing from this callback's
    // backend registration.
    let novable = match rgd
        .rd_loop_token_clt()
        .and_then(|clt_any| {
            clt_any.downcast_ref::<std::sync::Arc<majit_backend::CompiledLoopToken>>()
        })
        .and_then(|clt| clt.upgrade_loop_token())
        .and_then(|token| token.outermost_jitdriver_index)
        .and_then(|idx| driver.meta_interp().jitdriver_has_vinfo(idx))
    {
        Some(has_vinfo) => !has_vinfo,
        // Synthetic descr, evicted token, or stale driver slot: decline to the
        // existing recovery-layout fallback instead of choosing the wrong
        // resume-data shape for one of the two driver kinds.
        None => return false,
    };

    // 3. Extract resume payload.  Empty rd_numb → nothing to decode.
    let Some(rd_numb) = rgd.payload.rd_numb() else {
        return false;
    };
    if rd_numb.is_empty() {
        return false;
    }
    let rd_consts = rgd.payload.rd_consts().unwrap_or(&[]);
    let rd_virtuals_rcs = rgd.payload.rd_virtuals();
    let rd_pendingfields = rgd.payload.rd_pendingfields();

    // 4. resume.py _prepare_virtuals — convert RdVirtualInfo →
    //    VirtualInfo so the decoder can materialise lazily.
    let count = outputs.len() as i32;
    let rd_virtuals_converted: Option<Vec<resume::VirtualInfo>> = rd_virtuals_rcs.map(|rcs| {
        let num_virtuals = rcs.len();
        rcs.iter()
            .map(|rd| resume::rd_virtual_to_virtual_info(rd, rd_consts, count, num_virtuals))
            .collect()
    });
    let rd_virtuals_slice = rd_virtuals_converted.as_deref();

    // 5. Construct ResumeDataDirectReader.  `outputs` enters as the
    //    deadframe (the JITed exit code stored fail_args here).
    //    Snapshot all_liveness once so the slice outlives the reader.
    let all_liveness = pyre_jit_trace::state::liveness_info_snapshot();
    let mut deadframe: Vec<i64> = outputs.clone();
    // `cpu.get_ref_value(deadframe, num)` reads a GC-traced frame; this copy
    // is not one, so root its Ref slots for as long as the reader decodes
    // them. Materializing a virtual below can collect.
    let _deadframe_roots = unsafe {
        resume::DeadFrameRefRoots::enter(&mut deadframe, |index| {
            types.get(index) == Some(&majit_ir::Type::Ref)
        })
    };
    let allocator = crate::eval::PyreBlackholeAllocator;
    let mut reader = resume::ResumeDataDirectReader::new(
        rd_numb,
        rd_consts,
        &all_liveness,
        majit_backend::FailArgSource::Slice(&deadframe),
        Some(types),
        None,
        &allocator,
    );

    // 6. resume.py:1324-1325 — prepare virtuals/pendingfields, then
    //    consume the vref + vable sections that precede the per-frame
    //    sections.
    let _resume_roots =
        resume::prepare_resume_heap_with_roots(&mut reader, rd_virtuals_slice, rd_pendingfields);
    let vinfo_dyn: &dyn resume::VirtualizableInfo = driver_vinfo.as_ref();
    let vrefinfo_dyn: &dyn resume::VRefInfo = driver.meta_interp().virtualref_info();
    if std::env::var_os("PYRE_DEOPT_PROBE").is_some() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static HITS: AtomicU64 = AtomicU64::new(0);
        let n = HITS.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "[deopt-probe] cranelift vinfo hand-off #{n} descr_addr={descr_addr:#x} novable={novable}"
        );
    }
    let vinfo_arg: Option<&dyn resume::VirtualizableInfo> =
        if novable { None } else { Some(vinfo_dyn) };
    reader.consume_vref_and_vable(Some(vrefinfo_dyn), vinfo_arg, None);

    // 7. resume.py:1339 jitcodes[jitcode_pos] lookup — same shape as
    //    blackhole_resume_via_rd_numb's resolve_jitcode,
    //    but returns the (jitcode, pc, op_live) triple
    //    consume_all_sections_into_vec needs to compute the per-section
    //    liveness offset.
    let (op_live_i32, _op_catch_exception, _op_rvmprof_code) =
        pyre_jit_trace::state::blackhole_control_opcodes();
    // op_live is the `-live-` opcode byte that JitCode::get_live_vars_info
    // (translate/codewriter/jitcode.rs) uses to skip past the
    // op header.  state.rs returns it as i32 for the
    // `setup_cached_control_opcodes` API; we narrow here.  A negative
    // or out-of-range value means the control opcodes were not set up,
    // which would break the per-section walk — short-circuit to the
    // recovery_layout fallback instead.
    if !(0..=255).contains(&op_live_i32) {
        return false;
    }
    let op_live = op_live_i32 as u8;
    let resolve_jitcode = |jitcode_index: i32,
                           pc: i32|
     -> Option<(
        std::sync::Arc<majit_metainterp::jitcode::JitCode>,
        usize,
        u8,
    )> {
        if pc < 0 {
            return None;
        }
        let pyjitcode = pyre_jit_trace::state::pyjitcode_for_jitcode_index(jitcode_index)?;
        if pyjitcode.has_abort_opcode() {
            return None;
        }
        let resolved_pc = if pyjitcode.jitcode.can_decode_live_vars(pc as usize, op_live) {
            pc as usize
        } else {
            return None;
        };
        Some((pyjitcode.jitcode.clone(), resolved_pc, op_live))
    };

    // 8. Drive the per-section consume loop, appending decoded values
    //    into a fresh rebuilt vec.  Mirrors the
    //    `rebuild_state_after_failure(outputs, types, recovery)` walker:
    //    innermost-first concatenation of (i, r, f) sections.
    let mut rebuilt: Vec<i64> = Vec::with_capacity(outputs.len());
    if !reader.consume_all_sections_into_vec(&resolve_jitcode, &mut rebuilt) {
        // resolve_jitcode failure — leave outputs as-is so the
        // recovery_layout fallback path can take over.
        return false;
    }

    // 9. Replace outputs with rebuilt.
    *outputs = rebuilt;
    true
}

#[cfg(test)]
mod tests_bh_normalize_raise {
    use super::*;
    use pyre_interpreter::{PyErrorKind, compile_exec};

    #[test]
    fn bh_normalize_raise_varargs_rejects_builtin_callables_that_are_not_exception_classes() {
        let callable = pyre_interpreter::make_builtin_function(
            "len",
            pyre_interpreter::builtins::__majit_wrap_builtin_len,
        );
        let code = compile_exec("").expect("empty module should compile");
        let frame = pyre_interpreter::PyFrame::new(code);

        let frame_ptr = &*frame as *const pyre_interpreter::PyFrame;
        let result =
            bh_normalize_raise_varargs_with_frame(frame_ptr, callable, pyre_object::PY_NULL);
        let err = unsafe { pyre_interpreter::PyError::from_exc_object(result) };
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(
            err.message_text(),
            "exceptions must derive from BaseException"
        );
    }

    #[test]
    fn record_caught_blackhole_traceback_ignores_null_operands() {
        record_caught_blackhole_traceback(0, 0, 0, 0);
    }
}
