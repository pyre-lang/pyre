//! Bytecode evaluation loop — pure interpreter.
//!
//! JIT integration lives in pyre-jit/src/eval.rs. This module is
//! JIT-free: it processes bytecode instructions with no tracing,
//! no merge points, and no compiled-code hooks.

use crate::bytecode::{BinaryOperator, ComparisonOperator, Instruction};
use crate::*;
use crate::{
    ArithmeticOpcodeHandler, BranchOpcodeHandler, ConstantOpcodeHandler, ControlFlowOpcodeHandler,
    IterOpcodeHandler, LocalOpcodeHandler, NamespaceOpcodeHandler, OpcodeStepExecutor, PyError,
    PyErrorKind, PyResult, SharedOpcodeHandler, StackOpcodeHandler, StepResult, TruthOpcodeHandler,
    build_list_from_refs, build_map_from_refs, build_tuple_from_refs, decode_instruction_forward,
    ensure_range_iter, execute_opcode_step, unpack_sequence_exact_tos,
};
use crate::{locals_w, locals_w_mut};
use pyre_object::*;

use crate::call::call_callable;
use std::cell::Cell;

#[derive(Debug, Clone)]
pub struct Code {
    pub name: String,
    pub code: Option<PyObjectRef>,
}

impl Code {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            code: None,
        }
    }

    pub fn __repr__(&self) -> String {
        format!("<code {}>", self.name)
    }
}

// The current active exception (`sys.exc_info()` / bare `raise`) now lives
// on the per-thread `ExecutionContext` (`sys_exc_value`), reached via
// `get_current_exception` / `set_current_exception`; see those.
thread_local! {
    pub(crate) static CURRENT_FRAME: Cell<*mut PyFrame> = const { Cell::new(std::ptr::null_mut()) };

    static PYFRAME_ROOT_AREA: PyFrameRootArea = PyFrameRootArea {
        current_frame: CURRENT_FRAME.with(|frame| frame as *const _),
        last_exec_ctx: crate::call::capture_last_exec_ctx_cell(),
        import_roots: crate::importing::capture_import_root_area(),
        in_flight_exception: IN_FLIGHT_EXCEPTION.with(|cell| cell as *const _),
        bh_last_exception: majit_metainterp::blackhole::BH_LAST_EXC_VALUE
            .with(|cell| cell as *const _),
        guard_exception: majit_metainterp::blackhole::GUARD_EXC_VALUE
            .with(|cell| cell as *const _),
        jit_pending_exception: crate::stack_check::capture_jit_pending_exception_area(),
        pending_call_error: crate::call::capture_pending_call_error_area(),
        pending_hash_error: crate::baseobjspace::capture_pending_hash_error_area(),
    };
}

struct PyFrameRootArea {
    current_frame: *const Cell<*mut PyFrame>,
    last_exec_ctx: *const (),
    import_roots: *const (),
    in_flight_exception: *const Cell<PyObjectRef>,
    bh_last_exception: *const Cell<i64>,
    guard_exception: *const Cell<i64>,
    jit_pending_exception: *const Cell<i64>,
    pending_call_error: *const (),
    pending_hash_error: *const (),
}
use crate::pyframe::PyFrame;

/// Saves the previous `CURRENT_FRAME` and (when EC was modified) the
/// previous `ec.topframeref` so they can be restored on Drop. The two
/// pointers are pushed onto `majit_gc::shadow_stack` rather than a local
/// `Vec` — this matches RPython's `framework.py` shadow-stack
/// (rpython/memory/gctransform/shadowstack.py:281) and lets the GC's
/// root-walker forward both pointers in place when a minor
/// collection runs while the guard is on the stack.
pub struct CurrentFrameGuard {
    save_point: usize,
    ec: *mut PyExecutionContext,
    ec_top_root_index: Option<usize>,
}

impl Drop for CurrentFrameGuard {
    fn drop(&mut self) {
        // Read forwarded values from the shadow stack before pop_to so we
        // observe any in-place updates the GC may have made.
        let previous = majit_gc::shadow_stack::get(self.save_point);
        let previous_ec_top = self
            .ec_top_root_index
            .map(majit_gc::shadow_stack::get)
            .unwrap_or(majit_ir::GcRef::NULL);
        majit_gc::shadow_stack::pop_to(self.save_point);
        CURRENT_FRAME.with(|current| current.set(previous.0 as *mut PyFrame));
        if !self.ec.is_null() {
            unsafe {
                (*self.ec).topframeref = previous_ec_top.0 as *mut PyFrame;
            }
        }
    }
}

fn push_current_frame_previous_root(
    previous: *mut PyFrame,
    ec: *mut PyExecutionContext,
    previous_ec_top: *mut PyFrame,
) -> CurrentFrameGuard {
    let save_point = majit_gc::shadow_stack::push(majit_ir::GcRef(previous as usize));
    let ec_top_root_index = if ec.is_null() {
        None
    } else {
        Some(majit_gc::shadow_stack::push(majit_ir::GcRef(
            previous_ec_top as usize,
        )))
    };
    CurrentFrameGuard {
        save_point,
        ec,
        ec_top_root_index,
    }
}

/// Flat TLS read of the per-thread `CURRENT_FRAME` slot — the frame a builtin
/// was called from, since a builtin call creates no Python frame of its own.
/// Null when no frame is installed.
///
/// The slot is runtime-mutable, not a build-time constant, so the JIT
/// residualises the read instead of tracing into it (`@dont_look_inside`, the
/// [`get_current_exception`] shape).
#[majit_macros::dont_look_inside]
pub fn current_frame() -> *mut PyFrame {
    CURRENT_FRAME.with(|current| current.get())
}

pub fn install_current_frame(frame: &mut PyFrame) -> CurrentFrameGuard {
    let previous = CURRENT_FRAME.with(|current| {
        let previous = current.get();
        current.set(frame as *mut PyFrame);
        previous
    });
    // executioncontext.py `enter()` parity: link the frame into the
    // topframeref/f_backref chain so walkers (GC roots, sys._getframe)
    // can iterate all active frames. `eval_frame_plain` calls
    // `ExecutionContext::enter` before installing TLS-only state, but
    // the JIT portal path enters through this helper directly.
    let ec = crate::call::getexecutioncontext() as *mut PyExecutionContext;
    // `ExecutionContext::enter` refuses a frame that is already the top —
    // relinking it would make `f_backref` name the frame itself, and
    // `walk_pyframe_roots` follows `f_backref` with no cycle guard.  This door
    // owes the same refusal, because the enter is no longer performed only
    // here: `try_walker_call_assembler_self_recursive`
    // (`pyre-jit-trace/src/jitcode_dispatch/inline_call.rs`) records it ahead
    // of its CALL_ASSEMBLER, and the force and deopt legs can hand that same,
    // already-entered frame to a portal helper.  Already entered means this
    // guard owns neither the link nor its matching restore; the surrounding
    // `ExecutionContext.enter`/`leave` scope owns both.
    let enters_chain = !ec.is_null()
        && !std::ptr::eq(
            crate::executioncontext::vref_referent(unsafe { (*ec).topframeref }),
            frame as *mut PyFrame,
        );
    let previous_ec_top = if enters_chain {
        unsafe {
            let top = (*ec).topframeref;
            (*ec).topframeref = frame as *mut PyFrame;
            top
        }
    } else {
        std::ptr::null_mut()
    };
    if enters_chain || ec.is_null() {
        // Barrier for the same reason as `ExecutionContext::enter`: this is a
        // traced `Type::Ref` store and `frame` can be an old-generation frame
        // taking a young predecessor.
        pyre_object::gc_hook::try_gc_write_barrier(
            frame as *mut PyFrame as pyre_object::gc_hook::GCREF,
        );
        majit_gc::bh_probe_note_store(
            frame as *mut PyFrame as usize,
            crate::pyframe::PYFRAME_F_BACKREF_OFFSET,
            3,
        );
        frame.f_backref = if ec.is_null() {
            previous
        } else {
            previous_ec_top
        };
    }
    // A guard that did not write the execution-context chain owns no restore
    // either.  The surrounding `ExecutionContext.enter`/`leave` pair keeps
    // ownership of an already-entered frame; restoring its predecessor here
    // would unlink the live outer activation when this guard is dropped.
    push_current_frame_previous_root(
        previous,
        if enters_chain {
            ec
        } else {
            std::ptr::null_mut()
        },
        previous_ec_top,
    )
}

/// Install only the TLS current-frame root.
///
/// Use this after `ExecutionContext::enter()` has already linked
/// `frame.f_backref`.  PyPy has one frame chain (`ec.topframeref`);
/// pyre's `CURRENT_FRAME` is an extra GC/super() TLS root and must not
/// overwrite the RPython `f_backref` chain once EC owns it.
pub fn install_current_frame_tls_only(frame: &mut PyFrame) -> CurrentFrameGuard {
    let previous = CURRENT_FRAME.with(|current| {
        let previous = current.get();
        current.set(frame as *mut PyFrame);
        previous
    });
    push_current_frame_previous_root(previous, std::ptr::null_mut(), std::ptr::null_mut())
}

/// Re-anchor a caller frame across a callee call that may run a moving minor
/// collection.
///
/// The interpreter holds the running frame as a raw `&mut PyFrame`, which the
/// GC does not treat as a managed reference.  When the callee allocates enough
/// to trigger a minor collection, the caller frame is relocated and that raw
/// pointer is left aimed at the abandoned nursery copy.  A field write through
/// it afterwards — the `CALL` result push and its `valuestackdepth` bump —
/// then lands on dead memory, so the live frame keeps a stack depth one slot
/// short and the next opcode reads the wrong operands.
///
/// Pushing the frame onto the shadow stack lets the root walker forward it in
/// place during the collection; `live()` reads the forwarded pointer back.
/// This mirrors the JIT eval layer's `FrameRoot`.
///
/// The JIT models the anchor as its depth word. `#[repr(transparent)]` is
/// that declaration: the only sized field is `depth`, and a borrow of the
/// anchor is that word.
#[repr(transparent)]
pub struct FrameAnchor {
    depth: usize,
    /// The shadow stack is per-thread, so a depth taken on one thread names a
    /// different slot on another. The marker is what keeps an anchor from
    /// being sent or shared across threads now that the type is public.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl FrameAnchor {
    pub fn new(frame: &mut PyFrame) -> Self {
        unsafe { Self::from_raw(frame as *mut PyFrame) }
    }

    /// Anchor a frame the caller holds as a raw pointer.
    ///
    /// `executioncontext`, the frame typedef and the introspection builtins
    /// carry `*mut PyFrame` rather than a reference, because the same frame
    /// stays reachable through the `f_backref` chain it was read off.  Minting
    /// a `&mut` for the length of the anchor would claim an exclusivity none of
    /// them has.
    ///
    /// # Safety
    /// `frame` must be null or name a live `PyFrame`.  A null is an
    /// anticipated input for the residual helpers, which take the caller frame
    /// as a raw operand the emit site may not have; the root walker skips a
    /// null slot, so anchoring one costs a push and answers null from
    /// [`Self::live`].
    pub unsafe fn from_raw(frame: *mut PyFrame) -> Self {
        let depth = frame_anchor_push(frame);
        Self {
            depth,
            _not_send: std::marker::PhantomData,
        }
    }

    /// Read the anchored frame back out of its shadow-stack slot.
    ///
    /// The anchor is its `depth` word, and a borrow of that by-value struct
    /// is the same word, so `self.depth` is `depth`. The body is the
    /// `dont_look_inside` read `frame_anchor_live`.
    pub fn live(&self) -> *mut PyFrame {
        frame_anchor_live(self.depth)
    }
}

impl Drop for FrameAnchor {
    fn drop(&mut self) {
        frame_anchor_release(self.depth);
    }
}

/// Shadow-stack slot publication behind [`FrameAnchor`].  The slot ops live
/// in `majit_gc`, which has no jitcode graphs, so a walked handler graph
/// reaching them found only a symbolic path hash.  First-party
/// `dont_look_inside` wrappers with word-sized signatures keep each op a
/// bound residual the tracer can execute (`jit_trace_fnaddrs()` rows).
#[majit_macros::dont_look_inside]
pub fn frame_anchor_push(frame: *mut PyFrame) -> usize {
    majit_gc::shadow_stack::push(majit_ir::GcRef(frame as usize))
}

/// Read the anchored frame back from its shadow-stack slot; a collection
/// between push and read leaves the forwarded address here.
#[majit_macros::dont_look_inside]
pub fn frame_anchor_live(depth: usize) -> *mut PyFrame {
    majit_gc::shadow_stack::get(depth).0 as *mut PyFrame
}

/// Release the anchor's slot (and any deeper ones) on drop.
#[majit_macros::dont_look_inside]
pub fn frame_anchor_release(depth: usize) {
    majit_gc::shadow_stack::try_pop_to(depth);
}

/// One-word residual-call ABI for the three slot ops, and — as separate
/// functions below — for the two [`FrameAnchor`] methods a walked handler
/// graph reaches.
///
/// A value-returning residual is lowered as `(i64 x n) -> i64`; a raw
/// `*mut PyFrame` / `usize` signature is narrower than a word on wasm32,
/// where `call_indirect` type-checks its callee.
///
/// Each spelling gets its own bridge because `jit_trace_fnaddrs()` reads one
/// address back as one function: two unrelated paths sharing an address is
/// what `registered_paths_sharing_an_address_are_alias_spellings` refuses.
/// [`majit_ir::icf_identity`] keeps LLVM MergeFunctions from folding these
/// bridges onto `shadow_stack_*_word` (`drain_list_append`).
pub extern "C" fn frame_anchor_push_jit_abi(frame: *mut PyFrame) -> i64 {
    majit_ir::icf_identity!("eval::frame_anchor_push_jit_abi");
    frame_anchor_push(frame) as i64
}

/// One-word residual-call ABI for [`frame_anchor_live`].
pub extern "C" fn frame_anchor_live_jit_abi(depth: i64) -> i64 {
    majit_ir::icf_identity!("eval::frame_anchor_live_jit_abi");
    frame_anchor_live(depth as usize) as i64
}

/// One-word residual-call ABI for [`frame_anchor_release`].
pub extern "C" fn frame_anchor_release_jit_abi(depth: i64) {
    majit_ir::icf_identity!("eval::frame_anchor_release_jit_abi");
    frame_anchor_release(depth as usize);
}

/// One-word residual-call ABI for [`FrameAnchor::new`], which the `anchor`
/// handler graph residualizes rather than inlining.
///
/// The anchor is one word, and the trace keeps that word: the construction is
/// handed back without running `Drop`, exactly as the aggregate return did
/// when the method itself was the registered target.
pub extern "C" fn frame_anchor_new_jit_abi(frame: *mut PyFrame) -> i64 {
    majit_ir::icf_identity!("eval::frame_anchor_new_jit_abi");
    // SAFETY: the residual's slot holds the frame the walked graph read it
    // from, or null, which `from_raw` anticipates.
    let anchor = unsafe { FrameAnchor::from_raw(frame) };
    let depth = anchor.depth;
    std::mem::forget(anchor);
    depth as i64
}

/// One-word residual-call ABI for [`FrameAnchor::live`].
///
/// `front::mir` aliases an `Rvalue::Ref` over a bare local to that local's own
/// Variable without emitting an address-of, so the method's `&self` reaches
/// the residual as the one-word anchor's value — the depth — rather than a
/// pointer to it.
pub extern "C" fn frame_anchor_live_method_jit_abi(anchor: i64) -> i64 {
    majit_ir::icf_identity!("eval::frame_anchor_live_method_jit_abi");
    let anchor = std::mem::ManuallyDrop::new(FrameAnchor {
        depth: anchor as usize,
        _not_send: std::marker::PhantomData,
    });
    anchor.live() as i64
}

/// Forward one reference field, skipping the unset ones.
///
/// Upstream tests the slot before it invokes the visitor on either side of the
/// walk: `walk_stack_root` (shadowstack.py) reads the slot and calls `invoke`
/// only `if content:`, and `trace` / `_trace_slow_path` reach `callback`
/// through `points_to_valid_gc_object` (base.py).  The raw walkers below name a
/// fixed field list per carrier rather than a computed offset table, and on a
/// builtin function most of that list is unset — no closure, no defaults, no
/// annotations, no `__dict__` — so the test belongs here, ahead of the
/// visitor's indirect call and of the heap-membership probe behind it.  The
/// major root walk of a bare `print(1)` visited 39,360 empty slots out of
/// 60,341 roots before this.
#[inline]
fn visit_slot(slot: &mut PyObjectRef, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    if slot.is_null() {
        return;
    }
    // The cast reinterprets one machine word in place, the same spelling every
    // caller below uses for its own fields.
    visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
}

/// rpython/memory/gctransform/framework.py `root_walker.walk_roots` parity:
/// expose every live slot of `PyFrame.locals_cells_stack_w` on the active
/// f_backref chain as a GC root.
///
/// pyre's JIT-compiled code allocates W_IntObject / result boxes into the
/// nursery (`NewWithVtable` → `gc_alloc_typed_nursery_shim`). When the
/// nursery fills and a minor collection runs, only registered roots are
/// forwarded — an unforwarded nursery ref is left addressing a corpse.
/// `Nursery::reset` only rewinds the free pointer on native (it zero-fills
/// on wasm32, and writes the 0xAA poison only when that debug mode is on),
/// so the corpse keeps its forwarding header until something is allocated
/// over it and the stale ref reads whichever of the two it finds. The
/// interpreter stores live
/// refs in `PyFrame.locals_cells_stack_w`; without this walker those
/// slots turn into NULL-`ob_type` stale pointers on the next LOAD_FAST
/// (reproduced by `inline_helper` n >= 10000).
///
/// Walks 0..`valuestackdepth` entries because that range covers both
/// the always-live locals+cells prefix (slots `0..nlocals+ncells`,
/// written once at frame setup) and the operand stack region
/// (`nlocals+ncells..valuestackdepth`). Dead stack slots past
/// `valuestackdepth` are skipped.
/// Extra-root one prebuilt slot. A GC-managed object is already in the
/// heap: `inspector.py enumerate_all_roots` lists it and `gc.trace` finds
/// its children, so nested `walk_raw_*` would report those children as
/// extra roots. A `malloc_typed` / Box immortal is outside the arenas,
/// so the nested walk is its only tracing path.
unsafe fn visit_prebuilt_declaration(
    slot: &mut PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
    walk_immortal: bool,
) {
    if slot.is_null() {
        return;
    }
    unsafe {
        visitor(&mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef));
        if pyre_object::gc_hook::try_gc_owns_object(*slot as pyre_object::gc_hook::GCREF) {
            return;
        }
        walk_raw_function_roots(*slot, visitor);
        walk_raw_getset_roots(*slot, visitor);
        walk_raw_wrapped_function_roots(*slot, visitor);
        if walk_immortal {
            walk_raw_immortal_roots(*slot, visitor);
        }
    }
}

/// `MethodCache` trace: GC-owned slots are edges of the cache object.
/// A `malloc_typed` carrier has no header, so only its interior fields
/// are visited.
pub unsafe fn method_cache_custom_trace(
    _container: usize,
    visitor: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    unsafe {
        crate::baseobjspace::trace_method_cache_entries(&mut |slot| {
            if pyre_object::gc_hook::try_gc_owns_object(*slot as pyre_object::gc_hook::GCREF) {
                visitor(slot as *mut PyObjectRef as *mut majit_ir::GcRef);
                return;
            }
            let mut fwd = |r: &mut majit_ir::GcRef| visitor(r);
            walk_raw_function_roots(*slot, &mut fwd);
            walk_raw_getset_roots(*slot, &mut fwd);
            walk_raw_wrapped_function_roots(*slot, &mut fwd);
        });
    }
}

unsafe fn walk_raw_function_roots(
    value: PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        // `is_function_carrier`, not `is_function`: `SLOT_WRAPPER_TYPE` is
        // registered to the same `function_tid` as `FUNCTION_TYPE`
        // (`pyre-jit/src/eval.rs` `register_vtable_for_type`), so the
        // collector's census already says a slot wrapper carries Function's
        // offsets. The carriers are Box-immortal, which makes this walk their
        // only tracing path, and `typedef.rs` stamps `w_qualname`/`w_objclass`
        // onto one *before* `function_retag_slot_wrapper` swaps `ob_type` —
        // the narrower gate then refused the whole population afterwards.
        if value.is_null() || !crate::is_function_carrier(value) {
            return;
        }
        let func = &mut *(value as *mut crate::function::Function);
        visitor(&mut *(&mut func.code as *mut *const () as *mut majit_ir::GcRef));
        // The code object caches its own globals dict (`PyCode.w_globals`).
        // Reuse the same walk for managed custom tracing and bootstrap code.
        walk_raw_code_roots(func.code as PyObjectRef, visitor);
        // `function.py:51 self.name` — the wrapped `__name__`, stamped at
        // construction and replaced by `f.__name__ = ...`. Both stores already
        // run `function_write_barrier`, but a Box-immortal function is reached
        // only through this walk, so leaving the slot out gave the string a
        // rename installs no root at all. A managed function reaches the same
        // field through `FUNCTION_GC_TYPE_ID`'s offsets, which is why the gap
        // showed up on immortal functions alone.
        visit_slot(&mut func.w_name, visitor);
        visit_slot(&mut func.closure, visitor);
        visit_slot(&mut func.defs_w, visitor);
        visit_slot(&mut func.w_kw_defs, visitor);
        visit_slot(&mut func.w_module, visitor);
        visit_slot(&mut func.w_func_globals_obj, visitor);
        visit_slot(&mut func.w_builtins, visitor);
        visit_slot(&mut func.w_ann, visitor);
        visit_slot(&mut func.w_annotate, visitor);
        visit_slot(&mut func.w_func_dict, visitor);
        visit_slot(&mut func.w_typeparams, visitor);
        visit_slot(&mut func.w_doc, visitor);
        visit_slot(&mut func.w_qualname, visitor);
        visit_slot(&mut func.w_objclass, visitor);
        visit_slot(&mut func.w_text_signature, visitor);
        // BuiltinFunction.w_moduleobj is an ordinary movable module reference.
        // Builtin functions are immortal, so only this raw-root walker can
        // forward the slot during a collection.
        visit_slot(&mut func.w_moduleobj, visitor);
        visit_slot(&mut func.lifeline, visitor);
    }
}

/// Forward a `PyCode`'s cached globals, realized constants, qualname and
/// mapdict-method entries. This is both the managed wrapper's custom trace and
/// the explicit walk for bootstrap wrappers outside the collector. No-op for
/// non-code values.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn walk_raw_code_roots(
    value: PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        if value.is_null() || !crate::pycode::is_code(value) {
            return;
        }
        // A GC trace callback reports one object's direct edges; the mark
        // worklist provides the transitive traversal.
        walk_enrolled_code_roots(value, visitor);
    }
}

/// Report the direct fields of a `PyCode` already known to be a code wrapper.
///
/// [`walk_raw_code_roots`] first asks [`is_code`], which reads `ob_type`.
/// The collector's type-directed `PyCode` trace names code wrappers by type
/// id; it skips that predicate so a wrapper whose `ob_type` word is momentarily unreadable
/// (a just-written stable allocation, a header the visitor has not yet
/// forwarded) still has `co_consts_w` visited.
pub unsafe fn walk_enrolled_code_roots(
    value: PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    if value.is_null() {
        return;
    }
    unsafe {
        let code = &mut *(value as *mut crate::pycode::PyCode);
        visitor(&mut *(&mut code.w_globals as *mut PyObjectRef as *mut majit_ir::GcRef));
        // typedef.py `make_weakref_descr(PyCode)` adds the strong
        // `_lifeline_` field to PyCode.  Its own rweakrefs point weakly
        // back to this code object; tracing this edge keeps callbacks
        // alive exactly until their owner becomes unreachable.
        visitor(&mut *(&mut code.w_weakreflifeline as *mut PyObjectRef as *mut majit_ir::GcRef));
        // The realized `co_qualname` is an ordinary movable string object
        // shared by every function built from this code.
        visitor(&mut *(&mut code.w_qualname as *mut PyObjectRef as *mut majit_ir::GcRef));
        // `co_name` is realized and retained the same way.
        visitor(&mut *(&mut code.w_name as *mut PyObjectRef as *mut majit_ir::GcRef));
        // `PyCode.co_consts_w` is a GCREF to a `GcArray(OBJECTPTR)` whose
        // own trace names the items. An item store barriers the array
        // (`FixedObjectArray::set_ref`), so an old array holding a young
        // item is in the remembered set.
        visitor(
            &mut *(&mut code.co_consts_w as *mut *mut pyre_object::FixedObjectArray
                as *mut majit_ir::GcRef),
        );
        // `pycode.py _immutable_fields_ co_names_w[*]`. The array's own
        // trace names the interned strings.
        visitor(
            &mut *(&mut code.co_names_w as *mut *mut pyre_object::FixedObjectArray
                as *mut majit_ir::GcRef),
        );
        // mapdict.py CacheEntry.w_method is the cache's sole GC
        // reference. PyPy traces it as part of the live PyCode; do the same
        // here now that managed code wrappers reach this direct-field walker.
        if !code.mapdict_caches.is_null() {
            for entry in (&mut *code.mapdict_caches).iter_mut().flatten() {
                visitor(&mut *(&mut entry.w_method as *mut PyObjectRef as *mut majit_ir::GcRef));
            }
        }
    }
}

/// Mark the GC-managed children of a `W_BaseException`.
///
/// Exception allocation is a nursery `malloc_fixedsize`
/// (`alloc_exception_nursery` → `try_gc_alloc_collecting_rooted`, which runs
/// `collect_and_reserve` when the nursery is full). Immortal singletons and
/// the GC-allocation-failure fallback still use `malloc_typed`
/// (`interp_exceptions.rs`). For the `malloc_typed` family the collector
/// never traces the exception at all — the root visitor's
/// `is_managed_heap_object` guard short-circuits and `mark_object` is never
/// reached — and a minor moves a nursery instance, forwarding this slot
/// before it traces the children. Its
/// `args_w` tuple, `w_errno` / `w_strerror` / `w_filename` ints/strings,
/// `w_traceback` / `w_context` / `w_cause`, `w_dict`, … are ordinary GC-managed
/// objects, so when an exception is the only holder of those children (a caught
/// `except X as e` bound to a frame local) a collection sweeps them and a
/// later `e.args` / `e.errno` reads freed memory. Visit every GC pointer
/// slot of the instance layout in place — selected by `ob_type` so a
/// `_getusercls` instance also reaches mapdict `storage` — the same shape
/// `walk_raw_function_roots` / `walk_raw_getset_roots`
/// use for Box/`malloc_typed`-held children. No-op for non-exception values.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn walk_raw_exception_roots(
    value: PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        if value.is_null() {
            return;
        }
        // Positive predicate (see `walk_raw_getset_roots`): `!is_exception`
        // over a cross-crate bool is `UnaryNotUnknownOperand` to the annotator.
        if pyre_object::interp_exceptions::is_exception(value) {
            let offsets = pyre_object::interp_exceptions::exception_unmanaged_gc_ptr_offsets(value);
            for &offset in offsets {
                let slot = (value as usize + offset) as *mut PyObjectRef;
                visitor(&mut *(slot as *mut majit_ir::GcRef));
            }
        }
    }
}

/// Maximum recursion depth for `walk_raw_immortal_roots`.  Immortal objects
/// chain a few levels at most (`map(f, filter(p, iter(seq)))`, a view's dict,
/// a tuple of iterators); this bounds the walk against a malformed cycle — the
/// only cycle protection.
const IMMORTAL_WALK_MAX_DEPTH: u32 = 8;

/// Force-forward the managed children of any `malloc_typed`-immortal REGISTERED
/// pyre object reachable from a root slot (a frame value-stack/locals slot, or
/// an explicit `gc_roots::pin_root` shadow-stack slot).  Such objects are
/// outside the GC arenas, so the marker skips them and their registered
/// `gc_ptr_offsets` trace never fires; a managed child held solely through one
/// is freed by a collection (a bare `for x in <expr>:` whose iterator sits on
/// the value stack, a `d.keys()` view, an immortal iterator's source list, an
/// `_pickle.Unpickler` pinned across `load`).  Drive off the per-type offset
/// registry so every present and future immortal type is covered.  Runs on both
/// minor and major collections via the collection-kind-agnostic `visitor`.
///
/// # Safety
/// `value` must be a valid `PyObjectRef` or null, and `visitor` must accept the
/// forwarded child slots for the duration of the walk.
pub unsafe fn walk_raw_immortal_roots(
    value: PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe { walk_immortal_rec(value, visitor, 0) };
}

unsafe fn walk_immortal_rec(
    value: PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
    depth: u32,
) {
    unsafe {
        if value.is_null() || depth >= IMMORTAL_WALK_MAX_DEPTH {
            return;
        }
        // A tagged immediate is an int with no managed children; short-circuit
        // before any predicate below dereferences its `ob_type`. Gated on
        // `CAN_BE_TAGGED` (default false).
        if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(value) {
            return;
        }

        // (A) A `malloc_typed`-immortal REGISTERED pyre object: the marker
        // ignores it, so follow its registered `gc_ptr_offsets` in place and
        // recurse.  Managed registered objects (`try_gc_owns_object`) are
        // traced by the marker already, so leave their offsets to `None` and
        // fall through — re-walking a managed graph here is redundant.
        // Positive predicate (no `!` over a cross-crate bool): compute the
        // offsets only for a non-owned object.
        let immortal_offsets =
            if pyre_object::gc_hook::try_gc_owns_object(value as pyre_object::gc_hook::GCREF) {
                None
            } else {
                pyre_object::gc_hook::offsets_for_pytype((*value).ob_type)
            };
        if let Some(offsets) = immortal_offsets {
            for &off in offsets {
                let slot = (value as usize + off) as *mut PyObjectRef;
                visitor(&mut *(slot as *mut majit_ir::GcRef));
                // Re-read the slot AFTER the visitor so a relocated child is
                // the one recursed into.
                walk_immortal_rec(*slot, visitor, depth + 1);
            }
            return;
        }

        // (B) A managed container reached AS A CHILD of an immortal (depth>=1):
        // its immortal elements would otherwise stay untraced-through (the
        // marker forwards element slots but does not recurse into an immortal
        // element's own children).  Enumerate element slots in place and
        // recurse.  Top-level (depth 0) managed containers are left to the
        // marker — matches prior scope.
        if depth >= 1 {
            if pyre_object::is_list(value) {
                if let Some((ptr, n)) = pyre_object::listobject::w_list_object_items_ptr_len(value)
                {
                    for i in 0..n {
                        let s = ptr.add(i) as *mut PyObjectRef;
                        visitor(&mut *(s as *mut majit_ir::GcRef));
                        walk_immortal_rec(*s, visitor, depth + 1);
                    }
                }
            } else if pyre_object::is_tuple(value) {
                // Dispatch on the concrete tuple layout (general vs specialised
                // arity-2): a specialised `_oo` stores objects inline, so a
                // general-tuple block read would mis-cast its inline slots.
                pyre_object::tupleobject::w_tuple_walk_gc_refs(
                    value,
                    &mut |s: *mut PyObjectRef| {
                        visitor(&mut *(s as *mut majit_ir::GcRef));
                        walk_immortal_rec(*s, visitor, depth + 1);
                    },
                );
            } else if pyre_object::is_dict(value) {
                let strat = pyre_object::dictmultiobject::w_dict_get_strategy(value);
                strat.walk_gc_refs(value, &mut |s: *mut PyObjectRef| {
                    visitor(&mut *(s as *mut majit_ir::GcRef));
                    walk_immortal_rec(*s, visitor, depth + 1);
                });
            } else if pyre_object::setobject::is_set_or_frozenset(value) {
                // `set` and `frozenset` share the `W_SetObject` layout and the
                // marker's `set_object_custom_trace`, so one walker covers both.
                pyre_object::setobject::w_set_walk_gc_refs(value, &mut |s: *mut PyObjectRef| {
                    visitor(&mut *(s as *mut majit_ir::GcRef));
                    walk_immortal_rec(*s, visitor, depth + 1);
                });
            }
        }
    }
}

/// Mark the GC-reachable children of a `getset_descriptor`
/// (`GetSetProperty`).  The descriptor itself is Box-immortal
/// (`pyre_class` `allocate` → `malloc_typed`), so its `W_TYPE_GC_TYPE_ID`
/// custom trace never fires.  Its `fget`/`fset`/`fdel` getters are
/// GC-managed `try_gc_alloc_stable` functions — non-moving but still
/// *collectable* — so when a descriptor's only holder is a Box-immortal
/// type dict, nothing marks the getters reachable and the collector frees
/// them, leaving `descr.fget` dangling (a fresh `obj.__dict__` after a
/// collection then calls a freed getter → SIGSEGV).  Visit every
/// `PyObjectRef` field and recurse into the getter functions, the same
/// shape `walk_raw_function_roots` uses for Box-held function children.
/// No-op for non-descriptor values.
unsafe fn walk_raw_getset_roots(value: PyObjectRef, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    unsafe {
        if value.is_null() {
            return;
        }
        // Positive predicate: the annotator cannot lower `!` over a
        // cross-crate bool result (`UnaryNotUnknownOperand`), so guard with
        // a positive `if` rather than negating `is_getset_property`.
        if pyre_object::typedef::is_getset_property(value) {
            let d = &mut *(value as *mut pyre_object::typedef::GetSetProperty);
            visit_slot(&mut d.fget, visitor);
            visit_slot(&mut d.fset, visitor);
            visit_slot(&mut d.fdel, visitor);
            visit_slot(&mut d.doc, visitor);
            visit_slot(&mut d.reqcls, visitor);
            visit_slot(&mut d.name, visitor);
            visit_slot(&mut d.w_objclass, visitor);
            visit_slot(&mut d.w_qualname, visitor);
            // The getters are functions whose own children (code / globals /
            // defaults) must stay reachable as well.
            walk_raw_function_roots(d.fget, visitor);
            walk_raw_function_roots(d.fset, visitor);
            walk_raw_function_roots(d.fdel, visitor);
        }
    }
}

/// Mark the GC-reachable children of the function a `staticmethod` /
/// `classmethod` wraps.  A builtin type dict binds the wrapper, not the
/// function, so `walk_raw_function_roots` applied to the dict value stops at
/// the wrapper and never descends to `w_function`.  That function is
/// Box-immortal — it never moves and so is never traced — while the metadata
/// `TypeCache.build` stamps onto it (`w_qualname`, `w_objclass`) and its
/// lazily allocated `w_func_dict` are ordinary young objects.  Without this
/// walk a minor collection leaves those slots pointing into vacated nursery
/// memory (`str.maketrans`, `dict.fromkeys`).  No-op for other values.
unsafe fn walk_raw_wrapped_function_roots(
    value: PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        if value.is_null() {
            return;
        }
        // Positive predicates (see `walk_raw_getset_roots`): `!is_staticmethod`
        // over a cross-crate bool is `UnaryNotUnknownOperand` to the annotator.
        if pyre_object::function::is_staticmethod(value) {
            walk_raw_function_roots(
                pyre_object::function::w_staticmethod_get_func(value),
                visitor,
            );
        }
        if pyre_object::function::is_classmethod(value) {
            walk_raw_function_roots(
                pyre_object::function::w_classmethod_get_func(value),
                visitor,
            );
        }
    }
}

/// Box-immortal builtin types never have their `W_TYPE_GC_TYPE_ID` custom
/// trace fired, but their namespaces, `bases`, and `weak_subclasses` can hold
/// young GC objects after startup.  Walk every registered builtin type (and
/// the rare pre-GC heap-type fallback) as pinned roots so those children stay
/// live and their slots are forwarded.  GC-managed heap types reach the same
/// fields through their own `W_TYPE_GC_TYPE_ID` custom trace.
unsafe fn walk_builtin_type_dicts_gc(forward: &mut dyn FnMut(&mut PyObjectRef)) {
    unsafe {
        for addr in pyre_object::typeobject::snapshot_builtin_type_roots() {
            let w_type = addr as PyObjectRef;
            if w_type.is_null() {
                continue;
            }
            // Positive predicate (see `walk_raw_getset_roots`): `!is_type`
            // over a cross-crate bool is `UnaryNotUnknownOperand` to the
            // annotator, so guard with a positive `if`.
            if pyre_object::is_type(w_type) {
                if pyre_object::gc_hook::try_gc_owns_object(w_type as pyre_object::gc_hook::GCREF) {
                    continue;
                }
                // `bases` is a movable tuple created at class definition and
                // held only by the Box-immortal type; forward it in place.
                let bases_slot =
                    &mut (*(w_type as *mut pyre_object::typeobject::W_TypeObject)).bases;
                forward(bases_slot);
                let t = &mut *(w_type as *mut pyre_object::typeobject::W_TypeObject);
                // The metatype.  `type_object_custom_trace` forwards it too,
                // but never runs for the Box-immortal owners this walk stands
                // in for.  It is the immortal `type` for all but a type built
                // through `PyType_FromMetaclass`, whose metatype is an
                // ordinary object with no other root here.
                forward(&mut t.ob_header.w_class);
                forward(&mut t.w_name);
                forward(&mut t.w_qualname);
                forward(&mut t.w_doc);
                // Heap and builtin types both hold a managed W_DictObject.
                // Forward the field itself; the dict's custom trace walks its
                // keys and values during a major collection. During a minor,
                // an already-old dict is not recursively rescanned merely
                // because this root slot is visited, so explicitly descend
                // through the strategy storage as the translated prebuilt
                // object's trace function does.
                let dict_slot = &mut t.dict as *mut pyre_object::gc_hook::GCREF as *mut PyObjectRef;
                forward(&mut *dict_slot);
                forward(&mut t.lifeline);
                pyre_object::dictmultiobject::w_dict_walk_gc_refs(*dict_slot, &mut |slot| {
                    forward(slot)
                });
                // `weak_subclasses` holds `w_weakref_new` (`try_gc_alloc`)
                // young WEAKREF GcStructs whose only strong root is this
                // off-GC list; forward each slot in place so the WEAKREF
                // survives collection (its `weakptr` payload is invalidated
                // separately by the collector's weakref scan).  Without this,
                // the first collection reclaims the weakref and the base's
                // `weak_subclasses[i]` dangles — a UAF on the next
                // `mutated()` / `w_type_get_subclasses` deref.  The
                // `W_TYPE_GC_TYPE_ID` custom trace performs the same walk for
                // a heap type.
                if t.weak_subclasses.is_null() {
                    // No subclasses recorded.
                } else {
                    let subs = &mut *t.weak_subclasses;
                    for slot in subs.iter_mut() {
                        forward(
                            &mut *(slot as *mut *mut pyre_object::weakref::Weakref
                                as *mut PyObjectRef),
                        );
                    }
                }
                // `mro_w` is a stable type-9 GcArray block (`alloc_mro_block_gc`)
                // whose only strong root is this immortal type. The custom
                // trace `type_object_custom_trace` never fires for a Box-immortal
                // owner, so forward the block field slot here: the collector
                // marks the tid-9 block and its varsize walker forwards
                // items[0..len]. Omitting this lets the first major collection
                // sweep the block — a UAF on the next MRO read. Guard on GC
                // ownership so the `std::alloc` bootstrap fallback (not owned)
                // is left in place.
                if !t.mro_w.is_null()
                    && pyre_object::gc_hook::try_gc_owns_object(
                        t.mro_w as pyre_object::gc_hook::GCREF,
                    )
                {
                    forward(&mut *(std::ptr::addr_of_mut!(t.mro_w) as *mut PyObjectRef));
                }
            }
        }
    }
}

/// Whether `PYRE_INTERP_RETURN_LOG` asks the RETURN_VALUE handler to trace
/// every returning frame.
///
/// Read once: `finish_value` runs on every interpreted return, and `getenv`
/// takes a lock and scans the environment array, so an uncached read puts that
/// scan on the return path of every Python-level call.
#[cfg(not(feature = "sandbox"))]
#[majit_macros::elidable_cannot_raise]
fn interp_return_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_INTERP_RETURN_LOG").is_some())
}

/// Report a returning frame for `PYRE_INTERP_RETURN_LOG`.
#[cfg(not(feature = "sandbox"))]
#[majit_macros::dont_look_inside]
fn interp_return_log(frame: &PyFrame, value: PyObjectRef) {
    unsafe {
        let code_ptr = crate::pyframe::pyframe_get_pycode(frame);
        let name = if !code_ptr.is_null() {
            (*code_ptr).obj_name.as_str()
        } else {
            "?"
        };
        let arg0_intval = {
            let lw = locals_w!(frame);
            if !lw.is_empty() {
                let v = lw[0];
                if !v.is_null() && pyre_object::pyobject::is_int(v) {
                    Some(pyre_object::intobject::w_int_get_value(v))
                } else {
                    None
                }
            } else {
                None
            }
        };
        let ret_intval = if !value.is_null() && pyre_object::pyobject::is_int(value) {
            Some(pyre_object::intobject::w_int_get_value(value))
        } else {
            None
        };
        let f_back = frame.f_backref as usize;
        eprintln!(
            "[interp] return name={} arg0={:?} ret={:?} frame={:p} f_back=0x{:x} ret_ref=0x{:x}",
            name, arg0_intval, ret_intval, frame as *const _, f_back, value as usize
        );
    }
}

/// Whether the incminimark-parity minor-collection skip of clean prebuilt
/// structures is enabled (`PYRE_GC_PREBUILT_REMEMBER=0` opts out, restoring
/// the rescan-everything-every-minor behavior).
fn gc_prebuilt_remember_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        #[cfg(not(feature = "sandbox"))]
        {
            std::env::var("PYRE_GC_PREBUILT_REMEMBER").as_deref() != Ok("0")
        }
        // The host env is off-limits under sandbox; keep the parity default
        // (the prebuilt-remember minor-collection skip enabled).
        #[cfg(feature = "sandbox")]
        {
            true
        }
    })
}

/// Whether `PYRE_RERAISE_DIAG` asks `RERAISE` to describe an operand that is
/// not an exception before it raises the `TypeError`.
///
/// Read once: the check sits on the raise path, which is cold, but the flag is
/// also consulted from a `OnceLock` for the same reason as
/// [`interp_return_log_enabled`] — a `getenv` scans the environment array under
/// a lock.
#[cfg(not(feature = "sandbox"))]
#[majit_macros::elidable_cannot_raise]
fn reraise_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_RERAISE_DIAG").is_some())
}

#[cfg(feature = "sandbox")]
fn reraise_diag_enabled() -> bool {
    false
}

/// Describe a `RERAISE` operand that failed the `W_BaseException` check.
///
/// The operand is whatever `PUSH_EXC_INFO` pushed, and the compiler emits the
/// pair together, so a non-exception here means the frame's value-stack slot
/// was corrupted in between — never a program error. Report enough to tell the
/// two shapes apart: a NULL slot (a live slot that was cleared) versus a live
/// pointer whose object is no longer an exception (a stale or swept address).
fn reraise_bad_operand_diag(
    w_exc: PyObjectRef,
    oparg: u32,
    code_name: &str,
    last_instr: isize,
    opcode: &str,
    stack_base: usize,
    depth: usize,
    slots: &[(usize, PyObjectRef)],
) {
    // Addresses only. The operand that fails `is_exception` is often not a
    // PyObject, and reading `ob_type` there is the segfault this diag used to
    // hit on the yield-from `GeneratorExit` throw.
    let owned = !w_exc.is_null()
        && pyre_object::gc_hook::try_gc_owns_object(w_exc as pyre_object::gc_hook::GCREF);
    let tid = if owned {
        unsafe { majit_gc::header::header_of(w_exc as usize).read().type_id() }
    } else {
        u32::MAX
    };
    let stack = slots
        .iter()
        .map(|(index, value)| format!("{index}:{:#x}", *value as usize))
        .collect::<Vec<_>>()
        .join(", ");
    // Through the seam, not `eprintln!`: under sandbox the interpreter reaches
    // fd 2 only via `ops::write`, and the compile-out fence rejects a direct
    // `std::io::_eprint` here.
    crate::host_seam::emit_stderr(
        format!(
            "[reraise] code={code_name} oparg={oparg} lasti={last_instr} op={opcode} \
             base={stack_base} depth={depth} tos={:#x} gc_owned={owned} tid={tid} slots=[{stack}]\n",
            w_exc as usize,
        )
        .as_bytes(),
    );
}

pub fn capture_pyframe_root_area() -> *const () {
    PYFRAME_ROOT_AREA.with(|area| area as *const _ as *const ())
}

/// Whether `frame` is a collector-owned `PyFrame` rather than one of the
/// header-prefixed `std::alloc` frames used by interpreter snapshots.
///
/// PyPy's `PyFrame` is a GC object: the root callback exposes the frame and
/// the collector traces its fields later.  It never walks through a freshly
/// copied frame during `collect_roots_in_nursery`, when only direct roots have
/// been forwarded (`IncrementalMiniMarkGC.collect_roots_in_nursery`).  Pyre's
/// explicit frame walker is needed only for the non-GC fallback.  Reading the
/// header is safe for both representations because the fallback deliberately
/// carries a zeroed `GcHeader` too (`alloc_fixed_array_with_header`'s frame
/// counterpart in `pyframe.rs`).
#[inline]
unsafe fn is_gc_managed_pyframe(frame: *mut PyFrame) -> bool {
    if frame.is_null() {
        return false;
    }
    unsafe {
        (*majit_gc::header::header_of(frame as usize)).type_id()
            == crate::pyframe::PYFRAME_GC_TYPE_ID
    }
}

/// Continue only through the non-GC fallback frame chain.
///
/// A managed `PyFrame` or `JitVirtualRef` has already been exposed through the
/// slot visitor.  Its own type tracer owns the next edge; dereferencing it in
/// the root phase would run ahead of the collector's gray/remembered-set walk.
#[inline]
unsafe fn raw_chain_next_frame(f_backref: *mut PyFrame) -> *mut PyFrame {
    if f_backref.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        if (*majit_gc::header::header_of(f_backref as usize)).type_id() == 0 {
            f_backref
        } else {
            std::ptr::null_mut()
        }
    }
}

/// Forward the frame named by a `JitVirtualRef` stored in a frame-shaped slot.
///
/// Frame backrefs and `executioncontext.rs`'s `vref_referent` both allow such a slot to
/// hold a vref in place of a `*mut PyFrame`. Its leading word is the
/// `JIT_VIRTUAL_REF_VTABLE` magic, not a PyObject `ob_type`, so callers must
/// skip PyObject-shaped walks when this returns true. The vref is old-gen and
/// non-moving, but its `forced` frame may be young.
/// `virtualref.py is_virtual_ref(gcref)` supplies the predicate.
#[inline]
unsafe fn forward_virtual_ref_forced(
    value: *mut u8,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) -> bool {
    unsafe {
        if !majit_metainterp::virtualref::ptr_is_virtual_ref(value as *const u8) {
            return false;
        }
        let vref = value as *mut majit_metainterp::virtualref::JitVirtualRef;
        visitor(&mut *(&mut (*vref).forced as *mut *mut u8 as *mut majit_ir::GcRef));
        true
    }
}

/// Visit one `locals_cells_stack_w` slot as a GC root.
///
/// A slot holding a `JitVirtualRef` must skip the PyObject-shaped raw walks:
/// its leading word is the vtable magic rather than an `ob_type`.
unsafe fn walk_frame_value_slot(
    slot_ptr: *mut majit_ir::GcRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    // SAFETY: `slot_ptr` points to live `GcRef` storage whose allocation
    // outlives this walk. The visitor reads, conditionally forwards, and
    // stores back a `GcRef` (same layout as `*mut PyObject`).
    visitor(unsafe { &mut *slot_ptr });
    // A caught exception bound to a local (`except X as e`) is
    // `malloc_typed`-immortal, so the visitor above is a no-op
    // for it and its GC-managed children (`args_w`, `w_errno`,
    // …) are never traced. Forward them in place. Read the slot
    // AFTER the visitor so a relocated value is the live one.
    unsafe {
        let value = (*slot_ptr).0 as PyObjectRef;
        if forward_virtual_ref_forced(value as *mut u8, visitor) {
            return;
        }
        walk_raw_exception_roots(value, visitor);
        walk_raw_immortal_roots(value, visitor);
    }
}

/// How much of `locals_cells_stack_w` is reachable state.
///
/// `valuestackdepth` is an absolute index that starts at `stack_base()`
/// (`pyframe.rs`) and `pop` refuses to go below it, so for a running
/// frame it already covers the locals/cells prefix. `PyFrame::descr_clear`
/// is the one writer that breaks that: it rebinds every cell slot to a fresh
/// `w_cell_new` and then sets `valuestackdepth = 0`. Clamping to the raw
/// depth would walk zero slots for such a frame and drop cells that the
/// array still points at, so the prefix is the floor. The array is
/// `PY_NULL`-filled at allocation, so widening never reads uninitialised
/// slots.
unsafe fn walk_depth(f: &PyFrame, arr: &FixedObjectArray) -> usize {
    f.valuestackdepth.max(f.stack_base()).min(arr.len())
}

/// The module dicts one [`walk_pyframe_roots_area`] pass has already walked.
///
/// A frame chain names the same globals dict once per frame from that module,
/// and every frame on it names the same builtins dict, so the per-frame cell
/// walk ran over the same storage as many times as there were frames.  Upstream
/// reaches an object once per collection because `_collect_obj` sets VISITED
/// before pushing it (incminimark.py); these dicts carry no such
/// mark of their own — a Box-immortal one has no GC header at all — so the
/// pass carries the marks itself.
///
/// The window is fixed because the walk must not allocate: a frame chain names
/// a handful of distinct module dicts, and past the window every further one is
/// walked again, which is what every one of them got before.
struct WalkedModuleDicts {
    seen: [usize; Self::CAPACITY],
    len: usize,
}

impl WalkedModuleDicts {
    /// Above the number of loaded modules a frame chain can name, so the
    /// degradation past it stays theoretical: an interpreter with 184 modules
    /// imported still puts every distinct globals dict a stack can reach
    /// inside one 1 KB stack array.
    const CAPACITY: usize = 128;

    fn new() -> Self {
        Self {
            seen: [0; Self::CAPACITY],
            len: 0,
        }
    }

    /// Whether `w_dict` still needs walking on this pass, recording it if so.
    fn claim(&mut self, w_dict: PyObjectRef) -> bool {
        let addr = w_dict as usize;
        if addr == 0 || self.seen[..self.len].contains(&addr) {
            return false;
        }
        if self.len < Self::CAPACITY {
            self.seen[self.len] = addr;
            self.len += 1;
        }
        true
    }
}

/// Walk one captured thread's active frame and interpreter root state.
///
/// # Safety
/// `data` must come from [`capture_pyframe_root_area`], and the owning thread
/// must be quiesced.
pub unsafe fn walk_pyframe_roots_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let area = unsafe { &*(data as *const PyFrameRootArea) };
    // These are genuinely thread-local exception carriers, but every stopped
    // mutator's carrier must be visited during free-threaded collection.
    // PyPy reaches the equivalent values through each ExecutionContext /
    // thread state; keeping them in this registered per-mutator root area
    // preserves that ownership instead of resolving TLS on only the thread
    // which happened to initiate collection.
    unsafe {
        walk_in_flight_exception_area(area.in_flight_exception, visitor);
        walk_raw_exception_cell_area(area.bh_last_exception, visitor);
        walk_raw_exception_cell_area(area.guard_exception, visitor);
        walk_raw_exception_cell_area(area.jit_pending_exception, visitor);
        crate::call::walk_pending_call_error_area(area.pending_call_error, visitor);
        crate::baseobjspace::walk_pending_hash_error_area(area.pending_hash_error, visitor);
    }
    // incminimark.py:339-355 prebuilt-object scanning parity: a minor
    // collection scans an old/prebuilt object only when the write barrier
    // recorded a store into it since the previous minor collection
    // (`old_objects_pointing_to_young`); a major collection always traces
    // `prebuilt_root_objects`.  The Box-immortal structures walked below
    // (module dicts / cells, heap-type namespace dicts, method caches,
    // function fields) are pyre's prebuilt family; their mutation helpers
    // set `mark_prebuilt_roots_dirty`, so a clean bit during a minor
    // collection means no young pointer can be inside and the walks are
    // skipped.  Live-frame slots are real stack roots and are always walked.
    let is_minor = majit_gc::shadow_stack::extra_root_walk_kind()
        == majit_gc::shadow_stack::ExtraRootWalkKind::Minor;
    let scan_prebuilt = !is_minor
        || pyre_object::gc_roots::prebuilt_roots_dirty()
        || !gc_prebuilt_remember_enabled();
    let cf = unsafe { &*area.current_frame };
    // Kept across the whole frame loop: the dicts it names repeat, the walk
    // does not have to.  Not shared with `walk_import_roots_area` below, whose
    // `forward` visits strictly more per slot.
    let mut walked_module_dicts = WalkedModuleDicts::new();
    {
        // Forward `CURRENT_FRAME` itself: when the top frame is a
        // nursery-allocated `PyFrame`
        // (`emit_new_pyframe_inline_self_recursive`) the visitor copies
        // it to the survivor space and rewrites the cell to the new
        // address. For `std::alloc`-backed frames the visitor's
        // `is_nursery_object_start` guard short-circuits, leaving the
        // pointer untouched. `Cell::as_ptr()` exposes the storage
        // address; `*mut PyFrame` and `GcRef` share the `usize` repr
        // (`GcRef` is `#[repr(transparent)]`).
        //
        // SAFETY: `CURRENT_FRAME`'s storage is a thread-local `Cell`
        // that outlives this walker. We hold the with-borrow `cf` for
        // the duration of the visit so no other code mutates the cell.
        let cf_slot_ptr = cf.as_ptr() as *mut majit_ir::GcRef;
        visitor(unsafe { &mut *cf_slot_ptr });
        // Saved previous-frame / previous-ec-topframe roots now live on
        // `majit_gc::shadow_stack` (pushed by `push_current_frame_previous_root`)
        // and are forwarded by the GC's root walker; no extra visit here.

        let mut frame = cf.get();
        // Root the EC slots from the ambient TLS EC.  PyPy reaches the
        // ExecutionContext unconditionally through `space.threadlocals`,
        // independent of whether a frame is currently installed.
        let ambient_ec = unsafe {
            (&*(area.last_exec_ctx as *const Cell<*const PyExecutionContext>)).get()
                as *mut PyExecutionContext
        };
        let mut visit_ec_slots = |ec: *mut PyExecutionContext| {
            if ec.is_null() {
                return;
            }
            let top_slot = unsafe { &mut (*ec).topframeref as *mut *mut PyFrame };
            visitor(unsafe { &mut *(top_slot as *mut majit_ir::GcRef) });
            // `ExecutionContext.topframeref` is the executing-frame root
            // (`executioncontext.py enter`). A managed PyFrame is left to
            // `pyframe_object_custom_trace` after the gray/remembered-set
            // phase.
            unsafe { (*ec).walk_builtin_roots(visitor) };
            // `sys_exc_value` holds the active handler exception, which
            // is nursery-allocated and may move; forward it so the EC
            // slot is updated on a minor collection (the value-stack
            // copy alone is not authoritative for later EC reads).
            let exc_slot = unsafe { &mut (*ec).sys_exc_value as *mut PyObjectRef };
            visitor(unsafe { &mut *(exc_slot as *mut majit_ir::GcRef) });
            let async_exc_slot = unsafe { &mut (*ec).w_async_exception_type as *mut PyObjectRef };
            visitor(unsafe { &mut *(async_exc_slot as *mut majit_ir::GcRef) });
            // Exceptions may use the off-GC malloc_typed fallback.  In that
            // case forwarding the carrier above is a no-op, so trace its raw
            // GC-managed children just as `walk_in_flight_exception` does.
            // This is the post-PUSH_EXC_INFO owner of the same exception and
            // must preserve its traceback/frame graph identically.
            unsafe { walk_raw_exception_roots((*ec).sys_exc_value, visitor) };
            // `executioncontext.py:55` current_gen_or_coroutine is the head
            // of the running-generator chain.  The generator custom trace
            // follows each node's `previous_gen_or_coroutine` edge.
            let current_gen_slot =
                unsafe { &mut (*ec).current_gen_or_coroutine as *mut PyObjectRef };
            visitor(unsafe { &mut *(current_gen_slot as *mut majit_ir::GcRef) });
            let contextvar_slot = unsafe { &mut (*ec).contextvar_context as *mut PyObjectRef };
            visitor(unsafe { &mut *(contextvar_slot as *mut majit_ir::GcRef) });
            for hook in unsafe {
                [
                    &mut (*ec).w_asyncgen_firstiter_fn as *mut PyObjectRef,
                    &mut (*ec).w_asyncgen_finalizer_fn as *mut PyObjectRef,
                ]
            } {
                visitor(unsafe { &mut *(hook as *mut majit_ir::GcRef) });
            }
        };
        visit_ec_slots(ambient_ec);
        while !frame.is_null() {
            // A GC-managed PyFrame was exposed by the CURRENT_FRAME slot (or
            // by the preceding raw frame's f_backref slot).  Stop here and let
            // `pyframe_object_custom_trace` follow its fields after the
            // collector reaches the gray/remembered-set phase.
            if unsafe { is_gc_managed_pyframe(frame) } {
                break;
            }
            // SAFETY: PyFrame pointers on the f_backref chain are valid
            // for the duration of the enclosing `eval_with_jit` call. A
            // minor collection is always synchronous with respect to the
            // interpreter thread, so frames cannot be dropped mid-walk.
            //
            // The walk covers `[0, walk_depth)` — the locals/cells prefix
            // plus the live operand stack. Slots above it are popped
            // capacity, not roots. Non-ref slots are filtered by
            // `is_nursery_object_start` inside the collector, so a slot
            // holding a non-object word is harmless for the bump-pointer
            // nursery.
            //
            // Only a header-prefixed stdalloc frame reaches this branch.
            // Collector-owned frames stop above and are traced from the
            // survivor/old object by `pyframe_object_custom_trace`.
            let (arr_ptr, depth, next_frame) = unsafe {
                let f_back_slot = &mut (*(frame)).f_backref as *mut *mut PyFrame;
                visitor(&mut *(f_back_slot as *mut majit_ir::GcRef));
                // A direct raw-frame backref can continue the fallback chain.
                // A managed PyFrame or JitVirtualRef stops it: the collector's
                // type tracer follows that object's interior edges later.

                // pyframe.py:102 `self.pycode` — the running code object.
                // Visited as a root so a code object reachable only via
                // `frame.pycode` (e.g. `exec`'d code with no owning
                // Function) stays alive now that code objects are GC-managed.
                let pycode_slot = &mut (*(frame)).pycode as *mut *const ();
                visitor(&mut *(pycode_slot as *mut majit_ir::GcRef));
                // Forward the running code object's cached globals dict.  For
                // `exec`'d code with a movable (non-module) globals dict and no
                // owning Function, this frame is the only root that reaches it.
                walk_raw_code_roots((*(frame)).pycode as PyObjectRef, visitor);

                // PyFrame is normally a GC object in PyPy, so its GCREF
                // fields are traced before consumers dereference them.
                // pyre also has stdalloc-backed frames, so the frame root
                // walker must expose those fields explicitly.
                let locals_slot =
                    &mut (*(frame)).locals_cells_stack_w as *mut *mut pyre_object::FixedObjectArray;
                visitor(&mut *(locals_slot as *mut majit_ir::GcRef));
                // Strong edge to the `f_generator_wref` WEAKREF box
                // (`initialize_as_generator`), which must stay alive; the
                // collector invalidates its `weakptr` on its own.
                let generator_slot = &mut (*(frame)).f_generator_wref as *mut PyObjectRef;
                visitor(&mut *(generator_slot as *mut majit_ir::GcRef));
                let yielding_slot = &mut (*(frame)).w_yielding_from as *mut PyObjectRef;
                visitor(&mut *(yielding_slot as *mut majit_ir::GcRef));
                // pyframe.py:115-116 `self.builtin = ...` — the picked
                // builtin Module is a GC root.  Pyre stores it on
                // `frame.w_builtin` so `frame.get_builtin()` returns
                // the same object PyPy would; the LOAD_GLOBAL fallback
                // (`load_global_value` at eval.rs) reaches the
                // builtin's globals through `w_module_get_w_dict(self
                // .w_builtin)` — there is no separate storage-keyed
                // fast path field anymore.
                let w_builtin_slot = &mut (*(frame)).w_builtin as *mut PyObjectRef;
                visitor(&mut *(w_builtin_slot as *mut majit_ir::GcRef));
                let w_builtin = (*frame).w_builtin;
                if !w_builtin.is_null() && pyre_object::is_module(w_builtin) {
                    // Module is a Box-immortal carrier, but its module dict is
                    // now a non-moving GC object. Mark the header through the
                    // owning field so its custom trace can retain the values.
                    let w_dict_slot = &mut (*(w_builtin as *mut pyre_object::module::Module)).w_dict
                        as *mut PyObjectRef;
                    visitor(&mut *(w_dict_slot as *mut majit_ir::GcRef));
                }
                // pyframe.py `FrameDebugData.w_globals` / `w_locals`: only a
                // code/globals identity mismatch stores globals on the frame.
                // The common globals edge is reached through `pycode` above.
                // Both debug mappings must be forwarded before use.
                // `debugdata.w_locals` (the frame's locals
                // mapping object) and `w_f_trace` carry GCREFs that survive
                // the frame; forward both slots.  The locals mapping holds its
                // own bindings (module globals, class namespace, function
                // `locals()` dict, or an `exec` mapping), so forwarding the
                // object pointer keeps the whole namespace reachable.
                if !(*frame).debugdata.is_null() {
                    if pyre_object::gc_hook::try_gc_owns_object(
                        (*frame).debugdata as pyre_object::gc_hook::GCREF,
                    ) {
                        let debugdata_slot =
                            &mut (*frame).debugdata as *mut *mut crate::pyframe::FrameDebugData;
                        visitor(&mut *(debugdata_slot as *mut majit_ir::GcRef));
                    }
                    let d = &mut *(*frame).debugdata;
                    let w_globals_slot = &mut d.w_globals as *mut PyObjectRef;
                    visitor(&mut *(w_globals_slot as *mut majit_ir::GcRef));
                    let w_locals_slot = &mut d.w_locals as *mut PyObjectRef;
                    visitor(&mut *(w_locals_slot as *mut majit_ir::GcRef));
                    let w_extra_locals_slot = &mut d.w_extra_locals as *mut PyObjectRef;
                    visitor(&mut *(w_extra_locals_slot as *mut majit_ir::GcRef));
                    let w_f_trace_slot = &mut d.w_f_trace as *mut PyObjectRef;
                    visitor(&mut *(w_f_trace_slot as *mut majit_ir::GcRef));
                    let hidden_operationerr_slot = &mut d.hidden_operationerr as *mut PyObjectRef;
                    visitor(&mut *(hidden_operationerr_slot as *mut majit_ir::GcRef));
                }
                let live_obj = (*frame).get_w_globals();
                // For a W_ModuleDictObject the LOAD_GLOBAL read path consults the
                // authoritative `dstorage` cell map / `object_storage` /
                // strategy caches. Forward those movable
                // values here so a relocated global is not read back stale.
                // No-op for non-module dicts.  The picked builtin Module's
                // dict is consulted on a globals miss (`_load_global`
                // fallback), so forward it too.
                if scan_prebuilt {
                    let mut forward = |slot: &mut PyObjectRef| {
                        visitor(&mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef));
                        walk_raw_function_roots(*slot, visitor);
                    };
                    if walked_module_dicts.claim(live_obj) {
                        pyre_object::dictmultiobject::w_module_dict_walk_gc_cells(
                            live_obj,
                            &mut forward,
                        );
                    }
                    let w_builtin = (*frame).w_builtin;
                    if !w_builtin.is_null() && pyre_object::is_module(w_builtin) {
                        let w_builtin_dict = pyre_object::w_module_get_w_dict(w_builtin);
                        if walked_module_dicts.claim(w_builtin_dict) {
                            pyre_object::dictmultiobject::w_module_dict_walk_gc_cells(
                                w_builtin_dict,
                                &mut forward,
                            );
                        }
                    }
                }
                let f = &*frame;
                let next_frame = raw_chain_next_frame((*frame).f_backref);
                if f.locals_cells_stack_w.is_null() {
                    (std::ptr::null_mut::<PyObjectRef>(), 0, next_frame)
                } else {
                    let arr = &*f.locals_cells_stack_w;
                    (
                        arr.items_ptr() as *mut PyObjectRef,
                        walk_depth(f, arr),
                        next_frame,
                    )
                }
            };
            if !arr_ptr.is_null() && depth > 0 {
                for i in 0..depth {
                    let slot_ptr = unsafe { arr_ptr.add(i) } as *mut majit_ir::GcRef;
                    unsafe { walk_frame_value_slot(slot_ptr, visitor) };
                }
            }
            frame = next_frame;
        }
        // Box-immortal modules (and their Box-immortal dicts) are not
        // reachable transitively by the collector, so walk every loaded
        // module's dict storage as a pinned root source.  This covers
        // module-scope movable values bound in modules other than the
        // running frame's globals — e.g. `gc.collect` reached through
        // `gc.__dict__` on a fresh `LOAD_METHOD` after a collection.
        if scan_prebuilt {
            unsafe {
                let mut forward = |slot: &mut PyObjectRef| {
                    visitor(&mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef));
                    walk_raw_function_roots(*slot, visitor);
                    // getset descriptors are Box-immortal (custom trace never
                    // fires), so their collectable `fget`/`fset`/`fdel`
                    // functions must be marked reachable here or the getter
                    // dangles after a collection.
                    walk_raw_getset_roots(*slot, visitor);
                    walk_raw_wrapped_function_roots(*slot, visitor);
                };
                crate::importing::walk_import_roots_area(area.import_roots, &mut forward);
                // The `_mapdict_caches` LOAD_METHOD `w_method` slots
                // (mapdict.py:1418) and the stamped `w_globals` slots
                // (`pycode.py frame_stores_global`) used to ride here
                // as per-thread areas.  Their holder is an immortal code
                // object that outlives the stamping thread, so they moved to
                // process-global walkers registered in `pyre-jit::eval`
                // (`mapdict_method_cache_root_walker`,
                // `w_globals_stamped_code_root_walker`).
            }
        }
    }
}

/// The interpreter's process-global off-GC slots, as one root source.
///
/// Frame roots are not here: `PyFrame.locals_cells_stack_w` and the
/// thread-local exception carriers ride the per-mutator `PyFrameRootArea`
/// (`walk_pyframe_root_area`), which reaches every registered thread rather
/// than only the collecting one. What is left is genuinely process-global —
/// app-level interphook handles, the `threading` module's own roots, and the
/// faulthandler's — so it registers once for the process.
fn walk_interpreter_global_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    pyre_object::unicodeobject::walk_interned_strings_gc(&mut |slot| {
        visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
    });
    walk_global_prebuilt_roots(visitor);
    #[cfg(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    ))]
    {
        // The cast reinterprets a mirror's link slot as a GC reference in
        // place, so the two spellings of a machine word have to stay
        // interchangeable for the forwarding write to land on the slot.
        const _: () = assert!(
            std::mem::size_of::<PyObjectRef>() == std::mem::size_of::<majit_ir::GcRef>()
                && std::mem::align_of::<PyObjectRef>() == std::mem::align_of::<majit_ir::GcRef>()
        );
        let mut forward = |slot: &mut PyObjectRef| {
            visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
        };
        crate::cpyext::walk_gc_roots(&mut forward);
    }
    crate::executioncontext::walk_space_user_del_action_roots(visitor);
    crate::module::signal::interp_signal::walk_check_signal_action_roots(visitor);
    crate::module::sys::vm::walk_monitoring_tool_roots(visitor);
    crate::module::thread::walk_thread_roots(visitor);
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    crate::module::faulthandler::handler::walk_faulthandler_roots(visitor);
    if let Some(hooks) = crate::importing::optional_module_hooks() {
        (hooks.walk_global_roots)(visitor);
    }
}

/// Install the interpreter's process-global GC root walker with the majit-gc
/// collector.
///
/// Called once at process startup from the JIT driver / pyrex main.
/// Stored in a process-global fn-pointer cell (#396); calling again with
/// the same fn pointer is idempotent.
pub fn register_interpreter_global_root_walker() {
    majit_gc::shadow_stack::register_extra_root_walker(
        walk_interpreter_global_roots,
        "interpreter_global_roots",
    );
}

thread_local! {
    /// The exception currently being raised / propagated up the Rust call
    /// stack. Between the raising frame's `record_application_traceback` and
    /// the frame that finally catches it, the exception is held only in the
    /// Rust `PyError` value in flight — not on any frame's value stack or in
    /// `ec.sys_exc_value` yet. The instance is a nursery object
    /// (`alloc_exception_nursery`), so a minor collection moves it and a
    /// dispatch-loop safepoint would drop an unrooted traceback chain.
    /// Mirrors `tstate->current_exception`: keep the in-flight
    /// exception's traceback chain a GC root until the exception is caught.
    static IN_FLIGHT_EXCEPTION: Cell<PyObjectRef> = const { Cell::new(pyre_object::PY_NULL) };
}

/// Publish the exception now being raised / propagated so the GC root walker
/// keeps it (and its traceback chain) alive across a collection. Called from
/// `record_application_traceback`, the single chokepoint every raising frame
/// passes through.
///
/// Writes the runtime-mutable `IN_FLIGHT_EXCEPTION` thread-local, not a
/// build-time constant, so the JIT residualizes the call instead of tracing
/// into it (`@dont_look_inside`, the `gc_interp::at_outermost_activation`
/// shape). One single-word argument, `()` result, and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn set_in_flight_exception(exc: PyObjectRef) {
    IN_FLIGHT_EXCEPTION.with(|c| c.set(exc));
}

#[allow(dead_code)]
fn walk_in_flight_exception(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    IN_FLIGHT_EXCEPTION.with(|c| {
        unsafe { walk_in_flight_exception_area(c as *const _, visitor) };
    });
}

unsafe fn walk_in_flight_exception_area(
    c: *const Cell<PyObjectRef>,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let c = unsafe { &*c };
    let exc = c.get();
    if exc.is_null() {
        return;
    }
    // Forward the exception OBJECT slot itself. A nursery exception
    // (`w_exception_new_empty` → `alloc_exception_nursery`) moves; the
    // collector rewrites this cell, then recurses into its slots.
    let slot = c.as_ptr();
    unsafe { visitor(&mut *(slot as *mut majit_ir::GcRef)) };
    // Off-GC fallback exceptions are malloc_typed, so explicitly trace their
    // raw GC-managed children after forwarding the carrier itself.
    let exc = unsafe { *(slot as *const PyObjectRef) };
    unsafe { walk_raw_exception_roots(exc, visitor) };
}

/// Forward one raw `i64` exception carrier cell and trace the exception's
/// GC-managed children. Shared by every such carrier
/// (`BH_LAST_EXC_VALUE`, `GUARD_EXC_VALUE`, `TL_JIT_PENDING_EXCEPTION`) so they
/// cannot drift.
pub(crate) unsafe fn walk_raw_exception_cell_area(
    c: *const Cell<i64>,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let c = unsafe { &*c };
    if c.get() == 0 {
        return;
    }
    let slot = c.as_ptr();
    unsafe { visitor(&mut *(slot as *mut majit_ir::GcRef)) };
    let exc = c.get() as PyObjectRef;
    unsafe { walk_raw_exception_roots(exc, visitor) };
}

fn walk_global_prebuilt_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    // `reduce_protocol`'s app-level interphook handles are process-global
    // off-GC slots.  RPython stores them in the space's ordinary object graph;
    // expose the equivalent slots on every collection so both minor moves and
    // major marking preserve their functions and private globals namespace.
    crate::reduce_protocol::walk_handle_roots(visitor);
    // The aiter/anext app-level handles are the same off-GC-slot case.
    crate::async_operation::walk_handle_roots(visitor);
    crate::app_functional::walk_handle_roots(visitor);
    // These `space.fromcache` tables are off-GC slots; forward them on every
    // collection so a minor move updates the cached mapping pointers. They sit
    // with the ungated handle roots (not the gated prebuilt block) because each
    // is published lazily without `mark_prebuilt_roots_dirty`, so its possibly
    // young dicts must be forwarded on the first collection regardless of the
    // prebuilt-remember bit. `_compat_pickle`'s fix_imports tables reach this
    // block through `walk_prebuilt_slots` below.
    {
        let mut fwd = |slot: &mut PyObjectRef| {
            visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
        };
        // `space.fromcache(AuditHolder)`'s hooks are installed by running app
        // code, so a hook callable is young when it lands and cannot wait for
        // the prebuilt-roots scan below.
        crate::module::sys::vm::walk_audit_hooks_gc(&mut fwd);
        // `space.fromcache(CodecState)` publishes a young list and two young
        // dicts on first use and marks no prebuilt-roots dirty bit, so it
        // belongs with these two rather than behind the gate below.
        crate::module::_codecs::walk_codec_state_gc(&mut fwd);
        // `ast.get(space)`'s node types are the same shape: published on the
        // first `_ast` import without a dirty bit, and the only owner of the
        // classes between one `_ast` module dict and the next.
        crate::module::_ast::moduledef::walk_ast_state_gc(&mut fwd);
        // The exception accessor table is built on the first exception
        // class, and an accessor whose class has not installed it yet is
        // owned by nothing else.
        crate::builtins::walk_exception_getset_roots(&mut fwd);
        if let Some(hooks) = crate::importing::optional_module_hooks() {
            (hooks.walk_prebuilt_slots)(&mut fwd);
        }
    }
    let is_minor = majit_gc::shadow_stack::extra_root_walk_kind()
        == majit_gc::shadow_stack::ExtraRootWalkKind::Minor;
    let scan_prebuilt = !is_minor
        || pyre_object::gc_roots::prebuilt_roots_dirty()
        || !gc_prebuilt_remember_enabled();
    // incminimark.py collect_roots_in_nursery walks static pointers in
    // prebuilt non-gc structures on every minor. TypeDef.rawdict slots
    // are that family; visit_prebuilt_declaration also walks the raw
    // interiors of a malloc_typed value, covering interp2app._code.
    unsafe {
        let mut forward_declaration = |slot: &mut PyObjectRef| {
            visit_prebuilt_declaration(slot, visitor, true);
        };
        pyre_object::typedef::walk_typedef_roots(&mut forward_declaration);
    }
    if !scan_prebuilt {
        return;
    }
    unsafe {
        {
            let mut forward_declaration = |slot: &mut PyObjectRef| {
                visit_prebuilt_declaration(slot, visitor, true);
            };
            // Native Cache.content and TypeDef.rawdict have the same strong
            // GC reachability as their host dictionaries in PyPy.
            crate::baseobjspace::walk_object_space_cache_roots(&mut forward_declaration);
        }
        // `typeobject.py MethodCache` is one GC object. A fill write-barriers
        // that object; this walk names it once. Its trace visits the slots.
        {
            let mut forward_cache = |slot: &mut PyObjectRef| {
                visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
            };
            crate::baseobjspace::walk_method_cache_root(&mut forward_cache);
        }
        let mut forward = |slot: &mut PyObjectRef| {
            visit_prebuilt_declaration(slot, visitor, false);
        };
        walk_builtin_type_dicts_gc(&mut forward);
        // interp_posix.ApplevelForkCallbacks is another object-space cache.
        #[cfg(not(target_arch = "wasm32"))]
        crate::module::posix::interp_posix::walk_fork_callback_roots(&mut forward);
        // `space.sys.modules`, `w_default_importlib_import`, and
        // `space.builtin_modules` are process-owned prebuilt slots
        // (`SysModuleState` / `space`). Mutator stores into them take
        // `mark_prebuilt_roots_dirty`
        // (`incminimark.py remember_young_pointer_from_prebuilt`); this
        // gated walk is the `prebuilt_root_objects` scan. Interior
        // immortal-module cell words live here too, so a major's
        // end-of-marking rescan still sees modules installed after the
        // initial root snapshot.
        crate::importing::walk_process_import_roots(&mut forward);
    }
    if is_minor {
        pyre_object::gc_roots::clear_prebuilt_roots_dirty();
    }
}

/// Forward the GC slots a SUSPENDED generator's frame owns.
///
/// A suspended generator's frame is off the active `CURRENT_FRAME` /
/// `f_backref` chain that [`walk_pyframe_roots`] traverses, so its
/// locals/cells/valuestack and the generator's own slots are never
/// reached during root scanning.  The generator object's custom trace
/// (`pyre-jit` `generator_object_custom_trace`) calls this while marking
/// so the suspended frame's live references survive a collection.
///
/// Only the slots unique to the suspended frame are forwarded here.
/// The globals/builtin dict VALUES are not walked: a module dict is
/// rooted globally by `walk_module_dicts_gc`, and a GC-managed `exec`
/// globals dict is reached transitively once its (forwarded) object
/// pointer is marked — its own trace walks the values.  This deliberately
/// avoids the globals-proxy / module-dict-cell walk that
/// [`walk_pyframe_roots`] performs during root scanning, keeping the
/// marking-phase visit to plain slot forwarding.
pub fn walk_suspended_generator_frame(
    frame: *mut PyFrame,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    if frame.is_null() {
        return;
    }
    unsafe {
        let pycode_slot = &mut (*frame).pycode as *mut *const ();
        visitor(&mut *(pycode_slot as *mut majit_ir::GcRef));

        // The locals/cells/valuestack array pointer, then each element
        // slot — walked exactly as the per-frame body of
        // `walk_pyframe_roots` (the array pointer plus the full
        // fixed-length payload).
        let locals_slot =
            &mut (*frame).locals_cells_stack_w as *mut *mut pyre_object::FixedObjectArray;
        visitor(&mut *(locals_slot as *mut majit_ir::GcRef));
        if !(*frame).locals_cells_stack_w.is_null() {
            let arr = &*(*frame).locals_cells_stack_w;
            let base = arr.items_ptr() as *mut PyObjectRef;
            let len = walk_depth(&*frame, arr);
            for i in 0..len {
                visitor(&mut *(base.add(i) as *mut majit_ir::GcRef));
            }
        }

        // Same strong edge as `walk_pyframe_roots`: the `f_generator_wref`
        // WEAKREF box.  Its `weakptr` is weak, so the frame does not keep
        // its generator alive; the collector invalidates the `weakptr`.
        let generator_slot = &mut (*frame).f_generator_wref as *mut PyObjectRef;
        visitor(&mut *(generator_slot as *mut majit_ir::GcRef));
        let yielding_slot = &mut (*frame).w_yielding_from as *mut PyObjectRef;
        visitor(&mut *(yielding_slot as *mut majit_ir::GcRef));

        // Forward the builtin object pointer. The common globals object is
        // reached through `pycode`; the rare override is in debugdata below.
        let w_builtin_slot = &mut (*frame).w_builtin as *mut PyObjectRef;
        visitor(&mut *(w_builtin_slot as *mut majit_ir::GcRef));
        let w_builtin = (*frame).w_builtin;
        if !w_builtin.is_null() && pyre_object::is_module(w_builtin) {
            let w_dict_slot =
                &mut (*(w_builtin as *mut pyre_object::module::Module)).w_dict as *mut PyObjectRef;
            visitor(&mut *(w_dict_slot as *mut majit_ir::GcRef));
        }

        if !(*frame).debugdata.is_null() {
            if pyre_object::gc_hook::try_gc_owns_object(
                (*frame).debugdata as pyre_object::gc_hook::GCREF,
            ) {
                let debugdata_slot =
                    &mut (*frame).debugdata as *mut *mut crate::pyframe::FrameDebugData;
                visitor(&mut *(debugdata_slot as *mut majit_ir::GcRef));
            }
            let d = &mut *(*frame).debugdata;
            let w_globals_slot = &mut d.w_globals as *mut PyObjectRef;
            visitor(&mut *(w_globals_slot as *mut majit_ir::GcRef));
            let w_locals_slot = &mut d.w_locals as *mut PyObjectRef;
            visitor(&mut *(w_locals_slot as *mut majit_ir::GcRef));
            let w_extra_locals_slot = &mut d.w_extra_locals as *mut PyObjectRef;
            visitor(&mut *(w_extra_locals_slot as *mut majit_ir::GcRef));
            let w_f_trace_slot = &mut d.w_f_trace as *mut PyObjectRef;
            visitor(&mut *(w_f_trace_slot as *mut majit_ir::GcRef));
            let hidden_operationerr_slot = &mut d.hidden_operationerr as *mut PyObjectRef;
            visitor(&mut *(hidden_operationerr_slot as *mut majit_ir::GcRef));
        }
    }
}

/// Flat TLS read of the per-thread `CURRENT_EXCEPTION` slot.
///
/// `dont_look_inside` keeps the codewriter from following into the
/// `LocalKey::with` closure (no extractable graph); calls classify
/// `Residual` against the fnaddr registered in `jit_trace_fnaddrs()`,
/// mirroring the trace-side `get_current_exception_fn` cpu helper
/// binding (`codewriter.rs PlainCannotRaiseNoHeap`).
#[majit_macros::dont_look_inside]
pub fn get_current_exception() -> PyObjectRef {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return PY_NULL;
    }
    unsafe { (*ec).sys_exc_value }
}

/// `executioncontext.py sys_exc_info` — return the topmost handled
/// exception, including one parked on the running generator/coroutine chain.
/// The bytecode PUSH_EXC_INFO machinery deliberately continues to use
/// [`get_current_exception`] for the direct EC slot; app-level `sys.exception`
/// and bare raise use this logical view instead.
///
/// No `dont_look_inside`: the thread-local read this reaches is already
/// isolated behind [`crate::call::take_last_exec_ctx`]'s own residual, so the
/// rest of the body — the `sys_exc_value` slot read and the generator-chain
/// walk — is an ordinary graph the write analyzer can see.  That visibility is
/// load-bearing, not cosmetic: a graph-less callee contributes an EMPTY
/// read/write set (`collect_readwrite_effects`), and an empty set lets the
/// heap optimizer dead-store-eliminate the balanced `sys_exc_value`
/// save/restore this read sits between, which returns the pre-handler slot
/// instead of the handled exception.
pub fn get_sys_exception() -> PyObjectRef {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return PY_NULL;
    }
    unsafe { (*ec).sys_exc_info() }
}

/// Flat TLS write of the per-thread `CURRENT_EXCEPTION` slot — same
/// residual-leaf contract as [`get_current_exception`].
#[majit_macros::dont_look_inside]
pub fn set_current_exception(exc: PyObjectRef) {
    let ec = crate::call::getexecutioncontext() as *mut PyExecutionContext;
    if ec.is_null() {
        return;
    }
    unsafe {
        (*ec).sys_exc_value = exc;
    }
}

/// The value `PUSH_EXC_INFO` saves below the caught exception: the exception
/// being handled, or `space.w_None` when there is none.
pub fn current_exception_or_none() -> PyObjectRef {
    let prev = get_current_exception();
    if prev.is_null() {
        pyre_object::w_none()
    } else {
        prev
    }
}

/// `PyFrame._restore_exc_info`: reinstate the value `PUSH_EXC_INFO` saved,
/// as `ExecutionContext.set_sys_exc_info3` does — `None` means no exception
/// was being handled.  A null slot means the same.
pub fn restore_exc_info(w_prev: PyObjectRef) {
    let exc = if !w_prev.is_null() && unsafe { pyre_object::is_none(w_prev) } {
        PY_NULL
    } else {
        w_prev
    };
    set_current_exception(exc);
}

/// `pyopcode.py:764-766` — `raise Class` instantiates the class, and
/// `normalize_exception` then validates the result.  `space.call_function`
/// propagates the constructor's own error in RPython; pyre's returns
/// `PY_NULL` with the error parked in the pending-call slot, so an unchecked
/// null would both lose that error and report the raise as
/// "exceptions must derive from BaseException".
///
/// # Safety
/// `w_type` must be a live exception class (`exception_is_valid_obj_as_class_w`).
unsafe fn instantiate_raised_class(mut w_type: PyObjectRef) -> Result<PyObjectRef, PyError> {
    // `pyopcode.py RAISE_VARARGS` keeps the exception class live across
    // `space.call_function` so the wrong-result diagnostic can still name it.
    let result = pyre_object::with_roots!(w_type => unsafe {
        crate::call_function(w_type, &[])
    });
    if result.is_null() {
        return Err(crate::call::take_call_error()
            .unwrap_or_else(|| PyError::type_error("exceptions must derive from BaseException")));
    }
    if !unsafe { pyre_object::is_exception(result) } {
        return Err(crate::error::exception_from_call_type_error(w_type, result));
    }
    Ok(result)
}

/// The `from` cause of a `raise X from Y`, after the class call, together with
/// the wording owed if it turns out not to name an exception.
pub struct RaiseCause {
    /// The object `__cause__` receives.  Null when there is no cause.
    pub w_cause: PyObjectRef,
    /// `Some` when `w_cause` came out of a constructor that did not answer an
    /// exception instance, holding the TypeError that names both the class and
    /// what it answered.
    ///
    /// It is formatted here rather than where it is raised because only this
    /// side still holds the class: the frame slot the cause came out of is
    /// cleared by the `pop`, and keeping the class and the answer alive across
    /// the raised value's own constructor would owe them both a root.  The
    /// formatting itself is on the error path only.
    invalid: Option<PyError>,
}

/// Instantiate the `from` cause of a `raise X from Y` when it is an exception
/// class, and answer it unchanged otherwise.
///
/// The verdict is recorded but deliberately not raised here.
/// `pyopcode.py` instantiates and does nothing else; the check is
/// `error.py set_cause`, which runs after the raised value has been
/// popped and normalized.  Raising early lets `raise 5 from Cls` answer the
/// cause's TypeError where every implementation answers the raised value's
/// "exceptions must derive from BaseException".  [`attach_raise_cause`] is
/// where it is raised.
///
/// # TODO: inline back into RAISE_VARARGS
///
/// **Deviation.** RPython performs this inline inside `RAISE_VARARGS`
/// (`pypy/interpreter/pyopcode.py`, `space.call_function(w_cause)`
/// when `w_cause` is an exception class) without a named helper. Pyre
/// extracts the step into a standalone helper so the JIT raise BH path
/// (`pyre-jit/src/call_jit.rs`), the tracer (`pyre-jit-trace`) and the
/// interpreter raise path share one instantiation.
///
/// **When to fix.** When `bh_normalize_raise_varargs_with_frame` is removed or
/// rewritten — e.g. when the JIT BH path can dispatch the same inlined
/// `RAISE_VARARGS` sequence directly without a shared helper.
///
/// **How to fix.** Inline this body back into the `RAISE_VARARGS`
/// dispatch arm in `pyre-interpreter/src/pyopcode.rs` (mirroring
/// `pyopcode.py`), delete this standalone fn, and either route
/// the BH path through the inlined sequence or rewrite it to match
/// RPython's inline shape.
pub fn normalize_raise_cause(mut cause: PyObjectRef) -> Result<RaiseCause, PyError> {
    if !unsafe { crate::baseobjspace::exception_is_valid_obj_as_class_w(cause) } {
        return Ok(RaiseCause {
            w_cause: cause,
            invalid: None,
        });
    }
    // `space.call_function(w_cause)` propagates the constructor's own error;
    // pyre's answers `PY_NULL` with the error parked in the pending-call slot,
    // so an unchecked null would lose it and report the raise as having no
    // cause at all.
    // Same `RAISE_VARARGS` livevar as upstream's `w_cause`: the result-type
    // error below still reads the class after its constructor returned.
    let result = pyre_object::with_roots!(cause => unsafe {
        crate::call_function(cause, &[])
    });
    if result.is_null() {
        return Err(crate::call::take_call_error().unwrap_or_else(|| {
            PyError::type_error("exception causes must derive from BaseException")
        }));
    }
    // A constructor that answers something other than an exception instance
    // is wrong whatever it answered: `None` is not the `from None` spelling,
    // which never reaches this branch because `None` is not a class.
    let invalid = (!unsafe { pyre_object::is_exception(result) })
        .then(|| crate::error::exception_from_call_type_error(cause, result));
    Ok(RaiseCause {
        w_cause: result,
        invalid,
    })
}

pub fn attach_raise_cause(exc: PyObjectRef, cause: Option<RaiseCause>) -> Result<(), PyError> {
    // `pypy/interpreter/pyopcode.py do_raise` /
    // `pypy/interpreter/executioncontext.py:325 _normalize_exception` —
    // when a `raise` runs while another exception is being handled,
    // chain the in-flight one as the new `__context__` so tracebacks
    // can show "During handling of the above exception, another
    // exception occurred:". Skip self-context to avoid the obvious
    // cycle (re-raising the same exception object).  Both
    // `__context__` and `__cause__`/`__suppress_context__` writes land
    // in the typed slots on `W_BaseException` per
    // `interp_exceptions.py`.
    // `ExecutionContext.sys_exc_info()` is the logical handled-exception
    // view: when a generator is resumed from inside its caller's `except`,
    // `push_gen_or_coroutine` parks that caller exception on the generator
    // chain.  It must still become the context of a new exception raised by
    // the generator. `get_sys_exception` is residual, so this does not expose
    // the virtualizable frame to the tracer.
    crate::error::chain_context(exc, get_sys_exception());
    if let Some(RaiseCause { w_cause, invalid }) = cause
        && !w_cause.is_null()
    {
        // `error.py set_cause` checks the cause here, after the raised
        // value has been normalized: `raise Cls from 42` therefore runs `Cls()`
        // first, and a raised value that is not an exception answers its own
        // TypeError rather than this one.
        //
        // Which check it is depends on how the cause got here.  A cause that
        // was written as a class has already been called, and only an exception
        // instance will do; a cause written as an object is judged by
        // `_exception_getclass(space, w_cause, "exception causes")`, which also
        // admits `None` and whose wording is not the one `descr_setcause` uses
        // for `e.__cause__ = x`.
        if let Some(err) = invalid {
            return Err(err);
        }
        let valid = unsafe { pyre_object::is_none(w_cause) || pyre_object::is_exception(w_cause) };
        if !valid {
            return Err(PyError::type_error(
                "exception causes must derive from BaseException",
            ));
        }
        if unsafe { pyre_object::is_exception(exc) } {
            // `interp_exceptions.py descr_setcause` — writes
            // `w_cause` and flips `suppress_context` to True.
            unsafe {
                pyre_object::interp_exceptions::w_exception_set_cause(exc, w_cause);
                pyre_object::interp_exceptions::w_exception_set_suppress_context(exc, true);
            };
        }
    }
    Ok(())
}

/// pyopcode.py `cmp_exc_match(self, w_1, w_2)` line-by-line:
///
/// ```python
/// def cmp_exc_match(self, w_1, w_2):
///     space = self.space
///     if space.isinstance_w(w_2, space.w_tuple):
///         for w_type in space.fixedview(w_2):
///             if not space.exception_is_valid_class_w(w_type):
///                 raise oefmt(space.w_TypeError, CANNOT_CATCH_MSG)
///     elif not space.exception_is_valid_class_w(w_2):
///         raise oefmt(space.w_TypeError, CANNOT_CATCH_MSG)
///     return space.exception_match(space.type(w_1), w_2)
/// ```
///
/// `w_1` is `exc_value` (the exception instance, peeked from TOS at
/// pyopcode.py), `w_2` is `exc_type` (the type spec, popped at
/// :1851). `space.type(w_1)` is the exception's class.
///
/// pyopcode.py `CANNOT_CATCH_MSG`.
pub const CANNOT_CATCH_MSG: &str =
    "catching classes that do not inherit from BaseException is not allowed";

/// pyopcode.py — the class-validity gate of `cmp_exc_match`,
/// split out from `check_exc_match_against` so the bool-returning hot
/// helper keeps a 1-register C ABI suitable for residual JIT calls.
/// PyPy's `@jit.unroll_safe` `cmp_exc_match` inlines into the trace and
/// its `raise oefmt(...)` becomes a guard; pyre matches the structure
/// by keeping the raise on the caller side (the BC handler), which
/// likewise runs outside the JIT-traced bool-returning fast path.
/// `pyopcode.py cmp_exc_match` is `@jit.unroll_safe` because the
/// tuple-of-types loop is short and must unroll into the except-match
/// trace.
#[majit_macros::unroll_safe]
pub fn validate_check_exc_match_class(exc_type: PyObjectRef) -> Result<(), PyError> {
    unsafe {
        if pyre_object::is_tuple(exc_type) {
            let n = pyre_object::w_tuple_len(exc_type) as i64;
            for i in 0..n {
                if let Some(w_type) = pyre_object::w_tuple_getitem(exc_type, i)
                    && !crate::baseobjspace::exception_is_valid_class_w(w_type)
                {
                    return Err(PyError::type_error(CANNOT_CATCH_MSG));
                }
            }
        } else if !crate::baseobjspace::exception_is_valid_class_w(exc_type) {
            return Err(PyError::type_error(CANNOT_CATCH_MSG));
        }
    }
    Ok(())
}

/// `pyopcode.py check_except_star_type_valid` is `@jit.unroll_safe`.
#[majit_macros::unroll_safe]
fn validate_check_eg_match_class(mut exc_type: PyObjectRef) -> Result<(), PyError> {
    validate_check_exc_match_class(exc_type)?;
    let mut base_group = crate::builtins::lookup_exc_class("BaseExceptionGroup").unwrap();
    unsafe {
        if pyre_object::is_tuple(exc_type) {
            let n = pyre_object::w_tuple_len(exc_type) as i64;
            for i in 0..n {
                if let Some(w_type) = pyre_object::w_tuple_getitem(exc_type, i)
                    && pyre_object::with_roots!(base_group, exc_type =>
                        crate::baseobjspace::issubclass(w_type, base_group)
                    )?
                {
                    return Err(PyError::type_error(
                        "catching ExceptionGroup with except* is not allowed. Use except instead.",
                    ));
                }
            }
        } else if crate::baseobjspace::issubclass(exc_type, base_group)? {
            return Err(PyError::type_error(
                "catching ExceptionGroup with except* is not allowed. Use except instead.",
            ));
        }
    }
    Ok(())
}

pub fn check_exc_match_against(exc_value: PyObjectRef, exc_type: PyObjectRef) -> bool {
    // pyopcode.py `return space.exception_match(space.type(w_1), w_2)`.
    // `crate::typedef::r#type` is the `space.type` equivalent — it
    // resolves `w_class` for objects whose specific class was already
    // installed (post-`init_typeobjects`) AND for exception instances
    // whose `w_class` slot still holds the generic `EXCEPTION_TYPE`
    // stub (pre-registry-init internal `w_exception_new` callers, e.g.
    // `PyError::value_error`) by falling back to the `ExcKind`-tag
    // registry (`lookup_exc_class_for_kind`).
    //
    // The validity gate (pyopcode.py:1034-1039) lives in
    // `validate_check_exc_match_class` and is invoked by the BC handler
    // BEFORE this helper, mirroring PyPy's `@jit.unroll_safe` inlining
    // where `raise oefmt(...)` becomes a guard outside the bool-returning
    // residual call.  The 1-register `bool` ABI is preserved for
    // cranelift / dynasm residual-call codegen.
    let Some(w_exc_class) = crate::typedef::r#type(exc_value) else {
        return false;
    };
    crate::baseobjspace::exception_match(w_exc_class.as_ptr(), exc_type)
}

/// Try to dispatch an exception using the exception table or block stack.
///
/// Returns `true` if a handler was found (resume PC updated to handler),
/// `false` if the exception should propagate to the caller.
///
/// `err` is taken by `&mut` so the bytecode_trace_after_exception /
/// exception_trace plumbing can replace it with a tracer exception
/// (pyopcode.py `except OperationError as e: operr = e`); the
/// caller's `Err(err)` propagation then surfaces the replacement.
pub fn handle_exception(frame: &mut PyFrame, err: &mut PyError, next_instr: &mut usize) -> bool {
    handle_exception_with_context(frame, err, next_instr, ContextSource::GeneratorChain)
}

/// Where the implicit `__context__` of `err` comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ContextSource {
    /// `executioncontext.py sys_exc_info` — the logical handled
    /// exception, walking to the generator that parked one. An exception the
    /// frame itself produces is raised while the caller's handler is live, so
    /// it takes that handler's exception.
    GeneratorChain,
    /// The flat EC slot alone, for an exception *thrown into* a resumed
    /// generator. `push_gen_or_coroutine` has already swapped the generator's
    /// own handled exception into the slot; the caller's belongs to the caller,
    /// which merely delivered this exception rather than raising it under a
    /// live handler. A generator holding none of its own therefore keeps a null
    /// context.
    ResumedFrameOnly,
}

/// [`handle_exception`] with an explicit context source
/// (`pyframe.py:303-306` records the context of a thrown-in
/// `SApplicationException` before the handler search).
#[inline(never)]
#[majit_macros::dont_look_inside]
pub fn handle_exception_with_context(
    frame: &mut PyFrame,
    err: &mut PyError,
    next_instr: &mut usize,
    context_source: ContextSource,
) -> bool {
    // An internal corruption marker is not a real Python exception and must
    // never be dispatched via bytecode handlers.
    if err.kind == crate::PyErrorKind::BytecodeCorruption {
        return false;
    }
    // pyopcode.py:135-148 — exception trace plumbing:
    //   try:
    //       trace = self.get_w_f_trace()
    //       if trace is not None:
    //           self.getorcreatedebug().w_f_trace = None
    //       try:
    //           ec.bytecode_trace_after_exception(self)
    //       finally:
    //           if trace is not None:
    //               self.getorcreatedebug().w_f_trace = trace
    //   except OperationError as e:
    //       operr = e
    //   pytraceback.record_application_traceback(
    //       self.space, operr, self, self.last_instr)
    //   ec.exception_trace(self, operr)
    //
    // bytecode_trace_after_exception + exception_trace are gated on a
    // live tracefunc so the no-tracer hot path skips the f_trace
    // save/restore dance.  record_application_traceback runs
    // unconditionally per `:147-148`, so the traceback chain grows on
    // every exception regardless of trace state.
    // bytecode_trace_after_exception's exception is caught by the
    // surrounding `except OperationError` and replaces operr;
    // exception_trace's exception is NOT caught (line 148 stands
    // outside the except), so it short-circuits the unrollstack search
    // — pyre signals that by returning `false` after replacing `err`.
    // `pyopcode.py handle_operation_error(attach_tb=True)` —
    // the entire `if attach_tb:` block (bytecode_trace_after_exception,
    // record_application_traceback, exception_trace) is gated on
    // `attach_tb`.  RERAISE opcode raises `RaiseWithExplicitTraceback`
    // which routes through the `attach_tb=False` branch, so all three
    // tracing hooks are skipped per `:91-94`.  Pyre carries the same
    // intent via `PyError.attach_tb` set by `eval.rs::reraise`.
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    // Everything below allocates before it touches the frame again:
    // `to_exc_object` materialises the exception, `chain_context` builds the
    // `__context__` link, the trace hooks run arbitrary Python and
    // `record_application_traceback` allocates a `PyTraceback`.  A frame the
    // JIT built is a nursery object (`emit_new_pyframe_inline_with_params`),
    // so `frame` names the abandoned copy after any of them collect.  Anchor it
    // once and re-read at each point the frame is next used.
    let frame_anchor = FrameAnchor::new(frame);
    let exc_obj = err.to_exc_object();
    if err.exc_object.is_null() {
        err.set_exc_object(exc_obj);
    }
    // Nursery exception: the trace hooks, `record_application_traceback` and
    // `w_int_new` can move both the instance and the carrier. Pin the carrier
    // first — `pin_root` is a safepoint — and reload it the way
    // `expand_pop_roots` reloads every live variable. The instance word is
    // pinned too, because later calls take that word.
    let _exc_roots = pyre_object::gc_roots::push_roots();
    let mut err_slot = err.pin(&_exc_roots);
    let exc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(err.exc_object);
    err.reload(&_exc_roots, err_slot);
    // PyPy `PyFrame.handle_bytecode` calls `OperationError.record_context`
    // only on the ordinary OperationError arm. `RaiseWithExplicitTraceback`
    // (RERAISE) goes straight to `handle_operation_error(attach_tb=False)`:
    // the same exception is still propagating and must not acquire the
    // exception handled by an intervening finally block as a new context.
    //
    // `error.py record_context` records it once and then marks the
    // OperationError, so the frames the SAME error merely unwinds through never
    // re-derive it.  `PyError::context_recorded` is that mark, and it rides the
    // error outward because the dispatch loop moves the same value into
    // `Err(err)` on propagation.  The mark is set below whether or not an active
    // exception was found, mirroring the `finally`.
    //
    // # TODO: call `PyError::record_context`
    //
    // **Deviation.** The body is spelled here rather than going through the
    // method that holds it, so this reads as three steps where upstream reads
    // as one call.
    //
    // **Why.** `record_context`'s only callee is `chain_context`, which no
    // annotated graph reaches, so it is absent from the CallRegistry and the
    // method cannot be typed: Phase-A ends with `chain_context ... not
    // registered in CallRegistry`, and the method joins the rtyper
    // skip-subject census. Inlining it here does not cost an annotated graph
    // — `handle_exception` is already a census entry for reasons of its own.
    //
    // **How to fix.** Make `chain_context` reachable from a graph the
    // minting closure already walks, so the registry carries it; then this
    // becomes `err.record_context(context_source)` and the method annotates
    // wherever it is called.
    if err.attach_tb && !err.context_recorded {
        let last = match context_source {
            ContextSource::GeneratorChain => get_sys_exception(),
            ContextSource::ResumedFrameOnly => get_current_exception(),
        };
        crate::error::chain_context(pyre_object::gc_roots::shadow_stack_get(exc_slot), last);
        err.set_exc_object(pyre_object::gc_roots::shadow_stack_get(exc_slot));
        err.set_context_recorded(true);
    }
    let frame = unsafe { &mut *frame_anchor.live() };
    if err.attach_tb {
        if !ec.is_null() && unsafe { !(*ec).gettrace().is_null() } {
            // The materialized exception is a nursery object and lives only in
            // the `PyError` field plus the pin above. Publish the forwarded
            // address as the in-flight root before the trace hook runs
            // arbitrary Python. `record_application_traceback` re-publishes
            // the possibly-replaced operr below.
            err.set_exc_object(pyre_object::gc_roots::shadow_stack_get(exc_slot));
            set_in_flight_exception(err.exc_object);
            let saved_trace = frame.get_w_f_trace();
            let _trace_roots = pyre_object::gc_roots::push_roots();
            let saved_trace_slot = pyre_object::gc_roots::shadow_stack_len();
            let saved_trace = pyre_object::gc_roots::pin_root(saved_trace);
            if !saved_trace.is_null() {
                frame.getorcreatedebug(-1).w_f_trace = pyre_object::PY_NULL;
            }
            let after_exc_result =
                unsafe { (*ec).bytecode_trace_after_exception(frame as *mut PyFrame) };
            // The hook ran application code; restore the slot on the frame that
            // survived it.  `saved_trace` is a nursery function, so reload it
            // from the root rather than the pre-callback local.
            let frame = unsafe { &mut *frame_anchor.live() };
            let saved_trace = pyre_object::gc_roots::shadow_stack_get(saved_trace_slot);
            if !saved_trace.is_null() {
                frame.getorcreatedebug(-1).w_f_trace = saved_trace;
            }
            match after_exc_result {
                // pyopcode.py:144-145 — `except OperationError as e: operr = e`.
                // The replacement is a new carrier; publish it into the same
                // slot the later reloads read.
                Err(trace_err) => {
                    *err = trace_err;
                    err_slot = err.pin(&_exc_roots);
                }
                // The hook ran application code, so the field names the
                // pre-collection address; the pin holds the forwarded one.
                Ok(_) => {
                    err.reload(&_exc_roots, err_slot);
                    err.set_exc_object(pyre_object::gc_roots::shadow_stack_get(exc_slot));
                }
            }
        }
        // pyopcode.py:144-149 — after `except OperationError as e: operr = e`,
        // record/trace the (possibly tracer-replaced) operr, not the exception
        // captured before the trace hook ran.  Re-derive from `err`: an
        // unreplaced err returns the cached object; a replaced err
        // materialises the replacement.  Cache it so record and trace share
        // one object.
        let operr_obj = err.to_exc_object();
        if err.exc_object.is_null() {
            err.set_exc_object(operr_obj);
        }
        pyre_object::gc_roots::shadow_stack_set(exc_slot, err.exc_object);
        // `pyopcode.py pytraceback.record_application_traceback`
        // — prepends a `PyTraceback` wrapping the current frame onto
        // the exception's `w_traceback` chain.
        // `w_pytraceback_new` copies this pointer into the node it allocates,
        // so it has to be the address the frame has now.
        let frame = unsafe { &mut *frame_anchor.live() };
        unsafe {
            crate::pytraceback::record_application_traceback(
                pyre_object::gc_roots::shadow_stack_get(exc_slot),
                frame as *mut PyFrame,
                frame.last_instr as i64,
            );
        }
        err.reload(&_exc_roots, err_slot);
        err.set_exc_object(pyre_object::gc_roots::shadow_stack_get(exc_slot));
    }
    if err.attach_tb && !ec.is_null() && unsafe { !(*ec).gettrace().is_null() } {
        // `record_application_traceback` above allocates, so the frame address
        // is re-read from the anchor rather than reused.
        let frame = unsafe { &mut *frame_anchor.live() };
        // `pyopcode.py handle_operation_error` calls
        // `ec.exception_trace(self, operr)` with the live carrier, and
        // `executioncontext.py exception_trace` normalizes it in place to
        // build the `(w_type, w_value, w_traceback)` argument — including the
        // traceback read, so the caller does not assemble one.
        err.reload(&_exc_roots, err_slot);
        err.set_exc_object(pyre_object::gc_roots::shadow_stack_get(exc_slot));
        if let Err(trace_err) = unsafe { (*ec).exception_trace(frame as *mut PyFrame, err) } {
            // The call sits outside the trace-ticker recovery block, so a tracer
            // exception replaces the original error and propagates without
            // searching this frame for a handler for the original.
            *err = trace_err;
            return false;
        }
        // `exception_trace` normalizes the carrier in place, which keeps the
        // same exception, then runs the tracer; the pin, not the field, holds
        // its forwarded address.
        err.reload(&_exc_roots, err_slot);
        err.set_exc_object(pyre_object::gc_roots::shadow_stack_get(exc_slot));
    }
    // `attach_tb=False` (RaiseWithExplicitTraceback) suppresses the traceback
    // record for the frame that performed the re-raise only.  Once that frame's
    // record/trace decision is made, the flag is cleared so that if no handler
    // is found here and the exception propagates to the caller, that outer frame
    // records its own traceback entry — mirroring the special-exception being
    // unwrapped to a plain OperationError after one frame.
    err.set_attach_tb(true);
    // `record_application_traceback` and `exception_trace` above both allocate;
    // `pycode` is read off the frame, so a stale frame yields a stale code
    // object as well as a stale value stack.
    let frame = unsafe { &mut *frame_anchor.live() };
    // `pyopcode.py handle_operation_error`:
    // `self.getcode().lookup_exceptiontable(self.last_instr)`.
    // The method is `@jit.elidable`; call the PyCode wrapper so the
    // residual ABI keys on the code object, not a table slice.
    // pyre's `last_instr` is a rustpython code-unit index; the PyPy-shaped
    // `lookup_exceptiontable` lookup takes byte offsets, so multiply by 2.
    // (See pycode.rs: varint values are word offsets but the lookup
    // operates in byte space, mirroring `pycode.py`.)
    //
    // `frame.last_instr == -1` is the pre-first-opcode sentinel
    // (`pyframe.py` initialization).  An injected operr
    // (`eval_frame_plain_with_operr`) drives `handle_exception` before any
    // bytecode has executed, so the lookup must mirror PyPy
    // `pycode.py`: with `instr_offset == -1`, the first entry's
    // `start <= -1` is False and `start > -1` is True, returning the
    // `depth == -1` sentinel (no handler).  Skip the table lookup outright
    // rather than casting -1 to `u32::MAX` (panic in debug, wrap in
    // release).
    let lookup_result = if frame.last_instr < 0 {
        None
    } else {
        let pc_bytes = (frame.last_instr as u32) * 2;
        crate::pycode::unpack_exceptiontable_lookup(unsafe {
            crate::pycode::w_code_lookup_exceptiontable(
                frame.pycode as pyre_object::PyObjectRef,
                pc_bytes,
            )
        })
    };
    let pc_units = if frame.last_instr < 0 {
        0u32
    } else {
        frame.last_instr as u32
    };
    err.reload(&_exc_roots, err_slot);

    // `pypy/interpreter/pyopcode.py` exception-table dispatch.
    if let Some((target_bytes, depth, lasti)) = lookup_result {
        // `pyopcode.py:155-156` — depth is relative (0 = empty value
        // stack); convert to absolute by adding the frame's locals+cells
        // base, then drop the stack to that depth.
        let target_depth = frame.nlocals() + frame.ncells() + depth as usize;
        // `pyopcode.py handle_operation_error` → `dropvaluesuntil`.
        frame.dropvaluesuntil(target_depth);
        err.reload(&_exc_roots, err_slot);
        // `pyopcode.py:157-170` — lasti=True: push the raise-site offset
        // as an int below the exception, so RERAISE N can read it for
        // traceback/f_lineno correctness.  If this dispatch was triggered
        // by RERAISE (reraise_lasti from PyError, mirroring PyPy's
        // `handle_operation_error(reraise_lasti=...)`), use the original
        // raise-site lasti the RERAISE carried; otherwise use the current
        // instruction (the raising site itself).
        if lasti {
            let lasti_value: i64 = if err.reraise_lasti >= 0 {
                err.reraise_lasti as i64
            } else {
                pc_units as i64
            };
            frame.push(pyre_object::w_int_new(lasti_value));
            err.reload(&_exc_roots, err_slot);
            err.set_exc_object(pyre_object::gc_roots::shadow_stack_get(exc_slot));
        }
        // pyopcode.py: reraise_lasti is a local of handle_operation_error;
        // OperationError raised from this function carries no lasti.  Clear
        // here so a re-thrown PyError does not double-consume.
        err.set_reraise_lasti(-1);
        err.set_exc_object(pyre_object::gc_roots::shadow_stack_get(exc_slot));
        let exc_obj = err.to_exc_object();
        // Same shape as `opcode_build_list`: materialise, then push. The push
        // has to land on the frame the materialisation left live.
        let frame = unsafe { &mut *frame_anchor.live() };
        pyre_object::gc_roots::shadow_stack_set(exc_slot, exc_obj);
        frame.push(pyre_object::gc_roots::shadow_stack_get(exc_slot));
        // The exception is now on this frame's handler.  Drop the backend
        // `_store_exception` cells so a later compiled `GUARD_NO_EXCEPTION`
        // does not re-deliver the value this except already consumed.
        crate::runtime_ops::jit_clear_published_exception();
        // The decoded `target` is a byte offset; pyre's `next_instr` is a
        // code-unit index, so divide by 2.
        *next_instr = (target_bytes / 2) as usize;
        return true;
    }

    // `pyopcode.py:175-185` no-handler propagation: if this unwind was
    // triggered by RERAISE N, restore `last_instr` to the original
    // raise-site offset so `frame.f_lineno` reports the right line.
    if err.reraise_lasti >= 0 {
        frame.last_instr = err.reraise_lasti as isize;
    }
    err.set_reraise_lasti(-1);
    frame.set_frame_finished_execution(true);
    err.set_exc_object(pyre_object::gc_roots::shadow_stack_get(exc_slot));

    false
}

/// Execute a frame — pure interpreter, no JIT.
///
/// Crate-private: the canonical
/// surface is `PyFrame::run` / `PyFrame::execute_frame` (PyPy
/// `pyframe.py run` / `pyframe.py execute_frame`).  Retained as a
/// free function because pyre's JIT override mechanism (call.rs
/// `EVAL_OVERRIDE: OnceLock<EvalFn>` where `EvalFn = fn(&mut PyFrame,
/// Option<&mut FrameResumeArgs>) -> PyResult`) requires a `fn` pointer.  Rust
/// methods cannot be cast to `fn` pointers, so the canonical body stays as a
/// free function and the `EVAL_OVERRIDE.unwrap_or(eval_frame_plain)` fallback
/// (`call.rs`'s `get_eval_fn`) continues to reference it directly.
///
/// `resume` is `Some` only for a suspended frame being resumed
/// (`generator.py _invoke_execute_frame`); an ordinary call passes
/// `None` and enters at the frame's first instruction.
pub(crate) fn eval_frame_plain(
    frame: &mut PyFrame,
    resume: Option<&mut crate::call::FrameResumeArgs>,
) -> PyResult {
    frame.execute_frame_plain(resume)
}

/// pyframe.py execute_frame body — enter/call_trace/eval_loop/
/// return_trace/leave wrapping. When `operr` is Some, the generator's
/// throw() path routes it through handle_operation_error and sets
/// last_instr = next_instr - 1 before resuming (pyframe.py).
#[allow(dead_code)]
pub(crate) fn eval_frame_plain_with_operr(frame: &mut PyFrame, operr: Option<PyError>) -> PyResult {
    frame.execute_frame(None, operr)
}

enum FrameResume {
    Yielded(PyObjectRef),
    Dispatch(Option<PyError>),
}

/// pyframe.py `resume_execute_frame`.  A suspended `yield from` is
/// resumed only after `execute_frame` has entered the outer frame.  Clearing
/// `w_yielding_from` before the delegate call gives the running generator the
/// same transient `gi_yieldfrom is None` state as PyPy.
fn prepare_frame_resume(
    frame: &mut PyFrame,
    w_inputvalue: Option<PyObjectRef>,
    operr: Option<PyError>,
    throw_args: Option<([PyObjectRef; 3], usize)>,
) -> Result<FrameResume, PyError> {
    let mut pending_operr = operr;
    if !frame.w_yielding_from.is_null() {
        let w_yf = frame.w_yielding_from;
        frame.w_yielding_from = pyre_object::PY_NULL;
        let w_arg = w_inputvalue.unwrap_or_else(pyre_object::w_none);
        match crate::baseobjspace::resume_yield_from(
            frame,
            w_yf,
            w_arg,
            pending_operr.take(),
            throw_args,
        ) {
            Ok(Some(value)) => return Ok(FrameResume::Yielded(value)),
            // The delegate's StopIteration value is already on the outer
            // frame stack and its SEND completion target is installed.
            Ok(None) => return Ok(FrameResume::Dispatch(None)),
            // The delegate's error is thrown into this frame's own dispatch;
            // the sent value was the delegate's and is not resumed here.
            Err(err) => return Ok(FrameResume::Dispatch(Some(err))),
        }
    }
    if pending_operr.is_none()
        && let Some(w_arg_or_err) = w_inputvalue
    {
        // `pending_operr` is None on this path. Returning here keeps that
        // handle out of the live set across `resume_execute_frame`, which
        // can collect.
        let _ = frame.resume_execute_frame(w_arg_or_err)?;
        return Ok(FrameResume::Dispatch(None));
    }
    Ok(FrameResume::Dispatch(pending_operr))
}

/// Seed a suspended frame with its resumption payload, leaving it positioned to
/// dispatch from its own pc.
///
/// `Ok(Some(value))` means a suspended `yield from` delegate yielded and this
/// frame must not run at all; `Ok(None)` means dispatch.
///
/// Split out of `eval_frame_plain_with_resume` because the JIT portal performs
/// the same seeding before it enters: the merge point is keyed on the frame's
/// current pc, so the resumed state has to be in place first.
pub fn prepare_frame_resume_for_dispatch(
    frame: &mut PyFrame,
    resume: &mut crate::call::FrameResumeArgs,
) -> Result<Option<PyObjectRef>, PyError> {
    match prepare_frame_resume(
        frame,
        resume.w_inputvalue.take(),
        resume.operr.take(),
        resume.throw_args.take(),
    )? {
        FrameResume::Yielded(value) => Ok(Some(value)),
        FrameResume::Dispatch(Some(mut err)) => {
            let mut next_instr = frame.next_instr();
            if !handle_exception_with_context(
                frame,
                &mut err,
                &mut next_instr,
                ContextSource::ResumedFrameOnly,
            ) {
                return Err(err);
            }
            frame.last_instr = next_instr as isize - 1;
            Ok(None)
        }
        FrameResume::Dispatch(None) => Ok(None),
    }
}

/// Shadow-stack stand-in for `pyframe.py execute_frame`'s live `w_arg_or_err`.
///
/// RPython keeps that local rooted from graph entry through `call_trace` and
/// `resume_execute_frame` (`execute_frame.insert_stack_check_here`;
/// `shadowstack.py` inserts the root at every collection). One bracket covers
/// the whole span; each allocating call is followed by a reload from the slot.
///
/// Construction and reload are macros (`pin_resume_args!` /
/// `reload_resume_args!`) so they expand into the already-translated caller
/// (`execute_frame`, `eval_with_jit_inner`) rather than becoming their own
/// graphs. Upstream has no helper type: the local is a GC pointer and the
/// transformer writes the root.
pub struct ResumeArgPins {
    pub roots: pyre_object::gc_roots::RootScope,
    pub input_slot: Option<usize>,
    pub operr_slot: Option<usize>,
}

/// Open the `w_arg_or_err` bracket on `resume` and reload the locals from
/// the slots. Expands in the caller so `try_pin` is not a FunDecl.
#[macro_export]
macro_rules! pin_resume_args {
    ($resume:expr) => {{
        let input_word = match $resume.w_inputvalue {
            Some(v) if !v.is_null() => v,
            _ => pyre_object::PY_NULL,
        };
        let operr_word = match $resume.operr.as_ref() {
            Some(err) => err.as_raw() as pyre_object::PyObjectRef,
            None => pyre_object::PY_NULL,
        };
        if input_word.is_null() && operr_word.is_null() {
            None
        } else {
            let roots = pyre_object::gc_roots::push_roots();
            // One publish+normalize for the whole live set: sequential
            // `pin_root` would query after the first write and leave the
            // later value invisible to a foreign collection
            // (`RootScope::pin_roots`).
            let base = roots.pin_roots(&[input_word, operr_word]);
            let pins = $crate::eval::ResumeArgPins {
                roots,
                input_slot: if input_word.is_null() {
                    None
                } else {
                    Some(base)
                },
                operr_slot: if operr_word.is_null() {
                    None
                } else {
                    Some(base + 1)
                },
            };
            $crate::reload_resume_args!(Some(&pins), $resume);
            Some(pins)
        }
    }};
}

/// `shadowstack.py expand_pop_roots`: the local after a collecting call is
/// the slot, not the word from before the call.
#[macro_export]
macro_rules! reload_resume_args {
    ($pins:expr, $resume:expr) => {{
        if let Some(pins) = $pins {
            if let Some(slot) = pins.input_slot {
                let current = pins.roots.get(slot);
                $resume.w_inputvalue = if current.is_null() {
                    None
                } else {
                    Some(current)
                };
            }
            if let Some(slot) = pins.operr_slot
                && let Some(err) = $resume.operr.as_mut()
            {
                err.reload(&pins.roots, slot);
            }
        }
    }};
}

/// [`reload_resume_args!`] when the resume payload itself is `Option`.
#[macro_export]
macro_rules! reload_resume_args_opt {
    ($pins:expr, $resume_opt:expr) => {{
        if let Some(resume) = ($resume_opt).as_mut() {
            $crate::reload_resume_args!($pins, resume);
        }
    }};
}

pub(crate) fn eval_frame_plain_with_resume(
    frame: &mut PyFrame,
    resume: &mut crate::call::FrameResumeArgs,
    pins: Option<&ResumeArgPins>,
) -> PyResult {
    // Spend one unit of the recursion budget on this frame's activation and
    // give it back when it returns.  Every Python frame costs the same unit —
    // module body, called function, `exec`ed code, resumed generator — so the
    // depth `stack_check()` reads is the number of live Python frames.
    let _recursion_depth = crate::call::enter_recursive_frame(frame);
    frame.fix_array_ptrs();
    crate::reload_resume_args!(pins, resume);
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if ec.is_null() {
        crate::reload_resume_args!(pins, resume);
        if let Some(value) = prepare_frame_resume_for_dispatch(frame, resume)? {
            return Ok(value);
        }
        return eval_loop(frame, ec);
    }
    let execution_context = unsafe { &mut *ec };
    // executioncontext.py / threadlocals.py parity: the current
    // ExecutionContext is owned by the OS-thread locals and is installed by
    // thread bootstrap.  Entering an (including inlined) frame must not
    // replace that thread-owned slot from a frame field; doing so lets a
    // collapsed/stale translated frame identity poison every subsequent
    // `space.getexecutioncontext()` lookup.
    execution_context.enter(frame as *mut PyFrame);
    // `call_trace` / `return_trace` / `leave` run application Python, so a
    // nursery-born frame moves under them.  The raw `frame` argument is the
    // abandoned copy after the first callback; the JIT portal re-reads through
    // `FrameRoot`, and this interpreter entry does the same with `FrameAnchor`.
    let frame_anchor = FrameAnchor::new(frame);
    crate::reload_resume_args!(pins, resume);
    let mut got_exception = true;
    // pyframe.py PyFrame.execute_frame parity:
    //   try:
    //     ec.call_trace(self)            # outside inner try
    //     try:
    //       ... eval ...
    //     finally:
    //       ec.return_trace(self, w_exitvalue)
    //     got_exception = False
    //   finally:
    //     ec.leave(self, w_exitvalue, got_exception)
    //
    // call_trace lives in the outer try only — if it raises, neither the
    // eval body nor return_trace runs, but leave still does (because
    // enter() already executed).  Python finally semantics: a finally
    // block that raises replaces the prior exception (return_trace
    // overrides eval-body, leave overrides everything).
    let (mut outer_result, w_exitvalue) = (|| -> (PyResult, PyObjectRef) {
        // `execute_frame` calls `call_trace` before `resume_execute_frame`.
        // The sent value and `OperationError` (`error.py`) are GC locals
        // there (`w_arg_or_err`). The caller's `ResumeArgPins` already
        // roots them from `stack_check` through this hook and the resume.
        let trace = execution_context.call_trace(frame_anchor.live());
        crate::reload_resume_args!(pins, resume);
        if let Err(e) = trace {
            return (Err(e), pyre_object::w_none());
        }
        let mut inner_result = (|| -> PyResult {
            let frame = unsafe { &mut *frame_anchor.live() };
            crate::reload_resume_args!(pins, resume);
            if let Some(value) = prepare_frame_resume_for_dispatch(frame, resume)? {
                return Ok(value);
            }
            let frame = unsafe { &mut *frame_anchor.live() };
            let result = eval_loop(frame, ec)?;
            Ok(result)
        })();
        let mut w_exitvalue = match &inner_result {
            Ok(v) => *v,
            Err(_) => pyre_object::w_none(),
        };
        // `return_trace` runs application Python while the exit value and
        // the pending exception are still owed to `leave`.
        let return_trace_result = {
            let roots = pyre_object::gc_roots::push_roots();
            let exit_slot = roots.base();
            let exit = roots.pin_root(w_exitvalue);
            let result = match inner_result {
                Err(err) => {
                    let mut err = err;
                    let err_slot = err.pin(&roots);
                    let result = execution_context.return_trace(frame_anchor.live(), exit);
                    w_exitvalue = roots.get(exit_slot);
                    err.reload(&roots, err_slot);
                    inner_result = Err(err);
                    result
                }
                Ok(v) => {
                    let v = pyre_object::gc_roots::pin_root(v);
                    let result = execution_context.return_trace(frame_anchor.live(), exit);
                    w_exitvalue = roots.get(exit_slot);
                    inner_result = Ok(v);
                    result
                }
            };
            result
        };
        // Python finally: a finally-block exception replaces any
        // pending exception from the try-body. Only the all-OK path
        // advances to `got_exception = false`.
        // Both hooks hand the exit value back at its live address, and both
        // the local and `inner_result` carry the word taken before the
        // callback ran, so each result is rebuilt from what came back.
        let combined = match return_trace_result {
            Err(rt_err) => (Err(rt_err), w_exitvalue),
            Ok(live) => (inner_result.map(|_| live), live),
        };
        if combined.0.is_ok() {
            got_exception = false;
        }
        combined
    })();
    let leave_result = {
        let roots = pyre_object::gc_roots::push_roots();
        let result = match outer_result {
            Err(err) => {
                let mut err = err;
                let err_slot = err.pin(&roots);
                let result =
                    execution_context.leave(frame_anchor.live(), w_exitvalue, got_exception);
                err.reload(&roots, err_slot);
                outer_result = Err(err);
                result
            }
            Ok(v) => {
                let v = pyre_object::gc_roots::pin_root(v);
                let result =
                    execution_context.leave(frame_anchor.live(), w_exitvalue, got_exception);
                outer_result = Ok(v);
                result
            }
        };
        result
    };
    match leave_result {
        Err(leave_err) => Err(leave_err),
        Ok(live) => outer_result.map(|_| live),
    }
}

/// Resume interpretation after compiled code guard failure.
pub fn eval_loop_for_force(frame: &mut PyFrame) -> PyResult {
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    eval_loop(frame, ec)
}

fn eval_loop(frame: &mut PyFrame, ec: *mut crate::PyExecutionContext) -> PyResult {
    // Bump the monotonic frame eval-loop entry odometer: a user Python frame
    // is about to run bytecode.  The FBW FOR_ITER Option-C guard snapshots
    // this around a residual call to detect a body effect that ran through
    // user code (a side-effecting getter / dunder / module top level).
    crate::call::bump_frame_entry_count();
    // Count this interpreter activation so the JIT eval loop's GC safepoint
    // stops firing once the nesting is deep enough to hide a root
    // (PYRE_GC_INTERP root-completeness): module level and one called
    // function's loop still collect, and an `eval_loop_jit` re-entered from
    // inside an opcode handler observes depth >= 3 and skips collection.
    // No-op when the flag is off.
    let _eval_activation = pyre_object::gc_interp::EvalActivationGuard::enter();
    if _eval_activation.armed() {
        // Publish the process-stable configuration in the breaker word once
        // per activation, before the first dispatch. Compiled back-edges mask
        // this bit out.
        majit_ir::eval_breaker_word::set_gc_interp();
        // The periodic poll rides the action ticker. Without
        // `has_bytecode_counter` the ticker never wraps (`decrement_ticker`).
        crate::executioncontext::enable_gc_bytecode_counter();
    }
    let _current_frame_guard = if ec.is_null() {
        install_current_frame(frame)
    } else {
        install_current_frame_tls_only(frame)
    };
    let code = unsafe { &*crate::pyframe_get_pycode(frame) };
    let mut next_instr = frame.next_instr();

    loop {
        if next_instr >= code.instructions.len() {
            return Ok(w_none());
        }

        let pc = next_instr;
        frame.last_instr = pc as isize;
        // pypy/interpreter/pyopcode.py dispatch_bytecode, non-jitted arm:
        //   self.last_instr = intmask(next_instr)
        //   bytecode_only_trace only when a tracer (or reverse debugging) is
        //   installed; otherwise decrement_ticker, and action_dispatcher only
        //   once that goes negative. next_instr is reloaded only after those.
        if !ec.is_null() {
            if frame.take_failed_attr_before_opcode() {
                unsafe { (*ec).run_failed_attr_finalizers() };
            }
            // `pyopcode.py` `dispatch_bytecode` calls `bytecode_only_trace`
            // only when a tracer is installed, and reloads `next_instr` from
            // `last_instr` only after that hook or `action_dispatcher`.
            // `get_w_f_trace` stays off the no-tracer path.
            let needs_trace = unsafe { !(*ec).w_tracefunc.is_null() };
            if needs_trace {
                let trace_result = unsafe {
                    (*ec).bytecode_trace(
                        frame as *mut PyFrame,
                        crate::executioncontext::TICK_COUNTER_STEP,
                    )
                };
                // `handle_bytecode` wraps `dispatch_bytecode` in the same
                // `except OperationError` that routes an opcode error through
                // `handle_operation_error`. A signal delivered from
                // `bytecode_trace` must search this frame's exception blocks
                // at `last_instr`.
                if let Err(mut err) = trace_result {
                    if handle_exception(frame, &mut err, &mut next_instr) {
                        continue;
                    }
                    return Err(err);
                }
                if frame.last_instr as usize != pc {
                    next_instr = frame.last_instr as usize;
                    continue;
                }
            } else {
                let ticker = crate::executioncontext::space_decrement_ticker(
                    unsafe { &mut *ec },
                    crate::executioncontext::TICK_COUNTER_STEP as isize,
                );
                if ticker < 0 {
                    if let Err(mut err) = unsafe { (*ec).perform_actions(frame) } {
                        if handle_exception(frame, &mut err, &mut next_instr) {
                            continue;
                        }
                        return Err(err);
                    }
                    if frame.last_instr as usize != pc {
                        next_instr = frame.last_instr as usize;
                        continue;
                    }
                }
            }
        } else {
            // No EC means no ticker, so this frame polls the breaker word
            // itself, the interpreter GC poll included.
            if let Err(mut err) = crate::executioncontext::service_eval_breaker(
                std::ptr::null_mut(),
                frame as *mut PyFrame,
            ) {
                if handle_exception(frame, &mut err, &mut next_instr) {
                    continue;
                }
                return Err(err);
            }
            pyre_object::gc_interp::dispatch_safepoint(majit_ir::eval_breaker_word::load());
        }
        let (opcode_pc, instruction, op_arg) = decode_instruction_forward(code, pc)?;
        let fallthrough = opcode_pc + 1;
        // `decode_instruction_forward` absorbs any EXTENDED_ARG prefix
        // units, so the real opcode may sit past `pc`.  Re-point `last_instr`
        // at the opcode unit (`opcode_pc`) so a falling-through handler's
        // `next_instr()` (= last_instr + 1) lands at `fallthrough` rather than
        // re-dispatching the opcode unit that trailed an EXTENDED_ARG.
        // Mirrors interp_jit.py dispatch (`set_last_instr_from_next_instr`).
        frame.set_last_instr_from_next_instr(fallthrough);
        match execute_opcode_step(frame, code, instruction, op_arg, fallthrough) {
            Ok(StepResult::Continue) | Ok(StepResult::CloseLoop { loop_header_pc: _ }) => {
                next_instr = frame.next_instr();
            }
            Ok(StepResult::Return(result)) => return Ok(result),
            Ok(StepResult::Yield(result)) => {
                // interp_jit.py PyFrame.dispatch `except Yield`: a suspended
                // frame outlives this portal invocation, so eagerly publish
                // its redirected fields. Ordinary Return deliberately keeps
                // the FORCE_TOKEN/GUARD_NOT_FORCED_2 lazy protocol.
                let _ = majit_metainterp::jit::hint_force_virtualizable(frame as *mut PyFrame);
                return Ok(result);
            }
            Err(mut err) => {
                if handle_exception(frame, &mut err, &mut next_instr) {
                    continue;
                }
                return Err(err);
            }
        }
    }
}

/// CPython 3.14's failed-attribute refcount boundary: once `LOAD_ATTR` has
/// popped a finalizable receiver, an otherwise-unreferenced temporary runs its
/// finalizer before the surrounding exception handler continues. This is
/// observable in `test_io.test_error_through_destructor` for both native and
/// `_pyio` streams.
///
/// PyPy's `UserDelAction` only drains the queue; the collect that discovers a
/// just-popped temporary is the 3.14 adaptation. Arm only when this object is
/// itself registered for a finalizer. An MRO `__del__` lookup is the wrong
/// predicate: many types expose `__del__` without `register_finalizer` ever
/// seeing the instance, and a collect then finds nothing to run. A live
/// registered receiver (`f.xyzzy`) still triggers the collect, but stays
/// reachable so its finalizer does not run.
#[majit_macros::dont_look_inside]
pub(crate) fn finalize_failed_attr_receiver_now(obj: PyObjectRef) -> bool {
    !obj.is_null() && majit_gc::gc_object_finalizer_pending(obj as usize)
}

impl SharedOpcodeHandler for PyFrame {
    type Value = PyObjectRef;

    type Anchor = FrameAnchor;

    fn anchor(&mut self) -> Self::Anchor {
        FrameAnchor::new(self)
    }

    fn push_anchored(anchor: &Self::Anchor, value: Self::Value) -> Result<(), PyError> {
        // A JIT-created frame lives in the nursery and the allocating step may
        // have relocated it; push onto the forwarded live frame.
        unsafe { &mut *anchor.live() }.push(value);
        Ok(())
    }

    fn push_value(&mut self, value: Self::Value) -> Result<(), PyError> {
        self.push(value);
        Ok(())
    }

    fn pop_value(&mut self) -> Result<Self::Value, PyError> {
        Ok(self.pop())
    }

    fn peek_at(&mut self, depth: usize) -> Result<Self::Value, PyError> {
        Ok(PyFrame::peek_at(self, depth))
    }

    /// `pyopcode.py MAKE_FUNCTION` is `@jit.unroll_safe`.
    #[majit_macros::unroll_safe]
    fn make_function(&mut self, code_obj: Self::Value) -> Result<Self::Value, PyError> {
        // `pypy/interpreter/pyopcode.py MAKE_FUNCTION` stamps
        // `func.w_func_globals = self.w_globals` from the running
        // frame's dict object directly.  Pyre resolves the same
        // canonical sibling via `get_w_globals()` and threads it
        // through `make_function_from_code_obj_with_globals_obj` so
        // the freshly-created function's `__globals__` identity IS
        // the frame's view, with no second resolution that could surface a
        // different dict object.
        let w_globals = self.get_w_globals();
        // Capture the globals OBJECT only; the raw `*mut DictStorage` is
        // recovered from the object via the proxy back-link wherever a frame
        // built from this function still needs it.  Threading a raw here is
        // what dangled exec-defined functions when the exec temp storage was
        // freed (the `GlobalsBinding` leak), so it is dropped.
        Ok(crate::runtime_ops::make_function_from_code_obj_with_globals_obj(code_obj, w_globals))
    }

    fn call_callable(
        &mut self,
        callable: Self::Value,
        args: &[Self::Value],
    ) -> Result<Self::Value, PyError> {
        call_callable(self, callable, args)
    }

    fn build_list(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(build_list_from_refs(items))
    }

    fn build_tuple(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(build_tuple_from_refs(items))
    }

    /// `pyopcode.py BUILD_MAP` is `@jit.unroll_safe`.
    #[majit_macros::unroll_safe]
    fn build_map(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        build_map_from_refs(items)
    }

    fn store_subscr(
        &mut self,
        obj: Self::Value,
        key: Self::Value,
        value: Self::Value,
    ) -> Result<(), PyError> {
        setitem(obj, key, value).map(|_| ())
    }

    fn list_append(&mut self, list: Self::Value, value: Self::Value) -> Result<(), PyError> {
        unsafe { w_list_append(list, value) };
        Ok(())
    }

    fn unpack_sequence(
        &mut self,
        seq: Self::Value,
        count: usize,
    ) -> Result<Vec<Self::Value>, PyError> {
        unpack_sequence_exact_tos(seq, count)
    }

    fn load_attr(&mut self, obj: Self::Value, name: &str) -> Result<Self::Value, PyError> {
        // The receiver is already off the value stack, so this local is the
        // only thing keeping it reachable while `getattr_str` runs arbitrary
        // Python — and the failing branch below still has to read its type.
        let roots = pyre_object::gc_roots::push_roots();
        let obj_slot = roots.base();
        let obj = roots.pin_root(obj);
        getattr_str(obj, name).map_err(|error| {
            if finalize_failed_attr_receiver_now(roots.get(obj_slot)) {
                // Keep this write on the live virtualizable red frame.  The
                // opaque helper only answers the semantic type question;
                // writing `frame.flags` inside it would race the JIT's
                // register-resident copy of the same field.
                self.defer_failed_attr_until_pop_except();
            }
            error
        })
    }

    fn load_special_attr(&mut self, obj: Self::Value, name: &str) -> Result<Self::Value, PyError> {
        crate::baseobjspace::load_special_resolve(obj, name)
    }

    fn store_attr(
        &mut self,
        obj: Self::Value,
        name: &str,
        value: Self::Value,
    ) -> Result<(), PyError> {
        setattr_str(obj, name, value).map(|_| ())
    }
}

impl LocalOpcodeHandler for PyFrame {
    fn load_local_value(&mut self, idx: usize) -> Result<Self::Value, PyError> {
        Ok(locals_w!(self)[idx])
    }

    fn load_local_checked_value(&mut self, idx: usize) -> Result<Self::Value, PyError> {
        let value = locals_w!(self)[idx];
        if value.is_null() {
            // Name resolution stays on the failure path only
            // (pyopcode.py LOAD_FAST -> _load_fast_failed), keeping the
            // varnames read out of the non-raising path.
            let code = unsafe { &*crate::pyframe_get_pycode(self) };
            let name = if idx < code.varnames.len() {
                code.varnames[idx].as_str()
            } else {
                "<cell>"
            };
            return Err(PyError::unbound_local_error(format!(
                "cannot access local variable '{name}' where it is not associated with a value"
            )));
        }
        // Cell objects are valid even if their contents are PY_NULL
        // (needed for __class__ cell during class body execution).
        // The cell itself is non-null, so the check above passes.
        Ok(value)
    }

    fn store_local_value(&mut self, idx: usize, value: Self::Value) -> Result<(), PyError> {
        // STORE_FAST always writes directly to the slot.
        // Cell content updates use STORE_DEREF, not STORE_FAST.
        self.set_locals_w(idx, value);
        Ok(())
    }
}

impl NamespaceOpcodeHandler for PyFrame {
    /// PyPy: LOAD_NAME checks locals first (class body), then globals,
    /// then `__builtins__` via `load_global_value`'s fallback chain
    /// (pypy/interpreter/pyopcode.py LOAD_NAME → load_global).
    ///
    /// Non-dict mapping locals (`exec(src, g, mapping)`,
    /// `pypy/interpreter/pyopcode.py ensure_ns`) bypass the
    /// `*mut DictStorage` fast path and route through
    /// `space.getitem(w_locals, name)` directly per PyPy
    /// `pyopcode.py:LOAD_NAME` `space.finditem_str(w_locals, name)`.
    fn load_name_value(&mut self, name: &str, nameindex: usize) -> Result<Self::Value, PyError> {
        let w_locals = self.get_w_locals();
        if !w_locals.is_null() {
            // At module scope `initialize_frame_scopes` binds `w_locals` to the
            // very same object as `w_globals`, so the locals probe here is a
            // redundant copy of the globals lookup `load_global_value` runs
            // next: same dict, same builtins fallback, identical result. Skip
            // it when they are identical — both to avoid the double lookup and,
            // critically, to avoid materializing a throwaway `w_str` key on
            // every module-loop LOAD_NAME (`load_global_value` already probes
            // the globals dict borrow-based via `getitem_str` + the cell cache).
            let w_globals = self.get_w_globals();
            if !std::ptr::eq(w_locals, w_globals) {
                // pyopcode.py:967-968 `w_value = space.finditem_str(
                // self.getorcreatedebug().w_locals, varname)`. The probe is
                // `finditem_str`, not `getitem`: an exact dict answers from its
                // strategy with the borrowed key, so neither the wrapped key nor
                // a KeyError is built. A class body reads every one of its names
                // through here, and the names it does not bind itself — a module
                // global, a builtin — miss on every pass, which is the miss that
                // shortcut exists for. `finditem_str` keeps a dict subclass on
                // the generic path, where a `__getitem__` or `__missing__`
                // override still runs.
                // `pyopcode.py w_varname = self.getname_w(nameindex)`:
                // the generic arm names its key through `co_names_w`, so a
                // namespace that cannot use the borrowed-key shortcut — a
                // `dict` subclass, a non-dict mapping — wraps no key of its own
                // per execution either.
                if let Some(value) = crate::baseobjspace::finditem_str_named(
                    w_locals,
                    name,
                    self.pycode as PyObjectRef,
                    nameindex,
                )? {
                    return Ok(value);
                }
                // pyopcode.py:972 — a missing locals entry falls through to
                // `LOAD_GLOBAL_cached`.
            }
            return self.load_global_value(name, nameindex);
        }
        // No locals mapping bound (degenerate): fall through to globals.
        self.load_global_value(name, nameindex)
    }

    /// pyopcode.py STORE_NAME —
    /// `space.setitem_str(self.getorcreatedebug().w_locals, varname, w_value)`.
    ///
    /// `nameindex` addresses the `co_names_w` slot naming the key on the
    /// non-dict-mapping arm below; callers holding no index pass
    /// [`crate::pyopcode::NO_NAMEINDEX`].
    ///
    /// Writes straight to `w_locals` (the class namespace, or — at module
    /// scope — the globals dict). It must NOT route through `getdictscope`:
    /// that runs `fast2locals`, which would erase a module frame's
    /// `CO_FAST_HIDDEN` inlined-comprehension locals (their fast slot is NULL,
    /// the binding lives in `w_locals` via STORE_NAME) on every store.
    fn store_name_value(
        &mut self,
        name: &str,
        nameindex: usize,
        value: Self::Value,
    ) -> Result<(), PyError> {
        // Three allocations sit between the opcode's `pop_value` and the store:
        // `get_or_create_w_locals` when the frame has no mapping yet, and
        // `named_key_hash` / `w_code_getname_w_or_new` when the `co_names_w`
        // slot is not realized.  The popped value's only name is this word.
        let roots = pyre_object::gc_roots::push_roots();
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(value);
        let locals_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(self.get_or_create_w_locals());
        let hash = crate::baseobjspace::named_key_hash(name, self.pycode as PyObjectRef, nameindex);
        if store_name_into_dict(roots.get(locals_slot), name, hash, roots.get(value_slot)) {
            return Ok(());
        }
        let key = unsafe {
            crate::pycode::w_code_getname_w_or_new(self.pycode as PyObjectRef, nameindex, name)
        };
        crate::baseobjspace::setitem(roots.get(locals_slot), key, roots.get(value_slot))?;
        Ok(())
    }

    /// pypy/interpreter/pyopcode.py STORE_GLOBAL — bypasses w_locals
    /// and writes directly into w_globals so `exec("global x; x = 1", g, l)`
    /// lands `x` in `g` even when `l != g`.
    ///
    /// `nameindex` names the key through `co_names_w` on the dict-subclass arm;
    /// callers holding no index pass [`crate::pyopcode::NO_NAMEINDEX`].
    fn store_global_value(
        &mut self,
        name: &str,
        nameindex: usize,
        value: Self::Value,
    ) -> Result<(), PyError> {
        // `named_key_hash` / `w_code_getname_w_or_new` realize the name when
        // the `co_names_w` slot is not, and the mapping's own `__setitem__`
        // runs Python: the popped value's only name is this word.
        let roots = pyre_object::gc_roots::push_roots();
        let value_slot = roots.base();
        let _ = roots.pin_root(value);
        let globals_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(self.get_w_globals());
        let hash = crate::baseobjspace::named_key_hash(name, self.pycode as PyObjectRef, nameindex);
        if !roots.get(globals_slot).is_null()
            && !store_name_into_dict(roots.get(globals_slot), name, hash, roots.get(value_slot))
        {
            let key = unsafe {
                crate::pycode::w_code_getname_w_or_new(self.pycode as PyObjectRef, nameindex, name)
            };
            crate::baseobjspace::setitem(roots.get(globals_slot), key, roots.get(value_slot))?;
        }
        Ok(())
    }

    /// pypy/interpreter/pyopcode.py `_load_global` — first reads
    /// `w_globals`, then falls back to `self.get_builtin().getdictvalue
    /// (space, varname)`.  PyPy's `get_builtin()` returns the `Module`
    /// chosen at frame-creation time by `pick_builtin(w_globals)`
    /// (`pyframe.py` + `pypy/module/__builtin__/moduledef.py`),
    /// so `exec("x = len", {"__builtins__": {}})` raises `NameError`
    /// because the empty dict is the picked builtin.
    fn load_global_value(&mut self, name: &str, nameindex: usize) -> Result<Self::Value, PyError> {
        // `celldict.py _LOAD_GLOBAL_cached`: under the JIT, or when the
        // frame's globals is not the pycode's first-seen globals, the whole
        // cached path is bypassed via `_load_global_fallback` — both the
        // per-pycode `_globals_caches[nameindex]` slot and the strategy-level
        // `get_global_cache(varname)` install, because both would attach a
        // cache to a module that is not the one being executed.  Identity is
        // `pycode.w_globals is self.get_w_globals_storage()` — the wrapped
        // dict OBJECT on both sides (`w_code_get_w_globals` vs the frame's
        // `w_globals`).  Positive form (`load_attr_cached`) keeps the
        // annotator off the bare-`!` hazard; `we_are_jitted()` folds to
        // `ConstBool(true)` so the cache arm and its `Arc<Mutex<GlobalCache>>`
        // chase are dead-code-eliminated on the lifted graph.
        let mut w_globals = self.get_w_globals();
        let use_cache = if majit_metainterp::jit::we_are_jitted() {
            false
        } else {
            unsafe {
                let cwo = crate::pycode::w_code_get_w_globals(self.pycode as PyObjectRef);
                !cwo.is_null()
                    && std::ptr::eq(cwo, w_globals)
                    && !w_globals.is_null()
                    && pyre_object::dictmultiobject::is_module_dict(w_globals)
            }
        };
        // `_load_global` (pyopcode.py) reads the builtins through the METHOD
        // `self.get_builtin()`, not the `builtin` field.  Under the default
        // `honor__builtins__=False` the constructor never assigns that field
        // (pyframe.py) and the method answers `space.builtin` without
        // consulting the frame at all, so the field being unset must not be
        // observable here.  A frame the JIT materialized from a virtual is
        // exactly such a frame: reading the raw field made the builtins leg
        // silently find nothing there, raising `NameError` for a builtin that
        // plainly exists.
        if use_cache
            && let Some(value) =
                unsafe { load_global_cache_hit(self.pycode as PyObjectRef, nameindex) }
        {
            return Ok(value);
        }
        let mut w_builtin = self.get_builtin();
        if use_cache {
            let cache_hit: Option<PyObjectRef> = unsafe {
                pyre_object::with_roots!(w_builtin, w_globals => load_global_via_cache(
                    w_globals,
                    w_builtin,
                    name,
                    self.pycode as PyObjectRef,
                    nameindex,
                ))
            }?;
            if let Some(value) = cache_hit {
                return Ok(value);
            }
        }
        // `_load_global_fallback` → `_load_global` (`pyopcode.py`):
        // `space.finditem_str(self.get_w_globals(), varname)`.  finditem_str
        // takes a borrowed-string fast path for real W_DictObject /
        // W_ModuleDictObject layouts and dispatches a dict subclass through
        // the general mapping object, so a raising key `__eq__` propagates
        // instead of being swallowed as a miss.  A cached miss lands here too
        // and finds nothing again before the `NameError`.
        if !w_globals.is_null()
            && let Some(value) = pyre_object::with_roots!(w_builtin =>
                crate::baseobjspace::finditem_str_named(
                    w_globals,
                    name,
                    self.pycode as PyObjectRef,
                    nameindex,
                )
            )?
        {
            return Ok(value);
        }
        // `self.get_builtin().getdictvalue(space, varname)`.
        if !w_builtin.is_null() && unsafe { pyre_object::is_module(w_builtin) } {
            let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
            if !w_dict.is_null()
                && let Some(value) = crate::baseobjspace::finditem_str(w_dict, name)?
            {
                return Ok(value);
            }
        }
        // `pyopcode.py _load_global_failed`: NameError.
        Err(PyError::oefmt_name_error_text(
            format!("name '{name}' is not defined"),
            name,
        ))
    }

    fn null_value(&mut self) -> Result<Self::Value, PyError> {
        Ok(PY_NULL)
    }
}

impl StackOpcodeHandler for PyFrame {
    /// `pyopcode.py SWAP` is `@jit.unroll_safe`.
    #[majit_macros::unroll_safe]
    fn swap_values(&mut self, depth: usize) -> Result<(), PyError> {
        // `pyopcode.py SWAP`, peek/settop element-wise.  A
        // `<[T]>::swap` call hands the locals array to a callee, which the
        // codewriter can only residualize; the element-wise spelling stays
        // native array operations, and it is what the preceding comment
        // already claimed this function did.
        //
        // `assert oparg >= 2` is upstream's own precondition, and it is what
        // makes both indices known non-negative to the annotator.
        assert!(depth >= 2);
        // Both halves go through the `maybe_none` read: a stack slot holds
        // NULL while an inlined comprehension runs, where
        // `LOAD_FAST_AND_CLEAR` pushed an unbound local and cleared its slot.
        let w_top = self.peekvalue_maybe_none(0);
        let w_other = self.peekvalue_maybe_none(depth - 1);
        // The first store overwrites the only stack slot still holding
        // `w_top`.  `FixedObjectArray::set_ref` roots its receiver and the
        // value it was handed, and nothing else, across a barrier that can
        // wait behind a foreign collection — so an unrooted `w_top` would be
        // stale by the second store.  The GC transform carries every live
        // variable on the shadow stack across that point; spell the same
        // thing out here and read `w_top` back.
        let roots = pyre_object::gc_roots::push_roots();
        let top_slot = roots.base();
        let _ = roots.pin_root(w_top);
        self.settopvalue(w_other, 0);
        self.settopvalue(roots.get(top_slot), depth - 1);
        Ok(())
    }
}

/// `celldict.py _LOAD_GLOBAL_cached` slow-path: consult the
/// W_ModuleDictObject's `mstrategy.get_global_cache` for `name`,
/// chaining through `cache.builtincache` to the `__builtins__` Module
/// on a globals miss.  Returns `None` when the name is absent from
/// both globals and builtins.
///
/// Public extern alias so `runtime_ops::jit_load_name_from_namespace`
/// can reuse this cache path on a globals miss.
///
/// # Safety
/// `w_module_dict` must be a valid W_ModuleDictObject; `w_builtin`
/// may be null or a valid Module; `name` is the requested str key.
pub unsafe fn load_global_via_cache_extern(
    w_module_dict: PyObjectRef,
    w_builtin: PyObjectRef,
    name: &str,
) -> Option<PyObjectRef> {
    // JIT extern path: discard `space.finditem_str`'s `PyError` because
    // the C-ABI signature has no error channel.  For the builtins dict
    // (the only call site that can raise here in practice), `finditem_str`
    // only raises on a non-dict mapping with custom `__getitem__` — never
    // for the W_DictObject / W_ModuleDictObject backing real builtins.
    match unsafe { load_global_via_cache(w_module_dict, w_builtin, name, std::ptr::null_mut(), 0) }
    {
        Ok(v) => v,
        Err(_) => None,
    }
}

/// `pyopcode.py:855-859 space.setitem_str(w_ns, varname, w_value)` on a real
/// `W_DictObject` / `W_ModuleDictObject`: stores by borrowed `&str` through the
/// strategy without materializing a throwaway `w_str` (an overwrite reuses the
/// stored key; only a new name allocates one).  This is the raw mapping store,
/// not `__setitem__`, exactly as the object-keyed `setitem` resolves a dict.
///
/// Answers `false` when `w_ns` is not a dict, leaving the caller on the
/// object-keyed path: a dict subclass is an ordinary instance in pyre and must
/// keep its mapping identity and any `__setitem__` override, and a non-dict
/// mapping (`exec(src, g, mapping)`) has no strategy to store into.
///
/// `hash` is `name`'s digest when the caller holds it — the memo
/// `rstr.py ll_strhash` keeps in the shared `co_names_w` string
/// (`crate::baseobjspace::named_key_hash`), so a stored name is hashed once per
/// string rather than once per opcode.  Zero leaves the strategy to hash the
/// borrowed bytes.
fn store_name_into_dict(w_ns: PyObjectRef, name: &str, hash: i64, value: PyObjectRef) -> bool {
    if !unsafe { pyre_object::is_dict(w_ns) } {
        return false;
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_hashed(w_ns, name, hash, value);
    }
    true
}

/// STORE_NAME for a caller holding the wrapped name rather than a `co_names_w`
/// index — the JIT's `bh_store_name_fn`, whose `w_name` operand is the trace's
/// own `box_str_constant`.
///
/// Naming the key from that constant is what keeps the mapping arm from minting
/// an immortal `w_str` per execution, the job `co_names_w` (`pycode.py:127-129`)
/// does for the interpreter; the residual carries no index to reach a slot with.
///
/// # Safety
/// `w_name` must point to a valid `str` object.
pub unsafe fn store_name_value_w(
    frame: &mut PyFrame,
    w_name: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), PyError> {
    // `get_or_create_w_locals` allocates a mapping for a frame that has none,
    // so it collects; reach it before the key's payload is borrowed, because a
    // `&str` into a moved object outlives nothing.
    let roots = pyre_object::gc_roots::push_roots();
    let name_slot = roots.base();
    let _ = roots.pin_root(w_name);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(value);
    let w_locals = frame.get_or_create_w_locals();
    let w_name = roots.get(name_slot);
    let value = roots.get(value_slot);
    // The trace's own `box_str_constant` names the key, so it is the same
    // string object on every execution — `rstr.py ll_strhash`'s memo
    // makes it hashed once rather than once per store.
    let hash = unsafe { pyre_object::unicodeobject::w_str_hash_memoized(w_name) };
    // A lone surrogate has no `&str` dict key. `setitem` below keeps the
    // object (`W_UnicodeObject.text_w` returns the buffer verbatim).
    if let Some(name) = unsafe { pyre_object::unicodeobject::w_str_get_value_opt(w_name) }
        && store_name_into_dict(w_locals, name, hash, value)
    {
        return Ok(());
    }
    crate::baseobjspace::setitem(w_locals, w_name, value)?;
    Ok(())
}

/// DELETE_NAME counterpart of [`store_name_value_w`], for the JIT's
/// `bh_delete_name_fn`.  The trace already holds the key object, so it deletes
/// through that rather than naming the key again.
///
/// # Safety
/// `w_name` must point to a valid `str` object.
pub unsafe fn delete_name_w(frame: &mut PyFrame, w_name: PyObjectRef) -> Result<(), PyError> {
    // `get_or_create_w_locals` allocates for a frame with no mapping and
    // `delitem` runs a mapping's `__delitem__`; both collect, and the key is
    // read after each.
    let roots = pyre_object::gc_roots::push_roots();
    let name_slot = roots.base();
    let _ = roots.pin_root(w_name);
    let w_locals = frame.get_or_create_w_locals();
    crate::baseobjspace::delitem(w_locals, roots.get(name_slot)).map_err(|err| {
        if matches!(err.kind, PyErrorKind::KeyError) {
            let w_name = roots.get(name_slot);
            match unsafe { pyre_object::unicodeobject::w_str_get_value_opt(w_name) } {
                Some(name) => {
                    PyError::oefmt_name_error_text(format!("name '{name}' is not defined"), name)
                }
                None => {
                    let text = unsafe { pyre_object::unicodeobject::w_str_get_wtf8(w_name) };
                    let mut msg = rustpython_wtf8::Wtf8Buf::from("name '");
                    msg.push_wtf8(text);
                    msg.push_str("' is not defined");
                    let mut err = PyError::new(PyErrorKind::NameError, msg);
                    err.set_w_name_context(w_name);
                    err
                }
            }
        } else {
            err
        }
    })
}

/// STORE_GLOBAL counterpart of [`store_name_value_w`], for the JIT's
/// `bh_store_global_fn`.  `pyopcode.py` writes straight into `w_globals`.
///
/// # Safety
/// `w_name` must point to a valid `str` object.
pub unsafe fn store_global_value_w(
    frame: &mut PyFrame,
    w_name: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), PyError> {
    let hash = unsafe { pyre_object::unicodeobject::w_str_hash_memoized(w_name) };
    let w_globals = frame.get_w_globals();
    if !w_globals.is_null() {
        let stored = unsafe { pyre_object::unicodeobject::w_str_get_value_opt(w_name) }
            .is_some_and(|name| store_name_into_dict(w_globals, name, hash, value));
        if !stored {
            crate::baseobjspace::setitem(w_globals, w_name, value)?;
        }
    }
    Ok(())
}

/// `celldict.py _LOAD_GLOBAL_cached`: the `cache.getvalue(space)` and
/// `builtincache.getvalue(space)` hits, which run before
/// `load_global_via_cache` roots its operands; every other outcome falls
/// through to it.  The cell is copied out under the cache mutex and unwrapped
/// after the mutex is released, because unwrapping an `IntMutableCell`
/// allocates.  Residual for the same reason as `load_global_via_cache`.
#[majit_macros::dont_look_inside]
#[inline]
unsafe fn load_global_cache_hit(pycode: PyObjectRef, nameindex: usize) -> Option<PyObjectRef> {
    let cache = unsafe { crate::pycode::w_code_globals_caches_get(pycode, nameindex) }?;
    let cell = {
        let c = cache.lock();
        match c.cell {
            Some(v) => v,
            None if c.valid => c.builtincache.as_ref()?.lock().cell?,
            None => return None,
        }
    };
    Some(unsafe { pyre_object::celldict::unwrap_cell(cell) })
}

/// `celldict.py _LOAD_GLOBAL_cached`.  When `pycode` is
/// non-null, `pycode._globals_caches[nameindex]` is consulted before
/// `mstrategy.get_global_cache(name)`; on the slow path, the resolved
/// `cache.ref` (`celldict.py/353`) is installed into the slot.
///
/// Returns `Ok(Some(value))` on cache hit (globals or chained builtin),
/// `Ok(None)` on full miss, `Err(_)` when `space.finditem_str` raises
/// during the builtins fallback (`baseobjspace.py
/// W_Root.getdictvalue` → `space.finditem_str`).
///
/// The `GlobalCache` chase walks an `Arc<Mutex<GlobalCache>>` cache
/// structure the tracer cannot model (its `deref` reads past the
/// refcount / borrow-flag header, not a value-model identity).
/// `celldict.py _LOAD_GLOBAL_cached` bypasses the whole cache
/// under `jit.we_are_jitted()`, resolving the builtin through
/// `space.finditem_str` instead, so the cache is off-trace plumbing;
/// residualize it (`@jit.dont_look_inside`) like the sibling
/// `load_attr_caching`.
#[majit_macros::dont_look_inside]
unsafe fn load_global_via_cache(
    w_module_dict: PyObjectRef,
    w_builtin: PyObjectRef,
    name: &str,
    pycode: PyObjectRef,
    nameindex: usize,
) -> Result<Option<PyObjectRef>, PyError> {
    use pyre_object::celldict::unwrap_cell;
    use pyre_object::dictmultiobject::{DictOperationGuard, W_ModuleDictObject};
    let module_guard = DictOperationGuard::new(w_module_dict, &[w_builtin, pycode]);
    let w_module_dict = module_guard.root(0);
    let mut w_builtin = module_guard.root(1);
    let pycode = module_guard.root(2);
    // Body is a chain of unsafe-fn / raw-ptr ops on caller-supplied
    // PyObjectRefs; SAFETY contract is on the `unsafe fn` signature
    // (caller upholds w_module_dict / pycode / w_builtin validity).
    unsafe {
        // `celldict.py:292-313`: per-name slot fast path.  Read the slot,
        // upgrade the weakref; if the cache is alive, follow cell → builtincache
        // → builtins w_dict before falling through to the strategy lookup.
        if !pycode.is_null()
            && let Some(cache) = crate::pycode::w_code_globals_caches_get(pycode, nameindex)
        {
            // `celldict.py:295-313`: fast-path layout —
            //
            //     w_value = cache.getvalue(self.space)
            //     if w_value is not None:
            //         return w_value
            //     if cache.valid:
            //         builtincache = cache.builtincache
            //         if builtincache is not None:
            //             w_value = builtincache.getvalue(self.space)
            //             if w_value is not None:
            //                 return w_value
            //             # builtin getdictvalue + _load_global_failed
            //
            // The builtins fallback is GATED on `builtincache is not None`:
            // `get_global_cache` attaches one only for a name the globals
            // lack and the builtins hold, so any other miss falls through to
            // the slow path (`# either no cache or an invalid cache`).
            let (cell_opt, valid, bc_opt) = {
                let c = cache.lock();
                (c.cell, c.valid, c.builtincache.clone())
            };
            if let Some(v) = cell_opt {
                return Ok(Some(unwrap_cell(v)));
            }
            if valid && let Some(bc) = bc_opt {
                let bcell = bc.lock().cell;
                if let Some(v) = bcell {
                    return Ok(Some(unwrap_cell(v)));
                }
                // `celldict.py`: the builtin's cell went away, so ask
                // `self.get_builtin().getdictvalue(space, varname)`; the
                // `_load_global_failed` NameError is the caller's on `None`.
                if !w_builtin.is_null() && pyre_object::is_module(w_builtin) {
                    let w_builtin_dict = pyre_object::w_module_get_w_dict(w_builtin);
                    if !w_builtin_dict.is_null() {
                        return crate::baseobjspace::finditem_str(w_builtin_dict, name);
                    }
                }
            }
        }
        let raw = &mut *(w_module_dict as *mut W_ModuleDictObject);
        if raw.mstrategy.is_null() || raw.dstorage.is_null() {
            return Ok(None);
        }
        // `celldict.py ModuleDictStrategy.switch_to_object_strategy` swaps the
        // live strategy, so its module-only global-cache method is no longer
        // available.  Continue through the ordinary mapping lookup just as
        // `_load_global_fallback` does upstream.
        if pyre_object::dictmultiobject::w_module_dict_is_object_strategy(w_module_dict) {
            if let Some(value) = pyre_object::with_roots!(w_builtin =>
                crate::baseobjspace::finditem_str(w_module_dict, name)
            )? {
                return Ok(Some(value));
            }
            if !w_builtin.is_null() && pyre_object::is_module(w_builtin) {
                let w_builtin_dict = pyre_object::w_module_get_w_dict(w_builtin);
                if !w_builtin_dict.is_null() {
                    return crate::baseobjspace::finditem_str(w_builtin_dict, name);
                }
            }
            return Ok(None);
        }
        // `celldict.py:315-322`: the slow-path install just routes through
        // `w_globals.get_global_cache(varname)` and writes
        // `pycode._globals_caches[nameindex] = cache.ref`.
        // `get_global_cache` reads `space.builtin.w_dict` when
        // `honor__builtins__` is off, to attach the builtin's own cache.
        let w_space_builtin_dict = if crate::baseobjspace::HONOR_BUILTINS {
            pyre_object::PY_NULL
        } else {
            let ec = crate::call::getexecutioncontext();
            let w_space_builtin = if ec.is_null() {
                pyre_object::PY_NULL
            } else {
                (*ec).get_builtin()
            };
            if !w_space_builtin.is_null() && pyre_object::is_module(w_space_builtin) {
                pyre_object::w_module_get_w_dict(w_space_builtin)
            } else {
                pyre_object::PY_NULL
            }
        };
        let strategy =
            pyre_object::dictmultiobject::w_module_dict_module_strategy_mut(w_module_dict);
        let cache = strategy.get_global_cache(w_module_dict, name, w_space_builtin_dict);
        // `celldict.py:321/353 pycode._globals_caches[nameindex] = cache.ref`.
        if !pycode.is_null() {
            crate::pycode::w_code_globals_caches_set(pycode, nameindex, &cache);
        }
        // `_LOAD_GLOBAL_cached` lines 296-298: cache.getvalue hit.
        let cell_opt = cache.lock().cell;
        if let Some(v) = cell_opt {
            return Ok(Some(unwrap_cell(v)));
        }
        // `_load_global_fallback` → `_load_global` (`pyopcode.py`):
        // when globals miss, route through `self.get_builtin().getdictvalue(
        // space, varname)` which resolves via `space.finditem_str` per
        // `baseobjspace.py W_Root.getdictvalue`.  The caller threads
        // its frame's picked builtin in as `w_builtin`.
        if !w_builtin.is_null() && pyre_object::is_module(w_builtin) {
            let w_builtin_dict = pyre_object::w_module_get_w_dict(w_builtin);
            if !w_builtin_dict.is_null() {
                return crate::baseobjspace::finditem_str(w_builtin_dict, name);
            }
        }
        Ok(None)
    }
}

impl PyFrame {
    /// `pyopcode.py _report_stopiteration_sometimes`.
    ///
    /// ```python
    /// def _report_stopiteration_sometimes(self, w_iterator, operr):
    ///     ...
    ///     if (isinstance(w_iterator, GeneratorOrCoroutine) or
    ///             isinstance(w_iterator, AsyncGenASend) or
    ///             operr.has_any_traceback()):
    ///         self.space.getexecutioncontext().exception_trace(self, operr)
    /// ```
    ///
    /// Upstream calls the three arms "an approximative rule" for a case it
    /// says it cannot emulate.  Counting `exception` events in the consuming
    /// frame under `sys.settrace` on 3.14.6, each row the same on the first
    /// call and on the tenth:
    ///
    /// ```text
    /// for _ in CustomIter():   # Python-level __next__ raising StopIteration
    ///     -> [('__next__', 'StopIteration'), ('consume', 'StopIteration')]
    /// for _ in map(stop, [1]): # native __next__, Python-raised StopIteration
    ///     -> [('stop', 'StopIteration'), ('consume', 'StopIteration')]
    /// for _ in [1]:            -> []
    /// for _ in range(1):       -> []
    /// for _ in iter(step, 3):  -> []
    /// ```
    ///
    /// The traceback arm covers those rows: the report fires when the
    /// StopIteration carries a traceback and not otherwise.  The `map` row is
    /// what settles that this arm is about the traceback rather than about the
    /// iterator's kind: its `__next__` is native, and the event still fires,
    /// because the StopIteration passed through a Python frame on the way out.
    ///
    /// A generator's ending carries no traceback and is the iterator arm's
    /// own row.  3.14 answers it two ways for one code object -- silence on
    /// the first execution, which runs the unspecialised `_FOR_ITER` and jumps
    /// past `END_FOR`, then an event once the loop specialises to
    /// `FOR_ITER_GEN` and the end arrives at `INSTRUMENTED_END_FOR`, where
    /// `monitor_stop_iteration` mints it.  The event it mints is for a
    /// traceback-less StopIteration, so the traceback arm alone cannot reach
    /// it; the iterator arm answers a warm loop the way 3.14 does and a cold
    /// one the way pypy3 does, which is to say the same way every time.
    ///
    /// `pyre/extra_tests/parity_tests/foriter_stopiteration_exception_event.py`
    /// pins the table.
    ///
    /// The exhaustion an iterator signals from Rust never materialises an
    /// exception, so the common loop exit answers `has_any_traceback` false
    /// off a null carrier and never reaches the tracer test.
    fn _report_stopiteration_sometimes(
        &mut self,
        w_iterator: PyObjectRef,
        operr: &mut PyError,
    ) -> Result<(), PyError> {
        let generator = unsafe {
            pyre_object::generator::is_generator_or_coroutine(w_iterator)
                || pyre_object::generator::is_async_gen_asend(w_iterator)
        };
        if !generator && !operr.has_any_traceback() {
            return Ok(());
        }
        let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
        if ec.is_null() {
            return Ok(());
        }
        // The tracer runs Python, so the frame it is handed is read off the
        // anchor rather than out of `self`.
        let anchor = FrameAnchor::new(self);
        unsafe { (*ec).exception_trace(anchor.live(), operr) }
    }

    /// Route intrinsic operands through a helper of a module imported on
    /// first use.
    ///
    /// `pop` clears the frame slot, so the operands are livevars of this
    /// native frame alone. Importing runs the module body, and the lookup
    /// and the call run Python, so all three collect: root the operands the
    /// way `build_interpolation_op` roots its pops and read the forwarded
    /// words back for the call.
    fn call_module_helper(
        &mut self,
        module_name: &str,
        helper: &str,
        args: &[PyObjectRef],
    ) -> Result<(), PyError> {
        let _helper_roots = pyre_object::gc_roots::push_roots();
        let args_base = pyre_object::gc_roots::pin_roots(args);
        let anchor = FrameAnchor::new(self);
        let module = self.import_module(module_name)?;
        let module_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(module);
        let func = getattr_str(pyre_object::gc_roots::shadow_stack_get(module_slot), helper)?;
        let func_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(func);
        let live: Vec<PyObjectRef> = (0..args.len())
            .map(|index| pyre_object::gc_roots::shadow_stack_get(args_base + index))
            .collect();
        let result = call_callable(
            self,
            pyre_object::gc_roots::shadow_stack_get(func_slot),
            &live,
        )?;
        Self::push_anchored(&anchor, result)
    }
}

/// PyPy: pyopcode.py GET_ITER → space.iter(w_iterable)
///       pyopcode.py FOR_ITER → space.next(w_iterator)
impl IterOpcodeHandler for PyFrame {
    fn iter_value(&mut self, iterable: Self::Value) -> Result<Self::Value, PyError> {
        crate::baseobjspace::iter(iterable)
    }

    /// GET_ITER: convert iterable to iterator.
    /// PyPy: space.iter(w_iterable) → calls __iter__ or wraps in seq_iter.
    fn ensure_iter_value(&mut self, iter: Self::Value) -> Result<(), PyError> {
        unsafe {
            // mappingproxy iterates over its backing dict's keys.
            // dictproxyobject.py descr_iter → space.iter(self.w_mapping).
            // The backing mapping is a `dict`, which moves, and publishing it
            // to the stack slot runs the frame array's write barrier, whose
            // slow path can wait behind a foreign collection
            // (`FixedObjectArray::set_ref` reloads both operands across it for
            // that reason).  Pin the mapping on the read, so both the store and
            // the operand this dispatches on come out of the root slot rather
            // than the word that was live before the barrier.  The bracket ends
            // with the branch: past it the mapping is the frame's own stack
            // slot, which the collector updates.
            let iter = if pyre_object::is_dict_proxy(iter) {
                let roots = pyre_object::gc_roots::push_roots();
                let mapping_slot = roots.base();
                let _ = roots.pin_root(pyre_object::w_dict_proxy_get_mapping(iter));
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, roots.get(mapping_slot));
                roots.get(mapping_slot)
            } else {
                iter
            };
            // `range` sequence → fresh `W_IntRangeIterator` cursor; replace
            // the stack operand so FOR_ITER advances the iterator, not the
            // reusable range object.  (Mirrors the dict-proxy rewrite
            // above.)  This runs in the loop preheader, outside the traced
            // loop body, so the JIT's `for i in range(N)` fast path is
            // unaffected.
            if pyre_object::is_w_range(iter) {
                let it = pyre_object::w_range_iter(iter);
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, it);
                return Ok(());
            }
            // Already an iterator
            if pyre_object::is_range_iter(iter)
                || pyre_object::is_long_range_iter(iter)
                || pyre_object::is_seq_iter(iter)
                || pyre_object::is_list_iter(iter)
                || pyre_object::is_list_reverse_iter(iter)
                || pyre_object::is_tuple_iter(iter)
                || pyre_object::is_set_iterator(iter)
                || pyre_object::generator::is_generator(iter)
                || pyre_object::interp_itertools::is_repeat(iter)
                || pyre_object::interp_itertools::is_count(iter)
                || pyre_object::interp_itertools::is_takewhile(iter)
                || pyre_object::interp_itertools::is_dropwhile(iter)
                || pyre_object::interp_itertools::is_filterfalse(iter)
                || pyre_object::interp_itertools::is_pairwise(iter)
                || pyre_object::interp_itertools::is_cycle(iter)
                || pyre_object::interp_itertools::is_chain(iter)
                || pyre_object::dictmultiobject::is_dict_view_iterator(iter)
                || pyre_object::functional::is_enumerate(iter)
                || pyre_object::functional::is_reversed(iter)
                || pyre_object::functional::is_filter(iter)
                || pyre_object::functional::is_map(iter)
                || pyre_object::functional::is_zip(iter)
                || pyre_object::operation::is_callable_iterator(iter)
                || pyre_object::interp_sre::is_sre_scanner(iter)
                || crate::module::r#struct::is_unpack_iter(iter)
            {
                return Ok(());
            }
            // `pypy/objspace/std/dictmultiobject.py`
            // `W_DictViewKeysObject.descr_iter` (and values / items
            // siblings) returns a live `W_BaseDictMultiIterObject`. Pyre
            // produces a `W_BaseDictMultiIterObject` carrying the source
            // dict's `dictversion` counter so mid-iteration mutation
            // surfaces as `RuntimeError("dictionary changed size during
            // iteration")` per `:1719-1741 descr_next`.
            if pyre_object::dictmultiobject::is_dict_view(iter) {
                let kind = pyre_object::dictmultiobject::w_dict_view_get_kind(iter);
                let w_dict = pyre_object::dictmultiobject::w_dict_view_get_dict(iter);
                let it = pyre_object::dictmultiobject::w_dict_view_iterator_new(w_dict, kind);
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, it);
                return Ok(());
            }
            // list → W_FastListIterObject for an exact list; a subclass may override
            // `__iter__`, so route it through `space.iter`.
            if pyre_object::is_list(iter) {
                if pyre_object::is_exact_list(iter) {
                    let seq_iter = pyre_object::w_list_iter_new(iter);
                    let tos = self.valuestackdepth - 1;
                    self.set_locals_w(tos, seq_iter);
                    return Ok(());
                }
                let result = crate::baseobjspace::iter(iter)?;
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, result);
                return Ok(());
            }
            // tuple → seq_iter for an exact tuple; a subclass may override
            // `__iter__`, so route it through `space.iter`.
            if pyre_object::is_tuple(iter) {
                if pyre_object::is_exact_tuple(iter) {
                    let seq_iter = pyre_object::w_tuple_iter_new(iter);
                    let tos = self.valuestackdepth - 1;
                    self.set_locals_w(tos, seq_iter);
                    return Ok(());
                }
                let result = crate::baseobjspace::iter(iter)?;
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, result);
                return Ok(());
            }
            // str → list of 1-char strings → seq_iter
            if pyre_object::is_str(iter) {
                // Walk code points through the WTF-8 view so iterating a
                // surrogateescape / surrogatepass-decoded string yields its
                // lone surrogates instead of panicking in w_str_get_value.
                let _roots = pyre_object::gc_roots::push_roots();
                let iter_slot = pyre_object::gc_roots::pin_roots(&[iter]);
                let s =
                    pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(iter_slot))
                        .to_wtf8_buf();
                let chars_base = pyre_object::gc_roots::shadow_stack_len();
                for c in s.code_points() {
                    let mut one = rustpython_wtf8::Wtf8Buf::new();
                    one.push(c);
                    let _ =
                        pyre_object::gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(one));
                }
                let len = pyre_object::gc_roots::shadow_stack_len() - chars_base;
                let mut chars = Vec::with_capacity(len);
                for i in 0..len {
                    chars.push(pyre_object::gc_roots::shadow_stack_get(chars_base + i));
                }
                let char_list = pyre_object::w_list_new(chars);
                let seq_iter = pyre_object::w_seq_iter_new(char_list, len);
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, seq_iter);
                return Ok(());
            }
            // bytes/bytearray → seq_iter cursor over the bytes themselves, so
            // GET_ITER stays O(1) and a bytearray mutated mid-loop is observed
            // (baseobjspace::iter takes the same shape).
            if pyre_object::bytesobject::is_bytes_like(iter) {
                let len = pyre_object::bytesobject::bytes_like_len(iter);
                let seq_iter = pyre_object::w_seq_iter_new(iter, len);
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, seq_iter);
                return Ok(());
            }
            // dict → iterate over keys.
            // `pypy/objspace/std/dictmultiobject.py:W_DictMultiObject.descr_iter` returns
            // `W_DictMultiIterKeysObject` — pyre's
            // `W_BaseDictMultiIterObject` with kind=Keys plays the same
            // role, capturing the
            // dict's `dictversion` so mid-iteration mutation raises
            // `RuntimeError("dictionary changed size during
            // iteration")`.
            if pyre_object::is_dict(iter) {
                let it = pyre_object::dictmultiobject::w_dict_view_iterator_new(
                    iter,
                    pyre_object::dictmultiobject::DictViewKind::Keys,
                );
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, it);
                return Ok(());
            }
            // setobject.py W_BaseSetObject.descr_iter returns a live
            // W_SetIterObject over the set implementation.  Do not snapshot
            // the elements into a list: IteratorImplementation.next must see
            // a size change and make RuntimeError sticky.
            if pyre_object::is_set_or_frozenset(iter) {
                let set_iter = pyre_object::w_set_iter_new(iter);
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, set_iter);
                return Ok(());
            }
            // array.array → seq_iter cursor (interp_array.py descr_iter
            // returns space.newseqiter(self)).
            if pyre_object::interp_array::is_array(iter) {
                let len = pyre_object::interp_array::w_array_len(iter);
                let seq_iter = pyre_object::w_seq_iter_new(iter, len);
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, seq_iter);
                return Ok(());
            }
            // User-defined __iter__ — PyPy: space.iter → __iter__().
            // Instances, plus typed-payload builtins (e.g. deque) whose
            // type registers `__iter__` on its MRO.  Delegates to
            // baseobjspace::iter which handles type MRO and __getitem__
            // fallback (PyPy: space.iter → PyObject_GetIter → tp_iter or
            // PySeqIter_New).  Already-iterator payloads returned above, so
            // this only sees non-iterator containers.
            if pyre_object::is_instance(iter)
                || crate::typedef::r#type(iter).is_some_and(|t| {
                    crate::baseobjspace::lookup_in_type_where(t.as_ptr(), "__iter__").is_some()
                })
            {
                let result = crate::baseobjspace::iter(iter)?;
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, result);
                return Ok(());
            }
            // Type object: metaclass __iter__ (NOT the type's own MRO)
            // CPython: iter(X) calls type(X).__iter__(X)
            if pyre_object::is_type(iter) {
                // baseobjspace.py:76 — metaclass from w_class
                let mc = {
                    let w_class = (*iter).w_class;
                    let w_type_type = crate::typedef::w_type();
                    if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                        Some(w_class)
                    } else {
                        None
                    }
                };
                if let Some(metaclass) = mc
                    && let Some(method) = crate::baseobjspace::lookup_in_type(metaclass, "__iter__")
                {
                    let result = crate::call_function(method, &[iter]);
                    let tos = self.valuestackdepth - 1;
                    self.set_locals_w(tos, result);
                    return Ok(());
                }
            }
        }
        ensure_range_iter(iter)
    }

    /// FOR_ITER: advance the iterator one step.
    /// PyPy: space.next() → StopIteration means exhausted.
    fn iter_next(&mut self, mut iter: Self::Value) -> Result<Option<Self::Value>, PyError> {
        // baseobjspace::next walks the iterator protocol and raises
        // StopIteration for exhaustion.  All iterator kinds dispatch uniformly
        // through space.next here (pyopcode.py `w_nextitem =
        // self.space.next(w_iterator)`); the JIT specialises range/long-range/
        // seq by inlining this dispatch during tracing (trace_opcode.rs
        // iter_next), not by branching the interpreter opcode implementation.
        match pyre_object::with_roots!(iter => crate::baseobjspace::next(iter)) {
            Ok(result) => Ok(Some(result)),
            Err(e) => {
                let _stop_roots = pyre_object::gc_roots::push_roots();
                let iter_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(iter);
                let (stop, mut e) = e.matches_stop_iteration_keep();
                if stop {
                    self._report_stopiteration_sometimes(
                        pyre_object::gc_roots::shadow_stack_get(iter_slot),
                        &mut e,
                    )?;
                    Ok(None)
                } else {
                    Err(e)
                }
            }
        }
    }

    fn on_iter_exhausted_anchored(anchor: &Self::Anchor, target: usize) -> Result<(), PyError> {
        unsafe { &mut *anchor.live() }.set_last_instr_from_next_instr(target);
        Ok(())
    }
}

impl TruthOpcodeHandler for PyFrame {
    type Truth = bool;

    fn truth_value(&mut self, value: Self::Value) -> Result<Self::Truth, PyError> {
        truth_value(value)
    }

    fn bool_value_from_truth(
        &mut self,
        truth: Self::Truth,
        negate: bool,
    ) -> Result<Self::Value, PyError> {
        Ok(bool_value_from_truth(if negate { !truth } else { truth }))
    }

    fn unary_not_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        crate::opcode_ops::unary_not_value(value)
    }
}

impl ControlFlowOpcodeHandler for PyFrame {
    fn fallthrough_target(&mut self) -> usize {
        self.next_instr()
    }

    fn set_next_instr(&mut self, target: usize) -> Result<(), PyError> {
        self.set_last_instr_from_next_instr(target);
        Ok(())
    }

    fn close_loop(&mut self, target: usize) -> Result<StepResult<Self::Value>, PyError> {
        // Signal a back-edge to the main eval_loop, which handles
        // JIT counting and compiled code execution via try_back_edge_jit.
        Ok(StepResult::CloseLoop {
            loop_header_pc: target,
        })
    }

    /// pyopcode.py:180-183 RETURN_VALUE — frame_finished_execution = True
    /// when the returning path exits the frame (matched by StepResult::Return).
    fn finish_value(&mut self, value: Self::Value) -> Result<StepResult<Self::Value>, PyError> {
        #[cfg(not(feature = "sandbox"))]
        if interp_return_log_enabled() {
            interp_return_log(self, value);
        }
        self.set_frame_finished_execution(true);
        Ok(StepResult::Return(value))
    }
}

impl BranchOpcodeHandler for PyFrame {
    fn concrete_truth_as_bool(
        &mut self,
        _value: Self::Value,
        truth: Self::Truth,
    ) -> Result<bool, PyError> {
        Ok(truth)
    }
}

impl ArithmeticOpcodeHandler for PyFrame {
    fn binary_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: BinaryOperator,
    ) -> Result<Self::Value, PyError> {
        // Exact int/float pairs cannot run user code and do not need the
        // operands after the result allocation. Override / concat arms pin
        // inside `try_dispatch_binary_special` / `list_concat` the way
        // `compare` pins only `both_exact_builtin_instances_promoted == false`.
        binary_value(a, b, op)
    }

    fn compare_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: ComparisonOperator,
    ) -> Result<Self::Value, PyError> {
        // Same as `binary_value`: `compare` already publishes only on the
        // override arm (`both_exact_builtin_instances_promoted`).
        compare_value(a, b, op)
    }

    fn unary_negative_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        unary_negative_value(value)
    }

    fn unary_invert_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        unary_invert_value(value)
    }
}

impl ConstantOpcodeHandler for PyFrame {
    fn int_constant(&mut self, value: i64) -> Result<Self::Value, PyError> {
        Ok(w_int_new(value))
    }

    fn small_int_constant(&mut self, value: i64) -> Result<Self::Value, PyError> {
        Ok(w_small_int_const(value))
    }

    fn bigint_constant(&mut self, value: &crate::PyBigInt) -> Result<Self::Value, PyError> {
        Ok(w_long_new(value.clone()))
    }

    fn float_constant(&mut self, value: f64) -> Result<Self::Value, PyError> {
        Ok(w_float_new(value))
    }

    fn complex_constant(&mut self, re: f64, im: f64) -> Result<Self::Value, PyError> {
        Ok(pyre_object::complexobject::w_complex_new(re, im))
    }

    fn bool_constant(&mut self, value: bool) -> Result<Self::Value, PyError> {
        Ok(w_bool_from(value))
    }

    fn str_constant(&mut self, value: &rustpython_wtf8::Wtf8) -> Result<Self::Value, PyError> {
        Ok(box_str_constant(value))
    }

    fn bytes_constant(&mut self, value: &[u8]) -> Result<Self::Value, PyError> {
        Ok(pyre_object::bytesobject::w_bytes_from_bytes(value))
    }

    fn code_constant(
        &mut self,
        code: &crate::bytecode::CodeObject,
    ) -> Result<Self::Value, PyError> {
        // Reached only for a code constant nested inside a container constant
        // (e.g. a tuple element), which has no top-level `co_consts_w` slot;
        // realize a wrapper directly.  Top-level `LOAD_CONST` of a code constant
        // goes through `constant_at` below.
        //
        // Defensive: the compiler does not emit that shape, and both routes into
        // here are narrow.  `load_const_value` is entered either from the trait
        // default `constant_at` — which `PyFrame`, the only implementor,
        // overrides — or from `opcode_load_const` by way of
        // `OpcodeStepExecutor::load_const`, which no caller in the tree invokes
        // today, though `opcode_load_const` is registered as a JIT call target.
        // So the copy this makes is not on a measured path; see
        // `box_code_constant` for why removing it is not worth its price.
        Ok(crate::pycode::box_code_constant(code))
    }

    fn constant_at(
        &mut self,
        index: crate::bytecode::oparg::ConstIdx,
        _enclosing: &crate::bytecode::CodeObject,
    ) -> Result<Self::Value, PyError> {
        // `pyopcode.py getconstant_w(index) -> co_consts_w[index]`:
        // return the one object `self.pycode` holds at `index`.
        Ok(unsafe {
            crate::pycode::w_code_const(self.pycode as pyre_object::PyObjectRef, usize::from(index))
        })
    }

    fn none_constant(&mut self) -> Result<Self::Value, PyError> {
        Ok(w_none())
    }

    fn ellipsis_constant(&mut self) -> Result<Self::Value, PyError> {
        Ok(pyre_object::special::w_ellipsis())
    }

    fn slice_constant(
        &mut self,
        start: Self::Value,
        stop: Self::Value,
        step: Self::Value,
    ) -> Result<Self::Value, PyError> {
        Ok(pyre_object::w_slice_new(start, stop, step))
    }

    fn frozenset_constant(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(pyre_object::w_frozenset_from_items(items))
    }
}

/// `callmethod.py:66-78` fast-path discriminator: bind the receiver only
/// when the MRO descriptor `d` is a method-descriptor-typed function
/// (`flag_method_descriptor` — set on `function` alone, typedef.py:807)
/// that `getattr` surfaced unchanged.  The `d == attr` identity check is
/// the moral equivalent of `w_obj.getdictvalue(space, name) is None` plus
/// `has_object_getattribute()` (callmethod.py:46/67): an instance-dict
/// shadow, a descriptor `__get__` result, or a `__getattribute__` override
/// all hand `getattr` a different object than the raw descriptor.
/// Everything else takes the slow path (callmethod.py:79-82): the getattr
/// result is called as-is, with no self binding.
unsafe fn method_descriptor_bound(
    d: PyObjectRef,
    attr: PyObjectRef,
    obj: PyObjectRef,
) -> PyObjectRef {
    unsafe {
        if d != attr || !crate::is_function_carrier(d) {
            return PY_NULL;
        }
        // BuiltinFunction has no `__get__` (its typedef carries no
        // `method_descriptor` flag, function.py:783).
        if std::ptr::eq((*d).ob_type, &crate::BUILTIN_FUNCTION_TYPE as *const _) {
            PY_NULL
        } else {
            obj
        }
    }
}

/// Compute the `null_or_self` value LOAD_METHOD pushes alongside the
/// resolved attribute `attr` (the result of `getattr(obj, name)`).
///
/// Pure MRO inspection (`lookup_in_type` + descriptor-kind predicates) —
/// it never invokes a descriptor `__get__` or `__getattribute__`, so the
/// side effects already paid by the `getattr` that produced `attr` are not
/// repeated.  Shared by [`PyFrame::load_method`] and the blackhole residual
/// helper `bh_load_method_self_fn` so both bind self identically.
///
///  - method-descriptor function surfaced unchanged by getattr → bind
///    instance (self); see [`method_descriptor_bound`]
///  - EXACT classmethod → bind class (w_type); a wrapper subclass overriding
///    `__get__` produced `attr` through that override and takes no binding
///  - everything else (staticmethod / non-method descriptors / shadowed
///    or arbitrary class attrs) → no binding (NULL)
pub fn compute_load_method_bound(obj: PyObjectRef, attr: PyObjectRef, name: &str) -> PyObjectRef {
    unsafe {
        if pyre_object::is_method(attr) {
            return PY_NULL;
        }
        if pyre_object::is_instance(obj) {
            // callmethod.py:66-67 `w_value = w_obj.getdictvalue(space, name)`:
            // a shadowing instance attribute is what getattr returned for
            // every non-data descriptor — never bind self for it.  (Data
            // descriptors that win over the instance dict — property /
            // member — resolve to PY_NULL either way.)
            let _roots = pyre_object::gc_roots::push_roots();
            let base = pyre_object::gc_roots::pin_roots(&[obj, attr]);
            let shadowed = crate::objspace::std::mapdict::instance_node_getdictvalue(
                pyre_object::gc_roots::shadow_stack_get(base),
                rustpython_wtf8::Wtf8::new(name),
            )
            .is_some();
            let obj = pyre_object::gc_roots::shadow_stack_get(base);
            let attr = pyre_object::gc_roots::shadow_stack_get(base + 1);
            let w_type = pyre_object::w_instance_get_type(obj);
            // callmethod.py:33 `w_type.has_object_getattribute()`: a non-default
            // `__getattribute__` produced `attr` through the override rather
            // than the default descriptor path, so the MRO-shape binding
            // inference below does not apply.  `_PyObject_GetMethod` skips the
            // self-binding optimization for a custom `tp_getattro` (pushes
            // NULL), so an override returning the raw descriptor must call as a
            // plain function, not a bound method.  This is the same gate
            // `load_method_fast_path` applies before its fast path.
            //
            // The computing form, not the raw `uses_object_getattribute`
            // memo: that field starts `false`, `mutated()` resets it to
            // `false` on every type-dict write, and the only writer
            // (`getattribute_if_not_from_object`) is skipped while jitted.
            // Reading it raw therefore answers "custom `__getattribute__`"
            // for a type that inherits the default one, and the method load
            // drops `self` — while `load_method_fast_path`, which resolves
            // the same question by lookup, pushed the unbound function that
            // needs it.
            if !crate::baseobjspace::has_object_getattribute(w_type) {
                return PY_NULL;
            }
            let raw = crate::baseobjspace::lookup_in_type(w_type, name);
            match raw {
                _ if shadowed => PY_NULL,
                // staticmethod / classmethod wrappers: getattr already
                // unwrapped them, so the identity fast path below can never
                // match; an EXACT classmethod keeps its explicit cls binding.
                //
                // The classmethod test must be exact for the same reason
                // `classmethod_on_type_fast_path` is: a wrapper SUBCLASS
                // overriding `__get__` was resolved through that override, so
                // `attr` is whatever the override returned and there is no raw
                // function left for the class to be prepended to.  It falls
                // through to `method_descriptor_bound`, whose `d != attr` test
                // already answers no-binding.  The staticmethod arm needs no
                // such split — it answers PY_NULL either way.
                Some(d) if pyre_object::is_staticmethod(d) => PY_NULL,
                Some(d) if pyre_object::is_exact_classmethod(d) => w_type,
                Some(d) => method_descriptor_bound(d, attr, obj),
                None => {
                    // Not found in type MRO → found in instance __dict__.
                    // Instance __dict__ attrs bypass descriptor protocol.
                    PY_NULL
                }
            }
        } else if pyre_object::is_type(obj) {
            // Type receiver: PyPy resolves LOAD_METHOD through the
            // METAclass MRO (`space.type(w_obj)`), so a name found in the
            // type's own MRO reaches the call as a plain getattr value
            // with no binding.
            //
            // `is_type` reports the physical layout every type object shares,
            // not the metaclass, so read the metaclass and require it to be
            // `type`.  The shape inferred below is what
            // `type.__getattribute__` produces; a custom metaclass can
            // override `__getattribute__` or define a data descriptor of the
            // same name, and either one produced `attr` in place of the
            // class's own MRO entry — binding `cls` onto that value would
            // pass the class to something that never asked for it.
            let metatype_is_type = crate::typedef::r#type(obj)
                .is_some_and(|meta| std::ptr::eq(meta.as_ptr(), crate::typedef::w_type()));
            if !metatype_is_type {
                return PY_NULL;
            }
            let raw = crate::baseobjspace::lookup_in_type(obj, name);
            match raw {
                // Exact for the reason given on the instance arm above.
                Some(d) if pyre_object::is_exact_classmethod(d) => obj,
                Some(_) => PY_NULL, // found in own MRO → no binding
                None => {
                    // Not in the type's own MRO → resolved via the
                    // metaclass MRO; bind the type for a method-descriptor
                    // function getattr surfaced unchanged.
                    match crate::typedef::r#type(obj)
                        .and_then(|meta| crate::baseobjspace::lookup_in_type(meta.as_ptr(), name))
                    {
                        Some(d) => method_descriptor_bound(d, attr, obj),
                        None => PY_NULL,
                    }
                }
            }
        } else if let Some(w_type) =
            crate::typedef::r#type(obj).filter(|_| !pyre_object::is_module(obj))
        {
            // Builtin-storage receiver (list, str, ... and their
            // subclasses such as enum members) found via TypeDef; the
            // same fast-path discriminator applies — `dict.get` etc. are
            // FunctionWithFixedCode (interp2app) attrs that getattr
            // returns unchanged, while staticmethods (str.maketrans) and
            // classmethods (dict.fromkeys) were already unwrapped.
            //
            // Gated like the instance arm, and for the same reason: the MRO
            // shape below infers a binding from how getattr WOULD have
            // resolved the name, which describes nothing once an override
            // resolved it instead.  A subclass whose `__getattribute__`
            // hands back the class's own function reaches this arm on its
            // storage rather than the instance one, and binding there
            // prepends a receiver the call site never wrote.
            if !crate::baseobjspace::has_object_getattribute(w_type.as_ptr()) {
                return PY_NULL;
            }
            match crate::baseobjspace::lookup_in_type(w_type.as_ptr(), name) {
                Some(d) if pyre_object::is_staticmethod(d) => PY_NULL,
                // Exact for the reason given on the instance arm above.
                Some(d) if pyre_object::is_exact_classmethod(d) => w_type.as_ptr(),
                Some(d) => method_descriptor_bound(d, attr, obj),
                None => PY_NULL,
            }
        } else {
            PY_NULL
        }
    }
}

/// Shared WITH_EXCEPT_START call semantics for the interpreter and generated
/// JitCode residual.  Keeping the call here preserves the bytecode operation's
/// exact `__exit__(type, value, traceback)` argument construction on both paths.
pub fn with_except_start_values(
    exit_func: PyObjectRef,
    exit_self: PyObjectRef,
    val: PyObjectRef,
) -> PyObjectRef {
    let exc_type = crate::typedef::r#type(val).map_or(pyre_object::w_none(), |p| p.as_ptr());
    // pyopcode.py:1358-1362 reads W_BaseException.w_traceback directly while
    // building the `__exit__(type, value, traceback)` arguments.
    let exc_tb = if unsafe { pyre_object::is_exception(val) } {
        let tb = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(val) };
        if tb.is_null() {
            pyre_object::w_none()
        } else {
            tb
        }
    } else {
        pyre_object::w_none()
    };
    if exit_self.is_null() {
        crate::call_function(exit_func, &[exc_type, val, exc_tb])
    } else {
        crate::call_function(exit_func, &[exit_self, exc_type, val, exc_tb])
    }
}

/// Resolve the value half of CPython 3.14's fused `LOAD_SUPER_ATTR`.
///
/// The observable opcode still honours the `super` value loaded from globals.
/// For the exact builtin zero-argument case, pass the live frame explicitly to
/// PyPy's `_super_from_frame` equivalent instead of rediscovering it through
/// `ExecutionContext::gettopframe`: an inlined caller has its own red frame,
/// and that frame is what `_super_check` must read.  Keeping this as ordinary
/// interpreter source lets the translator generate the `_super_check` and
/// `W_Super.getattribute` call graph instead of teaching the tracer a second
/// implementation of either one.
pub fn load_super_attr_value_w(
    global_super: PyObjectRef,
    self_obj: PyObjectRef,
    cls: PyObjectRef,
    frame: *mut PyFrame,
    mut w_name: PyObjectRef,
    is_two_arg: bool,
    self_is_cell: bool,
    class_slot: isize,
) -> Result<PyObjectRef, PyError> {
    if !unsafe { pyre_object::is_str(w_name) } {
        return Err(PyError::runtime_error(format!(
            "LOAD_SUPER_ATTR requires a string name"
        )));
    }
    let exact_builtin_zero_arg =
        !is_two_arg && crate::builtins::is_builtin_super_type(global_super);
    let proxy = if is_two_arg {
        pyre_object::with_roots!(w_name =>
            crate::call::call_function_impl_result(global_super, &[cls, self_obj])
        )?
    } else if exact_builtin_zero_arg {
        pyre_object::with_roots!(w_name =>
            crate::builtins::builtin_super_from_frame_layout(frame, self_is_cell, class_slot)
        )?
    } else {
        pyre_object::with_roots!(w_name =>
            crate::call::call_function_impl_result(global_super, &[])
        )?
    };
    if exact_builtin_zero_arg {
        // The selected slot is PyPy's `W_Super.getattribute`, so enter that
        // ordinary interpreter body directly.  The generic `getattr` wrapper
        // would redispatch to the same slot after publishing a Rust slice with
        // `pin_roots`; individual pins are the translated shadow-stack
        // `setarrayitem`s and keep this generated call graph word-ABI-only.
        let _roots = pyre_object::gc_roots::push_roots();
        let operands = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(proxy);
        let _ = pyre_object::gc_roots::pin_root(w_name);
        return unsafe {
            crate::baseobjspace::super_getattribute_code_name(
                pyre_object::gc_roots::shadow_stack_get(operands),
                pyre_object::gc_roots::shadow_stack_get(operands + 1),
            )
        };
    }
    crate::baseobjspace::getattr(proxy, w_name)
}

/// Index-taking blackhole twin of [`load_super_attr_value_w`].
///
/// The opcode interpreter and generated JitCode carry PyPy's already-wrapped
/// `co_names_w[nameindex]`.  The standalone blackhole ABI still carries its
/// compact `(w_code, nameindex)` pair, so realize the same immutable name here
/// before entering the shared value path.
pub fn load_super_attr_value(
    global_super: PyObjectRef,
    self_obj: PyObjectRef,
    cls: PyObjectRef,
    frame: *mut PyFrame,
    w_code: PyObjectRef,
    name_idx: usize,
    is_two_arg: bool,
) -> Result<PyObjectRef, PyError> {
    let code = unsafe { &*(crate::w_code_get_ptr(w_code) as *const crate::CodeObject) };
    let Some(name) = code.names.get(name_idx) else {
        return Err(PyError::runtime_error(format!(
            "LOAD_SUPER_ATTR name index {name_idx} out of range"
        )));
    };
    // `w_code_getname_w_or_new` interns and can collect.  The code body is not
    // a GC object; the frame and the three objects the lookup still needs are.
    // Anchor before the intern and pass `live()` — `descriptor.py`
    // `_get_self_location` reads that frame afterwards.
    let frame_anchor = unsafe { FrameAnchor::from_raw(frame) };
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    let _ = roots.publish(&[global_super, self_obj, cls]);
    let w_name = unsafe { crate::pycode::w_code_getname_w_or_new(w_code, name_idx, name) };
    let (self_is_cell, class_slot) = crate::builtins::bare_super_frame_layout_words(code);
    load_super_attr_value_w(
        roots.get(base),
        roots.get(base + 1),
        roots.get(base + 2),
        frame_anchor.live(),
        w_name,
        is_two_arg,
        self_is_cell,
        class_slot,
    )
}

impl OpcodeStepExecutor for PyFrame {
    /// The generic default walks `small_int_constant` → `int_constant`
    /// → `w_int_new` and writes a translator-local `&INT_TYPE`.
    fn load_small_int(&mut self, value: i64) -> Result<(), PyError> {
        let w = w_small_int_const(value);
        self.push_value(w)
    }

    fn pop_top(&mut self) -> Result<(), PyError> {
        let _ = self.pop_value()?;
        self.failed_attr_after_stack_pop();
        Ok(())
    }

    /// SETUP_ANNOTATIONS — ensure `__annotations__` exists in the
    /// current locals namespace. PyPy: pyopcode.py SETUP_ANNOTATIONS
    /// (typeobject.py auto-fills the slot at class creation, but the
    /// pyre-equivalent flow runs the bytecode opcode and writes into
    /// the class_locals namespace just like CPython).
    fn setup_annotations(&mut self) -> Result<(), PyError> {
        // `if not self.space.finditem_str(w_locals, '__annotations__')`:
        // probe by item lookup, not membership — a custom mapping's
        // `__contains__` can disagree with `__getitem__`/KeyError.
        // The probe can run a mapping's `__getitem__`, and the key and the empty
        // dict are two more allocations; the mapping is read after every one of
        // them and the key after the dict.
        let roots = pyre_object::gc_roots::push_roots();
        let locals_slot = roots.base();
        let _ = roots.pin_root(self.get_or_create_w_locals());
        if crate::baseobjspace::finditem_str(roots.get(locals_slot), "__annotations__")?.is_none() {
            let key_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(unsafe { pyre_object::w_str_new("__annotations__") });
            let annotations_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(pyre_object::w_dict_new());
            crate::baseobjspace::setitem(
                roots.get(locals_slot),
                roots.get(key_slot),
                roots.get(annotations_slot),
            )?;
        }
        Ok(())
    }

    fn cleanup_throw(&mut self) -> Result<(), PyError> {
        let w_exc = self.pop_value()?;
        let mut err = unsafe { PyError::from_exc_object(w_exc) };
        let (stop, mut err) = err.matches_stop_iteration_keep();
        if !stop {
            // CPython 3.14 `CLEANUP_THROW` installs the existing exception and
            // jumps straight to `exception_unwind`; unlike the ordinary
            // opcode-error path it does not prepend another traceback entry.
            // This is the same explicit-reraise shape as PyPy's
            // `RaiseWithExplicitTraceback`, represented in pyre by
            // `attach_tb = false`.
            err.set_attach_tb(false);
            return Err(err);
        }

        self.pop_value()?;
        self.pop_value()?;
        let value =
            if !err.exc_object.is_null() && unsafe { pyre_object::is_exception(err.exc_object) } {
                crate::baseobjspace::getattr_str(err.exc_object, "value")
                    .unwrap_or_else(|_| pyre_object::w_none())
            } else {
                pyre_object::w_none()
            };
        // `(sub_iter, last_sent_val, exc_value -- none, value)`: the `END_SEND`
        // this jumps back to pops the pair, as it does SEND's `receiver, value`.
        self.push(pyre_object::w_none());
        self.push(value);
        Ok(())
    }

    /// WITH_EXCEPT_START — call __exit__ for the exceptional `with` exit.
    ///
    /// Stack layout the bytecode emits (bottom → top):
    ///   exit_func, exit_self, lasti, unused, val
    ///
    /// `val` (TOS) is the in-flight exception. LOAD_SPECIAL split the
    /// context manager's `__exit__` into `exit_func` (the function) and
    /// `exit_self` (the bound instance, or NULL). We call
    /// `exit_func(exit_self, type(val), val, val.__traceback__)` with
    /// `exit_self` prepended only when it is non-NULL, and push the result
    /// so the following TO_BOOL decides whether to suppress.
    fn with_except_start(&mut self) -> Result<(), PyError> {
        let depth = self.valuestackdepth;
        if depth < 5 {
            return Err(PyError::type_error(
                "WITH_EXCEPT_START requires five stack values",
            ));
        }
        // Indices first: a subscript evaluates its receiver before its index
        // expression, so arithmetic inside the brackets puts the subtraction's
        // overflow check between the `locals_cells_stack_w` read and its use.
        let (i_val, i_self, i_func) = (depth - 1, depth - 4, depth - 5);
        let val = locals_w!(self)[i_val];
        let exit_self = locals_w!(self)[i_self];
        let exit_func = locals_w!(self)[i_func];
        let anchor = FrameAnchor::new(self);
        let res = with_except_start_values(exit_func, exit_self, val);
        if res.is_null() {
            return Err(crate::call::take_call_error()
                .unwrap_or_else(|| crate::PyError::type_error("__exit__ failed")));
        }
        Self::push_anchored(&anchor, res)
    }

    // ── LoadCommonConstant ──
    fn load_common_constant(&mut self, cc: crate::bytecode::CommonConstant) -> Result<(), PyError> {
        // `LOAD_ASSERTION_ERROR` pushes the `AssertionError` class itself,
        // so `assert x` raises `AssertionError()` and `assert x, msg`
        // raises `AssertionError(msg)`.  The resolution is shared with the
        // JIT residual via `opcode_ops::load_common_constant_value`.
        let val = crate::opcode_ops::load_common_constant_value(cc);
        self.push(val);
        Ok(())
    }

    // ── PopJumpIfNone / PopJumpIfNotNone ──
    // CPython 3.13: replaces IS_OP + POP_JUMP_IF_TRUE/FALSE for None checks

    fn pop_jump_if_none(&mut self, target: usize) -> Result<(), PyError> {
        let val = self.pop();
        if unsafe { pyre_object::is_none(val) } || val.is_null() {
            self.set_last_instr_from_next_instr(target);
        }
        Ok(())
    }

    fn pop_jump_if_not_none(&mut self, target: usize) -> Result<(), PyError> {
        let val = self.pop();
        if !val.is_null() && !unsafe { pyre_object::is_none(val) } {
            self.set_last_instr_from_next_instr(target);
        }
        Ok(())
    }

    // ── Closures / cells ──

    /// PyPy: pyopcode.py LOAD_DEREF
    ///
    /// Reads cell/free variable. If the slot holds a cell object (from
    /// closure tuple via COPY_FREE_VARS), dereferences it. Otherwise
    /// reads the raw value (pyre's direct storage for cellvars).
    /// LOAD_DEREF — RustPython 3.13 uses unified index (same as LOAD_FAST).
    ///
    /// PyPy: pyopcode.py LOAD_DEREF → cell.get()
    /// If the slot holds a cell object, dereference it to get the value.
    fn load_deref(&mut self, idx: usize) -> Result<(), PyError> {
        let slot = locals_w!(self)[idx];
        let value = if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
            unsafe { pyre_object::w_cell_get(slot) }
        } else {
            slot
        };
        if value == PY_NULL {
            return Err(crate::pyframe::deref_unbound_error(self.code(), idx));
        }
        self.push(value);
        Ok(())
    }

    /// STORE_DEREF — unified index. Stores into cell if present.
    ///
    /// PyPy: pyopcode.py STORE_DEREF → cell.set(value)
    fn store_deref(&mut self, idx: usize) -> Result<(), PyError> {
        let value = self.pop();
        // CPython 3.14 compile.c keeps an explicit class-body `__class__`
        // assignment in the namespace (`STORE_NAME`) while the implicit cell
        // remains available for methods and is filled by `type.__new__`.
        // The pinned RustPython compiler emits STORE_DEREF for that assignment;
        // normalize its semantics at the opcode boundary.
        if crate::pyframe::class_scope_class_deref_is_name(self.code(), idx) {
            return self.store_name_value("__class__", crate::pyopcode::NO_NAMEINDEX, value);
        }
        let slot = locals_w!(self)[idx];
        if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
            unsafe { pyre_object::w_cell_set(slot, value) };
        } else {
            self.set_locals_w(idx, value);
        }
        Ok(())
    }

    /// LOAD_CLOSURE — unified index. Push cell object itself (not contents).
    ///
    /// PyPy: pyopcode.py LOAD_CLOSURE → push cell for closure capture.
    fn load_closure(&mut self, idx: usize) -> Result<(), PyError> {
        let cell = locals_w!(self)[idx];
        self.push(cell);
        Ok(())
    }

    /// MAKE_CELL — wrap the slot value in a Cell.
    ///
    /// CPython 3.13 / RustPython MAKE_CELL — create cell object in slot.
    /// Wraps the current value (PY_NULL if uninitialized) in a Cell.
    /// LoadFast on cell slots returns the cell object itself (needed for
    /// closure creation via BUILD_TUPLE + SET_FUNCTION_ATTRIBUTE).
    ///
    /// `initialize_frame_scopes` already installs an empty cell for every
    /// pure cellvar (a cellvar not sharing a locals slot).  A cellvar which
    /// does share a locals slot must always wrap that slot's raw value here,
    /// even when the Python argument itself happens to be a `cell` object.
    /// PyPy's `PyFrame.init_cells` has distinct argument and cell slots, so
    /// `cell.set(locals[argnum])` naturally preserves this extra level.
    /// Only the pre-installed pure-cellvar slot is already the cell which
    /// MAKE_CELL denotes (notably the implicit `__class__` cell).
    fn make_cell(&mut self, idx: usize) -> Result<(), PyError> {
        let current = locals_w!(self)[idx];
        let shares_local_slot = idx < self.code().varnames.len();
        if shares_local_slot || current.is_null() || !unsafe { pyre_object::is_cell(current) } {
            // pyframe.py `PyFrame.initialize_frame_scopes` `Cell(..., self.pycode.cell_families[i])` —
            // the cellvar this parameter slot was promoted for.
            let family = unsafe {
                crate::pycode::w_code_cell_family(self.pycode as pyre_object::PyObjectRef, idx)
            };
            let cell = pyre_object::w_cell_new(current, family);
            self.set_locals_w(idx, cell);
        }
        Ok(())
    }

    fn delete_deref(&mut self, idx: usize) -> Result<(), PyError> {
        if crate::pyframe::class_scope_class_deref_is_name(self.code(), idx) {
            return self.delete_name("__class__", crate::pyopcode::NO_NAMEINDEX);
        }
        // `pyopcode.py DELETE_DEREF`: fetch the cell, raise if empty, then
        // `cell.set(None)` — clear the cell *contents* (PY_NULL is the empty
        // marker), not the slot pointer that holds the cell.  The cell lives at
        // `locals_w!(self)[idx]`, the same slot `load_deref`/`store_deref` use.
        let slot = locals_w!(self)[idx];
        let is_cell = !slot.is_null() && unsafe { pyre_object::is_cell(slot) };
        let contents = if is_cell {
            unsafe { pyre_object::w_cell_get(slot) }
        } else {
            slot
        };
        if contents == PY_NULL {
            return Err(crate::pyframe::deref_unbound_error(self.code(), idx));
        }
        if is_cell {
            unsafe { pyre_object::w_cell_set(slot, PY_NULL) };
        } else {
            self.set_locals_w(idx, PY_NULL);
        }
        Ok(())
    }

    // ── Exception handling ──

    fn raise_varargs(&mut self, argc: usize) -> Result<(), PyError> {
        match argc {
            0 => {
                // Bare `raise` — re-raise current exception
                // PyPy: executioncontext.py sys_exc_info
                let exc = get_sys_exception();
                if exc.is_null() || unsafe { pyre_object::is_none(exc) } {
                    Err(PyError::runtime_error("No active exception to reraise"))
                } else if unsafe { pyre_object::is_exception(exc) } {
                    // RAISE_VARARGS(nbargs=0) re-raises the active exception via
                    // RaiseWithExplicitTraceback, i.e. without recording a fresh
                    // traceback for this frame.  Preserving the exception's existing
                    // w_traceback keeps its identity stable (mirroring the RERAISE
                    // opcode below), which the except* metadata check
                    // (_is_same_exception_metadata) relies on.
                    let mut err = unsafe { PyError::from_exc_object(exc) };
                    err.set_attach_tb(false);
                    Err(err)
                } else {
                    Err(PyError::runtime_error("No active exception to reraise"))
                }
            }
            1 => {
                // pyopcode.py:708-722 — cause=None, normalize exc.
                let w_value = self.pop();
                unsafe {
                    if crate::baseobjspace::exception_is_valid_obj_as_class_w(w_value) {
                        // The class is a nursery object only when it is a heap
                        // type. `instantiate_raised_class` allocates the
                        // instance (`malloc_fixedsize` → `collect_and_reserve`),
                        // so pin the class and re-read it afterwards.
                        let _roots = pyre_object::gc_roots::push_roots();
                        let value_slot = pyre_object::gc_roots::shadow_stack_len();
                        let _ = pyre_object::gc_roots::pin_root(w_value);
                        let result = instantiate_raised_class(
                            pyre_object::gc_roots::shadow_stack_get(value_slot),
                        )?;
                        // Re-read the class. `attach_raise_cause` does not use
                        // it; the read is what drops the pre-move local.
                        let _ = pyre_object::gc_roots::shadow_stack_get(value_slot);
                        attach_raise_cause(result, None)?;
                        Err(PyError::from_exc_object(result))
                    } else if pyre_object::is_exception(w_value) {
                        attach_raise_cause(w_value, None)?;
                        Err(PyError::from_exc_object(w_value))
                    } else {
                        Err(PyError::type_error(
                            "exceptions must derive from BaseException",
                        ))
                    }
                }
            }
            2 => {
                // pyopcode.py:704-722 — pop+normalize cause first, then exc.
                let raw_cause = self.pop();
                // `normalize_raise_cause` returns `Ok(null)` for a null/absent
                // cause; report absence as `None` so the value is never
                // `Some(null)` (mirrors the JIT paths call_jit.rs / trace_opcode.rs).
                let c = normalize_raise_cause(raw_cause)?;
                let cause = if c.w_cause.is_null() { None } else { Some(c) };
                let w_value = self.pop();
                unsafe {
                    if crate::baseobjspace::exception_is_valid_obj_as_class_w(w_value) {
                        // Root the class and the normalized `cause` across the
                        // instantiation. `instantiate_raised_class` allocates
                        // (`malloc_fixedsize` → `collect_and_reserve`), and both
                        // live only in this Rust local until `attach_raise_cause`
                        // reads them. Re-read each slot afterwards.
                        let _roots = pyre_object::gc_roots::push_roots();
                        let value_slot = pyre_object::gc_roots::shadow_stack_len();
                        let _ = pyre_object::gc_roots::pin_root(w_value);
                        let cause_slot = cause.as_ref().map(|c| {
                            let slot = pyre_object::gc_roots::shadow_stack_len();
                            let _ = pyre_object::gc_roots::pin_root(c.w_cause);
                            slot
                        });
                        let result = instantiate_raised_class(
                            pyre_object::gc_roots::shadow_stack_get(value_slot),
                        )?;
                        let _ = pyre_object::gc_roots::shadow_stack_get(value_slot);
                        let cause = cause.map(|mut c| {
                            if let Some(slot) = cause_slot {
                                c.w_cause = pyre_object::gc_roots::shadow_stack_get(slot);
                            }
                            c
                        });
                        attach_raise_cause(result, cause)?;
                        Err(PyError::from_exc_object(result))
                    } else if pyre_object::is_exception(w_value) {
                        attach_raise_cause(w_value, cause)?;
                        Err(PyError::from_exc_object(w_value))
                    } else {
                        Err(PyError::type_error(
                            "exceptions must derive from BaseException",
                        ))
                    }
                }
            }
            _ => Err(PyError::type_error("too many arguments for raise")),
        }
    }

    fn end_finally(&mut self) -> Result<(), PyError> {
        // Pop the exception or None from stack
        let _ = self.pop();
        Ok(())
    }

    // ── Import ──
    // PyPy: pyopcode.py IMPORT_NAME
    // Stack: [level, fromlist] → pops both, pushes module object.
    fn import_module(&mut self, name: &str) -> Result<PyObjectRef, PyError> {
        if let Some(m) = crate::importing::get_sys_module(name) {
            return Ok(m);
        }
        crate::importing::importhook(
            rustpython_wtf8::Wtf8::new(name),
            self.get_w_globals(),
            pyre_object::w_none(),
            0,
            crate::call::getexecutioncontext(),
        )
    }

    fn build_template_op(&mut self) -> Result<(), PyError> {
        // Stack: [strings, interpolations] (two tuples the compiler split).
        let interpolations = self.pop();
        let strings = self.pop();
        self.call_module_helper("_template", "_build_template", &[strings, interpolations])
    }

    fn build_interpolation_op(
        &mut self,
        conversion: u32,
        has_format_spec: bool,
    ) -> Result<(), PyError> {
        // Stack: [value, expression, format_spec?] — format_spec present only
        // when the oparg low bit is set, else it defaults to the empty string.
        let format = if has_format_spec {
            self.pop()
        } else {
            pyre_object::w_str_new("")
        };
        let expression = self.pop();
        let value = self.pop();
        let _interp_roots = pyre_object::gc_roots::push_roots();
        let triple = pyre_object::gc_roots::pin_roots(&[format, expression, value]);
        let format_slot = triple;
        let expression_slot = triple + 1;
        let value_slot = triple + 2;
        let conversion_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(conversion as i64));
        let anchor = FrameAnchor::new(self);
        let module = self.import_module("_template")?;
        let func = getattr_str(module, "_build_interpolation")?;
        let result = call_callable(
            self,
            func,
            &[
                pyre_object::gc_roots::shadow_stack_get(value_slot),
                pyre_object::gc_roots::shadow_stack_get(expression_slot),
                pyre_object::gc_roots::shadow_stack_get(conversion_slot),
                pyre_object::gc_roots::shadow_stack_get(format_slot),
            ],
        )?;
        Self::push_anchored(&anchor, result)
    }

    /// `pyopcode.py CALL_INTRINSIC_1`'s PEP 695 helpers. The trait default
    /// keeps the popped operand in a native local across the `_typing`
    /// import; a nursery operand does not survive that (`test.test_global`
    /// under a small nursery).
    fn typing_intrinsic_1(&mut self, helper: &str) -> Result<(), PyError> {
        let arg = self.pop_value()?;
        self.call_module_helper("_typing", helper, &[arg])
    }

    /// Two-operand variant: TOS is the second argument, TOS1 the first.
    fn typing_intrinsic_2(&mut self, helper: &str) -> Result<(), PyError> {
        let arg2 = self.pop_value()?;
        let arg1 = self.pop_value()?;
        self.call_module_helper("_typing", helper, &[arg1, arg2])
    }

    fn import_name(&mut self, name: &str, nameindex: usize) -> Result<(), PyError> {
        // `pyopcode.py IMPORT_NAME`: the popped fromlist / level stay
        // livevars across `getname_w` and `__import__`.  `intern_str_value`
        // (the no-table fallback of `w_code_getname_w_or_new`) and the
        // import itself both collect, so the copies must sit on the
        // shadow stack the way `build_interpolation_op` roots its pops.
        let _import_roots = pyre_object::gc_roots::push_roots();
        let fromlist_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(self.pop());
        let flag_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(self.pop());
        let anchor = FrameAnchor::new(self);
        let w_modulename = unsafe {
            crate::pycode::w_code_getname_w_or_new(self.pycode as PyObjectRef, nameindex, name)
        };
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_modulename);
        let w_obj = crate::importing::import_name(
            self,
            pyre_object::gc_roots::shadow_stack_get(name_slot),
            pyre_object::gc_roots::shadow_stack_get(fromlist_slot),
            pyre_object::gc_roots::shadow_stack_get(flag_slot),
        )?;
        Self::push_anchored(&anchor, w_obj)
    }

    // PyPy: pyopcode.py IMPORT_FROM
    // Stack: [module] → peek module, push getattr(module, name)
    fn import_from(&mut self, name: &str) -> Result<(), PyError> {
        let module = self.peek();
        let anchor = FrameAnchor::new(self);
        let attr = crate::importing::import_from(module, name)?;
        Self::push_anchored(&anchor, attr)
    }

    // ── ContainsOp (in / not in) ──
    // PyPy: pyopcode.py COMPARE_OP with 'in' / 'not in'

    fn contains_op(&mut self, invert: crate::bytecode::Invert) -> Result<(), PyError> {
        // `pyopcode.py CONTAINS_OP`: `w_2 = popvalue(); w_1 = popvalue();
        // space.contains_w(w_2, w_1)`. Exact-builtin membership does not
        // collect; `contains` pins only the subclass/override arm.
        let haystack = self.pop();
        let needle = self.pop();
        let anchor = FrameAnchor::new(self);
        let result = crate::baseobjspace::contains(haystack, needle)?;
        let inverted = match invert {
            crate::bytecode::Invert::No => result,
            crate::bytecode::Invert::Yes => !result,
        };
        Self::push_anchored(&anchor, pyre_object::w_bool_from(inverted))
    }

    // ── IsOp (is / is not) ──
    // PyPy: pyopcode.py COMPARE_OP with 'is' / 'is not'

    fn is_op(&mut self, invert: crate::bytecode::Invert) -> Result<(), PyError> {
        let b = self.pop();
        let a = self.pop();
        let invert = match invert {
            crate::bytecode::Invert::No => 0,
            crate::bytecode::Invert::Yes => 1,
        };
        self.push(crate::runtime_ops::is_op(a, b, invert));
        Ok(())
    }

    // ── ToBool ──
    // CPython 3.13: converts TOS to bool

    fn to_bool(&mut self) -> Result<(), PyError> {
        let val = self.pop();
        let anchor = FrameAnchor::new(self);
        let truth = crate::baseobjspace::is_true(val)?;
        Self::push_anchored(&anchor, pyre_object::w_bool_from(truth))
    }

    // ── DeleteSubscr ──

    fn delete_subscript(&mut self) -> Result<(), PyError> {
        let index = self.pop();
        let obj = self.pop();
        crate::baseobjspace::delitem(obj, index)?;
        Ok(())
    }

    // ── DeleteFast ──

    fn delete_fast(&mut self, idx: usize) -> Result<(), PyError> {
        if locals_w!(self)[idx].is_null() {
            let code = unsafe { &*crate::pyframe_get_pycode(self) };
            let name = if idx < code.varnames.len() {
                code.varnames[idx].as_str()
            } else {
                ""
            };
            return Err(PyError::unbound_local_error(format!(
                "cannot access local variable '{name}' where it is not associated with a value"
            )));
        }
        self.set_locals_w(idx, PY_NULL);
        Ok(())
    }

    // ── FormatSimple (str(TOS)) ──
    fn format_simple(&mut self) -> Result<(), PyError> {
        let val = self.pop();
        // `f'{x}'` → `PyObject_Format(x, NULL)`; a user `__format__` is
        // invoked with an empty spec, otherwise this is `str(value)`.
        let anchor = FrameAnchor::new(self);
        let s = crate::runtime_ops::format_value(val, pyre_object::PY_NULL)?;
        Self::push_anchored(&anchor, s)
    }

    // ── FormatWithSpec (format(TOS1, TOS)) ──
    fn format_with_spec(&mut self) -> Result<(), PyError> {
        let spec = self.pop();
        let val = self.pop();
        // `PyObject_Format(value, spec)` — dispatch to a user-defined
        // `__format__` when present, else apply the shared spec parser
        // (empty spec → `str(value)`).  `runtime_ops::format_value` keeps
        // f-string `{n:08.3f}` and `"{:08.3f}".format(n)` identical, and
        // reads a non-`str`/non-UTF-8 spec as empty rather than panicking.
        let anchor = FrameAnchor::new(self);
        let s = crate::runtime_ops::format_value(val, spec)?;
        Self::push_anchored(&anchor, s)
    }

    // ── ConvertValue (repr/str/ascii conversion) ──
    fn convert_value(&mut self, conv: crate::bytecode::ConvertValueOparg) -> Result<(), PyError> {
        let val = self.pop();
        // An instance dunder returns its str object (`DescrOperation.str`).
        // Every other shape is built as WTF-8, so a lone surrogate (a str,
        // or an exception whose single argument is a str) survives a Rust
        // `String`.  This is the path the `'%s' % x` rewrite takes.
        let code = crate::runtime_ops::convert_value_code(conv);
        let anchor = FrameAnchor::new(self);
        let converted = crate::runtime_ops::convert_value(val, code)?;
        Self::push_anchored(&anchor, converted)
    }

    // ── CopyFreeVars ──
    // CPython 3.13: copy n freevars from function closure to frame cell slots
    fn copy_free_vars(&mut self, _count: usize) -> Result<(), PyError> {
        // No-op — closure passing needs call-site integration
        // The closure tuple is on the Function, but COPY_FREE_VARS
        // runs inside the callee frame which doesn't have a reference to
        // the function object. Need to pass closure during frame creation.
        Ok(())
    }

    // ── SetFunctionAttribute ──
    /// CPython 3.13 SET_FUNCTION_ATTRIBUTE: pop attr, pop func, set, push func.
    /// Stack effect: (2) → (1)
    /// CPython 3.13 SET_FUNCTION_ATTRIBUTE: (attr, func -- func)
    /// attr = TOS1 (below), func = TOS (top).
    /// Pops both, sets attribute on func, pushes func back.
    fn set_function_attribute_with_flag(
        &mut self,
        flag: crate::bytecode::MakeFunctionFlag,
    ) -> Result<(), PyError> {
        use crate::bytecode::MakeFunctionFlag;
        let func = self.pop(); // TOS = function
        let attr = self.pop(); // TOS1 = attribute value (closure tuple etc.)
        match flag {
            MakeFunctionFlag::Closure => unsafe {
                crate::function_set_closure(func, attr);
            },
            MakeFunctionFlag::Defaults => unsafe {
                crate::function_set_defaults(func, attr);
            },
            MakeFunctionFlag::KwOnlyDefaults => unsafe {
                crate::function::function_set_kwdefaults_from_definition(func, attr);
            },
            MakeFunctionFlag::Annotations => {
                // `pypy/interpreter/function.py:553-559
                // fset_func_annotations` — MAKE_FUNCTION ANNOTATIONS
                // (oparg.rs:352 `MakeFunctionFlag::Annotations = 2`)
                // carries the eager annotations dict.  PyPy stores it
                // on `self.w_ann`; pyre stamps the typed
                // `Function.w_ann` slot directly so
                // `f.__annotations__ is f.__annotations__` holds
                // (the getattr arm reads the same field) instead of
                // routing through a side table.
                unsafe { crate::function::function_set_annotations(func, attr) };
            }
            MakeFunctionFlag::Annotate => {
                // PEP 649: lazy annotations.  `attr` is the
                // `__annotate__` callable the `__annotations__` getter
                // evaluates with `format=1` when the runtime dict is
                // requested; stored on the function's typed
                // `w_annotate` slot (CPython 3.14 `func_annotate`).
                // CPython 3.14's annotations compile scope is qualified as
                // `<target>.__annotate__`.  The external compiler currently
                // supplies only the enclosing scope, so finish the compiler
                // metadata at the bytecode operation that binds the annotate
                // function to its target.
                let roots = pyre_object::gc_roots::push_roots();
                let func_slot = roots.base();
                let _ = roots.pin_root(func);
                let attr_slot = func_slot + 1;
                let _ = roots.pin_root(attr);
                let mut qualname =
                    unsafe { crate::function::function_get_qualname(roots.get(func_slot)) };
                qualname.push_str(".__annotate__");
                let w_qualname = pyre_object::w_str_from_wtf8_managed(qualname);
                unsafe {
                    crate::function::function_set_qualname(roots.get(attr_slot), w_qualname);
                    crate::function::function_set_annotate_unchecked(
                        roots.get(func_slot),
                        roots.get(attr_slot),
                    );
                }
            }
            // `MakeFunctionFlag::TypeParams` names the tuple of
            // TypeVar / ParamSpec / TypeVarTuple bound by a PEP 695 generic
            // function, but the codegen never sets that bit: the parameters
            // reach the function through
            // `CALL_INTRINSIC_2 INTRINSIC_SET_FUNCTION_TYPE_PARAMS`, whose
            // `set_function_typeparams` writes `Function.w_typeparams`.
            // Accept the operand rather than panic on a bytecode stream that
            // does carry it.
            MakeFunctionFlag::TypeParams => {}
        }
        self.push(func);
        Ok(())
    }

    // ── PushExcInfo ──
    // PyPy: executioncontext.py enter_frame / normalize_exception
    fn push_exc_info(&mut self) -> Result<(), PyError> {
        let exc = self.pop();
        // Save previous exception, set current.  Routed through the
        // named TLS accessors (not a raw `CURRENT_EXCEPTION.with`
        // closure) so the codewriter sees two residual-callable leaves
        // with registered fnaddrs instead of an unresolvable
        // `LocalKey::with` monomorphization — the same per-thread slot
        // the compiled trace reads/writes through
        // `get_current_exception_fn` / `set_current_exception_fn`.
        let w_prev = current_exception_or_none();
        set_current_exception(exc);
        // `PUSH_EXC_INFO` transfers ownership from the propagating `PyError`
        // to the execution context.  Its `sys_exc_value` slot and raw
        // exception children are walked above, so the temporary propagation
        // root must no longer retain a completed handler's traceback.
        set_in_flight_exception(pyre_object::PY_NULL);
        // Push "previous exception" for later restore
        self.push(w_prev);
        // Push the exception value back
        self.push(exc);
        Ok(())
    }

    // ── CheckExcMatch ──
    // TOS = exception type to match, TOS1 = caught exception
    // Pops type, peeks exc, pushes bool result
    fn check_exc_match(&mut self) -> Result<(), PyError> {
        let exc_type = self.pop();
        let exc_value = self.peek();
        // pyopcode.py cmp_exc_match split:
        //   :1034-1039 — `validate_check_exc_match_class(exc_type)?`
        //                raises TypeError(CANNOT_CATCH_MSG) for invalid
        //                except targets (`raise oefmt(...)` upstream).
        //   :1040     — `check_exc_match_against(exc_value, exc_type)`
        //                computes the match boolean.
        // PyPy keeps both in a single `@jit.unroll_safe cmp_exc_match`;
        // the JIT inlines and the `raise` becomes a guard. Pyre splits
        // so the bool-returning hot helper keeps a 1-register ABI for
        // residual JIT calls; the validity gate runs in this BC handler
        // (outside the residual call path) and lifts to `?` propagation.
        validate_check_exc_match_class(exc_type)?;
        let matched = check_exc_match_against(exc_value, exc_type);
        self.push(pyre_object::w_bool_from(matched));
        Ok(())
    }

    fn check_eg_match(&mut self) -> Result<(), PyError> {
        // `pyopcode.py CHECK_EG_MATCH`: pop the type, peek the exception so it
        // stays on the value stack for the whole match, then either push None
        // or `settopvalue(rest)` + push the match.
        // The exception stays on the value stack, but the popped type is a
        // livevar of this frame across the validation, which runs
        // `__subclasscheck__`.
        let type_roots = pyre_object::gc_roots::push_roots();
        let type_slot = type_roots.base();
        let _ = type_roots.pin_root(self.pop());
        let anchor = FrameAnchor::new(self);
        validate_check_eg_match_class(type_roots.get(type_slot))?;
        let exc_value = unsafe { &mut *anchor.live() }.peek();
        let (matching, rest, wrapped_naked) = if unsafe { pyre_object::is_none(exc_value) } {
            (pyre_object::w_none(), pyre_object::w_none(), false)
        } else {
            crate::builtins::exception_group_match(exc_value, type_roots.get(type_slot))?
        };
        let _roots = pyre_object::gc_roots::push_roots();
        let result_base = pyre_object::gc_roots::pin_roots(&[matching, rest]);
        let matching = || pyre_object::gc_roots::shadow_stack_get(result_base);
        let rest = || pyre_object::gc_roots::shadow_stack_get(result_base + 1);
        let frame = unsafe { &mut *anchor.live() };
        if unsafe { pyre_object::is_none(matching()) } {
            frame.push(matching());
        } else {
            frame.settopvalue(rest(), 0);
            frame.push(matching());
            set_current_exception(matching());
        }
        if wrapped_naked {
            // The wrapper this opcode just built has an empty traceback, and the
            // exception inside it keeps its own chain, so nothing else ever names
            // this frame on the group.  A group re-raised out of the `except*`
            // clause reports only the frames it passes through afterwards without
            // the entry made here.  Both values are on the value stack already,
            // so the node allocation cannot collect them.
            let frame = unsafe { &mut *anchor.live() };
            unsafe {
                crate::pytraceback::record_application_traceback(
                    matching(),
                    frame as *mut PyFrame,
                    frame.last_instr as i64,
                );
            }
            // `record_application_traceback` publishes a propagating error in
            // `IN_FLIGHT_EXCEPTION`. This wrapper is no longer propagating:
            // CHECK_EG_MATCH has transferred it to the value stack and
            // `sys_exc_value`, exactly as `push_exc_info` does for an ordinary
            // caught exception. Leaving the side channel set makes the
            // temporary group a permanent collector root after POP_EXCEPT.
            set_in_flight_exception(pyre_object::PY_NULL);
        }
        Ok(())
    }

    fn prep_reraise_star(
        &mut self,
        orig: Self::Value,
        exceptions: Self::Value,
    ) -> Result<Self::Value, PyError> {
        crate::builtins::exception_group_prep_reraise_star(orig, exceptions)
    }

    // ── PopExcept ──
    fn pop_except(&mut self) -> Result<(), PyError> {
        // Restore previous exc_info from stack.  Named TLS accessor for
        // the same codewriter-resolvability reason as `push_exc_info`.
        let w_prev_exc = self.pop();
        restore_exc_info(w_prev_exc);
        self.failed_attr_after_pop_except();
        Ok(())
    }

    // ── Reraise ──
    // `pypy/interpreter/pyopcode.py RERAISE`.
    //
    // `RERAISE` reads the operand `PUSH_EXC_INFO` left on the value stack, so
    // this opcode only sees a non-exception when the frame's stack slot itself
    // went bad between the two. That is unreachable for a correct interpreter,
    // and the intermittent `test_asyncio` failures reach it — hence the
    // diagnostic below rather than a bare `TypeError`.
    fn reraise(&mut self, oparg: u32) -> Result<(), PyError> {
        // pyopcode.py:1357-1363
        let reraise_lasti: i32 = if oparg != 0 {
            // pyopcode.py:1361 — self.space.int_w(self.peekvalue(oparg))
            crate::baseobjspace::int_w(self.peekvalue(oparg as usize))? as i32
        } else {
            -1
        };
        // pyopcode.py:1364 — w_exc = self.popvalue()
        let w_exc = self.popvalue();
        // pyopcode.py:1367 — w_value = space.interp_w(W_BaseException, w_exc)
        if w_exc.is_null() || !unsafe { pyre_object::is_exception(w_exc) } {
            if reraise_diag_enabled() {
                // Snapshot the absolute slots, not a deref of TOS. A depth
                // miscount leaves the exception one slot away; a cleared slot
                // is NULL; a non-object bit pattern is the address itself.
                let depth = self.valuestackdepth;
                let stack_base = self.stack_base();
                let slots: Vec<(usize, PyObjectRef)> = (0..depth)
                    .map(|index| (index, locals_w!(self)[index]))
                    .collect();
                let code_ptr = unsafe { crate::pyframe::pyframe_get_pycode(self) };
                let code_name = if code_ptr.is_null() {
                    "?"
                } else {
                    unsafe { (*code_ptr).obj_name.as_str() }
                };
                let opcode = if self.last_instr >= 0 {
                    crate::pyopcode::decode_instruction_at(self.code(), self.last_instr as usize)
                        .map(|(instruction, _)| u8::from(instruction).to_string())
                        .unwrap_or_else(|| "?".to_string())
                } else {
                    "pre".to_string()
                };
                reraise_bad_operand_diag(
                    w_exc,
                    oparg,
                    code_name,
                    self.last_instr,
                    &opcode,
                    stack_base,
                    depth,
                    &slots,
                );
            }
            return Err(PyError::type_error(
                "exception must derive from BaseException",
            ));
        }
        // pyopcode.py:1368-1369 — w_type = space.type(w_exc); operr = OperationError(w_type, w_exc, w_value.w_traceback)
        let mut err = unsafe { PyError::from_exc_object(w_exc) };
        // pyopcode.py — raise RaiseWithExplicitTraceback(operr, reraise_lasti)
        err.set_attach_tb(false);
        err.set_reraise_lasti(reraise_lasti);
        Err(err)
    }

    // ── LoadFromDictOrGlobals ──
    // CPython 3.13: LOAD_FROM_DICT_OR_GLOBALS — try TOS dict first, then globals
    fn load_from_dict_or_globals(&mut self, name: &str, nameindex: usize) -> Result<(), PyError> {
        // `w_code_getname_w_or_new` realizes the `co_names_w` slot on its
        // first read, so the popped mapping crosses an allocation.
        let roots = pyre_object::gc_roots::push_roots();
        let mapping_slot = roots.base();
        let _ = roots.pin_root(self.pop());
        let key = unsafe {
            crate::pycode::w_code_getname_w_or_new(self.pycode as PyObjectRef, nameindex, name)
        };
        let anchor = FrameAnchor::new(self);
        match crate::baseobjspace::getitem(roots.get(mapping_slot), key) {
            Ok(value) => {
                // The mapping may be the raw type namespace.  `setdictvalue`
                // parks an `ObjectMutableCell` there; the loaded name is the
                // payload.
                let value = unsafe { pyre_object::celldict::unwrap_cell(value) };
                return Self::push_anchored(&anchor, value);
            }
            Err(err) if matches!(err.kind, PyErrorKind::KeyError) => {}
            Err(err) => return Err(err),
        }

        let value = unsafe { &mut *anchor.live() }.load_global_value(name, nameindex)?;
        Self::push_anchored(&anchor, value)
    }

    // ── LoadFromDictOrDeref ──
    // CPython 3.13: LOAD_FROM_DICT_OR_DEREF — used by the PEP 695 type-param
    // scope.  Pop the namespace mapping (TOS), try `mapping[name]`, then fall
    // back to the cell / free variable at `idx`.
    fn load_from_dict_or_deref(&mut self, idx: usize, name: &str) -> Result<(), PyError> {
        // A localsplus name has no `co_names_w` slot to realize into, so nothing
        // bounds how often this runs; interning is what keeps an immortal
        // string per execution from being an immortal string per execution.
        // The popped mapping crosses that allocation.
        let roots = pyre_object::gc_roots::push_roots();
        let mapping_slot = roots.base();
        let _ = roots.pin_root(self.pop());
        let key = pyre_object::unicodeobject::intern_str_value(name);
        let anchor = FrameAnchor::new(self);
        match crate::baseobjspace::getitem(roots.get(mapping_slot), key) {
            Ok(value) => {
                // The mapping is the raw type namespace (`__classdictcell__`).
                // An `ObjectMutableCell` parked by `setdictvalue` must not
                // surface as the bound.
                let value = unsafe { pyre_object::celldict::unwrap_cell(value) };
                return Self::push_anchored(&anchor, value);
            }
            Err(err) if matches!(err.kind, PyErrorKind::KeyError) => {}
            Err(err) => return Err(err),
        }
        // `getitem` may have run a `__getitem__`; everything below reads the
        // frame's own code object, globals and cells.
        let self_ = unsafe { &mut *anchor.live() };
        // CPython 3.14 addresses the outer `__class__` freevar in the
        // class-cell collision. The pinned compiler encodes the implicit
        // method cell instead; if no outer cell exists, it emits this opcode
        // where CPython uses LOAD_NAME, so preserve globals/builtins fallback.
        let deref_idx = if crate::pyframe::class_scope_class_deref_is_name(self_.code(), idx) {
            match crate::pyframe::class_scope_outer_class_freevar(self_.code()) {
                Some(free_idx) => free_idx,
                None => {
                    // This deref name has no `co_names` index, so do the
                    // uncached LOAD_NAME fallback directly rather than
                    // corrupting an unrelated per-code LOAD_GLOBAL cache slot.
                    let w_globals = self_.get_w_globals();
                    if !w_globals.is_null()
                        && let Some(value) = unsafe {
                            pyre_object::dictmultiobject::w_dict_getitem_str(w_globals, name)
                        }
                    {
                        return Self::push_anchored(&anchor, value);
                    }
                    // `self.get_builtin()`, not the `builtin` field — see
                    // `load_global_value`.
                    let w_builtin = self_.get_builtin();
                    if !w_builtin.is_null() && unsafe { pyre_object::is_module(w_builtin) } {
                        let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
                        if !w_dict.is_null()
                            && let Some(value) = crate::baseobjspace::finditem_str(w_dict, name)?
                        {
                            return Self::push_anchored(&anchor, value);
                        }
                    }
                    return Err(PyError::oefmt_name_error_text(
                        format!("name '{name}' is not defined"),
                        name,
                    ));
                }
            }
        } else {
            idx
        };
        let self_ = unsafe { &mut *anchor.live() };
        let slot = locals_w!(self_)[deref_idx];
        let value = if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
            unsafe { pyre_object::w_cell_get(slot) }
        } else {
            slot
        };
        if value == PY_NULL {
            return Err(crate::pyframe::deref_unbound_error(self_.code(), deref_idx));
        }
        Self::push_anchored(&anchor, value)
    }

    // ── GetLen ──
    fn get_len(&mut self, obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
        let len = crate::baseobjspace::len(obj)?;
        Ok(len)
    }

    // ── Pattern matching (PEP 634) ──
    // Each opcode is the stack shuffle around an `opcode_ops` value-level
    // helper; the JIT residuals (`bh_match_*_fn`) call the same helpers.
    fn match_mapping(&mut self) -> Result<(), PyError> {
        let subject = PyFrame::peek_at(self, 0);
        self.push(crate::opcode_ops::match_mapping_value(subject));
        Ok(())
    }

    fn match_sequence(&mut self) -> Result<(), PyError> {
        let subject = PyFrame::peek_at(self, 0);
        self.push(crate::opcode_ops::match_sequence_value(subject));
        Ok(())
    }

    // MATCH_KEYS: STACK[-1] = keys tuple, STACK[-2] = subject (neither popped).
    /// `pyopcode.py MATCH_KEYS` is `@jit.unroll_safe`.
    #[majit_macros::unroll_safe]
    fn match_keys(&mut self) -> Result<(), PyError> {
        let keys = PyFrame::peek_at(self, 0);
        let subject = PyFrame::peek_at(self, 1);
        let anchor = FrameAnchor::new(self);
        let result = crate::opcode_ops::match_keys_value(subject, keys)?;
        Self::push_anchored(&anchor, result)
    }

    // MATCH_CLASS count: STACK[-1] = keyword attr-name tuple, STACK[-2] = class,
    // STACK[-3] = subject (all popped). Push the extracted-attrs tuple on a
    // match, else None. `count` is the number of positional sub-patterns.
    /// `pyopcode.py _match_class` is `@jit.unroll_safe`.
    #[majit_macros::unroll_safe]
    fn match_class(&mut self, count: usize) -> Result<(), PyError> {
        let kwd_attrs = self.pop();
        let cls = self.pop();
        let subject = self.pop();
        let anchor = FrameAnchor::new(self);
        let result = crate::opcode_ops::match_class_value(subject, cls, kwd_attrs, count)?;
        Self::push_anchored(&anchor, result)
    }

    // ── LoadFastAndClear (comprehension scope) ──
    fn load_fast_and_clear(&mut self, idx: usize) -> Result<(), PyError> {
        let val = locals_w!(self)[idx];
        self.push(val);
        self.set_locals_w(idx, PY_NULL);
        Ok(())
    }

    // ── BuildSet ──
    /// `pyopcode.py BUILD_SET` is `@jit.unroll_safe`.
    #[majit_macros::unroll_safe]
    fn build_set(&mut self, count: usize) -> Result<(), PyError> {
        // Build as a set-like object backed by __data__ dict.
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            items.push(self.pop());
        }
        items.reverse();
        let anchor = FrameAnchor::new(self);
        let set_obj = crate::builtins::builtin_set_from_items(&items)?;
        Self::push_anchored(&anchor, set_obj)
    }

    // ── DictUpdate ──
    // [3.14-spec] `inst(DICT_UPDATE)` runs `PyDict_Update` and rewrites an
    // `AttributeError` it surfaced to "'<T>' object is not a mapping", so a
    // user `keys()` raising one reads the same way.  `PyFrame.DICT_UPDATE`
    // gates `space.ismapping_w` ahead of the update instead and lets such an
    // error through.  `dict_update_value` is the wrapper that draws the line
    // where 3.14 does.
    fn dict_update(&mut self, i: usize) -> Result<(), PyError> {
        let source = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        crate::opcode_ops::dict_update_value(dict, source)
    }

    // ── DictMerge ──
    // pypy/interpreter/pyopcode.py DICT_MERGE → _dict_merge
    // (pyopcode.py).
    fn dict_merge(&mut self, i: usize) -> Result<(), PyError> {
        let source = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        // pyopcode.py:1514 — callable = peekvalue(oparg + 2)
        // Stack after pop: [..., callable, NULL, args_tuple, dict]
        let w_callable = if self.valuestackdepth > i + 2 {
            PyFrame::peek_at(self, i + 2)
        } else {
            pyre_object::PY_NULL
        };
        crate::opcode_ops::dict_merge_value(dict, source, w_callable)
    }

    // ── MapAdd ──
    // PyPy: STORE_MAP/MAP_ADD; CPython: MAP_ADD
    // dict = STACK[-i-2]; dict[TOS1] = TOS; pop key+value
    fn map_add(&mut self, i: usize) -> Result<(), PyError> {
        let value = self.pop();
        let key = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        crate::opcode_ops::map_add_value(dict, key, value)
    }

    // ── SetAdd ──
    // PyPy: SET_ADD; CPython: SET_ADD
    // set = STACK[-i]; set.add(TOS); pop value
    fn set_add(&mut self, i: usize) -> Result<(), PyError> {
        let value = self.pop();
        let set = PyFrame::peek_at(self, i - 1);
        crate::opcode_ops::set_add_value(set, value)
    }

    // ── none_value ──
    fn none_value(&mut self) -> Result<PyObjectRef, PyError> {
        Ok(pyre_object::w_none())
    }

    // ── unary_positive ──
    // PyPy: UNARY_POSITIVE → space.pos(w_value)
    fn unary_positive(&mut self, val: PyObjectRef) -> Result<PyObjectRef, PyError> {
        crate::opcode_ops::unary_positive_value(val)
    }

    // ── list_to_tuple ──
    // PyPy intrinsic: convert list to tuple (used in star unpacking).
    fn list_to_tuple(&mut self, val: PyObjectRef) -> Result<PyObjectRef, PyError> {
        crate::opcode_ops::list_to_tuple_value(val)
    }

    fn async_gen_wrap(&mut self, val: PyObjectRef) -> Result<PyObjectRef, PyError> {
        Ok(pyre_object::generator::w_async_gen_value_wrapper_new(val))
    }

    // ── print_expr ──
    // PRINT_EXPR → sys.displayhook(value). Routing through the live hook lets
    // a rebound displayhook (doctest, IDLE) and a redirected sys.stdout take
    // effect instead of writing straight to the native stream.
    fn print_expr(&mut self, val: PyObjectRef) -> Result<(), PyError> {
        if let Some(sys_mod) = crate::importing::get_interpreter_sys_module() {
            // The opcode popped the value, so it is a livevar of the native
            // frame across the attribute lookup, which runs `__getattr__`.
            let roots = pyre_object::gc_roots::push_roots();
            let val_slot = roots.base();
            let _ = roots.pin_root(val);
            match crate::baseobjspace::getattr_str(sys_mod, "displayhook") {
                Ok(hook) => {
                    let r = crate::call_function(hook, &[roots.get(val_slot)]);
                    if r.is_null() {
                        return Err(crate::call::take_call_error().unwrap_or_else(|| {
                            PyError::runtime_error("displayhook raised an exception")
                        }));
                    }
                    return Ok(());
                }
                Err(e) if e.kind == PyErrorKind::AttributeError => {
                    return Err(PyError::runtime_error("lost sys.displayhook"));
                }
                Err(e) => return Err(e),
            }
        }
        // No `sys` yet (early bootstrap) — native repr print.
        if !unsafe { pyre_object::is_none(val) } {
            let s = unsafe { crate::display::py_repr_wtf8(val)? };
            crate::host_seam::emit_stdout(crate::display::wtf8_format!(s, "\n").as_bytes());
        }
        Ok(())
    }

    // ── delete_name ──
    // pypy/interpreter/pyopcode.py DELETE_NAME — delete from w_locals; KeyError → NameError.
    fn delete_name(&mut self, name: &str, nameindex: usize) -> Result<(), PyError> {
        // `space.delitem(w_locals, w_name)`; at module scope `w_locals` is the
        // globals dict, so a module DELETE_NAME routes through the canonical
        // W_DictObject too.  KeyError → NameError.
        // `w_code_getname_w_or_new` realizes the `co_names_w` slot on its first
        // read, so the mapping crosses an allocation before the delete.
        let roots = pyre_object::gc_roots::push_roots();
        let locals_slot = roots.base();
        let _ = roots.pin_root(self.get_or_create_w_locals());
        let key = unsafe {
            crate::pycode::w_code_getname_w_or_new(self.pycode as PyObjectRef, nameindex, name)
        };
        crate::baseobjspace::delitem(roots.get(locals_slot), key).map_err(|err| {
            if matches!(err.kind, PyErrorKind::KeyError) {
                PyError::oefmt_name_error_text(format!("name '{name}' is not defined"), name)
            } else {
                err
            }
        })?;
        Ok(())
    }

    // ── delete_global ──
    // pypy/interpreter/pyopcode.py DELETE_GLOBAL —
    //   `self.space.delitem(self.get_w_globals_storage(), w_varname)`.
    // CPython/PyPy dict deletion uses the dict's intrinsic strategy and does
    // not invoke a dict subclass's Python-level __delitem__.  Resolve pyre's
    // composed dict-subclass backing first, then use that same strategy path.
    fn delete_global(&mut self, name: &str) -> Result<(), PyError> {
        let w_globals = self.get_w_globals();
        let backing = unsafe { crate::type_methods::resolve_dict_backing(w_globals) };
        let found = !backing.is_null() && unsafe { pyre_object::w_dict_delitem_str(backing, name) };
        if !found {
            return Err(PyError::key_error(format!("'{name}'")));
        }
        Ok(())
    }

    // ── import_star ──
    // IMPORT_STAR — merge the module's public names into the locals
    // mapping (class body / exec-with-locals), not globals:
    //     w_locals = self.getdictscope()
    //     import_all_from(self.space, w_module, w_locals)
    //     self.setdictscope(w_locals)
    // `getdictscope` runs fast2locals so the mapping reflects the live
    // fast locals; `import_all_from_w` lands each `from module import *`
    // entry via `space.setitem(w_locals, name, value)` rather than the
    // `*mut DictStorage` fast path; `setdictscope` runs locals2fast to
    // write the merged mapping back into the frame's fast locals.
    fn import_star(&mut self) -> Result<(), PyError> {
        // `getdictscope` runs fast2locals and the merge lands each name
        // through `setitem`, so the popped module and the mapping are
        // livevars of this frame across both, and the frame itself crosses
        // the merge.
        let roots = pyre_object::gc_roots::push_roots();
        let module_slot = roots.base();
        let _ = roots.pin_root(self.pop());
        let anchor = FrameAnchor::new(self);
        let locals_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(self.getdictscope()?);
        crate::importing::import_all_from_w(roots.get(module_slot), roots.get(locals_slot))?;
        unsafe { &mut *anchor.live() }.setdictscope(roots.get(locals_slot))?;
        Ok(())
    }

    // ── load_build_class ──
    // PyPy pyopcode.py LOAD_BUILD_CLASS reads
    // `self.get_builtin().getdictvalue('__build_class__')`.  Python 3.14
    // reports a NameError when the selected builtin mapping has no entry.
    fn load_build_class(&mut self) -> Result<(), PyError> {
        let anchor = FrameAnchor::new(self);
        let bc = self.load_build_class_value()?;
        Self::push_anchored(&anchor, bc)
    }

    fn load_build_class_value(&mut self) -> Result<PyObjectRef, PyError> {
        let w_builtin = self.get_builtin();
        let bc = if !w_builtin.is_null() && unsafe { pyre_object::is_module(w_builtin) } {
            let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
            if w_dict.is_null() {
                None
            } else {
                crate::baseobjspace::finditem_str(w_dict, "__build_class__")?
            }
        } else {
            None
        };
        let Some(bc) = bc else {
            return Err(PyError::oefmt_name_error_text(
                "__build_class__ not found",
                "__build_class__",
            ));
        };
        Ok(bc)
    }

    // ── yield from / send ──
    fn get_yield_from_iter(&mut self) -> Result<(), PyError> {
        let iterable = self.pop();
        let anchor = FrameAnchor::new(self);
        // CPython 3.14 `GET_YIELD_FROM_ITER` / PyPy's coroutine-aware
        // `YIELD_FROM`: exact generators already are their iterator.  A
        // native coroutine is also sent to directly, but only when the
        // current frame is itself a coroutine or was marked by
        // `types.coroutine` with CO_ITERABLE_COROUTINE.  Calling ordinary
        // `iter()` here loses both halves of that distinction because native
        // coroutine objects intentionally expose no public `__iter__`.
        let iter = unsafe {
            if pyre_object::generator::is_coroutine(iterable) {
                let flags = self.code().flags;
                if !flags
                    .intersects(crate::CodeFlags::COROUTINE | crate::CodeFlags::ITERABLE_COROUTINE)
                {
                    return Err(PyError::type_error(
                        "cannot 'yield from' a coroutine object in a non-coroutine generator",
                    ));
                }
                iterable
            } else if pyre_object::generator::is_generator(iterable) {
                iterable
            } else {
                crate::baseobjspace::iter(iterable)?
            }
        };
        Self::push_anchored(&anchor, iter)
    }

    fn send_value(&mut self, target: usize) -> Result<(), PyError> {
        let value = self.pop();
        let iter = self.peek();
        let anchor = FrameAnchor::new(self);
        // `next_yield_from` keeps the awaitable live across `send_ex` /
        // `space.next`; both run Python and relocate a nursery-born
        // coroutine.  Reload it before the type check and before storing
        // `w_yielding_from`.
        let _roots = pyre_object::gc_roots::push_roots();
        let iter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(iter);
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(value);
        let iter = pyre_object::gc_roots::shadow_stack_get(iter_slot);
        let value = pyre_object::gc_roots::shadow_stack_get(value_slot);
        let result = if unsafe { pyre_object::is_none(value) } {
            // generator.py / pyopcode.py `next_yield_from`: coroutine
            // objects are not public iterators, but the interpreter's SEND
            // machinery resumes both GeneratorIterator and Coroutine through
            // their shared `send_ex(None)` path.
            if unsafe { pyre_object::generator::is_generator_or_coroutine(iter) } {
                crate::baseobjspace::generator_next_method(&[
                    pyre_object::gc_roots::shadow_stack_get(iter_slot),
                ])
            } else {
                crate::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(iter_slot))
            }
        } else {
            let send = crate::baseobjspace::getattr_str(
                pyre_object::gc_roots::shadow_stack_get(iter_slot),
                "send",
            )?;
            crate::call::call_function_impl_result(
                send,
                &[pyre_object::gc_roots::shadow_stack_get(value_slot)],
            )
        };
        let iter = pyre_object::gc_roots::shadow_stack_get(iter_slot);
        match result {
            Ok(result) => {
                // `send_ex` / `space.next` may collect; pin the yielded value
                // before the `w_yielding_from` store, which is itself a
                // write-barrier and must not observe a pre-move local.
                let result_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(result);
                let frame = unsafe { &mut *anchor.live() };
                frame.w_yielding_from = iter;
                if pyre_object::gc_hook::try_gc_owns_object(
                    frame as *mut PyFrame as pyre_object::gc_hook::GCREF,
                ) {
                    pyre_object::gc_hook::try_gc_write_barrier(
                        frame as *mut PyFrame as pyre_object::gc_hook::GCREF,
                    );
                }
                Self::push_anchored(
                    &anchor,
                    pyre_object::gc_roots::shadow_stack_get(result_slot),
                )
            }
            Err(e) => {
                let (stop, e) = e.matches_stop_iteration_keep();
                if stop {
                    let frame = unsafe { &mut *anchor.live() };
                    if std::ptr::eq(frame.w_yielding_from, iter) {
                        frame.w_yielding_from = pyre_object::PY_NULL;
                    }
                    // `pypy/interpreter/pyopcode.py next_yield_from`:
                    //     try:
                    //         w_stop_value = space.getattr(e.get_w_value(space),
                    //                                      space.newtext("value"))
                    //     except OperationError as e:
                    //         if not e.match(space, space.w_AttributeError):
                    //             raise
                    //         w_stop_value = space.w_None
                    //     self.pushvalue(w_stop_value)
                    //
                    // CPython 3.13 emits SEND with an EOI target; pyre's
                    // dispatch lands here on StopIteration and must surface
                    // the exception's `.value` as the yield-from result so
                    // `val = yield from inner()` captures `inner`'s return.
                    let value = if !e.exc_object.is_null()
                        && unsafe { pyre_object::is_exception(e.exc_object) }
                    {
                        crate::baseobjspace::getattr_str(e.exc_object, "value")
                            .unwrap_or_else(|_| pyre_object::w_none())
                    } else {
                        pyre_object::w_none()
                    };
                    Self::push_anchored(&anchor, value)?;
                    unsafe { &mut *anchor.live() }.set_last_instr_from_next_instr(target);
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }
    }

    fn end_send(&mut self) -> Result<(), PyError> {
        let result = self.pop();
        let _iter = self.pop();
        self.push(result);
        Ok(())
    }

    fn get_awaitable(&mut self, context: u32) -> Result<(), PyError> {
        // pyopcode.py GET_AWAITABLE.
        let w_iterable = self.pop();
        let anchor = FrameAnchor::new(self);
        let _roots = pyre_object::gc_roots::push_roots();
        let iterable_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_iterable);
        let w_iter = crate::baseobjspace::get_awaitable_iter(
            pyre_object::gc_roots::shadow_stack_get(iterable_slot),
            context,
        )?;
        // pyopcode.py:1604 guards a coroutine that is already being awaited
        // (`w_iter.get_delegate() is not None`) with RuntimeError.  pyre's
        // generator object has no delegate / `w_yielded_from` field, so the
        // reentrant-await case is instead caught at SEND by the generator
        // `running` flag.
        Self::push_anchored(&anchor, w_iter)
    }

    fn get_aiter(&mut self) -> Result<(), PyError> {
        let mut obj = self.pop();
        let anchor = FrameAnchor::new(self);
        let method = unsafe {
            pyre_object::with_roots!(obj => crate::baseobjspace::lookup_special(obj, "__aiter__"))?
        }
        .ok_or_else(|| {
            crate::PyError::type_error(format!(
                "'async for' requires an object with __aiter__ method, got {}",
                crate::type_methods::arg_type_name(obj)
            ))
        })?;
        let mut iter = crate::call::call_function_impl_result(method, &[])?;
        if unsafe {
            pyre_object::with_roots!(iter =>
                crate::baseobjspace::lookup_special(iter, "__anext__")
            )?
        }
        .is_none()
        {
            return Err(crate::PyError::type_error(format!(
                "'async for' received an object from __aiter__ that does not implement __anext__: {}",
                crate::type_methods::arg_type_name(iter)
            )));
        }
        Self::push_anchored(&anchor, iter)
    }

    fn get_anext(&mut self) -> Result<(), PyError> {
        let mut iter = self.peek();
        let anchor = FrameAnchor::new(self);
        let method = unsafe {
            pyre_object::with_roots!(iter =>
                crate::baseobjspace::lookup_special(iter, "__anext__")
            )?
        }
        .ok_or_else(|| {
            crate::PyError::type_error(format!(
                "'async for' requires an iterator with __anext__ method, got {}",
                crate::type_methods::arg_type_name(iter)
            ))
        })?;
        let mut next = crate::call::call_function_impl_result(method, &[])?;
        let awaitable =
            pyre_object::with_roots!(next => crate::baseobjspace::get_awaitable_iter(next, 0))
                .map_err(|mut cause| {
                    // CPython 3.14 `_PyEval_GetANext` uses
                    // `_PyErr_FormatFromCause` for *every* failure produced while
                    // converting `__anext__`'s result to an awaitable.  In
                    // particular, an exception raised by `result.__await__()` is the
                    // explicit cause of this TypeError; only an exception raised by
                    // `__anext__` itself propagates unchanged above.
                    let message = format!(
                        "'async for' received an invalid object from __anext__: {}",
                        crate::type_methods::arg_type_name(next)
                    );
                    let cause_obj = cause.to_exc_object();
                    let _roots = pyre_object::gc_roots::push_roots();
                    let _ = pyre_object::gc_roots::pin_root(cause_obj);
                    let cause_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                    let mut error = crate::PyError::type_error(message);
                    let error_obj = error.to_exc_object();
                    let cause_obj = pyre_object::gc_roots::shadow_stack_get(cause_slot);
                    unsafe {
                        pyre_object::interp_exceptions::w_exception_set_context(
                            error_obj, cause_obj,
                        );
                        pyre_object::interp_exceptions::w_exception_set_cause(error_obj, cause_obj);
                        pyre_object::interp_exceptions::w_exception_set_suppress_context(
                            error_obj, true,
                        );
                        crate::PyError::from_exc_object(error_obj)
                    }
                })?;
        Self::push_anchored(&anchor, awaitable)
    }

    fn end_async_for(&mut self) -> Result<(), PyError> {
        let exc = self.pop();
        let err = unsafe { crate::PyError::from_exc_object(exc) };
        if err.kind == crate::PyErrorKind::StopAsyncIteration {
            // The 3.14 exception table preserves the surrounding handler's
            // previous-exception entry below the asynchronous iterator; the
            // StopAsyncIteration edge itself pushes only `exc`.
            self.pop(); // asynchronous iterator
            Ok(())
        } else {
            Err(err)
        }
    }

    // ── load_method ──
    // PyPy: LOOKUP_METHOD — interpreter-only override.
    // For instances, pushes [attr, self] so CALL prepends self.
    // ── return_generator ──
    // CPython 3.12: RETURN_GENERATOR creates a generator from the current
    // frame and returns it to the caller. PyPy: generator.py GeneratorIterator.
    fn return_generator(&mut self) -> Result<(), PyError> {
        // When the generator function is already wrapped (CodeFlags::GENERATOR
        // detected in call_user_function_with_eval), RETURN_GENERATOR fires
        // during the first __next__() resume. It's a no-op in that case —
        // the generator object was already created at call time.
        // Push dummy value for the following POP_TOP to consume.
        self.push(pyre_object::w_none());
        Ok(())
    }

    // ── load_super_attr ──
    // CPython 3.12 LOAD_SUPER_ATTR: stack = [global_super, class, self]
    // → super(class, self).attr
    fn load_super_attr_with(
        &mut self,
        name_idx: usize,
        is_method: bool,
        is_two_arg: bool,
    ) -> Result<(), PyError> {
        let w_code = self.pycode as PyObjectRef;
        let name = &self.code().names[name_idx];
        // Resolve while the three opcode operands still live in the frame's
        // value stack.  First realization may allocate; after publication the
        // interned `co_names_w` entry is immutable, as in PyPy's `getname_w`.
        let w_name = unsafe { crate::pycode::w_code_getname_w_or_new(w_code, name_idx, name) };
        let (self_is_cell, class_slot) =
            crate::builtins::bare_super_frame_layout_words(self.code());
        let self_obj = self.pop();
        let cls = self.pop();
        let global_super = self.pop();
        let anchor = FrameAnchor::new(self);

        let result = load_super_attr_value_w(
            global_super,
            self_obj,
            cls,
            anchor.live(),
            w_name,
            is_two_arg,
            self_is_cell,
            class_slot,
        )?;

        // CPython _PySuper_Lookup: determines whether the resolved attr
        // is an unbound method (needs self binding) or a staticmethod /
        // classmethod (no self binding / bind class).
        if is_method {
            // getattr now returns a bound method via descriptor protocol.
            // Unwrap for the (func, self) pattern that CALL expects.
            if unsafe { pyre_object::is_method(result) } {
                let func = unsafe { pyre_object::w_method_get_func(result) };
                let recv = unsafe { pyre_object::w_method_get_self(result) };
                Self::push_anchored(&anchor, func)?;
                Self::push_anchored(&anchor, recv)?;
            } else {
                // staticmethod or classmethod — no self binding
                Self::push_anchored(&anchor, result)?;
                Self::push_anchored(&anchor, PY_NULL)?;
            }
        } else {
            // is_method=false: getattr already returned a bound method.
            Self::push_anchored(&anchor, result)?;
        }
        Ok(())
    }

    // For non-instances (modules etc.), pushes [attr, NULL].
    // The default trait impl always pushes [attr, NULL], which is what
    // the JIT tracer uses — no runtime branch in the shared path.
    fn load_method(&mut self, name: &str) -> Result<(), PyError> {
        let obj = self.pop();
        // callmethod.py:60-78 fast method path: a plain method descriptor in
        // the class, nothing shadowing it in the instance, on a type that uses
        // the default __getattribute__.  Pushes [w_descr, w_obj] (the unbound
        // function + receiver) so CALL_METHOD binds self without allocating a
        // Method wrapper.  Shared with the JIT tracer (trace_opcode.rs) so the
        // concrete and symbolic frames produce the identical stack shape.
        // `__getattribute__` allocates and relocates a JIT-created frame, so
        // the receiver is published before either lookup.
        let roots = pyre_object::gc_roots::push_roots();
        let obj_slot = roots.base();
        let _ = roots.pin_root(obj);
        if let Some((_, _, w_descr)) =
            unsafe { crate::baseobjspace::load_method_fast_path(roots.get(obj_slot), name) }
        {
            self.push(w_descr);
            self.push(roots.get(obj_slot));
            return Ok(());
        }
        let anchor = FrameAnchor::new(self);
        let attr = crate::baseobjspace::getattr_str(roots.get(obj_slot), name)?;
        let attr_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(attr);
        // LOOKUP_METHOD pushes (attr, null_or_self): the resolved attribute
        // first, then the bound receiver computed by the shared, side-effect
        // free binding decision (NULL when no self should be prepended).
        let bound = compute_load_method_bound(
            roots.get(obj_slot),
            pyre_object::gc_roots::shadow_stack_get(attr_slot),
            name,
        );
        let live = unsafe { &mut *anchor.live() };
        live.push(pyre_object::gc_roots::shadow_stack_get(attr_slot));
        live.push(bound);
        Ok(())
    }

    fn load_special(&mut self, name: &str) -> Result<(), PyError> {
        let obj = self.pop();
        // The descriptor `__get__` allocates and can relocate a JIT-created
        // frame; push onto the forwarded live frame.
        let anchor = FrameAnchor::new(self);
        let bound = crate::baseobjspace::load_special_resolve(obj, name)?;
        let live = unsafe { &mut *anchor.live() };
        live.push(bound);
        live.push(PY_NULL);
        Ok(())
    }

    /// pyopcode.py `LOAD_ATTR` — the interpreter consults the mapdict
    /// attribute cache only off-trace; under the JIT it does the plain
    /// `space.getattr`, which the trace folds via the type's `version_tag`.
    fn load_attr_cached(&mut self, name: &str, nameindex: usize) -> Result<(), PyError> {
        // pyopcode.py:1024 `if not jit.we_are_jitted():` — positive form keeps
        // the annotator off the bare-`!` hazard. The cache path's helpers are
        // `dont_look_inside`, so the JIT never traces into them.
        if majit_metainterp::jit::we_are_jitted() {
            return OpcodeStepExecutor::load_attr(self, name);
        }
        // Graceful underflow (`shared_opcode.rs`'s `opcode_load_attr` →
        // `pop_value()?`): a corrupted concrete-execution stack during
        // trace recording (e.g. a residual call the inline executor
        // could not perform) aborts the trace instead of panicking the
        // hard-asserting `pop()`.
        let obj = self.pop_value()?;
        // Popped, so nothing but this local still reaches the receiver across
        // the cache lookup, which runs arbitrary Python on a cache miss.
        let roots = pyre_object::gc_roots::push_roots();
        let obj_slot = roots.base();
        let obj = roots.pin_root(obj);
        let pycode = self.pycode as PyObjectRef;
        let anchor = FrameAnchor::new(self);
        let w_value = unsafe {
            crate::objspace::std::mapdict::load_attr_caching(pycode, obj, nameindex, name)
        }
        .map_err(|error| {
            if finalize_failed_attr_receiver_now(roots.get(obj_slot)) {
                unsafe { &mut *anchor.live() }.defer_failed_attr_until_pop_except();
            }
            error
        })?;
        Self::push_anchored(&anchor, w_value)
    }

    /// pyopcode.py `STORE_ATTR` — consults the mapdict attribute cache
    /// only off-trace; under the JIT it does the plain `space.setattr`, folded
    /// by the type's `version_tag`.
    fn store_attr_cached(&mut self, name: &str, nameindex: usize) -> Result<(), PyError> {
        // pyopcode.py:920 `if not jit.we_are_jitted():` — positive form.
        if majit_metainterp::jit::we_are_jitted() {
            return OpcodeStepExecutor::store_attr(self, name);
        }
        // pyopcode.py:918-919 — obj is the top of stack, value below it.
        // Graceful underflow like `opcode_store_attr` (`shared_opcode.rs`)
        // so a corrupted trace-recording stack aborts the trace instead of
        // panicking the hard-asserting `pop()`.
        let obj = self.pop_value()?;
        let value = self.pop_value()?;
        unsafe {
            crate::objspace::std::mapdict::store_attr_caching(
                self.pycode as PyObjectRef,
                obj,
                nameindex,
                name,
                value,
            )
        }
    }

    // ── call ──
    // PyPy: baseobjspace.py `call_valuestack` +
    // function.py `funccall_valuestack`.
    //
    // CPython 3.12+ CALL: stack is [callable, null_or_self, arg0..argN-1].
    // null_or_self is NULL for plain calls, `self` for method calls.
    fn call(&mut self, nargs: usize) -> Result<(), PyError> {
        // baseobjspace.py call_valuestack fast path: Function, including the
        // CALL_METHOD form.  callmethod.py counts a non-null `self` as
        // one extra argument while `dropvalues` remains the physical
        // `[callable, null_or_self, explicit args...]` width.  This is what
        // lets the translated interpreter expose an ordinary `_flat_pycall`
        // to the meta-tracer for `obj.method(...)`, just like PyPy.
        //
        // Guard: only enter when the value stack has at least nargs + 2
        // items above stack_base (callable + null_or_self + args).
        let stack_items = self.valuestackdepth.saturating_sub(self.stack_base());
        // `call_valuestack` diverts to `call_args_and_c_profile` only when the
        // frame is profiled AND the callable wraps a builtin code, so that the
        // `c_call` / `c_return` events a builtin owes are not skipped.  A
        // `Function` keeps `funccall_valuestack` under a profiler; its own
        // `call` / `return` events come from the `call_trace` / `return_trace`
        // bracket at the callee's activation, not from this dispatch.  The
        // test reads the raw callable, before the `_Method` unwrap below,
        // because upstream tests `w_func` at that same point.
        if stack_items >= nargs + 2 {
            // `stack_items >= nargs + 2` already proves both slots sit above
            // `stack_base`, so `peekvalue` is the same read as
            // `peekvalue_maybe_none` without the bounds re-check.
            let mut callable = self.peekvalue(nargs + 1);
            // `call_valuestack` tests `w_func` before the `_Method` unwrap
            // below. The peek above is that same raw callable.
            let profiled_builtin =
                self.get_is_being_profiled() && crate::function::is_builtin_code(callable);
            if !profiled_builtin {
                let mut null_or_self = self.peekvalue(nargs);
                // baseobjspace.py: `_Method` is not a generic callable
                // here.  Reuse its null/self stack slot for `w_instance`, unwrap
                // `w_function`, and continue through the identical Function
                // valuestack path.  Module aliases such as `random.gauss =
                // _inst.gauss` depend on this just as direct `obj.method()` calls
                // do; allocating an Arguments Vec for every alias call diverges
                // from PyPy's meta-traced interpreter shape.
                let mut is_func = !callable.is_null() && unsafe { crate::is_function(callable) };
                if !is_func
                    && !callable.is_null()
                    && null_or_self.is_null()
                    && unsafe { pyre_object::is_method(callable) }
                {
                    let receiver = unsafe { pyre_object::w_method_get_self(callable) };
                    let function = unsafe { pyre_object::w_method_get_func(callable) };
                    if !receiver.is_null()
                        && !function.is_null()
                        && unsafe { crate::is_function(function) }
                    {
                        self.settopvalue(receiver, nargs);
                        null_or_self = receiver;
                        callable = function;
                        is_func = true;
                    }
                }
                if is_func {
                    let methodcall = !null_or_self.is_null();
                    let call_nargs = nargs + usize::from(methodcall);
                    let anchor = FrameAnchor::new(self);
                    let result = crate::function::funccall_valuestack(
                        callable,
                        call_nargs,
                        self,
                        nargs + 2,
                        methodcall,
                    );
                    if result.is_null() {
                        return Err(crate::call::take_call_error()
                            .unwrap_or_else(|| crate::PyError::type_error("call failed")));
                    }
                    // baseobjspace.py `call_valuestack` self.pushvalue(w_result). The callee may
                    // have relocated this frame via a minor collection, so push
                    // onto the forwarded live frame, not the pre-call pointer.
                    unsafe { &mut *anchor.live() }.push(result);
                    return Ok(());
                }
            }
        }

        // Slow path: method call or non-Function callable.
        // Must allocate Vec for args.
        let mut args = Vec::with_capacity(nargs);
        for _ in 0..nargs {
            args.push(self.pop());
        }
        args.reverse();
        let null_or_self = self.pop();
        let callable = self.pop();

        let anchor = FrameAnchor::new(self);
        let result = if null_or_self.is_null() {
            call_callable(self, callable, &args)?
        } else {
            let mut full_args = Vec::with_capacity(1 + args.len());
            full_args.push(null_or_self);
            full_args.extend_from_slice(&args);
            call_callable(self, callable, &full_args)?
        };
        // The callee may have relocated this frame via a minor collection;
        // push onto the forwarded live frame, not the pre-call pointer.
        unsafe { &mut *anchor.live() }.push(result);
        Ok(())
    }

    // ── call_function_ex ──
    // pyopcode.py CALL_FUNCTION_EX:
    //     w_kwargs = self.popvalue() if has_kwarg else None
    //     w_args = self.popvalue()
    //     w_function = self.popvalue()
    //     args = self.argument_factory([], None, None,
    //                                  w_star=w_args,
    //                                  w_starstar=w_kwargs,
    //                                  w_function=w_function)
    //     w_result = self.space.call_args(w_function, args)
    //     self.pushvalue(w_result)
    //
    // argument.py Arguments.unpack_combined_starargs iterates w_star with
    // space.fixedview_unroll / space.listview_no_unpack, so arbitrary
    // iterables are accepted.
    fn call_function_ex(&mut self) -> Result<(), PyError> {
        let kwargs_or_null = self.pop();
        let args_obj = self.pop();
        let self_or_null = self.pop();
        let callable = self.pop();
        let anchor = FrameAnchor::new(self);
        let result =
            crate::call::call_function_ex(self, callable, self_or_null, args_obj, kwargs_or_null)?;
        // The callee may have relocated this frame via a minor collection;
        // push onto the forwarded live frame, not the pre-call pointer.
        unsafe { &mut *anchor.live() }.push(result);
        Ok(())
    }

    // ── call_kw ──
    // PyPy: CALL_FUNCTION_KW; CPython 3.13: CALL_KW
    // Stack: [callable, self_or_null, arg1, ..., argN, kwarg_names_tuple]
    /// CALL_KW — call with keyword arguments.
    ///
    /// PyPy: argument.py _match_signature
    /// Stack: [callable, null_or_self, arg0..argN-1, kwarg_names_tuple]
    /// The last `len(kwarg_names)` args are keyword args.
    ///
    /// Keyword resolution happens HERE (before frame creation) so the
    /// JIT eval loop sees correctly-positioned locals. PyPy does this
    /// in Arguments.parse_into_scope before the frame executes.
    /// `pyopcode.py CALL_FUNCTION_KW` is `@jit.unroll_safe`.
    #[majit_macros::unroll_safe]
    fn call_kw(&mut self, nargs: usize) -> Result<(), PyError> {
        let kwarg_names = self.pop();
        let mut args = Vec::with_capacity(nargs);
        for _ in 0..nargs {
            args.push(self.pop());
        }
        args.reverse();
        let self_or_null = self.pop();
        let callable = self.pop();

        let anchor = FrameAnchor::new(self);
        let result = crate::call::call_kw(self, callable, self_or_null, &args, kwarg_names)?;
        // The callee may have relocated this frame via a minor collection;
        // push onto the forwarded live frame, not the pre-call pointer.
        unsafe { &mut *anchor.live() }.push(result);
        Ok(())
    }

    // ── load_locals ──
    // PyPy: LOAD_LOCALS; CPython: LOAD_LOCALS
    // Pushes the current namespace dict onto the stack.
    fn load_locals(&mut self) -> Result<(), PyError> {
        // pyopcode.py:793-794:
        //   self.pushvalue(self.getorcreatedebug().w_locals)
        // In particular this must preserve a metaclass's custom `__prepare__`
        // mapping.  Rebuilding a plain dict loses mapping-subclass semantics;
        // treating a dict subclass as a non-dict can also accidentally copy
        // globals into the class namespace and make LOAD_FROM_DICT_OR_DEREF
        // select a global over its closure cell.
        let anchor = FrameAnchor::new(self);
        let w_locals = self.get_or_create_w_locals();
        Self::push_anchored(&anchor, w_locals)
    }

    // ── unpack_ex ──
    // PyPy: UNPACK_SEQUENCE with star; CPython: UNPACK_EX
    // `a, *b, c = iterable`
    /// `pyopcode.py UNPACK_EX` is `@jit.unroll_safe`.
    #[majit_macros::unroll_safe]
    fn unpack_ex(&mut self, args: crate::bytecode::UnpackExArgs) -> Result<(), PyError> {
        let before = args.before as usize;
        let after = args.after as usize;
        let value = self.pop();
        // `unpack_ex_slots` returns the `before + 1 + after` slots in TOS
        // order (head items, starred list, tail items); push bottom-first so
        // the first head item ends on top.
        let anchor = FrameAnchor::new(self);
        let slots = crate::runtime_ops::unpack_ex_slots(before, after, value)?;
        for item in slots.into_iter().rev() {
            Self::push_anchored(&anchor, item)?;
        }
        Ok(())
    }

    // ── delete_attr ──
    // PyPy: DELETE_ATTR → space.delattr(obj, name)
    fn delete_attr(&mut self, name: &str) -> Result<(), PyError> {
        let obj = self.pop();
        crate::baseobjspace::delattr_str(obj, name)?;
        Ok(())
    }

    // ── set_update ──
    // PyPy: set.update(iterable); CPython: SET_UPDATE
    fn set_update(&mut self, i: usize) -> Result<(), PyError> {
        let iterable = self.pop();
        let set = PyFrame::peek_at(self, i - 1);
        crate::opcode_ops::set_update_value(set, iterable)
    }

    // ── BuildSlice ──
    // CPython 3.13: BUILD_SLICE creates a slice object from 2 or 3 stack items
    fn build_slice(&mut self, argc: crate::bytecode::BuildSliceArgCount) -> Result<(), PyError> {
        use crate::bytecode::BuildSliceArgCount;
        let step = match argc {
            BuildSliceArgCount::Three => self.pop(),
            BuildSliceArgCount::Two => pyre_object::w_none(),
        };
        let stop = self.pop();
        let start = self.pop();
        self.push(pyre_object::w_slice_new(start, stop, step));
        Ok(())
    }

    // ── BinarySlice (a[b:c]) ──
    // PyPy: BINARY_SUBSCR with slice; CPython 3.13: BINARY_SLICE
    fn binary_slice(&mut self) -> Result<(), PyError> {
        let stop = self.pop();
        let start = self.pop();
        let obj = self.pop();
        let anchor = FrameAnchor::new(self);
        let result = crate::runtime_ops::binary_slice_values(obj, start, stop)?;
        Self::push_anchored(&anchor, result)
    }

    // ── StoreSlice (a[b:c] = d) ──
    // Stack (bottom→top): value, container, start, stop.
    fn store_slice(&mut self) -> Result<(), PyError> {
        let stop = self.pop();
        let start = self.pop();
        let container = self.pop();
        let value = self.pop();
        crate::runtime_ops::store_slice_values(container, start, stop, value)
    }

    // ── BuildString (f-string concatenation) ──
    // `pyopcode.py BUILD_STRING`: `Utf8StringBuilder`, then
    // `space.newutf8(builder.build(), builder.getlength())`.
    /// `pyopcode.py BUILD_STRING` is `@jit.unroll_safe`.
    #[majit_macros::unroll_safe]
    fn build_string(&mut self, count: usize) -> Result<(), PyError> {
        // `peekvalue` is index-from-top; reversing walks source order
        // (first pushed = index 0), the order `build_string_from_refs` reports.
        let mut parts = Vec::with_capacity(count);
        for i in (0..count).rev() {
            parts.push(self.peekvalue(i));
        }
        let result = crate::runtime_ops::build_string_from_refs(&parts)?;
        for _ in 0..count {
            let _ = self.pop();
        }
        self.push(result);
        Ok(())
    }

    // ── ListExtend ──
    // pypy/interpreter/pyopcode.py LIST_EXTEND — calls
    // `list.extend(iterable)`; on failure surfaces "Value after * must be
    // an iterable, not <T>" when the operand isn't iterable, else
    // re-raises the inner error.
    fn list_extend(&mut self, _i: usize) -> Result<(), PyError> {
        let iterable = self.pop();
        let list = self.peek();
        crate::opcode_ops::list_extend_value(list, iterable)
    }

    fn unsupported(
        &mut self,
        instruction: &Instruction,
    ) -> Result<StepResult<PyObjectRef>, PyError> {
        Err(PyError::type_error(format!(
            "unimplemented instruction: {instruction:?}"
        )))
    }
}

// ── JitState ↔ PyFrame conversion ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    // Language-level behaviour belongs in `pyre/extra_tests/snippets`, which
    // runs every script under CPython as well as both backends. What is left
    // here either reaches a Rust entry point no Python program can call, or
    // pins a value pyre and CPython 3.14 disagree about — a snippet asserting
    // the latter would have to fail on one of its own runners. Each test says
    // which of the two it is.

    fn run_exec_frame(source: &str) -> (PyResult, crate::pyframe::FrameBox) {
        // Module globals are now a celldict whose str keys hash through the
        // `hash_w` trampoline (production installs it before the first frame
        // via `init_jit_hooks`); mirror that here so frame construction can
        // seed the builtins.
        crate::test_hooks::install_hash_hook();
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let result = frame.execute_frame(None, None);
        (result, frame)
    }

    // Compiled code can record `ExecutionContext.enter` before handing the
    // same frame to a portal helper.  Installing the TLS current-frame root in
    // that helper must neither relink the frame nor unlink the surrounding
    // activation when the guard is dropped.
    #[test]
    fn test_install_current_frame_does_not_own_an_already_entered_chain() {
        let _ = run_exec_frame("pass");
        let code = compile_exec("pass").expect("compile failed");
        let mut frame = PyFrame::new(code);
        let frame_ptr = frame.as_mut_ptr();
        // The frame no longer carries its context, so resolve the thread's.
        // Read it AFTER building the frame: `PyFrame::new` re-stamps
        // `LAST_EXEC_CTX`, and this has to be the same context
        // `install_current_frame` will resolve.
        let ec = crate::call::getexecutioncontext() as *mut PyExecutionContext;
        assert!(!ec.is_null(), "the runtime must own an execution context");
        let outer_top = unsafe { (*ec).topframeref };

        unsafe { (*ec).enter(frame_ptr) };
        let linked_backref = unsafe { (*frame_ptr).f_backref };
        assert_eq!(linked_backref, outer_top);

        {
            let _guard = install_current_frame(unsafe { &mut *frame_ptr });
            assert_eq!(unsafe { (*frame_ptr).f_backref }, linked_backref);
        }
        assert_eq!(
            unsafe { (*ec).topframeref },
            frame_ptr,
            "the outer enter/leave scope, not the TLS guard, owns the unlink",
        );
        unsafe { (*ec).topframeref = outer_top };
    }

    // A frame value slot may hold a `JitVirtualRef`, whose leading magic word
    // would be read as a type pointer by the PyObject-shaped raw walks. Keep
    // the virtual-ref branch returning after it visits the `forced` field.
    #[test]
    fn test_frame_value_slot_holding_a_virtual_ref_skips_the_pyobject_walks() {
        use majit_metainterp::virtualref::{JIT_VIRTUAL_REF_VTABLE, JitVirtualRef, ObjectHeader};

        let mut vref = JitVirtualRef {
            super_: ObjectHeader {
                typeptr: JIT_VIRTUAL_REF_VTABLE,
            },
            virtual_token: std::ptr::null_mut(),
            forced: 0x1000usize as *mut u8,
        };
        let mut slot = majit_ir::GcRef(&mut vref as *mut JitVirtualRef as usize);
        let slot_address = &mut slot as *mut majit_ir::GcRef as usize;
        let forced_address = std::ptr::addr_of_mut!(vref.forced) as usize;
        let mut visited = Vec::new();

        unsafe {
            walk_frame_value_slot(&mut slot, &mut |root| {
                visited.push(root as *mut majit_ir::GcRef as usize);
            });
        }

        assert_eq!(visited, vec![slot_address, forced_address]);
    }

    // `exception_is_valid_obj_as_class_w` is a Rust predicate the raise path
    // consults; no Python expression evaluates it on its own.
    #[test]
    fn test_exception_is_valid_obj_as_class_w_matches_baseexception_subclass_rule() {
        let (_result, frame) = run_exec_frame("good = ValueError\nbad = int");
        let w_globals = frame.get_w_globals();
        let good =
            unsafe { pyre_object::w_dict_getitem_str(w_globals, "good") }.expect("missing good");
        let bad =
            unsafe { pyre_object::w_dict_getitem_str(w_globals, "bad") }.expect("missing bad");

        unsafe {
            assert!(crate::baseobjspace::exception_is_valid_obj_as_class_w(good));
            assert!(!crate::baseobjspace::exception_is_valid_obj_as_class_w(bad));
        }
    }

    // `load_method_fast_path` and `compute_load_method_bound` split one
    // LOAD_METHOD between them — the first picks the attribute, the second the
    // `null_or_self` pushed beside it — and no Python expression reaches
    // either alone.  They ask the same question about the receiver's
    // `__getattribute__`, so they have to answer it the same way: the fast
    // path pushes the UNBOUND function, and a `PY_NULL` companion calls it
    // with `self` missing.
    //
    // The state pinned here is the one a hot trace sees.  `mutated()` clears
    // `uses_object_getattribute` on every type-dict write, and the only writer
    // (`getattribute_if_not_from_object`) is skipped while jitted, so a type
    // reached only from compiled code keeps the flag `false` however ordinary
    // its `__getattribute__` is.
    #[test]
    fn test_load_method_bound_agrees_with_the_fast_path_on_an_unmemoized_type() {
        let (_result, frame) =
            run_exec_frame("class C:\n    def m(self):\n        return 1\nobj = C()");
        let w_globals = frame.get_w_globals();
        let obj =
            unsafe { pyre_object::w_dict_getitem_str(w_globals, "obj") }.expect("missing obj");

        unsafe {
            let w_type = pyre_object::w_instance_get_type(obj);
            // Read the descriptor before clearing the flag: `lookup_in_type`
            // is a plain MRO walk, while anything going through
            // `has_object_getattribute` would memoize the default back and
            // hide exactly what this pins.
            let w_descr = crate::baseobjspace::lookup_in_type(w_type, "m").expect("C.m");
            pyre_object::typeobject::w_type_set_uses_object_getattribute(w_type, false);

            assert!(
                std::ptr::eq(compute_load_method_bound(obj, w_descr, "m"), obj),
                "an unmemoized default `__getattribute__` must still bind self",
            );

            // The companion admits the same receiver, and pushes `w_descr`
            // unbound — which is what makes the NULL above a dropped `self`
            // rather than a missed optimisation.  Last, because it memoizes.
            let (_, _, fast_descr) =
                crate::baseobjspace::load_method_fast_path(obj, "m").expect("fast path admits C.m");
            assert!(std::ptr::eq(fast_descr, w_descr));
        }
    }

    // A lazy `PyError` is a Rust-side value; Python only ever sees the
    // instance `to_exc_object()` already materialised.
    #[test]
    fn test_to_exc_object_memoizes_lazy_exception() {
        // Bring up the managed heap / builtin exception types by running a
        // real frame first; `to_exc_object()` allocates the instance.
        let _ = run_exec_frame("pass");
        // A raw-message NameError is lazy: `exc_object` stays null until the
        // first `to_exc_object()` materialises it. The write-once memo
        // (`get_w_value`, error.py) must then return that same instance on
        // every later call instead of allocating a fresh one.
        let mut err = PyError::oefmt_name_error_text("name 'x' is not defined", "x");
        assert!(err.exc_object.is_null(), "raw-message error starts lazy");
        let first = err.to_exc_object();
        assert!(!first.is_null());
        assert_eq!(err.exc_object, first, "materialised instance is memoised");
        let second = err.to_exc_object();
        assert_eq!(first, second, "second call returns the memoised instance");
    }

    // The corrupted chain is written with `replace_op` after compilation; no
    // Python source the compiler accepts produces this bytecode.
    #[test]
    fn test_eval_loop_raises_on_malformed_extended_arg_chain() {
        let code = compile_exec("x = 1").expect("compile failed");
        unsafe {
            code.instructions.replace_op(0, Instruction::ExtendedArg);
            code.instructions.replace_op(1, Instruction::GetIter);
        }
        let mut frame = PyFrame::new(code);
        let err = frame
            .execute_frame(None, None)
            .expect_err("expected bytecode corruption");
        assert_eq!(err.kind, PyErrorKind::BytecodeCorruption);
        assert_eq!(err.message_text(), "bytecode corruption");
    }

    // `check_exc_match_against` is called here directly: the residual
    // `compare_value_from_tag` reaches it on a path an `except` clause cannot select.
    #[test]
    fn test_check_exc_match_against_matches_by_actual_type() {
        // pyopcode.py `return space.exception_match(space.type(w_1), w_2)`:
        // the left operand is matched by its *actual* type, never treated as
        // an unconditional success.  Guards the three shapes the residual
        // `compare_value_from_tag` (call_jit.rs) and the BC `check_exc_match` share:
        //   * a matching exception instance   -> true
        //   * a non-matching exception class  -> false (an `except` clause
        //     past the first must not spuriously match)
        //   * a non-exception value           -> false (matched by `type(v)`,
        //     whose MRO holds no exception class)
        let (_result, frame) = run_exec_frame(
            "exc = ValueError(\"boom\")\nplain = 5\nvalue_error = ValueError\ntype_error = TypeError",
        );
        let w_globals = frame.get_w_globals();
        let exc =
            unsafe { pyre_object::w_dict_getitem_str(w_globals, "exc") }.expect("missing exc");
        let plain =
            unsafe { pyre_object::w_dict_getitem_str(w_globals, "plain") }.expect("missing plain");
        let value_error = unsafe { pyre_object::w_dict_getitem_str(w_globals, "value_error") }
            .expect("missing value_error");
        let type_error = unsafe { pyre_object::w_dict_getitem_str(w_globals, "type_error") }
            .expect("missing type_error");

        assert!(check_exc_match_against(exc, value_error));
        assert!(!check_exc_match_against(exc, type_error));
        assert!(!check_exc_match_against(plain, value_error));
    }

    #[test]
    fn test_pyerror_matches_stop_iteration_uses_exception_mro() {
        let (result, frame) =
            run_exec_frame("class VS(ValueError, StopIteration):\n    pass\nexc = VS('done')");
        result.expect("exception subclass setup failed");
        let exc = unsafe { pyre_object::w_dict_getitem_str(frame.get_w_globals(), "exc") }
            .expect("missing exc");
        let err = unsafe { PyError::from_exc_object(exc) };

        // `find_best_base` gives VS StopIteration's instance layout even though
        // ValueError supplies `__new__`, so the instance carries that tag.
        assert_eq!(err.kind, PyErrorKind::StopIteration);
        // The object walk answers from the MRO, independently of the tag.
        assert!(crate::error::exception_object_matches_stop_iteration(exc));
        assert!(err.matches_stop_iteration());
        assert!(PyError::stop_iteration().matches_stop_iteration());
        assert!(!PyError::value_error("not exhausted").matches_stop_iteration());
    }

    // pyre materialises the rich-compare and `__iter__` rows in
    // `UnionType.__dict__`; CPython 3.14 leaves them to the slot table, so
    // `required <= UT.__dict__.keys()` is false there. Divergent by design,
    // hence not a snippet.
    #[test]
    fn test_union_type_exposes_cpython_314_richcompare_surface() {
        let source = r#"
UT = type(int | str)
required = {
    '__getattribute__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
    '__name__', '__qualname__', '__origin__', '__iter__', '__doc__',
}
assert required <= UT.__dict__.keys()
u = int | str
assert UT.__getattribute__(u, '__args__') == (int, str)
assert u.__name__ == 'Union'
assert u.__qualname__ == 'Union'
assert u.__origin__ is UT
assert u.__module__ == 'typing'
assert UT.__dict__['__iter__'] is None
try:
    iter(u)
except TypeError:
    pass
else:
    raise AssertionError('UnionType must disable __getitem__ iteration fallback')
assert UT.__ne__(u, int | str) is False
assert UT.__ne__(u, int | bytes) is True
assert UT.__ne__(u, int) is NotImplemented
assert UT.__lt__(u, int | bytes) is NotImplemented
assert UT.__le__(u, int | bytes) is NotImplemented
assert UT.__gt__(u, int | bytes) is NotImplemented
assert UT.__ge__(u, int | bytes) is NotImplemented
"#;
        let (result, _frame) = run_exec_frame(source);
        result.expect("UnionType explicit rich-compare surface failed");
    }

    // pyre spells the type in this AttributeError with its module prefix
    // (`__main__.make_type.<locals>.X`); CPython 3.14 prints the bare
    // qualname. Divergent by design, hence not a snippet.
    #[test]
    fn test_empty_member_slot_error_uses_fully_qualified_type_name() {
        let source = "\
def make_type():
    class X:
        __slots__ = 'a'
    return X
X = make_type()
try:
    X().a
except AttributeError as exc:
    result = str(exc)";
        let (res, frame) = run_exec_frame(source);
        res.expect("member slot AttributeError regression");
        unsafe {
            let value = w_dict_getitem_str(frame.get_w_globals(), "result").unwrap();
            assert_eq!(
                w_str_get_wtf8(value).as_str(),
                Ok("'__main__.make_type.<locals>.X' object has no attribute 'a'")
            );
        }
    }

    #[test]
    // `sort_seen` pins what `list.__sizeof__()` reports while `sort` holds the
    // receiver: pyre answers 56, CPython 3.14 answers 32 because it detaches
    // `ob_item` for the duration. Every other clause here matches CPython, but
    // that one keeps the whole fixture off the snippet runner.
    fn test_list_cpython_allocation_and_sizeof() {
        let source = "\
def allocation(value):
    return (value.__sizeof__() - type(value).__basicsize__) // 8

value = []
append_allocations = []
for item in range(20):
    value.append(item)
    append_allocations.append(allocation(value))

sources = (
    list([1, 2, 3, 4, 5]),
    list((1, 2, 3, 4, 5)),
    list(range(5)),
    list(item for item in range(5)),
    list(dict.fromkeys(range(5))),
    list(set(range(5))),
    list(dict.fromkeys(range(5)).keys()),
    list(dict.fromkeys(range(5)).values()),
    list(dict.fromkeys(range(5)).items()),
)

class HintTwenty:
    def __iter__(self):
        return iter(range(3))
    def __len__(self):
        return 20

sorted_value = [3, 2, 1]
sort_seen = []
def sort_key(item):
    sort_seen.append((len(sorted_value), sorted_value.__sizeof__()))
    return item
sorted_value.sort(key=sort_key)

noop_sorted = [3, 2, 1]
def noop_key(item):
    noop_sorted.clear()
    del noop_sorted[:]
    noop_sorted[:] = []
    return item
noop_sorted.sort(key=noop_key)

mucked_sorted = [3, 2, 1]
def mucked_key(item):
    if item == 3:
        mucked_sorted.append(9)
        mucked_sorted.pop()
    return item
try:
    mucked_sorted.sort(key=mucked_key)
except ValueError as exc:
    mucked_detected = str(exc) == 'list modified during sort'
else:
    mucked_detected = False

class Sub(list):
    pass
sub = Sub(range(3))

class HugeHint:
    def __iter__(self):
        return self
    def __next__(self):
        raise StopIteration
    def __length_hint__(self):
        return (1 << 63) - 1

nonempty = [1]
try:
    nonempty.extend(HugeHint())
except MemoryError:
    ignored_overflowing_hint = False
else:
    ignored_overflowing_hint = nonempty == [1]
try:
    list(HugeHint())
except MemoryError:
    empty_huge_hint_fails = True
else:
    empty_huge_hint_fails = False

result = (
    append_allocations == [4, 4, 4, 4, 8, 8, 8, 8,
                           16, 16, 16, 16, 16, 16, 16, 16,
                           24, 24, 24, 24]
    and [allocation(item) for item in sources] == [6, 6, 6, 8, 8, 8, 8, 8, 8]
    and allocation(list(HintTwenty())) == 8
    and sort_seen == [(0, 56), (0, 56), (0, 56)]
    and sorted_value == [1, 2, 3]
    and allocation(sorted_value) == 4
    and noop_sorted == [1, 2, 3]
    and mucked_sorted == [1, 2, 3]
    and mucked_detected
    and ignored_overflowing_hint
    and empty_huge_hint_fails
    and sub.__sizeof__() == Sub.__basicsize__ + 4 * 8
)
";
        let (res, frame) = run_exec_frame(source);
        res.expect("CPython list allocation metadata failed");
        unsafe {
            let result = w_dict_getitem_str(frame.get_w_globals(), "result").unwrap();
            assert!(crate::baseobjspace::is_true(result).unwrap());
        }
    }

    /// `walk_raw_function_roots` is the only tracing path a Box-immortal
    /// function has, so every slot `FUNCTION_GC_PTR_OFFSETS` declares must be
    /// forwarded by that walk unless it is deliberately exempt. The two lists
    /// drifted apart once: `w_name` was declared and not walked, so the string
    /// `f.__name__ = ...` installs had no root at all on an immortal function,
    /// while a managed one traced it through `FUNCTION_GC_TYPE_ID`. This runs
    /// the real walker and records the offsets it visits, so a field cannot be
    /// half-wired again.
    #[test]
    fn walk_raw_function_roots_covers_every_declared_gc_offset() {
        // A zeroed `Function` carries a real `ob_type` and nothing else:
        // `is_function_carrier` gates the walk on pointer identity with
        // `FUNCTION_TYPE`.
        let mut func: crate::function::Function =
            unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
        func.ob.ob_type = &crate::function::FUNCTION_TYPE as *const _;
        let base = &mut func as *mut crate::function::Function as usize;
        // Every declared slot but `code` is then stamped non-null, because
        // `visit_slot` skips an unset reference and a zeroed carrier would
        // report no coverage at all. The walk hands each of those slots to the
        // visitor as a pointer and dereferences none of them, so the carrier's
        // own address serves; `code` stays null because `walk_raw_code_roots`
        // does follow it.
        let code_offset = std::mem::offset_of!(crate::function::Function, code);
        for &offset in crate::function::FUNCTION_GC_PTR_OFFSETS.iter() {
            if offset != code_offset {
                unsafe { *((base + offset) as *mut usize) = base };
            }
        }

        let mut seen: Vec<usize> = Vec::new();
        {
            let mut record = |slot: &mut majit_ir::GcRef| {
                seen.push(slot as *mut majit_ir::GcRef as usize - base);
            };
            unsafe {
                walk_raw_function_roots(
                    &mut func as *mut crate::function::Function as PyObjectRef,
                    &mut record,
                );
            }
        }
        seen.sort_unstable();
        seen.dedup();

        // The same carrier retagged as a slot wrapper must still be walked:
        // `SLOT_WRAPPER_TYPE` shares `FUNCTION_TYPE`'s registered offsets, and
        // these carriers are immortal, so a refusal here is a population the
        // collector believes is traced and nothing traces.
        func.ob.ob_type = &crate::function::SLOT_WRAPPER_TYPE as *const _;
        let mut seen_wrapper: Vec<usize> = Vec::new();
        {
            let mut record = |slot: &mut majit_ir::GcRef| {
                seen_wrapper.push(slot as *mut majit_ir::GcRef as usize - base);
            };
            unsafe {
                walk_raw_function_roots(
                    &mut func as *mut crate::function::Function as PyObjectRef,
                    &mut record,
                );
            }
        }
        seen_wrapper.sort_unstable();
        seen_wrapper.dedup();
        assert_eq!(
            seen_wrapper, seen,
            "slot-wrapper carrier must be walked too"
        );

        // `name` is the one declared offset this walk skips: an immortal
        // function's name box comes from `malloc_raw`, so the collector never
        // owns it, while a managed function's GC box is reached through the
        // type id's offsets.
        let exempt = std::mem::offset_of!(crate::function::Function, name);
        let mut expected: Vec<usize> = crate::function::FUNCTION_GC_PTR_OFFSETS
            .iter()
            .copied()
            .filter(|off| *off != exempt)
            .collect();
        expected.sort_unstable();
        assert_eq!(seen, expected);
    }

    // `convert_value` is the CONVERT_VALUE helper an f-string reaches.
    // `DescrOperation.str` / `descr__str__` return the app-level dunder's
    // str object, so this pins that object rather than a WTF-8 copy of
    // its text. `Ascii` still encodes.
    #[test]
    fn test_convert_value_returns_the_instance_dunder_object() {
        let (result, frame) = run_exec_frame(
            "sentinel = 'SENTINEL'\n\
             class A:\n\
             \tdef __str__(self):\n\
             \t\treturn sentinel\n\
             \tdef __repr__(self):\n\
             \t\treturn sentinel\n\
             class B:\n\
             \tdef __repr__(self):\n\
             \t\treturn sentinel\n\
             class Bad:\n\
             \tdef __str__(self):\n\
             \t\treturn 1\n\
             \tdef __repr__(self):\n\
             \t\treturn sentinel\n\
             a = A()\n\
             b = B()\n\
             bad = Bad()\n",
        );
        assert!(result.is_ok(), "{result:?}");
        let w_globals = frame.get_w_globals();
        let get = |name| unsafe { pyre_object::w_dict_getitem_str(w_globals, name) }.expect(name);
        let sentinel = get("sentinel");
        let a = get("a");
        let b = get("b");
        let bad = get("bad");
        let same = |obj: pyre_object::PyObjectRef, conv: i64| {
            let got = crate::runtime_ops::convert_value(obj, conv).expect("convert_value");
            assert!(
                std::ptr::eq(got, sentinel),
                "convert_value conv={conv} returned a copy"
            );
        };
        same(a, 0);
        same(a, 1);
        same(a, 3);
        same(b, 0);
        same(b, 1);
        let exact = crate::runtime_ops::convert_value(sentinel, 0).expect("exact str");
        assert!(std::ptr::eq(exact, sentinel));
        // `ascii_from_object` encodes `__repr__`'s text. An all-ASCII repr
        // keeps that text and allocates a fresh str.
        let ascii = crate::runtime_ops::convert_value(a, 2).expect("ascii");
        assert!(!std::ptr::eq(ascii, sentinel));
        assert_eq!(
            unsafe { pyre_object::w_str_get_wtf8(ascii) }.as_str(),
            Ok("SENTINEL")
        );
        let err = crate::runtime_ops::convert_value(bad, 0).expect_err("non-str __str__");
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert!(
            err.message_text().contains("__str__ returned non-string"),
            "{}",
            err.message_text()
        );
    }

    static DECL_SHIPPED_WALK: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    thread_local! {
        static DECL_OWNED: std::cell::RefCell<Vec<usize>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    fn declaration_shipped_minor_walk(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        if !DECL_SHIPPED_WALK.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        walk_global_prebuilt_roots(visitor);
    }

    fn declaration_test_owns(addr: usize) -> bool {
        DECL_OWNED.with(|slots| slots.borrow().contains(&addr))
    }

    struct DeclarationTestRestore {
        prev_dirty: bool,
        roots_len: usize,
    }

    impl Drop for DeclarationTestRestore {
        fn drop(&mut self) {
            DECL_SHIPPED_WALK.store(false, std::sync::atomic::Ordering::Release);
            pyre_object::gc_hook::clear_gc_owns_object_hook();
            DECL_OWNED.with(|slots| slots.borrow_mut().clear());
            pyre_object::typedef::test_truncate_declaration_roots(self.roots_len);
            if self.prev_dirty {
                pyre_object::gc_roots::mark_prebuilt_roots_dirty();
            } else {
                pyre_object::gc_roots::clear_prebuilt_roots_dirty();
            }
        }
    }

    fn restore_declaration_test_state() -> DeclarationTestRestore {
        DeclarationTestRestore {
            prev_dirty: pyre_object::gc_roots::prebuilt_roots_dirty(),
            roots_len: pyre_object::typedef::test_declaration_roots_len(),
        }
    }

    /// A clean minor still walks TypeDef.rawdict slots (`incminimark.py`
    /// `collect_roots_in_nursery`: static pointers in prebuilt non-gc
    /// structures). A young pointer in a slot moves only when that walk runs.
    #[test]
    fn clean_minor_forwards_declaration_slot_only_when_the_container_is_traced() {
        use majit_gc::GcAllocator;
        use majit_gc::collector::{GcConfig, MiniMarkGC};
        use majit_gc::trace::TypeInfo;
        use std::sync::atomic::Ordering;

        let _hook_lock = pyre_object::gc_hook::hook_test_guard();
        // A collector requesting STW must hold the GIL.
        crate::module::thread::ensure_runtime_thread();
        let _restore = restore_declaration_test_state();
        let roots_len = _restore.roots_len;

        majit_gc::shadow_stack::register_extra_root_walker(
            declaration_shipped_minor_walk,
            "declaration_clean_minor",
        );
        pyre_object::gc_hook::register_gc_owns_object_hook(declaration_test_owns);

        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 4096,
            large_object_threshold: 2048,
            ..GcConfig::default()
        });
        let young_tid = gc.register_type(TypeInfo::simple(16));
        let young = gc.alloc_nursery_typed(young_tid, 16);
        DECL_OWNED.with(|slots| slots.borrow_mut().push(young.0));
        let _rooted = unsafe { pyre_object::typedef::TypeDefValue::root(young.0 as PyObjectRef) };
        let slot = pyre_object::typedef::test_last_declaration_slot();
        let young_before = young.0;
        pyre_object::gc_roots::clear_prebuilt_roots_dirty();

        gc.do_collect_nursery();
        assert_eq!(
            unsafe { *slot as usize },
            young_before,
            "clean minor without the shipped walk left the young slot unforwarded"
        );

        pyre_object::typedef::test_truncate_declaration_roots(roots_len);
        let young2 = gc.alloc_nursery_typed(young_tid, 16);
        DECL_OWNED.with(|slots| slots.borrow_mut().push(young2.0));
        let _rooted2 = unsafe { pyre_object::typedef::TypeDefValue::root(young2.0 as PyObjectRef) };
        let slot2 = pyre_object::typedef::test_last_declaration_slot();
        let young2_before = young2.0;
        pyre_object::gc_roots::clear_prebuilt_roots_dirty();
        DECL_SHIPPED_WALK.store(true, Ordering::Release);
        gc.do_collect_nursery();
        let forwarded = unsafe { *slot2 as usize };
        assert_ne!(forwarded, young2_before);
        assert!(gc.is_managed_heap_object(forwarded));
    }

    /// A `malloc_typed` interp2app is not a GC object. Its `_code` slot is
    /// forwarded by the per-slot interior walk, and the immortal header
    /// itself is not passed to the visitor.
    #[test]
    fn minor_forwards_interp2app_code_and_skips_the_immortal_header() {
        use majit_gc::GcAllocator;
        use majit_gc::collector::{GcConfig, MiniMarkGC};
        use majit_gc::trace::TypeInfo;
        use std::sync::atomic::Ordering;

        let _hook_lock = pyre_object::gc_hook::hook_test_guard();
        // A collector requesting STW must hold the GIL.
        crate::module::thread::ensure_runtime_thread();
        let _restore = restore_declaration_test_state();

        let descr = <pyre_object::gateway::interp2app as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR;
        pyre_object::gc_hook::register_pyre_class_offsets(
            descr.pytype_ptr as usize,
            descr.ptr_offsets,
        );

        majit_gc::shadow_stack::register_extra_root_walker(
            declaration_shipped_minor_walk,
            "declaration_interp2app_interior",
        );
        pyre_object::gc_hook::register_gc_owns_object_hook(declaration_test_owns);

        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 4096,
            large_object_threshold: 2048,
            ..GcConfig::default()
        });
        let young_tid = gc.register_type(TypeInfo::simple(16));
        let young = gc.alloc_nursery_typed(young_tid, 16);
        let young_before = young.0;
        DECL_OWNED.with(|slots| slots.borrow_mut().push(young_before));
        let gateway = unsafe {
            pyre_object::gateway::interp2app::new(young_before as PyObjectRef, "interior_probe")
        };
        assert!(!pyre_object::gc_hook::try_gc_owns_object(
            gateway as pyre_object::gc_hook::GCREF
        ));
        let _rooted = unsafe { pyre_object::typedef::TypeDefValue::root(gateway) };

        let mut seen = Vec::new();
        unsafe {
            let value = *pyre_object::typedef::test_last_declaration_slot();
            assert_eq!(value, gateway);
            assert!(!pyre_object::gc_hook::try_gc_owns_object(
                value as pyre_object::gc_hook::GCREF
            ));
            let mut record = |gcref: &mut majit_ir::GcRef| {
                seen.push(gcref.0);
            };
            walk_raw_function_roots(value, &mut record);
            walk_raw_getset_roots(value, &mut record);
            walk_raw_wrapped_function_roots(value, &mut record);
            walk_raw_immortal_roots(value, &mut record);
        }
        assert!(seen.contains(&young_before));
        assert!(
            !seen.contains(&(gateway as usize)),
            "immortal interp2app header was passed to the visitor"
        );

        pyre_object::gc_roots::clear_prebuilt_roots_dirty();
        DECL_SHIPPED_WALK.store(true, Ordering::Release);
        gc.do_collect_nursery();
        let forwarded =
            unsafe { (*(gateway as *mut pyre_object::gateway::interp2app))._code as usize };
        assert_ne!(forwarded, young_before);
        assert!(gc.is_managed_heap_object(forwarded));
    }
}
