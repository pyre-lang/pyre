use indexmap::IndexMap;
use std::cell::RefCell;
use std::sync::Arc;
/// runner.py: AbstractX86CPU — the Backend trait implementation.
///
/// This is the entry point for the dynasm backend, corresponding to
/// rpython/jit/backend/x86/runner.py AbstractX86CPU.
use std::sync::atomic::Ordering;

use majit_backend::deadframe::{ExitDescr, JitFrameDeadFrame};
use majit_backend::jitframe::{
    HostHeapGc, JitframeDescrFacts, check_jitframe_descr, jitframe_facts_at_cpu_init,
    jitframe_is_gc_object, jitframe_write_barrier, malloc_entry_jitframe, malloc_host_jitframe,
    malloc_jitframe, malloc_jitframe_no_collect, publish_standalone_jitframe_facts,
};
use majit_backend::libc_deadframe::LibcJitFrameDeadFrame;
use majit_backend::llmodel::{
    done_int_slot0, done_ref_slot0, park_or_free_done_entry_frame, prepare_done_raw_entry_frame,
    raw_done_entry_use_general, take_or_alloc_parked_entry_frame,
};
use majit_backend::{AsmInfo, Backend, BackendError, DeadFrame, JitCellToken};
// `gc_sync` hands out the concrete collector; the trait must be in scope for
// its methods to resolve on that type.
use majit_gc::GcAllocator;
use majit_ir::{FailDescr, GcRef, InputArgRc, OpRc, OpRef, Type, Value};

#[cfg(target_arch = "aarch64")]
use crate::aarch64::assembler::{AssemblerARM64 as Asm, CompiledCode};

#[cfg(target_arch = "aarch64")]
use crate::aarch64::cpu_ext::Aarch64CpuExt as ArchCpuExt;
use crate::arch;
use crate::codebuf;
use crate::jitframe::JitFrame;
#[cfg(target_arch = "x86_64")]
use crate::x86::assembler::{Assembler386 as Asm, CompiledCode};
#[cfg(target_arch = "x86_64")]
use crate::x86::cpu_ext::X86CpuExt as ArchCpuExt;

/// Global CALL_ASSEMBLER target registry.
///
/// RPython stores `descr._ll_function_addr` on the target token
/// (x86/assembler.py:599) so `CALL_ASSEMBLER` can resolve the callee
/// address directly from the descriptor. pyre identifies callee tokens
/// by `JitCellToken` object identity (`identity_key`), so a per-thread
/// `identity_key -> DynasmCaTarget` index is required.
///
/// Each entry retains a `Weak` to the callee's `CompiledLoopToken` so
/// `handle_call_assembler` (rewrite.py) can sample
/// `_ll_initial_locs` and `frame_info`. The strong owner is
/// `JitCellToken.compiled_loop_token` / `alive_loops`; a strong Arc
/// here would keep the CLT past `CompiledLoopToken.__del__`.
struct DynasmCaTarget {
    /// `_ll_function_addr` per `x86/assembler.py:599`, retained for
    /// redirect bookkeeping and diagnostics. Dynasm address resolution now
    /// reads the descr-carried token directly.
    code_addr: usize,
    /// `model.py` `CompiledLoopToken` — Weak to the owning
    /// `JitCellToken.compiled_loop_token`.
    compiled_loop_token: std::sync::Weak<majit_backend::CompiledLoopToken>,
    /// `pyjitpl.py:3629` `outermost_jitdriver_sd.index_of_virtualizable`.
    /// Captured from `JitCellToken.virtualizable_arg_index` when the compiled
    /// target registers.
    index_of_virtualizable: i32,
}

thread_local! {
    /// CALL_ASSEMBLER callee dispatch table.  RPython keeps the equivalent
    /// state on `cpu.assembler` (per-CPU); majit's JIT compiles and executes
    /// on a single thread, so a thread-local mirrors that per-context scope
    /// while keeping concurrent backend test binaries (each test runs on its
    /// own thread) from sharing a global map keyed by reused token numbers —
    /// a colliding token would otherwise let one thread bake another thread's
    /// compiled `code_addr` into a CALL_ASSEMBLER site.  The map is read only
    /// at compile time (the resolved address is baked into the code), so
    /// thread-local scope never narrows what production (single-threaded) can
    /// reach.
    static CALL_ASSEMBLER_TARGETS: RefCell<IndexMap<usize, DynasmCaTarget, rustc_hash::FxBuildHasher>> =
        const { RefCell::new(IndexMap::with_hasher(rustc_hash::FxBuildHasher)) };
}

fn unregister_dynasm_ca_target(token_key: usize) {
    let _ = CALL_ASSEMBLER_TARGETS.try_with(|cell| {
        cell.borrow_mut().swap_remove(&token_key);
    });
}

/// `rewrite.py` `handle_call_assembler` per-callee metadata
/// lookup, sourced from the registered `DynasmCaTarget`'s CLT Arc.
/// Mirrors `majit-backend-cranelift::compiler.rs`.
pub(crate) fn lookup_call_assembler_callee_locs(
    token_key: usize,
) -> Option<majit_gc::rewrite::CallAssemblerCalleeLocs> {
    CALL_ASSEMBLER_TARGETS.with(|cell| {
        let guard = cell.borrow();
        let target = guard.get(&token_key)?;
        let clt = target.compiled_loop_token.upgrade()?;
        // `JitFrameInfo` is `#[repr(C)]` and the Arc keeps the allocation
        // pinned, matching cranelift's `compiler.rs` pattern.
        let frame_info_ptr = clt.frame_info.data_ptr() as usize;
        let frame_depth = unsafe { (*clt.frame_info.data_ptr()).depth() as usize };
        let ll_initial_locs = clt._ll_initial_locs.lock().clone();
        Some(majit_gc::rewrite::CallAssemblerCalleeLocs {
            _ll_initial_locs: ll_initial_locs,
            frame_depth,
            frame_info_ptr,
            index_of_virtualizable: target.index_of_virtualizable,
        })
    })
}

/// The per-thread GC box, and the accessors every trampoline reaches it through.
///
/// `gc.py:30` `GcLLDescription.__init__` holds `self.gcdescr` as a plain field
/// on the backend descriptor — there is no per-thread allocator upstream — so
/// this cell is scaffolding, not a ported structure. `install_gc_box` fills it
/// for tests and embedders whose complete managed heap is thread-confined; a
/// runtime with shared managed values uses `install_gc_standalone` and the
/// `gc_sync` singleton.
///
/// Every accessor opens with `majit_gc::gc_box_installed()`, which without
/// `majit-gc/gc_box` is a constant `false` — so in a standalone build each one
/// folds to `None`, the thread-local becomes unreachable, and the trampolines
/// call `gc_sync` directly. The gate lives in `majit-gc` because a Cargo
/// feature is per-crate: this crate cannot `#[cfg]` on a feature of its
/// dependency, so the box is eliminated by the optimizer rather than by
/// conditional compilation.
mod gc_box {
    use std::cell::RefCell;

    thread_local! {
        /// llmodel.py self.gc_ll_descr — owned by the active dynasm
        /// backend on this thread. Stored as a thread-local so the
        /// backend-agnostic `majit_gc::ActiveGcGuardHooks` shims can
        /// reach the live allocator without taking a dynasm dependency.
        pub static DYNASM_ACTIVE_GC: RefCell<Option<Box<dyn majit_gc::GcAllocator>>> =
            const { RefCell::new(None) };
        /// Read-only mirror of the box address, for the queries that can fire
        /// while an in-progress allocation already holds the mutable borrow.
        static DYNASM_ACTIVE_GC_RAW: std::cell::Cell<Option<*mut dyn majit_gc::GcAllocator>> =
            const { std::cell::Cell::new(None) };
    }

    /// Apply `f` to this thread's GC box. `None` means there is no box, and
    /// the caller runs its `gc_sync` path instead.
    pub(super) fn with_ref<R>(f: impl FnOnce(&dyn majit_gc::GcAllocator) -> R) -> Option<R> {
        if !majit_gc::gc_box_installed() {
            return None;
        }
        DYNASM_ACTIVE_GC.with(|cell| cell.borrow().as_deref().map(f))
    }

    /// `&mut` counterpart of [`with_ref`], for allocation and write barriers.
    pub(super) fn with_mut<R>(f: impl FnOnce(&mut dyn majit_gc::GcAllocator) -> R) -> Option<R> {
        if !majit_gc::gc_box_installed() {
            return None;
        }
        DYNASM_ACTIVE_GC.with(|cell| {
            let mut guard = cell.borrow_mut();
            let raw: *mut dyn majit_gc::GcAllocator = guard.as_deref_mut()?;
            // SAFETY: `guard` holds the `RefCell` borrow for the whole `f`
            // call, and these callers are non-reentrant top-level mutator
            // trampolines, so the reborrow is exclusive and outlives `f`. The
            // raw round-trip is what lets the boxed `dyn + 'static` allocator
            // satisfy the `FnOnce(&mut dyn GcAllocator)` HRTB bound (same
            // shape `gc_op` gets from its `&'static mut` singleton).
            Some(f(unsafe { &mut *raw }))
        })
    }

    /// Read-only access that tolerates being reached from inside a collection.
    ///
    /// Structural adaptation: RPython's GC descriptor is a normal object
    /// reference, so `gc_current_object_address` can query ownership while a
    /// collection is already walking extra roots. Here the box sits behind a
    /// `RefCell` whose mutable borrow an in-progress allocation may hold, so
    /// that case reads the same allocator through the raw mirror rather than
    /// panicking across the extern slowpath.
    pub(super) fn with_reentrant_ref<R>(
        f: impl FnOnce(&dyn majit_gc::GcAllocator) -> R,
    ) -> Option<R> {
        if !majit_gc::gc_box_installed() {
            return None;
        }
        DYNASM_ACTIVE_GC.with(|cell| match cell.try_borrow() {
            Ok(guard) => guard.as_deref().map(f),
            // SAFETY: the mirror is published and cleared under the same
            // borrow as the box itself, so a non-null value points at the
            // live allocator, and this query only reads it.
            Err(_) => DYNASM_ACTIVE_GC_RAW.with(|raw| raw.get().map(|p| f(unsafe { &*p }))),
        })
    }

    /// `&mut` access for a top-level, never-reentrant op that has a defined
    /// answer when the box is busy: a borrow already held by an in-progress
    /// allocation yields `busy` rather than falling through to `gc_sync`.
    pub(super) fn with_mut_or_busy<R>(
        busy: R,
        f: impl FnOnce(&mut dyn majit_gc::GcAllocator) -> R,
    ) -> Option<R> {
        if !majit_gc::gc_box_installed() {
            return None;
        }
        DYNASM_ACTIVE_GC.with(|cell| {
            let mut guard = match cell.try_borrow_mut() {
                Ok(guard) => guard,
                Err(_) => return Some(busy),
            };
            let raw: *mut dyn majit_gc::GcAllocator = guard.as_deref_mut()?;
            // SAFETY: as in [`with_mut`].
            Some(f(unsafe { &mut *raw }))
        })
    }

    /// Whether this thread holds a box at all.
    pub(super) fn present() -> bool {
        majit_gc::gc_box_installed() && DYNASM_ACTIVE_GC.with(|cell| cell.borrow().is_some())
    }

    /// Store `gc` as this thread's box, publishing the raw mirror with it.
    pub(super) fn store(gc: Box<dyn majit_gc::GcAllocator>) {
        DYNASM_ACTIVE_GC.with(|cell| {
            let mut guard = cell.borrow_mut();
            *guard = Some(gc);
            let raw = guard
                .as_deref_mut()
                .map(|g| g as *mut dyn majit_gc::GcAllocator);
            DYNASM_ACTIVE_GC_RAW.with(|raw_cell| raw_cell.set(raw));
        });
    }

    /// Drop this thread's box and clear the raw mirror.
    ///
    /// The box goes first so reentrant ownership queries issued from its drop
    /// body still resolve old-heap addresses through the mirror.
    pub(super) fn clear() {
        DYNASM_ACTIVE_GC.with(|cell| {
            *cell.borrow_mut() = None;
        });
        DYNASM_ACTIVE_GC_RAW.with(|raw_cell| raw_cell.set(None));
    }
}

/// Read-only GC query for the guard hooks and codegen helpers.
///
/// - **Test box present**: a test owns the GC directly — apply `f` to the
///   boxed allocator.
/// - **No box, `gc_sync` initialized** (production): route to the
///   process-global singleton via `gc_query_reentrant`. These hooks can
///   fire during a collection's guard evaluation / extra-root walk, so
///   the reentrant read-only path (no second `gc_mutex`, no second
///   `&mut`) is required.
/// - **No box, `gc_sync` uninitialized** (no GC at all — unit tests):
///   returns `None` so callers keep their existing `.unwrap_or(default)`
///   / `.flatten()` behaviour.
pub(crate) fn with_dynasm_active_gc<R>(f: impl Fn(&dyn majit_gc::GcAllocator) -> R) -> Option<R> {
    if let Some(r) = gc_box::with_ref(&f) {
        return Some(r);
    }
    if majit_gc::gc_sync::is_initialized() {
        // The singleton is the concrete collector; the box path above is what
        // keeps this forwarder's argument a trait object.
        return Some(majit_gc::gc_sync::gc_query_reentrant(|gc| f(gc)));
    }
    None
}

/// GC write-barrier descriptor for machine-code generation.
///
/// Compilation may run outside the mutator thread that owns the GC box.
/// PyPy keeps this descriptor on `cpu.gc_ll_descr`; use
/// the current MiniMark layout in that case instead of silently omitting every
/// barrier. If an active collector explicitly reports no descriptor, preserve
/// that choice.
pub(crate) fn dynasm_write_barrier_descr() -> Option<majit_gc::WriteBarrierDescr> {
    match with_dynasm_active_gc(|gc| gc.get_write_barrier_descr()) {
        Some(descr) => descr,
        None => Some(majit_gc::WriteBarrierDescr::for_current_gc()),
    }
}

/// `&mut` counterpart of [`with_dynasm_active_gc`] for GC mutations
/// (allocation, write barriers). Same three-way routing: test box →
/// box; production (no box, `gc_sync` initialized) → `gc_sync::gc_op`;
/// no GC at all → `None` so callers keep their non-GC fallback.
///
/// These callers run at the top level of a JIT/blackhole trampoline
/// (mutator context), never inside a collection, so the non-reentrant
/// `gc_op` is correct here.
pub(crate) fn with_dynasm_active_gc_mut<R>(
    f: impl FnOnce(&mut dyn majit_gc::GcAllocator) -> R,
) -> Option<R> {
    if gc_box::present() {
        return gc_box::with_mut(f);
    }
    if majit_gc::gc_sync::is_initialized() {
        return Some(majit_gc::gc_sync::gc_op(|gc| f(gc)));
    }
    None
}

/// `cpu.gc_ll_descr`: the installed collector, or [`HostHeapGc`] —
/// `get_ll_description(None)` (gc.py) picks the Boehm descr when no
/// framework GC is configured, and this is that descr. Same three-way
/// routing as [`with_dynasm_active_gc_mut`], with the host descr in place of
/// `None`.
fn with_gc_ll_descr<R>(f: impl FnOnce(&mut dyn majit_gc::GcAllocator) -> R) -> R {
    if gc_box::present() {
        return gc_box::with_mut(f).expect("gc box present");
    }
    if majit_gc::gc_sync::is_initialized() {
        return majit_gc::gc_sync::gc_op(|gc| f(gc));
    }
    f(&mut HostHeapGc)
}

/// Logging and dump gates, read once per process. Each flag is itself a
/// cached env lookup; OR-ing them on every entry was five loads for a
/// steady run where every one is false.
#[inline]
fn exec_diag_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        crate::majit_log_enabled()
            || crate::majit_dump_enabled()
            || crate::dynasm_exec_diag_enabled()
            || majit_ir::debug::have_debug_prints()
            || crate::gc_freelist_diag_enabled()
    });
    *ENABLED
}

/// REF arguments of one `execute_token`, rooted across `malloc_jitframe`.
///
/// `ShadowStackFrameworkGCTransformer.push_roots` stores each live GCREF at
/// `root_stack_top` and bumps it. `walk_stack_root` writes a moved object
/// back into that slot. `pop_roots` (`Drop`) restores the top recorded
/// before the pushes. The handle is resolved once per call
/// (`gc_enter_roots_frame`), not once per ref.
struct EntryArgRoots {
    slot: Option<majit_gc::shadow_stack::ShadowStackSlot>,
    depth: usize,
}

impl EntryArgRoots {
    const fn inactive() -> Self {
        Self {
            slot: None,
            depth: 0,
        }
    }

    /// Push every `Value::Ref`, in argument order. No ref means no
    /// thread-local resolve and a `Drop` that does not touch the stack.
    fn push_refs(args: &[Value]) -> Self {
        let mut roots = Self::inactive();
        for arg in args {
            if let Value::Ref(value) = arg {
                roots.push_one(*value);
            }
        }
        roots
    }

    fn push_one(&mut self, value: GcRef) {
        let slot = match self.slot {
            Some(slot) => slot,
            None => {
                let slot = majit_gc::shadow_stack::shadow_stack_slot();
                self.depth = slot.depth();
                self.slot = Some(slot);
                slot
            }
        };
        slot.push(value);
    }

    /// The ref at `ref_index` among the pushed refs, after any forwarding.
    #[inline]
    fn get(&self, ref_index: usize) -> Option<GcRef> {
        let slot = self.slot?;
        Some(slot.get(self.depth + ref_index))
    }
}

impl Drop for EntryArgRoots {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            slot.pop_to(self.depth);
        }
    }
}

/// Arguments of one `llmodel.py execute_token` call.
///
/// `Typed` is the `Value` vector the general entry still carries. `Raw` is
/// the `unspecialize_value` words `warmstate.py maybe_compile_and_run` built;
/// the no-collector store writes each word with `set_int_value`.
#[derive(Copy, Clone)]
enum EntryWords<'a> {
    Typed(&'a [Value]),
    Raw(&'a [i64]),
}

impl std::fmt::Debug for EntryWords<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Typed(args) => std::fmt::Debug::fmt(args, f),
            Self::Raw(args) => std::fmt::Debug::fmt(args, f),
        }
    }
}

impl EntryWords<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Typed(args) => args.len(),
            Self::Raw(args) => args.len(),
        }
    }
}

/// The frame `make_execute_token` returns, before it is wrapped as a deadframe.
struct RanFrame {
    head: *mut JitFrame,
    tip: *mut JitFrame,
    gc_object: bool,
    num_slots: usize,
}

/// Frame allocated the same way [`alloc_entry_jitframe`] allocates the
/// entry (`llmodel.py` `realloc_frame`: `jitframe.JITFRAME.allocate`).
fn malloc_jitframe_like_entry(size_bytes: usize) -> *mut JitFrame {
    if !majit_gc::collector_installed() {
        malloc_host_jitframe(size_bytes)
    } else {
        with_gc_ll_descr(|gc| malloc_jitframe_no_collect(gc, size_bytes))
    }
}

/// `malloc_jitframe` (`execute_token`) for a compiled entry.
///
/// REF arguments are pushed before the allocation (`push_roots`). Rust's
/// stack is not traced; the frame is filled from the slots `walk_stack_root`
/// updates, and the guard's `Drop` is `pop_roots`. Returns whether the frame
/// is a collector object, which decides the deadframe that later owns it.
fn alloc_entry_jitframe(size_bytes: usize, args: &[Value]) -> (*mut JitFrame, bool, EntryArgRoots) {
    // `make_execute_token` allocates through `gc_ll_descr`. With no collector
    // installed that descr is `HostHeapGc` and the vtable query is a call
    // that always answers "host block".
    if !majit_gc::collector_installed() {
        return (
            malloc_jitframe_like_entry(size_bytes),
            false,
            EntryArgRoots::inactive(),
        );
    }
    with_gc_ll_descr(|gc| {
        let gc_object = jitframe_is_gc_object(gc);
        // `push_roots` before the collecting `malloc_jitframe`.
        let roots = if gc_object {
            EntryArgRoots::push_refs(args)
        } else {
            EntryArgRoots::inactive()
        };
        (malloc_entry_jitframe(gc, size_bytes), gc_object, roots)
    })
}

/// Whether the frames this thread's descr builds are collector objects.
fn jitframe_is_gc_managed() -> bool {
    with_gc_ll_descr(|gc| jitframe_is_gc_object(gc))
}

/// Store a GC allocator in the dynasm backend thread-local and register
/// the `majit_gc::set_active_*` function-pointer hooks, without
/// requiring a `DynasmBackend` instance.  This allows the GC subsystem
/// to be installed at boot (via `init_gc_subsystem`) before the JIT
/// driver is constructed.
/// Register all backend-agnostic `majit_gc::set_active_*` hooks to the
/// dynasm trampolines. Shared by `install_gc_box` (thread-confined heap: also
/// stores a box in TLS) and `install_gc_standalone` (shared heap: hooks only,
/// no box; the trampolines then route to the `gc_sync` singleton).
fn register_active_hooks(supports_guard_gc_type: bool, has_gcrootmap: bool) {
    majit_gc::set_active_gc_guard_hooks(majit_gc::ActiveGcGuardHooks {
        check_is_object: Some(dynasm_check_is_object),
        is_tagged_immediate: Some(dynasm_is_tagged_immediate),
        get_actual_typeid: Some(dynasm_get_actual_typeid),
        subclass_range: Some(dynasm_subclass_range),
        typeid_subclass_range: Some(dynasm_typeid_subclass_range),
        typeid_is_object: Some(dynasm_typeid_is_object),
        is_registered_type_id: Some(dynasm_is_registered_type_id),
        can_move: Some(dynasm_can_move),
        pin: Some(dynasm_pin),
        unpin: Some(dynasm_unpin),
        is_pinned: Some(dynasm_is_pinned),
        supports_guard_gc_type,
    });
    // The jitframe a compiled run returns stays alive as the deadframe until
    // the frontend has finished reading it, and is off the JF shadow stack for
    // that whole window. Publishing the set is what keeps its interior refs
    // rooted; see `libc_deadframe::LIVE_DEADFRAMES`.
    majit_gc::set_active_gc_deadframe_hooks(majit_gc::ActiveGcDeadFrameHooks {
        walk_live_deadframes: Some(majit_backend::libc_deadframe::walk_live_deadframes),
    });
    majit_gc::set_active_alloc_nursery_typed(Some(dynasm_alloc_nursery_typed));
    majit_gc::set_active_thread_has_gc_box(Some(gc_box::present));
    majit_gc::set_active_alloc_nursery_headerless_no_collect(Some(
        dynasm_alloc_nursery_headerless_no_collect,
    ));
    majit_gc::set_active_alloc_nursery_typed_with_placement(Some(
        dynasm_alloc_nursery_typed_with_placement,
    ));
    majit_gc::set_active_alloc_nursery_collecting_typed(Some(
        dynasm_alloc_nursery_collecting_typed,
    ));
    majit_gc::set_active_alloc_nursery_collecting_typed_rooted(Some(
        dynasm_alloc_nursery_collecting_typed_rooted,
    ));
    majit_gc::set_active_alloc_nursery_collecting_typed_roots(Some(
        dynasm_alloc_nursery_collecting_typed_roots,
    ));
    majit_gc::set_active_alloc_oldgen_typed(Some(dynasm_alloc_oldgen_typed));
    majit_gc::set_active_alloc_young_nonmoving_typed(Some(dynasm_alloc_young_nonmoving_typed));
    majit_gc::set_active_alloc_young_nonmoving_typed_no_collect(Some(
        dynasm_alloc_young_nonmoving_typed_no_collect,
    ));
    majit_gc::set_active_collect_generation(Some(dynasm_collect_generation));
    majit_gc::set_active_collect_step(Some(dynasm_collect_step));
    majit_gc::set_active_get_objects(Some(dynasm_get_objects));
    majit_gc::set_active_get_referents(Some(dynasm_get_referents));
    majit_gc::set_active_subgraph_has_pending_finalizer(Some(
        dynasm_subgraph_has_pending_finalizer,
    ));
    majit_gc::set_active_is_tracked(Some(dynasm_is_tracked));
    majit_gc::set_active_gcflag_hooks(
        Some(dynasm_get_gcflag_extra),
        Some(dynasm_toggle_gcflag_extra),
        Some(dynasm_get_gcflag_dummy),
    );
    majit_gc::set_active_get_rpy_memory_usage(Some(dynasm_get_rpy_memory_usage));
    majit_gc::set_active_get_rpy_type_index(Some(dynasm_get_rpy_type_index));
    majit_gc::set_active_get_rpy_roots(Some(dynasm_get_rpy_roots));
    majit_gc::set_active_get_rpy_referents(Some(dynasm_get_rpy_referents));
    majit_gc::set_active_is_app_level_object(Some(dynasm_is_app_level_object));
    majit_gc::set_active_dump_rpy_heap(Some(dynasm_dump_rpy_heap));
    majit_gc::set_active_get_typeids_text(Some(dynasm_get_typeids_text));
    majit_gc::set_active_get_typeids_list(Some(dynasm_get_typeids_list));
    majit_gc::set_active_add_memory_pressure(Some(dynasm_add_memory_pressure));
    majit_gc::set_active_maybe_collect_for_external_malloc(Some(
        dynasm_maybe_collect_for_external_malloc,
    ));
    majit_gc::set_active_total_memory_pressure(Some(dynasm_total_memory_pressure));
    majit_gc::set_active_collect_oldgen(Some(dynasm_collect_oldgen_nonmoving));
    majit_gc::set_active_heap_stats(Some(dynasm_heap_stats));
    majit_gc::set_active_gc_memory_stats(Some(dynasm_gc_memory_stats));
    majit_gc::set_active_major_threshold_reached(Some(dynasm_major_threshold_reached));
    majit_gc::set_active_minor_collections_since_major(Some(dynasm_minor_collections_since_major));
    majit_gc::set_active_root_hooks(Some(dynasm_gc_add_root), Some(dynasm_gc_remove_root));
    majit_gc::set_active_gc_owns_object(Some(dynasm_gc_owns_object));
    majit_gc::set_active_gc_shrink_array(Some(dynasm_gc_shrink_array));
    majit_gc::set_active_gc_varsize_layout(Some(dynasm_gc_varsize_layout));
    majit_gc::set_active_gc_is_nursery_object(Some(dynasm_gc_is_nursery_object));
    // Assumption: the active GC does not move objects.
    // `has_gcrootmap() == false` is the predicate available
    // (`GcLLDescr_boehm.gcrootmap is None`; CelGc answers false because
    // `collect_nursery` / `collect_full` are no-ops). `minimark.py
    // id_or_identityhash` is then the object's address. Leaving the hook
    // unset makes `hash_whatever` of a Ref green that address, so
    // `JitCell.get_uhash` of a constant green does not trampoline.
    if has_gcrootmap {
        majit_gc::set_active_gc_id_or_identityhash(Some(dynasm_id_or_identityhash));
    } else {
        majit_gc::set_active_gc_id_or_identityhash(None);
    }
    majit_gc::set_active_write_barrier(Some(dynasm_gc_write_barrier));
    majit_gc::set_active_write_barrier_before_move(Some(dynasm_gc_write_barrier_before_move));
    majit_gc::set_active_write_barrier_from_array(Some(dynasm_gc_write_barrier_from_array));
    majit_gc::set_active_writebarrier_before_copy(Some(dynasm_gc_writebarrier_before_copy));
    majit_gc::set_active_write_barrier_managed(Some(dynasm_gc_write_barrier_managed));
    majit_gc::set_active_finalizer_hooks(
        Some(dynasm_register_finalizer),
        Some(dynasm_finalizer_next_dead),
    );
}

/// Install a GC allocator box into TLS and register all `set_active_*`
/// hooks. Test path only — `set_gc_allocator` hands ownership of a real
/// allocator to the backend thread. Production uses
/// [`install_gc_standalone`], which registers the same hooks WITHOUT a
/// box so the trampolines fall through to `gc_sync`.
///
/// Two live boxes at once is not a supported configuration: `ConstPtr`
/// `MAJOR_LIVE` and `TRACE_OPS_GC_TYPE_ID` are process-global indexes
/// into one collector. Sequential replace on one thread is the test
/// shape.
fn install_gc_box(gc: Box<dyn majit_gc::GcAllocator>) -> JitframeDescrFacts {
    // This thread now answers heap queries from its own allocator, whose
    // nursery is not the singleton's, so the process-wide published range can
    // no longer stand in for `is_nursery_object`.
    majit_gc::disarm_published_nursery();
    let has_gcrootmap = gc.has_gcrootmap();
    majit_gc::note_gc_box_installed(has_gcrootmap);
    let supports_guard_gc_type = gc.supports_guard_gc_type();
    let facts = check_jitframe_descr(gc.as_ref());
    gc_box::store(gc);
    register_active_hooks(supports_guard_gc_type, has_gcrootmap);
    facts
}

/// Production path: register all `set_active_*` hooks WITHOUT storing a
/// box, so every trampoline routes to the process-global `gc_sync` singleton
/// (the per-thread GC box is the free-threading gap R4 removes).
///
/// The type table stays open. `gctypelayout.py encode_type_shapes_now`
/// closes it at translation; pyre closes it before the first reader so
/// JIT-only types can still be registered after startup.
pub fn install_gc_standalone() {
    let facts = majit_gc::gc_sync::gc_op(|gc| check_jitframe_descr(gc));
    publish_standalone_jitframe_facts(facts);
    let supports_guard_gc_type = majit_gc::gc_sync::gc_query(|gc| gc.supports_guard_gc_type());
    let has_gcrootmap = majit_gc::gc_sync::gc_query(|gc| gc.has_gcrootmap());
    register_active_hooks(supports_guard_gc_type, has_gcrootmap);
}

/// Drop the active dynasm GC box. Callers must go through this helper rather
/// than reaching the thread-local directly, otherwise the raw mirror used by
/// `dynasm_gc_owns_object`'s reentrant fallback would be left pointing at
/// freed memory.
pub fn clear_gc_allocator() {
    gc_box::clear();
}

/// TYPE_INFO / CLASSTYPE constants read by the dynasm assemblers for
/// `GUARD_IS_OBJECT` and `GUARD_SUBCLASS`.
///
/// RPython reads these from `self.cpu.gc_ll_descr` at codegen time
/// (`x86/assembler.py:1934-1939`, `1946-1969`). The Rust assembler is a
/// transient emitter without a borrow of `DynasmBackend`, so the runner
/// pre-fetches the same values once per trace and passes them in.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GuardGcTypeInfo {
    pub base_type_info: usize,
    pub shift_by: u8,
    pub sizeof_ti: usize,
    pub infobits_offset: usize,
    pub is_object_flag: u8,
    pub subclassrange_min_offset: usize,
}

fn dynasm_check_is_object(gcref: GcRef) -> bool {
    with_dynasm_active_gc(|gc| gc.check_is_object(gcref)).unwrap_or(false)
}

fn dynasm_is_tagged_immediate(addr: usize) -> bool {
    with_dynasm_active_gc(|gc| gc.is_tagged_immediate(addr)).unwrap_or(false)
}

fn dynasm_get_actual_typeid(gcref: GcRef) -> Option<u32> {
    with_dynasm_active_gc(|gc| gc.get_actual_typeid(gcref)).flatten()
}

fn dynasm_can_move(gcref: GcRef) -> bool {
    with_dynasm_active_gc(|gc| gc.can_move(gcref)).unwrap_or(false)
}

fn dynasm_pin(gcref: GcRef) -> bool {
    with_dynasm_active_gc_mut(|gc| gc.pin(gcref)).unwrap_or(false)
}

fn dynasm_unpin(gcref: GcRef) {
    with_dynasm_active_gc_mut(|gc| gc.unpin(gcref)).expect("missing active GC runtime");
}

fn dynasm_is_pinned(gcref: GcRef) -> bool {
    with_dynasm_active_gc(|gc| gc.is_pinned(gcref)).unwrap_or(false)
}

fn dynasm_subclass_range(classptr: usize) -> Option<(i64, i64)> {
    with_dynasm_active_gc(|gc| gc.subclass_range(classptr)).flatten()
}

fn dynasm_typeid_subclass_range(typeid: u32) -> Option<(i64, i64)> {
    with_dynasm_active_gc(|gc| gc.typeid_subclass_range(typeid)).flatten()
}

/// gc.py `get_nursery_free_addr` / `get_nursery_top_addr` parity:
/// the backend reads nursery slot addresses from the active GC descriptor,
/// NOT from a process-global singleton. Returns `(0, 0)` when no GC is
/// bound so the assembler falls back to the slow-path helper.
pub(crate) fn dynasm_nursery_addrs() -> (usize, usize) {
    with_dynasm_active_gc(|gc| (gc.nursery_free_addr(), gc.nursery_top_addr())).unwrap_or((0, 0))
}

/// [`dynasm_nursery_addrs`] for an inline headerless bump. A collector that
/// is not headerless-aware (`GcAllocator::headerless_nursery`) answers
/// `(0, 0)` as an absent nursery does, so the emitter keeps the allocation on
/// its helper instead of writing a raw base into a headered nursery.
pub(crate) fn dynasm_headerless_nursery_addrs() -> (usize, usize) {
    with_dynasm_active_gc(|gc| {
        if gc.headerless_nursery() {
            (gc.nursery_free_addr(), gc.nursery_top_addr())
        } else {
            (0, 0)
        }
    })
    .unwrap_or((0, 0))
}

/// `GcLLDescr_framework.max_size_of_young_obj`, consumed by
/// `malloc_cond_varsize` before it computes `itemsize * length`.  Zero is the
/// no-GC sentinel, paired with [`dynasm_nursery_addrs`]'s `(0, 0)` answer.
pub(crate) fn dynasm_max_size_of_young_obj() -> usize {
    with_dynasm_active_gc(|gc| gc.max_nursery_object_size()).unwrap_or(0)
}

/// Head of the recycle list an inline allocation takes from before it bumps.
/// Zero when the bound GC keeps none, and when none is bound.
pub(crate) fn dynasm_nursery_recycle_list_addr() -> usize {
    with_dynasm_active_gc(|gc| gc.nursery_recycle_list_addr()).unwrap_or(0)
}

pub(crate) fn dynasm_nursery_recycle_window_addr() -> usize {
    with_dynasm_active_gc(|gc| gc.nursery_recycle_window_addr()).unwrap_or(0)
}

/// Per-backend `CPU.load_supported_factors` (rewrite.py:1124 /
/// x86/runner.py / llmodel.py:39). x86 addressing scales natively by  allow-line-citation
/// 1/2/4/8, aarch64 has no scaled store form and always expects factor 1.
#[cfg(target_arch = "x86_64")]
fn gc_store_supported_factors() -> &'static [i64] {
    &[1, 2, 4, 8]
}

#[cfg(target_arch = "aarch64")]
fn gc_store_supported_factors() -> &'static [i64] {
    &[1]
}

/// Per-backend `CPU.supports_load_effective_address` — x86/runner.py
/// overrides model.py:22 base default `False` to `True`; aarch64/runner.py
/// inherits the base `False` (rewriter expands to INT_LSHIFT + INT_ADD +
/// INT_ADD per rewrite.py:1089-1098 instead of emitting LOAD_EFFECTIVE_ADDRESS).
#[cfg(target_arch = "x86_64")]
fn supports_load_effective_address() -> bool {
    true
}

#[cfg(target_arch = "aarch64")]
fn supports_load_effective_address() -> bool {
    false
}

fn dynasm_typeid_is_object(typeid: u32) -> Option<bool> {
    with_dynasm_active_gc(|gc| gc.typeid_is_object(typeid)).flatten()
}

fn dynasm_is_registered_type_id(typeid: u32) -> bool {
    with_dynasm_active_gc(|gc| (typeid as usize) < gc.type_count()).unwrap_or(false)
}

/// Published Boehm `malloc_fixedsize` address, or `None` when it must not run.
///
/// `gc.py` has one `GcLLDescr` per translation. An installed collector
/// owns fixed-size blocks; the hook's storage is unheadered and must not
/// be handed to it. `collector_installed` is an argument so a test can
/// force either arm without publishing a process-global collector.
fn boehm_fixedsize_hook(collector_installed: bool) -> Option<usize> {
    if collector_installed {
        return None;
    }
    let addr = majit_gc::malloc_fixedsize_addr();
    if addr == 0 { None } else { Some(addr) }
}

/// `GcLLDescr_boehm.malloc_fixedsize`: storage from the published function,
/// or `None` when [`majit_gc::set_malloc_fixedsize`] is unset.
///
/// `GC_malloc` returns zero-filled bytes (`malloc_zero_filled`). The caller
/// does not clear the block again.
pub(crate) fn call_malloc_fixedsize(size: usize) -> Option<*mut u8> {
    call_malloc_fixedsize_inner(size, majit_gc::collector_installed())
}

fn call_malloc_fixedsize_inner(size: usize, collector_installed: bool) -> Option<*mut u8> {
    let addr = boehm_fixedsize_hook(collector_installed)?;
    let malloc: extern "C" fn(usize) -> *mut u8 = unsafe { std::mem::transmute(addr) };
    Some(malloc(size))
}

/// Headerless no-collect nursery trampoline for backend-agnostic callers.
///
/// The metainterp's jitcode tracer allocates a `NEW` on a `headerless` descr
/// through here so the object lands in the interpreter's own collected pool
/// rather than the host heap, where its collector could not see it. Returns
/// null when no GC is bound, leaving the caller on its own path.
fn dynasm_alloc_nursery_headerless_no_collect(size: usize) -> GcRef {
    if let Some(r) = gc_box::with_mut(|g| g.alloc_nursery_headerless_no_collect(size)) {
        return r;
    }
    if majit_gc::gc_sync::is_initialized() {
        return majit_gc::gc_sync::gc_op(|g| g.alloc_nursery_headerless_no_collect(size));
    }
    GcRef::NULL
}

/// Host-side nursery allocation trampoline. Published via
/// `majit_gc::set_active_alloc_nursery_typed` from `set_gc_allocator`
/// so backend-agnostic callers (e.g. pyre-object `w_int_new`) can
/// route through the live dynasm-owned GC without taking a backend
/// dependency.
fn dynasm_alloc_nursery_typed(type_id: u32, size: usize) -> GcRef {
    // NOTE host-side allocation must not trigger collection: the
    // caller holds a raw `*mut u8` on the Rust stack that is NOT
    // registered as a GC root. Collection here would move the
    // freshly-allocated nursery object, leaving the caller with a
    // dangling pointer. Routing through `try_alloc_nursery_no_collect_typed`
    // falls back to old-gen on nursery full — stable across minor
    // collections that fire between here and the caller's store into a
    // tracked slot — and returns NULL when rawmalloc fails so the host helper
    // can raise `MemoryError`.
    if let Some(r) = gc_box::with_mut(|g| g.try_alloc_nursery_no_collect_typed(type_id, size)) {
        return r;
    }
    majit_gc::standalone_alloc_nursery_typed(type_id, size)
}

/// Placement-reporting companion of [`dynasm_alloc_nursery_typed`].
///
/// # Safety
/// `needs_write_barrier` must remain a valid mutable `bool` slot until this
/// call returns.
unsafe fn dynasm_alloc_nursery_typed_with_placement(
    type_id: u32,
    size: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    if let Some(r) = gc_box::with_mut(|g| unsafe {
        g.try_alloc_nursery_no_collect_typed_with_placement(type_id, size, needs_write_barrier)
    }) {
        return r;
    }
    majit_gc::gc_sync::gc_op(|g| unsafe {
        g.try_alloc_nursery_no_collect_typed_with_placement(type_id, size, needs_write_barrier)
    })
}

/// Host-side *collecting* nursery allocation trampoline. Unlike
/// [`dynasm_alloc_nursery_typed`], this runs a minor collection when the nursery
/// is full (instead of spilling to old-gen). Used only by the elidable bigint
/// payload helpers, which are invoked from a residual `CallR` whose gcmap roots
/// the trace's live set and which hold no unrooted GC pointer across the
/// allocation — so the embedded minor cycle is safe and dead bigints are
/// reclaimed instead of accumulating in old-gen.
fn dynasm_alloc_nursery_collecting_typed(type_id: u32, size: usize) -> GcRef {
    if let Some(r) = gc_box::with_mut(|g| g.alloc_nursery_typed(type_id, size)) {
        return r;
    }
    majit_gc::gc_sync::gc_op(|g| g.alloc_nursery_typed(type_id, size))
}

/// Rooted collecting allocation used when a residual helper has manufactured
/// one GC child on the native Rust stack before allocating its parent. The
/// MiniMark override keeps the root out of the dynamic root set unless the
/// nursery bump actually reaches `collect_and_reserve`.
///
/// # Safety
/// `root` must remain a valid mutable GC slot until this call returns.
/// `needs_write_barrier` must remain a valid mutable `bool` slot.
unsafe fn dynasm_alloc_nursery_collecting_typed_rooted(
    type_id: u32,
    size: usize,
    root: *mut GcRef,
    needs_write_barrier: *mut bool,
) -> GcRef {
    if let Some(r) = gc_box::with_mut(|g| unsafe {
        g.alloc_nursery_collecting_typed_rooted(type_id, size, root, needs_write_barrier)
    }) {
        return r;
    }
    unsafe {
        majit_gc::standalone_alloc_nursery_collecting_typed_rooted(
            type_id,
            size,
            root,
            needs_write_barrier,
        )
    }
}

unsafe fn dynasm_alloc_nursery_collecting_typed_roots(
    type_id: u32,
    size: usize,
    roots: *mut GcRef,
    root_count: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    if let Some(r) = gc_box::with_mut(|g| unsafe {
        g.alloc_fast_nursery_collecting_typed_roots(
            type_id,
            size,
            roots,
            root_count,
            needs_write_barrier,
        )
    }) {
        return r;
    }
    unsafe {
        majit_gc::standalone_alloc_fast_nursery_collecting_typed_roots(
            type_id,
            size,
            roots,
            root_count,
            needs_write_barrier,
        )
    }
}

/// Host-side old-gen allocation trampoline. Used by
/// pyre-object allocators (`w_int_new`, `w_float_new`) whose
/// callers cannot register the returned pointer as a GC root before
/// subsequent allocations. MiniMark's old-gen is mark-sweep
/// (non-moving), so the returned pointer is stable across minor and
/// major collections.
fn dynasm_alloc_oldgen_typed(type_id: u32, size: usize) -> GcRef {
    if let Some(r) = gc_box::with_mut(|g| g.alloc_oldgen_typed(type_id, size)) {
        return r;
    }
    // Grain never installs MiniMark. A typed blackhole NEW with no collector
    // is NULL (`bh_alloc_struct`), not a panic on the unset singleton.
    if !majit_gc::gc_sync::is_initialized() {
        return GcRef(0);
    }
    majit_gc::gc_sync::gc_op(|g| g.alloc_oldgen_typed(type_id, size))
}

/// Host-side young non-moving allocation trampoline
/// (`external_malloc(..., alloc_young=True)`): the address is as stable as
/// [`dynasm_alloc_oldgen_typed`]'s, but the next minor collection frees the
/// block unless a root or a traced edge reaches it. May itself run that
/// minor (`threshold_reached` then `minor_collection_with_major_progress`).
fn dynasm_alloc_young_nonmoving_typed(type_id: u32, size: usize) -> GcRef {
    if let Some(r) = gc_box::with_mut(|g| g.alloc_young_nonmoving_typed(type_id, size)) {
        return r;
    }
    if !majit_gc::gc_sync::is_initialized() {
        return GcRef(0);
    }
    majit_gc::gc_sync::gc_op(|g| g.alloc_young_nonmoving_typed(type_id, size))
}

/// [`dynasm_alloc_young_nonmoving_typed`] without the collection in front of
/// the birth, for the host constructors that fill the block from words held
/// on the Rust stack (`GcAllocator::alloc_young_nonmoving_typed_no_collect`).
fn dynasm_alloc_young_nonmoving_typed_no_collect(type_id: u32, size: usize) -> GcRef {
    if let Some(r) = gc_box::with_mut(|g| g.alloc_young_nonmoving_typed_no_collect(type_id, size)) {
        return r;
    }
    if !majit_gc::gc_sync::is_initialized() {
        return GcRef(0);
    }
    majit_gc::gc_sync::gc_op(|g| g.alloc_young_nonmoving_typed_no_collect(type_id, size))
}

/// Allocate the struct a `bh_new` / `bh_new_with_vtable` descr describes
/// (`llmodel.py`).
///
/// A GC-managed struct (real `type_id`) MUST be allocated through the GC so the
/// collector can trace its pointer fields: a resume-materialized virtual (e.g.
/// an inlined-callee `PyFrame`) holds a `locals_cells_stack` ref to its arrays,
/// and a raw `libc::malloc` block is invisible to the GC, so a minor collection
/// during the blackhole forward run frees those arrays out from under the
/// frame.  Allocate in the non-moving old generation (mark-sweep), mirroring
/// `w_int_new`/`w_float_new`: the blackhole register file and the deep forward
/// recursion capture raw pointers to the materialized struct that the resume
/// path does not re-root across the minor collections it triggers, so a moving
/// nursery object would leave those captures stale.  Old-gen keeps every
/// materialized pointer stable for the lifetime of the resume.
///
/// A headerless struct lives in the interpreter's own `headerless_structs` pool
/// and carries no `type_id` word at `ref - 8`, so it takes the headerless
/// nursery allocator instead: `alloc_oldgen_typed` returns
/// `base + GcHeader::SIZE`, which would shift every field offset the descr
/// carries.
///
/// Non-GC descrs (`type_id == 0`, raw buffers) keep the plain zeroed malloc.
/// A typed descr never does: absence/failure of its collector is NULL, the
/// translated `do_malloc_fixedsize_clear` failure edge.
fn bh_alloc_struct(sizedescr: &majit_jitcode::jitcode::BhDescr) -> *mut libc::c_void {
    // `MAJIT_GC_STRESS_TRACE_ALLOC`: a minor before the blackhole
    // materializes a struct, so a resume constant captured as an address
    // dies here instead of in a later collection.
    if majit_gc::gc_stress_trace_alloc_enabled()
        && majit_gc::gc_sync::is_initialized()
        && !majit_gc::gc_sync::in_gc_op()
    {
        majit_gc::gc_sync::gc_op(|g| g.do_collect_nursery());
    }
    let size = sizedescr.as_size();
    let type_id = sizedescr.resolve_gc_tid();
    let gc_ptr = if sizedescr.is_headerless() {
        majit_gc::alloc_nursery_headerless_no_collect(size).0
    } else {
        match type_id {
            0 => 0,
            type_id => dynasm_alloc_oldgen_typed(type_id, size).0,
        }
    };
    if gc_ptr != 0 {
        return gc_ptr as *mut libc::c_void;
    }
    // `GcLLDescr_framework._bh_malloc` allocates a GC struct through
    // `do_malloc_fixedsize_clear`; NULL is its OOM result and is converted to
    // `MemoryError` by blackhole.py `_get_method`.  Falling through to a raw
    // block for a typed descr loses the GC header and tracing layout.
    // A process with no collector (grain) has no header to lose: the blackhole
    // still has to run the `new` and continue to the next `jit_merge_point`.
    // A collector installed through the thread's GC box (`set_gc_allocator`)
    // without the `gc_sync` singleton is still a collector: its NULL is the
    // OOM result above, not a missing allocator.
    let collector_installed = majit_gc::gc_sync::is_initialized() || majit_gc::gc_box_installed();
    if type_id != 0 && !sizedescr.is_headerless() && collector_installed {
        return std::ptr::null_mut();
    }
    let ptr = unsafe { libc::malloc(size) };
    if !ptr.is_null() {
        unsafe { libc::memset(ptr, 0, size) };
    }
    ptr
}

/// User-level `gc.collect(n)` trampoline — drives
/// `GcAllocator::collect_generation` on the active dynasm-owned GC.
/// `interp_gc.py collect` runs `rgc.collect()` from app-level `gc.collect`;
/// this is the dynasm backend's edge of that path. Safety: callers must be at a safepoint
/// where every live PyObjectRef is either in a registered root, on the
/// Python value stack, or on the shadow stack — Rust-stack PyObjectRef
/// in nursery would dangle after the embedded minor cycle.
fn dynasm_collect_generation(generation: i64) {
    if gc_box::with_mut(|g| g.collect_generation(generation)).is_some() {
        return;
    }
    majit_gc::gc_sync::gc_op(|g| g.collect_generation(generation));
}

fn dynasm_collect_step() -> majit_gc::GcStepTransition {
    if let Some(transition) = gc_box::with_mut(|g| g.collect_step()) {
        return transition;
    }
    majit_gc::gc_sync::gc_op(|g| g.collect_step())
}

/// Takes `format_args!` rather than a built `String`: the callers sit on the
/// per-residual-call and per-trace-entry paths, where formatting the site name
/// for a diagnostic that is off costs a heap allocation every time.
#[inline]
fn debug_validate_oldgen_freeblocks(site: std::fmt::Arguments<'_>) {
    if !crate::gc_freelist_diag_enabled() {
        return;
    }
    debug_validate_oldgen_freeblocks_slow(site);
}

#[cold]
#[inline(never)]
fn debug_validate_oldgen_freeblocks_slow(site: std::fmt::Arguments<'_>) {
    let site = site.to_string();
    if gc_box::with_ref(|g| g.debug_validate_oldgen_freeblocks(&site)).is_some() {
        return;
    }
    majit_gc::gc_sync::gc_op(|g| g.debug_validate_oldgen_freeblocks(&site));
}

pub extern "C" fn dynasm_debug_validate_oldgen_freeblocks(site: u64, frame: usize) {
    // The call itself is only emitted under `MAJIT_TRACE_CALL_DIAG`, which is
    // what selects the failure-frame report below. The freelist walk is the
    // only half `MAJIT_GC_FREELIST_DIAG` owns, and it gates itself — so no
    // early return here, or enabling one diagnostic would need the other.
    if site >= 1_000_000 {
        let top = majit_gc::shadow_stack::jf_top_ptr().0;
        eprintln!(
            "[failure-frame] site={site} frame={frame:#x} top={top:#x} registered={} top_registered={}",
            majit_gc::shadow_stack::is_libc_jitframe(frame),
            majit_gc::shadow_stack::is_libc_jitframe(top),
        );
    }
    debug_validate_oldgen_freeblocks(format_args!("after residual site {site}"));
}

fn dynasm_get_objects(generation: i8, visitor: majit_gc::GetObjectsVisitorFn) {
    let mut visit = visitor;
    if gc_box::with_mut(|g| g.get_objects(generation, &mut visit)).is_some() {
        return;
    }
    majit_gc::gc_sync::gc_op(|g| g.get_objects(generation, &mut visit));
}

fn dynasm_get_referents(obj: majit_ir::GcRef, visitor: majit_gc::GetObjectsVisitorFn) {
    let mut visit = visitor;
    if gc_box::with_mut(|g| g.get_referents(obj, &mut visit)).is_some() {
        return;
    }
    // See `MajitGc::get_referents`: the fallback enters the collector, so the
    // argument is published and reloaded rather than passed as the raw local.
    majit_gc::gc_sync::gc_op_with_root(obj, |g, obj| g.get_referents(obj, &mut visit));
}

fn dynasm_subgraph_has_pending_finalizer(roots: &[majit_ir::GcRef]) -> bool {
    if let Some(found) = gc_box::with_mut(|g| g.subgraph_has_pending_finalizer(roots)) {
        return found;
    }
    majit_gc::gc_sync::gc_op(|g| g.subgraph_has_pending_finalizer(roots))
}

fn dynasm_is_tracked(obj: majit_ir::GcRef) -> bool {
    if let Some(tracked) = gc_box::with_mut(|g| g.is_tracked(obj)) {
        return tracked;
    }
    majit_gc::gc_sync::gc_op_with_root(obj, |g, obj| g.is_tracked(obj))
}

fn dynasm_get_gcflag_extra(obj: majit_ir::GcRef) -> bool {
    gc_box::with_mut(|g| g.get_gcflag_extra(obj))
        .unwrap_or_else(|| majit_gc::gc_sync::gc_op(|g| g.get_gcflag_extra(obj)))
}

fn dynasm_toggle_gcflag_extra(obj: majit_ir::GcRef) {
    if gc_box::with_mut(|g| g.toggle_gcflag_extra(obj)).is_none() {
        majit_gc::gc_sync::gc_op(|g| g.toggle_gcflag_extra(obj));
    }
}

fn dynasm_get_gcflag_dummy(obj: majit_ir::GcRef) -> bool {
    gc_box::with_mut(|g| g.get_gcflag_dummy(obj))
        .unwrap_or_else(|| majit_gc::gc_sync::gc_op(|g| g.get_gcflag_dummy(obj)))
}

fn dynasm_get_rpy_memory_usage(obj: majit_ir::GcRef) -> Option<usize> {
    if let Some(size) = gc_box::with_mut(|g| g.get_rpy_memory_usage(obj)) {
        return size;
    }
    majit_gc::gc_sync::gc_op_with_root(obj, |g, obj| g.get_rpy_memory_usage(obj))
}

fn dynasm_get_rpy_type_index(obj: majit_ir::GcRef) -> Option<usize> {
    if let Some(index) = gc_box::with_mut(|g| g.get_rpy_type_index(obj)) {
        return index;
    }
    majit_gc::gc_sync::gc_op_with_root(obj, |g, obj| g.get_rpy_type_index(obj))
}

fn dynasm_get_rpy_roots(visitor: majit_gc::GetObjectsVisitorFn) -> bool {
    let mut visit = visitor;
    if let Some(supported) = gc_box::with_mut(|g| g.get_rpy_roots(&mut visit)) {
        return supported;
    }
    majit_gc::gc_sync::gc_op(|g| g.get_rpy_roots(&mut visit))
}

fn dynasm_get_rpy_referents(obj: majit_ir::GcRef, visitor: majit_gc::GetObjectsVisitorFn) -> bool {
    let mut visit = visitor;
    if let Some(supported) = gc_box::with_mut(|g| g.get_rpy_referents(obj, &mut visit)) {
        return supported;
    }
    majit_gc::gc_sync::gc_op_with_root(obj, |g, obj| g.get_rpy_referents(obj, &mut visit))
}

fn dynasm_is_app_level_object(obj: majit_ir::GcRef) -> bool {
    if let Some(is_object) = gc_box::with_mut(|g| g.is_app_level_object(obj)) {
        return is_object;
    }
    majit_gc::gc_sync::gc_op_with_root(obj, |g, obj| g.is_app_level_object(obj))
}

fn dynasm_dump_rpy_heap(fd: i32) -> Result<bool, i32> {
    if let Some(result) = gc_box::with_mut(|g| g.dump_rpy_heap(fd)) {
        return result;
    }
    majit_gc::gc_sync::gc_op(|g| g.dump_rpy_heap(fd))
}

fn dynasm_get_typeids_text() -> Option<Vec<u8>> {
    with_dynasm_active_gc(|g| g.get_typeids_text()).flatten()
}

fn dynasm_get_typeids_list() -> Option<Vec<usize>> {
    with_dynasm_active_gc(|g| g.get_typeids_list()).flatten()
}

fn dynasm_add_memory_pressure(size: isize, object: GcRef) {
    if gc_box::with_mut(|g| g.add_memory_pressure(size, object)).is_some() {
        return;
    }
    if object.is_null() {
        majit_gc::gc_sync::gc_op(|g| g.add_memory_pressure(size, object));
    } else {
        majit_gc::gc_sync::gc_op_with_root(object, |g, object| g.add_memory_pressure(size, object));
    }
}

fn dynasm_maybe_collect_for_external_malloc(totalsize: usize) -> bool {
    if let Some(result) = gc_box::with_mut(|g| g.maybe_collect_for_external_malloc(totalsize)) {
        return result;
    }
    majit_gc::gc_sync::gc_op(|g| g.maybe_collect_for_external_malloc(totalsize))
}

fn dynasm_total_memory_pressure() -> isize {
    if let Some(result) = gc_box::with_mut(|g| g.total_memory_pressure()) {
        return result;
    }
    majit_gc::gc_sync::gc_op(|g| g.total_memory_pressure())
}

/// Non-moving old-gen-only major. Reclaims stable-allocated interp int/float
/// without moving the nursery, so the interpreter safepoint can fire it under
/// an active JIT (nursery non-empty) — unlike [`dynasm_collect_generation`], whose
/// embedded minor would relocate a Rust-stack nursery PyObjectRef.
fn dynasm_collect_oldgen_nonmoving() {
    if gc_box::with_mut(|g| g.collect_oldgen_nonmoving()).is_some() {
        return;
    }
    majit_gc::gc_sync::gc_op(|g| g.collect_oldgen_nonmoving());
}

fn dynasm_register_finalizer(fq_index: usize, obj: GcRef, trigger: majit_gc::FinalizerTriggerFn) {
    with_dynasm_active_gc_mut(|gc| gc.register_finalizer(fq_index, obj, trigger));
}

fn dynasm_finalizer_next_dead(fq_index: usize) -> Option<GcRef> {
    with_dynasm_active_gc_mut(|gc| gc.finalizer_next_dead(fq_index)).flatten()
}

/// Report `(oldgen_total, nursery_used)` for the interpreter GC safepoint.
fn dynasm_heap_stats() -> (usize, usize) {
    if let Some(r) = gc_box::with_mut(|g| g.heap_byte_stats()) {
        return r;
    }
    majit_gc::gc_sync::gc_op(|g| g.heap_byte_stats())
}

fn dynasm_gc_memory_stats() -> majit_gc::GcMemoryStats {
    if let Some(r) = gc_box::with_mut(|g| g.gc_memory_stats()) {
        return r;
    }
    majit_gc::gc_sync::gc_op(|g| g.gc_memory_stats())
}

/// Report whether the GC wants a major collection, for the interpreter GC
/// safepoint (incminimark.py `threshold_reached`).
fn dynasm_major_threshold_reached() -> bool {
    if let Some(r) = gc_box::with_mut(|g| g.major_threshold_reached()) {
        return r;
    }
    majit_gc::gc_sync::gc_op(|g| g.major_threshold_reached())
}

fn dynasm_minor_collections_since_major() -> usize {
    if let Some(r) = gc_box::with_mut(|g| g.minor_collections_since_major()) {
        return r;
    }
    majit_gc::gc_sync::gc_op(|g| g.minor_collections_since_major())
}

/// Host-side root-register trampoline. Bridges
/// `majit_gc::gc_add_root` to the active backend's `RootSet`.
///
/// # Safety
/// Caller must keep `slot` valid until [`dynasm_gc_remove_root`] is
/// called with the same pointer.
unsafe fn dynasm_gc_add_root(slot: *mut GcRef) {
    if gc_box::with_mut(|g| unsafe { g.add_root(slot) }).is_some() {
        return;
    }
    unsafe { majit_gc::gc_sync::gc_op_add_root(slot) };
}

/// Companion to [`dynasm_gc_add_root`].
fn dynasm_gc_remove_root(slot: *mut GcRef) {
    if gc_box::with_mut(|g| g.remove_root(slot)).is_some() {
        return;
    }
    majit_gc::gc_sync::gc_op(|g| g.remove_root(slot));
}

/// Host-side write-barrier trampoline for GC-managed objects updated
/// outside compiled code.
fn dynasm_gc_write_barrier_before_move(obj: GcRef) {
    if gc_box::with_mut(|g| g.writebarrier_before_move(obj)).is_some() {
        return;
    }
    // Root-free, for the reason `MiniMarkGc::write_barrier` (majit-gc/src/lib.rs) states.
    majit_gc::gc_sync::gc_op(|g| g.writebarrier_before_move(obj.0));
}

fn dynasm_gc_write_barrier_from_array(obj: GcRef, index: usize) {
    if gc_box::with_mut(|g| g.write_barrier_from_array(obj, index)).is_some() {
        return;
    }
    majit_gc::gc_sync::gc_op(|g| g.write_barrier_from_array(obj, index));
}

fn dynasm_gc_writebarrier_before_copy(
    source: GcRef,
    dest: GcRef,
    source_start: usize,
    dest_start: usize,
    length: usize,
) -> bool {
    if let Some(handled) = gc_box::with_mut(|g| {
        g.writebarrier_before_copy(source, dest, source_start, dest_start, length)
    }) {
        return handled;
    }
    if majit_gc::gc_sync::is_initialized() {
        return majit_gc::gc_sync::gc_op(|g| {
            majit_gc::GcAllocator::writebarrier_before_copy(
                g,
                source,
                dest,
                source_start,
                dest_start,
                length,
            )
        });
    }
    true
}

fn dynasm_gc_write_barrier(obj: GcRef) {
    if gc_box::with_mut(|g| g.write_barrier(obj)).is_some() {
        return;
    }
    // Root-free, for the reason `MiniMarkGc::write_barrier` (majit-gc/src/lib.rs) states.
    majit_gc::gc_sync::gc_op(|g| g.write_barrier(obj));
}

fn dynasm_gc_write_barrier_managed(obj: GcRef) {
    if gc_box::with_mut(|g| g.write_barrier_managed(obj)).is_some() {
        return;
    }
    majit_gc::gc_sync::gc_op(|g| g.write_barrier_managed(obj));
}

fn dynasm_id_or_identityhash(addr: usize) -> usize {
    // `boehm.py` `ll_identityhash`: `h = ~cast_adr_to_int(addr)`.
    // `GcLLDescr_boehm.gcrootmap` is `None`, which is `!collector_installed`.
    if !majit_gc::collector_installed() {
        return !addr;
    }
    // `Trace::refresh_from_gc` calls this from a root walk. `gc_op` already
    // holds the collector; a second one panics in debug
    // (`reentrant &mut gc_op`).
    if majit_gc::gc_sync::in_gc_op() {
        return majit_gc::gc_sync::gc_query_reentrant(|g| g.id_or_identityhash_reentrant(addr));
    }
    // A box whose borrow is already held by an in-progress alloc answers with
    // the raw `addr`, not with the singleton's id: this is a top-level op, so
    // the busy borrow means the box is mid-allocation, not that it is absent.
    if let Some(r) = gc_box::with_mut_or_busy(addr, |gc| gc.id_or_identityhash(addr)) {
        return r;
    }
    // minimark.py `id_or_identityhash`: only a nursery object moves to a
    // shadow. With no collector on this thread and no process singleton,
    // there is no nursery the object could be in, so the identity is `addr`.
    if majit_gc::gc_sync::is_initialized() {
        return majit_gc::gc_sync::gc_op(|g| g.id_or_identityhash(addr));
    }
    addr
}

/// Host-side `is_managed_heap_object` trampoline. Lets host-side
/// allocators (`pyre_object::dealloc_items_block`) discriminate
/// `try_gc_alloc_stable`-allocated blocks from `std::alloc`-backed
/// fallback blocks during the L1/L2 stepping-stone window. Returns
/// `false` when no GC is installed (caller falls through to
/// `std::alloc::dealloc`).
fn dynasm_gc_owns_object(addr: usize) -> bool {
    // This query can fire reentrantly from an extra-root walker mid-collection,
    // so both arms are read-only: `with_reentrant_ref` for the box, and the
    // reentrant singleton read for everything else.
    if let Some(r) = gc_box::with_reentrant_ref(|gc| gc.is_managed_heap_object(addr)) {
        return r;
    }
    majit_gc::gc_sync::is_initialized()
        && majit_gc::gc_sync::gc_query_reentrant(|g| g.is_managed_heap_object(addr))
}

/// `llop.shrink_array`.  This mutates a GC-owned object's length word, so it
/// follows allocation and write barriers through the exclusive GC-operation
/// path.  In particular, it must not use `gc_query_reentrant`: that helper is
/// reserved for bounded read-only queries made while a collection may already
/// hold the collector's `&mut`.
fn dynasm_gc_shrink_array(addr: usize, smaller_length: usize) -> bool {
    with_dynasm_active_gc_mut(|gc| gc.shrink_array(addr, smaller_length)).unwrap_or(false)
}

fn dynasm_gc_varsize_layout(addr: usize) -> Option<majit_gc::GcVarSizeLayout> {
    let obj = GcRef(addr);
    if let Some(r) = gc_box::with_reentrant_ref(|gc| gc.varsize_layout(obj)) {
        return r;
    }
    if majit_gc::gc_sync::is_initialized() {
        majit_gc::gc_sync::gc_query_reentrant(|g| g.varsize_layout(obj))
    } else {
        None
    }
}

fn dynasm_gc_is_nursery_object(addr: usize) -> bool {
    if let Some(r) = gc_box::with_reentrant_ref(|gc| gc.is_nursery_object(addr)) {
        return r;
    }
    majit_gc::gc_sync::is_initialized()
        && majit_gc::gc_sync::gc_query_reentrant(|g| g.is_nursery_object(addr))
}

/// `gc.py:51` malloc-helper OOM signaling.
///
/// `do_malloc_fixedsize_clear` raises `MemoryError` on failure;
/// translation lowers that to "store the singleton in
/// `cpu.pos_exc_value`, return NULL".  pyre malloc helpers return 0
/// directly, so this wrapper performs the equivalent thread-local
/// store before propagating the NULL up to the JIT-emitted
/// `CHECK_MEMORY_ERROR` (`x86/assembler.py:2334`,
/// `aarch64/assembler.py:1845`).  When no provider is registered
/// (typical for unit tests), `JIT_EXC_VALUE` stays 0 and Layer 4's
/// `cast_instance_to_gcref(memory_error)` fallback in
/// `compile.py:1095` will fill in the gap.
#[inline]
fn oom_signal_if_zero(result: u64) -> u64 {
    if result == 0 {
        let v = majit_backend::memory_error_singleton_ref();
        if v != 0 {
            // llmodel.py `_store_exception` parity — `jit_exc_raise`
            // sets both `JIT_EXC_VALUE` and the typeptr-derived
            // `JIT_EXC_TYPE`, matching the translated `raise MemoryError`
            // sequence inside RPython's `do_malloc_fixedsize_clear`.
            crate::jit_exc_raise(v);
        }
    }
    result
}

/// _build_malloc_slowpath parity: nursery overflow slow path.
///
/// Called from JIT-compiled code when inline nursery bump allocation
/// fails (new_free > nursery_top). total_size includes GcHeader.
///
/// Returns payload pointer (after GcHeader), matching fast-path semantics.
pub extern "C" fn dynasm_nursery_slowpath(total_size: u64) -> u64 {
    let gc_hdr = majit_gc::header::GcHeader::SIZE;
    let result =
        with_dynasm_active_gc_mut(|gc| gc.alloc_nursery(total_size as usize - gc_hdr).0 as u64);
    let ptr = result.unwrap_or_else(|| unsafe {
        // `libc::calloc` returns NULL on real host OOM; preserve that
        // NULL through to the trampoline's `TEST rax, rax; JZ propagate`
        // (assembler.py:300-302).  Adding `gc_hdr` unconditionally
        // masked OOM as a "valid" near-zero pointer and let the JIT
        // continue past the failure.
        let raw = libc::calloc(1, total_size as usize) as u64;
        if raw == 0 { 0 } else { raw + gc_hdr as u64 }
    });
    if crate::gc_freelist_diag_enabled() {
        let nursery = dynasm_gc_is_nursery_object(ptr as usize);
        eprintln!("[malloc-slow] total={total_size} payload={ptr:#x} nursery={nursery}");
        debug_validate_oldgen_freeblocks(format_args!("after malloc slowpath"));
    }
    if majit_ir::debug::have_debug_prints() {
        majit_ir::debug::log_one(
            "jit-backend",
            &format!("nursery-frame total_size={total_size} payload=0x{ptr:x}"),
        );
    }
    ptr
}

/// `malloc_cond_varsize` headerless slow path.
///
/// The assembler passes the length, not a wrapping `length * itemsize +
/// base_size + 7`. `gc.py` `malloc_cond_varsize` sends that slow path to
/// `ovfcheck`; a negative length or an overflowing product returns null
/// so the callsite's `emit_propagate_memory_error_if_null` raises.
pub extern "C" fn dynasm_nursery_slowpath_headerless_varsize(
    length: i64,
    itemsize: u64,
    base_size: u64,
) -> u64 {
    let Some(size) = headerless_varsize_alloc_size(length, itemsize, base_size) else {
        return 0;
    };
    dynasm_nursery_slowpath_headerless(size)
}

fn headerless_varsize_alloc_size(length: i64, itemsize: u64, base_size: u64) -> Option<u64> {
    if length < 0 {
        return None;
    }
    let bytes = (length as u64)
        .checked_mul(itemsize)?
        .checked_add(base_size)?
        .checked_add(7)?;
    Some(bytes & !7)
}

/// Headerless nursery overflow slow path.
///
/// `size` is the exact allocation size.  Returns the allocation base,
/// matching the headerless fast path.  Headered collectors such as MiniMarkGC
/// are not headerless-aware and must fail via `alloc_nursery_headerless`.
pub extern "C" fn dynasm_nursery_slowpath_headerless(size: u64) -> u64 {
    let ptr = with_dynasm_active_gc_mut(|gc| gc.alloc_nursery_headerless(size as usize).0 as u64)
        .unwrap_or_else(|| unsafe { libc::calloc(1, size as usize) as u64 });
    if majit_ir::debug::have_debug_prints() {
        majit_ir::debug::log_one(
            "jit-backend",
            &format!("nursery-headerless size={size} base=0x{ptr:x}"),
        );
    }
    ptr
}

/// malloc_cond_varsize_frame slow path for JITFRAME allocation.
///
/// `frame_size` is `jfi_frame_size`: bytes from the JITFRAME payload base
/// through the trailing array, i.e. it excludes the GC header that the
/// allocator prepends internally.
pub extern "C" fn dynasm_nursery_slowpath_jitframe(frame_size: u64) -> u64 {
    with_gc_ll_descr(|gc| malloc_jitframe(gc, frame_size as usize)) as u64
}

/// `_build_malloc_slowpath(kind='var')` parity: varsize nursery
/// overflow.  Called with `(base_size, item_size, length)`; returns the
/// payload pointer.
///
///
/// aarch64 only. x86 builds the per-kind trampolines of
/// `_build_malloc_slowpath(kind)` (`'var'`, `'str'`, `'unicode'`), which call
/// `dynasm_malloc_array` / `dynasm_malloc_str` / `dynasm_malloc_unicode`.
/// The aarch64 `malloc_cond_varsize` still collapses the three kinds into
/// this one helper.
pub extern "C" fn dynasm_nursery_slowpath_varsize(
    base_size: u64,
    item_size: u64,
    length: u64,
    type_id: u64,
) -> u64 {
    let gc_hdr = majit_gc::header::GcHeader::SIZE;
    let result = with_dynasm_active_gc_mut(|gc| {
        gc.alloc_varsize_typed(
            type_id as u32,
            base_size as usize,
            item_size as usize,
            length as usize,
        )
        .0 as u64
    });
    result.unwrap_or_else(|| {
        let Some(total) = (item_size as usize)
            .checked_mul(length as usize)
            .and_then(|var_size| (base_size as usize).checked_add(var_size))
            .and_then(|payload_size| gc_hdr.checked_add(payload_size))
        else {
            return 0;
        };
        unsafe {
            // `libc::calloc` returns NULL on real OOM; the previous
            // unconditional `raw + gc_hdr` masked failure as a tiny
            // non-zero "valid" payload pointer and let the JIT continue
            // past the failure.  Mirror `dynasm_nursery_slowpath`'s
            // OOM-null preservation so the caller's TEST/JZ propagate
            // path can fire on real OOM.
            let raw = libc::calloc(1, total) as u64;
            if raw == 0 { 0 } else { raw + gc_hdr as u64 }
        }
    })
}

fn dynasm_raw_varsize_alloc_typed_and_set_len(
    type_id: u32,
    base_size: usize,
    item_size: usize,
    length_ofs: usize,
    length: usize,
) -> u64 {
    let Some(var_bytes) = item_size.checked_mul(length) else {
        return 0;
    };
    let Some(payload_size) = base_size.checked_add(var_bytes) else {
        return 0;
    };
    let Some(total_size) = majit_gc::header::GcHeader::SIZE.checked_add(payload_size) else {
        return 0;
    };
    unsafe {
        let raw = libc::calloc(1, total_size) as *mut u8;
        if raw.is_null() {
            return 0;
        }
        *(raw as *mut majit_gc::header::GcHeader) = majit_gc::header::GcHeader::new(type_id);
        let obj = raw.add(majit_gc::header::GcHeader::SIZE);
        *(obj.add(length_ofs) as *mut usize) = length;
        obj as u64
    }
}

fn dynasm_alloc_varsize_typed_and_set_len(
    type_id: u32,
    base_size: usize,
    item_size: usize,
    length_ofs: usize,
    length: usize,
) -> u64 {
    dynasm_alloc_varsize_typed_and_set_len_maybe_clear(
        type_id, base_size, item_size, length_ofs, length, false,
    )
}

fn dynasm_fixedsize_hook_varsize(
    base_size: usize,
    item_size: usize,
    length_ofs: usize,
    length: usize,
) -> Option<u64> {
    let addr = majit_gc::malloc_fixedsize_addr();
    if addr == 0 {
        return None;
    }
    let Some(var_bytes) = item_size.checked_mul(length) else {
        return Some(0);
    };
    let Some(payload_size) = base_size.checked_add(var_bytes) else {
        return Some(0);
    };
    let func: extern "C" fn(usize) -> *mut u8 = unsafe { std::mem::transmute(addr) };
    let ptr = func(payload_size.max(1));
    if ptr.is_null() {
        return Some(0);
    }
    unsafe {
        *ptr.add(length_ofs).cast::<usize>() = length;
    }
    Some(ptr as u64)
}

fn dynasm_alloc_varsize_typed_and_set_len_maybe_clear(
    type_id: u32,
    base_size: usize,
    item_size: usize,
    length_ofs: usize,
    length: usize,
    clear: bool,
) -> u64 {
    let result = with_dynasm_active_gc_mut(|gc| {
        let obj = gc.alloc_varsize_typed(type_id, base_size, item_size, length);
        if obj.is_null() {
            0
        } else {
            let payload = obj.0 as *mut u8;
            if clear {
                let nbytes = base_size.saturating_add(item_size.saturating_mul(length));
                unsafe {
                    core::ptr::write_bytes(payload, 0, nbytes);
                }
            }
            unsafe {
                *payload.add(length_ofs).cast::<usize>() = length;
            }
            obj.0 as u64
        }
    });
    result.unwrap_or_else(|| {
        // `GcLLDescr.malloc_fixedsize` when the portal published one
        // (`set_malloc_fixedsize`). The block is the caller's heap, not a
        // libc array: length is still stamped at `lendescr`.
        if let Some(hooked) =
            dynasm_fixedsize_hook_varsize(base_size, item_size, length_ofs, length)
        {
            return hooked;
        }
        let raw = dynasm_raw_varsize_alloc_typed_and_set_len(
            type_id, base_size, item_size, length_ofs, length,
        );
        // The raw fallback already calloc's, so the length stamp is enough.
        let _ = clear;
        raw
    })
}

/// Backend leftover path for `NEW_ARRAY` / `NEW_ARRAY_CLEAR` when GC rewrite
/// did not lower the op. Must allocate a typed GC array and stamp length at
/// the descr's `lendescr` offset — libc malloc + store-at-+8 is the RPython
/// string header, not `ItemsBlock` / `GcArray` (length at offset 0).
pub extern "C" fn dynasm_malloc_new_array(
    base_size: u64,
    item_size: u64,
    length_ofs: u64,
    type_id: u64,
    num_elem: u64,
    clear: u64,
) -> u64 {
    oom_signal_if_zero(dynasm_alloc_varsize_typed_and_set_len_maybe_clear(
        type_id as u32,
        base_size as usize,
        item_size as usize,
        length_ofs as usize,
        num_elem as usize,
        clear != 0,
    ))
}

pub extern "C" fn dynasm_malloc_array(item_size: u64, type_id: u64, num_elem: u64) -> u64 {
    oom_signal_if_zero(dynasm_alloc_varsize_typed_and_set_len(
        type_id as u32,
        std::mem::size_of::<usize>(),
        item_size as usize,
        0,
        num_elem as usize,
    ))
}

pub extern "C" fn dynasm_malloc_array_nonstandard(
    base_size: u64,
    item_size: u64,
    length_ofs: u64,
    type_id: u64,
    num_elem: u64,
) -> u64 {
    oom_signal_if_zero(dynasm_alloc_varsize_typed_and_set_len(
        type_id as u32,
        base_size as usize,
        item_size as usize,
        length_ofs as usize,
        num_elem as usize,
    ))
}

/// Typed varsize allocation for RPython `STR` / `UNICODE`. The string layout
/// always stores its length at byte offset 8; the ArrayDescr supplies the
/// collector type id plus base/item sizes.
pub extern "C" fn dynasm_malloc_lowlevel_string(
    type_id: u64,
    base_size: u64,
    item_size: u64,
    length: u64,
) -> u64 {
    oom_signal_if_zero(dynasm_alloc_varsize_typed_and_set_len(
        type_id as u32,
        base_size as usize,
        item_size as usize,
        BUILTIN_STRING_LEN_OFFSET,
        length as usize,
    ))
}

/// Old-generation twin of [`dynasm_malloc_array`], selected by
/// `gen_malloc_array` for a `non_moving` array descr.  Same signature.
pub extern "C" fn dynasm_malloc_array_oldgen(item_size: u64, type_id: u64, num_elem: u64) -> u64 {
    oom_signal_if_zero(dynasm_alloc_oldgen_varsize_typed_and_set_len(
        type_id as u32,
        std::mem::size_of::<usize>(),
        item_size as usize,
        0,
        num_elem as usize,
    ))
}

/// Old-generation twin of [`dynasm_malloc_array_nonstandard`].
pub extern "C" fn dynasm_malloc_array_nonstandard_oldgen(
    base_size: u64,
    item_size: u64,
    length_ofs: u64,
    type_id: u64,
    num_elem: u64,
) -> u64 {
    oom_signal_if_zero(dynasm_alloc_oldgen_varsize_typed_and_set_len(
        type_id as u32,
        base_size as usize,
        item_size as usize,
        length_ofs as usize,
        num_elem as usize,
    ))
}

fn dynasm_raw_fixedsize_alloc_typed(type_id: u32, size: usize) -> u64 {
    let Some(total_size) = majit_gc::header::GcHeader::SIZE.checked_add(size) else {
        return 0;
    };
    unsafe {
        let raw = libc::calloc(1, total_size) as *mut u8;
        if raw.is_null() {
            return 0;
        }
        *(raw as *mut majit_gc::header::GcHeader) = majit_gc::header::GcHeader::new(type_id);
        let obj = raw.add(majit_gc::header::GcHeader::SIZE);
        obj as u64
    }
}

fn dynasm_alloc_oldgen_varsize_typed_and_set_len(
    type_id: u32,
    base_size: usize,
    item_size: usize,
    length_ofs: usize,
    length: usize,
) -> u64 {
    let Some(payload_size) = item_size
        .checked_mul(length)
        .and_then(|var_size| base_size.checked_add(var_size))
    else {
        return 0;
    };
    let obj = dynasm_alloc_oldgen_typed(type_id, payload_size);
    let ptr = obj.0;
    if ptr != 0 {
        unsafe {
            *((ptr as *mut u8).add(length_ofs) as *mut usize) = length;
        }
    }
    ptr as u64
}

/// The old-generation twin of [`dynasm_alloc_fixedsize_typed_or_raw`], for a
/// size descr that demands a non-moving address at any size.  Same OOM contract.
fn dynasm_alloc_oldgen_typed_or_raw(type_id: u32, payload_size: usize) -> u64 {
    let result = with_dynasm_active_gc_mut(|gc| {
        let obj = gc.alloc_oldgen_typed(type_id, payload_size);
        if obj.is_null() { 0 } else { obj.0 as u64 }
    });
    match result {
        None => dynasm_raw_fixedsize_alloc_typed(type_id, payload_size),
        Some(v) => v,
    }
}

fn dynasm_alloc_fixedsize_typed_or_raw(type_id: u32, payload_size: usize) -> u64 {
    let result = with_dynasm_active_gc_mut(|gc| {
        let obj = gc.alloc_nursery_typed(type_id, payload_size);
        if obj.is_null() { 0 } else { obj.0 as u64 }
    });
    // gc.py:51 contract: helper returns NULL on OOM so
    // CHECK_MEMORY_ERROR can convert it into a MemoryError.  Only
    // `None` (no active runtime — typical for unit tests) falls back
    // to a raw alloc; `Some(0)` (a real OOM from the registered
    // allocator) propagates as 0 unchanged, mirroring
    // `dynasm_nursery_slowpath` above.
    match result {
        None => dynasm_raw_fixedsize_alloc_typed(type_id, payload_size),
        Some(v) => v,
    }
}

/// gc.py `malloc_big_fixedsize(size, tid)` — fixed-size object
/// large enough to skip the nursery.
///
/// Upstream calls `do_malloc_fixedsize_clear`, which is the ordinary
/// `malloc_fixedsize`, so the size decides the arm: over `nonlarge_max` it
/// takes `external_malloc(typeid, 0, alloc_young=True)`.  `alloc_nursery_typed`
/// is that entry point here — it dispatches on size, and only its small arm is
/// the nursery — so the object is born *young* raw-malloced and can die at the
/// next minor.  Header is stamped with the type id so callers MUST
/// NOT emit a separate `gen_initialize_tid`.
pub extern "C" fn dynasm_malloc_big_fixedsize(size: u64, type_id: u64) -> u64 {
    // The CALL_R arg is `total = payload + GcHeader::SIZE` (built by
    // `handle_new` in rewrite.rs to include the GC header).  The
    // runtime allocators (`alloc_with_type`, `alloc_nursery_typed`)
    // and the raw fallback both prepend the GC header themselves and
    // expect a payload-only size, so subtract `HDR` here once —
    // mirroring `dynasm_nursery_slowpath` above
    // (`alloc_nursery(total_size - gc_hdr)`).
    let payload = (size as usize).saturating_sub(majit_gc::header::GcHeader::SIZE);
    // `GcLLDescr_boehm.malloc_fixedsize` (`rewrite.py gen_malloc_fixedsize`,
    // Boehm arm): no HDR, the block is the object, vtable at offset 0.
    // `handle_new` still passes the headered total; the payload is
    // `descr.size()`. Unset keeps `dynasm_alloc_fixedsize_typed_or_raw`.
    if let Some(ptr) = call_malloc_fixedsize(payload) {
        return oom_signal_if_zero(ptr as u64);
    }
    oom_signal_if_zero(dynasm_alloc_fixedsize_typed_or_raw(type_id as u32, payload))
}

/// `malloc_big_fixedsize`'s old-generation twin, selected by
/// `gen_malloc_fixedsize` for a `non_moving` size descr.  Takes the same
/// arguments and stamps the type id the same way; only the allocator differs.
pub extern "C" fn dynasm_malloc_big_fixedsize_oldgen(size: u64, type_id: u64) -> u64 {
    let payload = (size as usize).saturating_sub(majit_gc::header::GcHeader::SIZE);
    // Same `GcLLDescr_boehm.malloc_fixedsize` arm as
    // `dynasm_malloc_big_fixedsize`: one heap, no second header.
    if let Some(ptr) = call_malloc_fixedsize(payload) {
        return oom_signal_if_zero(ptr as u64);
    }
    oom_signal_if_zero(dynasm_alloc_oldgen_typed_or_raw(type_id as u32, payload))
}

/// gc.py `malloc_str(length)` — but the upstream closure captures
/// `str_type_id` from `self.str_descr.tid` at generate-time.  `extern
/// "C" fn` cannot capture, so the type id is threaded through the
/// CALL_R as an explicit Signed arg and the calldescr's first param is
/// it (see `make_malloc_str_calldescr`).
pub extern "C" fn dynasm_malloc_str(type_id: u64, length: u64) -> u64 {
    oom_signal_if_zero(dynasm_alloc_varsize_typed_and_set_len(
        type_id as u32,
        BUILTIN_STR_TOKEN_BASE_SIZE,
        1,
        BUILTIN_STRING_LEN_OFFSET,
        length as usize,
    ))
}

/// gc.py `malloc_unicode(length)` — see `dynasm_malloc_str` for the
/// closure-vs-extern type-id threading rationale.
pub extern "C" fn dynasm_malloc_unicode(type_id: u64, length: u64) -> u64 {
    oom_signal_if_zero(dynasm_alloc_varsize_typed_and_set_len(
        type_id as u32,
        BUILTIN_UNICODE_TOKEN_BASE_SIZE,
        4,
        BUILTIN_STRING_LEN_OFFSET,
        length as usize,
    ))
}

/// opassembler.py:956-976: non-array write barrier slow path, for the case
/// whose base is a JITFRAME.
///
/// `gc.write_barrier` re-checks null, nursery membership, heap membership and
/// the pyobject-header registry before touching the remembered set. The frame
/// is the one base that needs those: it can be a block built off the GC
/// (`jitframe::alloc_off_gc_jitframe`), and a forwarded nursery header reads
/// `0xFF` at the flag byte, so the inline test admits addresses the barrier
/// must still decline.
pub extern "C" fn dynasm_write_barrier(obj_ptr: u64) {
    with_dynasm_active_gc_mut(|gc| gc.write_barrier(majit_ir::GcRef(obj_ptr as usize)));
}

/// opassembler.py:956-976: non-array write barrier slow path, for an ordinary
/// store's base.
///
/// `gc.py get_write_barrier_fn` resolves to
/// `framework.py:538-544 gcdata.gc.remember_young_pointer`, whose own comment
/// is "We know that 'addr_struct' has GCFLAG_TRACK_YOUNG_PTRS so far"
/// (`incminimark.py`) — the inline test already made that true, so
/// the helper neither repeats it nor guards. The base here is whatever the GC
/// rewriter emitted `COND_CALL_GC_WB` for, which is a collector-allocated
/// object with a real header.
pub extern "C" fn dynasm_jit_remember_young_pointer(obj_ptr: u64) {
    with_dynasm_active_gc_mut(|gc| {
        gc.jit_remember_young_pointer(majit_ir::GcRef(obj_ptr as usize))
    });
}

/// opassembler.py:953-960: array write barrier slow path.
/// Calls jit_remember_young_pointer_from_array(obj) which handles
/// the CARDS_SET transition for HAS_CARDS arrays.
pub extern "C" fn dynasm_write_barrier_from_array(obj_ptr: u64) {
    with_dynasm_active_gc_mut(|gc| {
        gc.jit_remember_young_pointer_from_array(majit_ir::GcRef(obj_ptr as usize))
    });
}

/// llmodel.py `write_ref_at_mem`: the write barrier implied by the
/// framework GC transformer around every blackhole ref store
/// (`bh_setfield_gc_r`, `bh_setarrayitem_gc_r`, `bh_setinteriorfield_gc_r`).
/// The blackhole interpreter is not the JIT, so no inline TRACK_YOUNG_PTRS
/// test precedes the call. `is_managed_heap_object` first guards the
/// pyre-specific case where a reconstructed frame slot is not a GC-managed
/// object (RPython's frame structs/arrays always are), then `write_barrier`
/// flag-checks (gc.write_barrier → do_write_barrier). Mirrors cranelift's
/// `write_barrier_if_managed`.
fn dynasm_write_barrier_if_managed(obj_ptr: u64) {
    if obj_ptr == 0 {
        return;
    }
    with_dynasm_active_gc_mut(|gc| {
        let obj = majit_ir::GcRef(obj_ptr as usize);
        if gc.is_managed_heap_object(obj.0) {
            gc.write_barrier(obj);
        }
    });
}

/// `_build_frame_realloc_slowpath` parity (assembler.py:143-189):
/// JIT-side helper invoked when `_check_frame_depth` detects that
/// `jf_frame.length < expected_depth`.  Allocates a wider JITFRAME,
/// copies the live slots, threads `jf_forward = new_frame`, and
/// returns the new pointer; the JIT-emitted slowpath body then writes
/// `rbp = rax` so subsequent frame-relative loads/stores land on the
/// reallocated frame.
///
/// # Safety
/// - `old_jf` must be the live, resolved frame of the running loop/bridge.
/// - The caller (the JIT-emitted slowpath body) must have already
///   spilled all live registers into the old frame via
///   `_push_all_regs_to_jitframe`; this helper relies on the GC seeing
///   those slots through the gcmap pushed at the same site.
pub unsafe extern "C" fn dynasm_realloc_frame(
    old_jf: *mut JitFrame,
    expected_depth: isize,
) -> *mut JitFrame {
    let base_ofs = DynasmBackend::get_baseofs_of_frame_field() as isize;
    let new_jf = unsafe {
        majit_backend::jitframe::realloc_frame(
            old_jf,
            expected_depth,
            base_ofs,
            |size_bytes| malloc_jitframe_like_entry(size_bytes as usize),
            |new_jf| with_gc_ll_descr(|gc| jitframe_write_barrier(gc, new_jf)),
        )
    };
    // `frame.jf_forward = new_frame` is a GC-pointer store on the old
    // frame. `realloc_frame` documents this caller-side barrier, matching
    // the framework transform around llmodel.py.
    with_gc_ll_descr(|gc| jitframe_write_barrier(gc, old_jf));
    if crate::majit_log_enabled() {
        eprintln!(
            "[dynasm][realloc-frame] old={old_jf:p} new={new_jf:p} expected_depth={expected_depth}"
        );
    }
    new_jf
}

/// runner.py AbstractX86CPU — concrete Backend implementation.
pub struct DynasmBackend {
    /// `rpython/jit/backend/model.py self.tracker = CPUTotalTracker()`
    /// parity — per-instance `cpu.tracker` exposed via
    /// [`Backend::cpu_tracker`].  Held behind `Arc` so the same
    /// counters are shared with the paired `JitProfiler` (which
    /// borrows the same `Arc` during [`crate::pyjitpl::MetaInterp::new`]
    /// setup in metainterp) — reads through `Profiler.get_counter`
    /// and writes through [`majit_backend::record_compiled_loop_token`]
    /// / [`CompiledLoopToken::compiling_a_bridge`] hit one shared
    /// store.
    cpu_tracker: Arc<majit_backend::CpuTotalTracker>,
    /// `llsupport/asmmemmgr.py` `cpu.asmmemmgr` parity.  The
    /// handle is owned by this CPU/backend and shared with every compiled code
    /// block and per-CPU helper buffer; its retained inner arena is
    /// process-owned so worker-backend teardown cannot discard reusable pages.
    asm_memory_manager: Arc<majit_backend::AsmMemoryManager>,
    /// Next unique trace ID.
    next_trace_id: u64,
    /// Next header PC (green key).
    next_header_pc: u64,
    /// Constant pool for the next compilation, keyed by OpRef raw index.
    /// Each `Const` carries its value (read via `as_raw_i64()`) and type
    /// (read via `get_type()` for the GC rewriter's `v.type` check), so
    /// there is no separate type side-table.
    constants: majit_ir::ConstMap<majit_ir::Const>,
    /// llmodel.py:64-69 self.vtable_offset — byte offset of the typeptr
    /// field inside instance objects. None when gcremovetypeptr is enabled.
    vtable_offset: Option<usize>,
    /// Byte offset of the class word beside the type word. `None` until
    /// `Backend::set_w_class_offset`. `bh_new_with_vtable` writes
    /// `resolve_w_class_obj(vtable)` there when both are set.
    w_class_offset: Option<usize>,
    /// `llmodel.py` `AbstractLLCPU.subclassrange_min_offset`, from
    /// `rclass.OBJECT_VTABLE`. Byte offset of `subclassrange_min` inside
    /// the class object. `None` until the portal configures it.
    subclassrange_min_offset: Option<usize>,
    /// `compile.py:665` `setattr(cpu, name, descr)` per-cpu attachments,
    /// held in a heap-pinned `Arc<CpuDescrCell>` so the
    /// pointer baked into the CALL_ASSEMBLER helper call site
    /// (`compile_loop` / `compile_bridge`) stays valid even when
    /// `DynasmBackend` is moved (the metainterp stores it by value; tests
    /// hold stack-local `DynasmBackend::new()`).  Compiled traces clone
    /// the `Arc` into `CompiledCode` so the attachments outlive the
    /// owning backend — matches the lifetime guarantee RPython gets from
    /// `cpu` being a long-lived Python object.
    descr_attachments: crate::guard::CpuDescrHandle,
    /// Cell address of `done_with_this_frame_descr_int`, published when the
    /// singleton is attached. The entry compares `jf_descr` to this word;
    /// `descr_attachments`' lock is not taken on that path.
    done_int_cell: std::sync::atomic::AtomicUsize,
    /// Cell address of `done_with_this_frame_descr_ref`. Same role as
    /// [`Self::done_int_cell`] for `compile.py DoneWithThisFrameDescrRef`.
    done_ref_cell: std::sync::atomic::AtomicUsize,
    /// Arch-specific per-CPU state PyPy keeps on `Assembler386` /
    /// `AssemblerARM64` (e.g. `self.malloc_slowpath`,
    /// `self.propagate_exception_path` at `assembler.py:63,344` and
    /// `aarch64/assembler.py`).  PyPy's assembler is one-per-CPU;
    /// pyre's `Asm` is per-`compile_loop`/`compile_bridge`, so the
    /// per-CPU stash lives here instead.  See
    /// `crate::x86::cpu_ext::X86CpuExt` /
    /// `crate::aarch64::cpu_ext::Aarch64CpuExt`.
    arch_cpu_ext: ArchCpuExt,
    /// `llmodel.py` `AbstractLLCPU.gc_ll_descr` answers used by raw DONE
    /// (`make_execute_token` / `malloc_jitframe`). Replaced with the descr
    /// in [`Self::set_gc_allocator`]; [`JitframeDescrFacts::HOST`] when the
    /// descr has no `JITFRAME` type id.
    jitframe_facts: JitframeDescrFacts,
}

impl Default for DynasmBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl DynasmBackend {
    /// Legacy test-only entry point.  Production code routes the typed
    /// pool through `Backend::set_constants_pool`; this raw-`i64`
    /// helper is retained for in-crate tests that construct
    /// `IndexMap<u32, i64>` literals by hand. Each raw value is wrapped
    /// as a `ConstInt` — the only constant kind these fixtures build.
    pub fn set_constants(
        &mut self,
        constants: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher>,
    ) {
        self.constants = constants
            .iter()
            .map(|(&k, &v)| (k, majit_ir::Const::Int(v)))
            .collect();
    }

    #[inline]
    fn raw_mem_ptr(addr: i64, offset: i64) -> Option<usize> {
        if addr == 0 {
            majit_backend::note_null_mem_access();
            return None;
        }
        Some((addr as usize).wrapping_add(offset as usize))
    }

    /// llmodel.py read_int_at_mem(gcref, ofs, size, sign).
    fn read_int_at_mem(&self, addr: i64, offset: i64, size: usize, sign: bool) -> i64 {
        let Some(ptr) = Self::raw_mem_ptr(addr, offset) else {
            return 0;
        };
        unsafe {
            match (size, sign) {
                (1, true) => (ptr as *const i8).read_unaligned() as i64,
                (1, false) => (ptr as *const u8).read_unaligned() as i64,
                (2, true) => (ptr as *const i16).read_unaligned() as i64,
                (2, false) => (ptr as *const u16).read_unaligned() as i64,
                (4, true) => (ptr as *const i32).read_unaligned() as i64,
                (4, false) => (ptr as *const u32).read_unaligned() as i64,
                _ => (ptr as *const i64).read_unaligned(),
            }
        }
    }

    /// llmodel.py write_int_at_mem(gcref, ofs, size, newvalue).
    fn write_int_at_mem(&self, addr: i64, offset: i64, size: usize, newvalue: i64) {
        let Some(ptr) = Self::raw_mem_ptr(addr, offset) else {
            return;
        };
        unsafe {
            match size {
                1 => (ptr as *mut u8).write_unaligned(newvalue as u8),
                2 => (ptr as *mut u16).write_unaligned(newvalue as u16),
                4 => (ptr as *mut u32).write_unaligned(newvalue as u32),
                _ => (ptr as *mut i64).write_unaligned(newvalue),
            }
        }
    }

    /// llmodel.py read_float_at_mem(gcref, ofs).
    fn read_float_at_mem(&self, addr: i64, offset: i64) -> f64 {
        let Some(ptr) = Self::raw_mem_ptr(addr, offset) else {
            return 0.0;
        };
        unsafe { (ptr as *const f64).read_unaligned() }
    }

    /// llmodel.py write_float_at_mem(gcref, ofs, newvalue).
    fn write_float_at_mem(&self, addr: i64, offset: i64, newvalue: f64) {
        let Some(ptr) = Self::raw_mem_ptr(addr, offset) else {
            return;
        };
        unsafe { (ptr as *mut f64).write_unaligned(newvalue) }
    }

    pub fn new() -> Self {
        // `rpython/jit/backend/model.py` `AbstractCPU.__init__` parity:
        // the cpu is constructed with no attached descrs.  The
        // `DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`
        // singletons are attached later by
        // `compile.make_and_attach_done_descrs([self, cpu])` during
        // `MetaInterpStaticData.finish_setup` (pyjitpl.py).
        let asm_memory_stats = Arc::new(majit_backend::AsmMemoryManagerStats::default());
        let asm_memory_manager =
            majit_backend::AsmMemoryManager::new(Arc::clone(&asm_memory_stats));
        DynasmBackend {
            cpu_tracker: Arc::new(majit_backend::CpuTotalTracker::default()),
            asm_memory_manager: Arc::clone(&asm_memory_manager),
            next_trace_id: 1,
            next_header_pc: 0,
            constants: majit_ir::ConstMap::default(),
            vtable_offset: None,
            w_class_offset: None,
            subclassrange_min_offset: None,
            descr_attachments: Arc::new(crate::guard::CpuDescrCell::default()),
            done_int_cell: std::sync::atomic::AtomicUsize::new(0),
            done_ref_cell: std::sync::atomic::AtomicUsize::new(0),
            arch_cpu_ext: ArchCpuExt::new(asm_memory_manager),
            jitframe_facts: jitframe_facts_at_cpu_init(),
        }
    }

    /// Bridge entry-pointer lookup by source guard `(trace_id, fail_index_per_trace)`.
    /// Reads `CompiledLoopToken.compiled_bridge_addrs`, written when the
    /// matching `CompiledCode` is pushed onto `asmmemmgr_blocks`. Returns
    /// `0` if none — `assembler.py` treats `adr_jump_offset == 0`
    /// uniformly as "patched / no entry".
    ///
    /// The map is append-only and insert-overwrites, so retracing the same
    /// guard resolves to the most recently compiled bridge — that is the
    /// one `bridge_was_compiled` / `compiled_bridge_fail_descr_layouts`
    /// ask about and the one subsequent guard failures should dispatch into.
    pub fn lookup_bridge_addr(
        &self,
        token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> usize {
        token
            .compiled_loop_token_expect()
            .compiled_bridge_addrs
            .lock()
            .get(&(source_trace_id, source_fail_index))
            .copied()
            .unwrap_or(0)
    }

    /// Test helper: attach synthetic per-cpu `DoneWithThisFrame*` +
    /// `ExitFrameWithExceptionDescrRef` descrs, mirroring the state
    /// `MetaInterpStaticData.attach_descrs_to_cpu(cpu)` leaves the
    /// backend in at `finish_setup` (pyjitpl.py).  Production
    /// code reaches this state through `MetaInterp::new`; backend-
    /// only unit/integration tests that skip the metainterp call
    /// this to get a populated cpu before running `compile_loop`.
    pub fn attach_default_test_descrs(&mut self) {
        // `compile.py make_and_attach_done_descrs` +
        // `pyjitpl.py` `self.cpu.propagate_exception_descr = exc_descr`
        // parity: attach the class-distinct DoneWithThisFrameDescr* /
        // ExitFrameWithExceptionDescrRef plus a PropagateExceptionDescr
        // stand-in that the metainterp would mint through
        // `MetaInterp::new` / `MetaInterpStaticData.finish_setup`.
        // Backend-only tests that skip the metainterp call this to land
        // the same descrs the runtime classifier expects — and so that
        // `X86CpuExt::ensure_propagate_exception_path` can bake a
        // non-zero descr pointer into the propagate trampoline
        // (matching PyPy's `setup_once` ordering, which builds
        // trampolines after `finish_setup` has installed every CPU
        // descr).
        majit_backend::make_and_attach_done_descrs(&mut [self as &mut dyn Backend]);
        // `compile.py PropagateExceptionDescr` parity: backend-only
        // tests still need the same descr class identity that production
        // `MetaInterpStaticData.finish_setup` installs.
        let propagate: majit_ir::DescrRef = Arc::new(majit_backend::PropagateExceptionDescr::new());
        <Self as Backend>::set_propagate_exception_descr(self, propagate);
        // `pyjitpl.py self.cpu.setup_once()` parity — production
        // reaches `cpu.setup_once()` via `MetaInterpStaticData::_setup_once`
        // (`pyjitpl.py`) on first JIT entry, AFTER every descr
        // setter has run.  Backend-only tests bypass the metainterp gate,
        // so call `setup_once` here directly once the descrs are in place
        // — analogous to PyPy test helpers that explicitly call
        // `cpu.setup_once()` after manual descr attachment (e.g.
        // `rpython/jit/backend/ppc/test/test_regalloc_3.py`).
        <Self as Backend>::setup_once(self);
    }

    /// Active vtable_offset for the assembler to consume during codegen.
    pub fn vtable_offset(&self) -> Option<usize> {
        self.vtable_offset
    }

    /// `compile.py` `make_and_attach_done_descrs` parity: expose
    /// the six per-cpu-instance descrs as raw pointers for emission
    /// consumers (Assembler386 / AssemblerARM64 FINISH + CALL_ASSEMBLER
    /// sites).  The metainterp attaches the real descrs through
    /// `Backend::set_done_with_this_frame_descr_*` during
    /// `MetaInterpStaticData.finish_setup` (pyjitpl.py); before that
    /// the per-cpu fallback descrs installed by `DynasmBackend::new()`
    /// answer, so backend-only integration tests see distinct, non-zero
    /// pointers per result type without ever consulting per-thread state.
    pub(crate) fn attached_descr_ptrs(&self) -> crate::guard::AttachedDescrPtrs {
        self.descr_attachments.read().descr_ptrs()
    }

    /// `Arc` clone of the attachment handle, for compiled traces to
    /// keep alive alongside their executable buffer.  The `Arc`'s
    /// payload (the `CpuDescrCell`) lives at a heap-
    /// pinned address; `Arc::as_ptr(&clone)` is baked by emission into
    /// the CALL_ASSEMBLER helper call site as a compile-time immediate
    /// (same role as RPython's `self.cpu` closure capture in the
    /// translated code).  Cloning into `CompiledCode` keeps the pointee
    /// alive past any subsequent `DynasmBackend` drop — matches the
    /// lifetime guarantee RPython gets from `cpu` being a long-lived
    /// Python object.
    pub(crate) fn cpu_handle(&self) -> crate::guard::CpuDescrHandle {
        Arc::clone(&self.descr_attachments)
    }

    /// Pin a newly compiled loop/bridge's baked `FailDescrCell`s on the
    /// owning CLT.  Called after every `compile_loop` / `compile_bridge`
    /// returns so `recover_fail_descr_cell` can safely recover a descr
    /// even when a bridge JUMPs into another compiled loop and that
    /// loop's guard fires before control returns to the bridge's owning
    /// token.
    ///
    /// Lifetime root: `token.compiled_loop_token.asmmemmgr_gcreftracers`
    /// (`model.py:294`, `llsupport/assembler.py get_asmmemmgr_gcreftracers`  allow-line-citation
    /// `get_asmmemmgr_gcreftracers`).  Pushes one tracer per
    /// `register_fail_descrs` call mirroring
    /// `assembler.py:822 gcreftracers.append(tracer)`; the tracer owns
    /// the cell `Arc`s for the lifetime of the compiled loop so the
    /// `FailDescrCell` addresses baked into machine code stay live until
    /// `free_loop_and_bridges` drops the CLT (`llmodel.py`).
    pub fn register_fail_descrs(
        &self,
        token: &majit_backend::JitCellToken,
        cells: &Arc<majit_ir::FailDescrStore>,
    ) {
        // `assembler.py:820-823` parity: each call appends one tracer.
        // `clt.asmmemmgr_gcreftracers` is the sole lifetime root for the
        // baked descrs (`model.py` / `llmodel.py free_loop_and_bridges
        // free_loop_and_bridges`).  The cells' addresses are baked into
        // machine code, so the tracer must keep the same `Box`es alive —
        // clone the `Arc`, not the cells.
        if let Some(clt) = token.compiled_loop_token() {
            let tracer: Arc<dyn std::any::Any + Send + Sync> = cells.clone();
            clt.asmmemmgr_gcreftracers.lock().push(tracer);
        }
    }

    /// `assembler.py:822 gcreftracers.append(tracer)` parity for the
    /// per-loop reference-constant `GcTable`.  The table's base address
    /// is baked into machine code by the `LoadFromGcTable` genops, so the
    /// strong `Arc` must outlive the compiled trace.
    /// `clt.asmmemmgr_gcreftracers` is that lifetime root (`model.py:294`
    /// / `llmodel.py free_loop_and_bridges`); when the CLT drops,
    /// the table frees and its `Weak` in the gcreftracer registry is
    /// reaped lazily, so deregistration needs no `free_loop` hook.
    fn register_gc_table(
        &self,
        token: &majit_backend::JitCellToken,
        table: Arc<majit_gc::GcTable>,
    ) {
        if let Some(clt) = token.compiled_loop_token() {
            let tracer: Arc<dyn std::any::Any + Send + Sync> = table.clone();
            clt.asmmemmgr_gcreftracers.lock().push(tracer);
        }
        // `gcreftracer.py` `llop.gc_writebarrier(tr)`: the table enters
        // this MiniMark's remembered set for one minor.
        let _ = with_dynasm_active_gc_mut(|gc| gc.remember_gc_table(&table));
    }

    /// `gcreftracer.py` `make_framework_tracer` / `assembler.py` `_allgcrefs`:
    /// rewrite's gcref list is a GC object, so a collection between
    /// `rewrite_assembler` and `patch_gcref_table` forwards the entries.
    /// A Rust `Vec<GcRef>` is not a root. Park the list in a heap table
    /// and write-barrier it (`llop.gc_writebarrier(tr)`) before assemble
    /// can collect, then copy [`GcTable::slots`] into the in-code array.
    fn pin_rewrite_gcrefs(&self, gcrefs: &[GcRef]) -> Option<Arc<majit_gc::GcTable>> {
        if gcrefs.is_empty() {
            return None;
        }
        let table = majit_gc::GcTable::from_gcrefs(gcrefs);
        let _ = with_dynasm_active_gc_mut(|gc| gc.remember_gc_table(&table));
        Some(table)
    }

    fn gcrefs_after_assemble(pinned: Option<&majit_gc::GcTable>, gcrefs: Vec<GcRef>) -> Vec<GcRef> {
        match pinned {
            Some(table) => table.slots(),
            None => gcrefs,
        }
    }

    // `set_constants_pool`, `set_next_trace_id`, and `set_next_header_pc`
    // are provided via the `Backend` trait impl below so
    // `compile_tmp_callback` and other backend-agnostic consumers can
    // reach them through `&mut dyn Backend`.

    /// gc.py:525-531 parity: build a GcRewriterImpl from the active GC.
    ///
    /// Built unconditionally. Upstream never asks whether a collector
    /// exists before rewriting: `aarch64/regalloc.py:188` (and
    /// `x86/regalloc.py`) call `cpu.gc_ll_descr.rewrite_assembler`
    /// straight through, and `gc.py:109-112` defines it on the base
    /// `GcLLDescription`, so `GcLLDescr_boehm` — a configuration with no
    /// nursery and no write barrier — still runs the whole pass. The
    /// collector decides which *arms* fire inside `rewrite()`, never
    /// whether the pass runs.
    ///
    /// That distinction is load-bearing because the pass is not only about
    /// the GC. `transform_to_gc_load` (rewrite.py) is the memory-op
    /// lowering, and it reads only descrs and CPU addressing capability:
    /// skip it and `RAW_LOAD_I` never becomes `GC_LOAD_INDEXED_I`
    /// (rewrite.py:228-232 via :199-205), so a variable index box reaches a
    /// backend whose only surviving load op wants a *constant* offset
    /// (`aarch64/regalloc.py:535` `op.getarg(1).getint()`).
    ///
    /// Four fields come from the collector and two more take their boehm values
    /// when there is none (gc.py:151-162). With none installed
    /// they take upstream's base-class values: `can_use_nursery_malloc`
    /// returns False (gc.py, inherited by `GcLLDescr_boehm`), spelled
    /// here as `max_nursery_size: 0`, and `write_barrier_descr = None`
    /// (gc.py:156). Allocation then declines to the `dynasm_malloc_*`
    /// helpers, which already answer without a collector by falling back to
    /// a raw alloc (`dynasm_alloc_fixedsize_typed_or_raw`). Note this default
    /// differs deliberately from `dynasm_write_barrier_descr`, which
    /// substitutes the MiniMark layout when the *mutator thread* is not the
    /// one compiling; here `None` from `with_dynasm_active_gc` means no
    /// collector is registered at all (the `gc_sync` singleton arm has
    /// already been consulted), and emitting barriers for a program that
    /// has no collector would call through an absent helper.
    fn gc_rewriter(&self) -> majit_gc::rewrite::GcRewriterImpl {
        let collector = with_dynasm_active_gc(|gc| {
            (
                gc.nursery_free_addr(),
                gc.nursery_top_addr(),
                gc.max_nursery_object_size(),
                gc.get_write_barrier_descr(),
                gc.headerless_fixedsize(),
            )
        });
        // gc.py `get_ll_description(gcdescr)`: `gcdescr is None`
        // selects `GcLLDescr_boehm`, so upstream has no "no collector" state
        // at all — the configuration with none installed IS boehm, and
        // gc.py:151-162 is its field block. Two of the four collector-sourced
        // values already take the base-class answer here
        // (`can_use_nursery_malloc -> False`, gc.py, spelled
        // `max_nursery_size: 0`; `write_barrier_descr = None`, gc.py:156);
        // this flag carries the other two, below.
        let is_boehm = collector.is_none();
        let (nursery_free_addr, nursery_top_addr, max_nursery_size, wb_descr, headerless_fixedsize) =
            collector.unwrap_or((0, 0, 0, None, false));
        majit_gc::rewrite::GcRewriterImpl {
            nursery_free_addr,
            nursery_top_addr,
            max_nursery_size,
            // gc.py:401 `gc_ll_descr.write_barrier_descr` — ask the active
            // collector rather than assuming the MiniMark layout, so the
            // rewriter and `emit_write_barrier_fastpath_for_base` agree on
            // whether barriers exist at all. A collector that needs none
            // reports `None` (`gc.py GcLLDescr_boehm`), and
            // `rewrite.py:393` then emits no `COND_CALL_GC_WB*`.
            wb_descr,
            jitframe_info: crate::jitframe_layout().and_then(|info| info.jitframe_descrs),
            // rewrite.py:673 — read compiled_loop_token._ll_initial_locs +
            // ptr2int(compiled_loop_token.frame_info), both sourced from the
            // CLT Arc on the registered DynasmCaTarget (model.py:292-338).
            call_assembler_callee_locs: Some(Box::new(|ltd| {
                ltd.token_handle_any()
                    .and_then(|any| any.downcast_ref::<std::sync::Arc<JitCellToken>>())
                    .and_then(|token| lookup_call_assembler_callee_locs(token.identity_key()))
            })),
            // x86/runner.py:31 `load_supported_factors = (1, 2, 4, 8)`
            // vs llmodel.py:39 default `(1,)` used by the aarch64
            // backend (which has no scaled store addressing mode and
            // asserts boxes[3].getint() == 1 in its regalloc — see
            // `consider_gc_store_indexed` cfg(target_arch = "aarch64")).
            load_supported_factors: gc_store_supported_factors(),
            // x86/runner.py:22 overrides model.py:22 base default
            // `False` to `True`.  aarch64/runner.py inherits the
            // base `False`, so the rewriter expands LEA to
            // INT_LSHIFT + INT_ADD + INT_ADD (rewrite.py:1089-1098)
            // even though the aarch64 backend has a native
            // `genop_load_effective_address` lowering.
            supports_load_effective_address: supports_load_effective_address(),
            // incminimark.py `malloc_zero_filled = False` when a
            // collector answers: clear_gc_fields / clear_varsize_gc_fields
            // emit the per-object GC-pointer initialization
            // rewrite.py:498-535 requires. With none installed the value is
            // gc.py `GcLLDescr_boehm.malloc_zero_filled = True` and both
            // are inert (rewrite.py:499-500, :521-522), which is what this
            // configuration's allocation actually does: every malloc reaches
            // a raw fallback built on `libc::calloc`.
            malloc_zero_filled: is_boehm,
            // gc.py `self.memcpy_fn = memcpy_fn` cast through
            // `cast_ptr_to_adr` + `cast_adr_to_int` (rewrite.py).
            memcpy_fn: majit_ir::memcpy_fn_addr(),
            // gc.py:40-43 `self.memcpy_descr = get_call_descr(...)`.
            memcpy_descr: majit_ir::make_memcpy_calldescr(),
            // gc.py:46 `self.str_descr = get_array_descr(self, rstr.STR)`.
            str_descr: builtin_string_array_descr(majit_ir::OpCode::Newstr)
                .expect("Newstr must produce a str ArrayDescr"),
            // gc.py `self.unicode_descr = get_array_descr(self, rstr.UNICODE)`.
            unicode_descr: builtin_string_array_descr(majit_ir::OpCode::Newunicode)
                .expect("Newunicode must produce a unicode ArrayDescr"),
            // gc.py:48 `self.str_hash_descr = get_field_descr(self, rstr.STR, 'hash')`.
            str_hash_descr: builtin_string_hash_field_descr(majit_ir::OpCode::Strhash)
                .expect("Strhash must produce a str hash FieldDescr"),
            // gc.py `self.unicode_hash_descr = get_field_descr(self, rstr.UNICODE, 'hash')`.
            unicode_hash_descr: builtin_string_hash_field_descr(majit_ir::OpCode::Unicodehash)
                .expect("Unicodehash must produce a unicode hash FieldDescr"),
            // gc.py:33-37 `self.fielddescr_vtable = get_field_descr(
            // self, rclass.OBJECT, 'typeptr')`.  pyre always emits
            // a typeptr slot (no `gcremovetypeptr` build), so we
            // install Some unconditionally.
            fielddescr_vtable: Some(majit_ir::make_vtable_field_descr()),
            // gc.py:394 `self.fielddescr_tid = get_field_descr(self,
            // self.GCClass.HDR, 'tid')` — framework GC.  pyre's GC
            // is always framework-style; gen_initialize_tid translates
            // the descr's offset by `-HDR_SIZE` because pyre's HDR
            // sits before the object pointer. gc.py:157
            // `GcLLDescr_boehm.fielddescr_tid = None` makes
            // gen_initialize_tid (rewrite.py) emit nothing, which is
            // right with no collector: the raw malloc fallbacks stamp the
            // header themselves, so a second tid GC_STORE has no producer to
            // agree with.
            fielddescr_tid: (!is_boehm).then(majit_ir::make_tid_field_descr),
            malloc_array_fn: dynasm_malloc_array as *const () as i64,
            malloc_array_nonstandard_fn: dynasm_malloc_array_nonstandard as *const () as i64,
            malloc_array_oldgen_fn: dynasm_malloc_array_oldgen as *const () as i64,
            malloc_array_nonstandard_oldgen_fn: dynasm_malloc_array_nonstandard_oldgen as *const ()
                as i64,
            malloc_str_fn: dynasm_malloc_str as *const () as i64,
            malloc_unicode_fn: dynasm_malloc_unicode as *const () as i64,
            malloc_big_fixedsize_fn: dynasm_malloc_big_fixedsize as *const () as i64,
            malloc_big_fixedsize_oldgen_fn: dynasm_malloc_big_fixedsize_oldgen as *const () as i64,
            malloc_array_descr: majit_ir::make_malloc_array_calldescr(),
            malloc_array_nonstandard_descr: majit_ir::make_malloc_array_nonstandard_calldescr(),
            malloc_str_descr: majit_ir::make_malloc_str_calldescr(),
            malloc_unicode_descr: majit_ir::make_malloc_unicode_calldescr(),
            malloc_big_fixedsize_descr: majit_ir::make_malloc_big_fixedsize_calldescr(),
            standard_array_basesize: std::mem::size_of::<usize>(),
            standard_array_length_ofs: 0,
            headerless_fixedsize,
        }
    }

    /// rewrite.py:345 parity: run GC rewriter on ops before assembly.
    /// Returns the rewritten ops plus the per-loop reference-constant
    /// list (`rewrite.py:352 gcrefs_output_list`) the caller turns into a
    /// `GcTable`. The list is empty when the trace references no reference
    /// constants.
    fn prepare_ops_for_compile(
        &mut self,
        inputargs: &[InputArgRc],
        ops: &[OpRc],
    ) -> (Vec<OpRc>, Vec<GcRef>) {
        let num_inputs = inputargs.len() as u32;
        // rewrite.py assemble_loop mutates the same ResOperation objects.
        // `pos` and `descr` are interior-mutable, so the incoming `OpRc`
        // identities stay shared with the optimizer — no `Op` clone.
        for (op_idx, op) in ops.iter().enumerate() {
            if op.result_type() != Type::Void && op.pos().get().is_none() {
                let pos = num_inputs + op_idx as u32;
                op.pos().set(match op.result_type() {
                    Type::Int => OpRef::int_op(pos),
                    Type::Float => OpRef::float_op(pos),
                    Type::Ref => OpRef::ref_op(pos),
                    Type::Void => unreachable!("filtered above"),
                });
            }
        }
        // rewrite.py:489 parity: inject str_descr/unicode_descr for NEWSTR/NEWUNICODE
        inject_builtin_string_descrs(ops);
        {
            let rewriter = self.gc_rewriter();
            use majit_gc::GcRewriter;
            // `GcRewriterAssembler.rewrite` does not copy a constant pool.
            // `RewriteState::resolve_constant` only reads it, so the
            // backend's map stays put.
            let (result, gcrefs) = rewriter.rewrite_for_gc_with_constants(ops, &self.constants);
            (result, gcrefs)
        }
    }

    /// llmodel.py:53-54: store gc_ll_descr on the cpu instance.
    ///
    /// Dynasm does not have cranelift's runtime-id indirection, so it
    /// mirrors wasm: the live allocator is stored in a thread-local and
    /// exposed through backend-agnostic `majit_gc::ActiveGcGuardHooks`.
    pub fn set_gc_allocator(&mut self, mut gc: Box<dyn majit_gc::GcAllocator>) {
        gc.freeze_types();
        self.jitframe_facts = install_gc_box(gc);
    }

    /// No-op, as on cranelift and wasm: the GC rewriter lowers every
    /// `New` / `NewWithVtable` (`rewrite.py handle_new`) to the active
    /// collector's allocation, so no compiled allocation is left to route.
    pub fn set_new_via_gc(&mut self, _enabled: bool) {}

    /// llmodel.py:64-69 self.vtable_offset configuration.
    pub fn set_vtable_offset(&mut self, offset: Option<usize>) {
        self.vtable_offset = offset;
    }

    /// `AbstractLLCPU.subclassrange_min_offset`. Published for
    /// `optimizer.py` `_check_subclass` as well as this CPU's assembler.
    pub fn set_subclassrange_min_offset(&mut self, offset: Option<usize>) {
        self.subclassrange_min_offset = offset;
        majit_backend::set_cpu_subclassrange_min_offset(offset);
    }

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer to its registered GC type id via the
    /// installed gc_ll_descr (the GC backend supplied through
    /// set_gc_allocator).
    pub fn lookup_typeid_from_classptr(&self, classptr: usize) -> Option<u32> {
        with_dynasm_active_gc(|gc| gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr))
            .flatten()
    }

    /// Pre-fetch the GC TYPE_INFO constants that RPython's assembler reads
    /// from `cpu.gc_ll_descr` while emitting `GUARD_IS_OBJECT` and
    /// `GUARD_SUBCLASS`.
    fn collect_guard_gc_type_info(&self) -> Option<GuardGcTypeInfo> {
        with_dynasm_active_gc(|gc| {
            if !gc.supports_guard_gc_type() {
                return None;
            }
            let (base_type_info, shift_by, sizeof_ti) = gc.get_translated_info_for_typeinfo();
            let (infobits_offset, is_object_flag) = gc.get_translated_info_for_guard_is_object();
            Some(GuardGcTypeInfo {
                base_type_info,
                shift_by,
                sizeof_ti,
                infobits_offset,
                is_object_flag,
                subclassrange_min_offset: gc.subclassrange_min_offset(),
            })
        })
        .flatten()
    }

    /// Pre-compute classptr → expected_typeid pairs for every GuardClass /
    /// GuardNonnullClass operand seen in `ops`. RPython resolves these on
    /// demand inside `_cmp_guard_class` (assembler.py); pyre's
    /// dynasm assembler runs without a borrow of `self`, so we materialize
    /// the resolver as a IndexMap up front.
    fn collect_classptr_typeid_table(
        &self,
        ops: &[OpRc],
        const_pool: &majit_ir::ConstMap<majit_ir::Const>,
    ) -> indexmap::IndexMap<i64, u32> {
        let mut table = indexmap::IndexMap::new();
        if self.vtable_offset.is_some() || with_dynasm_active_gc(|_| ()).is_none() {
            // vtable_offset path doesn't need typeid lookups; without a
            // gc_ll_descr there is nothing to resolve anyway.
            return table;
        }
        for op in ops {
            if matches!(
                op.opcode,
                majit_ir::OpCode::GuardClass | majit_ir::OpCode::GuardNonnullClass
            ) && op.num_args() >= 2
            {
                let class_arg = op.arg(1).to_opref();
                // history.py — inline-Const carries its class pointer
                // directly.  The optimizer may also leave it as a plain
                // const-pool OpRef (short preamble / constant folding), which
                // regalloc resolves through the constants map in
                // `RegisterManager::loc`; resolve it the same way here so the
                // codegen-side `Loc::Immed` always has a matching typeid entry.
                let classptr = class_arg
                    .const_int_value()
                    .or_else(|| const_pool.get(&class_arg.raw()).map(|c| c.as_raw_i64()));
                if let Some(classptr) = classptr
                    && let Some(tid) = self.lookup_typeid_from_classptr(classptr as usize)
                {
                    table.insert(classptr, tid);
                }
            }
        }
        table
    }

    /// Pre-compute classptr → `(subclassrange_min, subclassrange_max)` for
    /// every constant `GuardSubclass` expected-class operand.
    ///
    /// RPython reads these fields from `loc_check_against_class.getint()` at
    /// codegen time (`x86/assembler.py:1971-1974`). This table is the
    /// smallest dynasm-side equivalent of that object-field read; it is not a
    /// per-box side table.
    fn collect_classptr_subclass_range_table(
        &self,
        ops: &[OpRc],
    ) -> indexmap::IndexMap<i64, (i64, i64)> {
        let mut table = indexmap::IndexMap::new();
        if with_dynasm_active_gc(|_| ()).is_none() {
            return table;
        }
        for op in ops {
            if op.opcode == majit_ir::OpCode::GuardSubclass && op.num_args() >= 2 {
                let class_arg = op.arg(1).to_opref();
                // history.py — inline-Const carries its class pointer directly.
                let classptr = class_arg.const_int_value();
                if let Some(classptr) = classptr
                    && let Some(range) =
                        with_dynasm_active_gc(|gc| gc.subclass_range(classptr as usize)).flatten()
                {
                    table.insert(classptr, range);
                }
            }
        }
        table
    }

    fn get_compiled(token: &JitCellToken) -> &CompiledCode {
        token
            .compiled
            .get()
            .expect("token has no compiled code")
            .downcast_ref::<CompiledCode>()
            .expect("compiled data is not CompiledCode")
    }

    fn input_slot(position: usize) -> usize {
        arch::JITFRAME_FIXED_SIZE + position
    }

    /// Parity with `BaseRegalloc._set_initial_bindings`:
    /// `_ll_initial_locs` stores `loc.value - base_ofs`, measured in bytes
    /// from `FIRST_ITEM_OFFSET`, not input-order slot numbers.
    fn input_initial_loc(position: usize) -> i32 {
        (Self::input_slot(position) * crate::jitframe::SIZEOFSIGNED) as i32
    }

    /// `llmodel.py get_latest_descr`: cast `jf_descr` through
    /// `AbstractDescr.show`. The word is a [`majit_ir::FailDescrCell`]
    /// address for both guard cells and the cpu-attached singletons,
    /// `propagate_exception_descr` included: its `handle_fail` reads
    /// `grab_exc_value` off the deadframe (`compile.py`
    /// `PropagateExceptionDescr`).
    fn find_descr_by_ptr(&self, ptr: usize) -> majit_backend::deadframe::ExitDescr {
        assert_ne!(
            ptr, 0,
            "find_descr_by_ptr: jf_descr was not written; refusing to recover a null FailDescrCell"
        );
        unsafe { majit_backend::deadframe::ExitDescr::from_cell(ptr) }
    }

    /// `rpython/jit/backend/x86/assembler.py:599` parity: store
    /// `_ll_function_addr` after the loop is fully assembled and retain the
    /// compiled-loop metadata for GC rewrite callbacks.
    fn register_call_assembler_target(token: &JitCellToken, code_addr: usize) {
        let token_key = token.identity_key();
        let index_of_virtualizable = token.virtualizable_arg_index().map_or(-1_i32, |i| i as i32);
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm][ca-target] register token={} addr=0x{:x}",
                token.number(),
                code_addr
            );
        }
        CALL_ASSEMBLER_TARGETS.with(|cell| {
            let mut guard = cell.borrow_mut();
            match guard.get_mut(&token_key) {
                Some(existing) => {
                    existing.code_addr = code_addr;
                }
                None => {
                    let clt = token.compiled_loop_token_expect();
                    clt.set_ca_unregister(token_key, unregister_dynasm_ca_target);
                    guard.insert(
                        token_key,
                        DynasmCaTarget {
                            code_addr,
                            compiled_loop_token: Arc::downgrade(&clt),
                            index_of_virtualizable,
                        },
                    );
                }
            }
        });
    }

    fn redirect_call_assembler_target(old_key: usize, new_addr: usize) {
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm][ca-target] redirect token_key={} addr=0x{:x}",
                old_key, new_addr
            );
        }
        CALL_ASSEMBLER_TARGETS.with(|cell| {
            if let Some(existing) = cell.borrow_mut().get_mut(&old_key) {
                existing.code_addr = new_addr;
            }
        });
    }

    /// `rpython/jit/backend/llsupport/llmodel.py get_baseofs_of_frame_field`
    /// `get_baseofs_of_frame_field(self)` — offset from a `JITFRAME` base
    /// to the first frame-array item. Used by `_set_initial_bindings`
    /// (regalloc.py:865) and `update_frame_info` (model.py) for
    /// `jfi_frame_size` accounting (jitframe.py).
    fn get_baseofs_of_frame_field() -> i64 {
        crate::jitframe::FIRST_ITEM_OFFSET as i64
    }
}

impl DynasmBackend {
    /// `llmodel.py make_execute_token`: allocate, store args, call.
    ///
    /// The entry address is `looptoken._ll_function_addr`. Downcasting the
    /// compiled buffer is only for the debug dumps, which are off on a
    /// steady run.
    fn run_compiled_frame(&self, token: &JitCellToken, args: &[Value]) -> RanFrame {
        self.run_compiled_entry(token, EntryWords::Typed(args))
    }

    /// [`Self::run_compiled_frame`] for `unspecialize_value` words when no
    /// collector is installed. The caller converts to `Value`s first when
    /// `majit_gc::collector_installed()` is set, so this store never roots.
    fn run_compiled_frame_raw(&self, token: &JitCellToken, args: &[i64]) -> RanFrame {
        debug_assert!(!majit_gc::collector_installed());
        self.run_compiled_entry(token, EntryWords::Raw(args))
    }

    fn run_compiled_entry(&self, token: &JitCellToken, args: EntryWords<'_>) -> RanFrame {
        let diag = exec_diag_enabled();
        let compiled = diag.then(|| Self::get_compiled(token));
        let entry = match compiled {
            Some(code) => code.entry_ptr(),
            None => token.ll_function_addr() as *const u8,
        };

        // jitframe.py — every JITFRAME carries a non-null JITFRAMEINFO
        // so the bridge-entry `_check_frame_depth` realloc slowpath
        // (`_frame_realloc_slowpath` → `dynasm_realloc_frame`) can read
        // `jfi_frame_depth` / `jfi_frame_size` to size a grown frame.
        // Mirror the CALL_ASSEMBLER callee template
        // (`lookup_call_assembler_callee_locs` above): the Arc-pinned
        // `frame_info` pointer is stable, and `jfi_frame_depth` alone sizes
        // the frame — the same sizing the JIT uses for the callee frames it
        // allocates itself, so the former `.max(args/fail*4/64)` cushion was
        // a runner-only over-allocation with no upstream basis.
        let clt = unsafe { &*token.compiled_loop_token_ptr() };
        // `jfi_frame_depth` / `jfi_frame_size` live in the CLT's
        // `JITFRAMEINFO`. The pointer is stable for the token's life;
        // `depth()` is the word `rewrite.py` loads, without the mutex.
        let (fi_ptr, num_slots) = {
            let info = unsafe { &*clt.frame_info.data_ptr() };
            (
                info as *const majit_backend::JitFrameInfo,
                info.depth() as usize,
            )
        };
        // `jfi_frame_depth` is floored at `JITFRAME_FIXED_SIZE + inputargs`
        // by construction, so every input slot fits.  Assert it (release):
        // dropping the `.max(64)` cushion removes the silent-OOB mask, so a
        // depth/arg mismatch must surface loudly instead of corrupting.
        let n_args = args.len();
        assert!(
            num_slots >= Self::input_slot(n_args),
            "execute_token: frame depth {num_slots} < input top {} for {} args",
            Self::input_slot(n_args),
            n_args
        );
        let frame_bytes = JitFrame::alloc_size(num_slots);
        // No collector: `malloc_jitframe` is a host block and the input refs
        // are not forwarded. Skip the shadow-stack scope. The steady
        // finish-with-an-int case reuses the token's parked frame
        // (`llmodel.py execute_token` bump) instead of a TLS free list.
        // `EntryWords::Raw` is only used on that path: a collector converts
        // the words back to `Value`s and takes `EntryWords::Typed`.
        let (jf_ptr, gc_object, arg_roots) = match args {
            EntryWords::Typed(values) if majit_gc::collector_installed() => {
                let (ptr, gc_object, roots) = alloc_entry_jitframe(frame_bytes, values);
                (ptr, gc_object, Some(roots))
            }
            _ => (
                take_or_alloc_parked_entry_frame(token, frame_bytes),
                false,
                None,
            ),
        };
        unsafe { JitFrame::init(jf_ptr, fi_ptr, num_slots) };

        match args {
            EntryWords::Typed(values) => {
                let mut ref_index = 0;
                for (i, arg) in values.iter().enumerate() {
                    let raw = match arg {
                        Value::Int(v) => *v,
                        Value::Ref(r) => {
                            let current = match arg_roots.as_ref() {
                                Some(roots) => roots.get(ref_index).unwrap_or(*r),
                                None => *r,
                            };
                            ref_index += 1;
                            current.0 as i64
                        }
                        Value::Float(f) => f.to_bits() as i64,
                        Value::Void => 0,
                    };
                    unsafe {
                        crate::llmodel::set_int_value(jf_ptr, Self::input_slot(i), raw as isize)
                    };
                }
            }
            // `llmodel.py execute_token`: each word is already the frame slot.
            EntryWords::Raw(words) => {
                for (i, &word) in words.iter().enumerate() {
                    unsafe {
                        crate::llmodel::set_int_value(jf_ptr, Self::input_slot(i), word as isize)
                    };
                }
            }
        }
        // `pop_roots`: the pushes span `malloc_jitframe` only. Leaving them
        // through compiled execution would rescan the inputs on every collection.
        drop(arg_roots);
        // llmodel.py execute_token: `llop.gc_writebarrier(lltype.Void, ll_frame)`
        // after the inputs are stored. A frame the allocation placed outside
        // the nursery reaches the next minor collection only through the
        // remembered set.
        if gc_object {
            with_gc_ll_descr(|gc| jitframe_write_barrier(gc, jf_ptr));
        }

        // Each flag is folded into `diag`. A steady run takes the one
        // false test and does not call into the loggers.
        if diag && majit_ir::debug::have_debug_prints() {
            let _s = majit_ir::debug::scope("jit-running");
            for i in 0..n_args {
                let raw = unsafe {
                    crate::llmodel::get_int_value_direct(jf_ptr, Self::input_slot(i)) as i64
                };
                let rendered = match args {
                    EntryWords::Typed(values) => format!("{raw:#018x} ({:?})", values[i]),
                    EntryWords::Raw(words) => format!("{raw:#018x} ({:?})", words[i]),
                };
                majit_ir::debug::debug_print(&format!("  arg[{i}] = {rendered}"));
            }
            majit_ir::debug::debug_print(&format!(
                "execute_token: entry={entry:?} jf_ptr={jf_ptr:?} num_args={} num_slots={num_slots} code_len={}",
                n_args,
                compiled.unwrap().buffer.len()
            ));
        }

        if diag && crate::majit_dump_enabled() {
            // Independent debug toggle — MAJIT_DUMP must produce output
            // regardless of whether MAJIT_LOG is set, so emit via plain
            // eprintln (debug_print would silently no-op without
            // MAJIT_LOG and lose the dump).
            let compiled = compiled.unwrap();
            let rawstart = codebuf::buffer_ptr(&compiled.buffer);
            let code = unsafe { std::slice::from_raw_parts(rawstart, compiled.buffer.len()) };
            eprintln!(
                "[dynasm] CODE DUMP ({} bytes at {:?}, entry {:?}):",
                code.len(),
                rawstart,
                entry
            );
            for (i, chunk) in code.chunks(4).enumerate() {
                let word = u32::from_le_bytes([
                    chunk.first().copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                ]);
                eprint!("{word:08x} ");
                if (i + 1) % 8 == 0 {
                    eprintln!();
                }
            }
            eprintln!();
        }

        // Debug: verify bridge patches are visible
        if diag && crate::majit_log_enabled() {
            for descr in compiled.unwrap().fail_descrs.iter() {
                if let Some(fd) = descr.as_fail_descr() {
                    let bridge_addr =
                        self.lookup_bridge_addr(token, fd.trace_id(), fd.fail_index_per_trace());
                    if bridge_addr != 0 && fd.adr_jump_offset() == 0 {
                        eprintln!(
                            "[dynasm] bridge-patched guard fi={} bridge_addr={:#x} ajo=0 (patched)",
                            fd.fail_index_per_trace(),
                            bridge_addr
                        );
                    }
                }
            }
        }

        // llmodel.py `make_execute_token` fixes the entry signature as
        // `(jitframe, threadlocal_addr) -> jitframe`, and `:317-323` reads the
        // address with `llop.threadlocalref_addr` before the call. The compiled
        // prologue (gen_shadowstack_header) / epilogue
        // (gen_footer_shadowstack) push/pop the jf_ptr onto the shadow
        // stack inline, matching aarch64/assembler.py/1438 — no
        // manual push_jf/pop_jf_to around the call.
        let func: unsafe extern "C" fn(*mut JitFrame, *const i64) -> *mut JitFrame =
            unsafe { std::mem::transmute(entry) };
        if diag && crate::dynasm_exec_diag_enabled() {
            let compiled = compiled.unwrap();
            eprintln!(
                "[dynasm-exec] trace={} header={} entry={entry:p} len={} args={args:?}",
                compiled.trace_id,
                compiled.header_pc,
                compiled.buffer.len(),
            );
        }
        if diag && crate::gc_freelist_diag_enabled() {
            let trace_id = compiled.unwrap().trace_id;
            debug_validate_oldgen_freeblocks(format_args!("before trace {trace_id}"));
        }
        let result_jf = unsafe { func(jf_ptr, crate::jit_threadlocalref_base()) };
        if diag && crate::gc_freelist_diag_enabled() {
            let trace_id = compiled.unwrap().trace_id;
            debug_validate_oldgen_freeblocks(format_args!("after trace {trace_id}"));
        }

        if diag && crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] execute_token returned: result_jf={:?} (expected={:?}) same={}",
                result_jf,
                jf_ptr,
                result_jf == jf_ptr
            );
        }

        RanFrame {
            head: jf_ptr,
            tip: result_jf,
            gc_object,
            num_slots,
        }
    }

    /// `llmodel.py return ll_frame` — the deadframe IS the frame the run
    /// returned. A collector object takes a movable owner-root slot; a
    /// host frame is owned, and its chain freed, by the deadframe.
    fn deadframe_from_run(&self, _token: &JitCellToken, ran: RanFrame) -> DeadFrame {
        let jf_descr_raw = unsafe { crate::llmodel::get_latest_descr(ran.tip) };
        let descr = self.find_descr_by_ptr(jf_descr_raw);
        let descr_fd = descr.as_fail_descr();

        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] descr: fi={} finish={} types={} rd_locs={:?}",
                descr_fd.fail_index_per_trace(),
                descr_fd.is_finish(),
                descr_fd.fail_arg_types().len(),
                descr_fd.rd_locs()
            );
        }

        if ran.gc_object {
            DeadFrame::JitFrame(JitFrameDeadFrame::new(
                GcRef(ran.tip as usize),
                descr,
                None,
                None,
            ))
        } else {
            DeadFrame::LibcJitFrame(unsafe {
                LibcJitFrameDeadFrame::owning(ran.head, ran.tip, ran.num_slots, descr, None)
            })
        }
    }

    /// `DoneWithThisFrameDescrInt.get_result` on a frame `make_execute_token`
    /// just returned.
    ///
    /// `release_under_collector` is the int rule: a host frame is freed even
    /// when a collector is installed. A collector frame (`gc_object`) stays
    /// the deadframe. Ref finishes do not use this; see `done_ref_from_ran`.
    #[inline(always)]
    fn done_word_from_ran<T>(
        &self,
        token: &JitCellToken,
        ran: RanFrame,
        release_under_collector: bool,
        is_done: fn(&Self, usize) -> bool,
        read: fn(*mut JitFrame) -> T,
    ) -> Result<T, DeadFrame> {
        let host_finish =
            !ran.gc_object && (release_under_collector || !majit_gc::collector_installed());
        if host_finish {
            let descr_raw = unsafe { crate::llmodel::get_latest_descr(ran.tip) };
            if is_done(self, descr_raw) {
                // Slot 0. That descr's `rd_locs` stays empty (`set_rd_locs`
                // is resume-guard only), so the word `genop_finish` stored
                // is `jf_frame[0]`.
                let value = unsafe { read(JitFrame::resolve(ran.tip)) };
                park_or_free_done_entry_frame(token, ran.head, ran.tip, ran.gc_object);
                return Ok(value);
            }
        }
        Err(self.deadframe_from_run(token, ran))
    }

    /// `DoneWithThisFrameDescrInt.get_result` on a frame
    /// `make_execute_token` just returned. A collector frame, or any other
    /// descr, becomes the deadframe the general path reads.
    fn done_int_from_ran(&self, token: &JitCellToken, ran: RanFrame) -> Result<i64, DeadFrame> {
        self.done_word_from_ran(token, ran, true, Self::finish_is_done_int, done_int_slot0)
    }

    /// Exec-diag entries keep the layered `execute_token` path. The value
    /// rebuild and the raw-frame run are shared; the finish call is the int
    /// or ref sibling.
    #[inline(always)]
    fn execute_token_done_raw_general<T>(
        &self,
        token: &JitCellToken,
        args: &[i64],
        on_collector: fn(&Self, &JitCellToken, &[Value]) -> Result<T, DeadFrame>,
        on_ran: fn(&Self, &JitCellToken, RanFrame) -> Result<T, DeadFrame>,
    ) -> Result<T, DeadFrame> {
        if majit_gc::collector_installed() {
            let kinds = token.inputarg_types();
            let values: smallvec::SmallVec<[Value; 8]> = args
                .iter()
                .enumerate()
                .map(|(i, &word)| {
                    let kind = kinds.get(i).copied().unwrap_or(Type::Int);
                    majit_backend::value_from_unspecialized_word(word, kind)
                })
                .collect();
            return on_collector(self, token, &values);
        }
        let ran = self.run_compiled_frame_raw(token, args);
        on_ran(self, token, ran)
    }

    #[cold]
    #[inline(never)]
    fn execute_token_done_int_raw_general(
        &self,
        token: &JitCellToken,
        args: &[i64],
    ) -> Result<i64, DeadFrame> {
        self.execute_token_done_raw_general(
            token,
            args,
            Self::execute_token_done_int,
            Self::done_int_from_ran,
        )
    }

    /// Any exit other than `DoneWithThisFrameDescrInt` builds the deadframe.
    #[cold]
    #[inline(never)]
    fn raw_entry_deadframe(
        &self,
        token: &JitCellToken,
        head: *mut JitFrame,
        tip: *mut JitFrame,
        num_slots: usize,
    ) -> DeadFrame {
        let ran = RanFrame {
            head,
            tip,
            gc_object: self.jitframe_facts.is_gc_object,
            num_slots,
        };
        self.deadframe_from_run(token, ran)
    }

    /// `jf_descr` equals the `DoneWithThisFrameDescrInt` cell
    /// `set_done_with_this_frame_descr_int` published.
    ///
    /// `compile.py make_and_attach_done_descrs` attaches that singleton
    /// before any compiled code runs, so the exit is one compare.
    ///
    /// `#[inline(always)]`: the raw finish path compares `jf_descr` on every
    /// return. Taking this function's address for the shared cold body must
    /// not turn that compare into a call.
    #[inline(always)]
    fn finish_is_done_int(&self, descr_raw: usize) -> bool {
        let cached = self
            .done_int_cell
            .load(std::sync::atomic::Ordering::Acquire);
        descr_raw != 0 && descr_raw == cached
    }

    /// `jf_descr` equals the `DoneWithThisFrameDescrRef` cell
    /// `set_done_with_this_frame_descr_ref` published.
    #[inline(always)]
    fn finish_is_done_ref(&self, descr_raw: usize) -> bool {
        let cached = self
            .done_ref_cell
            .load(std::sync::atomic::Ordering::Acquire);
        descr_raw != 0 && descr_raw == cached
    }

    /// `compile.py DoneWithThisFrameDescrRef.get_result` on the frame
    /// `make_execute_token` just returned.
    ///
    /// `warmstate.py execute_assembler` reads that word and drops the
    /// deadframe with nothing allocated between the two. The word is loaded
    /// here and the frame released without `JitFrameDeadFrame::new` and
    /// without an `OwnerRootGuard` on the result. A nursery frame stays the
    /// collector's. A host frame goes through `release_done_int_frame` (park
    /// a single unforwarded frame, otherwise free the chain). Any other
    /// descr still builds the deadframe from the unresolved `RanFrame`.
    fn done_ref_from_ran(&self, token: &JitCellToken, ran: RanFrame) -> Result<usize, DeadFrame> {
        // Host frames keep the descr load on the pointer the run returned
        // (`done_word_from_ran`). A nursery frame may come back as a
        // forwarding stub, so its descr is read from `JitFrame::resolve`.
        let tip = unsafe { JitFrame::resolve(ran.tip) };
        let descr_ptr = if ran.gc_object { tip } else { ran.tip };
        let descr_raw = unsafe { crate::llmodel::get_latest_descr(descr_ptr) };
        if self.finish_is_done_ref(descr_raw) {
            let value = done_ref_slot0(tip);
            if !ran.gc_object {
                // Original `ran.tip`, not the resolved pointer: the
                // single-frame park check stays `tip == head`.
                park_or_free_done_entry_frame(token, ran.head, ran.tip, ran.gc_object);
            }
            return Ok(value);
        }
        Err(self.deadframe_from_run(token, ran))
    }

    #[cold]
    #[inline(never)]
    fn execute_token_done_ref_raw_general(
        &self,
        token: &JitCellToken,
        args: &[i64],
    ) -> Result<usize, DeadFrame> {
        self.execute_token_done_raw_general(
            token,
            args,
            Self::execute_token_done_ref,
            Self::done_ref_from_ran,
        )
    }

    /// Raw finish entry. Allocate, store `args`, call the token, and return
    /// `(head, tip, num_slots, jf_descr)`.
    ///
    /// `llmodel.py` `AbstractLLCPU.make_execute_token`. Both raw finish paths
    /// read slot 0 of `tip` afterwards. Allocation and the slot stores are
    /// shared (`prepare_done_raw_entry_frame`); the call is this backend's
    /// `(jitframe, threadlocal_addr)` convention.
    #[inline(always)]
    fn run_done_raw_entry(
        &self,
        token: &JitCellToken,
        args: &[i64],
    ) -> (*mut JitFrame, *mut JitFrame, usize, usize) {
        let entry = token.ll_function_addr() as *const u8;
        let clt = unsafe { &*token.compiled_loop_token_ptr() };
        let num_slots = unsafe { (*clt.frame_info.data_ptr()).depth() as usize };
        let facts = self.jitframe_facts;
        let jf_ptr = unsafe {
            prepare_done_raw_entry_frame(token, args, Self::input_slot(0), num_slots, facts)
        };
        let func: unsafe extern "C" fn(*mut JitFrame, *const i64) -> *mut JitFrame =
            unsafe { std::mem::transmute(entry) };
        let returned = unsafe { func(jf_ptr, crate::jit_threadlocalref_base()) };
        // Host frames keep the descr load on the pointer the run returned
        // (`done_word_from_ran`). A moving nursery may come back as a
        // forwarding stub (`gen_footer_shadowstack` / `_call_footer`
        // pop then `mov x0, x29` / `emit_call_footer_raw`);
        // `JitframeDescrFacts::resolve_returned_frame` chases that when
        // the descr has a gcrootmap, else `jitframe_resolve` (`jf_forward`).
        let tip = if facts.is_gc_object {
            facts.resolve_returned_frame(returned)
        } else {
            returned
        };
        let descr_raw = unsafe { crate::llmodel::get_latest_descr(tip) };
        (jf_ptr, tip, num_slots, descr_raw)
    }

    /// Cell address published by `set_done_with_this_frame_descr_int`.
    /// Zero until `make_and_attach_done_descrs` runs.
    pub fn done_with_this_frame_descr_int_cell(&self) -> usize {
        self.done_int_cell
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// The attached cell `descr_ptrs` hands to codegen.
    pub fn attached_done_with_this_frame_descr_int(&self) -> usize {
        self.descr_attachments
            .read()
            .descr_ptrs()
            .done_with_this_frame_descr_int
    }
}

impl Backend for DynasmBackend {
    fn vtable_offset(&self) -> Option<usize> {
        self.vtable_offset
    }

    fn w_class_offset(&self) -> Option<usize> {
        self.w_class_offset
    }

    fn set_w_class_offset(&mut self, offset: Option<usize>) {
        self.w_class_offset = offset;
    }

    fn cpu_tracker(&self) -> &Arc<majit_backend::CpuTotalTracker> {
        &self.cpu_tracker
    }

    fn assembler_memory_stats(&self) -> (usize, usize) {
        majit_backend::process_assembler_memory_stats()
    }

    fn compile_loop(
        &mut self,
        inputargs: &[InputArgRc],
        ops: &[OpRc],
        token: &JitCellToken,
    ) -> Result<AsmInfo, BackendError> {
        // `gctypelayout.py encode_type_shapes_now` closes `type_info_group`
        // at translation. Close before `collect_guard_gc_type_info` reads it.
        // Once frozen this is a query; backend-only tests never reach the
        // metainterp compile entry.
        majit_gc::ensure_type_registry_closed();
        let _writing = majit_backend::AssemblerWriting::enter();
        // `x86/assembler.py:514` parity: PyPy creates the
        // `CompiledLoopToken` inside `assemble_loop`, and that's where
        // the `cpu.tracker.total_compiled_loops` bump and the
        // `jit-mem-looptoken-alloc` debug section fire.  Pyre's eager
        // CLT creation makes that point unreachable from
        // `CompiledLoopToken::new`; defer both to here so the counter
        // matches PyPy at the same structural moment.
        if let Some(clt) = token.compiled_loop_token() {
            majit_backend::record_compiled_loop_token(&self.cpu_tracker, &clt);
        }
        token.set_inputarg_types(inputargs.iter().map(|ia| ia.tp.get()).collect());
        let trace_id = self.next_trace_id;
        self.next_trace_id += 1;
        let header_pc = self.next_header_pc;
        // gc.py rewrite_assembler parity: run GC rewriter before regalloc.
        let (prepared_ops, gcrefs) = self.prepare_ops_for_compile(inputargs, ops);
        let pinned_gcrefs = self.pin_rewrite_gcrefs(&gcrefs);
        // The assembler stores the typed `Const` pool directly; each box
        // variant carries its own type (`Const::get_type`).
        let const_pool = std::mem::take(&mut self.constants);
        if crate::trace_ops_diag_id() == Some(trace_id) {
            let constants: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> = const_pool
                .iter()
                .map(|(&key, value)| (key, value.as_raw_i64()))
                .collect();
            eprintln!(
                "--- dynasm loop prepared ops trace={trace_id} header={header_pc} ---\n{}",
                majit_ir::format_trace(&prepared_ops, &constants),
            );
        }
        let typeid_table = self.collect_classptr_typeid_table(&prepared_ops, &const_pool);
        let guard_gc_type_info = self.collect_guard_gc_type_info();
        let subclass_range_table = self.collect_classptr_subclass_range_table(&prepared_ops);
        let attached_descrs = self.attached_descr_ptrs();
        let cpu_handle = self.cpu_handle();
        // PyPy's `setup_once` (`llsupport/assembler.py`) is what
        // builds the per-CPU malloc / propagate trampolines, but the
        // pyre `Backend::setup_once` hook isn't yet wired into every
        // tracing entry (`force_start_tracing` builds the trace ctx
        // inline rather than going through `setup_tracing`).  Ensure
        // the trampoline is materialised lazily on the first
        // `compile_loop`/`compile_bridge` instead — idempotent and
        // cheap after the cache hit, matching PyPy's "build once per
        // CPU" semantics without requiring every trace-start path to
        // remember to call `_setup_once`.
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let malloc_slowpath_fixed = self
            .arch_cpu_ext
            .ensure_malloc_slowpath_fixed(&self.descr_attachments);
        #[cfg(target_arch = "x86_64")]
        let malloc_slowpath_headerless = self
            .arch_cpu_ext
            .ensure_malloc_slowpath_headerless(&self.descr_attachments);
        #[cfg(target_arch = "x86_64")]
        let malloc_slowpath_varsize = self
            .arch_cpu_ext
            .ensure_malloc_slowpath_varsize(&self.descr_attachments);
        #[cfg(target_arch = "x86_64")]
        let malloc_slowpath_str = self
            .arch_cpu_ext
            .ensure_malloc_slowpath_str(&self.descr_attachments);
        #[cfg(target_arch = "x86_64")]
        let malloc_slowpath_unicode = self
            .arch_cpu_ext
            .ensure_malloc_slowpath_unicode(&self.descr_attachments);
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let wb_slowpath = self.arch_cpu_ext.ensure_wb_slowpath();
        // `setup_once`: `propagate_exception_path`, `_frame_realloc_slowpath`,
        // `stack_check_slowpath`. Passed in like `wb_slowpath`. The stack
        // check stays 0 until `insert_stack_check` is registered; the next
        // compile retries.
        #[cfg(target_arch = "x86_64")]
        let propagate_exception_path = self
            .arch_cpu_ext
            .ensure_propagate_exception_path(&self.descr_attachments);
        #[cfg(target_arch = "x86_64")]
        let frame_realloc_slowpath = self.arch_cpu_ext.ensure_frame_realloc_slowpath();
        #[cfg(target_arch = "x86_64")]
        let stack_check_slowpath = self
            .arch_cpu_ext
            .ensure_stack_check_slowpath(&self.descr_attachments);
        // `setup_once` builds `cond_call_slowpath` after `wb_slowpath`.
        #[cfg(target_arch = "x86_64")]
        let cond_call_slowpath = self.arch_cpu_ext.ensure_cond_call_slowpath();
        // `setup_once` builds `failure_recovery_code` before the slow paths.
        #[cfg(target_arch = "x86_64")]
        let failure_recovery_code = self.arch_cpu_ext.ensure_failure_recovery();
        let mut asm = Asm::new(
            Arc::clone(&self.asm_memory_manager),
            trace_id,
            header_pc,
            const_pool,
            self.vtable_offset,
            self.subclassrange_min_offset,
            typeid_table,
            guard_gc_type_info,
            subclass_range_table,
            attached_descrs,
            cpu_handle,
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            malloc_slowpath_fixed,
            #[cfg(target_arch = "x86_64")]
            malloc_slowpath_headerless,
            #[cfg(target_arch = "x86_64")]
            malloc_slowpath_varsize,
            #[cfg(target_arch = "x86_64")]
            malloc_slowpath_str,
            #[cfg(target_arch = "x86_64")]
            malloc_slowpath_unicode,
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            wb_slowpath,
            #[cfg(target_arch = "x86_64")]
            propagate_exception_path,
            #[cfg(target_arch = "x86_64")]
            frame_realloc_slowpath,
            #[cfg(target_arch = "x86_64")]
            stack_check_slowpath,
            #[cfg(target_arch = "x86_64")]
            cond_call_slowpath,
            #[cfg(target_arch = "x86_64")]
            failure_recovery_code,
            inputargs,
            &prepared_ops,
        );
        // `assembler.py assemble_loop`: `reserve_gcref_table(allgcrefs)`
        // opens the code block with one word per reference constant, and
        // `patch_gcref_table` fills them in once the block is materialized.
        asm.reserve_gcref_table(gcrefs.len());
        let compiled = asm.assemble_loop()?;
        // `assembler.py patch_pending_failure_recoveries`'s
        // `clt.invalidate_positions.append(...)`, deferred to here because the
        // addresses are only absolute once the buffer is materialised.
        token.record_invalidate_positions(
            codebuf::write_invalidate_positions,
            compiled.invalidate_positions.clone(),
        );

        let rawstart = codebuf::buffer_ptr(&compiled.buffer) as usize;
        let code_addr = compiled.entry_ptr() as usize;
        let code_size = compiled.buffer.len();
        let gcrefs = Self::gcrefs_after_assemble(pinned_gcrefs.as_deref(), gcrefs);
        // SAFETY: `reserve_gcref_table` reserved `gcrefs.len()` words at
        // `rawstart`, in the arena block the token's CLT keeps alive.
        let gc_table =
            (!gcrefs.is_empty()).then(|| unsafe { majit_gc::GcTable::in_code(rawstart, &gcrefs) });
        let frame_depth = compiled.frame_depth.load(Ordering::Acquire) as i64;
        Self::register_call_assembler_target(token, code_addr);
        self.register_fail_descrs(token, &compiled.fail_descrs);
        if let Some(table) = gc_table {
            self.register_gc_table(token, table);
        }

        // `compile.py record_loop_or_bridge`: for each ResumeDescr
        // in the newly-compiled trace, stamp the owning CompiledLoopToken.
        // RPython predicates the stamp on `isinstance(descr, ResumeDescr)`
        // (`compile.py:185`); pyre uses the `is_resume_guard()` trait
        // method (descr.rs), implemented true on the
        // `ResumeGuardDescr` family (compile.rs).
        //
        // `compile.py:183-186` walks `loop.operations` and writes
        // `op.descr.rd_loop_token` directly.  Pyre's `compiled.fail_descrs`
        // hold the same `DescrRef`s the metainterp stamped onto each
        // guard op (unified descr), so the write lands on the
        // same `ResumeGuardDescr` Arc upstream targets.
        if let Some(clt) = token.compiled_loop_token() {
            for descr in compiled.fail_descrs.iter() {
                if !descr.is_resume_guard() {
                    continue;
                }
                if let Some(fd) = descr.as_fail_descr() {
                    fd.set_rd_loop_token_clt(std::sync::Arc::clone(&clt)
                        as std::sync::Arc<dyn std::any::Any + Send + Sync>);
                }
            }
        }

        // `rpython/jit/backend/x86/assembler.py:513-526` initializes the
        // per-loop `CompiledLoopToken` fields at assemble_loop entry:
        //   * frame_info is allocated and assigned (line 526-530)
        //   * looptoken.compiled_loop_token = clt (line 514)
        // pyre eagerly creates the CLT in `JitCellToken::new`, so the
        // equivalent here is populating its fields with the real values
        // computed during assembly.
        let baseofs = Self::get_baseofs_of_frame_field();
        if let Some(clt) = token.compiled_loop_token() {
            // `x86/assembler.py:526-530` frame_info = malloc_aligned + set
            // jfi_frame_depth/jfi_frame_size. pyre's frame_info lives on
            // the CLT already; just populate via update_frame_depth.
            clt.frame_info
                .lock()
                .update_frame_depth(baseofs, frame_depth);
            // `llsupport/regalloc.py` `_set_initial_bindings` —
            // each input lands at `loc.value - base_ofs =
            // (JITFRAME_FIXED_SIZE + i) * SIZEOFSIGNED` so the GcStores
            // synthesized by `handle_call_assembler` (rewrite.py)
            // hit the actual input slots, not the managed-register save
            // area at the head of `jf_frame`. The list length must match
            // `inputargs.len()` so `handle_call_assembler` can index it.
            let locs: Vec<i32> = (0..inputargs.len()).map(Self::input_initial_loc).collect();
            *clt._ll_initial_locs.lock() = locs;
        }
        // `x86/assembler.py:599` `looptoken._ll_function_addr =
        // rawstart + functionpos`. pyre stores the single entry point
        // so `_ll_function_addr` = compiled-code base.
        token.set_ll_function_addr(code_addr);
        // `x86/assembler.py assemble_loop` `looptoken._ll_raw_start = rawstart`.
        token.set_ll_raw_start(rawstart);
        token.set_compiled(Box::new(compiled));

        Ok(AsmInfo {
            code_addr,
            code_size,
        })
    }

    fn set_constants_pool(&mut self, constants: majit_ir::ConstMap<majit_ir::Const>) {
        self.constants = constants;
    }

    fn set_next_trace_id(&mut self, trace_id: u64) {
        self.next_trace_id = trace_id;
    }

    fn set_next_header_pc(&mut self, header_pc: u64) {
        self.next_header_pc = header_pc;
    }

    fn set_done_with_this_frame_descr_void(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .update(|a| a.done_with_this_frame_descr_void = Some(descr));
    }
    fn set_done_with_this_frame_descr_int(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .update(|a| a.done_with_this_frame_descr_int = Some(descr));
        let ptr = self
            .descr_attachments
            .read()
            .descr_ptrs()
            .done_with_this_frame_descr_int;
        self.done_int_cell
            .store(ptr, std::sync::atomic::Ordering::Release);
    }
    fn set_done_with_this_frame_descr_ref(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .update(|a| a.done_with_this_frame_descr_ref = Some(descr));
        let ptr = self
            .descr_attachments
            .read()
            .descr_ptrs()
            .done_with_this_frame_descr_ref;
        self.done_ref_cell
            .store(ptr, std::sync::atomic::Ordering::Release);
    }
    fn set_done_with_this_frame_descr_float(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .update(|a| a.done_with_this_frame_descr_float = Some(descr));
    }
    fn set_exit_frame_with_exception_descr_ref(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .update(|a| a.exit_frame_with_exception_descr_ref = Some(descr));
    }
    fn set_propagate_exception_descr(&mut self, descr: majit_ir::DescrRef) {
        // x86/assembler.py `_build_propagate_exception_path` parity:
        // PyPy bakes `propagate_exception_descr` into the per-CPU
        // propagate trampoline at setup time.  Pyre defers the bake to
        // `X86CpuExt::ensure_propagate_exception_path` (x86/cpu_ext.rs),
        // which reads the descr pointer directly from
        // `descr_attachments` and embeds it in the helper.
        //
        // The metainterp wiring re-installs the descr several times
        // during init (`attach_descrs_to_cpu`, `register_jitdriver_sd`,
        // and `attach_default_test_descrs` for backend-only tests).
        // The common case is the *same* `Arc<Descr>` arriving twice —
        // idempotent, no observable effect — so a pointer-equal
        // `Arc` returns immediately.
        //
        // PyPy's lifecycle binds `propagate_exception_descr` before
        // `cpu.setup_once()` (`pyjitpl.py` precedes
        // `pyjitpl.py`) and never swaps it afterwards.
        // Pyre upholds the same invariant: a *different* `Arc`
        // arriving after the propagate / malloc trampolines have
        // already baked the previous descr pointer would leave
        // already-compiled loops/bridges writing the orphaned pointer
        // into `jf_descr` on OOM, missing the propagate-exception
        // dispatch.  Rather than patch the baked immediates (PyPy
        // never does so), refuse the swap with a clear assert.  The
        // pre-bake path (no trampoline yet) keeps overwriting freely
        // — the next `ensure_*_path` will bake the fresh pointer.
        if self
            .descr_attachments
            .read()
            .propagate_exception_descr
            .as_ref()
            .is_some_and(|existing| std::sync::Arc::ptr_eq(existing, &descr))
        {
            return;
        }
        assert!(
            !self.arch_cpu_ext.has_propagate_dependent_caches(),
            "set_propagate_exception_descr called with a fresh Arc after \
             per-CPU propagate/malloc trampolines have already baked the \
             previous descr pointer; bind propagate_exception_descr once \
             before backend.setup_once() (pyjitpl.py:2273-2283 ordering)"
        );
        self.descr_attachments
            .update(|a| a.propagate_exception_descr = Some(descr));
    }

    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArgRc],
        ops: &[OpRc],
        original_token: &JitCellToken,
        _caller_recovery_layout: Option<&majit_backend::ExitRecoveryLayout>,
    ) -> Result<AsmInfo, BackendError> {
        // Same close as `compile_loop`: bridge codegen reads `type_info_group`.
        majit_gc::ensure_type_registry_closed();
        let _writing = majit_backend::AssemblerWriting::enter();
        // `x86/runner.py:100-101` parity:
        //   clt = original_loop_token.compiled_loop_token
        //   clt.compiling_a_bridge()
        // Bumps this backend's `cpu.tracker.total_compiled_bridges`,
        // the per-loop `bridges_count`, and emits the
        // `jit-mem-looptoken-alloc` debug section.
        if let Some(clt) = original_token.compiled_loop_token() {
            clt.compiling_a_bridge(&self.cpu_tracker);
        }
        let trace_id = self.next_trace_id;
        self.next_trace_id += 1;

        let arglocs = Asm::rebuild_faillocs_from_descr(fail_descr, inputargs);
        let (prepared_ops, gcrefs) = self.prepare_ops_for_compile(inputargs, ops);
        let pinned_gcrefs = self.pin_rewrite_gcrefs(&gcrefs);
        // format_trace reads raw `i64` values; the assembler stores the
        // typed `Const` pool directly (type rides on `Const::get_type`).
        let const_pool = std::mem::take(&mut self.constants);
        let trace_ops_diag = crate::trace_ops_diag_id() == Some(trace_id);
        if trace_ops_diag {
            let constants: IndexMap<u32, i64, rustc_hash::FxBuildHasher> = const_pool
                .iter()
                .map(|(&k, c)| (k, c.as_raw_i64()))
                .collect();
            eprintln!(
                "--- dynasm bridge prepared ops (trace_id={}, fail_index={}) ---\n{}",
                trace_id,
                fail_descr.fail_index_per_trace(),
                majit_ir::format_trace(&prepared_ops, &constants)
            );
        }
        let typeid_table = self.collect_classptr_typeid_table(&prepared_ops, &const_pool);
        let guard_gc_type_info = self.collect_guard_gc_type_info();
        let subclass_range_table = self.collect_classptr_subclass_range_table(&prepared_ops);
        let attached_descrs = self.attached_descr_ptrs();
        let cpu_handle = self.cpu_handle();
        // PyPy's `setup_once` (`llsupport/assembler.py`) is what
        // builds the per-CPU malloc / propagate trampolines, but the
        // pyre `Backend::setup_once` hook isn't yet wired into every
        // tracing entry (`force_start_tracing` builds the trace ctx
        // inline rather than going through `setup_tracing`).  Ensure
        // the trampoline is materialised lazily on the first
        // `compile_loop`/`compile_bridge` instead — idempotent and
        // cheap after the cache hit, matching PyPy's "build once per
        // CPU" semantics without requiring every trace-start path to
        // remember to call `_setup_once`.
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let malloc_slowpath_fixed = self
            .arch_cpu_ext
            .ensure_malloc_slowpath_fixed(&self.descr_attachments);
        #[cfg(target_arch = "x86_64")]
        let malloc_slowpath_headerless = self
            .arch_cpu_ext
            .ensure_malloc_slowpath_headerless(&self.descr_attachments);
        #[cfg(target_arch = "x86_64")]
        let malloc_slowpath_varsize = self
            .arch_cpu_ext
            .ensure_malloc_slowpath_varsize(&self.descr_attachments);
        #[cfg(target_arch = "x86_64")]
        let malloc_slowpath_str = self
            .arch_cpu_ext
            .ensure_malloc_slowpath_str(&self.descr_attachments);
        #[cfg(target_arch = "x86_64")]
        let malloc_slowpath_unicode = self
            .arch_cpu_ext
            .ensure_malloc_slowpath_unicode(&self.descr_attachments);
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let wb_slowpath = self.arch_cpu_ext.ensure_wb_slowpath();
        // Same once-per-CPU helpers as `compile_loop`.
        #[cfg(target_arch = "x86_64")]
        let propagate_exception_path = self
            .arch_cpu_ext
            .ensure_propagate_exception_path(&self.descr_attachments);
        #[cfg(target_arch = "x86_64")]
        let frame_realloc_slowpath = self.arch_cpu_ext.ensure_frame_realloc_slowpath();
        #[cfg(target_arch = "x86_64")]
        let stack_check_slowpath = self
            .arch_cpu_ext
            .ensure_stack_check_slowpath(&self.descr_attachments);
        // Same once-per-CPU `cond_call_slowpath` as `compile_loop`.
        #[cfg(target_arch = "x86_64")]
        let cond_call_slowpath = self.arch_cpu_ext.ensure_cond_call_slowpath();
        // Same once-per-CPU `failure_recovery_code` as `compile_loop`.
        #[cfg(target_arch = "x86_64")]
        let failure_recovery_code = self.arch_cpu_ext.ensure_failure_recovery();
        let mut asm = Asm::new(
            Arc::clone(&self.asm_memory_manager),
            trace_id,
            0,
            const_pool,
            self.vtable_offset,
            self.subclassrange_min_offset,
            typeid_table,
            guard_gc_type_info,
            subclass_range_table,
            attached_descrs,
            cpu_handle,
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            malloc_slowpath_fixed,
            #[cfg(target_arch = "x86_64")]
            malloc_slowpath_headerless,
            #[cfg(target_arch = "x86_64")]
            malloc_slowpath_varsize,
            #[cfg(target_arch = "x86_64")]
            malloc_slowpath_str,
            #[cfg(target_arch = "x86_64")]
            malloc_slowpath_unicode,
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            wb_slowpath,
            #[cfg(target_arch = "x86_64")]
            propagate_exception_path,
            #[cfg(target_arch = "x86_64")]
            frame_realloc_slowpath,
            #[cfg(target_arch = "x86_64")]
            stack_check_slowpath,
            #[cfg(target_arch = "x86_64")]
            cond_call_slowpath,
            #[cfg(target_arch = "x86_64")]
            failure_recovery_code,
            inputargs,
            &prepared_ops,
        );
        // `assembler.py assemble_bridge`: `reserve_gcref_table(allgcrefs)`
        // opens the bridge's own code block (same as compile_loop).
        asm.reserve_gcref_table(gcrefs.len());

        let _orig_compiled = Self::get_compiled(original_token);

        // `assembler.py assemble_bridge` reads the locations off the
        // `faildescr` it was handed — the descr the guard actually failed on —
        // and patches that same descr's jump below; `compile.py
        // ResumeGuardDescr.compile_and_attach` is what hands it over.
        //
        // `jitdriver.rs` `start_bridge_tracing` takes the bridge's inputarg
        // count from `fail_descr.fail_arg_types().len()`, while
        // `_update_bindings` zips that inputarg vector against this location
        // vector, so both must come off the one descr. Reading the locations
        // off a `(trace_id, fail_index)` lookup let the two lengths disagree —
        // measured on `pyre/bench/fannkuch.py`, 17 inputargs against 15
        // locations — and every inputarg past the shorter list then bound
        // nothing at all (`RegisterManager.loc: box InputArgInt(1844) not
        // found`).
        let compiled = asm.assemble_bridge(fail_descr, &arglocs)?;
        // `llsupport/assembler.py assemble_bridge` keeps `self.current_clt =
        // original_loop_token.compiled_loop_token` for the whole emission, so a
        // bridge's guards join the loop's list.  The list was emptied by any
        // invalidation that already happened, which is what makes this bridge
        // start valid.
        original_token.record_invalidate_positions(
            codebuf::write_invalidate_positions,
            compiled.invalidate_positions.clone(),
        );

        let rawstart = codebuf::buffer_ptr(&compiled.buffer) as usize;
        let bridge_addr = compiled.entry_ptr() as usize;
        let code_size = compiled.buffer.len();
        let gcrefs = Self::gcrefs_after_assemble(pinned_gcrefs.as_deref(), gcrefs);
        // SAFETY: as in `compile_loop`.
        let gc_table =
            (!gcrefs.is_empty()).then(|| unsafe { majit_gc::GcTable::in_code(rawstart, &gcrefs) });
        if crate::dynasm_exec_diag_enabled() {
            eprintln!(
                "[dynasm-bridge] trace={trace_id} source={}:{} addr={bridge_addr:#x} len={code_size}",
                fail_descr.trace_id(),
                fail_descr.fail_index_per_trace(),
            );
        }
        // `rpython/jit/backend/x86/assembler.py` `assemble_bridge`:
        // `frame_depth = max(current_clt.frame_info.jfi_frame_depth,
        //                    frame_depth_no_fixed_size + JITFRAME_FIXED_SIZE)`
        // → `self.update_frame_depth(frame_depth)` which calls
        // `self.current_clt.frame_info.update_frame_depth(baseofs, frame_depth)`.
        //
        // Without this, self-recursive CALL_ASSEMBLER allocates callee
        // jitframes with the loop's original (smaller) jfi_frame_depth,
        // so regalloc spill slots past that depth overrun into the next
        // nursery allocation (observed: jf_L[272] aliases the next
        // callee's jf_frame_info, so llfi ends up at input0).
        let bridge_frame_depth = compiled.frame_depth.load(Ordering::Acquire) as i64;
        let baseofs = Self::get_baseofs_of_frame_field();
        if let Some(clt) = original_token.compiled_loop_token() {
            clt.frame_info
                .lock()
                .update_frame_depth(baseofs, bridge_frame_depth);
        }
        // Keep the existing loop CompiledCode.frame_depth in lockstep
        // (same rationale as `redirect_call_assembler` — dynasm codegen
        // reads `CompiledCode.frame_depth` in addition to
        // `frame_info.jfi_frame_depth`).
        Self::get_compiled(original_token)
            .frame_depth
            .fetch_max(bridge_frame_depth as usize, Ordering::Release);

        // assembler.py:987 patch_jump_for_descr — redirect guard to bridge.
        let guard_fd = fail_descr;
        let ajo = guard_fd.adr_jump_offset();
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm-bridge] patch: trace_id={} fail_index={} adr_jump_offset=0x{:x} bridge_addr=0x{:x}",
                guard_fd.trace_id(),
                guard_fd.fail_index_per_trace(),
                ajo,
                bridge_addr
            );
        }
        if ajo != 0 {
            Asm::patch_jump_for_descr(guard_fd, bridge_addr);
        } else {
            majit_ir::debug::log_one(
                "jit-backend",
                "dynasm-bridge WARNING: adr_jump_offset=0, bridge NOT patched!",
            );
        }

        // llmodel.py free_loop_and_bridges asmmemmgr_blocks parity: store the entire
        // bridge CompiledCode on the owning loop token. This keeps
        // both the arena-backed mapped code AND the fail_descrs
        // (DescrRef) alive. Recovery stubs embed raw pointers to
        // these Arcs — dropping them would create dangling pointers
        // when a bridge-internal guard fires.  RPython's asmmemmgr
        // ties code blocks and their resume descriptors to the same
        // compiled_loop_token lifetime.
        //
        // `compile.py record_loop_or_bridge` attaches a
        // bridge's resume descrs to the original loop's CLT, so the
        // tracer batch lands on `original_token`'s
        // `asmmemmgr_gcreftracers` (`model.py`).
        self.register_fail_descrs(original_token, &compiled.fail_descrs);
        if let Some(table) = gc_table {
            self.register_gc_table(original_token, table);
        }

        // `compile.py record_loop_or_bridge`: a bridge's ResumeDescrs
        // inherit the original loop's CompiledLoopToken.  See the
        // sibling `compile_loop` site for the parity rationale on the
        // `is_resume_guard()` predicate (`compile.py:185`).
        if let Some(clt) = original_token.compiled_loop_token() {
            for descr in compiled.fail_descrs.iter() {
                if !descr.is_resume_guard() {
                    continue;
                }
                if let Some(fd) = descr.as_fail_descr() {
                    fd.set_rd_loop_token_clt(std::sync::Arc::clone(&clt)
                        as std::sync::Arc<dyn std::any::Any + Send + Sync>);
                }
            }
        }

        let source_guard = compiled.source_guard;
        let clt = original_token.compiled_loop_token_expect();
        clt.asmmemmgr_blocks.lock().push(Box::new(compiled));
        if let Some(key) = source_guard {
            clt.compiled_bridge_addrs.lock().insert(key, bridge_addr);
        }

        Ok(AsmInfo {
            code_addr: bridge_addr,
            code_size,
        })
    }

    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame {
        let ran = self.run_compiled_frame(token, args);
        self.deadframe_from_run(token, ran)
    }

    /// `warmstate.py execute_assembler` int fast path, on the frame
    /// `make_execute_token` just returned.
    ///
    /// `genop_finish` stores the result at `jf_frame[0]` and
    /// `handle_fail_done_with_this_frame` reads that slot for
    /// `done_with_this_frame_descr_int`. The frame is released here, which
    /// is the deadframe drop the general path would run after `get_int_value`.
    fn execute_token_done_int(
        &self,
        token: &JitCellToken,
        args: &[Value],
    ) -> Result<i64, DeadFrame> {
        let ran = self.run_compiled_frame(token, args);
        self.done_int_from_ran(token, ran)
    }

    /// `llmodel.py execute_token` allocates the frame, stores the words, and
    /// calls. `warmstate.py execute_assembler` reads `get_latest_descr` and,
    /// for `DoneWithThisFrameDescrInt`, returns `get_result` (slot 0).
    #[inline]
    fn execute_token_done_int_raw(
        &self,
        token: &JitCellToken,
        args: &[i64],
    ) -> Result<i64, DeadFrame> {
        if raw_done_entry_use_general(exec_diag_enabled()) {
            return self.execute_token_done_int_raw_general(token, args);
        }
        let (jf_ptr, tip, num_slots, descr_raw) = self.run_done_raw_entry(token, args);
        if self.finish_is_done_int(descr_raw) {
            // A GC frame was already resolved in `run_done_raw_entry`.
            // Host frames walk `jf_forward` (`jitframe.py` `jitframe_resolve`).
            let live = if self.jitframe_facts.is_gc_object {
                tip
            } else {
                unsafe { JitFrame::resolve_forward(tip) }
            };
            let value = done_int_slot0(live);
            park_or_free_done_entry_frame(token, jf_ptr, tip, self.jitframe_facts.is_gc_object);
            return Ok(value);
        }
        Err(self.raw_entry_deadframe(token, jf_ptr, tip, num_slots))
    }

    /// `warmstate.py execute_assembler` ref fast path. `done_ref_from_ran`
    /// reads slot 0 and releases the frame; a nursery frame is not freed.
    fn execute_token_done_ref(
        &self,
        token: &JitCellToken,
        args: &[Value],
    ) -> Result<usize, DeadFrame> {
        let ran = self.run_compiled_frame(token, args);
        self.done_ref_from_ran(token, ran)
    }

    /// Same B1 shape as [`Backend::execute_token_done_int_raw`]: inline hot
    /// path, cold general fallback when exec diag is on. Slot 0 is
    /// `get_ref_value_direct`.
    #[inline]
    fn execute_token_done_ref_raw(
        &self,
        token: &JitCellToken,
        args: &[i64],
    ) -> Result<usize, DeadFrame> {
        if raw_done_entry_use_general(exec_diag_enabled()) {
            return self.execute_token_done_ref_raw_general(token, args);
        }
        let (jf_ptr, tip, num_slots, descr_raw) = self.run_done_raw_entry(token, args);
        if self.finish_is_done_ref(descr_raw) {
            // A GC frame was already resolved in `run_done_raw_entry`.
            // Host frames walk `jf_forward` (`jitframe.py` `jitframe_resolve`).
            let live = if self.jitframe_facts.is_gc_object {
                tip
            } else {
                unsafe { JitFrame::resolve_forward(tip) }
            };
            let value = done_ref_slot0(live);
            park_or_free_done_entry_frame(token, jf_ptr, tip, self.jitframe_facts.is_gc_object);
            return Ok(value);
        }
        Err(self.raw_entry_deadframe(token, jf_ptr, tip, num_slots))
    }

    /// Override execute_token_ints_raw to return the FULL jitframe
    /// content (all slots), matching Cranelift's behavior.
    /// RPython: the deadframe IS the jitframe — all slots are accessible.
    fn execute_token_ints_raw(
        &self,
        token: &JitCellToken,
        args: &[i64],
    ) -> majit_backend::RawExecResult {
        // Same rationale as `execute_token`: the inline probe emitted by
        // `_call_header` (x86/aarch64 assembler.rs) is now the sole
        // stack-overflow detection site, so no runner-level probe is
        // needed here.
        let compiled = Self::get_compiled(token);
        let entry = compiled.entry_ptr();

        // Same non-null JITFRAMEINFO + `jfi_frame_depth` sizing as
        // `execute_token` (jitframe.py) — the bridge realloc slowpath
        // needs the frame_info, and the depth matches cranelift's
        // `max_output_slots`-sized raw outputs (compiler.rs).
        let clt = unsafe { &*token.compiled_loop_token_ptr() };
        let (fi_ptr, num_slots) = {
            let info = unsafe { &*clt.frame_info.data_ptr() };
            (
                info as *const majit_backend::JitFrameInfo,
                info.depth() as usize,
            )
        };
        assert!(
            num_slots >= Self::input_slot(args.len()),
            "execute_token_ints_raw: frame depth {num_slots} < input top {} for {} args",
            Self::input_slot(args.len()),
            args.len()
        );
        let (jf_ptr, gc_object, _arg_roots) =
            alloc_entry_jitframe(JitFrame::alloc_size(num_slots), &[]);
        unsafe { JitFrame::init(jf_ptr, fi_ptr, num_slots) };

        for (i, &val) in args.iter().enumerate() {
            unsafe { crate::llmodel::set_int_value(jf_ptr, Self::input_slot(i), val as isize) };
        }
        // llmodel.py `execute_token`: `llop.gc_writebarrier(ll_frame)`.
        if gc_object {
            with_gc_ll_descr(|gc| jitframe_write_barrier(gc, jf_ptr));
        }

        let func: unsafe extern "C" fn(*mut JitFrame, *const i64) -> *mut JitFrame =
            unsafe { std::mem::transmute(entry) };
        let result_jf = unsafe { func(jf_ptr, crate::jit_threadlocalref_base()) };

        let jf_descr_raw = unsafe { crate::llmodel::get_latest_descr(result_jf) };
        let descr = self.find_descr_by_ptr(jf_descr_raw);
        let descr_fd = descr.as_fail_descr();

        let fail_arg_types = descr_fd.fail_arg_types();
        let num_fail_args = fail_arg_types.len();
        let mut outputs: Vec<i64> = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            outputs.push(unsafe { crate::llmodel::get_int_value_direct(result_jf, i) as i64 });
        }
        // PyPy `llmodel.py _decode_pos` parity: read each
        // fail-arg slot from `descr.rd_locs[i]`.  Out-of-range index
        // (synthetic descrs without rd_locs) falls back to identity.
        let rd_locs_len = descr_fd.rd_locs().len();
        let mut typed_outputs = Vec::with_capacity(num_fail_args);
        for (i, fail_arg_type) in fail_arg_types.iter().enumerate() {
            let raw = if i < rd_locs_len {
                match crate::guard::decode_rd_loc_slot(descr_fd, i) {
                    Some(slot) => outputs.get(slot).copied().unwrap_or(0),
                    None => 0,
                }
            } else {
                outputs.get(i).copied().unwrap_or(0)
            };
            typed_outputs.push(match fail_arg_type {
                Type::Ref => Value::Ref(GcRef(raw as usize)),
                Type::Float => Value::Float(f64::from_bits(raw as u64)),
                // `Type::Void` is the resume.py:411-417 hole sentinel —
                // the slot's value is reconstructed from the resume
                // snapshot (TAGCONST/TAGVIRTUAL), not from the deadframe.
                // Surfacing `Value::Void` keeps the gcmap and downstream
                // type-tag dispatchers from misclassifying it as a live
                // `Ref` and leaking a NULL `GcRef`.
                Type::Void => Value::Void,
                Type::Int => Value::Int(raw),
            });
        }
        let descr_arc = descr.to_arc();
        let exit_layout = Some(crate::guard::layout_for_fail_descr(
            &descr_arc,
            descr_fd.fail_index_per_trace(),
            descr_fd.trace_id(),
        ));

        // grab_exc_value (llmodel.py): read jf_guard_exc off the deadframe
        // tip before the libc jitframe chain is freed (same as execute_token).
        let exception_value = GcRef(unsafe { (*result_jf).jf_guard_exc });
        let savedata = GcRef(unsafe { (*result_jf).jf_savedata });
        let guard_value_operand = majit_backend::guard_value_counter_slot(descr_fd)
            .map(|slot| unsafe { crate::llmodel::get_int_value_direct(result_jf, slot) as i64 });

        if !gc_object {
            unsafe { majit_backend::libc_deadframe::free_jitframe_chain(jf_ptr) };
        }

        majit_backend::RawExecResult {
            outputs,
            typed_outputs,
            exit_layout,
            savedata: (!savedata.is_null()).then_some(savedata),
            exception_value,
            fail_index: descr_fd.fail_index_per_trace(),
            trace_id: descr_fd.trace_id(),
            is_finish: descr_fd.is_finish(),
            is_exit_frame_with_exception: descr_fd.is_exit_frame_with_exception(),
            status: descr_fd.get_status(),
            guard_value_operand,
            descr_arc,
        }
    }

    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr {
        match frame {
            DeadFrame::JitFrame(data) => data.fail_descr.as_fail_descr(),
            DeadFrame::LibcJitFrame(data) => data.fail_descr.as_fail_descr(),
            DeadFrame::Boxed(_) => panic!("dynasm deadframe is a jitframe"),
        }
    }

    fn force(&self, force_token: GcRef) -> Option<DeadFrame> {
        if force_token.is_null() {
            return None;
        }
        let frame = unsafe { JitFrame::resolve(force_token.0 as *mut JitFrame) };
        let force_descr = unsafe { (*frame).jf_force_descr };
        assert_ne!(force_descr, 0, "force: jf_force_descr is null");
        unsafe { (*frame).jf_descr = force_descr };

        let descr = self.fail_descr_arc_from_addr(force_descr);
        let num_slots = descr
            .as_fail_descr()
            .expect("force descriptor must implement FailDescr")
            .fail_arg_types()
            .len();
        // `llmodel.py force` casts the resolved frame to a GCREF and
        // returns it — the forced frame IS the deadframe, and it belongs to
        // the compiled run that is still executing, so this deadframe borrows
        // it rather than taking the chain over.
        if jitframe_is_gc_managed() {
            Some(DeadFrame::JitFrame(JitFrameDeadFrame::borrowing(
                GcRef(frame as usize),
                ExitDescr::owned(descr),
                None,
            )))
        } else {
            Some(DeadFrame::LibcJitFrame(unsafe {
                LibcJitFrameDeadFrame::borrowing(
                    frame,
                    num_slots,
                    majit_backend::deadframe::ExitDescr::owned(descr),
                    None,
                )
            }))
        }
    }

    fn is_force_token_armed(&self, force_token: GcRef) -> bool {
        if force_token.is_null() {
            return false;
        }
        let frame = unsafe { JitFrame::resolve(force_token.0 as *mut JitFrame) };
        unsafe { (*frame).jf_force_descr != 0 }
    }

    fn get_latest_descr_arc(&self, frame: &DeadFrame) -> Arc<dyn majit_ir::Descr> {
        match frame {
            DeadFrame::JitFrame(data) => data.fail_descr.to_arc(),
            DeadFrame::LibcJitFrame(data) => data.fail_descr.to_arc(),
            DeadFrame::Boxed(_) => panic!("dynasm deadframe is a jitframe"),
        }
    }

    /// `cpu.grab_exc_value(deadframe)` (llmodel.py): return the
    /// `jf_guard_exc` value captured off the deadframe tip in `execute_token`.
    /// The exc=True failure-recovery stub stored pos_exc_value there for
    /// must_save_exception guards (GUARD_EXCEPTION / GUARD_NO_EXCEPTION /
    /// GUARD_NOT_FORCED); other guards leave it NULL.
    fn grab_exc_value(&self, frame: &DeadFrame) -> GcRef {
        match frame {
            DeadFrame::JitFrame(data) => data.grab_exc_value(),
            DeadFrame::LibcJitFrame(data) => data.exc_value(),
            DeadFrame::Boxed(_) => panic!("dynasm deadframe is a jitframe"),
        }
    }

    fn set_savedata_ref(&self, frame: &mut DeadFrame, data: GcRef) {
        match frame {
            DeadFrame::JitFrame(jf) => {
                // llmodel.py set_savedata_ref is a GCREF field store.
                let mut data_slot = data.0 as i64;
                let depth = majit_gc::shadow_stack::resume_ref_roots_depth();
                unsafe {
                    majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_mut(
                        &mut data_slot,
                    ));
                }
                majit_gc::gc_write_barrier(jf.jf_gcref());
                majit_gc::shadow_stack::pop_resume_ref_roots_to(depth);
                jf.set_savedata_ref(GcRef(data_slot as usize));
            }
            DeadFrame::LibcJitFrame(jf) => jf.set_savedata_ref(data),
            DeadFrame::Boxed(_) => panic!("dynasm deadframe is a jitframe"),
        }
    }

    fn get_savedata_ref(&self, frame: &DeadFrame) -> Option<GcRef> {
        let r = match frame {
            DeadFrame::JitFrame(jf) => jf.get_savedata_ref(),
            DeadFrame::LibcJitFrame(jf) => jf.get_savedata_ref(),
            DeadFrame::Boxed(_) => panic!("dynasm deadframe is a jitframe"),
        };
        if r.is_null() { None } else { Some(r) }
    }

    fn clear_stored_exception(&self) {
        crate::jit_exc_clear();
    }

    /// `llmodel.py free_loop_and_bridges` parity.  When the CLT
    /// drops, `asmmemmgr_gcreftracers` releases its strong refs to the
    /// baked `FailDescrCell` Arcs; subsequent recovery from a stale
    /// address would be UB, but the JIT-emitted code holding the address
    /// is also retired when its CLT drops, so no live caller can reach a
    /// stale `descr_addr`.
    ///
    /// `compile_loop` inserts the token into `CALL_ASSEMBLER_TARGETS`
    /// (`runner.rs`) and that map stores a strong
    /// `Arc<CompiledLoopToken>`; without removal here the CLT — and the
    /// fail-descr cells it pins — would live for the entire process
    /// lifetime.  Drop the entry so the Arc chain unwinds.
    fn free_loop(&mut self, token: &JitCellToken) {
        unregister_dynasm_ca_target(token.identity_key());
    }

    fn fail_descr_arc_from_addr(&self, descr_addr: usize) -> majit_ir::DescrRef {
        // `history.py AbstractDescr.show(cpu, descr_gcref) =
        // cast_gcref_to_instance(...)` parity.  `descr_addr` is the thin
        // pointer to the `FailDescrCell` that was baked at codegen time;
        // recovery is a direct `Arc::from_raw` with a refcount bump.
        // Safety: the cell is kept alive by `clt.asmmemmgr_gcreftracers`
        // for the life of the executing JIT code (`model.py`).
        unsafe { majit_ir::recover_fail_descr_cell(descr_addr) }
    }

    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64 {
        match frame {
            DeadFrame::JitFrame(data) => data.get_int(index),
            DeadFrame::LibcJitFrame(data) => data.get_int(index),
            DeadFrame::Boxed(_) => panic!("dynasm deadframe is a jitframe"),
        }
    }

    fn get_value_direct(&self, frame: &DeadFrame, slot: usize) -> i64 {
        match frame {
            DeadFrame::JitFrame(data) => data.get_int_at_slot(slot),
            DeadFrame::LibcJitFrame(data) => data.get_int_at_slot(slot),
            DeadFrame::Boxed(_) => panic!("dynasm deadframe is a jitframe"),
        }
    }

    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64 {
        match frame {
            DeadFrame::JitFrame(data) => data.get_float(index),
            DeadFrame::LibcJitFrame(data) => data.get_float(index),
            DeadFrame::Boxed(_) => panic!("dynasm deadframe is a jitframe"),
        }
    }

    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> GcRef {
        match frame {
            DeadFrame::JitFrame(data) => data.get_ref(index),
            DeadFrame::LibcJitFrame(data) => data.get_ref(index),
            DeadFrame::Boxed(_) => panic!("dynasm deadframe is a jitframe"),
        }
    }

    fn invalidate_loop(&self, token: &JitCellToken) {
        let _writing = majit_backend::AssemblerWriting::enter();
        // `model.py:145` activates the guards in the loop AND in its attached
        // bridges. Both compile against the loop's `invalidate_positions`, and
        // `invalidate` is what writes the branch into each of them.
        token.invalidate();
    }

    // assembler.py:1138 redirect_call_assembler
    fn redirect_call_assembler(
        &self,
        old: &JitCellToken,
        new: &JitCellToken,
    ) -> Result<(), BackendError> {
        let _writing = majit_backend::AssemblerWriting::enter();
        let old_compiled = Self::get_compiled(old);
        let new_compiled = Self::get_compiled(new);
        // x86/assembler.py:1146-1151 update_frame_info parity: propagate
        // new loop's frame depth onto the old token and every token in
        // its existing redirect chain, using the `baseofs` obtained from
        // `cpu.get_baseofs_of_frame_field()` so `jfi_frame_size` follows
        // jitframe.py `base_ofs + new_depth * SIZEOFSIGNED`.
        let baseofs = Self::get_baseofs_of_frame_field();
        if let (Some(new_clt), Some(old_clt)) =
            (new.compiled_loop_token(), old.compiled_loop_token())
        {
            // Seed new's CompiledLoopToken.frame_info.jfi_frame_depth
            // from the backend-specific compiled code depth so
            // update_frame_info has a non-zero value to propagate.
            let new_depth = new_compiled.frame_depth.load(Ordering::Acquire);
            new_clt
                .frame_info
                .lock()
                .update_frame_depth(baseofs, new_depth as i64);
            // model.py update_frame_info — pass old CLT with a
            // weak ref for the "append self to chain" step (line 328
            // `new_loop_tokens.append(weakref.ref(oldlooptoken))`).
            let old_weak = Arc::downgrade(&old_clt);
            new_clt.update_frame_info(&old_clt, old_weak, baseofs);
            // Keep the backend-specific frame_depth in lockstep so bridge
            // codegen's existing readers (CompiledCode.frame_depth) also
            // see the propagated value. TODO: RPython
            // reads the depth back from `compiled_loop_token.frame_info`;
            // dynasm's codegen reads `CompiledCode.frame_depth`. Writing
            // both keeps the orthodox field authoritative while the
            // reader migration lands.
            old_compiled
                .frame_depth
                .fetch_max(new_depth, Ordering::Release);
        }
        let old_addr = old_compiled.entry_ptr();
        let new_addr = new_compiled.entry_ptr();
        Asm::redirect_call_assembler(old, new, old_addr, new_addr);
        Self::redirect_call_assembler_target(old.identity_key(), new_addr as usize);
        Ok(())
    }

    fn bridge_was_compiled(
        &self,
        token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> bool {
        self.lookup_bridge_addr(token, source_trace_id, source_fail_index) != 0
    }

    /// `patch_pending_failure_recoveries` stamps every assembled guard's
    /// recovery stub address into `adr_jump_offset`, and `patch_jump_for_descr`
    /// zeroes it once the guard's jump has been redirected into a bridge. A
    /// non-zero offset therefore still names the guard's own stub: no bridge.
    /// Zero is left to the token map, since a guard that never received a
    /// stub reads the same way as a patched one.
    fn bridge_attached(&self, descr: &dyn majit_ir::FailDescr) -> Option<bool> {
        (descr.adr_jump_offset() != 0).then_some(false)
    }

    fn bh_new(&self, sizedescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        bh_alloc_struct(sizedescr) as i64
    }

    fn bh_new_with_vtable(&self, sizedescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        let vtable = sizedescr.get_vtable();
        // `GcLLDescr_boehm._bh_malloc` → `malloc_fixedsize`. A published
        // function replaces the collector allocation; the caller still
        // stores the vtable at `vtable_offset`.
        let ptr = if let Some(ptr) = call_malloc_fixedsize(sizedescr.as_size()) {
            ptr.cast()
        } else {
            bh_alloc_struct(sizedescr)
        };
        if !ptr.is_null() {
            // llmodel.py:780-782 writes the type word at vtable_offset.
            // The class word uses the same host-configured offset.
            unsafe {
                majit_backend::write_new_with_vtable_header(
                    ptr.cast(),
                    vtable,
                    self.vtable_offset,
                    self.w_class_offset,
                );
            }
        }
        ptr as i64
    }

    /// llmodel.py bh_new_array / bh_new_array_clear.
    fn bh_new_array(&self, length: i64, arraydescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        let Ok(length) = usize::try_from(length) else {
            return 0;
        };
        let (base_size, itemsize, _sign) = arraydescr.unpack_arraydescr_size();
        let len_offset = arraydescr
            .array_len_offset()
            .expect("bh_new_array requires ArrayDescr.lendescr");
        // descr.py `ArrayDescr.get_type_id(): assert self.tid` —
        // allocation requires a real GC type id; tid=0 means the descr
        // never went through `gc.py:548 set_type_id` and the GC tracer
        // would lack the per-item visit shape.
        // `BhDescr::resolve_gc_tid` maps the serialized `path_hash` cache key
        // back to the allocated GC tid (`gc.py:544-549`) so the header carries
        // the per-item visit shape the tracer reads.
        let type_id = arraydescr.resolve_gc_tid();
        assert!(
            type_id != 0,
            "bh_new_array requires ArrayDescr.tid (descr.py:340) — got 0"
        );
        // Old-gen for the same reason as `bh_new_with_vtable`: a resume
        // materializes the inlined frame's `locals_cells_stack` array and the
        // blackhole holds it (via the frame) across the minor collections the
        // deep forward recursion triggers.  A non-moving array keeps the frame's
        // field valid without a cross-generation write barrier, and keeps the
        // whole materialized graph in one generation.
        dynasm_alloc_oldgen_varsize_typed_and_set_len(
            type_id, base_size, itemsize, len_offset, length,
        ) as i64
    }

    /// llmodel.py bh_new_array_clear = bh_new_array.
    fn bh_new_array_clear(&self, length: i64, arraydescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        self.bh_new_array(length, arraydescr)
    }

    /// `LLtypeMixin.bh_newstr` → `gc_ll_descr.gc_malloc_str`.
    fn bh_newstr(&self, length: i64) -> i64 {
        let Ok(length) = u64::try_from(length) else {
            return 0;
        };
        dynasm_malloc_str(majit_gc::lowlevel_str_type_id() as u64, length) as i64
    }

    /// `LLtypeMixin.bh_newunicode` → `gc_ll_descr.gc_malloc_unicode`.
    fn bh_newunicode(&self, length: i64) -> i64 {
        let Ok(length) = u64::try_from(length) else {
            return 0;
        };
        dynasm_malloc_unicode(majit_gc::lowlevel_unicode_type_id() as u64, length) as i64
    }

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer through the installed gc_ll_descr.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        self.lookup_typeid_from_classptr(classptr)
    }

    /// llmodel.py bh_call_i: ABI-correct dispatch via the shared call stub.
    ///
    /// Routes through `majit_backend::call_stub::bh_call_i_dispatch`, whose
    /// signature is built in `arg_classes` declaration order to match
    /// `descr.py` / `descr.py create_call_stub`.
    fn bh_call_i(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> i64 {
        assert_ne!(func, 0, "bh_call_i: null function pointer");
        majit_backend::call_stub::verify_result_type(calldescr.result_type, "iS");
        unsafe {
            majit_backend::call_stub::bh_call_i_with_descr(
                func as usize,
                args_i,
                args_r,
                args_f,
                calldescr,
            )
        }
    }

    /// llmodel.py bh_call_r: GcRef-returning parallel of `bh_call_i`.
    /// `lltype.Ptr(lltype.GcStruct, ...)` lowers to a host pointer that
    /// matches `i64` on 64-bit, so we transmute via the shared int
    /// dispatcher and wrap the result.
    fn bh_call_r(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        assert_ne!(func, 0, "bh_call_r: null function pointer");
        majit_backend::call_stub::verify_result_type(calldescr.result_type, "r");
        let raw = unsafe {
            majit_backend::call_stub::bh_call_i_with_descr(
                func as usize,
                args_i,
                args_r,
                args_f,
                calldescr,
            )
        };
        majit_ir::GcRef(raw as usize)
    }

    /// llmodel.py bh_call_f / descr.py create_call_stub
    /// (`RESULT == lltype.Float`) parity: route through the f64-typed
    /// dispatcher so an f64-returning C callee delivers via xmm0 / d0
    /// instead of rax / x0.
    fn bh_call_f(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> f64 {
        assert_ne!(func, 0, "bh_call_f: null function pointer");
        majit_backend::call_stub::verify_result_type(calldescr.result_type, "fL");
        unsafe {
            majit_backend::call_stub::bh_call_f_with_descr(
                func as usize,
                args_i,
                args_r,
                args_f,
                calldescr,
            )
        }
    }

    /// llmodel.py bh_call_v / descr.py create_call_stub
    /// (`RESULT == lltype.Void`) parity: dispatch the funcptr through
    /// the void-typed `bh_call_v_dispatch` so a genuinely void C callee
    /// is called with the right C-ABI signature. Re-routing through
    /// `bh_call_i_dispatch` (a `extern "C" fn(...) -> i64` transmute)
    /// reads garbage from rax/x0 for true void returns.
    fn bh_call_v(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) {
        assert_ne!(func, 0, "bh_call_v: null function pointer");
        majit_backend::call_stub::verify_result_type(calldescr.result_type, "v");
        unsafe {
            majit_backend::call_stub::bh_call_v_with_descr(
                func as usize,
                args_i,
                args_r,
                args_f,
                calldescr,
            );
        }
    }

    /// llmodel.py bh_raw_load_i(addr, offset, descr).
    fn bh_raw_load_i(
        &self,
        addr: i64,
        offset: i64,
        descr: &majit_jitcode::jitcode::BhDescr,
    ) -> i64 {
        // llmodel.py: ofs, size, sign = self.unpack_arraydescr_size(descr)
        // ofs == 0 always for raw lengthless arrays (llmodel.py assert)
        let size = descr.as_itemsize();
        let sign = descr.is_item_signed();
        // llmodel.py: return self.read_int_at_mem(addr, offset, size, sign)
        self.read_int_at_mem(addr, offset, size, sign)
    }

    /// llmodel.py bh_raw_store_i(addr, offset, newvalue, descr).
    fn bh_raw_store_i(
        &self,
        addr: i64,
        offset: i64,
        newvalue: i64,
        descr: &majit_jitcode::jitcode::BhDescr,
    ) {
        // llmodel.py: ofs, size, _ = self.unpack_arraydescr_size(descr)
        // ofs == 0 always for raw lengthless arrays (llmodel.py assert)
        let size = descr.as_itemsize();
        // llmodel.py: self.write_int_at_mem(addr, offset, size, newvalue)
        self.write_int_at_mem(addr, offset, size, newvalue);
    }

    /// llmodel.py bh_raw_load_f(addr, offset, descr).
    fn bh_raw_load_f(
        &self,
        addr: i64,
        offset: i64,
        _descr: &majit_jitcode::jitcode::BhDescr,
    ) -> f64 {
        // llmodel.py: return self.read_float_at_mem(addr, offset)
        self.read_float_at_mem(addr, offset)
    }

    /// llmodel.py bh_raw_store_f(addr, offset, newvalue, descr).
    fn bh_raw_store_f(
        &self,
        addr: i64,
        offset: i64,
        newvalue: f64,
        _descr: &majit_jitcode::jitcode::BhDescr,
    ) {
        // llmodel.py: self.write_float_at_mem(addr, offset, newvalue)
        self.write_float_at_mem(addr, offset, newvalue);
    }

    /// `llmodel.py bh_getfield_gc_i` →
    /// `read_int_at_mem(struct, ofs, size, sign)`.  Threads the per-field
    /// `(offset, size, sign)` tuple from `BhDescr.unpack_fielddescr_size`
    /// to the size dispatch in `llmodel.py`.
    fn bh_getfield_gc_i(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) -> i64 {
        let (offset, size, sign) = fielddescr.unpack_fielddescr_size();
        self.read_int_at_mem(struct_ptr, offset as i64, size, sign)
    }

    /// `llmodel.py bh_setfield_gc_i` →
    /// `write_int_at_mem(struct, ofs, size, value)`.  Sign discarded by
    /// `unpack_fielddescr_size` consumer (`llmodel.py`); only
    /// `(offset, size)` reach the store.
    fn bh_setfield_gc_i(
        &self,
        struct_ptr: i64,
        value: i64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let (offset, size, _sign) = fielddescr.unpack_fielddescr_size();
        self.write_int_at_mem(struct_ptr, offset as i64, size, value);
    }

    fn bh_setfield_gc_r(
        &self,
        struct_ptr: i64,
        value: GcRef,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let offset = fielddescr.as_offset();
        // llmodel.py `bh_setfield_gc_r` → `write_ref_at_mem`: the write
        // barrier is implied by the framework GC transformer around the ref
        // store (`transform_generic_set`, before the bare store), identical
        // to the array ref setters. The blackhole has no inline
        // TRACK_YOUNG_PTRS test, so use the managed-guarded flag-checking
        // barrier (see bh_setarrayitem_gc_r). Without it, a young value
        // stored into an old managed struct is not remembered and the next
        // minor collection can free it while still referenced.
        dynasm_write_barrier_if_managed(struct_ptr as u64);
        unsafe { *((struct_ptr as *mut u8).add(offset) as *mut usize) = value.0 };
    }

    /// llmodel.py bh_getarrayitem_gc_i: ofs=base_size, size+sign
    /// from `unpack_arraydescr_size`; route through `read_int_at_mem`
    /// at `gcref + ofs + index*size`.
    fn bh_getarrayitem_gc_i(
        &self,
        array_ptr: i64,
        index: i64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) -> i64 {
        let (base_size, itemsize, sign) = arraydescr.unpack_arraydescr_size();
        let offset = (base_size as i64) + index * (itemsize as i64);
        self.read_int_at_mem(array_ptr, offset, itemsize, sign)
    }

    /// model.py / llmodel.py bh_arraylen_gc.
    /// Read the length word from `arraydescr.lendescr.offset`.
    fn bh_arraylen_gc(&self, array_ptr: i64, arraydescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        let ofs = arraydescr
            .array_len_offset()
            .expect("bh_arraylen_gc requires ArrayDescr.lendescr");
        self.read_int_at_mem(array_ptr, ofs as i64, std::mem::size_of::<usize>(), true)
    }

    /// llmodel.py bh_getarrayitem_gc_r: ofs=base_size, item width
    /// fixed at `WORD` (8 bytes).  Direct deref of `*const usize` mirrors
    /// `bh_getfield_gc_r`'s pattern so the GcRef carries the raw machine
    /// word from memory.
    fn bh_getarrayitem_gc_r(
        &self,
        array_ptr: i64,
        index: i64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) -> majit_ir::GcRef {
        let base_size = arraydescr.array_base_size();
        let offset = (base_size as i64) + index * 8;
        let raw = unsafe { *((array_ptr as *const u8).offset(offset as isize) as *const usize) };
        majit_ir::GcRef(raw)
    }

    /// llmodel.py bh_getarrayitem_gc_f: ofs=base_size, item
    /// width fixed at `sizeof(FLOATSTORAGE)` (8 bytes).  Routes through
    /// `read_float_at_mem` for the same `read_unaligned` safety as the
    /// field sibling.
    fn bh_getarrayitem_gc_f(
        &self,
        array_ptr: i64,
        index: i64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) -> f64 {
        let base_size = arraydescr.array_base_size();
        let offset = (base_size as i64) + index * 8;
        self.read_float_at_mem(array_ptr, offset)
    }

    /// llmodel.py bh_setarrayitem_gc_i.
    fn bh_setarrayitem_gc_i(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: i64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let (base_size, itemsize, _sign) = arraydescr.unpack_arraydescr_size();
        let offset = (base_size as i64) + index * (itemsize as i64);
        self.write_int_at_mem(array_ptr, offset, itemsize, newvalue);
    }

    /// llmodel.py bh_setarrayitem_gc_r.
    fn bh_setarrayitem_gc_r(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: majit_ir::GcRef,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let base_size = arraydescr.array_base_size();
        let offset = (base_size as i64) + index * 8;
        // llmodel.py `bh_setarrayitem_gc_r`: the generic flag-checking
        // barrier, then the store. The blackhole has no inline
        // TRACK_YOUNG_PTRS test ahead of the call, so it must not reuse the
        // JIT-only `jit_remember_young_pointer_from_array` (which assumes the
        // flag was already tested and would unconditionally remember an
        // untracked target — e.g. a reconstructed frame array outside the GC
        // heap — corrupting the remembered set).
        dynasm_write_barrier_if_managed(array_ptr as u64);
        unsafe {
            *((array_ptr as *mut u8).offset(offset as isize) as *mut usize) = newvalue.0;
        }
    }

    /// llmodel.py bh_setarrayitem_gc_f.
    fn bh_setarrayitem_gc_f(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: f64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let base_size = arraydescr.array_base_size();
        let offset = (base_size as i64) + index * 8;
        self.write_float_at_mem(array_ptr, offset, newvalue);
    }

    /// llmodel.py bh_setinteriorfield_gc_i.  Interior address is
    /// `array_base + arraydescr.basesize + fielddescr.offset + index *
    /// arraydescr.itemsize`; the integer field is `field_size` bytes wide.
    fn bh_setinteriorfield_gc_i(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: i64,
        descr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let majit_jitcode::jitcode::BhDescr::InteriorField { array, field } = descr else {
            panic!("bh_setinteriorfield_gc_i: descr is not an InteriorField: {descr:?}");
        };
        let (base_size, itemsize, _) = array.unpack_arraydescr_size();
        let (foff, fsize, _) = field.unpack_fielddescr_size();
        let offset = (base_size as i64) + (foff as i64) + index * (itemsize as i64);
        self.write_int_at_mem(array_ptr, offset, fsize, newvalue);
    }

    /// llmodel.py bh_setinteriorfield_gc_r.  Pointer-typed
    /// interior field: `WORD`-wide store plus the array write barrier.
    fn bh_setinteriorfield_gc_r(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: majit_ir::GcRef,
        descr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let majit_jitcode::jitcode::BhDescr::InteriorField { array, field } = descr else {
            panic!("bh_setinteriorfield_gc_r: descr is not an InteriorField: {descr:?}");
        };
        let (base_size, itemsize, _) = array.unpack_arraydescr_size();
        let (foff, _, _) = field.unpack_fielddescr_size();
        let offset = (base_size as i64) + (foff as i64) + index * (itemsize as i64);
        unsafe {
            *((array_ptr as *mut u8).offset(offset as isize) as *mut usize) = newvalue.0;
        }
        // Blackhole store — no inline TRACK_YOUNG_PTRS test precedes this
        // call, so use the managed-guarded flag-checking barrier (see
        // bh_setarrayitem_gc_r).
        dynasm_write_barrier_if_managed(array_ptr as u64);
    }

    /// llmodel.py bh_setinteriorfield_gc_f.  Float-typed interior
    /// field: `sizeof(FLOATSTORAGE)`-wide store via `write_float_at_mem`.
    fn bh_setinteriorfield_gc_f(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: f64,
        descr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let majit_jitcode::jitcode::BhDescr::InteriorField { array, field } = descr else {
            panic!("bh_setinteriorfield_gc_f: descr is not an InteriorField: {descr:?}");
        };
        let (base_size, itemsize, _) = array.unpack_arraydescr_size();
        let (foff, _, _) = field.unpack_fielddescr_size();
        let offset = (base_size as i64) + (foff as i64) + index * (itemsize as i64);
        self.write_float_at_mem(array_ptr, offset, newvalue);
    }

    /// llmodel.py bh_getfield_gc_f delegates to read_float_at_mem.
    /// `getfield_vable_f/rd>f` and the floating-point array reader rely
    /// on this — the trait default returns 0.0, which silently produces
    /// wrong results during blackhole resume on float vable fields.
    fn bh_getfield_gc_f(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) -> f64 {
        let (offset, size, _) = fielddescr.unpack_fielddescr_size();
        let Some(ptr) = Self::raw_mem_ptr(struct_ptr, offset as i64) else {
            return 0.0;
        };
        unsafe { majit_backend::llmodel::read_float_at_mem_sized(ptr, 0, size) }
    }

    /// llmodel.py bh_setfield_gc_f delegates to write_float_at_mem.
    /// Mirror of `bh_getfield_gc_f`; the trait default is a silent no-op
    /// which loses writes from `setfield_vable_f/rfd` during resume.
    fn bh_setfield_gc_f(
        &self,
        struct_ptr: i64,
        value: f64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let (offset, size, _) = fielddescr.unpack_fielddescr_size();
        let Some(ptr) = Self::raw_mem_ptr(struct_ptr, offset as i64) else {
            return;
        };
        unsafe { majit_backend::llmodel::write_float_at_mem_sized(ptr, 0, size, value) }
    }

    fn compiled_fail_descr_layouts(
        &self,
        token: &JitCellToken,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = Self::get_compiled(token);
        let trace_id = compiled.trace_id;
        Some(
            compiled
                .fail_descrs
                .iter()
                .enumerate()
                .map(|(idx, d)| crate::guard::layout_for_fail_descr(&d.descr, idx as u32, trace_id))
                .collect(),
        )
    }

    fn compiled_trace_fail_descr_layouts(
        &self,
        token: &JitCellToken,
        trace_id: u64,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = Self::get_compiled(token);
        if compiled.trace_id == trace_id {
            return Some(
                compiled
                    .fail_descrs
                    .iter()
                    .enumerate()
                    .map(|(idx, d)| {
                        crate::guard::layout_for_fail_descr(&d.descr, idx as u32, trace_id)
                    })
                    .collect(),
            );
        }
        // Search bridge fail_descrs in asmmemmgr_blocks.
        let blocks_clt = token.compiled_loop_token_expect();
        let blocks = blocks_clt.asmmemmgr_blocks.lock();
        for block in blocks.iter() {
            if let Some(bridge) = block.downcast_ref::<CompiledCode>()
                && bridge.trace_id == trace_id
            {
                return Some(
                    bridge
                        .fail_descrs
                        .iter()
                        .enumerate()
                        .map(|(idx, d)| {
                            crate::guard::layout_for_fail_descr(&d.descr, idx as u32, trace_id)
                        })
                        .collect(),
                );
            }
        }
        None
    }

    fn compiled_bridge_fail_descr_layouts(
        &self,
        original_token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        // RPython faildescr lookup is by object identity, never misses.
        // majit query-style callers (`bridge_was_compiled` etc.) probe
        // by (trace_id, fail_index) and must treat the miss as `None`
        // — match cranelift's `?` semantics in
        // `compiler.rs`'s `compiled_bridge_fail_descr_layouts`.
        let bridge_addr =
            self.lookup_bridge_addr(original_token, source_trace_id, source_fail_index);
        if bridge_addr == 0 {
            return None;
        }
        let blocks_clt = original_token.compiled_loop_token_expect();
        let blocks = blocks_clt.asmmemmgr_blocks.lock();
        for block in blocks.iter() {
            if let Some(bridge) = block.downcast_ref::<CompiledCode>() {
                let addr = bridge.entry_ptr() as usize;
                if addr == bridge_addr {
                    let bridge_trace_id = bridge.trace_id;
                    return Some(
                        bridge
                            .fail_descrs
                            .iter()
                            .enumerate()
                            .map(|(idx, d)| {
                                crate::guard::layout_for_fail_descr(
                                    &d.descr,
                                    idx as u32,
                                    bridge_trace_id,
                                )
                            })
                            .collect(),
                    );
                }
            }
        }
        None
    }

    /// `pyjitpl.py self.cpu.setup_once()` parity, dispatched by
    /// `MetaInterpStaticData::_setup_once` under the
    /// `globaldata.initialized` gate (`pyjitpl.py`).  All
    /// per-CPU descrs (notably `propagate_exception_descr` via
    /// `set_propagate_exception_descr`) must already be installed
    /// when this runs; the helpers we materialise here bake those
    /// descr pointers as immediates and assert non-zero on build.
    ///
    /// PyPy's `llsupport/assembler.py setup_once` builds the
    /// propagate trampoline + every `_build_malloc_slowpath` variant
    /// (`fixed` / `varsize` / `str` / `unicode`).
    fn setup_once(&mut self) {
        #[cfg(target_arch = "x86_64")]
        {
            self.arch_cpu_ext
                .ensure_propagate_exception_path(&self.descr_attachments);
            self.arch_cpu_ext
                .ensure_malloc_slowpath_fixed(&self.descr_attachments);
            self.arch_cpu_ext
                .ensure_malloc_slowpath_headerless(&self.descr_attachments);
            self.arch_cpu_ext
                .ensure_malloc_slowpath_varsize(&self.descr_attachments);
            self.arch_cpu_ext
                .ensure_malloc_slowpath_str(&self.descr_attachments);
            self.arch_cpu_ext
                .ensure_malloc_slowpath_unicode(&self.descr_attachments);
            self.arch_cpu_ext.ensure_wb_slowpath();
        }
        #[cfg(target_arch = "aarch64")]
        self.arch_cpu_ext.ensure_wb_slowpath();
    }

    /// `backend/<arch>/__init__.py` parity — pyre's dynasm backend
    /// spans x86 and aarch64; the active target_arch picks the name.
    fn backend_name(&self) -> &'static str {
        #[cfg(target_arch = "x86_64")]
        {
            "x86"
        }
        #[cfg(target_arch = "aarch64")]
        {
            "aarch64"
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            "dynasm"
        }
    }

    fn finish_once(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_or_identityhash_without_collector_returns_bitwise_not() {
        std::thread::spawn(|| {
            assert!(!gc_box::present());
            let marker = 0usize;
            let addr = &marker as *const usize as usize;
            let got = dynasm_id_or_identityhash(addr);
            // `boehm.py` `ll_identityhash`: `h = ~cast_adr_to_int(addr)`.
            // The function takes that arm on `!collector_installed()`, and
            // another test in this binary can install the gcrootmap half of
            // it at any time.  Both halves are set-only, so reading `false`
            // after the call proves the call took the Boehm arm.
            if !majit_gc::collector_installed() {
                assert_eq!(got, !addr);
            }
        })
        .join()
        .expect("id_or_identityhash must not panic without a collector");
    }

    #[test]
    fn reference_value_read_does_not_become_a_substructure_address() {
        let referent = 123usize;
        let field_words = [0usize, &referent as *const usize as usize];
        let backend = DynasmBackend::new();
        for flag in [
            majit_ir::descr::ArrayFlag::Pointer,
            majit_ir::descr::ArrayFlag::Struct,
        ] {
            let fd = majit_ir::descr::SimpleFieldDescr::new_with_name(
                0,
                std::mem::size_of::<usize>(),
                std::mem::size_of::<usize>(),
                majit_ir::Type::Ref,
                false,
                flag,
                "Node.value".to_string(),
                "value",
            );
            let bh = majit_jitcode::jitcode::BhDescr::from_field_descr(&fd);
            assert_eq!(
                backend.bh_getfield_gc_r(field_words.as_ptr() as i64, &bh),
                majit_ir::GcRef(field_words[1])
            );
        }
    }

    #[test]
    fn headerless_varsize_slowpath_rejects_negative_and_overflowing_lengths() {
        assert_eq!(dynasm_nursery_slowpath_headerless_varsize(-1, 8, 16), 0);
        assert_eq!(
            dynasm_nursery_slowpath_headerless_varsize(i64::MAX, 2, 0),
            0
        );
    }

    #[test]
    fn boehm_malloc_fixedsize_yields_to_an_installed_collector() {
        extern "C" fn hook(size: usize) -> *mut u8 {
            unsafe { libc::malloc(size.max(1)) as *mut u8 }
        }
        let prev = majit_gc::malloc_fixedsize_addr();
        majit_gc::set_malloc_fixedsize(Some(hook));
        // An installed collector suppresses the hook.
        // The flag is process-global and set-only, so the true arm is driven
        // by the argument the live `collector_installed()` call passes.
        assert!(call_malloc_fixedsize_inner(8, true).is_none());
        assert!(boehm_fixedsize_hook(false).is_some());
        if majit_gc::collector_installed() {
            assert!(call_malloc_fixedsize(8).is_none());
        } else {
            let ptr = call_malloc_fixedsize(8).expect("published hook");
            assert!(!ptr.is_null());
            unsafe { libc::free(ptr as *mut libc::c_void) };
        }
        if prev == 0 {
            majit_gc::set_malloc_fixedsize(None);
        } else {
            let restored: extern "C" fn(usize) -> *mut u8 = unsafe { std::mem::transmute(prev) };
            majit_gc::set_malloc_fixedsize(Some(restored));
        }
    }

    use majit_backend::Backend;
    use majit_backend::jitframe::{
        FIRST_ITEM_OFFSET, JF_DESCR_OFS, JF_FORCE_DESCR_OFS, JF_FORWARD_OFS, JF_FRAME_INFO_OFS,
        JF_FRAME_OFS, JF_GUARD_EXC_OFS, JF_SAVEDATA_OFS, JITFRAME_FIXED_SIZE, LENGTHOFS, SIGN_SIZE,
    };
    use majit_gc::collector::{GcConfig, MiniMarkGC};
    use majit_gc::header::header_of;
    use majit_gc::trace::TypeInfo;
    use majit_ir::descr::SimpleArrayDescr;
    use majit_ir::operand::Operand;
    use majit_ir::{
        CallDescr, DescrRef, EffectInfo, ExtraEffect, InputArg, OopSpecIndex, Op, OpCode, Type,
        Value,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};

    #[test]
    fn varsize_slowpaths_return_null_on_size_overflow() {
        assert_eq!(dynasm_nursery_slowpath_varsize(0, 2, u64::MAX, 0), 0);
        assert_eq!(
            dynasm_alloc_oldgen_varsize_typed_and_set_len(1, 0, 2, 0, usize::MAX),
            0
        );
    }

    #[test]
    fn bridge_fail_locations_drop_resume_holes_like_rpython() {
        let descr =
            majit_backend::make_resume_guard_descr_typed(vec![Type::Int, Type::Ref, Type::Int]);
        let fail_descr = descr.as_fail_descr().expect("resume guard fail descr");
        fail_descr.set_rd_locs(vec![0, 0xFFFF, 1].into());
        // pyjitpl.py initialize_state_from_guard_failure filters the hole
        // before the history is built, so only the two live boxes reach the
        // backend bridge.
        let inputargs = [InputArg::new_int_rc(0), InputArg::new_int_rc(1)];

        let locs = Asm::rebuild_faillocs_from_descr(fail_descr, &inputargs);

        assert_eq!(locs.len(), inputargs.len());
    }

    fn install_test_libc_jitframe_tracer() {
        majit_gc::shadow_stack::register_libc_jitframe_tracer(
            majit_backend::jitframe::jitframe_custom_trace,
        );
    }

    use majit_ir::forwarding::bound_operand_from_opref as rb;

    #[derive(Debug)]
    struct TestPlainCallDescr {
        arg_types: Vec<Type>,
        result_type: Type,
    }

    impl majit_ir::Descr for TestPlainCallDescr {
        fn index(&self) -> u32 {
            u32::MAX
        }

        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestPlainCallDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }

        fn result_type(&self) -> Type {
            self.result_type
        }

        fn result_size(&self) -> usize {
            8
        }

        fn get_extra_info(&self) -> &EffectInfo {
            static INFO: EffectInfo =
                EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
            &INFO
        }
    }

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> majit_ir::OpRc {
        let bx: Vec<Operand> = args.iter().map(|a| rb(*a)).collect();
        let op = Op::new(opcode, &bx);
        op.pos().set(OpRef::op_typed(pos, opcode.result_type()));
        OpRc::new(op)
    }

    fn make_plain_call_descr(arg_types: Vec<Type>, result_type: Type) -> DescrRef {
        Arc::new(TestPlainCallDescr {
            arg_types,
            result_type,
        })
    }

    extern "C" fn return_ref_passthrough(arg: i64) -> i64 {
        arg
    }

    static TEST_HELPER_ALLOC_TYPE_ID: AtomicU32 = AtomicU32::new(u32::MAX);

    const TEST_HELPER_MARKER: i64 = 0x5a5a5a5a_i64;

    extern "C" fn alloc_marked_ref() -> i64 {
        crate::runner::gc_box::DYNASM_ACTIVE_GC.with(|cell| {
            let mut guard = cell.borrow_mut();
            let gc = guard
                .as_mut()
                .expect("alloc_marked_ref requires an active dynasm GC");
            let type_id = TEST_HELPER_ALLOC_TYPE_ID.load(Ordering::Relaxed);
            assert_ne!(type_id, u32::MAX, "test helper type id not initialized");
            let obj = gc.alloc_nursery_typed(type_id, 16);
            unsafe {
                *(obj.0 as *mut i64) = TEST_HELPER_MARKER;
            }
            obj.0 as i64
        })
    }

    extern "C" fn alloc_marked_ref_collecting() -> i64 {
        crate::runner::gc_box::DYNASM_ACTIVE_GC.with(|cell| {
            let mut guard = cell.borrow_mut();
            let gc = guard
                .as_mut()
                .expect("alloc_marked_ref_collecting requires an active dynasm GC");
            let type_id = TEST_HELPER_ALLOC_TYPE_ID.load(Ordering::Relaxed);
            assert_ne!(type_id, u32::MAX, "test helper type id not initialized");
            let obj = gc.alloc_nursery_typed(type_id, 16);
            unsafe {
                *(obj.0 as *mut i64) = TEST_HELPER_MARKER;
            }
            let root_depth = majit_gc::shadow_stack::depth();
            let ss_idx = majit_gc::shadow_stack::push(obj);
            let _bump = gc.alloc_nursery_typed(type_id, 80);
            let updated = majit_gc::shadow_stack::get(ss_idx);
            majit_gc::shadow_stack::pop_to(root_depth);
            updated.0 as i64
        })
    }

    extern "C" fn collect_nursery_via_dynasm_gc() {
        crate::runner::gc_box::DYNASM_ACTIVE_GC.with(|cell| {
            let mut guard = cell.borrow_mut();
            let gc = guard
                .as_mut()
                .expect("collect_nursery_via_dynasm_gc requires an active dynasm GC");
            gc.collect_nursery();
        })
    }

    /// `jitframe.py` — the `JITFRAME` shape a collector with a type table must
    /// carry before `check_jitframe_descr` lets it be installed.
    fn register_jitframe_type(gc: &mut MiniMarkGC) -> u32 {
        let id = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        majit_gc::GcAllocator::set_jitframe_type_id(gc, id);
        id
    }

    fn install_call_assembler_test_layout(jitframe_tid: u32) {
        crate::register_jitframe_layout(crate::JitFrameLayoutInfo {
            jitframe_descrs: Some(majit_gc::rewrite::JitFrameDescrs {
                jitframe_tid,
                jitframe_fixed_size: JITFRAME_FIXED_SIZE,
                jf_frame_info_ofs: JF_FRAME_INFO_OFS,
                jf_descr_ofs: JF_DESCR_OFS,
                jf_force_descr_ofs: JF_FORCE_DESCR_OFS,
                jf_savedata_ofs: JF_SAVEDATA_OFS,
                jf_guard_exc_ofs: JF_GUARD_EXC_OFS,
                jf_forward_ofs: JF_FORWARD_OFS,
                jf_frame_ofs: JF_FRAME_OFS,
                jf_frame_baseitemofs: FIRST_ITEM_OFFSET,
                jf_frame_lengthofs: JF_FRAME_OFS + LENGTHOFS,
                sign_size: SIGN_SIZE,
                jf_frame_itemsize: SIGN_SIZE,
            }),
        });
    }

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr
    /// Verify the dynasm backend's gc_ll_descr round-trips a registered
    /// vtable→type_id mapping (the same contract Cranelift uses).
    #[test]
    fn test_backend_typeid_from_classptr_via_gc_ll_descr() {
        let mut gc = MiniMarkGC::new();
        let int_tid = gc.register_type(TypeInfo::simple(16));
        let int_vtable: usize = 0x2222_3300;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, int_vtable, int_tid);

        let mut backend = DynasmBackend::new();
        register_jitframe_type(&mut gc);
        backend.set_gc_allocator(Box::new(gc));

        let resolved = backend.get_typeid_from_classptr_if_gcremovetypeptr(int_vtable);
        assert_eq!(resolved, Some(int_tid));
        let unknown = backend.get_typeid_from_classptr_if_gcremovetypeptr(0xCAFE_F00D);
        assert_eq!(unknown, None);
    }

    #[test]
    fn test_backend_installs_active_gc_guard_hooks() {
        let mut gc = MiniMarkGC::new();
        let obj_tid = gc.register_type(TypeInfo::object(16));
        let obj = gc.alloc_with_type(obj_tid, 16);

        let mut backend = DynasmBackend::new();
        register_jitframe_type(&mut gc);
        backend.set_gc_allocator(Box::new(gc));

        assert!(majit_gc::supports_guard_gc_type());
        assert!(majit_gc::check_is_object(obj));
        assert_eq!(majit_gc::get_actual_typeid(obj), Some(obj_tid));
        assert_eq!(majit_gc::typeid_is_object(obj_tid), Some(true));
    }

    #[test]
    fn test_input_initial_locs_match_frame_relative_entry_offsets() {
        assert_eq!(
            DynasmBackend::input_initial_loc(0),
            (DynasmBackend::input_slot(0) * crate::jitframe::SIZEOFSIGNED) as i32
        );
        assert_eq!(
            DynasmBackend::input_initial_loc(1),
            (DynasmBackend::input_slot(1) * crate::jitframe::SIZEOFSIGNED) as i32
        );
    }

    #[test]
    fn compile_loop_records_token_inputarg_types() {
        let mut backend = DynasmBackend::new();
        // `compile.py make_and_attach_done_descrs` parity:
        // FINISH emission stamps the cpu-attached singleton Arc into
        // `compiled.fail_descrs`; without attachment the backend has
        // nothing to push.  Production reaches this state through
        // `MetaInterp::new`; backend-only tests must invoke this helper.
        backend.attach_default_test_descrs();
        let inputargs = vec![InputArg::new_ref_rc(0), InputArg::new_int_rc(1)];
        // Match the typed `InputArg{Ref,Int}` boxes registered by the
        // backend regalloc — variant-aware Eq makes Untyped(N) and
        // InputArg{Ref,Int}(N) distinct keys.
        let ops = vec![mk_op(
            OpCode::Finish,
            &[OpRef::input_arg_ref(0), OpRef::input_arg_int(1)],
            OpRef::NONE.raw(),
        )];

        let token = JitCellToken::new(1499);
        backend.compile_loop(&inputargs, &ops, &token).unwrap();

        assert_eq!(token.inputarg_types().to_vec(), vec![Type::Ref, Type::Int]);
    }

    #[test]
    fn compile_loop_accepts_nonzero_inputarg_indices() {
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let inputargs = vec![InputArg::new_int_rc(10), InputArg::new_int_rc(20)];
        let ops = vec![
            mk_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(10), OpRef::input_arg_int(20)],
                30,
            ),
            mk_op(OpCode::Finish, &[OpRef::int_op(30)], OpRef::NONE.raw()),
        ];

        let token = JitCellToken::new(1500);
        backend.compile_loop(&inputargs, &ops, &token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(4), Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 9);
    }

    #[test]
    fn uint_mul_high_then_wrapping_mul_keeps_both_operands() {
        // `consider_finish` publishes only its first argument. The high
        // half is a second argument so `UINT_MUL_HIGH` stays live:
        // `has_no_side_effect` drops a result nobody reads, and then
        // only `INT_MUL` would run. Both inputs stay live across the
        // high multiply, which clobbers EAX and EDX.
        fn product(a: i64, b: i64) -> i64 {
            let mut backend = DynasmBackend::new();
            backend.attach_default_test_descrs();
            let ops = vec![
                mk_op(
                    OpCode::UintMulHigh,
                    &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
                    2,
                ),
                mk_op(
                    OpCode::IntMul,
                    &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
                    3,
                ),
                mk_op(
                    OpCode::Finish,
                    &[OpRef::int_op(3), OpRef::int_op(2)],
                    OpRef::NONE.raw(),
                ),
            ];
            let token = JitCellToken::new(1501);
            backend
                .compile_loop(
                    &[InputArg::new_int_rc(0), InputArg::new_int_rc(1)],
                    &ops,
                    &token,
                )
                .unwrap();
            let frame = backend.execute_token(&token, &[Value::Int(a), Value::Int(b)]);
            backend.get_int_value(&frame, 0)
        }
        for (a, b) in [(6i64, 7i64), (3125867703, 3442092617), (-1, 3)] {
            let got = product(a, b) as u64;
            let exp = (a as u64 as u128).wrapping_mul(b as u64 as u128) as u64;
            assert_eq!(got, exp, "{a:#x} * {b:#x}");
        }
    }

    #[test]
    fn test_gc_alloc_and_init_with_configured_runtime() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::new();
        gc.register_type(TypeInfo::simple(16));
        gc.register_type(TypeInfo::simple(24));

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let mut consts: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        consts.insert(10000, 32_i64);
        consts.insert(10001, -8_i64);
        consts.insert(10002, 1_i64);
        consts.insert(10003, 8_i64);
        consts.insert(10004, 0_i64);
        consts.insert(10005, 0xDEAD_i64);
        consts.insert(10006, 16_i64);
        consts.insert(10007, 1_i64);
        backend.set_constants(consts);
        register_jitframe_type(&mut gc);
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![];
        let ops = vec![
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(10000)], 0),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10001),
                    OpRef::int_op(10002),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10004),
                    OpRef::int_op(10005),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10006),
                    OpRef::int_op(10007),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Finish, &[OpRef::ref_op(0)], OpRef::NONE.raw()),
        ];

        let token = JitCellToken::new(1500);
        backend.compile_loop(&inputargs, &ops, &token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        let obj = backend.get_ref_value(&frame, 0);
        assert!(!obj.is_null());
        assert_eq!(unsafe { (*header_of(obj.0)).type_id() }, 1);
        assert_eq!(unsafe { *(obj.0 as *const u64) }, 0xDEAD);
        assert_eq!(unsafe { *((obj.0 + 16) as *const i64) }, 1);
    }

    #[test]
    fn test_gc_varsize_alloc_and_length_init_with_configured_runtime() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        let array_tid = gc.register_type(TypeInfo::varsize(8, 8, 0, false, Vec::new()));

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let mut consts: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        consts.insert(10000, 0_i64); // FLAG_ARRAY
        consts.insert(10001, 8_i64); // item size / length-store width
        consts.insert(10002, 0_i64); // length offset
        backend.set_constants(consts);
        register_jitframe_type(&mut gc);
        backend.set_gc_allocator(Box::new(gc));

        let alloc = mk_op(
            OpCode::CallMallocNurseryVarsize,
            &[
                OpRef::int_op(10000),
                OpRef::int_op(10001),
                OpRef::input_arg_int(0),
            ],
            1,
        );
        alloc.setdescr(Arc::new(SimpleArrayDescr::new(
            0,
            8,
            8,
            array_tid,
            Type::Int,
        )));
        let inputargs = vec![InputArg::new_int_rc(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_int(0)], OpRef::NONE.raw()),
            alloc,
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(1),
                    OpRef::int_op(10002),
                    OpRef::input_arg_int(0),
                    OpRef::int_op(10001),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Finish, &[OpRef::ref_op(1)], OpRef::NONE.raw()),
        ];

        let token = JitCellToken::new(1501);
        backend.compile_loop(&inputargs, &ops, &token).unwrap();
        let frame = backend.execute_token(&token, &[Value::Int(3)]);
        let obj = backend.get_ref_value(&frame, 0);

        assert!(!obj.is_null());
        assert_eq!(unsafe { (*header_of(obj.0)).type_id() }, array_tid);
        assert_eq!(unsafe { *(obj.0 as *const i64) }, 3);

        // A length beyond `max_size_of_young_obj` must take the collecting
        // helper edge, not overflow the inline size calculation or bump past
        // the nursery top.  `external_malloc` returns this object born-old.
        let large_length = 140_000;
        let frame = backend.execute_token(&token, &[Value::Int(large_length)]);
        let obj = backend.get_ref_value(&frame, 0);
        assert!(!obj.is_null());
        assert_eq!(unsafe { (*header_of(obj.0)).type_id() }, array_tid);
        assert_eq!(unsafe { *(obj.0 as *const i64) }, large_length);
        assert!(!majit_gc::can_move(obj));
    }

    #[test]
    fn test_collecting_alloc_preserves_initialized_header_and_payload() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 96,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        gc.register_type(TypeInfo::simple(24));

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let mut consts: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        consts.insert(10000, 32_i64);
        consts.insert(10001, -8_i64);
        consts.insert(10002, 1_i64);
        consts.insert(10003, 8_i64);
        consts.insert(10004, 0_i64);
        consts.insert(10005, 0xDEAD_i64);
        consts.insert(10006, 16_i64);
        consts.insert(10007, 1_i64);
        backend.set_constants(consts);
        register_jitframe_type(&mut gc);
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![];
        let ops = vec![
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(10000)], 0),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10001),
                    OpRef::int_op(10002),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10004),
                    OpRef::int_op(10005),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10006),
                    OpRef::int_op(10007),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(10000)], 1),
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(10000)], 2),
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(10000)], 3),
            mk_op(
                OpCode::Finish,
                &[
                    OpRef::ref_op(0),
                    OpRef::ref_op(1),
                    OpRef::ref_op(2),
                    OpRef::ref_op(3),
                ],
                OpRef::NONE.raw(),
            ),
        ];

        let token = JitCellToken::new(1501);
        backend.compile_loop(&inputargs, &ops, &token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        let obj = backend.get_ref_value(&frame, 0);
        assert!(!obj.is_null());
        assert_eq!(unsafe { (*header_of(obj.0)).type_id() }, 1);
        assert_eq!(unsafe { *(obj.0 as *const u64) }, 0xDEAD);
        assert_eq!(unsafe { *((obj.0 + 16) as *const i64) }, 1);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_varsize_frame_fastpath_does_not_overlap_previous_object_payload() {
        let mut gc = MiniMarkGC::new();
        gc.register_type(TypeInfo::simple(24));

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let mut consts: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        consts.insert(10000, 32_i64);
        consts.insert(10001, -8_i64);
        consts.insert(10002, 1_i64);
        consts.insert(10003, 8_i64);
        consts.insert(10004, 0_i64);
        consts.insert(10005, 16_i64);
        consts.insert(10006, 111_i64);
        consts.insert(10007, 3_i64);
        backend.set_constants(consts);
        register_jitframe_type(&mut gc);
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_int_rc(0)];
        let ops = vec![
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(10000)], 0),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10001),
                    OpRef::int_op(10002),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10004),
                    OpRef::int_op(10004),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10005),
                    OpRef::int_op(10006),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::CallMallocNurseryVarsizeFrame,
                &[OpRef::input_arg_int(0)],
                1,
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(1),
                    OpRef::int_op(10001),
                    OpRef::int_op(10007),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Finish, &[OpRef::ref_op(0)], OpRef::NONE.raw()),
        ];

        let token = JitCellToken::new(1502);
        backend.compile_loop(&inputargs, &ops, &token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(64)]);
        let obj = backend.get_ref_value(&frame, 0);
        assert!(!obj.is_null());
        assert_eq!(unsafe { *((obj.0 + 16) as *const i64) }, 111);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_varsize_frame_gcstore_round_trips_first_user_slot() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 16);

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let mut consts: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        consts.insert(10000, 264_i64);
        consts.insert(10001, 256_i64);
        consts.insert(10002, 8_i64);
        backend.set_constants(consts);
        register_jitframe_type(&mut gc);
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref_rc(0)];
        let ops = vec![
            mk_op(
                OpCode::CallMallocNurseryVarsizeFrame,
                &[OpRef::int_op(10000)],
                1,
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(1),
                    OpRef::int_op(10001),
                    OpRef::input_arg_ref(0),
                    OpRef::int_op(10002),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcLoadR,
                &[OpRef::ref_op(1), OpRef::int_op(10001), OpRef::int_op(10002)],
                2,
            ),
            mk_op(OpCode::Finish, &[OpRef::ref_op(2)], OpRef::NONE.raw()),
        ];

        let token = JitCellToken::new(1503);
        backend.compile_loop(&inputargs, &ops, &token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(payload)]);
        assert_eq!(backend.get_ref_value(&frame, 0), payload);
    }

    /// `genop_guard_guard_nonnull_class` funnels the null test into the one
    /// guard `Jcc`, and `patch_jump_for_descr` rewrites that `Jcc`'s target
    /// field: once a bridge is attached, a null object and an object of the
    /// wrong class both reach it.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_guard_nonnull_class_null_and_wrong_class_reach_the_bridge() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let wrong_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 16);
        let wrong = gc.alloc_with_type(wrong_tid, 16);

        let payload_vtable: usize = 0x4444_6600;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, payload_vtable, payload_tid);

        majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
        install_call_assembler_test_layout(jitframe_tid);
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref_rc(0)];
        let mut constants: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        constants.insert(100, payload_vtable as i64);
        backend.set_constants(constants);

        let token = JitCellToken::new(1605);
        let guard = mk_op(
            OpCode::GuardNonnullClass,
            &[OpRef::input_arg_ref(0), OpRef::int_op(100)],
            OpRef::NONE.raw(),
        );
        guard.setfailargs(vec![rb(OpRef::input_arg_ref(0))].into());
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_ref(0)], OpRef::NONE.raw()),
            guard,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(0)],
                OpRef::NONE.raw(),
            ),
        ];
        backend.compile_loop(&inputargs, &ops, &token).unwrap();

        let passed = backend.execute_token(&token, &[Value::Ref(payload)]);
        assert!(backend.get_latest_descr(&passed).is_finish());
        assert_eq!(backend.get_ref_value(&passed, 0), payload);

        let null_failed = backend.execute_token(&token, &[Value::Ref(GcRef::NULL)]);
        assert!(!backend.get_latest_descr(&null_failed).is_finish());
        let wrong_failed = backend.execute_token(&token, &[Value::Ref(wrong)]);
        assert!(!backend.get_latest_descr(&wrong_failed).is_finish());
        let guard_descr = backend.get_latest_descr_arc(&wrong_failed);
        assert_ne!(guard_descr.as_fail_descr().unwrap().adr_jump_offset(), 0);

        backend.set_constants(indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher));
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_ref(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(0)],
                OpRef::NONE.raw(),
            ),
        ];
        backend
            .compile_bridge(
                guard_descr.as_fail_descr().unwrap(),
                &inputargs,
                &bridge_ops,
                &token,
                None,
            )
            .unwrap();
        assert_eq!(guard_descr.as_fail_descr().unwrap().adr_jump_offset(), 0);

        for (arg, expected) in [
            (GcRef::NULL, GcRef::NULL),
            (wrong, wrong),
            (payload, payload),
        ] {
            let frame = backend.execute_token(&token, &[Value::Ref(arg)]);
            assert!(backend.get_latest_descr(&frame).is_finish());
            assert_eq!(backend.get_ref_value(&frame, 0), expected);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_bridge_materializes_register_ref_inputs_for_resolve_opref_ops() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let wrong_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 16);

        let wrong_vtable: usize = 0x2222_4400;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, wrong_vtable, wrong_tid);

        majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
        install_call_assembler_test_layout(jitframe_tid);
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref_rc(0)];
        let mut constants: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        constants.insert(100, wrong_vtable as i64);
        backend.set_constants(constants);

        let token = JitCellToken::new(1604);
        let guard = mk_op(
            OpCode::GuardClass,
            &[OpRef::input_arg_ref(0), OpRef::int_op(100)],
            OpRef::NONE.raw(),
        );
        guard.setfailargs(vec![rb(OpRef::input_arg_ref(0))].into());
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_ref(0)], OpRef::NONE.raw()),
            guard,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(0)],
                OpRef::NONE.raw(),
            ),
        ];
        backend.compile_loop(&inputargs, &ops, &token).unwrap();

        let failed = backend.execute_token(&token, &[Value::Ref(payload)]);
        let _guard_fail_index = backend.get_latest_descr(&failed).fail_index();
        let _guard_trace_id = backend.get_latest_descr(&failed).trace_id();
        let guard_descr = backend.get_latest_descr_arc(&failed);

        let mut bridge_constants: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        bridge_constants.insert(200, return_ref_passthrough as *const () as usize as i64);
        backend.set_constants(bridge_constants);
        let bridge_value = mk_op(
            OpCode::CondCallValueR,
            &[OpRef::input_arg_ref(0), OpRef::int_op(200)],
            1,
        );
        bridge_value.setdescr(make_plain_call_descr(vec![], Type::Ref));
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_ref(0)], OpRef::NONE.raw()),
            bridge_value,
            mk_op(OpCode::Finish, &[OpRef::ref_op(1)], OpRef::NONE.raw()),
        ];
        backend
            .compile_bridge(
                guard_descr.as_fail_descr().unwrap(),
                &inputargs,
                &bridge_ops,
                &token,
                None,
            )
            .unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(payload)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), payload);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_bridge_materializes_two_register_ref_inputs_before_unused_raw_load() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let frame_tid = gc.register_type(TypeInfo::simple(24));
        let other_tid = gc.register_type(TypeInfo::simple(16));
        let wrong_tid = gc.register_type(TypeInfo::simple(16));
        let frame_payload = gc.alloc_with_type(frame_tid, 24);
        let second_payload = gc.alloc_with_type(other_tid, 16);

        const MARKER: i64 = 0x1357_2468_1122_i64;
        unsafe {
            *((frame_payload.0 as *mut u8).add(16) as *mut i64) = MARKER;
        }

        let wrong_vtable: usize = 0x3333_5500;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, wrong_vtable, wrong_tid);

        majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
        install_call_assembler_test_layout(jitframe_tid);
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref_rc(0), InputArg::new_ref_rc(1)];
        let mut constants: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        constants.insert(100, wrong_vtable as i64);
        backend.set_constants(constants);

        let token = JitCellToken::new(1618);
        let guard = mk_op(
            OpCode::GuardClass,
            &[OpRef::input_arg_ref(1), OpRef::int_op(100)],
            OpRef::NONE.raw(),
        );
        guard.setfailargs(vec![rb(OpRef::input_arg_ref(0)), rb(OpRef::input_arg_ref(1))].into());
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
                OpRef::NONE.raw(),
            ),
            guard,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(1)],
                OpRef::NONE.raw(),
            ),
        ];
        backend.compile_loop(&inputargs, &ops, &token).unwrap();

        let failed = backend.execute_token(
            &token,
            &[Value::Ref(frame_payload), Value::Ref(second_payload)],
        );
        let _guard_fail_index = backend.get_latest_descr(&failed).fail_index();
        let _guard_trace_id = backend.get_latest_descr(&failed).trace_id();
        let guard_descr = backend.get_latest_descr_arc(&failed);

        backend.set_constants(indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher));
        let field_descr: DescrRef =
            Arc::new(majit_ir::SimpleFieldDescr::new(0, 16, 8, Type::Int, false));
        let getfield = mk_op(OpCode::GetfieldRawI, &[OpRef::input_arg_ref(0)], 2);
        getfield.setdescr(field_descr);
        let bridge_ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
                OpRef::NONE.raw(),
            ),
            getfield,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(1)],
                OpRef::NONE.raw(),
            ),
        ];
        backend
            .compile_bridge(
                guard_descr.as_fail_descr().unwrap(),
                &inputargs,
                &bridge_ops,
                &token,
                None,
            )
            .unwrap();

        let frame = backend.execute_token(
            &token,
            &[Value::Ref(frame_payload), Value::Ref(second_payload)],
        );
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), second_payload);
    }

    extern "C" fn return_int_passthrough(arg: i64) -> i64 {
        arg
    }

    static COND_CALL_VALUE_I_CALLS: AtomicI64 = AtomicI64::new(0);
    static COND_CALL_VALUE_R_CALLS: AtomicI64 = AtomicI64::new(0);

    extern "C" fn cond_call_value_i_helper(arg: i64) -> i64 {
        COND_CALL_VALUE_I_CALLS.fetch_add(1, Ordering::SeqCst);
        arg.wrapping_mul(10)
    }

    extern "C" fn cond_call_value_r_helper(arg: i64) -> i64 {
        COND_CALL_VALUE_R_CALLS.fetch_add(1, Ordering::SeqCst);
        let _ = arg;
        0x5678
    }

    /// A `COND_CALL_VALUE_I` argument that is an op result — not an inputarg —
    /// and is still live after the call.
    ///
    /// `prepare_op_cond_call_value_i` (`_prepare_op_cond_call`) places extra
    /// args in `argument_regs`. Naming `i2` after the call keeps it live so
    /// a wrong extra-arg location surfaces in the helper result. `i2` is
    /// deliberately NOT equal to `i1`.
    #[test]
    fn test_cond_call_value_passes_a_register_resident_op_result_argument() {
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();

        // arg 0 is the cond-call predicate, passed 0 so the call is taken.
        let inputargs = vec![InputArg::new_int_rc(0), InputArg::new_int_rc(1)];
        let mut constants: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        constants.insert(200, return_int_passthrough as *const () as usize as i64);
        backend.set_constants(constants);

        let cond_call = mk_op(
            OpCode::CondCallValueI,
            &[
                OpRef::input_arg_int(0),
                OpRef::int_op(200),
                OpRef::int_op(2),
            ],
            3,
        );
        cond_call.setdescr(make_plain_call_descr(vec![Type::Int], Type::Int));

        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
                OpRef::NONE.raw(),
            ),
            // i2 = i1 + i1, an op result distinct from every inputarg value.
            mk_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(1), OpRef::input_arg_int(1)],
                2,
            ),
            cond_call,
            // Naming i2 again keeps it live across the cond-call.
            mk_op(
                OpCode::Finish,
                &[OpRef::int_op(3), OpRef::int_op(2)],
                OpRef::NONE.raw(),
            ),
        ];

        let token = JitCellToken::new(1617);
        backend.compile_loop(&inputargs, &ops, &token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(7)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 14);
    }

    /// HIT: non-null IntAdd result (kept live in Finish) is the op result and
    /// the helper is not called. MISS: null value, helper called once with the
    /// extra arg, result is the helper word. Sequential so the call counter
    /// is not shared with a parallel test.
    #[test]
    fn test_cond_call_value_i_hit_and_miss() {
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();

        let inputargs = vec![InputArg::new_int_rc(0), InputArg::new_int_rc(1)];
        let mut constants: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        constants.insert(200, cond_call_value_i_helper as *const () as usize as i64);
        backend.set_constants(constants.clone());

        let hit = mk_op(
            OpCode::CondCallValueI,
            &[
                OpRef::int_op(2),
                OpRef::int_op(200),
                OpRef::input_arg_int(1),
            ],
            3,
        );
        hit.setdescr(make_plain_call_descr(vec![Type::Int], Type::Int));
        let hit_ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(0)],
                2,
            ),
            hit,
            mk_op(
                OpCode::Finish,
                &[OpRef::int_op(3), OpRef::int_op(2)],
                OpRef::NONE.raw(),
            ),
        ];
        let hit_token = JitCellToken::new(1620);
        backend
            .compile_loop(&inputargs, &hit_ops, &hit_token)
            .unwrap();
        COND_CALL_VALUE_I_CALLS.store(0, Ordering::SeqCst);
        let frame = backend.execute_token(&hit_token, &[Value::Int(11), Value::Int(5)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 22);
        assert_eq!(COND_CALL_VALUE_I_CALLS.load(Ordering::SeqCst), 0);

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        backend.set_constants(constants.clone());
        let miss = mk_op(
            OpCode::CondCallValueI,
            &[
                OpRef::input_arg_int(0),
                OpRef::int_op(200),
                OpRef::int_op(2),
            ],
            3,
        );
        miss.setdescr(make_plain_call_descr(vec![Type::Int], Type::Int));
        let miss_ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(1), OpRef::input_arg_int(1)],
                2,
            ),
            miss,
            mk_op(
                OpCode::Finish,
                &[OpRef::int_op(3), OpRef::int_op(2)],
                OpRef::NONE.raw(),
            ),
        ];
        let miss_token = JitCellToken::new(1621);
        backend
            .compile_loop(&inputargs, &miss_ops, &miss_token)
            .unwrap();
        COND_CALL_VALUE_I_CALLS.store(0, Ordering::SeqCst);
        let frame = backend.execute_token(&miss_token, &[Value::Int(0), Value::Int(7)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 140);
        assert_eq!(COND_CALL_VALUE_I_CALLS.load(Ordering::SeqCst), 1);
    }

    /// HIT: non-null ref value is the result and the helper is not called.
    /// MISS: null ref, helper called once, result is the helper word.
    #[test]
    fn test_cond_call_value_r_hit_and_miss() {
        let inputargs = vec![InputArg::new_ref_rc(0), InputArg::new_ref_rc(1)];
        let mut constants: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        constants.insert(200, cond_call_value_r_helper as *const () as usize as i64);

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        backend.set_constants(constants.clone());
        let hit = mk_op(
            OpCode::CondCallValueR,
            &[
                OpRef::input_arg_ref(0),
                OpRef::int_op(200),
                OpRef::input_arg_ref(1),
            ],
            2,
        );
        hit.setdescr(make_plain_call_descr(vec![Type::Ref], Type::Ref));
        let hit_ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
                OpRef::NONE.raw(),
            ),
            hit,
            mk_op(
                OpCode::Finish,
                &[OpRef::ref_op(2), OpRef::input_arg_ref(0)],
                OpRef::NONE.raw(),
            ),
        ];
        let hit_token = JitCellToken::new(1622);
        backend
            .compile_loop(&inputargs, &hit_ops, &hit_token)
            .unwrap();
        COND_CALL_VALUE_R_CALLS.store(0, Ordering::SeqCst);
        let value = GcRef(0x1234);
        let extra = GcRef(0x9);
        let frame = backend.execute_token(&hit_token, &[Value::Ref(value), Value::Ref(extra)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), value);
        assert_eq!(COND_CALL_VALUE_R_CALLS.load(Ordering::SeqCst), 0);

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        backend.set_constants(constants);
        let miss = mk_op(
            OpCode::CondCallValueR,
            &[
                OpRef::input_arg_ref(0),
                OpRef::int_op(200),
                OpRef::input_arg_ref(1),
            ],
            2,
        );
        miss.setdescr(make_plain_call_descr(vec![Type::Ref], Type::Ref));
        let miss_ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
                OpRef::NONE.raw(),
            ),
            miss,
            mk_op(OpCode::Finish, &[OpRef::ref_op(2)], OpRef::NONE.raw()),
        ];
        let miss_token = JitCellToken::new(1623);
        backend
            .compile_loop(&inputargs, &miss_ops, &miss_token)
            .unwrap();
        COND_CALL_VALUE_R_CALLS.store(0, Ordering::SeqCst);
        let frame = backend.execute_token(&miss_token, &[Value::Ref(GcRef(0)), Value::Ref(extra)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), GcRef(0x5678));
        assert_eq!(COND_CALL_VALUE_R_CALLS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_label_uses_absolute_jitframe_input_slots_for_resolve_opref_ops() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 16);

        majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
        install_call_assembler_test_layout(jitframe_tid);
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref_rc(0)];
        let mut constants: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        constants.insert(200, return_ref_passthrough as *const () as usize as i64);
        backend.set_constants(constants);

        let token = JitCellToken::new(1605);
        let passthrough = mk_op(
            OpCode::CondCallValueR,
            &[OpRef::input_arg_ref(0), OpRef::int_op(200)],
            1,
        );
        passthrough.setdescr(make_plain_call_descr(vec![], Type::Ref));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_ref(0)], OpRef::NONE.raw()),
            passthrough,
            mk_op(OpCode::Finish, &[OpRef::ref_op(1)], OpRef::NONE.raw()),
        ];
        backend.compile_loop(&inputargs, &ops, &token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(payload)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), payload);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_plain_call_returns_fresh_gc_ref_without_call_assembler() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 96,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));

        majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
        install_call_assembler_test_layout(jitframe_tid);
        TEST_HELPER_ALLOC_TYPE_ID.store(payload_tid, Ordering::Relaxed);

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let mut consts: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        consts.insert(205, alloc_marked_ref as *const () as usize as i64);
        backend.set_constants(consts);
        backend.set_gc_allocator(Box::new(gc));

        let plain_call = mk_op(OpCode::CallR, &[OpRef::int_op(205)], 0);
        plain_call.setdescr(make_plain_call_descr(vec![], Type::Ref));
        let ops = vec![
            plain_call,
            mk_op(OpCode::Finish, &[OpRef::ref_op(0)], OpRef::NONE.raw()),
        ];
        let token = JitCellToken::new(1612);
        backend.compile_loop(&[], &ops, &token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        let result = backend.get_ref_value(&frame, 0);
        assert!(!result.is_null());
        unsafe {
            assert_eq!(*(result.0 as *const i64), TEST_HELPER_MARKER);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_plain_call_preserves_ref_result_across_collecting_helper_call() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 96,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));

        majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
        install_call_assembler_test_layout(jitframe_tid);
        TEST_HELPER_ALLOC_TYPE_ID.store(payload_tid, Ordering::Relaxed);

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let mut consts: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        consts.insert(
            206,
            alloc_marked_ref_collecting as *const () as usize as i64,
        );
        backend.set_constants(consts);
        backend.set_gc_allocator(Box::new(gc));

        let plain_call = mk_op(OpCode::CallR, &[OpRef::int_op(206)], 0);
        plain_call.setdescr(make_plain_call_descr(vec![], Type::Ref));
        let ops = vec![
            plain_call,
            mk_op(OpCode::Finish, &[OpRef::ref_op(0)], OpRef::NONE.raw()),
        ];
        let token = JitCellToken::new(1613);
        backend.compile_loop(&[], &ops, &token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        let result = backend.get_ref_value(&frame, 0);
        assert!(!result.is_null());
        unsafe {
            assert_eq!(*(result.0 as *const i64), TEST_HELPER_MARKER);
        }
    }

    /// Raw `DONE_REF` after a residual that forces a minor collection.
    /// The entry jitframe is a nursery object (`gen_shadowstack_header`);
    /// `_call_footer` pops that slot then returns the fp register, which
    /// may still name the corpse. Slot 0 is the moved input ref.
    #[test]
    fn test_raw_done_ref_returns_moved_object_after_minor_collection() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let root = gc.alloc_with_type(payload_tid, 16);
        assert!(gc.is_in_nursery(root.0));
        unsafe {
            *(root.0 as *mut u64) = 0xABCDEF01;
        }

        let jitframe_tid = register_jitframe_type(&mut gc);
        install_call_assembler_test_layout(jitframe_tid);

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let mut consts: indexmap::IndexMap<u32, i64, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        consts.insert(
            210,
            collect_nursery_via_dynasm_gc as *const () as usize as i64,
        );
        backend.set_constants(consts);
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref_rc(0)];
        let collect = mk_op(OpCode::CallN, &[OpRef::int_op(210)], OpRef::NONE.raw());
        collect.setdescr(make_plain_call_descr(vec![], Type::Void));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_ref(0)], OpRef::NONE.raw()),
            collect,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(0)],
                OpRef::NONE.raw(),
            ),
        ];
        let token = JitCellToken::new(1618);
        backend.compile_loop(&inputargs, &ops, &token).unwrap();

        let raw = match backend.execute_token_done_ref_raw(&token, &[root.0 as i64]) {
            Ok(value) => value,
            Err(_) => panic!("raw done-ref after collecting residual"),
        };
        assert_ne!(raw, root.0, "minor collection must move the nursery ref");
        assert_eq!(unsafe { *(raw as *const u64) }, 0xABCDEF01);
    }

    #[test]
    fn bh_new_with_vtable_writes_type_and_class_words() {
        use majit_backend::Backend;
        const VTABLE: usize = 0x1111_0000;
        const W_CLASS: i64 = 0x2222_0000;
        majit_ir::descr::set_w_class_obj_resolver(|vtable| (vtable == VTABLE).then_some(W_CLASS));
        let mut backend = DynasmBackend::new();
        backend.set_vtable_offset(Some(0));
        backend.set_w_class_offset(Some(std::mem::size_of::<usize>()));
        let descr = majit_jitcode::jitcode::BhDescr::Size {
            size: 32,
            type_id: 0,
            vtable: VTABLE as u64,
            owner: String::new(),
            all_fielddescrs: Vec::new(),
            is_gc_managed: false,
        };
        let ptr = backend.bh_new_with_vtable(&descr);
        assert_ne!(ptr, 0);
        unsafe {
            let words = ptr as *const usize;
            assert_eq!(*words, VTABLE);
            assert_eq!(*words.add(1), W_CLASS as usize);
        }
    }

    #[test]
    fn test_jit_threadlocalref_base_round_trips_slot_contents() {
        crate::jit_threadlocalref_set(0, 0x1234);
        crate::jit_threadlocalref_set(8, 0x5678);
        let base = crate::jit_threadlocalref_base();
        assert!(!base.is_null());
        unsafe {
            assert_eq!(*base.add(0), 0x1234);
            assert_eq!(*base.add(1), 0x5678);
        }
    }
}

// ── rewrite.py:489 parity: inject str_descr/unicode_descr ──
//
// Token semantics come from `symbolic.get_array_token(rstr.STR/UNICODE, ...)`
// and `symbolic.get_field_token(rstr.STR/UNICODE, 'hash', ...)` (see
// `rpython/jit/backend/llsupport/symbolic.py`). The layout encoded by
// `rstr.STR.become(GcStruct('rpy_string', ('hash', Signed), ('chars',
// Array(Char, hints={'extra_item_after_alloc': 1}))))`
// (`rpython/rtyper/lltypesystem/rstr.py`) is:
//
//   [ hash (WORD) | chars.length (WORD) | chars[0..n] | +1 extra null ]
//
// `get_array_token` returns `basesize = before_array_part +
// carray.items.offset + extra_item_after_alloc`, so for STR the token
// `basesize` is 17 (not 16) — rewrite.py:295 then subtracts 1 for the
// extra null character when emitting STR{GET,SET}ITEM. UNICODE has no
// `extra_item_after_alloc` hint, so its token `basesize` is 16.
//
// Hash lives in its own field (the `hash` struct member), separate from
// the array tail. rewrite.py:283-294 reads it with
// `get_field_token(..., 'hash', ...)`, not `get_array_token(...)`.

/// `symbolic.get_field_token(rstr.STR/UNICODE, 'hash', ...).offset`.
const BUILTIN_STRING_HASH_OFFSET: usize = majit_backend::BUILTIN_STRING_HASH_OFFSET;
/// `symbolic.get_field_token(..., 'hash', ...).size` — assert == WORD at
/// rewrite.py:286,292.
const BUILTIN_STRING_HASH_SIZE: usize = std::mem::size_of::<usize>();
/// `symbolic.get_array_token(rstr.STR/UNICODE, ...).ofs_length` =
/// `before_array_part + carray.length.offset`.
const BUILTIN_STRING_LEN_OFFSET: usize = majit_backend::BUILTIN_STRING_LEN_OFFSET;
/// STR token `basesize` — `before_array_part(8) + carray.items.offset(8) +
/// extra_item_after_alloc(1) = 17`.
const BUILTIN_STR_TOKEN_BASE_SIZE: usize = majit_backend::BUILTIN_STR_TOKEN_BASE_SIZE;
/// UNICODE token `basesize` — `before_array_part(8) + carray.items.offset(8)
/// = 16` (no extra_item_after_alloc).
const BUILTIN_UNICODE_TOKEN_BASE_SIZE: usize = 2 * std::mem::size_of::<usize>();

#[derive(Debug)]
struct BuiltinFieldDescr {
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
}

impl majit_ir::Descr for BuiltinFieldDescr {
    fn as_field_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
        Some(self)
    }
}

impl majit_ir::FieldDescr for BuiltinFieldDescr {
    fn offset(&self) -> usize {
        self.offset
    }
    fn field_size(&self) -> usize {
        self.field_size
    }
    fn field_type(&self) -> Type {
        self.field_type
    }
    fn is_field_signed(&self) -> bool {
        self.signed
    }
}

#[derive(Debug)]
struct BuiltinArrayDescr {
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
    len_descr: Arc<BuiltinFieldDescr>,
}

impl majit_ir::Descr for BuiltinArrayDescr {
    fn as_array_descr(&self) -> Option<&dyn majit_ir::ArrayDescr> {
        Some(self)
    }
}

impl majit_ir::ArrayDescr for BuiltinArrayDescr {
    fn base_size(&self) -> usize {
        self.base_size
    }
    fn item_size(&self) -> usize {
        self.item_size
    }
    fn type_id(&self) -> u32 {
        self.type_id
    }
    fn item_type(&self) -> Type {
        self.item_type
    }
    fn is_item_signed(&self) -> bool {
        self.signed
    }
    fn len_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
        Some(self.len_descr.as_ref())
    }
}

/// `symbolic.get_array_token(rstr.STR/UNICODE, ...)` token triple wrapped
/// as an `ArrayDescr`.  Fed to NEW{STR,UNICODE} / STR{LEN,GETITEM,SETITEM}
/// / UNICODE{LEN,GETITEM,SETITEM} / COPY{STR,UNICODE}CONTENT — every op
/// that upstream dispatches through `get_array_token` at
/// `rewrite.py:273-318`.  STR{,UNICODE}HASH takes a separate FieldDescr
/// (see `builtin_string_hash_field_descr` below).
fn builtin_string_array_descr(opcode: majit_ir::OpCode) -> Option<majit_ir::DescrRef> {
    use majit_ir::OpCode;
    let (base_size, item_size, type_id) = match opcode {
        OpCode::Newstr
        | OpCode::Strlen
        | OpCode::Strgetitem
        | OpCode::Strsetitem
        | OpCode::Copystrcontent => (
            BUILTIN_STR_TOKEN_BASE_SIZE,
            1,
            majit_gc::lowlevel_str_type_id(),
        ),
        OpCode::Newunicode
        | OpCode::Unicodelen
        | OpCode::Unicodegetitem
        | OpCode::Unicodesetitem
        | OpCode::Copyunicodecontent => (
            BUILTIN_UNICODE_TOKEN_BASE_SIZE,
            4,
            majit_gc::lowlevel_unicode_type_id(),
        ),
        _ => return None,
    };
    let len_descr = Arc::new(BuiltinFieldDescr {
        offset: BUILTIN_STRING_LEN_OFFSET,
        field_size: BUILTIN_STRING_HASH_SIZE,
        field_type: Type::Int,
        signed: false,
    });
    Some(Arc::new(BuiltinArrayDescr {
        base_size,
        item_size,
        type_id,
        item_type: Type::Int,
        signed: false,
        len_descr,
    }))
}

/// `symbolic.get_field_token(rstr.STR/UNICODE, 'hash', ...)` wrapped as a
/// FieldDescr.  rewrite.py:283-294 reads STRHASH/UNICODEHASH via
/// `get_field_token`, not `get_array_token`.  Kept separate so the two
/// upstream token helpers have independent pyre counterparts.
fn builtin_string_hash_field_descr(opcode: majit_ir::OpCode) -> Option<majit_ir::DescrRef> {
    use majit_ir::OpCode;
    if !matches!(opcode, OpCode::Strhash | OpCode::Unicodehash) {
        return None;
    }
    Some(Arc::new(BuiltinFieldDescr {
        offset: BUILTIN_STRING_HASH_OFFSET,
        field_size: BUILTIN_STRING_HASH_SIZE,
        field_type: Type::Int,
        // rewrite.py:288,293 pass `sign=True` for STR/UNICODE hash — the
        // `hash` struct field is `Signed`.
        signed: true,
    }))
}

fn inject_builtin_string_descrs(ops: &[OpRc]) {
    for op in ops {
        if op.has_descr() {
            continue;
        }
        if let Some(descr) = builtin_string_array_descr(op.opcode) {
            op.setdescr(descr);
        } else if let Some(descr) = builtin_string_hash_field_descr(op.opcode) {
            op.setdescr(descr);
        }
    }
}
