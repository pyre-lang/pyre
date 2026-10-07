//! The collector the matcher runs on.
//!
//! **A JIT-enabled RPython build cannot be translated without a GC**, so
//! neither should this crate run without one. `--gc=none` and `--gc=ref` are
//! real translation options (`translationoption.py:65-82`), but they select a
//! `gctransformer` of `"none"`/`"ref"`, and `gc.py get_ll_description`
//! resolves `GcLLDescr_<gctransformer>` against a module defining only
//! `GcLLDescr_boehm` (`gc.py`) and `GcLLDescr_framework` (`gc.py`);
//! anything else raises `NotImplementedError("GC transformer %r not supported
//! by the JIT backend")`. Both descrs that do exist inherit `malloc_jitframe`
//! from the base `GcLLDescription` (`gc.py`), where it is
//! `jitframe.JITFRAME.allocate` and so `lltype.malloc(JITFRAME,
//! frame_info.jfi_frame_depth)` (`jitframe.py`) — a GC allocation, with no
//! arm anywhere that puts a JITFRAME somewhere else. The matcher's own `Regex`
//! nodes are GC objects too. Running majit's side of the comparison with no
//! collector at all made the JIT take a fallback path that no translatable
//! configuration takes: `runner::alloc_jitframe` has nothing to allocate a
//! JITFRAME under, so every compiled entry gets one from the process allocator
//! and frees it on the way out, which on the `and`/`or` row is once per input
//! character.
//!
//! The sequence below is `pyre-jit`'s `init_gc_subsystem` with everything
//! pyre-specific removed — the same four steps
//! `majit-metainterp/tests/allocs_per_compiled_entry.rs` calls "the three calls
//! pyre reaches through `init_gc_subsystem`":
//!
//! 1. store the singleton,
//! 2. register this thread as a mutator (which takes the GIL and holds it),
//! 3. register the JITFRAME shape and publish its type id to the descr —
//!    before the install, because installing freezes the type table,
//! 4. install the backend's `majit_gc::set_active_*` hooks.

use std::sync::OnceLock;

/// Done once per process. `store_singleton` is itself idempotent, but the type
/// registration and the id publication are not, and `register_thread` asserts
/// against being called twice on one thread.
static INSTALLED: OnceLock<()> = OnceLock::new();

/// Install the collector, once.
///
/// Registers the CALLING thread as the mutator, so it must be called from the
/// thread that will run the matcher — which is `main`, the analogue of the
/// translated binary this crate is compared against.
pub fn install() {
    INSTALLED.get_or_init(|| {
        majit_gc::gc_sync::store_singleton(Box::new(majit_gc::collector::MiniMarkGC::new()));
        majit_gc::shadow_stack::register_mutator();
        majit_gc::gc_sync::register_thread();
        // `jitframe.py` `rgc.register_custom_trace_hook(JITFRAME, ...)` and
        // the descr-owned id every backend allocates its frames under
        // (`pyjitpl.rs register_active_backend_jitframe_gc_type`, which is
        // `eval.rs`'s own registration).
        majit_gc::gc_sync::gc_op(|gc| {
            majit_metainterp::register_active_backend_jitframe_gc_type(gc);
        });
        majit_metainterp::install_active_backend_gc_standalone();
    });
}
