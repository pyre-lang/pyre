// The remaining lint hits are spread across the generated-JIT integration
// layer (`call_jit`, `eval`, and `jit/*`) and come from in-progress RPython
// parity scaffolding plus macro-expanded control flow. Keep this crate-level
// suppression until those ports are collapsed into generated modules or deleted.
// TODO(pyre-jit-cleanup): move these to generated modules/items as each
// scaffolding block graduates.
#![allow(
    clippy::arc_with_non_send_sync,
    clippy::bind_instead_of_map,
    clippy::clone_on_copy,
    clippy::collapsible_if,
    clippy::collapsible_match,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_auto_deref,
    clippy::for_kv_map,
    clippy::int_plus_one,
    clippy::len_zero,
    clippy::let_and_return,
    clippy::manual_next_back,
    clippy::map_entry,
    clippy::mutable_key_type,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::new_without_default,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::ptr_eq,
    clippy::question_mark,
    clippy::redundant_closure_call,
    clippy::result_large_err,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unnecessary_get_then_check,
    clippy::unnecessary_map_or,
    clippy::unneeded_struct_pattern,
    clippy::unwrap_or_default,
    clippy::useless_conversion,
    unsafe_op_in_unsafe_fn,
    unused_assignments,
    unused_doc_comments,
    unused_imports,
    unused_macros,
    unused_mut
)]

//! pyre-jit: Auto-generated JIT for pyre.
//!
//! This crate is the Rust equivalent of RPython's `rpython/jit/` —
//! it contains automatically generated tracing code produced by
//! `majit-translate` during build time.
//!
//! The interpreter (`pyre-interpreter`) has zero JIT dependencies.
//! All JIT logic lives here, generated from static analysis of
//! the interpreter's source code.
//!
//! Architecture:
//! ```text
//! pyre-interpreter (pure interpreter)
//!       ↓ (source analysis at build time)
//! majit-translate (RPython translator)
//!       ↓ (code generation)
//! pyre-jit (this crate)
//!   ├── jit_trace_gen.rs (auto-generated trace functions)
//!   ├── driver.rs (JIT driver integration)
//!   └── lib.rs (public API)
//! ```

// `wasm-web` / `wasm-host` select the wasm JIT backend's host binding and are
// meaningful only on wasm32; reject misconfigurations early with a clear
// message instead of letting them surface as undefined-symbol failures.
#[cfg(all(feature = "wasm-web", feature = "wasm-host"))]
compile_error!("features `wasm-web` and `wasm-host` are mutually exclusive; enable exactly one");
#[cfg(all(
    any(feature = "wasm-web", feature = "wasm-host"),
    not(target_arch = "wasm32")
))]
compile_error!("features `wasm-web` / `wasm-host` require target_arch = \"wasm32\"");

pub mod call_jit;
pub mod driver;
pub mod eval;
pub mod jit;
mod trace_verify;

// Re-export auto-generated trace functions from pyre-jit-trace
pub use pyre_jit_trace::jitcode_runtime::{
    descr_demand_summary, descr_set_counts, descr_set_jit_stats, descr_spelling_gate_recheck_now,
    field_descr_identity_census_now, field_position_counts, field_position_jit_stats,
    field_position_unresolved_report,
};
pub use pyre_jit_trace::{
    trace_box_float, trace_box_int, trace_float_binop, trace_float_compare, trace_int_binop,
    trace_int_binop_ovf, trace_int_compare, trace_unbox_float, trace_unbox_int,
};

pub fn fbw_diag_counter(i: usize) -> u64 {
    pyre_jit_trace::trace::fbw_diag::get(i)
}

pub use pyre_jit_trace::state::PyreJitState;
/// The `[jit-stats]` key of each `fbw_diag` tally, in index order — declared
/// beside the counters themselves, so a slot cannot be added there and go
/// unprinted here. Joined against [`fbw_diag_counter`] by index.
pub use pyre_jit_trace::trace::fbw_diag::LABELS as FBW_DIAG_LABELS;

pub fn spec_census_summary() -> String {
    pyre_jit_trace::jitcode_dispatch::spec_census_summary()
}

pub fn fbw_depth_census_summary() -> String {
    pyre_jit_trace::jitcode_dispatch::fbw_depth_census_summary()
}

/// The `PYPY_GC_*` variables the collector sizes itself from, and the setter
/// for an embedder whose platform hands the process no environment to read them
/// out of. Re-exported so such a host reaches them through the crate it already
/// builds the interpreter with.
pub use majit_gc::collector::{GC_ENV_NAMES, set_supplied_env as set_gc_supplied_env};

/// The `PYRE_NO_JIT` / `MAJIT_NO_BRIDGE` variables the JIT knobs resolve
/// against, and the setter for an embedder whose platform hands the process
/// no environment to read them out of. Re-exported so such a host reaches
/// them through the crate it already builds the interpreter with.
pub use majit_metainterp::jit_env::{JIT_ENV_NAMES, set_supplied_env as set_jit_supplied_env};

/// Diagnostic only: `(oldgen_total_bytes, nursery_used_bytes)` of the wasm
/// backend's GC on this thread. Lets the wasm runner attribute guest
/// linear-memory growth to GC-retained objects vs. host-heap allocations.
#[cfg(target_arch = "wasm32")]
pub fn wasm_gc_heap_stats() -> (usize, usize) {
    majit_backend_wasm::active_gc_heap_stats()
}

/// Diagnostic only: `(minor_collections, major_collections)` of the wasm
/// backend's GC on this thread. Companion to [`wasm_gc_heap_stats`].
#[cfg(target_arch = "wasm32")]
pub fn wasm_gc_collection_counts() -> (usize, usize) {
    majit_backend_wasm::active_gc_collection_counts()
}

/// `(minor_collections, major_collections, heap_bytes, nursery_bytes)` for the
/// process GC, or all zeroes before one exists.
///
/// These are the GC-side counterpart of the `[jit-stats]` tallies, and they
/// exist for the same reason: they are DETERMINISTIC. The same program run
/// twice collects the same number of times, so a change that allocates less
/// shows up here even on a machine whose clock cannot resolve it — a minor
/// collection happens per nursery-full, so `minor_collections` is a direct
/// proxy for cumulative allocation, which `heap_bytes` (a current-usage
/// reading) is not.
pub fn gc_diag_counters() -> (usize, usize, usize, usize) {
    use majit_gc::GcAllocator;
    if !majit_gc::gc_sync::is_initialized() {
        return (0, 0, 0, 0);
    }
    majit_gc::gc_sync::gc_query_reentrant(|gc| {
        let (minor, major) = gc.collection_counts();
        let (heap, nursery) = gc.heap_byte_stats();
        (minor, major, heap, nursery)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `rpython/rlib/rstack.py` `stack_check_slowpath` raises `_StackOverflow`
    /// into `pos_exception` / `pos_exc_value` (`llmodel.py` `_store_exception`).
    /// `pyre_stack_check_slowpath_for_backend` publishes through the raiser
    /// `install_jit_call_bridge` registers (`register_jit_exc_raiser` →
    /// `jit_exc_raise`). The pending slot stays empty: `park_jit_pending_error`
    /// is its only producer.
    #[test]
    #[cfg(any(target_arch = "wasm32", feature = "dynasm", feature = "cranelift"))]
    fn stack_overflow_probe_raises_into_backend_exception_cells() {
        use pyre_interpreter::stack_check;
        use pyre_object::interp_exceptions::{ExcKind, w_exception_kind_checked};

        // The slowpath is reached from generated code, which runs with the GIL
        // held; driving it directly makes this thread the mutator instead.
        pyre_interpreter::module::thread::ensure_runtime_thread();

        // Install the same backend bridge call_jit::install_jit_call_bridge
        // wires up at runtime, so the registered slowpath addresses
        // really are the ones the cranelift / dynasm prologues invoke,
        // and `jit_publish_exception` has a raiser.
        crate::call_jit::install_jit_call_bridge();
        clear_backend_exception_cells();

        stack_check::reset_stack_base();
        assert!(
            stack_check::drain_jit_pending_exception().is_ok(),
            "pending slot starts empty"
        );

        let sp = {
            let probe: usize = 0;
            &probe as *const usize as usize
        };
        let above = sp.saturating_add(2 * stack_check::MAX_STACK_SIZE);
        // stack.c:40 baseptr is read from the TLS mirror, not the
        // global cache; plant both so the slowpath sees a base far
        // above the current SP and trips the real-overflow branch.
        stack_check::plant_stack_end_for_tests(above);
        assert_eq!(
            stack_check::pyre_stack_check_slowpath_for_backend(sp),
            1,
            "backend slowpath must signal overflow"
        );
        assert!(
            !stack_check::is_jit_overflow_pending(),
            "slowpath must not park the overflow in the pending slot",
        );
        assert!(
            stack_check::drain_jit_pending_exception().is_ok(),
            "pending slot stays empty"
        );
        let value = backend_exception_value();
        assert_ne!(value, 0, "raiser must write pos_exc_value");
        let kind = unsafe { w_exception_kind_checked(value as pyre_object::PyObjectRef) };
        assert_eq!(kind, Some(ExcKind::RecursionError));
        clear_backend_exception_cells();

        // Restore defaults so the rest of the test binary sees a sane
        // budget. The synthetic high base above would make
        // set_recursion_limit's internal stack_check trip, so reset
        // the base first.
        stack_check::reset_stack_base();
        stack_check::set_recursion_limit(stack_check::DEFAULT_RECURSION_LIMIT)
            .expect("default limit");
    }

    /// `pos_exc_value` the registered raiser wrote. Each backend keeps its
    /// own cell; `store_jit_exception` writes every backend that this build
    /// compiled in.
    #[cfg(any(target_arch = "wasm32", feature = "dynasm", feature = "cranelift"))]
    fn backend_exception_value() -> i64 {
        #[cfg(target_arch = "wasm32")]
        {
            return majit_backend_wasm::jit_exc_value_peek();
        }
        #[cfg(feature = "dynasm")]
        {
            majit_backend_dynasm::jit_exc_value_peek()
        }
        #[cfg(all(feature = "cranelift", not(feature = "dynasm")))]
        {
            majit_backend_cranelift::jit_exc_value_peek()
        }
    }

    #[cfg(any(target_arch = "wasm32", feature = "dynasm", feature = "cranelift"))]
    fn clear_backend_exception_cells() {
        #[cfg(target_arch = "wasm32")]
        majit_backend_wasm::jit_exc_clear();
        #[cfg(feature = "dynasm")]
        majit_backend_dynasm::jit_exc_clear();
        #[cfg(feature = "cranelift")]
        majit_backend_cranelift::jit_exc_clear();
    }
}
