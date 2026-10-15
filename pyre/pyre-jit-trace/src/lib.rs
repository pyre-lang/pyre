#![allow(
    // Trace-time code is generated from, or structurally paired with, the
    // RPython/PyPy interpreter pipeline. JitCode ownership must stay shared
    // across registries, and explicit frame/descriptor arguments, indexed
    // register walks, raw-handle wrappers, and generated control flow retain
    // the translator's source shape.
    clippy::approx_constant,
    clippy::arc_with_non_send_sync,
    clippy::cloned_ref_to_slice_refs,
    clippy::collapsible_if,
    clippy::declare_interior_mutable_const,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::empty_line_after_doc_comments,
    clippy::eq_op,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::filter_map_bool_then,
    clippy::if_same_then_else,
    clippy::int_plus_one,
    clippy::let_and_return,
    clippy::manual_contains,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::manual_map,
    clippy::manual_memcpy,
    clippy::manual_option_zip,
    clippy::manual_pattern_char_comparison,
    clippy::manual_range_contains,
    clippy::manual_repeat_n,
    clippy::map_identity,
    clippy::missing_const_for_thread_local,
    clippy::missing_safety_doc,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::nonminimal_bool,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::ptr_eq,
    clippy::question_mark,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unnecessary_lazy_evaluations,
    clippy::unnecessary_map_or,
    clippy::unnecessary_mut_passed,
    clippy::useless_conversion,
    clippy::vec_box,
    clippy::while_let_on_iterator,
    non_snake_case,
    private_interfaces,
    unsafe_op_in_unsafe_fn,
    unused_doc_comments,
    unused_imports,
    unused_unsafe,
    unused_variables
)]

//! pyre-jit-trace: Trace-time JIT for pyre.
//!
//! This crate contains MIFrame (the meta-interpreter frame) and all
//! trace-time logic. It is compiled as a separate compilation unit
//! from pyre-jit's eval_loop_jit to prevent MIFrame's monomorphization
//! of `execute_opcode_step<E>` from bloating the eval loop's codegen.

// Self-alias so include!()'d codegen written for `majit-translate`'s
// crate name keeps compiling when its source is also `include!`d into
// this crate's `generated*` modules (jit_trace_gen.rs).  Allows generic
// bounds like `F: pyre_jit_trace::walker_frame_ops::WalkerFrameOps` to
// resolve from both sides.
extern crate self as pyre_jit_trace;

#[cfg(test)]
mod allocation_counter {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct CountingAllocator;

    static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            }
            ptr
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc_zeroed(layout) };
            if !ptr.is_null() {
                LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) };
            LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
            if !new_ptr.is_null() {
                if new_size >= layout.size() {
                    LIVE_BYTES.fetch_add(new_size - layout.size(), Ordering::Relaxed);
                } else {
                    LIVE_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
                }
            }
            new_ptr
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    pub fn live_bytes() -> usize {
        LIVE_BYTES.load(Ordering::Relaxed)
    }
}

pub mod assembler;
pub mod callbacks;
#[cfg(test)]
mod codegen_cache;
pub mod ctor_continuation;
pub mod descr;
pub mod driver;
pub mod frame_layout;
pub mod helpers;
pub mod jitcode_dispatch;
pub mod jitcode_runtime;
pub mod liveness;
pub mod llbc_fingerprint;
pub mod operator_continuation;
pub mod py_coord;
pub mod pyjitcode;
pub mod pyjitpl;
pub mod pypyjit_driver_layout;
/// `pypy/module/pypyjit/policy.py`, which the build script links for the
/// translation (`build/prepass.rs`); compiled here for its tests only.
#[cfg(test)]
#[path = "../../pyre-interpreter/src/module/pypyjit/policy.rs"]
mod pypyjit_policy;
pub mod pyre_cpu;
pub mod runtime_fnaddr_patch;
pub mod state;
pub mod super_inst_expand;
mod trace_opcode;

pub mod genentry_state;
#[cfg(test)]
mod ullbc_semantic;
pub mod unpack_state;
pub use pyjitcode::{PyJitCode, PyJitCodeMetadata};
pub mod trace;
pub mod virtualizable_gen;
pub mod virtualizable_spec;
pub mod walker_frame_ops;

/// Build a standalone trace context on a libtest thread after installing the
/// interpreter's test-only dict hash hooks for that thread.
#[cfg(test)]
pub(crate) fn trace_ctx_for_test(num_inputs: usize) -> majit_metainterp::TraceCtx {
    pyre_interpreter::test_hooks::install_hash_hook();
    majit_metainterp::TraceCtx::for_test(num_inputs)
}

/// Put the original address back. `intern` finds a slot by the address
/// currently stored there, so a forwarded slot would mint a second index.
#[cfg(test)]
pub(crate) struct RestoreConstPtr {
    pub index: u32,
    pub addr: majit_ir::GcRef,
}

#[cfg(test)]
impl RestoreConstPtr {
    pub fn intern(addr: usize) -> Self {
        let addr = majit_ir::GcRef(addr);
        Self {
            index: majit_ir::const_ptr_table::intern(addr),
            addr,
        }
    }
}

#[cfg(test)]
impl Drop for RestoreConstPtr {
    fn drop(&mut self) {
        majit_ir::const_ptr_table::set_slot(self.index, self.addr);
    }
}

// pyre-jit-trace local invariant: PyFrame's `_virtualizable_` declares
// exactly one extra red (ec, see `virtualizable_gen.rs`'s `extra_reds` and
// `pypy/module/pypyjit/interp_jit.py reds = ['frame', 'ec']`).
// `majit-macros::virtualizable!` itself is generic over `extra_reds.len()`
// (mod.rs), so this assertion is *pyre-local*. Tracing-time helpers
// that seed/push the ec slot rely on this invariant; bumping it requires
// re-auditing every ec wiring callsite.
const _: () = assert!(
    virtualizable_gen::NUM_EXTRA_REDS == 1,
    "pyre's PyFrame virtualizable layout requires exactly one extra red (ec)",
);

/// `MAJIT_PROBE_SUBSCR` env-var gate cached once on first read. The
/// state.rs/jitcode_dispatch.rs probe sites are on hot paths; sampling
/// `std::env::var_os` on every cache hit would dominate the cost when
/// the probe is off.
pub(crate) fn probe_subscr_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("MAJIT_PROBE_SUBSCR").is_some())
}

/// Auto-generated trace functions from majit-translate.
#[allow(dead_code, unsafe_op_in_unsafe_fn, unused_imports, unused_variables)]
pub mod generated {
    use pyre_interpreter::bytecode::{BinaryOperator, ComparisonOperator};
    include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));
}

// Re-export top-level auto-generated functions for crate-level access.
// Keep generated-code lint allowances scoped to this include wrapper.
#[allow(dead_code, unsafe_op_in_unsafe_fn, unused_variables)]
mod generated_root {
    use pyre_interpreter::bytecode::{BinaryOperator, ComparisonOperator};
    include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));
}
pub use generated_root::*;

// Fixed trace helpers (operator dispatch tables, concrete computation,
// unbox/box/binop trace primitives, typed `generated_*` operations). These
// are hand-maintained Rust — the `pyjitpl.py`/`executor.py` analogs — not
// translator output, so they live in a real module. Re-exported at crate
// root because call sites reference them as `crate::<name>`.
#[allow(unsafe_op_in_unsafe_fn, unused_imports, unused_variables)]
mod trace_helpers;
pub use trace_helpers::*;
