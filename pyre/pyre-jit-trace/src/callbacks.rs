//! Function pointer table for pyre-jit → pyre-jit-trace callback bridge.
//!
//! pyre-jit-trace cannot depend on pyre-jit (that would be circular).
//! Instead, pyre-jit registers function pointers at init time, and
//! pyre-jit-trace calls them through this table.

use pyre_interpreter::CodeObject;
use pyre_object::PyObjectRef;
use std::sync::OnceLock;

/// Callback table populated by pyre-jit at initialization.
///
/// All function pointers are `extern "C"` JIT helpers from `call_jit.rs`
/// and eval.rs driver access.
pub struct CallJitCallbacks {
    // call_jit.rs helpers
    pub callee_frame_helper: fn(usize) -> Option<*const ()>,
    pub recursive_force_cache_safe: fn(PyObjectRef) -> bool,
    pub jit_drop_callee_frame: *const (),
    pub jit_force_callee_frame: *const (),
    pub jit_force_recursive_call_1: *const (),
    pub jit_force_recursive_call_argraw_boxed_1: *const (),
    pub jit_force_self_recursive_call_argraw_boxed_1: *const (),
    pub jit_create_callee_frame_1: *const (),
    pub jit_create_callee_frame_1_raw_int: *const (),
    pub jit_create_self_recursive_callee_frame_1: *const (),
    pub jit_create_self_recursive_callee_frame_1_raw_int: *const (),
    // eval.rs driver access (opaque pointer to JitDriverPair)
    pub driver_pair: fn() -> *mut u8,
    /// codewriter.py:make_jitcodes parity: build the majit JitCode for
    /// `code` through CallControl.get_jitcode + the pending-graph drain, then
    /// publish the same populated PyJitCode Arc into trace-side staticdata.
    /// Returns false when the graph cannot be represented as a JitCode.
    pub ensure_majit_jitcode: fn(*const CodeObject, *const ()) -> bool,
    /// Drain the backend `_store_exception` cells (`jit_exc_clear`).  Called
    /// from the authoritative residual-call executor's raise arm so a raise
    /// recorded into `last_exc_value` during tracing does not also leave the
    /// backend exception cells set — `pyjitpl.py execute_raised` writes
    /// only `metainterp.last_exc_value`, never the backend cells.
    pub drain_backend_jit_exc: fn(),
    /// Forward `payload.w_code` of every jitcode owned by the running
    /// `CodeWriter`'s `CallControl.jitcodes`. That dict is the owner while
    /// the drain compiles a queued skeleton; `MetaInterpStaticData.jitcodes`
    /// only sees the populated result after install. `walk_jitcode_code_roots_in`
    /// calls this so one walker covers both owners.
    pub walk_writer_jitcode_code_roots: fn(&mut dyn FnMut(&mut majit_ir::GcRef)),
}

// Safety: function pointers are 'static and never mutated after init
unsafe impl Send for CallJitCallbacks {}
unsafe impl Sync for CallJitCallbacks {}

// RPython's translated helper addresses live on MetaInterpStaticData: one
// immutable table shared by every execution context. The callbacks route
// thread-specific operations (notably `driver_pair`) through their functions;
// duplicating the table itself per thread only makes a newly-created Python
// thread observe an uninitialized module.
static CALLBACKS: OnceLock<&'static CallJitCallbacks> = OnceLock::new();

/// Register the callback table. Called once from pyre-jit's eval init.
pub fn init(cb: &'static CallJitCallbacks) {
    if CALLBACKS.set(cb).is_err() {
        assert!(
            std::ptr::eq(*CALLBACKS.get().unwrap(), cb),
            "CallJitCallbacks initialized twice with different tables"
        );
    }
}

/// Get the callback table. Panics if not initialized.
#[inline]
pub fn get() -> &'static CallJitCallbacks {
    CALLBACKS
        .get()
        .copied()
        .expect("CallJitCallbacks not initialized")
}

/// Optional callback table lookup for cold paths that can fall back to
/// skeleton-only behavior in tests before pyre-jit initializes callbacks.
#[inline]
pub fn try_get() -> Option<&'static CallJitCallbacks> {
    CALLBACKS.get().copied()
}
