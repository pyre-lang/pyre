//! Global Interpreter Lock — `pypy/module/thread/gil.py`.
//!
//! The lock itself lives in `majit_gc::rgil` (the `rpython/translator/c/src`
//! side). What this module adds is the half that belongs to the object space:
//! a thread holds the GIL for as long as it runs pyre code, so without a
//! periodic hand-off a compute-bound thread would never let another one run.
//! `GILReleaseAction` is that hand-off — an action registered on the ticker
//! which yields the GIL every `sys.getcheckinterval()` bytecodes.

use std::sync::OnceLock;

use crate::executioncontext::{
    AsyncAction, AsyncActionControl, AsyncActionOps, ExecutionContext, PeriodicAsyncAction,
    PeriodicAsyncActionOps,
};
use crate::pyframe::PyFrame;
use pyre_object::PyObjectRef;

/// gil.py `GILReleaseAction` — "an action called every
/// `sys.checkinterval` bytecodes. It releases the GIL to give some other
/// thread a chance to run."
pub struct GilReleaseAction {
    base: PeriodicAsyncAction,
}

impl GilReleaseAction {
    fn new(space: PyObjectRef) -> Box<Self> {
        Box::new(Self {
            base: *PeriodicAsyncAction::new(space),
        })
    }
}

impl AsyncActionOps for GilReleaseAction {
    /// gil.py `perform`: request `rgil.yield_thread()`.
    ///
    /// The dispatcher performs the request after this method returns so its
    /// `&mut GilReleaseAction` has ended before another thread dispatches the
    /// same process-owned action.  This is the Rust ownership boundary around
    /// the otherwise line-for-line upstream hand-off.
    fn perform(
        &mut self,
        _ec: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<AsyncActionControl, crate::PyError> {
        Ok(AsyncActionControl::YieldGil)
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base.base
    }
}

impl PeriodicAsyncActionOps for GilReleaseAction {}

static GIL_RELEASE_ACTION: OnceLock<usize> = OnceLock::new();

/// gil.py `GILThreadLocals.initialize` — "add the GIL-releasing callback
/// as an action on the space".
///
/// `use_bytecode_counter=True` is what puts it at the end of the periodic list
/// (executioncontext.py:503-504: "hack to put the release-the-GIL one at the
/// end of the list"), behind the signal check. Idempotent; the actionflag
/// holds the action's heap address, so the process-owned flag retains it for
/// the runtime lifetime.
pub fn initialize(ec: &mut ExecutionContext) {
    GIL_RELEASE_ACTION.get_or_init(|| {
        let action: &'static mut GilReleaseAction = Box::leak(GilReleaseAction::new(ec.space()));
        action.register_periodic_action(ec.actionflag.shared_mut(), true);
        action as *mut GilReleaseAction as usize
    });
}

/// Trace the process-owned periodic action's object-space reference.  The
/// action is a translated object-space child in PyPy; pyre's leaked Rust box
/// needs the corresponding explicit non-stack root.
pub(super) fn walk_action_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let Some(&addr) = GIL_RELEASE_ACTION.get() else {
        return;
    };
    let action = unsafe { &mut *(addr as *mut GilReleaseAction) };
    let slot = &mut action.base.base.space;
    if !slot.is_null() {
        visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
    }
}

/// gil.py `GILThreadLocals.setup_threads` — "enable threads in the object
/// space, if they haven't already been". Returns whether this call is the one
/// that set them up, which is a property of the space and not of the calling
/// thread.
///
/// The flag itself is `pyre_object::gil_ready` (`gil_ready?`). This function
/// still allocates before publishing, and the publish invalidates traces
/// that folded the zero.
pub fn setup_threads(ec: &mut ExecutionContext) -> bool {
    debug_assert!(
        majit_gc::rgil::am_i_holding_the_gil(),
        "setup_threads needs the GIL"
    );
    let first = !pyre_object::gil_ready::gil_ready_is_set();
    if first {
        // gil.py `setup_threads` allocates before publishing the flag.
        majit_gc::rgil::allocate();
        pyre_object::gil_ready::publish_gil_ready();
    }
    initialize(ec);
    first
}

/// gil.py `GILThreadLocals.threads_initialized`, reached through
/// `os_thread.py threads_initialized(space)`.
pub fn threads_initialized() -> bool {
    pyre_object::gil_ready::gil_ready_is_set()
}
