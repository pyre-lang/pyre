//! JIT driver access from pyre-jit-trace.
//!
//! make_green_key is a pure function. driver_pair() is accessed through
//! callbacks since the JIT_DRIVER TLS lives in pyre-jit/eval.rs.

use crate::callbacks;
use crate::state::PyreJitState;

/// RPython green_key = pypyjit greens `[next_instr, is_being_profiled,
/// pycode]` (interp_jit.py). The returned u64 is the full
/// `JitCell.get_uhash` over the typed green tuple (warmstate.py),
/// so the hash-form lookup and the typed marker-path lookup
/// (`lookup_chain_with_key`) agree on the same cell.
#[inline(always)]
pub fn make_green_key(code_ptr: *const (), pc: usize, is_being_profiled: bool) -> u64 {
    majit_ir::pypyjit_greenkey_uhash(pc, is_being_profiled, code_ptr as u64)
}

/// The typed form of [`make_green_key`]: the greens themselves, not a fold of
/// them.
///
/// A warmstate call that reaches a CELL needs this — `JitCell.comparekey`
/// (warmstate.py `def comparekey(self, *greenargs2)`) is what picks one
/// cell out of a bucket, and every cell is `JitCell(*greenargs)`
/// (`_ensure_jit_cell_at_key`). The u64 form stays correct for calls that
/// only move a counter.
pub fn make_green_key_typed(
    code_ptr: *const (),
    pc: usize,
    is_being_profiled: bool,
) -> majit_ir::GreenKey {
    majit_ir::pypyjit_greenkey(pc, is_being_profiled, code_ptr as u64)
}

/// Read the pypyjit profiling green from the concrete frame for a walk.
#[inline]
pub fn frame_is_being_profiled(frame_addr: usize) -> bool {
    let frame = frame_addr as *const pyre_interpreter::pyframe::PyFrame;
    !frame.is_null() && unsafe { (*frame).get_is_being_profiled() }
}

/// Type alias for the JIT driver pair. Must match pyre-jit/eval.rs JitDriverPair.
pub type JitDriverPair = (
    majit_metainterp::JitDriver<PyreJitState>,
    std::sync::Arc<majit_metainterp::virtualizable::VirtualizableInfo>,
);

/// Get the JIT driver pair through callbacks.
#[inline]
pub fn driver_pair() -> &'static mut JitDriverPair {
    let ptr = (callbacks::get().driver_pair)();
    unsafe { &mut *(ptr as *mut JitDriverPair) }
}

/// Get the JIT driver pair, or `None` when no callback table is installed.
///
/// A skeleton walk drives `jitcode_dispatch` directly with no pyre-jit eval
/// behind it, so there is no driver to reach. Cold bookkeeping paths that only
/// annotate an outcome take this spelling and skip themselves, rather than
/// panicking in `callbacks::get`.
#[inline]
pub fn try_driver_pair() -> Option<&'static mut JitDriverPair> {
    let ptr = (callbacks::try_get()?.driver_pair)();
    Some(unsafe { &mut *(ptr as *mut JitDriverPair) })
}
