//! `gil.py` `GILThreadLocals.gil_ready`.
//!
//! `_immutable_fields_ = ['gil_ready?']`. The word stays 0 until
//! `setup_threads` publishes it, and it never returns to 0. List iteration
//! reads it so the stripe lock stays out of a trace while the process is
//! still single-threaded. Publishing invalidates loops that folded the zero.
//!
//! The word is a plain `static mut` field, not an atomic. A traced
//! `Atomic::load` does not lower (`Acquire` drops the function;
//! `Relaxed` aliases the load to the atomic object), and an immutable
//! `static` folds to its initializer.

use std::sync::Arc;

/// Process-wide cell. One field, `repr(C)`, so the translated read is
/// offset 0 / 8 bytes / `Int`.
#[repr(C)]
pub struct GilReadyState {
    /// 0 until the first `setup_threads`, then 1.
    pub gil_ready: i64,
}

const _: () = assert!(std::mem::offset_of!(GilReadyState, gil_ready) == 0);
const _: () = assert!(std::mem::size_of::<GilReadyState>() == 8);

/// `static mut` so a translated read is a field load of the live cell,
/// not the initializer folded into the flow graph.
pub static mut GIL_READY_STATE: GilReadyState = GilReadyState { gil_ready: 0 };

/// Hidden `mutate_gil_ready` for the field spelled `gil_ready?`.
/// Not a field of [`GIL_READY_STATE`]: the value itself stays the single
/// `i64` at offset 0.
static GIL_READY_WATCHERS: crate::quasiimmut::QuasiImmutField =
    crate::quasiimmut::QuasiImmutField::new();

/// `quasiimmut.py get_current_qmut_instance` for `GILThreadLocals.gil_ready`.
pub fn gil_ready_current_qmut() -> Arc<crate::quasiimmut::QuasiImmut> {
    GIL_READY_WATCHERS.get_current_qmut_instance()
}

/// Plain field read of [`GIL_READY_STATE`]. Not `dont_look_inside`.
///
/// `#[inline(never)]` keeps this body in its own flow graph. A lowering
/// failure then residualizes one leaf call instead of dropping
/// `list_iter_descr_next`.
#[inline(never)]
pub fn gil_ready_word() -> i64 {
    // Raw place copy: edition 2024 rejects a reference to `static mut`.
    // SAFETY: the cell is published once, from 0 to 1, before any other
    // thread exists (`setup_threads` runs on the parent before `spawn`).
    // After that the word is only read.
    unsafe { GIL_READY_STATE.gil_ready }
}

#[inline]
pub fn gil_ready_is_set() -> bool {
    gil_ready_word() != 0
}

/// `gil.py setup_threads` publishes the flag only after `rgil.allocate`.
/// Invalidate first, then store, so a loop that folded the zero fails its
/// `GUARD_NOT_INVALIDATED` before any other thread observes 1.
pub fn publish_gil_ready() {
    GIL_READY_WATCHERS.invalidate_then_store(|| {
        // SAFETY: same single-writer protocol as [`gil_ready_word`]. The
        // watcher lock is held across the store.
        unsafe {
            GIL_READY_STATE.gil_ready = 1;
        }
    });
}
