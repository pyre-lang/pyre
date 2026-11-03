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
//!
//! The hidden `mutate_gil_ready` field lives on this STRUCT
//! (`rclass.py InstanceRepr._setup_repr` / `quasiimmut.py
//! get_mutate_field_name`).

use std::sync::Arc;

/// Process-wide cell. `repr(C)` pins `gil_ready` at offset 0 / 8 bytes /
/// `Int` and the hidden watcher immediately after it.
#[repr(C)]
#[majit_macros::jit_immutable_fields("gil_ready?")]
pub struct GilReadyState {
    /// 0 until the first `setup_threads`, then 1.
    pub gil_ready: i64,
    /// Hidden `mutate_gil_ready` for the field spelled `gil_ready?`.
    pub mutate_gil_ready: std::sync::atomic::AtomicPtr<crate::quasiimmut::QuasiImmut>,
    pub mutate_gil_ready_lock: std::sync::atomic::AtomicPtr<crate::quasiimmut::QuasiImmutLock>,
}

crate::quasi_immut_words!(GilReadyState, mutate_gil_ready, mutate_gil_ready_lock);

impl GilReadyState {
    #[inline]
    pub fn mutate_gil_ready(&self) -> crate::quasiimmut::QuasiImmutField<'_> {
        crate::quasiimmut::QuasiImmutField::new(&self.mutate_gil_ready, &self.mutate_gil_ready_lock)
    }
}

const _: () = assert!(std::mem::offset_of!(GilReadyState, gil_ready) == 0);
const _: () = assert!(std::mem::offset_of!(GilReadyState, mutate_gil_ready) == 8);
const _: () = assert!(std::mem::size_of::<GilReadyState>() == 8 + 2 * std::mem::size_of::<usize>());

/// `static mut` so a translated read is a field load of the live cell,
/// not the initializer folded into the flow graph.
pub static mut GIL_READY_STATE: GilReadyState = GilReadyState {
    gil_ready: 0,
    mutate_gil_ready: crate::quasiimmut::null_mutate_word(),
    mutate_gil_ready_lock: crate::quasiimmut::null_mutate_lock(),
};

/// `quasiimmut.py get_current_qmut_instance` for `GILThreadLocals.gil_ready`.
pub fn gil_ready_current_qmut() -> Arc<crate::quasiimmut::QuasiImmut> {
    // Raw place: edition 2024 rejects a reference to `static mut`.
    // SAFETY: same single-writer protocol as [`gil_ready_word`].
    unsafe {
        (*(&raw const GIL_READY_STATE))
            .mutate_gil_ready()
            .get_current_qmut_instance()
    }
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
    // SAFETY: same single-writer protocol as [`gil_ready_word`]. The
    // watcher lock is held across the store. `&raw mut` is the edition-2024
    // place for a `static mut` field.
    unsafe {
        let state = &raw mut GIL_READY_STATE;
        crate::quasi_immut_store!((*state).mutate_gil_ready() =>
            (*state).gil_ready = 1;
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutate_gil_ready_offset_lands_on_the_static_field() {
        let state = &raw const GIL_READY_STATE;
        // Field projection of `static mut` is a mutable-static place.
        let mutate = unsafe { &raw const GIL_READY_STATE.mutate_gil_ready };
        let via = (state as *const u8)
            .wrapping_add(std::mem::offset_of!(GilReadyState, mutate_gil_ready));
        assert_eq!(via, mutate as *const u8);
    }

    /// `publish_gil_ready` unlinks the instance `gil_ready_current_qmut`
    /// handed out (`quasiimmut.py make_invalidation_function._invalidate_now`).
    /// Restore the unpublished word so later list-lock tests still see 0.
    #[test]
    fn publish_gil_ready_invalidates_the_current_qmut() {
        struct RestoreGilReadyWord;
        impl Drop for RestoreGilReadyWord {
            fn drop(&mut self) {
                // SAFETY: test-only restore of the unpublished word.
                unsafe {
                    (*(&raw mut GIL_READY_STATE)).gil_ready = 0;
                }
            }
        }
        let _restore = RestoreGilReadyWord;
        let recorded = gil_ready_current_qmut();
        assert!(recorded.is_current());
        publish_gil_ready();
        assert!(!recorded.is_current());
        assert_eq!(gil_ready_word(), 1);
    }
}
