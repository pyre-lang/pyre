//! Process-global fn-pointer hook cells.
//!
//! [`FnPtrCell`] is the free-threading replacement for a
//! `thread_local! { static X: Cell<Option<fn(..)>> }` GC hook. Before the GC
//! became a process-global singleton (`gc_sync`), each thread could own a
//! distinct GC and installed its own hook fn pointers; now every thread
//! installs the *same* pointer, so the per-thread copy is redundant and — for
//! a collector that may run on an arbitrary thread — unsound (a thread that
//! never ran the install path would collect with the hook unset).
//!
//! The value is a bare `fn(..)` pointer, which is thin and pointer-sized, so
//! it is stored bit-for-bit in an [`AtomicPtr<()>`] with the null pointer
//! standing in for `None`. This keeps the read side a single acquire atomic
//! load plus a null check — no lock, no heap — matching the cost of the TLS
//! access it replaces on the allocation hot path.
//!
//! Registration mirrors [`crate::shadow_stack::register_extra_root_walker`]:
//! the hooks are process-global constants fixed once at boot (the runtime
//! substitute for RPython's translation-time weaving against a global
//! `gcdata`). Overwrite and clear (`None`) remain supported because the test
//! harness installs mock hooks and resets them; production installs each hook
//! exactly once under a `Once`.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicPtr, Ordering};

/// A process-global cell holding an optional bare `fn` pointer of type `F`.
///
/// `F` must be a bare function-pointer type (`fn(..) -> ..`), which is
/// pointer-sized and thin; this is enforced by a size `debug_assert` on every
/// access and would be a compile-time-visible mismatch for any wider `F`.
pub struct FnPtrCell<F: Copy> {
    ptr: AtomicPtr<()>,
    _marker: PhantomData<F>,
}

// SAFETY: the stored value is a code address (a bare fn pointer), which is
// itself `Send + Sync`; all access is through the `AtomicPtr`, which provides
// the synchronization. `PhantomData<F>` carries no runtime state.
unsafe impl<F: Copy> Sync for FnPtrCell<F> {}

impl<F: Copy> FnPtrCell<F> {
    /// Compile-time guard: `F` must be pointer-sized, i.e. a bare `fn(..)`
    /// pointer. `transmute_copy` does not check sizes, so a wider `F` would
    /// read out of bounds in `get`/`set`; forcing this associated const in
    /// both turns that into a build error instead of release-only UB.
    const SIZE_OK: () = assert!(
        std::mem::size_of::<F>() == std::mem::size_of::<*mut ()>(),
        "FnPtrCell<F>: F must be a pointer-sized bare fn pointer",
    );

    /// A cell with no hook installed (`None`).
    pub const fn new() -> Self {
        Self {
            ptr: AtomicPtr::new(std::ptr::null_mut()),
            _marker: PhantomData,
        }
    }

    /// Read the installed hook, or `None` if unset/cleared.
    ///
    /// `Acquire` pairs with the `Release` store in [`Self::set`] so a reader
    /// that observes the pointer also observes everything the installer
    /// published before it. On x86-64/aarch64 an acquire load is a plain load,
    /// so this is no more expensive than the thread-local read it replaces.
    #[inline]
    pub fn get(&self) -> Option<F> {
        let () = Self::SIZE_OK;
        let p = self.ptr.load(Ordering::Acquire);
        if p.is_null() {
            None
        } else {
            // SAFETY: `p` is non-null and was produced by `set` from a value of
            // type `F` (a pointer-sized fn pointer, per SIZE_OK); reinterpreting
            // its bits back to `F` round-trips the original pointer.
            Some(unsafe { std::mem::transmute_copy::<*mut (), F>(&p) })
        }
    }

    /// Install (`Some`) or clear (`None`) the hook.
    ///
    /// Production installs each hook once at boot; the setter tolerates being
    /// called again with the same value (idempotent) and supports clearing so
    /// the `#[cfg(test)]` isolation seam can reset between tests.
    pub fn set(&self, f: Option<F>) {
        let () = Self::SIZE_OK;
        let raw: *mut () = match f {
            None => std::ptr::null_mut(),
            Some(f) => {
                // SAFETY: `F` is a pointer-sized bare fn pointer (per SIZE_OK);
                // reinterpret its bits as an erased data pointer for storage.
                // `set(None)` stores null, which `get` reads back as `None`.
                unsafe { std::mem::transmute_copy::<F, *mut ()>(&f) }
            }
        };
        self.ptr.store(raw, Ordering::Release);
    }
}

impl<F: Copy> Default for FnPtrCell<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Declare a process-global fn-pointer hook cell backed by [`FnPtrCell`].
///
/// Replaces a `thread_local! { static NAME: Cell<Option<Fn>> = .. }` hook with
/// a process-global `static NAME: FnPtrCell<Fn>`. The accessor functions
/// (`set_*` / `try_*` / getters) stay hand-written so their attributes
/// (`#[dont_look_inside]`, `extern "C"`) and per-hook fallback values are
/// preserved; they just read/write `NAME` via [`FnPtrCell::get`] /
/// [`FnPtrCell::set`] instead of `NAME.with(..)`.
///
/// ```ignore
/// majit_gc::global_hook!(static ACTIVE_CAN_MOVE: CanMoveFn);
/// ```
#[macro_export]
macro_rules! global_hook {
    ($(#[$attr:meta])* $vis:vis static $name:ident : $fnty:ty) => {
        $(#[$attr])*
        $vis static $name: $crate::hook_cell::FnPtrCell<$fnty> =
            $crate::hook_cell::FnPtrCell::new();
        $crate::paste::paste! {
            #[cfg(not(target_arch = "wasm32"))]
            #[::majit_ir::linkme::distributed_slice(::majit_ir::helper_fnaddr::FNPTRCELL_STATICS)]
            #[linkme(crate = ::majit_ir::linkme)]
            #[allow(non_upper_case_globals)]
            static [<__FNPTRCELL_STATIC_ $name>]: ::majit_ir::helper_fnaddr::PrebuiltStaticAddr =
                ::majit_ir::helper_fnaddr::PrebuiltStaticAddr::new(
                    ::core::concat!(::core::module_path!(), "::", stringify!($name)),
                    &$name as *const _ as *const (),
                );
            #[cfg(target_arch = "wasm32")]
            #[::ctor::ctor(unsafe)]
            #[allow(non_snake_case)]
            fn [<__fnptrcell_register_ $name>]() {
                ::majit_ir::helper_fnaddr::register_fnptrcell_static(
                    ::core::concat!(::core::module_path!(), "::", stringify!($name)),
                    &$name as *const _ as *const (),
                );
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    type UnaryFn = fn(usize) -> usize;

    static CELL: FnPtrCell<UnaryFn> = FnPtrCell::new();

    fn double(x: usize) -> usize {
        x * 2
    }
    fn triple(x: usize) -> usize {
        x * 3
    }

    #[test]
    fn none_when_unset() {
        let cell: FnPtrCell<UnaryFn> = FnPtrCell::new();
        assert!(cell.get().is_none());
    }

    #[test]
    fn roundtrips_install_overwrite_clear() {
        let cell: FnPtrCell<UnaryFn> = FnPtrCell::new();
        cell.set(Some(double));
        assert_eq!(cell.get().unwrap()(21), 42);
        // overwrite with a different value
        cell.set(Some(triple));
        assert_eq!(cell.get().unwrap()(2), 6);
        // clear
        cell.set(None);
        assert!(cell.get().is_none());
    }

    #[test]
    fn static_cell_usable() {
        CELL.set(Some(double));
        assert_eq!(CELL.get().unwrap()(5), 10);
        CELL.set(None);
    }
}
