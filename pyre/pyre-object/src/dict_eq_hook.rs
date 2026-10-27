//! Thread-local space-callable hooks consumed by pyre-object's dict
//! key dispatch.
//!
//! `pypy/objspace/std/dictmultiobject.py+ ObjectDictStrategy`
//! routes key equality + hashing through `space.eq_w` / `space.hash_w`
//! so user-defined `__eq__` / `__hash__` resolve through the standard
//! comparison protocol.  pyre's `pyre-object` crate is below
//! `pyre-interpreter` in the dependency graph and cannot call into
//! `baseobjspace::eq_w` directly; the hook module mirrors the
//! `gc_hook` pattern: pyre-interpreter (via the pyre-jit init point)
//! installs the trampoline at startup, and pyre-object readers fall
//! back to the builtin equality path when no hook is installed
//! (single-crate tests, snapshot builds before init).
//!
//! Layering: this module has no external dependencies.  Wire-up lives
//! in `pyre-jit::eval`.

use crate::PyObjectRef;
use std::cell::Cell;

/// `pypy/interpreter/baseobjspace.py W_ObjectSpace.eq_w`
/// signature: returns `True` when `a` and `b` are equal per the
/// standard `__eq__` protocol.  PyPy's `space.eq_w` raises on `__eq__`
/// errors; the trampoline cannot return a `Result` across the dict
/// probe, so on error it calls [`signal_eq_error`] and returns `false`.
/// Checked dict ops convert the flag to a `DictKeyError` after the
/// probe; the concrete `PyError` rides the interpreter pending slot.
pub type EqWHookFn = unsafe fn(a: PyObjectRef, b: PyObjectRef) -> bool;

/// `pypy/interpreter/baseobjspace.py is_none W_ObjectSpace.hash_w`
/// signature: returns the `__hash__` digest as `i64` (matching
/// CPython `Py_hash_t`).  On error (unhashable type, user `__hash__`
/// raised, etc.) the hook calls [`signal_hash_error`] and returns 0;
/// callers that need error propagation use [`take_hash_error`] after
/// the hash call.
pub type HashWHookFn = unsafe fn(obj: PyObjectRef) -> i64;

/// `unicodeobject.py W_UnicodeObject.descr_hash` digest computed directly
/// from a key's WTF-8 bytes, with no intervening `W_UnicodeObject`.  Lets a
/// str-keyed dict GET probe (`getitem_str`) land in the same bucket an
/// `object_key_for(w_str_new(key))` would, without materializing the
/// throwaway str — matching PyPy's string-strategy `getitem_str`
/// (`dictmultiobject.py`), which hashes the raw RPython str.
/// `ptr`/`len` describe a valid WTF-8 range for the duration of the call;
/// the `(ptr, len)` ABI keeps the residual fnaddr FFI-clean (a `&[u8]` fat
/// pointer is not).  Str hashing never fails, so there is no error channel.
pub type HashStrHookFn = unsafe fn(ptr: *const u8, len: usize) -> i64;

/// `pypy/objspace/std/typeobject.py compares_by_identity
/// W_TypeObject.compares_by_identity` trampoline.  Walks the type's
/// MRO via `lookup_in_type('__eq__')` / `('__hash__')` and compares
/// against the object-default to decide if identity comparison is
/// observable-equivalent.  Updates the type's
/// `compares_by_identity_status` cache slot.
pub type ComparesByIdentityHookFn = unsafe fn(w_type: PyObjectRef) -> bool;

thread_local! {
    static EQ_W_HOOK: Cell<Option<EqWHookFn>> = const { Cell::new(None) };
    static HASH_W_HOOK: Cell<Option<HashWHookFn>> = const { Cell::new(None) };
    static HASH_STR_HOOK: Cell<Option<HashStrHookFn>> = const { Cell::new(None) };
    static COMPARES_BY_IDENTITY_HOOK: Cell<Option<ComparesByIdentityHookFn>> = const { Cell::new(None) };
    /// Error flag set by the hash hook when `space.hash_w` encounters
    /// an unhashable type or a user `__hash__` raises.  Only presence
    /// is observed (the error payload itself travels through the
    /// interpreter-side pending-error slot the trampoline fills before
    /// signalling).  Null means no error.
    static HASH_W_ERROR: Cell<PyObjectRef> = const { Cell::new(std::ptr::null_mut()) };
    /// Error flag set by the `eq_w` hook when a user `__eq__` (or the
    /// `__bool__` of its result) raises during a key comparison.  The
    /// hash flag fires at key-construction time; this one fires during
    /// the bucket probe, so checked dict ops consult it *after* the
    /// `IndexMap` access.  Presence-only, same as the hash flag; the
    /// payload travels through the interpreter pending-error slot.
    static EQ_W_ERROR: Cell<PyObjectRef> = const { Cell::new(std::ptr::null_mut()) };
    /// Set while a bucket probe runs on the callback-free fast path.
    /// `dict_keys_equal` then answers from its builtin type ladder alone and
    /// never enters `space.eq_w`, so no user code observes or mutates the
    /// container mid-probe.
    static CALLBACK_FREE_PROBE: Cell<bool> = const { Cell::new(false) };
    /// Raised by `dict_keys_equal` when a comparison inside a callback-free
    /// probe would have needed `space.eq_w`.  The probe's result is then
    /// meaningless and the caller re-runs the operation on the path that
    /// tolerates a re-entrant callback.
    static CALLBACK_FREE_PROBE_BROKEN: Cell<bool> = const { Cell::new(false) };
}

/// Enter a callback-free probe.  Clears the broken flag so
/// [`end_callback_free_probe`] reports only this probe's comparisons.
#[majit_macros::dont_look_inside]
pub extern "C" fn begin_callback_free_probe() {
    CALLBACK_FREE_PROBE.with(|cell| cell.set(true));
    CALLBACK_FREE_PROBE_BROKEN.with(|cell| cell.set(false));
}

/// Leave a callback-free probe, returning `true` when a comparison needed
/// `space.eq_w` and the probe's result must therefore be discarded.
#[majit_macros::dont_look_inside]
pub extern "C" fn end_callback_free_probe() -> bool {
    CALLBACK_FREE_PROBE.with(|cell| cell.set(false));
    CALLBACK_FREE_PROBE_BROKEN.with(|cell| cell.replace(false))
}

/// True while a callback-free probe is in progress.
#[majit_macros::dont_look_inside]
pub extern "C" fn callback_free_probe_active() -> bool {
    CALLBACK_FREE_PROBE.with(|cell| cell.get())
}

/// True once a comparison in the current callback-free probe hit a key pair
/// the builtin ladder cannot decide.  Read mid-probe by callers that mutate
/// only after a clean lookup.
#[majit_macros::dont_look_inside]
pub extern "C" fn callback_free_probe_broken() -> bool {
    CALLBACK_FREE_PROBE_BROKEN.with(|cell| cell.get())
}

/// Record that the current callback-free probe reached a comparison the
/// builtin ladder cannot decide.
#[majit_macros::dont_look_inside]
pub extern "C" fn break_callback_free_probe() {
    CALLBACK_FREE_PROBE_BROKEN.with(|cell| cell.set(true));
}

/// Signal that the most recent `hash_w` call failed.  Called by the
/// hash hook trampoline when `try_hash_value` returns `Err`.  The
/// caller observes the flag via [`take_hash_error`].
/// `dont_look_inside`: thread-local write, residualizes via the
/// registered fnaddr.  `obj` must be a valid PyObjectRef.
#[majit_macros::dont_look_inside]
pub extern "C" fn signal_hash_error(obj: PyObjectRef) {
    HASH_W_ERROR.with(|cell| cell.set(obj));
}

/// Consume the pending hash error flag, returning `true` if an error
/// was signalled since the last `take`.  Callers only branch on
/// presence; the error payload is retrieved from the interpreter-side
/// pending-error slot.  `dont_look_inside`: thread-local read,
/// residualizes via the registered fnaddr.
#[majit_macros::dont_look_inside]
pub extern "C" fn take_hash_error() -> bool {
    HASH_W_ERROR.with(|cell| {
        let obj = cell.get();
        if obj.is_null() {
            false
        } else {
            cell.set(std::ptr::null_mut());
            true
        }
    })
}

/// Signal that the most recent `eq_w` key comparison raised.  Called by
/// the eq hook trampoline when `space.eq_w` returns `Err`.  Checked dict
/// ops observe the flag with [`take_eq_error`] after the bucket probe.
/// `dont_look_inside`: thread-local write, residualizes via the
/// registered fnaddr.  `obj` must be a valid PyObjectRef.
#[majit_macros::dont_look_inside]
pub extern "C" fn signal_eq_error(obj: PyObjectRef) {
    EQ_W_ERROR.with(|cell| cell.set(obj));
}

/// Consume the pending eq error flag, returning `true` if a comparison
/// raised since the last `take`.  Callers branch on presence only; the
/// error payload is retrieved from the interpreter-side pending-error
/// slot.  `dont_look_inside`: thread-local read, residualizes via the
/// registered fnaddr.
#[majit_macros::dont_look_inside]
pub extern "C" fn take_eq_error() -> bool {
    EQ_W_ERROR.with(|cell| {
        let obj = cell.get();
        if obj.is_null() {
            false
        } else {
            cell.set(std::ptr::null_mut());
            true
        }
    })
}

/// Peek the pending eq error flag without consuming it.
/// `dict_keys_equal` consults this at the top of every comparison so
/// that once `space.eq_w` has raised during a bucket probe, the
/// remaining comparisons in that probe are skipped: the Rust `Eq`
/// callback cannot abort an `IndexMap` scan, but suppressing further
/// user `__eq__` calls means no extra comparison runs and the FIRST
/// exception is the one retained — matching `r_dict(space.eq_w, ...)`
/// which raises at the first comparison (`dictmultiobject.py`).
/// `dont_look_inside`: thread-local read, residualizes via the
/// registered fnaddr (the `take_eq_error` twin).
#[majit_macros::dont_look_inside]
pub extern "C" fn eq_error_pending() -> bool {
    EQ_W_ERROR.with(|cell| !cell.get().is_null())
}

/// Install the `eq_w` callback for this thread.  Called from
/// `pyre-jit::eval`'s init alongside the other thread-local hooks.
pub fn register_eq_w_hook(hook: EqWHookFn) {
    EQ_W_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the `eq_w` callback on this thread.
pub fn clear_eq_w_hook() {
    EQ_W_HOOK.with(|cell| cell.set(None));
}

/// Install the `hash_w` callback for this thread.  Companion to
/// `register_eq_w_hook`; together they let `dict_keys_equal` enforce
/// r_dict semantics (eq_w + hash_w pair, per `dictmultiobject.py:1210
/// r_dict(space.eq_w, space.hash_w, force_non_null=True)`).
pub fn register_hash_w_hook(hook: HashWHookFn) {
    HASH_W_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the `hash_w` callback on this thread.
pub fn clear_hash_w_hook() {
    HASH_W_HOOK.with(|cell| cell.set(None));
}

/// True when an `eq_w` hook is installed on this thread.
/// `dont_look_inside`: thread-local read, residualizes via the registered
/// fnaddr (same pattern as [`has_hash_w_hook`]).
#[majit_macros::dont_look_inside]
pub extern "C" fn has_eq_w_hook() -> bool {
    EQ_W_HOOK.with(|cell| cell.get().is_some())
}

/// Invoke the installed `eq_w` hook; returns `false` when no hook is
/// installed — gate with [`has_eq_w_hook`] (the [`try_eq_w`] wrapper does).
/// Not `unsafe` for ABI-surface reasons (residual calls dispatch through a
/// plain fnaddr); `a` / `b` must nonetheless be valid PyObjectRefs (null
/// tolerated as per PyPy's `is_w` shortcut at `baseobjspace.py`).
#[majit_macros::dont_look_inside]
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub extern "C" fn eq_w_hooked(a: PyObjectRef, b: PyObjectRef) -> bool {
    EQ_W_HOOK.with(|cell| cell.get().map(|f| unsafe { f(a, b) }).unwrap_or(false))
}

/// Invoke the installed `eq_w` hook.  Returns `None` when no hook is
/// installed (pyre-object lib tests, pre-init snapshot tools); the
/// caller falls back to the limited-type builtin equality so existing
/// behavior is preserved.
///
/// # Safety
/// `a` / `b` must be valid PyObjectRefs (null tolerated as per
/// PyPy's `is_w` shortcut at `baseobjspace.py`).
#[inline]
pub unsafe fn try_eq_w(a: PyObjectRef, b: PyObjectRef) -> Option<bool> {
    if has_eq_w_hook() {
        Some(eq_w_hooked(a, b))
    } else {
        None
    }
}

/// Invoke the installed `hash_w` hook.  Returns `None` when no hook
/// is installed; callers treat that as "hash unavailable" and fall
/// back to eq-only comparison (which preserves pre-Item-1.2 behavior
/// on snapshot tools / single-crate tests).
///
/// When the hook signals an error (unhashable type, user `__hash__`
/// raised), it calls [`signal_hash_error`] before returning 0.
/// Callers that need error propagation should call
/// [`take_hash_error`] after this function returns `Some(0)`.
///
/// # Safety
/// `obj` must be a valid PyObjectRef (null tolerated).
#[inline]
pub unsafe fn try_hash_w(obj: PyObjectRef) -> Option<i64> {
    if has_hash_w_hook() {
        Some(hash_w_hooked(obj))
    } else {
        None
    }
}

/// True when a `hash_w` hook is installed on this thread.
/// `dont_look_inside`: the thread-local read has no liftable RPython
/// shape; calls residualize via the registered fnaddr (same pattern
/// as `gc_hook::try_gc_write_barrier`).
#[majit_macros::dont_look_inside]
pub extern "C" fn has_hash_w_hook() -> bool {
    HASH_W_HOOK.with(|cell| cell.get().is_some())
}

/// Invoke the installed `hash_w` hook; returns 0 when no hook is
/// installed — gate with [`has_hash_w_hook`] (the [`try_hash_w`]
/// wrapper does).  Not `unsafe` for ABI-surface reasons (residual
/// calls dispatch through a plain fnaddr); `obj` must nonetheless be
/// a valid PyObjectRef (null tolerated).
#[majit_macros::dont_look_inside]
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub extern "C" fn hash_w_hooked(obj: PyObjectRef) -> i64 {
    HASH_W_HOOK.with(|cell| cell.get().map(|f| unsafe { f(obj) }).unwrap_or(0))
}

/// Install the `hash_str` callback for this thread.  Companion to
/// [`register_hash_w_hook`]: it computes the same digest `hash_w` yields
/// for a `str`, but straight from the WTF-8 bytes, so str-keyed `getitem_str`
/// probes avoid building a throwaway `W_UnicodeObject`.  Installed at the
/// same boot point as the other hooks.
pub fn register_hash_str_hook(hook: HashStrHookFn) {
    HASH_STR_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the `hash_str` callback on this thread.
pub fn clear_hash_str_hook() {
    HASH_STR_HOOK.with(|cell| cell.set(None));
}

/// True when a `hash_str` hook is installed on this thread.
/// `dont_look_inside`: thread-local read, residualizes via the registered
/// fnaddr (same pattern as [`has_hash_w_hook`]).
#[majit_macros::dont_look_inside]
pub extern "C" fn has_hash_str_hook() -> bool {
    HASH_STR_HOOK.with(|cell| cell.get().is_some())
}

/// Invoke the installed `hash_str` hook; returns 0 when none is installed —
/// gate with [`has_hash_str_hook`] (the [`try_hash_str`] wrapper does).
/// `ptr`/`len` must describe a valid WTF-8 byte range (a zero-length range
/// with a dangling-but-aligned `ptr` is fine).
#[majit_macros::dont_look_inside]
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub extern "C" fn hash_str_hooked(ptr: *const u8, len: usize) -> i64 {
    HASH_STR_HOOK.with(|cell| cell.get().map(|f| unsafe { f(ptr, len) }).unwrap_or(0))
}

/// [`hash_str_hooked`] over the rstr `STR` of a text — one word, as
/// `rstr.py ll_strhash` takes the STR.  The digest is the installed hook's
/// (Python-level `hash()`), not `STR.hash`.
#[majit_macros::dont_look_inside]
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "`key` is a live rstr `STR` the caller holds (one GC word, as `rstr.py ll_strhash` takes it)"
)]
pub extern "C" fn hash_str_hooked_str(key: *mut crate::unicodeobject::Utf8Str) -> i64 {
    let bytes = unsafe { crate::unicodeobject::utf8_payload_bytes(key) };
    hash_str_hooked(bytes.as_ptr(), bytes.len())
}

/// Invoke the installed `hash_str` hook over the STR key.  Returns `None` when no
/// hook is installed (pyre-object lib tests without the str hook, pre-init
/// snapshot tools); the str-keyed GET helpers then fall back to the
/// wrapping `object_key_for(w_str_from_storage(key))` path so behavior is preserved.
#[inline]
pub fn try_hash_str(key: *mut crate::unicodeobject::Utf8Str) -> Option<i64> {
    if has_hash_str_hook() {
        Some(hash_str_hooked_str(key))
    } else {
        None
    }
}

/// Diagnostic panic for the single-hash contract.  Every dict key is
/// hashed through the installed `hash_w` trampoline; there is no second,
/// structural hashing path (`dictmultiobject.py:1210
/// r_dict(space.eq_w, space.hash_w, force_non_null=True)`).  Production
/// installs the hook at boot (`pyre-jit::eval::init_jit_hooks`, before the
/// first statement); `#[cfg(test)]` modules install it per test thread.  A
/// `None` from [`try_hash_w`] at a key-construction site therefore means a
/// dict was built before the hook was registered (or on a thread that
/// never registered it) — a setup bug, not a recoverable condition.
#[cold]
#[inline(never)]
pub fn missing_hash_hook() -> ! {
    panic!(
        "dict key hashing requires the hash_w hook; register it via \
         register_hash_w_hook before constructing object/str-keyed dicts"
    );
}

/// Install the `compares_by_identity` callback for this thread.
/// Called from `pyre-jit::eval`'s init alongside the other
/// thread-local hooks.
pub fn register_compares_by_identity_hook(hook: ComparesByIdentityHookFn) {
    COMPARES_BY_IDENTITY_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the `compares_by_identity` callback on this thread.
pub fn clear_compares_by_identity_hook() {
    COMPARES_BY_IDENTITY_HOOK.with(|cell| cell.set(None));
}

/// Invoke the installed `compares_by_identity` hook.  Returns `None`
/// when no hook is installed (pyre-object lib tests, pre-init
/// snapshot tools); callers fall back to the conservative
/// `false` (i.e. presume `OVERRIDES_EQ_CMP_OR_HASH`).
///
/// # Safety
/// `w_type` must be a valid PyObjectRef pointing at a `W_TypeObject`
/// (null tolerated).
#[inline]
pub unsafe fn try_compares_by_identity(w_type: PyObjectRef) -> Option<bool> {
    if has_compares_by_identity_hook() {
        Some(compares_by_identity_hooked(w_type))
    } else {
        None
    }
}

/// True when a `compares_by_identity` hook is installed on this
/// thread.  `dont_look_inside`: thread-local read, residualizes via
/// the registered fnaddr.
#[majit_macros::dont_look_inside]
pub extern "C" fn has_compares_by_identity_hook() -> bool {
    // Materialise a datum no sibling carries, behind an optimisation barrier.
    // This predicate and [`has_eq_w_hook`] are both registered residual-call
    // targets (`jit_fnaddr.rs`), and `runtime_fnaddr_patch` re-pairs a
    // build-time address with the runtime one by that address alone, so two
    // bodies folded onto one address make the pairing ambiguous. Which pair a
    // folding linker picks moves with unrelated layout changes, so the bodies
    // must differ by construction; `drain_list_append` keeps its
    // `#[inline(never)]` forwarding call for the same reason.
    std::hint::black_box(4u32);
    COMPARES_BY_IDENTITY_HOOK.with(|cell| cell.get().is_some())
}

/// Invoke the installed `compares_by_identity` hook; returns `false`
/// when no hook is installed — gate with
/// [`has_compares_by_identity_hook`] (the
/// [`try_compares_by_identity`] wrapper does).  `w_type` must be a
/// valid PyObjectRef pointing at a `W_TypeObject` (null tolerated).
#[majit_macros::dont_look_inside]
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub extern "C" fn compares_by_identity_hooked(w_type: PyObjectRef) -> bool {
    COMPARES_BY_IDENTITY_HOOK.with(|cell| cell.get().map(|f| unsafe { f(w_type) }).unwrap_or(false))
}
