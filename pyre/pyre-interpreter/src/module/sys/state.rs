//! `sys` module interpreter-level state.
//!
//! PyPy stores `sys.recursionlimit` on the `sys` module instance itself
//! (`pypy/module/sys/moduledef.py self.recursionlimit = 1000`,
//! read/written through `space.sys.recursionlimit` in
//! `pypy/module/sys/vm.py:92-96`). Pyre is single-space so the module
//! singleton maps 1:1 to a static here, but the storage is scoped to
//! the `sys` module (not to the stack-check subsystem) so the data
//! structure lives in the same namespace as its upstream owner.

use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

use pyre_object::PyObjectRef;

/// Default recursion limit, matching CPython / PyPy. `Module.__init__`
/// sets `self.recursionlimit = 1000` in the PyPy tree
/// (`pypy/module/sys/moduledef.py`).
pub const DEFAULT_RECURSION_LIMIT: i32 = 1000;

/// Hard upper bound for `sys.setrecursionlimit`, matching the silent
/// clamp at `pypy/module/sys/vm.py`.
pub const MAX_RECURSION_LIMIT: i32 = 1_000_000;

/// `space.sys.recursionlimit` parity. `sys.setrecursionlimit` writes
/// here, `sys.getrecursionlimit` reads from here. The stack-check
/// subsystem (`crate::stack_check`) consults this value when the user
/// raises/lowers the budget, but otherwise keeps its own derived
/// byte-budget (`PYRE_STACKTOOBIG.length`) hot in L1.
///
/// Stored a word wide rather than as an `i32` so a compiled trace can read it
/// with the word-sized raw load it uses for the eval-breaker word, instead of
/// residualizing a call to reach it.  `set_recursion_limit` clamps its
/// argument at zero, so the stored value is always a non-negative `i32`.
static RECURSION_LIMIT: AtomicUsize = AtomicUsize::new(DEFAULT_RECURSION_LIMIT as usize);

/// Width of that word, in bytes.  The activation seam's load descriptor must
/// use exactly this size: a narrower load truncates the limit and a wider one
/// reads past it into the adjacent static, and either way the compiled
/// recursion answers to a bound the interpreter never set.
pub const RECURSION_LIMIT_WORD_SIZE: usize = std::mem::size_of::<AtomicUsize>();

/// PyPy `pypy/module/sys/system.py:15-18`.
pub const DEFAULT_MAX_STR_DIGITS: i32 = 4300;
pub const MAX_STR_DIGITS_THRESHOLD: i32 = 640;

/// `pypy/module/sys/state.py class State` — `space.fromcache(State)`.
/// `w_int_max_str_digits` is a field of this instance, not a module-level
/// atomic.  The atomic is the free-threaded stand-in for assigning the
/// field (`set_int_max_str_digits` writes `state.w_int_max_str_digits`).
pub struct SysState {
    int_max_str_digits: AtomicI32,
    lazy_imports_mode: AtomicI32,
    lazy_imports_filter: AtomicUsize,
    lazy_modules: AtomicUsize,
    lazy_pending_submodules: AtomicUsize,
}

impl SysState {
    pub const fn new() -> Self {
        Self {
            int_max_str_digits: AtomicI32::new(DEFAULT_MAX_STR_DIGITS),
            lazy_imports_mode: AtomicI32::new(0),
            lazy_imports_filter: AtomicUsize::new(0),
            lazy_modules: AtomicUsize::new(0),
            lazy_pending_submodules: AtomicUsize::new(0),
        }
    }

    pub fn walk_roots(&self, forward: &mut dyn FnMut(&mut pyre_object::PyObjectRef)) {
        walk_atomic_ptr(&self.lazy_imports_filter, forward);
        walk_atomic_ptr(&self.lazy_modules, forward);
        walk_atomic_ptr(&self.lazy_pending_submodules, forward);
    }
}

fn walk_atomic_ptr(slot: &AtomicUsize, forward: &mut dyn FnMut(&mut PyObjectRef)) {
    let mut p = slot.load(Ordering::Relaxed) as PyObjectRef;
    if !p.is_null() {
        forward(&mut p);
        slot.store(p as usize, Ordering::Relaxed);
    }
}

impl Default for SysState {
    fn default() -> Self {
        Self::new()
    }
}

/// `state.py get(space)` → the prebuilt space's State field.
fn sys_state() -> &'static SysState {
    crate::baseobjspace::object_space().sys_state()
}

/// `space.sys.recursionlimit` getter. Matches
/// `pypy/module/sys/vm.py getrecursionlimit return space.newint(space.sys.recursionlimit)`.
///
/// The value is mutable sys-module state, so tracing must read it at runtime
/// rather than bake the build process's atomic value into a JitCode.
#[majit_macros::dont_look_inside]
pub fn recursion_limit() -> i32 {
    RECURSION_LIMIT.load(Ordering::Relaxed) as i32
}

/// Address of the word [`recursion_limit`] reads, for the activation seam's
/// raw load.  Traces do not outlive the process, so the address a recording
/// observes is the one its compiled form runs against.
pub fn recursion_limit_addr() -> usize {
    std::ptr::addr_of!(RECURSION_LIMIT) as usize
}

/// `space.sys.recursionlimit = new_limit` parity
/// (`pypy/module/sys/vm.py:96`).
#[inline]
pub fn set_recursion_limit(new_limit: i32) {
    RECURSION_LIMIT.store(new_limit.max(0) as usize, Ordering::Relaxed);
}

#[inline]
pub fn int_max_str_digits() -> i32 {
    sys_state().int_max_str_digits.load(Ordering::Relaxed)
}

/// `pypy/module/sys/state.py:set_int_max_str_digits` validation.
pub fn set_int_max_str_digits(maxdigits: i32) -> Result<(), crate::PyError> {
    if maxdigits == 0 || maxdigits >= MAX_STR_DIGITS_THRESHOLD {
        sys_state()
            .int_max_str_digits
            .store(maxdigits, Ordering::Relaxed);
        Ok(())
    } else {
        Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            format!("maxdigits {maxdigits} must be 0 or larger than {MAX_STR_DIGITS_THRESHOLD}"),
        ))
    }
}

/// `PyImport_GetLazyImportsMode` / `PyImport_LAZY_NORMAL` / `LAZY_ALL`.
pub fn lazy_imports_mode() -> i32 {
    sys_state().lazy_imports_mode.load(Ordering::Relaxed)
}

pub fn set_lazy_imports_mode(mode: i32) {
    sys_state().lazy_imports_mode.store(mode, Ordering::Relaxed);
}

pub fn lazy_imports_filter() -> Option<PyObjectRef> {
    let p = sys_state().lazy_imports_filter.load(Ordering::Acquire) as PyObjectRef;
    if p.is_null() { None } else { Some(p) }
}

/// `PyImport_SetLazyImportsFilter`.
pub fn set_lazy_imports_filter(filter: PyObjectRef) -> Result<(), crate::PyError> {
    let stored = if filter.is_null() || unsafe { pyre_object::is_none(filter) } {
        0
    } else if !crate::baseobjspace::callable_w(filter) {
        return Err(crate::PyError::value_error(
            "filter provided but is not callable",
        ));
    } else {
        pyre_object::gc_roots::mark_prebuilt_roots_dirty();
        filter as usize
    };
    sys_state()
        .lazy_imports_filter
        .store(stored, Ordering::Release);
    Ok(())
}

pub fn lazy_modules() -> PyObjectRef {
    sys_state().lazy_modules.load(Ordering::Acquire) as PyObjectRef
}

pub fn set_lazy_modules(set: PyObjectRef) {
    sys_state()
        .lazy_modules
        .store(set as usize, Ordering::Release);
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
}

pub fn lazy_pending_submodules() -> PyObjectRef {
    sys_state().lazy_pending_submodules.load(Ordering::Acquire) as PyObjectRef
}

pub fn ensure_lazy_pending_submodules() -> PyObjectRef {
    let state = sys_state();
    let existing = state.lazy_pending_submodules.load(Ordering::Acquire) as PyObjectRef;
    if !existing.is_null() {
        return existing;
    }
    let d = pyre_object::w_dict_new();
    state
        .lazy_pending_submodules
        .store(d as usize, Ordering::Release);
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    d
}

#[cfg(test)]
mod tests {
    #[test]
    fn int_max_str_digits_lives_on_the_space_state() {
        crate::typedef::init_typeobjects();
        assert_eq!(super::int_max_str_digits(), super::DEFAULT_MAX_STR_DIGITS);
        super::set_int_max_str_digits(1000).unwrap();
        assert_eq!(super::int_max_str_digits(), 1000);
        super::set_int_max_str_digits(super::DEFAULT_MAX_STR_DIGITS).unwrap();
        assert!(super::set_int_max_str_digits(1).is_err());
    }
}

/// Reset to the default value. Used by unit tests that need a clean
/// recursion-limit state between runs.
#[cfg(test)]
pub fn reset_recursion_limit_for_tests() {
    RECURSION_LIMIT.store(DEFAULT_RECURSION_LIMIT as usize, Ordering::Relaxed);
}
