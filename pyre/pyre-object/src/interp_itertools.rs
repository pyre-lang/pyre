//! itertools module — iterator objects.
//!
//! PyPy: pypy/module/itertools/interp_itertools.py
//!
//! Line-by-line port of the W_Count / W_Repeat / W_Cycle / W_Chain etc.
//! classes. Each class becomes a `#[repr(C)]` struct with a static PyType.

use crate::pyobject::*;
use pyre_macros::pyre_class;

// W_Count.next_w's read/add/write is indivisible under PyPy's GIL.
// The stripe covers a mutator that has dropped the GIL (see objlock).
static COUNT_LOCKS: crate::objlock::LockFamily = crate::objlock::LockFamily::new();

pub type CountGuard = crate::objlock::ObjectLockGuard;

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
/// The guard is the zero word while `gil_ready` is unpublished.
pub unsafe fn w_count_lock(obj: PyObjectRef) -> CountGuard {
    let ready = crate::gil_ready::gil_ready_word();
    let lock = if ready != 0 {
        unsafe { w_count_lock_acquire(obj) }
    } else {
        0
    };
    unsafe { crate::objlock::ObjectLockGuard::from_word(lock) }
}

/// # Safety
/// `obj` must be a live `count`. The caller must release the returned handle
/// exactly once on the acquiring thread.
#[majit_macros::dont_look_inside]
pub unsafe fn w_count_lock_acquire(obj: PyObjectRef) -> usize {
    // A nursery object can move while its guard is held.  Stripe on its stable
    // class identity, not the movable instance address, so every operation on
    // one count continues to acquire the same lock after collection.
    COUNT_LOCKS.acquire(unsafe { crate::mapdict::instance_getclass(obj) } as usize)
}

pub fn count_locks_after_fork_child() {
    COUNT_LOCKS.after_fork_child();
}

// ── W_Count — pypy/module/itertools/interp_itertools.py:class W_Count ──
//
// ```python
// class W_Count(W_Root):
//     def __init__(self, space, w_firstval, w_step):
//         self.space = space
//         self.w_c = w_firstval
//         self.w_step = w_step
//
//     def iter_w(self):
//         return self
//
//     def next_w(self):
//         w_c = self.w_c
//         self.w_c = self.space.add(w_c, self.w_step)
//         return w_c
// ```
//
// The receiver stores `w_c` (current value) and `w_step` which are both
// PyObjectRef so that count(1.5, 0.5) works for float too.

#[pyre_class("itertools.count", type_id = 24, static_name = "COUNT", user_layout)]
pub struct W_Count {
    pub w_c: PyObjectRef,
    pub w_step: PyObjectRef,
}

pub fn w_count_new(
    w_firstval: PyObjectRef,
    w_step: PyObjectRef,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).
    let _roots = crate::gc_roots::push_roots();
    let w_firstval = crate::gc_roots::pin_root(w_firstval);
    let w_step = crate::gc_roots::pin_root(w_step);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_Count::allocate_instance(
        W_Count {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_c: w_firstval,
            w_step,
        },
        w_subtype,
    )
}

/// Check if an object is a `W_Count`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_count(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &COUNT_TYPE) || py_type_check(obj, &COUNT_USER_TYPE) }
}

/// Read the current `w_c` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_get_c(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Count)).w_c }
}

/// Write the current `w_c` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_set_c(obj: PyObjectRef, v: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Count)).w_c = v;
        crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    }
}

/// Read the `w_step` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_get_step(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Count)).w_step }
}

// ── W_Repeat — pypy/module/itertools/interp_itertools.py:class W_Repeat ──
//
// ```python
// class W_Repeat(W_Root):
//     def __init__(self, space, w_obj, w_times):
//         self.space = space
//         self.w_obj = w_obj
//         if w_times is None:
//             self.counting = False
//             self.count = 0
//         else:
//             self.counting = True
//             self.count = max(self.space.int_w(w_times), 0)
//
//     def next_w(self):
//         if self.counting:
//             if self.count <= 0:
//                 raise OperationError(self.space.w_StopIteration, self.space.w_None)
//             self.count -= 1
//         return self.w_obj
// ```

#[pyre_class("itertools.repeat", type_id = 25, static_name = "REPEAT", user_layout)]
pub struct W_Repeat {
    pub w_obj: PyObjectRef,
    pub counting: bool,
    pub count: i64,
}

pub fn w_repeat_new(
    w_obj: PyObjectRef,
    w_times: Option<i64>,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let (counting, count) = match w_times {
        None => (false, 0),
        Some(n) => (true, n.max(0)),
    };
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).
    let _roots = crate::gc_roots::push_roots();
    let w_obj = crate::gc_roots::pin_root(w_obj);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_Repeat::allocate_instance(
        W_Repeat {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_obj,
            counting,
            count,
        },
        w_subtype,
    )
}

/// Check if an object is a `W_Repeat`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_repeat(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &REPEAT_TYPE) || py_type_check(obj, &REPEAT_USER_TYPE) }
}

/// Read the `w_obj` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Repeat)).w_obj }
}

/// Read the `counting` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_counting(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_Repeat)).counting }
}

/// Read the `count` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_count(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_Repeat)).count }
}

/// Decrement the `count` field by 1.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_dec_count(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Repeat)).count -= 1;
    }
}

// ── W_TakeWhile — pypy/module/itertools/interp_itertools.py:class W_TakeWhile ──
//
// ```python
// class W_TakeWhile(W_Root):
//     def __init__(self, space, w_predicate, w_iterable):
//         self.space = space
//         self.w_predicate = w_predicate
//         self.w_iterable = space.iter(w_iterable)
//         self.stopped = False
// ```
//
// `next_w` lives in the interpreter (`baseobjspace::next`) because it
// calls the predicate.

#[pyre_class("itertools.takewhile", static_name = "TAKEWHILE", user_layout)]
pub struct W_TakeWhile {
    pub w_predicate: PyObjectRef,
    pub w_iterable: PyObjectRef,
    pub stopped: bool,
}

/// `w_iterable` must already be an iterator (`space.iter` applied by the
/// caller, matching `W_TakeWhile.__init__`).
pub fn w_takewhile_new(
    w_predicate: PyObjectRef,
    w_iterable: PyObjectRef,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).
    let _roots = crate::gc_roots::push_roots();
    let w_predicate = crate::gc_roots::pin_root(w_predicate);
    let w_iterable = crate::gc_roots::pin_root(w_iterable);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_TakeWhile::allocate_instance(
        W_TakeWhile {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_predicate,
            w_iterable,
            stopped: false,
        },
        w_subtype,
    )
}

/// Check if an object is a `W_TakeWhile`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_takewhile(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &TAKEWHILE_TYPE) || py_type_check(obj, &TAKEWHILE_USER_TYPE) }
}

// ── W_DropWhile — pypy/module/itertools/interp_itertools.py:class W_DropWhile ──
//
// ```python
// class W_DropWhile(W_Root):
//     def __init__(self, space, w_predicate, w_iterable):
//         self.space = space
//         self.w_predicate = w_predicate
//         self.w_iterable = space.iter(w_iterable)
//         self.started = False
// ```

#[pyre_class("itertools.dropwhile", static_name = "DROPWHILE", user_layout)]
pub struct W_DropWhile {
    pub w_predicate: PyObjectRef,
    pub w_iterable: PyObjectRef,
    pub started: bool,
}

/// `w_iterable` must already be an iterator (`space.iter` applied by the
/// caller, matching `W_DropWhile.__init__`).
pub fn w_dropwhile_new(
    w_predicate: PyObjectRef,
    w_iterable: PyObjectRef,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).
    let _roots = crate::gc_roots::push_roots();
    let w_predicate = crate::gc_roots::pin_root(w_predicate);
    let w_iterable = crate::gc_roots::pin_root(w_iterable);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_DropWhile::allocate_instance(
        W_DropWhile {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_predicate,
            w_iterable,
            started: false,
        },
        w_subtype,
    )
}

/// Check if an object is a `W_DropWhile`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_dropwhile(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &DROPWHILE_TYPE) || py_type_check(obj, &DROPWHILE_USER_TYPE) }
}

// ── W_FilterFalse — pypy/module/itertools/interp_itertools.py:class W_FilterFalse ──
//
// Subclass of `W_Filter` (`pypy/module/__builtin__/functional.py`)
// with `reverse = True`:
//
// ```python
// class W_Filter(W_Root):
//     reverse = False # set to True in itertools
//     def __init__(self, space, w_predicate, w_iterable):
//         self.space = space
//         if space.is_w(w_predicate, space.w_None):
//             self.w_predicate = None
//         else:
//             self.w_predicate = w_predicate
//         self.w_iterable = space.iter(w_iterable)
// ```
//
// `w_predicate` is PY_NULL when the Python-level predicate was None.

#[pyre_class("itertools.filterfalse", static_name = "FILTERFALSE", user_layout)]
pub struct W_FilterFalse {
    pub w_predicate: PyObjectRef,
    pub w_iterable: PyObjectRef,
}

/// `w_iterable` must already be an iterator; `w_predicate` is PY_NULL
/// for a None predicate (`W_Filter.__init__`).
pub fn w_filterfalse_new(
    w_predicate: PyObjectRef,
    w_iterable: PyObjectRef,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).
    let _roots = crate::gc_roots::push_roots();
    if !w_predicate.is_null() {
        let _ = crate::gc_roots::pin_root(w_predicate);
    }
    let w_iterable = crate::gc_roots::pin_root(w_iterable);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_FilterFalse::allocate_instance(
        W_FilterFalse {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_predicate,
            w_iterable,
        },
        w_subtype,
    )
}

/// Check if an object is a `W_FilterFalse`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_filterfalse(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FILTERFALSE_TYPE) || py_type_check(obj, &FILTERFALSE_USER_TYPE) }
}

// ── W_ISlice — pypy/module/itertools/interp_itertools.py:class W_ISlice ──
//
// ```python
// class W_ISlice(W_Root):
//     def __init__(self, space, w_iterable, w_startstop, args_w):
//         self.iterable = space.iter(w_iterable)
//         ...
//         self.count = rffi.cast(rffi.UNSIGNED, 0)
//         self.next = rffi.cast(rffi.UNSIGNED, start)
//         self.stop = stop
//         self.step = rffi.cast(rffi.UNSIGNED, step)
// ```
//
// A null `iterable` is PyPy's `self.iterable = None` exhausted sentinel.
// The remaining fields retain the exact unsigned/signed split used by the
// upstream object: count/next/step are `rffi.UNSIGNED`, while -1 in `stop`
// represents an unbounded slice.
#[pyre_class("itertools.islice", static_name = "ISLICE", user_layout)]
pub struct W_ISlice {
    pub iterable: PyObjectRef,
    pub count: usize,
    pub next: usize,
    pub stop: i64,
    pub step: usize,
}

/// `iterable` must already be an iterator (`W_ISlice.__init__` applies
/// `space.iter`).  All numeric arguments have already passed `arg_int_w`.
pub fn w_islice_new(
    iterable: PyObjectRef,
    start: i64,
    stop: i64,
    step: i64,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let iterable = crate::gc_roots::pin_root(iterable);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_ISlice::allocate_instance(
        W_ISlice {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            iterable,
            count: 0,
            next: start as usize,
            stop,
            step: step as usize,
        },
        w_subtype,
    )
}

/// Check if an object is a `W_ISlice`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_islice(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ISLICE_TYPE) || py_type_check(obj, &ISLICE_USER_TYPE) }
}

/// Clear the live source after exhaustion, matching
/// `W_ISlice.next_w/_ignore_items`.
///
/// # Safety
/// `obj` must be a valid `W_ISlice`.
#[inline]
pub unsafe fn w_islice_clear_iterable(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ISlice)).iterable = std::ptr::null_mut();
    }
}

// ── W_Batched — CPython 3.14 Modules/itertoolsmodule.c:batchedobject ──
//
// PyPy's current itertools port predates `batched`; Python 3.14 is the
// project's compatibility authority for this newer type.  Preserve the
// upstream object shape: one live iterator, a signed Py_ssize_t-sized batch
// count whose -1 value latches exhaustion, and the strict flag.
#[pyre_class("itertools.batched", static_name = "BATCHED", user_layout)]
pub struct W_Batched {
    pub it: PyObjectRef,
    pub batch_size: isize,
    pub strict: bool,
}

/// `it` must already be an iterator (`batched_new_impl` applies
/// `PyObject_GetIter` before allocating the instance).
pub fn w_batched_new(
    it: PyObjectRef,
    batch_size: isize,
    strict: bool,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let it = crate::gc_roots::pin_root(it);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_Batched::allocate_instance(
        W_Batched {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            it,
            batch_size,
            strict,
        },
        w_subtype,
    )
}

/// Check if an object is a `W_Batched`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_batched(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BATCHED_TYPE) || py_type_check(obj, &BATCHED_USER_TYPE) }
}

/// Latch the iterator as exhausted and release its source, matching
/// `batched_next`'s `batch_size = -1` / `Py_CLEAR(bo->it)` paths.
///
/// # Safety
/// `obj` must be a valid `W_Batched`.
#[inline]
pub unsafe fn w_batched_set_exhausted(obj: PyObjectRef) {
    unsafe {
        let batched = &mut *(obj as *mut W_Batched);
        batched.batch_size = -1;
        batched.it = std::ptr::null_mut();
    }
}

// ── W_Product — pypy/module/itertools/interp_itertools.py:W_Product ──
//
// ```python
// class W_Product(W_Root):
//     def __init__(self, space, args_w, w_repeat):
//         self.gears = [...]
//         self.indices = [0] * len(self.gears)
//         self.lst = None
//         self.stopped = False
// ```
//
// Each translated RPython list is represented by an owned W_ListObject.  The
// outer `gears` list owns list snapshots of the input pools; `indices` and
// `lst` preserve the upstream mutable per-iterator state.  A null `lst`
// represents PyPy's None sentinel before the first result and after rollover.
#[pyre_class("itertools.product", static_name = "PRODUCT", user_layout)]
pub struct W_Product {
    pub gears: PyObjectRef,
    pub indices: PyObjectRef,
    pub lst: PyObjectRef,
    pub stopped: bool,
}

pub fn w_product_new(
    gears: PyObjectRef,
    indices: PyObjectRef,
    stopped: bool,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let gears = crate::gc_roots::pin_root(gears);
    let indices = crate::gc_roots::pin_root(indices);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_Product::allocate_instance(
        W_Product {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            gears,
            indices,
            lst: std::ptr::null_mut(),
            stopped,
        },
        w_subtype,
    )
}

/// Check if an object is a `W_Product`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_product(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &PRODUCT_TYPE) || py_type_check(obj, &PRODUCT_USER_TYPE) }
}

// ── W_Combinations — PyPy interp_itertools.py:W_Combinations ───────
//
// `pool_w`, `indices`, and `last_result_w` are the translated equivalents of
// PyPy's three list attributes.  A null last_result_w is the initial None
// sentinel.  The signed machine-word `r` matches RPython `int`.
#[pyre_class("itertools.combinations", static_name = "COMBINATIONS", user_layout)]
pub struct W_Combinations {
    pub pool_w: PyObjectRef,
    pub indices: PyObjectRef,
    pub r: isize,
    pub last_result_w: PyObjectRef,
    pub stopped: bool,
}

pub fn w_combinations_new(
    pool_w: PyObjectRef,
    indices: PyObjectRef,
    r: isize,
    stopped: bool,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let pool_w = crate::gc_roots::pin_root(pool_w);
    let indices = crate::gc_roots::pin_root(indices);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_Combinations::allocate_instance(
        W_Combinations {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            pool_w,
            indices,
            r,
            last_result_w: std::ptr::null_mut(),
            stopped,
        },
        w_subtype,
    )
}

/// Check if an object is a `W_Combinations`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_combinations(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &COMBINATIONS_TYPE) || py_type_check(obj, &COMBINATIONS_USER_TYPE) }
}

// ── W_CombinationsWithReplacement — PyPy interp_itertools.py ───────
//
// PyPy subclasses W_Combinations and inherits the same five attributes.
// Rust's concrete GC layouts are flat, so repeat those fields in the same
// order while the interpreter shares the inherited next-state algorithm.
#[pyre_class(
    "itertools.combinations_with_replacement",
    static_name = "COMBINATIONS_WITH_REPLACEMENT",
    user_layout
)]
pub struct W_CombinationsWithReplacement {
    pub pool_w: PyObjectRef,
    pub indices: PyObjectRef,
    pub r: isize,
    pub last_result_w: PyObjectRef,
    pub stopped: bool,
}

pub fn w_combinations_with_replacement_new(
    pool_w: PyObjectRef,
    indices: PyObjectRef,
    r: isize,
    stopped: bool,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let pool_w = crate::gc_roots::pin_root(pool_w);
    let indices = crate::gc_roots::pin_root(indices);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_CombinationsWithReplacement::allocate_instance(
        W_CombinationsWithReplacement {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            pool_w,
            indices,
            r,
            last_result_w: std::ptr::null_mut(),
            stopped,
        },
        w_subtype,
    )
}

/// Check if an object is a `W_CombinationsWithReplacement`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_combinations_with_replacement(obj: PyObjectRef) -> bool {
    unsafe {
        py_type_check(obj, &COMBINATIONS_WITH_REPLACEMENT_TYPE)
            || py_type_check(obj, &COMBINATIONS_WITH_REPLACEMENT_USER_TYPE)
    }
}

// ── W_Permutations — PyPy interp_itertools.py:W_Permutations ───────
//
// `pool_w`, `indices`, and `cycles` are the translated RPython list
// attributes.  PyPy leaves indices/cycles unset when r > len(pool_w), which
// is represented by null list pointers while `stopped` is true.
#[pyre_class("itertools.permutations", static_name = "PERMUTATIONS", user_layout)]
pub struct W_Permutations {
    pub pool_w: PyObjectRef,
    pub r: isize,
    pub stopped: bool,
    pub raised_stop_iteration: bool,
    pub indices: PyObjectRef,
    pub cycles: PyObjectRef,
    pub started: bool,
}

pub fn w_permutations_new(
    pool_w: PyObjectRef,
    r: isize,
    stopped: bool,
    indices: PyObjectRef,
    cycles: PyObjectRef,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let pool_w = crate::gc_roots::pin_root(pool_w);
    if !indices.is_null() {
        let _ = crate::gc_roots::pin_root(indices);
    }
    if !cycles.is_null() {
        let _ = crate::gc_roots::pin_root(cycles);
    }
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_Permutations::allocate_instance(
        W_Permutations {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            pool_w,
            r,
            stopped,
            raised_stop_iteration: stopped,
            indices,
            cycles,
            started: false,
        },
        w_subtype,
    )
}

/// Check if an object is a `W_Permutations`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_permutations(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &PERMUTATIONS_TYPE) || py_type_check(obj, &PERMUTATIONS_USER_TYPE) }
}

// ── W_GroupBy / W_GroupByIterator — PyPy interp_itertools.py ───────
//
// The parent owns the live iterator and shared key/value cursor.  A strong
// `w_currgrouper` edge mirrors PyPy's attribute and invalidates an older group
// by identity whenever the parent advances.
#[pyre_class("itertools.groupby", static_name = "GROUPBY", user_layout)]
pub struct W_GroupBy {
    pub w_iterator: PyObjectRef,
    pub w_keyfunc: PyObjectRef,
    pub w_tgtkey: PyObjectRef,
    pub w_currkey: PyObjectRef,
    pub w_currvalue: PyObjectRef,
    pub w_currgrouper: PyObjectRef,
}

pub fn w_groupby_new(
    w_iterator: PyObjectRef,
    w_keyfunc: PyObjectRef,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let w_iterator = crate::gc_roots::pin_root(w_iterator);
    let w_keyfunc = crate::gc_roots::pin_root(w_keyfunc);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_GroupBy::allocate_instance(
        W_GroupBy {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_iterator,
            w_keyfunc,
            w_tgtkey: std::ptr::null_mut(),
            w_currkey: std::ptr::null_mut(),
            w_currvalue: std::ptr::null_mut(),
            w_currgrouper: std::ptr::null_mut(),
        },
        w_subtype,
    )
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_groupby(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &GROUPBY_TYPE) || py_type_check(obj, &GROUPBY_USER_TYPE) }
}

#[pyre_class("itertools._grouper", static_name = "GROUPBY_ITERATOR")]
pub struct W_GroupByIterator {
    pub groupby: PyObjectRef,
    pub w_tgtkey: PyObjectRef,
}

pub fn w_groupby_iterator_new(groupby: PyObjectRef, w_tgtkey: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let groupby = crate::gc_roots::pin_root(groupby);
    let w_tgtkey = crate::gc_roots::pin_root(w_tgtkey);
    let grouper = W_GroupByIterator::allocate_stable(W_GroupByIterator {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        groupby,
        w_tgtkey,
    });
    unsafe {
        (*(groupby as *mut W_GroupBy)).w_currgrouper = grouper;
        crate::gc_hook::try_gc_write_barrier(groupby as crate::gc_hook::GCREF);
    }
    grouper
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_groupby_iterator(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &GROUPBY_ITERATOR_TYPE) }
}

// ── W_TeeChainedListNode / W_TeeIterable — PyPy interp_itertools.py ─
//
// Each shared node owns one cached item and the next node.  Every tee copy
// holds its own cursor into this same chain while sharing the source iterator.
#[pyre_class(
    "itertools._tee_dataobject",
    static_name = "TEE_DATAOBJECT",
    weakrefable
)]
pub struct W_TeeChainedListNode {
    pub w_next: PyObjectRef,
    pub w_obj: PyObjectRef,
    pub running: bool,
}

pub fn w_tee_chained_list_node_new() -> PyObjectRef {
    W_TeeChainedListNode::allocate_stable(W_TeeChainedListNode {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_next: std::ptr::null_mut(),
        w_obj: std::ptr::null_mut(),
        running: false,
        lifeline: PY_NULL,
    })
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_tee_dataobject(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &TEE_DATAOBJECT_TYPE) }
}

#[pyre_class("itertools._tee", static_name = "TEE_ITERABLE", weakrefable)]
pub struct W_TeeIterable {
    pub w_iterator: PyObjectRef,
    pub w_chained_list: PyObjectRef,
}

pub fn w_tee_iterable_new(w_iterator: PyObjectRef, w_chained_list: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let w_iterator = crate::gc_roots::pin_root(w_iterator);
    if !w_chained_list.is_null() {
        let _ = crate::gc_roots::pin_root(w_chained_list);
    }
    W_TeeIterable::allocate_stable(W_TeeIterable {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_iterator,
        w_chained_list,
        lifeline: PY_NULL,
    })
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_tee_iterable(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &TEE_ITERABLE_TYPE) }
}

// ── W_Compress — pypy/module/itertools/interp_itertools.py:W_Compress ──
//
// ```python
// class W_Compress(W_Root):
//     def __init__(self, space, w_data, w_selectors):
//         self.space = space
//         self.w_data = space.iter(w_data)
//         self.w_selectors = space.iter(w_selectors)
// ```
//
// `next_w` lives in the interpreter because it invokes both iterators and
// selector truth testing.  The two live iterators are traced GC edges.

#[pyre_class("itertools.compress", static_name = "COMPRESS", user_layout)]
pub struct W_Compress {
    pub w_data: PyObjectRef,
    pub w_selectors: PyObjectRef,
}

/// Both arguments must already have had `space.iter` applied, matching
/// `W_Compress.__init__`.
pub fn w_compress_new(
    w_data: PyObjectRef,
    w_selectors: PyObjectRef,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let w_data = crate::gc_roots::pin_root(w_data);
    let w_selectors = crate::gc_roots::pin_root(w_selectors);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_Compress::allocate_instance(
        W_Compress {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_data,
            w_selectors,
        },
        w_subtype,
    )
}

/// Check if an object is a `W_Compress`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_compress(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &COMPRESS_TYPE) || py_type_check(obj, &COMPRESS_USER_TYPE) }
}

// ── W_StarMap — pypy/module/itertools/interp_itertools.py:W_StarMap ──
//
// ```python
// class W_StarMap(W_Root):
//     def __init__(self, space, w_fun, w_iterable):
//         self.space = space
//         self.w_fun = w_fun
//         self.w_iterable = self.space.iter(w_iterable)
// ```
//
// `next_w` lives in the interpreter because it expands the next object into
// call arguments and invokes `w_fun`.  Both fields are traced GC edges.

#[pyre_class("itertools.starmap", static_name = "STARMAP", user_layout)]
pub struct W_StarMap {
    pub w_fun: PyObjectRef,
    pub w_iterable: PyObjectRef,
}

/// `w_iterable` must already have had `space.iter` applied.
pub fn w_starmap_new(
    w_fun: PyObjectRef,
    w_iterable: PyObjectRef,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let w_fun = crate::gc_roots::pin_root(w_fun);
    let w_iterable = crate::gc_roots::pin_root(w_iterable);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_StarMap::allocate_instance(
        W_StarMap {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_fun,
            w_iterable,
        },
        w_subtype,
    )
}

/// Check if an object is a `W_StarMap`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_starmap(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &STARMAP_TYPE) || py_type_check(obj, &STARMAP_USER_TYPE) }
}

// ── W_Accumulate — pypy/module/itertools/interp_itertools.py:W_Accumulate ──
//
// PyPy keeps the live iterator, optional binary function, running total, and
// pending initial value on the iterator object.  PY_NULL represents PyPy's
// internal `None` sentinel for `w_func` and `w_total`; `w_initial` is the
// Python-level value and is reset to Python None after it is yielded.

#[pyre_class("itertools.accumulate", static_name = "ACCUMULATE", user_layout)]
pub struct W_Accumulate {
    pub w_iterable: PyObjectRef,
    pub w_func: PyObjectRef,
    pub w_total: PyObjectRef,
    pub w_initial: PyObjectRef,
}

pub fn w_accumulate_new(
    w_iterable: PyObjectRef,
    w_func: PyObjectRef,
    w_initial: PyObjectRef,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let w_iterable = crate::gc_roots::pin_root(w_iterable);
    if !w_func.is_null() {
        let _ = crate::gc_roots::pin_root(w_func);
    }
    let w_initial = crate::gc_roots::pin_root(w_initial);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_Accumulate::allocate_instance(
        W_Accumulate {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_iterable,
            w_func,
            w_total: std::ptr::null_mut(),
            w_initial,
        },
        w_subtype,
    )
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_accumulate(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ACCUMULATE_TYPE) || py_type_check(obj, &ACCUMULATE_USER_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_accumulate_set_total(obj: PyObjectRef, w_value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Accumulate)).w_total = w_value;
        crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_accumulate_set_initial(obj: PyObjectRef, w_value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Accumulate)).w_initial = w_value;
        crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    }
}

// ── W_ZipLongest — pypy/module/itertools/interp_itertools.py:W_ZipLongest ──

#[pyre_class("itertools.zip_longest", static_name = "ZIP_LONGEST", user_layout)]
pub struct W_ZipLongest {
    /// A Python list of live iterators; exhausted entries become Python None.
    pub w_iterators: PyObjectRef,
    pub w_fillvalue: PyObjectRef,
    pub active: i64,
}

pub fn w_zip_longest_new(
    w_iterators: PyObjectRef,
    w_fillvalue: PyObjectRef,
    active: i64,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let w_iterators = crate::gc_roots::pin_root(w_iterators);
    let w_fillvalue = crate::gc_roots::pin_root(w_fillvalue);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_ZipLongest::allocate_instance(
        W_ZipLongest {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_iterators,
            w_fillvalue,
            active,
        },
        w_subtype,
    )
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_zip_longest(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ZIP_LONGEST_TYPE) || py_type_check(obj, &ZIP_LONGEST_USER_TYPE) }
}

// ── W_Pairwise — pypy/module/itertools/interp_itertools.py:class W_Pairwise ──
//
// ```python
// class W_Pairwise(W_Root):
//     def __init__(self, space, w_iterator):
//         self.space = space
//         self.w_iterator = w_iterator
//         self.w_prev = None
// ```
//
// `w_prev` is PY_NULL until the first `next_w`.

#[pyre_class("itertools.pairwise", static_name = "PAIRWISE", user_layout)]
pub struct W_Pairwise {
    pub w_iterator: PyObjectRef,
    pub w_prev: PyObjectRef,
}

/// `w_iterator` must already be an iterator (`W_Pairwise__new__` applies
/// `space.iter`).
pub fn w_pairwise_new(w_iterator: PyObjectRef, w_subtype: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).
    let _roots = crate::gc_roots::push_roots();
    let w_iterator = crate::gc_roots::pin_root(w_iterator);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_Pairwise::allocate_instance(
        W_Pairwise {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_iterator,
            w_prev: std::ptr::null_mut(),
        },
        w_subtype,
    )
}

/// Check if an object is a `W_Pairwise`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_pairwise(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &PAIRWISE_TYPE) || py_type_check(obj, &PAIRWISE_USER_TYPE) }
}

/// Read the source iterator.
///
/// # Safety
/// `obj` must point to a valid `W_Pairwise`.
#[inline]
pub unsafe fn w_pairwise_get_iterator(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Pairwise)).w_iterator }
}

/// Read the previous item (`PY_NULL` before the first result).
///
/// # Safety
/// `obj` must point to a valid `W_Pairwise`.
#[inline]
pub unsafe fn w_pairwise_get_prev(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Pairwise)).w_prev }
}

/// Store the previous item. `W_Pairwise` is allocated in old-gen, while an
/// iterator may return a nursery object, so this is an old-to-young store.
///
/// # Safety
/// `obj` must point to a valid `W_Pairwise`.
#[inline]
pub unsafe fn w_pairwise_set_prev(obj: PyObjectRef, w_prev: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Pairwise)).w_prev = w_prev;
        crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    }
}

// ── W_Cycle — pypy/module/itertools/interp_itertools.py:class W_Cycle ──
//
// ```python
// class W_Cycle(W_Root):
//     def __init__(self, space, w_iterable):
//         self.space = space
//         self.saved_w = []
//         self.w_iterable = space.iter(w_iterable)
//         self.index = 0    # 0 during the first iteration; > 0 afterwards
// ```
//
// `next_w` (in `baseobjspace::next`) pulls from `w_iterable` on the first
// pass, appending each element to `saved`; once the source is exhausted it
// replays `saved` forever.  `index` is 0 during the first pass and > 0 once
// cycling.  Unlike the predicate iterators (whose referents are rooted by
// the live predicate/iterable), `saved` is owned solely by the W_Cycle, so
// the GC must trace it — the type is registered in the JIT GC driver
// (`pyre-jit/src/eval.rs`) via `register_pyre_class` in AUTO-ID mode.
#[pyre_class("itertools.cycle", static_name = "CYCLE", user_layout)]
pub struct W_Cycle {
    pub w_iterable: PyObjectRef,
    pub saved: PyObjectRef,
    pub index: i64,
}

/// `w_iterable` must already be an iterator (`cycle`'s registrar applies
/// `space.iter`).  Allocates an empty `saved` list.
pub fn w_cycle_new(w_iterable: PyObjectRef, w_subtype: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).  Minting
    // `saved` allocates, and a self-iterating `list` subclass reaches here as
    // a movable header, so `w_iterable` is read back out of its root slot
    // rather than from the parameter copy that collection left pre-move.
    // `w_subtype` is pinned across that same allocation (`objspace.py`
    // `allocate_instance`).
    let roots = crate::gc_roots::push_roots();
    let iterable_slot = roots.base();
    let _ = roots.pin_root(w_iterable);
    let subtype_slot = iterable_slot + 1;
    let _ = roots.pin_root(w_subtype);
    let saved = crate::listobject::w_list_new(Vec::new());
    let saved = roots.pin_root(saved);
    W_Cycle::allocate_instance(
        W_Cycle {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_iterable: roots.get(iterable_slot),
            saved,
            index: 0,
        },
        roots.get(subtype_slot),
    )
}

/// Check if an object is a `W_Cycle`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_cycle(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &CYCLE_TYPE) || py_type_check(obj, &CYCLE_USER_TYPE) }
}

// ── W_Chain — pypy/module/itertools/interp_itertools.py:class W_Chain ──
//
// ```python
// class W_Chain(W_Root):
//     def __init__(self, space, w_iterables):
//         self.space = space
//         self.w_iterables = w_iterables
//         self.w_it = None
//
//     def _advance(self):
//         self.w_it = self.space.iter(self.space.next(self.w_iterables))
//
//     def next_w(self):
//         if not self.w_it:
//             self._advance()     # may raise StopIteration
//         while True:
//             try:
//                 return self.space.next(self.w_it)
//             except OperationError as e:
//                 if e.match(self.space, self.space.w_StopIteration):
//                     self.w_it = None
//                     self._advance()
//                 else:
//                     raise
// ```
//
// `w_iterables` is an iterator over the source iterables; `w_it` is the
// current active sub-iterator (PY_NULL until the first `next_w`, and reset
// to PY_NULL each time a sub-iterator is exhausted).  Both pointer fields
// are owned solely by the W_Chain, so the GC must trace them — the type is
// registered in the JIT GC driver (`pyre-jit/src/eval.rs`) via
// `register_pyre_class` in AUTO-ID mode.
#[pyre_class("itertools.chain", static_name = "CHAIN", user_layout)]
pub struct W_Chain {
    pub w_iterables: PyObjectRef,
    pub w_it: PyObjectRef,
}

/// `w_iterables` must already be an iterator over the source iterables
/// (`chain` / `chain.from_iterable` apply `space.iter`).  `w_it` starts
/// PY_NULL (no active sub-iterator yet).
pub fn w_chain_new(w_iterables: PyObjectRef, w_subtype: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).
    let _roots = crate::gc_roots::push_roots();
    let w_iterables = crate::gc_roots::pin_root(w_iterables);
    let w_subtype = crate::gc_roots::pin_root(w_subtype);
    W_Chain::allocate_instance(
        W_Chain {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_iterables,
            w_it: std::ptr::null_mut(),
        },
        w_subtype,
    )
}

/// Check if an object is a `W_Chain`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_chain(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &CHAIN_TYPE) || py_type_check(obj, &CHAIN_USER_TYPE) }
}

/// Read the `w_iterables` field of a `W_Chain`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `W_Chain`.
#[inline]
pub unsafe fn w_chain_get_iterables(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Chain)).w_iterables }
}

/// Read the `w_it` field of a `W_Chain`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `W_Chain`.
#[inline]
pub unsafe fn w_chain_get_it(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Chain)).w_it }
}

/// Store the `w_iterables` field of a `W_Chain`.  Reassigning a pointer
/// field can record an old→young edge, so the GC write barrier runs.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `W_Chain`.
#[inline]
pub unsafe fn w_chain_set_iterables(obj: PyObjectRef, w_value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Chain)).w_iterables = w_value;
        crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    }
}

/// Store the `w_it` field of a `W_Chain`.  Reassigning a pointer field can
/// record an old→young edge, so the GC write barrier runs.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `W_Chain`.
#[inline]
pub unsafe fn w_chain_set_it(obj: PyObjectRef, w_value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Chain)).w_it = w_value;
        crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_count_gc_type_id_matches_descr() {
        assert_eq!(W_COUNT_GC_TYPE_ID, 24);
        assert_eq!(
            <W_Count as crate::lltype::GcType>::type_id(),
            W_COUNT_GC_TYPE_ID
        );
        assert_eq!(
            <W_Count as crate::lltype::GcType>::SIZE,
            W_COUNT_OBJECT_SIZE
        );
    }

    #[test]
    fn w_repeat_gc_type_id_matches_descr() {
        assert_eq!(W_REPEAT_GC_TYPE_ID, 25);
        assert_eq!(
            <W_Repeat as crate::lltype::GcType>::type_id(),
            W_REPEAT_GC_TYPE_ID
        );
        assert_eq!(
            <W_Repeat as crate::lltype::GcType>::SIZE,
            W_REPEAT_OBJECT_SIZE
        );
    }

    #[test]
    fn w_takewhile_object_size_matches_descr() {
        // Auto-id `allocate_stable` (GC-managed): tid assigned at JIT init.
        assert_eq!(
            <W_TakeWhile as crate::lltype::GcType>::SIZE,
            W_TAKEWHILE_OBJECT_SIZE
        );
    }

    #[test]
    fn w_dropwhile_object_size_matches_descr() {
        // Auto-id `allocate_stable` (GC-managed): tid assigned at JIT init.
        assert_eq!(
            <W_DropWhile as crate::lltype::GcType>::SIZE,
            W_DROPWHILE_OBJECT_SIZE
        );
    }

    #[test]
    fn w_filterfalse_object_size_matches_descr() {
        // Auto-id `allocate_stable` (GC-managed): tid assigned at JIT init.
        assert_eq!(
            <W_FilterFalse as crate::lltype::GcType>::SIZE,
            W_FILTERFALSE_OBJECT_SIZE
        );
    }

    #[test]
    fn w_islice_gc_descriptor_traces_source_iterator() {
        assert_eq!(
            W_ISLICE_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_ISlice, ob.w_class),
                std::mem::offset_of!(W_ISlice, iterable),
            ]
        );
        assert_eq!(
            <W_ISlice as crate::lltype::GcType>::SIZE,
            W_ISLICE_OBJECT_SIZE
        );
    }

    #[test]
    fn w_batched_gc_descriptor_traces_source_iterator() {
        assert_eq!(
            W_BATCHED_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_Batched, ob.w_class),
                std::mem::offset_of!(W_Batched, it),
            ]
        );
        assert_eq!(
            <W_Batched as crate::lltype::GcType>::SIZE,
            W_BATCHED_OBJECT_SIZE
        );
    }

    #[test]
    fn w_product_gc_descriptor_traces_all_owned_lists() {
        assert_eq!(
            W_PRODUCT_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_Product, ob.w_class),
                std::mem::offset_of!(W_Product, gears),
                std::mem::offset_of!(W_Product, indices),
                std::mem::offset_of!(W_Product, lst),
            ]
        );
        assert_eq!(
            <W_Product as crate::lltype::GcType>::SIZE,
            W_PRODUCT_OBJECT_SIZE
        );
    }

    #[test]
    fn w_combinations_gc_descriptor_traces_all_owned_lists() {
        assert_eq!(
            W_COMBINATIONS_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_Combinations, ob.w_class),
                std::mem::offset_of!(W_Combinations, pool_w),
                std::mem::offset_of!(W_Combinations, indices),
                std::mem::offset_of!(W_Combinations, last_result_w),
            ]
        );
        assert_eq!(
            <W_Combinations as crate::lltype::GcType>::SIZE,
            W_COMBINATIONS_OBJECT_SIZE
        );
    }

    #[test]
    fn w_combinations_with_replacement_gc_descriptor_traces_all_owned_lists() {
        assert_eq!(
            W_COMBINATIONS_WITH_REPLACEMENT_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_CombinationsWithReplacement, ob.w_class),
                std::mem::offset_of!(W_CombinationsWithReplacement, pool_w),
                std::mem::offset_of!(W_CombinationsWithReplacement, indices),
                std::mem::offset_of!(W_CombinationsWithReplacement, last_result_w),
            ]
        );
        assert_eq!(
            <W_CombinationsWithReplacement as crate::lltype::GcType>::SIZE,
            W_COMBINATIONS_WITH_REPLACEMENT_OBJECT_SIZE
        );
    }

    #[test]
    fn w_permutations_gc_descriptor_traces_all_owned_lists() {
        assert_eq!(
            W_PERMUTATIONS_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_Permutations, ob.w_class),
                std::mem::offset_of!(W_Permutations, pool_w),
                std::mem::offset_of!(W_Permutations, indices),
                std::mem::offset_of!(W_Permutations, cycles),
            ]
        );
        assert_eq!(
            <W_Permutations as crate::lltype::GcType>::SIZE,
            W_PERMUTATIONS_OBJECT_SIZE
        );
    }

    #[test]
    fn w_groupby_gc_descriptor_traces_shared_cursor() {
        assert_eq!(
            W_GROUPBY_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_GroupBy, ob.w_class),
                std::mem::offset_of!(W_GroupBy, w_iterator),
                std::mem::offset_of!(W_GroupBy, w_keyfunc),
                std::mem::offset_of!(W_GroupBy, w_tgtkey),
                std::mem::offset_of!(W_GroupBy, w_currkey),
                std::mem::offset_of!(W_GroupBy, w_currvalue),
                std::mem::offset_of!(W_GroupBy, w_currgrouper),
            ]
        );
        assert_eq!(
            <W_GroupBy as crate::lltype::GcType>::SIZE,
            W_GROUPBY_OBJECT_SIZE
        );
    }

    #[test]
    fn w_groupby_iterator_gc_descriptor_traces_parent_and_key() {
        assert_eq!(
            W_GROUPBY_ITERATOR_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_GroupByIterator, ob.w_class),
                std::mem::offset_of!(W_GroupByIterator, groupby),
                std::mem::offset_of!(W_GroupByIterator, w_tgtkey),
            ]
        );
        assert_eq!(
            <W_GroupByIterator as crate::lltype::GcType>::SIZE,
            W_GROUPBY_ITERATOR_OBJECT_SIZE
        );
    }

    #[test]
    fn w_tee_dataobject_gc_descriptor_traces_item_and_next() {
        assert_eq!(
            W_TEE_DATAOBJECT_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_TeeChainedListNode, ob.w_class),
                std::mem::offset_of!(W_TeeChainedListNode, w_next),
                std::mem::offset_of!(W_TeeChainedListNode, w_obj),
                std::mem::offset_of!(W_TeeChainedListNode, lifeline),
            ]
        );
        assert_eq!(
            <W_TeeChainedListNode as crate::lltype::GcType>::SIZE,
            W_TEE_DATAOBJECT_OBJECT_SIZE
        );
    }

    #[test]
    fn w_tee_iterable_gc_descriptor_traces_source_and_cursor() {
        assert_eq!(
            W_TEE_ITERABLE_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_TeeIterable, ob.w_class),
                std::mem::offset_of!(W_TeeIterable, w_iterator),
                std::mem::offset_of!(W_TeeIterable, w_chained_list),
                std::mem::offset_of!(W_TeeIterable, lifeline),
            ]
        );
        assert_eq!(
            <W_TeeIterable as crate::lltype::GcType>::SIZE,
            W_TEE_ITERABLE_OBJECT_SIZE
        );
    }

    #[test]
    fn w_compress_gc_descriptor_traces_both_iterator_fields() {
        assert_eq!(W_COMPRESS_GC_PTR_OFFSETS.len(), 3);
        assert_eq!(
            W_COMPRESS_GC_PTR_OFFSETS[0],
            std::mem::offset_of!(W_Compress, ob.w_class)
        );
        assert_eq!(
            W_COMPRESS_GC_PTR_OFFSETS[1],
            std::mem::offset_of!(W_Compress, w_data)
        );
        assert_eq!(
            W_COMPRESS_GC_PTR_OFFSETS[2],
            std::mem::offset_of!(W_Compress, w_selectors)
        );
        assert_eq!(
            <W_Compress as crate::lltype::GcType>::SIZE,
            W_COMPRESS_OBJECT_SIZE
        );
    }

    #[test]
    fn w_starmap_gc_descriptor_traces_function_and_iterator() {
        assert_eq!(W_STARMAP_GC_PTR_OFFSETS.len(), 3);
        assert_eq!(
            W_STARMAP_GC_PTR_OFFSETS[0],
            std::mem::offset_of!(W_StarMap, ob.w_class)
        );
        assert_eq!(
            W_STARMAP_GC_PTR_OFFSETS[1],
            std::mem::offset_of!(W_StarMap, w_fun)
        );
        assert_eq!(
            W_STARMAP_GC_PTR_OFFSETS[2],
            std::mem::offset_of!(W_StarMap, w_iterable)
        );
        assert_eq!(
            <W_StarMap as crate::lltype::GcType>::SIZE,
            W_STARMAP_OBJECT_SIZE
        );
    }

    #[test]
    fn w_accumulate_gc_descriptor_traces_live_state() {
        assert_eq!(W_ACCUMULATE_GC_PTR_OFFSETS.len(), 5);
        assert_eq!(
            W_ACCUMULATE_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_Accumulate, ob.w_class),
                std::mem::offset_of!(W_Accumulate, w_iterable),
                std::mem::offset_of!(W_Accumulate, w_func),
                std::mem::offset_of!(W_Accumulate, w_total),
                std::mem::offset_of!(W_Accumulate, w_initial),
            ]
        );
        assert_eq!(
            <W_Accumulate as crate::lltype::GcType>::SIZE,
            W_ACCUMULATE_OBJECT_SIZE
        );
    }

    #[test]
    fn w_zip_longest_gc_descriptor_traces_iterators_and_fillvalue() {
        assert_eq!(W_ZIP_LONGEST_GC_PTR_OFFSETS.len(), 3);
        assert_eq!(
            W_ZIP_LONGEST_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_ZipLongest, ob.w_class),
                std::mem::offset_of!(W_ZipLongest, w_iterators),
                std::mem::offset_of!(W_ZipLongest, w_fillvalue),
            ]
        );
        assert_eq!(
            <W_ZipLongest as crate::lltype::GcType>::SIZE,
            W_ZIP_LONGEST_OBJECT_SIZE
        );
    }

    #[test]
    fn w_pairwise_gc_descriptor_traces_iterator_and_previous_item() {
        assert_eq!(
            W_PAIRWISE_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_Pairwise, ob.w_class),
                std::mem::offset_of!(W_Pairwise, w_iterator),
                std::mem::offset_of!(W_Pairwise, w_prev),
            ]
        );
        assert_eq!(
            <W_Pairwise as crate::lltype::GcType>::SIZE,
            W_PAIRWISE_OBJECT_SIZE
        );
    }

    // W_Cycle is registered in AUTO-ID mode (no `type_id = N`), so the GC
    // tid is stamped at JIT-driver init rather than asserted against a
    // constant.  What must hold is that both traced edges — the source
    // `w_iterable` and the owned `saved` replay buffer — are reported to
    // the collector, and that the descriptor reflects the struct's size.
    #[test]
    fn w_cycle_gc_descriptor_traces_both_pointer_fields() {
        assert_eq!(W_CYCLE_GC_PTR_OFFSETS.len(), 3);
        assert_eq!(
            W_CYCLE_GC_PTR_OFFSETS[0],
            std::mem::offset_of!(W_Cycle, ob.w_class)
        );
        assert_eq!(
            W_CYCLE_GC_PTR_OFFSETS[1],
            std::mem::offset_of!(W_Cycle, w_iterable)
        );
        assert_eq!(
            W_CYCLE_GC_PTR_OFFSETS[2],
            std::mem::offset_of!(W_Cycle, saved)
        );
        assert_eq!(
            <W_Cycle as crate::lltype::GcType>::SIZE,
            W_CYCLE_OBJECT_SIZE
        );
    }

    // W_Chain is registered in AUTO-ID mode (no `type_id = N`), so the GC
    // tid is stamped at JIT-driver init rather than asserted against a
    // constant.  What must hold is that both traced edges — the source
    // `w_iterables` iterator and the current sub-iterator `w_it` — are
    // reported to the collector, and that the descriptor reflects the
    // struct's size.
    #[test]
    fn w_chain_gc_descriptor_traces_both_pointer_fields() {
        assert_eq!(W_CHAIN_GC_PTR_OFFSETS.len(), 3);
        assert_eq!(
            W_CHAIN_GC_PTR_OFFSETS[0],
            std::mem::offset_of!(W_Chain, ob.w_class)
        );
        assert_eq!(
            W_CHAIN_GC_PTR_OFFSETS[1],
            std::mem::offset_of!(W_Chain, w_iterables)
        );
        assert_eq!(
            W_CHAIN_GC_PTR_OFFSETS[2],
            std::mem::offset_of!(W_Chain, w_it)
        );
        assert_eq!(
            <W_Chain as crate::lltype::GcType>::SIZE,
            W_CHAIN_OBJECT_SIZE
        );
    }
}
