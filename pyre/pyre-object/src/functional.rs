//! `pypy/module/__builtin__/functional.py` line-by-line ports for built-in iterator functionals.

use crate::pyobject::*;
use majit_rlib::rbigint::{RBigInt as BigInt, RBigIntGcRoot, live_rbigint};
use pyre_macros::pyre_class;

// ── functional.rs ─────────────────────────────────────────────

// `pypy/module/__builtin__/functional.py W_Enumerate` line-by-line port.
//
// ```python
// class W_Enumerate(W_Root):
//     def __init__(self, w_iter_or_list, start, w_start):
//         self.w_iter_or_list = w_iter_or_list
//         self.index = start
//         self.w_index = w_start
//     ...
// ```
//
// `w_iter_or_list` is either the source iterator (general case) OR
// the source list itself when the start is zero and the source is an exact
// list. `w_enumerate_new` preserves that representation; reduce and iteration
// code must therefore handle both the stored-list and stored-iterator forms.
//
// `index: i64` is the fast counter; once it overflows i64, `w_index`
// carries the bigint value (PyPy line 297-303
// `space.add(w_index, space.newint(1))` after `rarithmetic.ovfcheck`).

#[pyre_class("enumerate", static_name = "ENUMERATE", user_layout)]
pub struct W_Enumerate {
    /// `functional.py:225 self.w_iter_or_list` — either the source
    /// iterator (general case) or the source list itself
    /// (start == 0 + exact-list source).  When the iterator is
    /// exhausted, set to `PY_NULL` per `:294-295`.
    pub w_iter_or_list: PyObjectRef,
    /// `functional.py:226 self.index` — i64 fast-path counter.  When
    /// negative, indicates the bigint slot below is active.
    pub index: i64,
    /// `functional.py:227 self.w_index` — bigint counter activated
    /// after i64 overflow per `:298-301`.  `PY_NULL` when the i64
    /// fast path is still active.
    pub w_index: PyObjectRef,
}

/// Allocate a `W_Enumerate`.  Mirrors `functional.py:222-227 __init__`.
/// `w_index` may be `PY_NULL` (i64 fast-path) or a bigint
/// `PyObjectRef` (overflow path).
pub fn w_enumerate_new(
    w_iter_or_list: PyObjectRef,
    start: i64,
    w_index: PyObjectRef,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::pin_roots(&[w_iter_or_list, w_index, w_subtype]);
    W_Enumerate::allocate_instance(
        W_Enumerate {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_iter_or_list: crate::gc_roots::shadow_stack_get(base),
            index: start,
            w_index: crate::gc_roots::shadow_stack_get(base + 1),
        },
        crate::gc_roots::shadow_stack_get(base + 2),
    )
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_enumerate(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ENUMERATE_TYPE) || py_type_check(obj, &ENUMERATE_USER_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `W_Enumerate`.
#[inline]
pub unsafe fn w_enumerate_get_iter_or_list(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Enumerate)).w_iter_or_list }
}

/// # Safety
/// `obj` must point to a valid `W_Enumerate`.
#[inline]
pub unsafe fn w_enumerate_set_iter_or_list(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Enumerate)).w_iter_or_list = value;
        crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    }
}

/// # Safety
/// `obj` must point to a valid `W_Enumerate`.
#[inline]
pub unsafe fn w_enumerate_get_index(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_Enumerate)).index }
}

/// # Safety
/// `obj` must point to a valid `W_Enumerate`.
#[inline]
pub unsafe fn w_enumerate_set_index(obj: PyObjectRef, value: i64) {
    unsafe {
        (*(obj as *mut W_Enumerate)).index = value;
    }
}

/// # Safety
/// `obj` must point to a valid `W_Enumerate`.
#[inline]
pub unsafe fn w_enumerate_get_w_index(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Enumerate)).w_index }
}

/// # Safety
/// `obj` must point to a valid `W_Enumerate`.
#[inline]
pub unsafe fn w_enumerate_set_w_index(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Enumerate)).w_index = value;
        crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    }
}

#[cfg(test)]
mod enumerate_tests {
    use super::*;

    #[test]
    fn w_enumerate_object_size_matches_descr() {
        // Auto-id `allocate_stable` (GC-managed): the tid is assigned at JIT
        // init, so there is no compile-time `W_ENUMERATE_GC_TYPE_ID` to check.
        assert_eq!(
            <W_Enumerate as crate::lltype::GcType>::SIZE,
            W_ENUMERATE_OBJECT_SIZE
        );
    }
}

// ── functional.rs ─────────────────────────────────────────────

// `pypy/module/__builtin__/functional.py W_ReversedIterator`
// line-by-line port.
//
// ```python
// class W_ReversedIterator(W_Root):
//     def __init__(self, space, w_sequence):
//         self.remaining = space.len_w(w_sequence) - 1
//         if not space.issequence_w(w_sequence):
//             raise oefmt(space.w_TypeError, ...)
//         self.w_sequence = w_sequence
// ```
//
// A lazy reverse iterator over a sequence: `descr_next` does
// `getitem(w_sequence, remaining)` then decrements `remaining`.  When
// exhausted, `w_sequence` is dropped to `PY_NULL` and `remaining` to
// `-1` (`:392-393`, `:403-404`).  This replaces the earlier eager
// materialisation into a `seq_iter`, restoring the lazy CPython 3.14 /
// PyPy `reversed` object whose `__reduce__` is
// `(reversed, (sequence,), remaining)`.

#[pyre_class("reversed", static_name = "REVERSED", user_layout)]
pub struct W_ReversedIterator {
    /// `functional.py:359 self.w_sequence` — the source sequence; set to
    /// `PY_NULL` once the iterator is exhausted (`:393`, `:404`).
    pub w_sequence: PyObjectRef,
    /// `functional.py self.remaining` — index of the next element to
    /// yield, counting down from `len(seq) - 1`; `-1` once exhausted.
    pub remaining: i64,
}

/// Allocate a `W_ReversedIterator`.  Mirrors `functional.py
/// __init__` with `remaining` already computed as `len(seq) - 1` by the
/// caller.
pub fn w_reversed_new(
    w_sequence: PyObjectRef,
    remaining: i64,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::pin_roots(&[w_sequence, w_subtype]);
    W_ReversedIterator::allocate_instance(
        W_ReversedIterator {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_sequence: crate::gc_roots::shadow_stack_get(base),
            remaining,
        },
        crate::gc_roots::shadow_stack_get(base + 1),
    )
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_reversed(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &REVERSED_TYPE) || py_type_check(obj, &REVERSED_USER_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `W_ReversedIterator`.
#[inline]
pub unsafe fn w_reversed_get_sequence(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ReversedIterator)).w_sequence }
}

/// # Safety
/// `obj` must point to a valid `W_ReversedIterator`.
#[inline]
pub unsafe fn w_reversed_set_sequence(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ReversedIterator)).w_sequence = value;
        crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    }
}

/// # Safety
/// `obj` must point to a valid `W_ReversedIterator`.
#[inline]
pub unsafe fn w_reversed_get_remaining(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_ReversedIterator)).remaining }
}

/// # Safety
/// `obj` must point to a valid `W_ReversedIterator`.
#[inline]
pub unsafe fn w_reversed_set_remaining(obj: PyObjectRef, value: i64) {
    unsafe {
        (*(obj as *mut W_ReversedIterator)).remaining = value;
    }
}

// ── functional.rs ─────────────────────────────────────────────

// `pypy/module/__builtin__/functional.py W_Map` line-by-line port,
// extended with the CPython 3.14 `strict` keyword (mirrors `zip`).
//
// ```python
// class W_Map(W_Root):
//     def __init__(self, space, w_fun, args_w):
//         self.w_fun = w_fun
//         self.iterators_w = build_iterators_from_args(space, args_w)
// ```
//
// A lazy map: `descr_next` pulls one item from each sub-iterator, then
// `call(w_fun, *items)`.  Stops when the shortest sub-iterator is
// exhausted; in `strict` mode a length mismatch raises `ValueError`.  This
// replaces the earlier eager materialisation into a `seq_iter`.

#[pyre_class("map", static_name = "MAP", user_layout)]
pub struct W_Map {
    /// `functional.py:843 self.w_fun` — the mapped callable.
    pub w_fun: PyObjectRef,
    /// `functional.py:844 self.iterators_w` — a `list` of sub-iterators, one
    /// per input iterable (`build_iterators_from_args`).
    pub w_iterators: PyObjectRef,
    /// CPython 3.14 `strict` flag; `descr_setstate` toggles it.
    pub strict: bool,
}

/// Allocate a `W_Map`.  `w_iterators` is a `list` of already-built
/// iterators (`build_iterators_from_args`).
pub fn w_map_new(
    w_fun: PyObjectRef,
    w_iterators: PyObjectRef,
    strict: bool,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::pin_roots(&[w_fun, w_iterators, w_subtype]);
    W_Map::allocate_instance(
        W_Map {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_fun: crate::gc_roots::shadow_stack_get(base),
            w_iterators: crate::gc_roots::shadow_stack_get(base + 1),
            strict,
        },
        crate::gc_roots::shadow_stack_get(base + 2),
    )
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_map(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &MAP_TYPE) || py_type_check(obj, &MAP_USER_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `W_Map`.
#[inline]
pub unsafe fn w_map_get_fun(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Map)).w_fun }
}

/// # Safety
/// `obj` must point to a valid `W_Map`.
#[inline]
pub unsafe fn w_map_get_iterators(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Map)).w_iterators }
}

/// # Safety
/// `obj` must point to a valid `W_Map`.
#[inline]
pub unsafe fn w_map_get_strict(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_Map)).strict }
}

/// # Safety
/// `obj` must point to a valid `W_Map`.
#[inline]
pub unsafe fn w_map_set_strict(obj: PyObjectRef, value: bool) {
    unsafe {
        (*(obj as *mut W_Map)).strict = value;
    }
}

/// `functional.py W_Map.iter_w` — `return self`.
///
/// # Safety
/// `obj` must point to a valid `W_Map`.
pub unsafe fn w_map_iter(obj: PyObjectRef) -> PyObjectRef {
    obj
}

// ── functional.rs ─────────────────────────────────────────────

// `pypy/module/__builtin__/functional.py W_Filter` line-by-line
// port.
//
// ```python
// class W_Filter(W_Root):
//     def __init__(self, space, w_predicate, w_iterable):
//         if space.is_w(w_predicate, space.w_None):
//             self.w_predicate = None
//         else:
//             self.w_predicate = w_predicate
//         self.w_iterable = space.iter(w_iterable)
// ```
//
// A lazy filter: `descr_next` pulls from `w_iterable` until the predicate
// (or truthiness, when the predicate is `None`) passes.  This replaces the
// earlier eager materialisation into a `seq_iter`, restoring the lazy
// `filter` object whose `__reduce__` is `(filter, (predicate, iterable))`.

#[pyre_class("filter", static_name = "FILTER", user_layout)]
pub struct W_Filter {
    /// `functional.py:921-924 self.w_predicate` — the predicate callable, or
    /// `PY_NULL` when the Python-level predicate was `None`.
    pub w_predicate: PyObjectRef,
    /// `functional.py:925 self.w_iterable` — the source iterator
    /// (`space.iter(w_iterable)`).
    pub w_iterable: PyObjectRef,
}

/// Allocate a `W_Filter`.  `w_iterable` must already be an iterator;
/// `w_predicate` is `PY_NULL` for a `None` predicate (`__init__`).
pub fn w_filter_new(
    w_predicate: PyObjectRef,
    w_iterable: PyObjectRef,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::pin_roots(&[w_predicate, w_iterable, w_subtype]);
    W_Filter::allocate_instance(
        W_Filter {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_predicate: crate::gc_roots::shadow_stack_get(base),
            w_iterable: crate::gc_roots::shadow_stack_get(base + 1),
        },
        crate::gc_roots::shadow_stack_get(base + 2),
    )
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_filter(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FILTER_TYPE) || py_type_check(obj, &FILTER_USER_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `W_Filter`.
#[inline]
pub unsafe fn w_filter_get_predicate(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Filter)).w_predicate }
}

/// # Safety
/// `obj` must point to a valid `W_Filter`.
#[inline]
pub unsafe fn w_filter_get_iterable(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Filter)).w_iterable }
}

/// `functional.py W_Filter.iter_w` — `return self`.
///
/// # Safety
/// `obj` must point to a valid `W_Filter`.
pub unsafe fn w_filter_iter(obj: PyObjectRef) -> PyObjectRef {
    obj
}

// ── functional.rs ─────────────────────────────────────────────

// `pypy/module/__builtin__/functional.py W_Zip` line-by-line port.
//
// ```python
// class W_Zip(W_Root):
//     def __init__(self, space, args_w, strict=False):
//         self.strict = strict
//         self.iterators_w = build_iterators_from_args(space, args_w)
// ```
//
// A lazy zip: `descr_next` pulls one item from each sub-iterator and
// returns the tuple, stopping when the shortest is exhausted; in `strict`
// mode a length mismatch raises `ValueError`.  `descr_setstate` toggles
// `strict`.  This replaces the earlier eager materialisation into a
// `seq_iter`.

#[pyre_class("zip", static_name = "ZIP", user_layout)]
pub struct W_Zip {
    /// `functional.py:1016 self.iterators_w` — a `list` of sub-iterators, one
    /// per input iterable (`build_iterators_from_args`).
    pub w_iterators: PyObjectRef,
    /// `functional.py self.strict`; `descr_setstate` toggles it.
    pub strict: bool,
    /// `functional.py:1017 self._iteration_progress` — number of iterators
    /// already consumed in the current tuple, used by strict mismatch
    /// reporting when a later iterator stops.
    pub iteration_progress: usize,
}

/// `functional.py W_Zip._immutable_fields_ = ["w_fun", "iterators_w", "strict"]`.
/// `w_fun` is the W_Map leftover; W_Zip stores `iterators_w` and `strict`.
/// `#[jit_immutable_fields]` cannot wrap `#[pyre_class]` (both consume the
/// item), so this is the harvest marker `harvest_immutable_fields_from_llbcs`
/// reads (`_immutable_fields_<Struct>`).
#[doc(hidden)]
#[allow(non_upper_case_globals, dead_code)]
pub const _immutable_fields_W_Zip: &str = "w_iterators,strict";

/// Allocate a `W_Zip`.  `w_iterators` is a `list` of already-built
/// iterators (`build_iterators_from_args`).
pub fn w_zip_new(w_iterators: PyObjectRef, strict: bool, w_subtype: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::pin_roots(&[w_iterators, w_subtype]);
    W_Zip::allocate_instance(
        W_Zip {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_iterators: crate::gc_roots::shadow_stack_get(base),
            strict,
            iteration_progress: 0,
        },
        crate::gc_roots::shadow_stack_get(base + 1),
    )
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_zip(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ZIP_TYPE) || py_type_check(obj, &ZIP_USER_TYPE) }
}

/// `functional.py W_Zip.iter_w` — `return self`.
///
/// # Safety
/// `obj` must point to a valid `W_Zip`.
pub unsafe fn w_zip_iter(obj: PyObjectRef) -> PyObjectRef {
    obj
}

/// # Safety
/// `obj` must point to a valid `W_Zip`.
#[inline]
pub unsafe fn w_zip_get_iterators(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Zip)).w_iterators }
}

/// # Safety
/// `obj` must point to a valid `W_Zip`.
#[inline]
pub unsafe fn w_zip_get_strict(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_Zip)).strict }
}

/// # Safety
/// `obj` must point to a valid `W_Zip`.
#[inline]
pub unsafe fn w_zip_set_strict(obj: PyObjectRef, value: bool) {
    unsafe {
        (*(obj as *mut W_Zip)).strict = value;
    }
}

/// # Safety
/// `obj` must point to a valid `W_Zip`.
#[inline]
pub unsafe fn w_zip_set_iteration_progress(obj: PyObjectRef, value: usize) {
    unsafe {
        (*(obj as *mut W_Zip)).iteration_progress = value;
    }
}

/// # Safety
/// `obj` must point to a valid `W_Zip`.
#[inline]
pub unsafe fn w_zip_get_iteration_progress(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_Zip)).iteration_progress }
}
/// Machine-int range iterator object.
///
/// Layout: `[ob_type | current: i64 | remaining: i64 | step: i64]`
/// The JIT reads `current` and `remaining` via `GetfieldGcI` and writes
/// both slots via `SetfieldGcI` when advancing the loop counter, matching
/// `pypy/module/__builtin__/functional.py W_IntRangeIterator`.
#[pyre_class("range_iterator", type_id = 6, static_name = "RANGE_ITER")]
pub struct W_IntRangeIterator {
    pub current: i64,
    pub remaining: i64,
    pub step: i64,
}

/// Field offsets of inline scalar slots — consumed by JIT field-access
/// IR (`pyre-jit/src/jit/codewriter.rs` GetfieldGcI / SetfieldGcI).
/// The macro's auto-generated `W_RANGE_ITER_GC_PTR_OFFSETS` holds only the
/// header `w_class` edge here (no PyObjectRef payload field) and does not
/// depend on these.
pub const RANGE_ITER_CURRENT_OFFSET: usize = std::mem::offset_of!(W_IntRangeIterator, current);
pub const RANGE_ITER_REMAINING_OFFSET: usize = std::mem::offset_of!(W_IntRangeIterator, remaining);
pub const RANGE_ITER_STEP_OFFSET: usize = std::mem::offset_of!(W_IntRangeIterator, step);

/// Allocate a new `W_IntRangeIterator` on the heap.
pub fn w_range_iter_new(current: i64, remaining: i64, step: i64) -> PyObjectRef {
    debug_assert!(remaining >= 0);
    W_IntRangeIterator::allocate(W_IntRangeIterator {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        current,
        remaining,
        step,
    })
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_range_iter_new(current: i64, remaining: i64, step: i64) -> PyObjectRef {
    w_range_iter_new(current, remaining, step)
}

/// Advance the range iterator and return the next value, or `None` if exhausted.
///
/// # Safety
/// `obj` must point to a valid `W_IntRangeIterator`.
pub unsafe fn w_range_iter_next(obj: PyObjectRef) -> Option<PyObjectRef> {
    unsafe {
        if is_range_iter_one_arg(obj) {
            let v = w_range_iter_one_arg_next(obj);
            return if v.is_null() { None } else { Some(v) };
        }
        if is_range_iter_step_one(obj) {
            let v = w_range_iter_step_one_next(obj);
            return if v.is_null() { None } else { Some(v) };
        }
        let iter = obj as *mut W_IntRangeIterator;
        if (*iter).remaining > 0 {
            let current = (*iter).current;
            let step = (*iter).step;
            (*iter).current = current + step;
            (*iter).remaining -= 1;
            Some(crate::intobject::w_int_new(current))
        } else {
            None
        }
    }
}

/// Check whether a range iterator has another element without advancing it.
///
/// # Safety
/// `obj` must point to a valid `W_IntRangeIterator`.
pub unsafe fn w_range_iter_has_next(obj: PyObjectRef) -> bool {
    unsafe { w_range_iter_remaining(obj) > 0 }
}

/// `functional.py W_IntRangeStepOneIterator` — the shape `descr_iter` picks
/// when the `range` was built without a step argument and does not start at
/// zero.
///
/// Layout: `[ob_type | current: i64 | stop: i64 | start: i64]`
/// `stop` is the sole `_immutable_fields_` entry, so a compiled loop hoists
/// the bound and the per-iteration cursor work is one load, one add and one
/// store — `step` is the constant 1 and `remaining` is derived as
/// `stop - current`.
#[pyre_class("range_iterator", type_id = 156, static_name = "RANGE_ITER_STEP_ONE")]
pub struct W_IntRangeStepOneIterator {
    pub current: i64,
    pub stop: i64,
    pub start: i64,
}

/// `functional.py W_IntRangeStepOneIterator.next`
///
/// # Safety
/// `obj` must point to a valid `W_IntRangeStepOneIterator`.
#[inline(never)]
pub unsafe fn w_range_iter_step_one_next(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let iter = obj as *mut W_IntRangeStepOneIterator;
        let current = (*iter).current;
        if current < (*iter).stop {
            (*iter).current = current + 1;
            return crate::intobject::w_int_new(current);
        }
        PY_NULL
    }
}

/// `functional.py W_IntRangeOneArgIterator` — the shape for `range(stop)`.
/// Its values are always `>= 0`, which is the property the JIT reads off the
/// class.
///
/// Layout: `[ob_type | current: i64 | stop: i64]`
#[pyre_class("range_iterator", type_id = 157, static_name = "RANGE_ITER_ONE_ARG")]
pub struct W_IntRangeOneArgIterator {
    pub current: i64,
    pub stop: i64,
}

/// `functional.py W_IntRangeOneArgIterator.next`
///
/// # Safety
/// `obj` must point to a valid `W_IntRangeOneArgIterator`.
#[inline(never)]
pub unsafe fn w_range_iter_one_arg_next(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let iter = obj as *mut W_IntRangeOneArgIterator;
        let current = (*iter).current;
        if current < (*iter).stop {
            // Advance before allocating: `w_int_new` can collect, and the
            // cursor write must not land through a stale pointer.
            (*iter).current = current + 1;
            return crate::intobject::w_int_new(current);
        }
        PY_NULL
    }
}

pub const RANGE_ITER_STEP_ONE_CURRENT_OFFSET: usize =
    std::mem::offset_of!(W_IntRangeStepOneIterator, current);
pub const RANGE_ITER_STEP_ONE_STOP_OFFSET: usize =
    std::mem::offset_of!(W_IntRangeStepOneIterator, stop);
pub const RANGE_ITER_STEP_ONE_START_OFFSET: usize =
    std::mem::offset_of!(W_IntRangeStepOneIterator, start);
pub const RANGE_ITER_ONE_ARG_CURRENT_OFFSET: usize =
    std::mem::offset_of!(W_IntRangeOneArgIterator, current);
pub const RANGE_ITER_ONE_ARG_STOP_OFFSET: usize =
    std::mem::offset_of!(W_IntRangeOneArgIterator, stop);

/// Allocate a `W_IntRangeStepOneIterator` positioned at `start`.
///
/// The header splits the two identities the way the specialised tuples do:
/// `ob_type` is this shape's own static, which is what `GuardClass` compares
/// and what lets a trace reach `stop` at a fixed offset, while `w_class` is
/// the `range_iterator` tag, so all three machine-int shapes are one Python
/// type.
pub fn w_range_iter_step_one_new(start: i64, stop: i64) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let full = W_IntRangeStepOneIterator {
        ob: PyObject {
            ob_type: &RANGE_ITER_STEP_ONE_TYPE as *const PyType,
            w_class: get_instantiate(&RANGE_ITER_TYPE),
        },
        current: start,
        stop,
        start,
    };
    crate::lltype::malloc_typed(full) as PyObjectRef
}

/// Allocate a `W_IntRangeOneArgIterator` positioned at zero.
pub fn w_range_iter_one_arg_new(stop: i64) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let full = W_IntRangeOneArgIterator {
        ob: PyObject {
            ob_type: &RANGE_ITER_ONE_ARG_TYPE as *const PyType,
            w_class: get_instantiate(&RANGE_ITER_TYPE),
        },
        current: 0,
        stop,
    };
    crate::lltype::malloc_typed(full) as PyObjectRef
}

/// Whether `obj` is the `step == 1` shape.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_range_iter_step_one(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &RANGE_ITER_STEP_ONE_TYPE) }
}

/// Whether `obj` is the `range(stop)` shape.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_range_iter_one_arg(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &RANGE_ITER_ONE_ARG_TYPE) }
}

/// Whether `obj` is the general three-field shape, and not one of the two
/// `step == 1` specialisations.  The JIT needs this exact answer because the
/// three layouts put different fields at the same offsets; interpreter code
/// wants [`is_range_iter`], which answers for all three.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_range_iter_general(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &RANGE_ITER_TYPE) }
}

/// Check if an object is a range iterator, in any of the three machine-int
/// shapes — the `isinstance(w_obj, W_IntRangeIterator)` answer.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_range_iter(obj: PyObjectRef) -> bool {
    unsafe {
        is_range_iter_general(obj) || is_range_iter_step_one(obj) || is_range_iter_one_arg(obj)
    }
}

/// Read the `(current, remaining, step)` triple of a range iterator.
///
/// # Safety
/// `obj` must point to a valid `W_IntRangeIterator`.
pub unsafe fn w_range_iter_fields(obj: PyObjectRef) -> (i64, i64, i64) {
    unsafe {
        if is_range_iter_one_arg(obj) {
            let iter = obj as *const W_IntRangeOneArgIterator;
            return ((*iter).current, (*iter).stop - (*iter).current, 1);
        }
        if is_range_iter_step_one(obj) {
            let iter = obj as *const W_IntRangeStepOneIterator;
            return ((*iter).current, (*iter).stop - (*iter).current, 1);
        }
        let iter = obj as *const W_IntRangeIterator;
        ((*iter).current, (*iter).remaining, (*iter).step)
    }
}

/// Count of elements not yet produced by a range iterator.
///
/// # Safety
/// `obj` must point to a valid `W_IntRangeIterator`.
pub unsafe fn w_range_iter_remaining(obj: PyObjectRef) -> i64 {
    unsafe {
        if is_range_iter_one_arg(obj) {
            let iter = obj as *const W_IntRangeOneArgIterator;
            return (*iter).stop - (*iter).current;
        }
        if is_range_iter_step_one(obj) {
            let iter = obj as *const W_IntRangeStepOneIterator;
            return (*iter).stop - (*iter).current;
        }
        let iter = obj as *const W_IntRangeIterator;
        (*iter).remaining
    }
}

/// Restore the machine range iterator's live cursor fields.
///
/// # Safety
/// `obj` must point to a valid `W_IntRangeIterator`; `remaining` must be
/// non-negative and `current` must be the value yielded next.
pub unsafe fn w_range_iter_set_cursor(obj: PyObjectRef, current: i64, remaining: i64) {
    unsafe {
        // The two `step == 1` shapes derive `remaining` from `stop`, which is
        // immutable, so the cursor is the whole state they carry.
        if is_range_iter_one_arg(obj) {
            (*(obj as *mut W_IntRangeOneArgIterator)).current = current;
            return;
        }
        if is_range_iter_step_one(obj) {
            (*(obj as *mut W_IntRangeStepOneIterator)).current = current;
            return;
        }
        let iter = obj as *mut W_IntRangeIterator;
        (*iter).current = current;
        (*iter).remaining = remaining;
    }
}

#[cfg(test)]
mod range_iter_tests {
    use super::*;
    use crate::intobject::w_int_get_value;

    #[test]
    fn test_range_iter_basic() {
        let iter = w_range_iter_new(0, 3, 1);
        unsafe {
            assert!(is_range_iter(iter));
            assert!(!is_int(iter));

            let v0 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v0), 0);

            let v1 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v1), 1);

            let v2 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v2), 2);

            assert!(w_range_iter_next(iter).is_none());
        }
    }

    #[test]
    fn test_range_iter_start_stop() {
        let iter = w_range_iter_new(5, 3, 1);
        unsafe {
            let v0 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v0), 5);

            let v1 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v1), 6);

            let v2 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v2), 7);

            assert!(w_range_iter_next(iter).is_none());
        }
    }

    #[test]
    fn test_range_iter_negative_step() {
        let iter = w_range_iter_new(5, 3, -1);
        unsafe {
            let v0 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v0), 5);

            let v1 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v1), 4);

            let v2 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v2), 3);

            assert!(w_range_iter_next(iter).is_none());
        }
    }

    #[test]
    fn test_range_iter_empty() {
        let iter = w_range_iter_new(5, 0, 1);
        unsafe {
            assert!(!w_range_iter_has_next(iter));
            assert!(w_range_iter_next(iter).is_none());
        }
    }

    #[test]
    fn test_range_iter_has_next_is_pure_probe() {
        let iter = w_range_iter_new(0, 2, 1);
        unsafe {
            assert!(w_range_iter_has_next(iter));
            assert!(w_range_iter_has_next(iter));
            let v0 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v0), 0);
        }
    }

    #[test]
    fn test_range_iter_field_offsets() {
        assert_eq!(RANGE_ITER_CURRENT_OFFSET, 16);
        assert_eq!(RANGE_ITER_REMAINING_OFFSET, 24);
        assert_eq!(RANGE_ITER_STEP_OFFSET, 32);
    }

    #[test]
    fn test_range_iter_one_arg_walks_and_reports_itself_as_a_range_iter() {
        let iter = w_range_iter_one_arg_new(3);
        unsafe {
            assert!(is_range_iter(iter));
            assert!(is_range_iter_one_arg(iter));
            assert!(!is_range_iter_general(iter));
            assert_eq!(w_range_iter_fields(iter), (0, 3, 1));

            for expected in 0..3 {
                assert!(w_range_iter_has_next(iter));
                let item = w_range_iter_next(iter).unwrap();
                assert_eq!(w_int_get_value(item), expected);
            }
            assert!(!w_range_iter_has_next(iter));
            assert!(w_range_iter_next(iter).is_none());
            assert_eq!(w_range_iter_remaining(iter), 0);
        }
    }

    #[test]
    fn test_range_iter_step_one_walks_from_its_start() {
        let iter = w_range_iter_step_one_new(5, 8);
        unsafe {
            assert!(is_range_iter(iter));
            assert!(is_range_iter_step_one(iter));
            assert_eq!(w_range_iter_fields(iter), (5, 3, 1));

            for expected in 5..8 {
                let item = w_range_iter_next(iter).unwrap();
                assert_eq!(w_int_get_value(item), expected);
            }
            assert!(w_range_iter_next(iter).is_none());
        }
    }

    #[test]
    fn test_range_iter_set_cursor_moves_a_step_one_shape() {
        let iter = w_range_iter_step_one_new(0, 10);
        unsafe {
            // The remaining count is derived from the immutable `stop`, so the
            // cursor is the whole state a restore has to write.
            w_range_iter_set_cursor(iter, 7, 3);
            assert_eq!(w_range_iter_fields(iter), (7, 3, 1));
            let item = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(item), 7);
        }
    }

    #[test]
    fn test_promote_step_picks_the_iterator_shape() {
        unsafe {
            // `range(4)` — no step argument, starting at zero.
            let one_arg = w_range_iter(w_range_from_i64(0, 4, 1, true));
            assert!(is_range_iter_one_arg(one_arg));

            // `range(2, 6)` — no step argument, but not from zero.
            let step_one = w_range_iter(w_range_from_i64(2, 6, 1, true));
            assert!(is_range_iter_step_one(step_one));

            // `range(0, 4, 1)` — the step is one, but it was spelled out, so
            // the promotion does not apply.
            let general = w_range_iter(w_range_new_i64(0, 4, 1));
            assert!(is_range_iter_general(general));
        }
    }

    #[test]
    fn test_empty_promoted_range_iter_remaining_is_non_negative() {
        unsafe {
            // `range(-1)` — one-arg promotion, raw stop is negative.
            let one_arg = w_range_iter(w_range_from_i64(0, -1, 1, true));
            assert!(is_range_iter_one_arg(one_arg));
            assert_eq!(w_range_iter_remaining(one_arg), 0);
            assert_eq!(w_range_iter_fields(one_arg), (0, 0, 1));
            assert!(!w_range_iter_has_next(one_arg));

            // `range(3, 1)` — step-one promotion, raw stop sits below start.
            let step_one = w_range_iter(w_range_from_i64(3, 1, 1, true));
            assert!(is_range_iter_step_one(step_one));
            assert_eq!(w_range_iter_remaining(step_one), 0);
            assert_eq!(w_range_iter_fields(step_one), (3, 0, 1));
            assert!(!w_range_iter_has_next(step_one));
        }
    }
}

// ── Range sequence object ──
//
// `pypy/module/__builtin__/functional.py W_Range`:
// an immutable arithmetic sequence carrying `(start, stop, step)`.  PyPy
// stores the three bounds as wrapped ints, so a range can describe values
// beyond a machine word; pyre keeps the same three wrapped fields.
//
// The hot `for i in range(n)` loop never reads these fields — `iter()`
// produces a `W_IntRangeIterator` (i64, JIT-specialized) when every bound
// fits a machine word, and a `W_LongRangeIterator` otherwise, mirroring
// PyPy's `rangeiterator` / `longrange_iterator` split.

/// `range(start, stop, step)` sequence object.
///
/// `w_length` is precomputed once at construction (`descr_new` →
/// `compute_range_length`) and read back by `descr_len` / `descr_bool`.
// Explicit `type_id` (not auto): `RANGE_DESCR_GROUP` in pyre-jit-trace bakes
// `W_RANGE_GC_TYPE_ID` at compile time to virtualize range GET_ITER/FOR_ITER,
// so the tid must be a const. 112 is the tail registration slot (right after
// W_Enumerate = 111); register_pyre_class drift-checks it against the runtime
// assignment order.
#[pyre_class("range", type_id = 112, static_name = "RANGE")]
pub struct W_Range {
    pub start: PyObjectRef,
    pub stop: PyObjectRef,
    pub step: PyObjectRef,
    pub length: PyObjectRef,
    /// `descr_new`'s `promote_step` — true when the constructor was given no
    /// step argument.  `descr_iter` reads it to pick the `step == 1`
    /// iterator shapes; a range built as `range(0, 10, 1)` carries a step of
    /// one but not the promotion, and keeps the general iterator.
    pub promote_step: bool,
}

pub const RANGE_START_OFFSET: usize = std::mem::offset_of!(W_Range, start);
pub const RANGE_STOP_OFFSET: usize = std::mem::offset_of!(W_Range, stop);
pub const RANGE_STEP_OFFSET: usize = std::mem::offset_of!(W_Range, step);
pub const RANGE_LENGTH_OFFSET: usize = std::mem::offset_of!(W_Range, length);
pub const RANGE_PROMOTE_STEP_OFFSET: usize = std::mem::offset_of!(W_Range, promote_step);

/// `functional.py get_len_of_range`. Callers keep `step != 0`.
///
/// After the `step < 0` swap, `diff` uses wrapping `r_uint` subtraction and
/// unsigned `//`; `as i64` is `intmask`. Truncating unsigned `/` matches
/// floor on the non-negative numerator the comment there records.
#[inline]
pub fn get_len_of_range(lo: i64, hi: i64, step: i64) -> i64 {
    debug_assert_ne!(step, 0);
    let (lo, hi, step) = if step < 0 {
        (hi, lo, step.wrapping_neg())
    } else {
        (lo, hi, step)
    };
    if lo < hi {
        let uhi = hi as u64;
        let ulo = lo as u64;
        let diff = uhi.wrapping_sub(ulo).wrapping_sub(1);
        (diff / (step as u64)).wrapping_add(1) as i64
    } else {
        0
    }
}

/// `functional.py W_Range.descr_new` `allocate_instance` / `__init__`.
/// Length is computed by the interpreter's `compute_range_length`.
///
/// `malloc_typed_managed` is the boxing `_int_add` / specialised-tuple
/// constructors use; `fuse_boxing_alloc` rewrites it to `new_with_vtable`.
/// `allocate_stable`'s `push_roots` is a symbolic residual and aborts
/// `RANGE_NEW_*_DESCENT`.
#[inline(never)]
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub fn w_range_allocate(
    start: PyObjectRef,
    stop: PyObjectRef,
    step: PyObjectRef,
    length: PyObjectRef,
    promote_step: bool,
) -> PyObjectRef {
    crate::lltype::malloc_typed_managed(W_Range {
        ob: PyObject {
            ob_type: &RANGE_TYPE as *const PyType,
            w_class: get_instantiate(&RANGE_TYPE),
        },
        start,
        stop,
        step,
        length,
        promote_step,
    }) as PyObjectRef
}

/// Box three machine-int bounds and the `get_len_of_range` length.
fn w_range_from_i64(start: i64, stop: i64, step: i64, promote_step: bool) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let _ = crate::gc_roots::pin_root(crate::intobject::w_int_new(start));
    let start_slot = crate::gc_roots::shadow_stack_len() - 1;
    let _ = crate::gc_roots::pin_root(crate::intobject::w_int_new(stop));
    let stop_slot = crate::gc_roots::shadow_stack_len() - 1;
    let _ = crate::gc_roots::pin_root(crate::intobject::w_int_new(step));
    let step_slot = crate::gc_roots::shadow_stack_len() - 1;
    let length = crate::gc_roots::pin_root(crate::intobject::w_int_new(get_len_of_range(
        start, stop, step,
    )));
    w_range_allocate(
        crate::gc_roots::shadow_stack_get(start_slot),
        crate::gc_roots::shadow_stack_get(stop_slot),
        crate::gc_roots::shadow_stack_get(step_slot),
        length,
        promote_step,
    )
}

/// Convenience constructor wrapping three machine-int bounds.  The explicit
/// step spells `promote_step = false`, matching `range(start, stop, step)`.
pub fn w_range_new_i64(start: i64, stop: i64, step: i64) -> PyObjectRef {
    w_range_from_i64(start, stop, step, false)
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_w_range(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &RANGE_TYPE) }
}

/// `descr_new`'s `promote_step` — whether the range was built without a step
/// argument.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
#[inline]
pub unsafe fn w_range_promote_step(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_Range)).promote_step }
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_exact_w_range(obj: PyObjectRef) -> bool {
    unsafe { std::ptr::eq((*obj).ob_type, &RANGE_TYPE) }
}

/// Read the wrapped `(start, stop, step)` triple of a range object.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
#[inline]
pub unsafe fn w_range_fields(obj: PyObjectRef) -> (PyObjectRef, PyObjectRef, PyObjectRef) {
    let r = obj as *const W_Range;
    unsafe { ((*r).start, (*r).stop, (*r).step) }
}

/// Read an int/long/bool operand as a `BigInt`.
///
/// # Safety
/// `obj` must be a valid int, long, or bool object.
pub unsafe fn range_obj_to_bigint(obj: PyObjectRef) -> BigInt {
    unsafe {
        // `is_int` is true for a bool (`BOOL_TYPE`), so test `is_bool` first;
        // a bool reads through `w_bool_get_value`, not the int accessor.
        if is_bool(obj) {
            BigInt::from(crate::boolobject::w_bool_get_value(obj) as i64)
        } else if is_int(obj) {
            BigInt::from(crate::intobject::w_int_get_value(obj))
        } else {
            // `space.bigint_w(W_LongObject)` returns the stored rbigint
            // reference. The host ownership adapter is a shallow clone, but
            // source translation aliases the existing GC payload.
            crate::longobject::w_long_get_value(obj).translated_alias()
        }
    }
}

/// Wrap a `BigInt` as a machine int when it fits, otherwise a long.
pub fn range_bigint_to_obj(value: BigInt) -> PyObjectRef {
    let value = RBigIntGcRoot::new(value);
    if crate::longobject::jit_bigint_to_i64_fits(&value) != 0 {
        crate::intobject::w_int_new(crate::longobject::jit_bigint_to_i64_value(&value))
    } else {
        crate::longobject::w_long_new(value.translated_alias())
    }
}

/// A single int/long/bool bound as an i64 when it fits, else `None`.
///
/// # Safety
/// `obj` must be a valid int/long/bool object.
pub unsafe fn range_obj_as_i64(obj: PyObjectRef) -> Option<i64> {
    unsafe {
        // `is_int` is true for a bool (`BOOL_TYPE`), so test `is_bool` first.
        if is_bool(obj) {
            Some(crate::boolobject::w_bool_get_value(obj) as i64)
        } else if is_int(obj) {
            Some(crate::intobject::w_int_get_value(obj))
        } else {
            let value = crate::longobject::w_long_get_value(obj);
            if crate::longobject::jit_bigint_to_i64_fits(value) != 0 {
                Some(crate::longobject::jit_bigint_to_i64_value(value))
            } else {
                None
            }
        }
    }
}

/// The `(start, stop, step)` triple as machine ints when all three fit a
/// machine word (the common case); `None` if any bound is a bignum.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_fields_i64(obj: PyObjectRef) -> Option<(i64, i64, i64)> {
    unsafe {
        let (s, e, t) = w_range_fields(obj);
        // Sequential `space.int_w` like `W_Range.descr_iter`: a `?` from
        // `Option<i64>` into `Option<(i64, i64, i64)>` is a FromResidual
        // conversion the linux rtyper declines.
        let Some(start) = range_obj_as_i64(s) else {
            return None;
        };
        let Some(stop) = range_obj_as_i64(e) else {
            return None;
        };
        let Some(step) = range_obj_as_i64(t) else {
            return None;
        };
        Some((start, stop, step))
    }
}

/// The precomputed wrapped element count — `descr_len → self.w_length`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
#[inline]
pub unsafe fn w_range_length(obj: PyObjectRef) -> PyObjectRef {
    let r = obj as *const W_Range;
    unsafe { (*r).length }
}

/// The element count as a machine int when it fits, else `None`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
#[inline]
pub unsafe fn w_range_length_i64(obj: PyObjectRef) -> Option<i64> {
    unsafe { range_obj_as_i64(w_range_length(obj)) }
}

/// `descr_bool → space.nonzero(self.w_length)` — a range is truthy iff it
/// has at least one element.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_bool(obj: PyObjectRef) -> bool {
    unsafe { !range_obj_to_bigint(w_range_length(obj)).is_zero() }
}

/// `functional.py W_Range.descr_iter` — `space.int_w` on start/stop/step/length
/// picks a machine-int `rangeiterator`; `OverflowError` falls through to
/// `W_LongRangeIterator`. A promoted step uses `W_IntRangeOneArgIterator`
/// (`start == 0`) or `W_IntRangeStepOneIterator`, both keyed off the exclusive
/// end `start + length` (see the empty-promoted-end gate below).
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_iter(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let (start_obj, stop_obj, step_obj) = w_range_fields(obj);
        let length_obj = w_range_length(obj);
        // Sequential `space.int_w` like `descr_iter`'s try/except: any bound
        // that does not fit a machine word takes the long iterator.
        let Some(start) = range_obj_as_i64(start_obj) else {
            return w_long_range_iter_new(start_obj, step_obj, length_obj);
        };
        let Some(_stop) = range_obj_as_i64(stop_obj) else {
            return w_long_range_iter_new(start_obj, step_obj, length_obj);
        };
        let Some(step) = range_obj_as_i64(step_obj) else {
            return w_long_range_iter_new(start_obj, step_obj, length_obj);
        };
        let Some(length) = range_obj_as_i64(length_obj) else {
            return w_long_range_iter_new(start_obj, step_obj, length_obj);
        };
        if w_range_promote_step(obj) {
            // [3.14-spec] empty promoted end. `W_Range.descr_iter` builds
            // `W_IntRangeOneArgIterator(space, stop)` /
            // `W_IntRangeStepOneIterator(space, start, stop)` from the raw
            // stop, so `iter(range(-1)).__length_hint__()` is -1 and
            // `operator.length_hint` raises `ValueError: __length_hint__()
            // should return >= 0` (measured on pypy3). `rangeobject.c
            // range_iter` stores `get_len_of_range` (`len >= 0`):
            // `type(it).__name__, length_hint, reduce` of `iter(range(-1))`
            // is `range_iterator 0 0 (iter, (range(0, 0),), None)` and of
            // `iter(range(3, 1))` is `0 (iter, (range(3, 3),), None)` on
            // 3.14.6 (measured). The exclusive end is `start + length`
            // (length >= 0, step 1). `_immutable_fields_ = ['stop']` on both
            // specialised classes is about mutation, not the constructor
            // value; `descr_iter` carries no `@jit.*`.
            debug_assert!(length >= 0);
            let end = start + length;
            if start == 0 {
                return w_range_iter_one_arg_new(end);
            }
            return w_range_iter_step_one_new(start, end);
        }
        // [3.14-spec] one-past overflow gate. `W_Range.descr_iter` returns
        // `W_IntRangeIterator(space, start, length, step)` with no one-past
        // check. `rangeobject.c range_iter` falls back to the long iterator
        // when `start + len*step` overflows: `type(iter(range(sys.maxsize-5,
        // sys.maxsize, 10))).__name__` is `longrange_iterator` on 3.14.6 and
        // `range_iterator` on pypy3 (measured). `functional.py` `W_Range` /
        // `W_IntRangeIterator` carry no load-bearing `@jit.*` or
        // `_immutable_fields_`.
        let Some(product) = length.checked_mul(step) else {
            return w_long_range_iter_new(start_obj, step_obj, length_obj);
        };
        let Some(_one_past) = start.checked_add(product) else {
            return w_long_range_iter_new(start_obj, step_obj, length_obj);
        };
        w_range_iter_new(start, length, step)
    }
}

/// [3.14-spec] machine-int reverse. `W_Range.descr_reversed` always builds
/// `W_LongRangeIterator` via `space.add`/`mul`/`neg`, but PyPy names that
/// type `range_iterator` too; pyre's long iterator is `longrange_iterator`.
/// `rangeobject.c range_reverse` returns `range_iterator` when the reversed
/// span fits a C long. `type(reversed(range(10))).__name__` is
/// `range_iterator` on 3.14.6 and on pypy3 (measured), so the word iterator
/// keeps pyre on both. Overflow falls through to
/// `(start + (length-1)*step, -step, length)`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_reversed(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let (start_obj, _stop_obj, step_obj) = w_range_fields(obj);
        let length_obj = w_range_length(obj);
        // Sequential `space.int_w` like `descr_iter`. Overflow of
        // `last = start + (len-1)*step` or the one-past bound uses
        // `checked_*` rather than i128 (`i64::try_from`), which the
        // linux rtyper declines.
        if let (Some(start), Some(step), Some(len)) = (
            range_obj_as_i64(start_obj),
            range_obj_as_i64(step_obj),
            range_obj_as_i64(length_obj),
        ) {
            if len == 0 {
                return w_range_iter_new(0, 0, 1);
            }
            if let (Some(adj), Some(neg_step)) = (len.checked_sub(1), step.checked_neg())
                && let Some(product) = adj.checked_mul(step)
                && let Some(last) = start.checked_add(product)
                && let Some(len_prod) = len.checked_mul(neg_step)
                && let Some(_one_past) = last.checked_add(len_prod)
            {
                return w_range_iter_new(last, len, neg_step);
            }
        }
        // `range_obj_to_bigint` of a machine int and each boxing collect;
        // the fields and the boxed last item come back off the shadow stack.
        let _roots = crate::gc_roots::push_roots();
        let (start, _stop, step) = w_range_fields(obj);
        let len_obj = w_range_length(obj);
        let field_base = crate::gc_roots::pin_roots(&[start, step, len_obj]);
        let start_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base,
        )));
        let step_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base + 1,
        )));
        let len_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base + 2,
        )));
        let adj = RBigIntGcRoot::new(len_b.int_sub(1));
        let product = RBigIntGcRoot::new(adj.mul(&step_b));
        let lastitem = start_b.add(&product);
        let w_lastitem = range_bigint_to_obj(lastitem);
        let lastitem_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(w_lastitem);
        let w_negstep = range_bigint_to_obj(step_b.neg());
        w_long_range_iter_new(
            crate::gc_roots::shadow_stack_get(lastitem_slot),
            w_negstep,
            crate::gc_roots::shadow_stack_get(field_base + 2),
        )
    }
}

/// `_compute_item` — the member at `index` (negative folded against the
/// length), or `None` when out of bounds (`IndexError` at the call site).
/// `index` is the already-`__index__`'d operand as a `BigInt`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_compute_item(obj: PyObjectRef, index: &BigInt) -> Option<PyObjectRef> {
    unsafe {
        // `fromint` of length / start / step allocates.  Pin the range and
        // its field pointers and re-read them, the same shape as
        // `range_hash_method` (`functional.py _compute_item` / `_compute_item0`).
        // `index` is rooted before the first of them.
        let mut idx = RBigIntGcRoot::new(index.translated_alias());
        let _roots = crate::gc_roots::push_roots();
        let obj = crate::gc_roots::pin_root(obj);
        let obj_slot = crate::gc_roots::shadow_stack_len() - 1;
        let (start, _stop, step) = w_range_fields(obj);
        let field_base = crate::gc_roots::pin_roots(&[start, step]);
        let len_obj = w_range_length(crate::gc_roots::shadow_stack_get(obj_slot));
        let len_slot = crate::gc_roots::pin_roots(&[len_obj]);
        let len_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            len_slot,
        )));
        if idx.int_lt(0) {
            idx = RBigIntGcRoot::new(idx.add(&len_b));
        }
        if *idx >= *len_b || idx.int_lt(0) {
            return None;
        }
        let start_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base,
        )));
        let step_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base + 1,
        )));
        let product = RBigIntGcRoot::new(idx.mul(&step_b));
        let value = start_b.add(&product);
        Some(range_bigint_to_obj(value))
    }
}

/// `_contains_long` — O(1) membership test for an integer `item`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_contains_bigint(obj: PyObjectRef, item: &BigInt) -> bool {
    unsafe {
        // `range_obj_to_bigint` of a machine int collects; `item` and the
        // fields are rooted before the first conversion.
        let item = live_rbigint(item);
        let _roots = crate::gc_roots::push_roots();
        let (start, stop, step) = w_range_fields(obj);
        let field_base = crate::gc_roots::pin_roots(&[start, stop, step]);
        let start_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base,
        )));
        let stop_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base + 1,
        )));
        let step_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base + 2,
        )));
        if *step_b > BigInt::zero() {
            // positive steps: start <= ob < stop
            if !(*start_b <= *item && *item < *stop_b) {
                return false;
            }
        } else {
            // negative steps: stop < ob <= start
            if !(*stop_b < *item && *item <= *start_b) {
                return false;
            }
        }
        // The stride must not invalidate membership.
        let diff = RBigIntGcRoot::new(item.sub(&start_b));
        (&*diff % &*step_b).is_zero()
    }
}

/// `descr_index` — the position of `item` (known to be in range), i.e.
/// `(item - start) // step`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_index_of(obj: PyObjectRef, item: &BigInt) -> PyObjectRef {
    unsafe {
        // `range_obj_to_bigint` of a machine int collects; `item` and the
        // fields are rooted before the first conversion.
        let item = live_rbigint(item);
        let _roots = crate::gc_roots::push_roots();
        let (start, _stop, step) = w_range_fields(obj);
        let field_base = crate::gc_roots::pin_roots(&[start, step]);
        let start_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base,
        )));
        let step_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base + 1,
        )));
        let diff = RBigIntGcRoot::new(item.sub(&start_b));
        let value = &*diff / &*step_b;
        range_bigint_to_obj(value)
    }
}

/// `descr_eq` — two ranges are equal iff they generate the same sequence:
/// equal lengths, and for a non-empty range equal start and (for length
/// > 1) equal step.  The caller has already established both operands are
/// >    ranges.
///
/// # Safety
/// `a` and `b` must point to valid `W_Range` objects.
pub unsafe fn w_range_eq(a: PyObjectRef, b: PyObjectRef) -> bool {
    unsafe {
        // functional.py W_Range.descr_eq: `space.eq_w` on length / start /
        // step collects (`rbigint.py fromint` allocates Digits).
        // gct_fv_gc_malloc reloads `self` / `w_other` and the field
        // pointers; pin them for the native path.
        let _roots = crate::gc_roots::push_roots();
        let base = crate::gc_roots::pin_roots(&[a, b]);
        let la = RBigIntGcRoot::new(range_obj_to_bigint(w_range_length(
            crate::gc_roots::shadow_stack_get(base),
        )));
        let lb = RBigIntGcRoot::new(range_obj_to_bigint(w_range_length(
            crate::gc_roots::shadow_stack_get(base + 1),
        )));
        if *la != *lb {
            return false;
        }
        let (astart, _astop, astep) = w_range_fields(crate::gc_roots::shadow_stack_get(base));
        let (bstart, _bstop, bstep) = w_range_fields(crate::gc_roots::shadow_stack_get(base + 1));
        if la.int_eq(0) {
            return true;
        }
        let field_base = crate::gc_roots::pin_roots(&[astart, astep, bstart, bstep]);
        let astart_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base,
        )));
        let bstart_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base + 2,
        )));
        if *astart_b != *bstart_b {
            return false;
        }
        if la.int_eq(1) {
            return true;
        }
        let astep_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base + 1,
        )));
        let bstep_b = RBigIntGcRoot::new(range_obj_to_bigint(crate::gc_roots::shadow_stack_get(
            field_base + 3,
        )));
        *astep_b == *bstep_b
    }
}

/// Number of elements in a `(start, stop, step)` range —
/// `functional.py get_len_of_range`.
pub fn range_length(start: i64, stop: i64, step: i64) -> i64 {
    get_len_of_range(start, stop, step)
}

// ── Long range iterator ──
//
// `pypy/module/__builtin__/functional.py W_LongRangeIterator` — the cursor `iter()`
// produces for a range whose bounds exceed a machine word.  `start`, `step`
// and `len` are set once at construction; `index` is a wrapped integer that
// advances by one each step (`self.w_index`), so the cursor keeps arbitrary
// precision and never overflows for a range longer than a machine word.

#[pyre_class("longrange_iterator", static_name = "LONG_RANGE_ITER")]
pub struct W_LongRangeIterator {
    pub start: PyObjectRef,
    pub step: PyObjectRef,
    pub len: PyObjectRef,
    pub index: PyObjectRef,
}

/// Allocate a `W_LongRangeIterator`.
pub fn w_long_range_iter_new(
    start: PyObjectRef,
    step: PyObjectRef,
    len: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let _ = crate::gc_roots::pin_root(start);
    let start_slot = crate::gc_roots::shadow_stack_len() - 1;
    let _ = crate::gc_roots::pin_root(step);
    let step_slot = crate::gc_roots::shadow_stack_len() - 1;
    let _ = crate::gc_roots::pin_root(len);
    let len_slot = crate::gc_roots::shadow_stack_len() - 1;
    let index = crate::gc_roots::pin_root(crate::intobject::w_int_new(0));
    let start = crate::gc_roots::shadow_stack_get(start_slot);
    let step = crate::gc_roots::shadow_stack_get(step_slot);
    let len = crate::gc_roots::shadow_stack_get(len_slot);
    W_LongRangeIterator::allocate_stable(W_LongRangeIterator {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        start,
        step,
        len,
        index,
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_long_range_iter(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &LONG_RANGE_ITER_TYPE) }
}

/// Number of elements not yet produced — `__length_hint__`.
///
/// # Safety
/// `obj` must point to a valid `W_LongRangeIterator`.
pub unsafe fn w_long_range_iter_len(obj: PyObjectRef) -> BigInt {
    unsafe {
        let it = obj as *const W_LongRangeIterator;
        let len = RBigIntGcRoot::new(range_obj_to_bigint((*it).len));
        let index = RBigIntGcRoot::new(range_obj_to_bigint((*it).index));
        let rem = RBigIntGcRoot::new(len.sub(&index));
        if rem.int_lt(0) {
            BigInt::from(0)
        } else {
            rem.translated_alias()
        }
    }
}

/// Read the `(start, step, len, index)` fields of a long-range iterator
/// as wrapped int/long objects.
///
/// # Safety
/// `obj` must point to a valid `W_LongRangeIterator`.
pub unsafe fn w_long_range_iter_fields(
    obj: PyObjectRef,
) -> (PyObjectRef, PyObjectRef, PyObjectRef, PyObjectRef) {
    let it = obj as *const W_LongRangeIterator;
    unsafe { ((*it).start, (*it).step, (*it).len, (*it).index) }
}

/// Restore the arbitrary-precision range iterator's wrapped index field.
///
/// # Safety
/// `obj` must point to a valid `W_LongRangeIterator` and `index` must be a
/// wrapped int/long in the inclusive interval `[0, len]`.
pub unsafe fn w_long_range_iter_set_index(obj: PyObjectRef, index: PyObjectRef) {
    let it = obj as *mut W_LongRangeIterator;
    unsafe {
        (*it).index = index;
        crate::gc_hook::try_gc_write_barrier(it as crate::gc_hook::GCREF);
    }
}

/// Whether the long-range iterator has elements left — peek, non-mutating.
///
/// # Safety
/// `obj` must point to a valid `W_LongRangeIterator`.
pub unsafe fn w_long_range_iter_has_next(obj: PyObjectRef) -> bool {
    unsafe {
        let it = obj as *const W_LongRangeIterator;
        let index = RBigIntGcRoot::new(range_obj_to_bigint((*it).index));
        let len = RBigIntGcRoot::new(range_obj_to_bigint((*it).len));
        *index < *len
    }
}

/// Advance the long-range iterator and return the next value, or `None`
/// once exhausted — `start + index * step`.
///
/// # Safety
/// `obj` must point to a valid `W_LongRangeIterator`.
pub unsafe fn w_long_range_iter_next(obj: PyObjectRef) -> Option<PyObjectRef> {
    unsafe {
        // The iterator is rewritten after the rbigint arithmetic below, and
        // every step of it allocates digits.
        let _roots = crate::gc_roots::push_roots();
        let iter_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::pin_root(obj);
        let it = obj as *mut W_LongRangeIterator;
        let index = RBigIntGcRoot::new(range_obj_to_bigint((*it).index));
        let len = RBigIntGcRoot::new(range_obj_to_bigint((*it).len));
        if *index >= *len {
            return None;
        }
        let start = RBigIntGcRoot::new(range_obj_to_bigint((*it).start));
        let step = RBigIntGcRoot::new(range_obj_to_bigint((*it).step));
        // `w_result = self.w_index * self.w_step + self.w_start`, then
        // `self.w_index = self.w_index + 1` (wrapped, arbitrary precision).
        // `next_index` is wrapped before `value`; that first allocation may
        // collect while the computed item exists only as an unboxed rbigint.
        // RPython's GC transform roots this local automatically.
        let product = RBigIntGcRoot::new(index.mul(&step));
        let value = RBigIntGcRoot::new(start.add(&product));
        let next_index = range_bigint_to_obj(index.int_add(1));
        let next_index = crate::gc_roots::pin_root(next_index);
        let it = crate::gc_roots::shadow_stack_get(iter_slot) as *mut W_LongRangeIterator;
        (*it).index = next_index;
        crate::gc_hook::try_gc_write_barrier(it as crate::gc_hook::GCREF);
        Some(range_bigint_to_obj(value.translated_alias()))
    }
}

#[cfg(test)]
mod range_obj_tests {
    use super::*;

    #[test]
    fn w_range_object_size_matches_descr() {
        // Auto-id `allocate_stable` (GC-managed): the tid is assigned at JIT
        // init, so there is no compile-time `W_RANGE_GC_TYPE_ID` to check.
        assert_eq!(
            <W_Range as crate::lltype::GcType>::SIZE,
            W_RANGE_OBJECT_SIZE
        );
    }

    #[test]
    fn range_length_matches_cpython() {
        assert_eq!(range_length(0, 5, 1), 5);
        assert_eq!(range_length(0, 0, 1), 0);
        assert_eq!(range_length(5, 0, 1), 0);
        assert_eq!(range_length(0, 10, 3), 4);
        assert_eq!(range_length(10, 0, -1), 10);
        assert_eq!(range_length(10, 0, -3), 4);
        assert_eq!(get_len_of_range(0, 5, 1), 5);
        assert_eq!(get_len_of_range(10, 0, -3), 4);
        // wrapping `r_uint` form: `intmask((u64::MAX - 1) / 1 + 1) == -1`.
        assert_eq!(get_len_of_range(i64::MIN, i64::MAX, 1), -1);
    }

    #[test]
    fn w_range_fields_i64_roundtrip() {
        let r = w_range_new_i64(0, 10, 2);
        unsafe {
            assert_eq!(w_range_fields_i64(r), Some((0, 10, 2)));
        }
    }

    #[test]
    fn long_range_iter_yields_values() {
        let it = w_long_range_iter_new(
            crate::intobject::w_int_new(5),
            crate::intobject::w_int_new(2),
            crate::intobject::w_int_new(3),
        );
        unsafe {
            assert!(is_long_range_iter(it));
            let v0 = w_long_range_iter_next(it).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(v0), 5);
            let v1 = w_long_range_iter_next(it).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(v1), 7);
            let v2 = w_long_range_iter_next(it).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(v2), 9);
            assert!(w_long_range_iter_next(it).is_none());
        }
    }

    #[test]
    fn iter_routes_increment_overflow_to_longrange() {
        unsafe {
            // Word-fit bounds, but the forward `start + length*step` overflows
            // i64, so a machine-int iterator would loop forever — must use the
            // bignum iterator instead.
            let fwd = w_range_new_i64(i64::MAX - 5, i64::MAX, 10);
            assert!(is_long_range_iter(w_range_iter(fwd)));
            // The reversed walk's one-past `start - step` underflows i64 here,
            // so `reversed()` must likewise use the bignum iterator.
            let rev = w_range_new_i64(i64::MIN, i64::MIN + 50, 10);
            assert!(is_long_range_iter(w_range_reversed(rev)));
            // A plain range stays on the machine-int (JIT) iterator both ways.
            let n = w_range_new_i64(0, 10, 2);
            assert!(is_range_iter(w_range_iter(n)));
            assert!(is_range_iter(w_range_reversed(n)));
        }
    }
}
