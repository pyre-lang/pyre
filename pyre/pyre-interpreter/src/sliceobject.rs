//! Interpreter-level slice helpers.
//!
//! PyPy equivalent: `pypy/objspace/std/sliceobject.py`. Only the helpers
//! that the interpreter layer uses are ported here; the data type for
//! `slice` objects lives in `pyre-object::sliceobject`.

use pyre_object::{PyObjectRef, pyobject::is_none};

/// sliceobject.py `_eval_slice_index(space, w_int)`.
///
/// Returns `w_int.__index__()` as an `i64`, converting to `TypeError`
/// when the object has no `__index__` method.
pub fn eval_slice_index(w_int: PyObjectRef) -> Result<i64, crate::PyError> {
    // A null bound is silently misread on wasm (offset 0 is valid memory), so
    // `getindex_w` reads `ob_type` at guest offset 0 and returns garbage. Trap
    // loudly instead so a wasm run pins the null-Ref provenance.
    assert!(
        !w_int.is_null(),
        "eval_slice_index: slice bound Ref is null (wasm offset-0 silent-null); getindex_w would read ob_type at guest offset 0"
    );
    // Own RootScope so the BINARY_SLICE caller can erase its bracket:
    // this body is depth-neutral (`is_stack_depth_neutral_fn`) even
    // though `getindex_w` is in the sensitive set. `shadow_stack_erase`
    // scalar-replaces the pins; `getindex_w` still runs `__index__`.
    let _roots = pyre_object::gc_roots::push_roots();
    let w_int = pyre_object::gc_roots::pin_root(w_int);
    // One return so the RootScope Drop is a regular close, not an
    // unwind-only cleanup `body_is_depth_neutral` misses.

    match crate::baseobjspace::getindex_w(w_int) {
        Ok(v) => Ok(v),
        Err(e) if e.kind == crate::PyErrorKind::TypeError => Err(crate::PyError::new(
            crate::PyErrorKind::TypeError,
            "slice indices must be integers or have an __index__ method".to_string(),
        )),
        Err(e) => Err(e),
    }
}

/// sliceobject.py `adapt_lower_bound(space, size, w_index)`.
///
/// Converts `w_index` via `__index__`, normalizes negatives against
/// `size`, and clamps at zero.
pub fn adapt_lower_bound(size: i64, w_index: PyObjectRef) -> Result<i64, crate::PyError> {
    let index = adapt_bound(size, eval_slice_index(w_index)?);
    debug_assert!(index >= 0);
    Ok(index)
}

/// `eval_slice_index` for a bound that may not be `None`
/// (`_PyEval_SliceIndexNotNone`), which is how a sequence `index` method
/// converts its optional start/stop: `[1, 2].index(2, None)` is a TypeError
/// even though `[1, 2][None:]` is not.  The message drops the `or None` the
/// slicing form offers.
pub fn eval_slice_index_not_none(mut w_int: PyObjectRef) -> Result<i64, crate::PyError> {
    // `_PyIndex_Check` gates the substituted message and is a type test, so a
    // bound that has `__index__` is converted exactly once, by
    // `eval_slice_index`.  Probing with a conversion instead would run a user
    // slot twice and relabel everything it raises — an `OverflowError` for a
    // result too wide for a word, or the slot's own exception — as this
    // TypeError, which upstream reserves for a type with no `__index__` at all.
    if unsafe {
        is_none(w_int) || !pyre_object::with_roots!(w_int => crate::builtins::index_check(w_int))
    } {
        return Err(crate::PyError::type_error(
            "slice indices must be integers or have an __index__ method",
        ));
    }
    eval_slice_index(w_int)
}

/// `unwrap_start_stop` for the bounds of a sequence `index` method, which
/// rejects `None` rather than reading it as "this side unbounded".
pub fn unwrap_start_stop_not_none(
    size: i64,
    w_start: PyObjectRef,
    w_end: PyObjectRef,
) -> Result<(i64, i64), crate::PyError> {
    let (start, end) = index_bounds_not_none(w_start, w_end)?;
    Ok(adapt_start_stop(size, start, end))
}

/// The conversion half of [`unwrap_start_stop_not_none`], for a receiver whose
/// length a bound's own `__index__` can change: convert both bounds here, read
/// the length afterwards, and normalize with [`adapt_start_stop`].
pub fn index_bounds_not_none(
    w_start: PyObjectRef,
    w_end: PyObjectRef,
) -> Result<(i64, i64), crate::PyError> {
    // Converted one at a time, as `slice_unpack`: the first bound's
    // `__index__` is user code that can collect, so the second is read back
    // from the shadow stack rather than carried in the argument it arrived
    // in. A caller that roots the container does not root these two.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[w_start, w_end]);
    let start = eval_slice_index_not_none(pyre_object::gc_roots::shadow_stack_get(base))?;
    let end = eval_slice_index_not_none(pyre_object::gc_roots::shadow_stack_get(base + 1))?;
    Ok((start, end))
}

/// The normalization half: fold negative bounds by `size` and floor at 0.
pub fn adapt_start_stop(size: i64, start: i64, end: i64) -> (i64, i64) {
    (adapt_bound(size, start), adapt_bound(size, end))
}

/// The negative-index normalization `adapt_lower_bound` applies once the
/// bound has already been converted to an `i64`.
///
/// Spelled as a plain add rather than `saturating_add`: `index` is negative
/// here and `size` is a length, so the sum stays inside the word — it can
/// only move an out-of-word `i64::MIN` upwards — and the saturating form is
/// an unregistered foreign leaf that stops the enclosing graph, the same
/// reason `type_methods.rs args_given` avoids `saturating_sub`.
pub(crate) fn adapt_bound(size: i64, index: i64) -> i64 {
    if index >= 0 {
        return index;
    }
    let folded = index + size;
    if folded < 0 { 0 } else { folded }
}

/// sliceobject.py `unwrap_start_stop(space, size, w_start, w_end)`.
///
/// Returns `(start, end)` after negative-index normalization. `None`
/// maps to `(0, size)`.
pub fn unwrap_start_stop(
    size: i64,
    w_start: PyObjectRef,
    w_end: PyObjectRef,
) -> Result<(i64, i64), crate::PyError> {
    // Rooted and read back for the same reason as
    // `unwrap_start_stop_not_none`.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[w_start, w_end]);
    let w_start = pyre_object::gc_roots::shadow_stack_get(base);
    let start = if unsafe { is_none(w_start) } {
        0
    } else {
        adapt_lower_bound(size, w_start)?
    };
    let w_end = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let end = if unsafe { is_none(w_end) } {
        debug_assert!(size >= 0);
        size
    } else {
        adapt_lower_bound(size, w_end)?
    };
    Ok((start, end))
}

/// sliceobject.py `W_SliceObject.unpack(space)`.
///
/// Evaluates `start`/`stop`/`step` through `__index__` **without** reading
/// the container length: an `__index__` method may mutate the container, so
/// the length must be consulted only afterwards (`_unpack_slice`). `None`
/// endpoints map to the open-ended sentinels that [`slice_adjust_indices`]
/// then clamps. A zero `step` raises `ValueError`.
///
/// The bounds are converted one at a time. `__index__` is user code that
/// can collect, so a bound not yet reached is read back from the shadow
/// stack.
pub fn slice_unpack(
    w_start: PyObjectRef,
    w_stop: PyObjectRef,
    w_step: PyObjectRef,
) -> Result<(i64, i64, i64), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_start);
    let _ = pyre_object::gc_roots::pin_root(w_stop);
    let _ = pyre_object::gc_roots::pin_root(w_step);
    let start_slot = base;
    let stop_slot = base + 1;
    let step_slot = base + 2;
    let step = if unsafe { is_none(pyre_object::gc_roots::shadow_stack_get(step_slot)) } {
        1
    } else {
        let step = eval_slice_index(pyre_object::gc_roots::shadow_stack_get(step_slot))?;
        if step == 0 {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                "slice step cannot be zero".to_string(),
            ));
        }
        step
    };
    let start = if unsafe { is_none(pyre_object::gc_roots::shadow_stack_get(start_slot)) } {
        if step < 0 { i64::MAX } else { 0 }
    } else {
        eval_slice_index(pyre_object::gc_roots::shadow_stack_get(start_slot))?
    };
    let stop = if unsafe { is_none(pyre_object::gc_roots::shadow_stack_get(stop_slot)) } {
        if step < 0 { i64::MIN } else { i64::MAX }
    } else {
        eval_slice_index(pyre_object::gc_roots::shadow_stack_get(stop_slot))?
    };
    Ok((start, stop, step))
}

/// sliceobject.py `W_SliceObject.adjust_indices(start, stop, step, length)`.
///
/// Pure arithmetic: clamps the unpacked `(start, stop, step)` against
/// `length`, clipping out-of-bounds endpoints consistently with
/// extended-slice handling, and returns the triple plus the resulting
/// `slicelength`.
pub fn slice_adjust_indices(
    mut start: i64,
    mut stop: i64,
    step: i64,
    length: i64,
) -> (i64, i64, i64, i64) {
    if start < 0 {
        start += length;
        if start < 0 {
            start = if step < 0 { -1 } else { 0 };
        }
    } else if start >= length {
        start = if step < 0 { length - 1 } else { length };
    }
    if stop < 0 {
        stop += length;
        if stop < 0 {
            stop = if step < 0 { -1 } else { 0 };
        }
    } else if stop >= length {
        stop = if step < 0 { length - 1 } else { length };
    }
    let slicelength = if (step < 0 && stop >= start) || (step > 0 && start >= stop) {
        0
    } else if step < 0 {
        (stop - start + 1) / step + 1
    } else {
        (stop - start - 1) / step + 1
    };
    (start, stop, step, slicelength)
}

/// sliceobject.py `W_SliceObject.indices3(space, length)`.
///
/// Computes the `(start, stop, step)` triple for a slice over a sequence
/// of `length` items — [`slice_unpack`] then [`slice_adjust_indices`]. A
/// zero `step` raises `ValueError`.
pub fn indices3(
    w_start: PyObjectRef,
    w_stop: PyObjectRef,
    w_step: PyObjectRef,
    length: i64,
) -> Result<(i64, i64, i64), crate::PyError> {
    let (start, stop, step) = slice_unpack(w_start, w_stop, w_step)?;
    let (start, stop, step, _) = slice_adjust_indices(start, stop, step, length);
    Ok((start, stop, step))
}
