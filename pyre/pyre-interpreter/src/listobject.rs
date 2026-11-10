//! Interpreter-level list helpers that need access to `space.eq_w`.
//!
//! PyPy equivalent: method bodies on `ListStrategy.find_or_count`
//! (`pypy/objspace/std/listobject.py`) and the descr callers that
//! propagate exceptions from `space.eq_w`. The raw list container
//! (`W_ListObject`) and typed fast paths live in
//! `pyre-object::listobject`; the generic `__eq__` loop lives here where
//! the object space is available.

use pyre_object::{PyObjectRef, listobject::ListFindFast};

use crate::PyError;

/// Outcome of `W_ListObject.find_or_count`. PyPy signals "not found" via
/// a Python `ValueError`; pyre keeps the unwrap into `i64` at the
/// descr-level callers so the rare not-found case doesn't allocate a
/// `PyError` inside the inner loop.
pub enum FindOrCountResult {
    Index(i64),
    Count(i64),
    NotFound,
}

/// `listobject.py` `W_ListObject.find_or_count`.
///
/// Dispatches to the current strategy's typed fast path via
/// `w_list_find_or_count_fast`; when that signals `NeedsGeneric`, runs
/// the generic `ListStrategy.find_or_count` loop
/// (`listobject.py:941-957`) using `space.eq_w`.
/// `x in xs` under the list mutation lock.  `IntegerListStrategy.find_or_count`
/// scans the unboxed pool when the needle is still a plain int; otherwise
/// this falls back to the generic `eq_w` loop.  The lock acquire is a GC
/// safepoint, so the list and needle are rooted and reloaded.  Not
/// elidable: the list is mutable.
///
/// The lock stays in this residual so the `contains_list` dispatcher
/// remains walkable (`w_list_append` documents the same split).
#[majit_macros::dont_look_inside]
pub fn contains_int_list_locked(obj: PyObjectRef, w_item: PyObjectRef) -> Result<bool, PyError> {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::publish_roots(&[obj, w_item]);
        pyre_object::gc_roots::normalize_roots(root_base, 2);
        let obj = pyre_object::gc_roots::shadow_stack_get(root_base);
        let lock = pyre_object::listobject::w_list_lock_acquire(obj);
        let obj = pyre_object::gc_roots::shadow_stack_get(root_base);
        let w_item = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        let integer_plain = pyre_object::listobject::w_list_strategy(obj)
            == pyre_object::listobject::ListStrategy::Integer
            && pyre_object::listobject::is_plain_int1(w_item)
            && pyre_object::is_int(w_item);
        if integer_plain {
            let found = matches!(
                pyre_object::listobject::w_list_find_or_count_fast(obj, w_item, 0, i64::MAX, false),
                ListFindFast::Found(_)
            );
            pyre_object::listobject::w_list_lock_release(lock);
            return Ok(found);
        }
        pyre_object::listobject::w_list_lock_release(lock);
        let obj = pyre_object::gc_roots::shadow_stack_get(root_base);
        let w_item = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        w_list_find_or_count(obj, w_item, 0, i64::MAX, false)
            .map(|result| matches!(result, FindOrCountResult::Index(_)))
    }
}

/// Residual entry for [`contains_int_list_locked`].  A stored object's
/// `__eq__` on the generic fallback can run Python.
#[majit_macros::jit_may_force]
pub extern "C" fn jit_list_contains_int(haystack: PyObjectRef, needle: PyObjectRef) -> i64 {
    match contains_int_list_locked(haystack, needle) {
        Ok(found) => i64::from(found),
        Err(err) => crate::runtime_ops::jit_publish_residual_error(err),
    }
}

pub fn w_list_find_or_count(
    obj: PyObjectRef,
    w_item: PyObjectRef,
    start: i64,
    stop: i64,
    count: bool,
) -> Result<FindOrCountResult, PyError> {
    match unsafe {
        pyre_object::listobject::w_list_find_or_count_fast(obj, w_item, start, stop, count)
    } {
        ListFindFast::Found(i) => return Ok(FindOrCountResult::Index(i)),
        ListFindFast::Count(n) => return Ok(FindOrCountResult::Count(n)),
        ListFindFast::NotFound => return Ok(FindOrCountResult::NotFound),
        ListFindFast::NeedsGeneric => {}
    }
    // listobject.py ListStrategy.find_or_count:
    //     while i < stop and i < w_list.length():
    //         if space.eq_w(w_list.getitem(i), w_item):
    //             ...
    //         i += 1
    //     raise ValueError / return count
    let mut i = start.max(0);
    let mut result: i64 = 0;
    // `eq_w` re-enters Python and may collect; the list and needle are raw
    // locals re-read each iteration, so pin them on the shadow stack.
    let _roots = pyre_object::gc_roots::push_roots();
    let pair = pyre_object::gc_roots::pin_roots(&[obj, w_item]);
    let obj_slot = pair;
    let item_slot = pair + 1;
    while i < stop
        && i < unsafe { pyre_object::w_list_len(pyre_object::gc_roots::shadow_stack_get(obj_slot)) }
            as i64
    {
        let w_curr = match unsafe {
            pyre_object::w_list_getitem(pyre_object::gc_roots::shadow_stack_get(obj_slot), i)
        } {
            Some(v) => v,
            None => break,
        };
        let _item_roots = pyre_object::gc_roots::push_roots();
        let w_curr = pyre_object::gc_roots::pin_root(w_curr);
        if crate::baseobjspace::eq_w(w_curr, pyre_object::gc_roots::shadow_stack_get(item_slot))? {
            if count {
                result += 1;
            } else {
                return Ok(FindOrCountResult::Index(i));
            }
        }
        i += 1;
    }
    if count {
        Ok(FindOrCountResult::Count(result))
    } else {
        Ok(FindOrCountResult::NotFound)
    }
}

/// `listobject.py` `W_ListObject.descr_remove`.
///
/// Runs `find_or_count(value, 0, sys.maxint)`, pops at the returned
/// index when still within bounds (listobject.py:791 guard against
/// `eq_w`-triggered mutations), raises `ValueError` otherwise.
pub fn w_list_remove(obj: PyObjectRef, w_value: PyObjectRef) -> Result<(), PyError> {
    // The scan runs the elements' `__eq__`, which is a collection point.
    // `w_list_find_or_count` pins and reloads for its own loop, but that scope
    // drops when it returns, so the length read and the pop below would
    // address the list at its pre-move header.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let obj = pyre_object::gc_roots::pin_root(obj);
    let i = match w_list_find_or_count(obj, w_value, 0, i64::MAX, false)? {
        FindOrCountResult::Index(i) => i,
        FindOrCountResult::NotFound => {
            return Err(PyError::new(
                crate::PyErrorKind::ValueError,
                "list.remove(x): x not in list".to_string(),
            ));
        }
        FindOrCountResult::Count(_) => unreachable!("find_or_count with count=false returns Count"),
    };
    // listobject.py: `if i < self.length():  # otherwise list was mutated`
    let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let length = unsafe { pyre_object::w_list_len(obj) } as i64;
    if i < length {
        unsafe {
            pyre_object::listobject::w_list_pop(obj, i, true);
        }
    }
    Ok(())
}
