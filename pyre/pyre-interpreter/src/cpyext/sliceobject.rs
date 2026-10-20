//! `slice` -- PyPy `cpyext/sliceobject.py`.
//!
//! A slice mirror carries its three bounds as fields, which the comment above
//! `slice_attach` calls out: "Slice objects directly expose their members as
//! PyObject.  Don't change them!".  cffi reads them that way, and no entry
//! point stands in for the read.

use super::object::argument;
use super::pyobject::{self, CPyObject};
use super::typeobject::CPyTypeObject;
use pyre_object::PyObjectRef;
use std::ffi::c_int;

/// C-visible `PySliceObject`, the twin of the struct in
/// `include/pyre3.15t/sliceobject.h`.
///
/// Field order is `sliceobject.py PySliceObjectFields`, which names them
/// `start`, `step`, `stop`; the header spells the order CPython's does, and
/// the two agree because every reader names a field rather than an offset.
#[repr(C)]
pub struct CPySliceObject {
    pub ob_base: CPyObject,
    pub start: *mut CPyObject,
    pub stop: *mut CPyObject,
    pub step: *mut CPyObject,
}

/// The layout is written out twice -- here and in the header -- so a field
/// added to one without the other stops compiling rather than writing
/// somewhere unclaimed.
const _: () = {
    assert!(std::mem::offset_of!(CPySliceObject, ob_base) == 0);
    assert!(std::mem::offset_of!(CPySliceObject, start) == size_of::<CPyObject>());
    assert!(
        std::mem::offset_of!(CPySliceObject, stop) == size_of::<CPyObject>() + size_of::<usize>()
    );
    assert!(
        std::mem::offset_of!(CPySliceObject, step)
            == size_of::<CPyObject>() + 2 * size_of::<usize>()
    );
    assert!(size_of::<CPySliceObject>() == size_of::<CPyObject>() + 3 * size_of::<usize>());
};

/// The interpreter's slice type, or null before it is built.
fn slice_type() -> PyObjectRef {
    crate::typedef::gettypeobject(&pyre_object::sliceobject::SLICE_TYPE)
}

/// What `tp_basicsize` a synthesized mirror of `w_type` carries --
/// `sliceobject.py basestruct=PySliceObject.TO` for the slice type, and 0 for
/// every other type, which asks for the plain header.
pub(super) fn basicsize(w_type: PyObjectRef) -> isize {
    let slice_type = slice_type();
    match !slice_type.is_null() && std::ptr::eq(w_type, slice_type) {
        true => size_of::<CPySliceObject>() as isize,
        false => 0,
    }
}

/// A mirror's slice fields, or `None` when it is not a slice mirror.
///
/// The type decides it, as it does for a frame mirror and for the same reason:
/// a size test would answer yes for any C-defined type whose own storage
/// reaches this far.
fn fields(raw: *mut CPyObject) -> Option<*mut CPySliceObject> {
    if raw.is_null() {
        return None;
    }
    let tp = unsafe { (*raw).ob_type };
    // `as_pyobj` rather than a realizing read: this runs while a mirror is
    // being deallocated, where nothing may build an interpreter object.
    let slice_tp = pyobject::as_pyobj(slice_type()) as *mut CPyTypeObject;
    if tp.is_null() || slice_tp.is_null() || !std::ptr::eq(tp, slice_tp) {
        return None;
    }
    assert!(
        unsafe { (*tp).tp_basicsize } >= size_of::<CPySliceObject>() as isize,
        "a slice mirror was allocated at the plain PyObject size"
    );
    Some(raw as *mut CPySliceObject)
}

/// Fill a freshly allocated mirror of `w_obj` when `w_obj` is a slice --
/// `sliceobject.py slice_attach`.
pub(super) fn attach(raw: *mut CPyObject, w_obj: PyObjectRef) {
    if w_obj.is_null() || !unsafe { pyre_object::sliceobject::is_slice(w_obj) } {
        return;
    }
    let Some(py_slice) = fields(raw) else {
        return;
    };
    // `make_ref` allocates, so the slice is read back through the mirror
    // before each bound rather than kept in a local: the block's address does
    // not move and the slice's does.
    let slice = || unsafe { (*raw).ob_pyre_link };
    let start = pyobject::make_ref(unsafe { pyre_object::sliceobject::w_slice_get_start(slice()) });
    let stop = pyobject::make_ref(unsafe { pyre_object::sliceobject::w_slice_get_stop(slice()) });
    let step = pyobject::make_ref(unsafe { pyre_object::sliceobject::w_slice_get_step(slice()) });
    unsafe {
        (*py_slice).start = start;
        (*py_slice).stop = stop;
        (*py_slice).step = step;
    }
}

/// Release the references a slice mirror owns -- `sliceobject.py
/// slice_dealloc`.
pub(super) fn forget_block(raw: *mut CPyObject) {
    let Some(py_slice) = fields(raw) else {
        return;
    };
    unsafe {
        for reference in [(*py_slice).start, (*py_slice).stop, (*py_slice).step] {
            pyobject::decref(reference);
        }
        (*py_slice).start = std::ptr::null_mut();
        (*py_slice).stop = std::ptr::null_mut();
        (*py_slice).step = std::ptr::null_mut();
    }
}

/// `slice(low, high)` — the object the sequence-slice entry points hand to
/// `getitem`, as `space.getslice` does (`baseobjspace.py`).
///
/// The two bounds are held on the Rust stack across each other's allocation,
/// which is what `w_int_new`'s stable (old-gen) allocator exists to allow
/// (`gc_hook.rs` `register_gc_alloc_stable_hook`); `w_slice_new` pins them
/// itself before the slice's own allocation.
pub(super) fn range_slice(low: isize, high: isize) -> PyObjectRef {
    pyre_object::sliceobject::w_slice_new(
        pyre_object::w_int_new(low as i64),
        pyre_object::w_int_new(high as i64),
        pyre_object::w_none(),
    )
}

/// A NULL bound reads as `None` (`sliceobject.py:57-64`).
fn or_none(value: PyObjectRef) -> PyObjectRef {
    if value.is_null() {
        pyre_object::w_none()
    } else {
        value
    }
}

/// The `(start, stop, step)` of a `slice`, or a recorded `SystemError`.
///
/// `PyErr_BadInternalCall` is what every one of these answers for a non-slice
/// (`sliceobject.py:79-80`).
fn bounds(slice: *mut CPyObject) -> Option<(PyObjectRef, PyObjectRef, PyObjectRef)> {
    let value = argument(slice)?;
    if !unsafe { pyre_object::sliceobject::is_slice(value) } {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return None;
    }
    // Three field reads with nothing that allocates between them, so none of
    // the bounds can move before the caller consumes them.
    unsafe {
        Some((
            pyre_object::sliceobject::w_slice_get_start(value),
            pyre_object::sliceobject::w_slice_get_stop(value),
            pyre_object::sliceobject::w_slice_get_step(value),
        ))
    }
}

/// Reject a NULL out-parameter rather than writing through it.  Upstream
/// declares these `Py_ssize_t *` and does not test them; pyre does, for the
/// reason `PyIter_Send` does.
fn out_parameters(pointers: &[*mut isize]) -> bool {
    if pointers.iter().any(|pointer| pointer.is_null()) {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return false;
    }
    true
}

/// `PySlice_New(start, stop, step)` (`sliceobject.py`).
///
/// # Safety
/// Each argument must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySlice_New(
    start: *mut CPyObject,
    stop: *mut CPyObject,
    step: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([start, stop, step]);
    let start = or_none(unsafe { pyobject::from_ref(start) });
    let stop = or_none(unsafe { pyobject::from_ref(stop) });
    let step = or_none(unsafe { pyobject::from_ref(step) });
    pyobject::make_ref(pyre_object::sliceobject::w_slice_new(start, stop, step))
}

/// `PySlice_Unpack(slice, *start, *stop, *step)` — the bounds through
/// `__index__`, with no length consulted (`sliceobject.py:107-133`).
///
/// A step at the bottom of the range is clamped to `-PY_SSIZE_T_MAX`, so that
/// negating it stays representable.
///
/// # Safety
/// `slice` must be a live reference and the three out-parameters writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySlice_Unpack(
    slice: *mut CPyObject,
    start: *mut isize,
    stop: *mut isize,
    step: *mut isize,
) -> c_int {
    if !out_parameters(&[start, stop, step]) {
        return -1;
    }
    let Some((w_start, w_stop, w_step)) = bounds(slice) else {
        return -1;
    };
    let unpacked = crate::sliceobject::slice_unpack(w_start, w_stop, w_step);
    let Some((low, high, stride)) = super::pyerrors::trap(unpacked) else {
        return -1;
    };
    unsafe {
        *step = stride.max(-(isize::MAX as i64)) as isize;
        *start = low as isize;
        *stop = high as isize;
    }
    0
}

/// `PySlice_AdjustIndices(length, *start, *stop, step)` — clamp an unpacked
/// triple against `length` and answer the slice length
/// (`sliceobject.py` `adjust_indices`).
///
/// Upstream cannot fail; -1 here is the NULL-out-parameter rejection, which is
/// otherwise a store through a null pointer.
///
/// # Safety
/// `start` and `stop` must be readable and writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySlice_AdjustIndices(
    length: isize,
    start: *mut isize,
    stop: *mut isize,
    step: isize,
) -> isize {
    if !out_parameters(&[start, stop]) {
        return -1;
    }
    let (low, high, _, slicelength) = crate::sliceobject::slice_adjust_indices(
        unsafe { *start } as i64,
        unsafe { *stop } as i64,
        step as i64,
        length as i64,
    );
    unsafe {
        *start = low as isize;
        *stop = high as isize;
    }
    slicelength as isize
}

/// `PySlice_GetIndices(slice, length, *start, *stop, *step)`
/// (`sliceobject.py:86-104`).
///
/// # Safety
/// `slice` must be a live reference and the three out-parameters writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySlice_GetIndices(
    slice: *mut CPyObject,
    length: isize,
    start: *mut isize,
    stop: *mut isize,
    step: *mut isize,
) -> c_int {
    if !out_parameters(&[start, stop, step]) {
        return -1;
    }
    let Some((w_start, w_stop, w_step)) = bounds(slice) else {
        return -1;
    };
    let indices = crate::sliceobject::indices3(w_start, w_stop, w_step, length as i64);
    let Some((low, high, stride)) = super::pyerrors::trap(indices) else {
        return -1;
    };
    unsafe {
        *start = low as isize;
        *stop = high as isize;
        *step = stride as isize;
    }
    0
}

/// `PySlice_GetIndicesEx(slice, length, *start, *stop, *step, *slicelength)`
/// (`sliceobject.py:68-83`).
///
/// The header spells this as the macro over `PySlice_Unpack` and
/// `PySlice_AdjustIndices` that every extension compiles against; the exported
/// function is the same composition, so the two agree.
///
/// # Safety
/// `slice` must be a live reference and the four out-parameters writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySlice_GetIndicesEx(
    slice: *mut CPyObject,
    length: isize,
    start: *mut isize,
    stop: *mut isize,
    step: *mut isize,
    slicelength: *mut isize,
) -> c_int {
    if !out_parameters(&[slicelength]) {
        return -1;
    }
    if unsafe { PySlice_Unpack(slice, start, stop, step) } < 0 {
        unsafe { *slicelength = 0 };
        return -1;
    }
    unsafe { *slicelength = PySlice_AdjustIndices(length, start, stop, *step) };
    0
}

/// `PySlice_Check(ob)`.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySlice_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::sliceobject::is_slice(object) }) as c_int
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PySlice_New as *const ());
    std::hint::black_box(PySlice_Unpack as *const ());
    std::hint::black_box(PySlice_AdjustIndices as *const ());
    std::hint::black_box(PySlice_GetIndices as *const ());
    std::hint::black_box(PySlice_GetIndicesEx as *const ());
    std::hint::black_box(PySlice_Check as *const ());
}
