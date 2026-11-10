//! Native `memoryview` object — `W_MemoryView`
//! (`pypy/objspace/std/memoryobject.py`) over a [`BufferView`]
//! (`pypy/interpreter/buffer.py`) over a byte-level
//! [`Buffer`](crate::buffer::Buffer) (`rpython/rlib/buffer.py`).
//!
//! `W_MemoryView` holds an off-heap `*const BufferView`; the view carries the
//! geometry plus the backing `Buffer`, so the three layers stay distinct
//! rather than collapsed into one struct.  Byte access reads/writes the LIVE
//! backing object through the view's `Buffer` — no copy — so a view observes
//! later mutations of its source and writes through to it.
//!
//! Because the refs the collector must keep alive (the backing exporter, the
//! `.obj` exporter, the format / shape / strides objects) live *inside* the
//! off-heap view rather than as inline `PyObjectRef` fields, the macro-derived
//! `ptr_offsets` cannot reach them; the JIT driver registers
//! `memoryview_object_custom_trace` (`pyre-jit/src/eval.rs`) to walk the view,
//! mirroring `W_ListObject` / `W_TupleObject`'s off-block trace.

use crate::bufferview::BufferView;
use crate::pyobject::*;
use pyre_macros::pyre_class;

/// A `memoryview` — a view over a byte backing, behind an off-heap
/// [`BufferView`].
#[pyre_class("memoryview", static_name = "MEMORYVIEW", weakrefable)]
pub struct W_MemoryView {
    /// `self.view` (`memoryobject.py`).  Never null for a live memoryview; the
    /// geometry and backing live here, off the GC heap, reached by the custom
    /// trace.
    pub view: *const BufferView,
    /// `self._hash` — the cached content hash, `-1` until computed
    /// (`memoryobject.py:92`).
    pub w_hash: i64,
    /// Flips on `release()` / context-manager exit.
    pub released: bool,
    /// CPython 3.14 `PyMemoryViewObject.exports`: active buffers exported by
    /// this view.  `release()` is forbidden while this is positive; hash and
    /// contiguous hex temporarily increment it across re-entrant callbacks.
    pub exports: i64,
    /// `W_Root`'s per-object weakref lifeline.  PyPy's memoryview typedef
    /// installs `make_weakref_descr(W_MemoryView)`, so the lifeline belongs
    /// to this exact view and dies with it; it must not live in a global
    /// side-table.
    pub w_weakreflifeline: PyObjectRef,
    /// CPython 3.14 `_Py_MEMORYVIEW_RESTRICTED`: a temporary view handed to
    /// a Python `__release_buffer__` implementation for a native buffer
    /// slot.  It remains readable during the callback, but may not be used
    /// to manufacture another view that outlives the release.
    pub restricted: bool,
    /// `owns_export` (`memoryobject.py`).  True for a view built directly over
    /// an exporter (holds one `_exports` count on the backing); false for a
    /// derived view (slice, copy, cast, `toreadonly`), which `memoryobject.py`
    /// builds as `W_MemoryView(view, owns_export=False)` and which shares the
    /// original's export.  Only an owning view releases the underlying export.
    pub owns_export: bool,
}

/// CPython 3.14 `PyBufferWrapper` (`Objects/typeobject.c`): the owner placed
/// in a `Py_buffer` when a Python `__buffer__` method supplies the actual
/// memoryview.  Keeping this as a real GC object (rather than a side table or
/// fields flattened onto `W_MemoryView`) preserves both `.obj` identity and
/// the upstream `mv`/`obj` lifetime shape.
#[pyre_class("_buffer_wrapper", static_name = "BUFFER_WRAPPER")]
pub struct W_BufferWrapper {
    /// The exact memoryview returned by `obj.__buffer__(flags)`.
    pub w_mv: PyObjectRef,
    /// The original Python buffer exporter.
    pub w_obj: PyObjectRef,
}

/// Allocate a [`BufferView`] off the GC heap (a stationary `Box`, like a
/// list's `ItemsBlock`).  The collector reaches its refs through
/// `memoryview_object_custom_trace` and never moves the box.
pub fn bufferview_alloc(view: BufferView) -> *const BufferView {
    Box::into_raw(Box::new(view)) as *const BufferView
}

/// Allocate the `W_MemoryView` header in GC old-gen with no view attached
/// yet (mirrors `w_set_new` allocating an empty body).
///
/// Old-gen (`try_gc_alloc_stable`, non-moving mark-sweep) so the object
/// carries `TRACK_YOUNG_PTRS` and `memoryview_object_custom_trace` runs on a
/// minor collection once the header sits in the remembered set — without
/// that, the managed `format` / `shape` / `strides` objects reachable only
/// through the view's off-heap box would be swept.  Allocating the header
/// *before* the ref-bearing box keeps no half-built box alive across the
/// collection this call may trigger: the caller re-reads its pinned refs
/// from the shadow stack (post-relocation) and attaches the box with
/// [`w_memoryview_set_view`].  Falls back to `malloc_typed` when no GC hook
/// is installed (unit tests).
pub fn w_memoryview_alloc_header(released: bool, owns_export: bool) -> PyObjectRef {
    let payload = W_MemoryView {
        ob: PyObject {
            ob_type: &MEMORYVIEW_TYPE as *const PyType,
            w_class: crate::pyobject::get_instantiate(&MEMORYVIEW_TYPE),
        },
        view: std::ptr::null(),
        w_hash: -1,
        released,
        exports: 0,
        w_weakreflifeline: PY_NULL,
        restricted: false,
        owns_export,
    };
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(
        <W_MemoryView as crate::lltype::GcType>::type_id(),
        <W_MemoryView as crate::lltype::GcType>::SIZE,
    );
    if !raw.is_null() {
        unsafe { std::ptr::write(raw as *mut W_MemoryView, payload) };
        // Nursery-full spill lands in old-gen with TRACK_YOUNG_PTRS still
        // set (`spill_to_oldgen_or_null`). `w_class` is an old-gen type.
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    } else {
        if crate::gc_hook::gc_alloc_failed_pending() {
            // W_MemoryView.__init__ / copy: a refused header is fatal; NoRoute
            // still uses the host fallback.
            crate::gc_hook::gc_alloc_failed(<W_MemoryView as crate::lltype::GcType>::SIZE);
        } else {
            crate::lltype::malloc_typed(payload) as PyObjectRef
        }
    }
}

/// Allocate CPython's `_buffer_wrapper(mv, obj)`, rooting both constructor
/// arguments across the managed allocation.
///
/// Managed and non-moving, the same shape `w_memoryview_new` above uses: the
/// wrapper's only job is to hold `w_mv`/`w_obj` alive, and a `malloc_typed`
/// immortal is outside the sweep set, so the marker latches GCFLAG_VISITED on
/// it at the first major and never re-marks those two children afterwards.
pub fn w_buffer_wrapper_new(w_mv: PyObjectRef, w_obj: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let sp = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_mv);
    let _ = crate::gc_roots::pin_root(w_obj);
    W_BufferWrapper::allocate_stable(W_BufferWrapper {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_mv: crate::gc_roots::shadow_stack_get(sp),
        w_obj: crate::gc_roots::shadow_stack_get(sp + 1),
    })
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_w_buffer_wrapper(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BUFFER_WRAPPER_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_buffer_wrapper_mv(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BufferWrapper)).w_mv }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_buffer_wrapper_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BufferWrapper)).w_obj }
}

/// `Py_CLEAR(bw->mv); Py_CLEAR(bw->obj)` after the wrapper's single release.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_buffer_wrapper_clear(obj: PyObjectRef) {
    unsafe {
        let wrapper = obj as *mut W_BufferWrapper;
        (*wrapper).w_mv = PY_NULL;
        (*wrapper).w_obj = PY_NULL;
    }
}

/// Attach the off-heap view to a header from [`w_memoryview_alloc_header`]
/// and fire the GC write barrier so the old-gen `W_MemoryView` enters the
/// remembered set; `memoryview_object_custom_trace` then forwards the view's
/// refs on the next minor collection.  Mirrors `set_write_barrier` after a
/// store into a set's off-heap items.
///
/// # Safety
/// `mv` must point to a `W_MemoryView` from [`w_memoryview_alloc_header`] and
/// `view` to a live [`bufferview_alloc`] box whose `PyObjectRef`s are already
/// the post-collection (relocated) pointers.
pub unsafe fn w_memoryview_set_view(mv: PyObjectRef, view: *const BufferView) {
    unsafe {
        (*(mv as *mut W_MemoryView)).view = view;
    }
    crate::gc_hook::try_gc_write_barrier(mv as crate::gc_hook::GCREF);
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_w_memoryview(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &MEMORYVIEW_TYPE) }
}

/// The off-heap view backing `obj`.
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView` whose `view` is live.  Once
/// `w_memoryview_set_released` drops the box and nulls `view`, this
/// dereferences a null pointer — every accessor must gate on
/// `w_memoryview_released` first (as the interpreter methods do).
#[inline]
pub unsafe fn w_memoryview_view(obj: PyObjectRef) -> &'static BufferView {
    unsafe { &*(*(obj as *const W_MemoryView)).view }
}

macro_rules! mv_view_obj {
    ($name:ident, $accessor:ident) => {
        /// # Safety
        /// `obj` must point to a valid `W_MemoryView`.
        #[inline]
        pub unsafe fn $name(obj: PyObjectRef) -> PyObjectRef {
            unsafe { w_memoryview_view(obj).$accessor() }
        }
    };
}
macro_rules! mv_view_scalar {
    ($name:ident, $accessor:ident, $ty:ty) => {
        /// # Safety
        /// `obj` must point to a valid `W_MemoryView`.
        #[inline]
        pub unsafe fn $name(obj: PyObjectRef) -> $ty {
            unsafe { w_memoryview_view(obj).$accessor() }
        }
    };
}

/// The exporter actually read/written (bytes / bytearray / array.array) — the
/// `.obj` of the root storage.
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_backing(obj: PyObjectRef) -> PyObjectRef {
    unsafe { w_memoryview_view(obj).backing().w_obj() }
}

mv_view_obj!(w_memoryview_obj, w_obj);
mv_view_scalar!(w_memoryview_itemsize, itemsize, i64);
mv_view_scalar!(w_memoryview_ndim, ndim, i64);
mv_view_scalar!(w_memoryview_offset, offset, i64);
mv_view_scalar!(w_memoryview_length, length, i64);
mv_view_scalar!(w_memoryview_readonly, readonly, bool);

/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_released(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_MemoryView)).released }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_memoryview_restricted(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_MemoryView)).restricted }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_memoryview_set_restricted(obj: PyObjectRef, restricted: bool) {
    unsafe {
        (*(obj as *mut W_MemoryView)).restricted = restricted;
    }
}

/// Number of buffers currently exported by this memoryview.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_memoryview_exports(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_MemoryView)).exports }
}

/// Temporarily protect the view from `release()` across a callback.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_memoryview_exports_incref(obj: PyObjectRef) {
    unsafe { (*(obj as *mut W_MemoryView)).exports += 1 }
}

/// End a temporary/exported-buffer protection interval.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_memoryview_exports_decref(obj: PyObjectRef) {
    unsafe {
        let mv = obj as *mut W_MemoryView;
        debug_assert!((*mv).exports > 0);
        (*mv).exports -= 1;
    }
}

/// The cached content hash (`self._hash`), `-1` until computed.
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_hash(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_MemoryView)).w_hash }
}

/// Store the computed content hash.  A plain scalar store — no write
/// barrier (a barrier guards `PyObjectRef` stores only).
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_set_hash(obj: PyObjectRef, hash: i64) {
    unsafe {
        (*(obj as *mut W_MemoryView)).w_hash = hash;
    }
}

/// `owns_export` — whether this view holds an `_exports` count on its backing
/// (true for a root view over an exporter, false for a derived view).
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_owns_export(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_MemoryView)).owns_export }
}

/// Flip `released` without touching the view box, so a re-entrant `release()`
/// is already a no-op while the view is still readable.  The exporter's
/// `__release_buffer__` runs in exactly that window: it is handed this
/// memoryview and reads its backing to identify the export it is undoing.
/// The flag is an inline scalar, so no write barrier is needed (a barrier
/// guards `PyObjectRef` stores only).
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_mark_released(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_MemoryView)).released = true;
    }
}

/// Drop the off-heap `BufferView` box (reclaiming any nested `Buffer::Sub`
/// boxes through `Box`'s recursive drop glue) and null `view`.  Mirrors
/// `descr_release` clearing `self.view = None` (`memoryobject.py`) so the
/// backing / view graph is dropped eagerly on release rather than lingering
/// until the header is GC-collected.  Idempotent: a second call finds `view`
/// already null.
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`, and no borrow handed out by
/// [`w_memoryview_view`] may outlive this call.
#[inline]
pub unsafe fn w_memoryview_drop_view(obj: PyObjectRef) {
    unsafe {
        let mv = obj as *mut W_MemoryView;
        let view_ptr = (*mv).view as *mut BufferView;
        if !view_ptr.is_null() {
            drop(Box::from_raw(view_ptr));
            (*mv).view = std::ptr::null();
        }
    }
}

/// Release the view: flip `released` and drop the view box together, for the
/// paths that run no exporter callback in between.
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_set_released(obj: PyObjectRef) {
    unsafe {
        w_memoryview_mark_released(obj);
        w_memoryview_drop_view(obj);
    }
}

/// `strides[0]` — the signed byte step between consecutive elements of a
/// 1-D view.  A contiguous view has `strides[0] == itemsize`; a strided
/// slice (`m[::2]`, `m[::-1]`) carries `parent_stride * step`, possibly
/// negative.  Falls back to `itemsize` when the tuple is unexpectedly empty.
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_stride0(obj: PyObjectRef) -> i64 {
    unsafe { w_memoryview_view(obj).stride0() }
}

/// The view's format string content (`memoryview.format`) read without
/// wrapping a fresh string object — for callers that only need the bytes.
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_format_str(obj: PyObjectRef) -> &'static str {
    unsafe { w_memoryview_view(obj).format_str() }
}

/// The view's shape as native `i64` extents (`memoryview.shape`).
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_native_shape(obj: PyObjectRef) -> Vec<i64> {
    unsafe { w_memoryview_view(obj).native_shape() }
}

/// The view's strides as native `i64` byte steps (`memoryview.strides`).
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_native_strides(obj: PyObjectRef) -> Vec<i64> {
    unsafe { w_memoryview_view(obj).native_strides() }
}

/// The view's suboffsets as native `i64` byte offsets, empty for a view that
/// has none (`memoryview.suboffsets`).
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_native_suboffsets(obj: PyObjectRef) -> Vec<i64> {
    unsafe { w_memoryview_view(obj).native_suboffsets() }
}
