//! `pypy/interpreter/generator.py` — Python generator iterator.
//!
//! Wraps a suspended frame. __next__() resumes the frame until
//! YIELD_VALUE (produces a value) or RETURN_VALUE (raises StopIteration).
//!
//! # Safety
//! The `w_generator_*` / `w_coroutine_*` accessors below cast their argument
//! to [`GeneratorIterator`] and read or write a field at a fixed offset, with
//! no kind check of their own. Each one therefore requires `obj` to point at a
//! live `GeneratorIterator` — a generator, coroutine or async generator. Any
//! other object reads or, worse, writes an unrelated byte of its payload.

use crate::pyobject::*;
use pyre_macros::pyre_class;

pub static GENERATOR_TYPE: PyType = crate::pyobject::new_pytype_with_weakref(
    "generator",
    std::mem::offset_of!(GeneratorIterator, lifeline),
);
pub static COROUTINE_TYPE: PyType = crate::pyobject::new_pytype_with_weakref(
    "coroutine",
    std::mem::offset_of!(GeneratorIterator, lifeline),
);
pub static ASYNC_GENERATOR_TYPE: PyType = crate::pyobject::new_pytype_with_weakref(
    "async_generator",
    std::mem::offset_of!(GeneratorIterator, lifeline),
);

/// Generator object: holds a boxed frame that can be resumed.
///
/// The frame is stored as a raw pointer to avoid generic type parameters
/// in the object layout (keeps it JIT-compatible).
#[repr(C)]
pub struct GeneratorIterator {
    pub ob: PyObject,
    /// Opaque pointer to the suspended PyFrame (Box<PyFrame>).
    /// NULL when the generator is exhausted. `llmemory.GCREF`.
    pub frame_ptr: crate::gc_hook::GCREF,
    /// `generator.py` `self.pycode = frame.pycode`.  This remains owned by
    /// the generator after `frame` is cleared on exhaustion.
    pub pycode: PyObjectRef,
    /// Whether the generator has been started (first __next__ called).
    pub started: bool,
    /// Whether the generator is exhausted.
    pub exhausted: bool,
    /// Whether the generator is currently executing (prevents reentrant calls).
    /// PyPy: GeneratorIterator.running
    pub running: bool,
    /// Per-generator writable `__name__` override. NULL means read the
    /// suspended frame code's original name.
    pub name: PyObjectRef,
    /// Per-generator writable `__qualname__` override. NULL means read the
    /// suspended frame code's original qualified name.
    pub qualname: PyObjectRef,
    /// PyPy: `Coroutine.w_cr_origin`. Generators keep this at `PY_NULL`;
    /// coroutines initialise it to `None` until origin tracking captures a
    /// tuple of frame summaries.
    pub cr_origin: PyObjectRef,
    /// PyPy: `Coroutine._warned_unawaited`.
    pub warned_unawaited: bool,
    /// `generator.py` `GeneratorOrCoroutine.saved_operr`.  Pyre stores
    /// the materialized exception value carried by `ExecutionContext` rather
    /// than an `OperationError`, but preserves the same per-generator owner.
    pub saved_exc_value: PyObjectRef,
    /// `generator.py` `previous_gen_or_coroutine`: the execution-context
    /// linked-list edge while this generator is running.
    pub previous_gen_or_coroutine: PyObjectRef,
    /// `AsyncGenerator.hooks_inited` / `ag_running` / `w_finalizer`.
    /// Generator and coroutine instances keep the neutral values.
    pub hooks_inited: bool,
    pub ag_running: bool,
    pub w_finalizer: PyObjectRef,
    /// `generator.py` `make_weakref_descr` `_lifeline_`, shared by generator,
    /// coroutine and async-generator instances.
    pub lifeline: PyObjectRef,
}

/// PyPy `generator.py CoroutineWrapper`: the iterator returned by
/// `Coroutine.__await__`, holding the coroutine as its sole GC edge.
#[pyre_class("coroutine_wrapper", static_name = "COROUTINE_WRAPPER")]
pub struct CoroutineWrapper {
    pub coroutine: PyObjectRef,
}

/// PyPy `generator.py AsyncGenValueWrapper`: distinguishes a value yielded
/// by an async generator from a value yielded by an await inside its body.
#[pyre_class(
    "async_generator_wrapped_value",
    static_name = "ASYNC_GEN_VALUE_WRAPPER"
)]
pub struct AsyncGenValueWrapper {
    pub w_value: PyObjectRef,
}

/// Shared state values from PyPy `AsyncGenABase`.
pub const ASYNC_GEN_STATE_INIT: u8 = 0;
pub const ASYNC_GEN_STATE_ITER: u8 = 1;
pub const ASYNC_GEN_STATE_CLOSED: u8 = 2;

#[pyre_class("async_generator_asend", static_name = "ASYNC_GEN_ASEND")]
pub struct AsyncGenASend {
    pub async_gen: PyObjectRef,
    pub w_value_to_send: PyObjectRef,
    pub state: u8,
}

#[pyre_class("async_generator_athrow", static_name = "ASYNC_GEN_ATHROW")]
pub struct AsyncGenAThrow {
    pub async_gen: PyObjectRef,
    /// `PY_NULL` identifies `aclose()`; otherwise these are the arguments to
    /// `athrow()` and optional values use Python `None`.
    pub w_exc_type: PyObjectRef,
    pub w_exc_value: PyObjectRef,
    pub w_exc_tb: PyObjectRef,
    pub state: u8,
}

/// GC type id assigned to `GeneratorIterator` at JitDriver init time.
pub const W_GENERATOR_GC_TYPE_ID: u32 = 32;

/// Fixed payload size (`framework.py:811`).
pub const W_GENERATOR_OBJECT_SIZE: usize = std::mem::size_of::<GeneratorIterator>();

impl crate::lltype::GcType for GeneratorIterator {
    fn type_id() -> u32 {
        W_GENERATOR_GC_TYPE_ID
    }
    const SIZE: usize = W_GENERATOR_OBJECT_SIZE;
}

fn w_generator_or_coroutine_new(
    frame_ptr: crate::gc_hook::GCREF,
    pycode: PyObjectRef,
    kind: GeneratorKind,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let pycode_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(pycode);
    let frame_owned = !frame_ptr.is_null() && crate::gc_hook::try_gc_owns_object(frame_ptr);
    // A static pin: an unmanaged frame's slot holds null.
    let frame_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(if frame_owned {
        frame_ptr as PyObjectRef
    } else {
        PY_NULL
    });
    let (ob_type, w_class) = match kind {
        GeneratorKind::Generator => (
            &GENERATOR_TYPE as *const PyType,
            get_instantiate(&GENERATOR_TYPE),
        ),
        GeneratorKind::Coroutine => (
            &COROUTINE_TYPE as *const PyType,
            get_instantiate(&COROUTINE_TYPE),
        ),
        GeneratorKind::AsyncGenerator => (
            &ASYNC_GENERATOR_TYPE as *const PyType,
            get_instantiate(&ASYNC_GENERATOR_TYPE),
        ),
    };
    let pycode = crate::gc_roots::shadow_stack_get(pycode_slot);
    let frame_ptr = if frame_owned {
        crate::gc_roots::shadow_stack_get(frame_slot) as crate::gc_hook::GCREF
    } else {
        frame_ptr
    };
    let value = GeneratorIterator {
        ob: PyObject { ob_type, w_class },
        frame_ptr,
        pycode,
        started: false,
        exhausted: false,
        running: false,
        name: PY_NULL,
        qualname: PY_NULL,
        cr_origin: if kind == GeneratorKind::Coroutine {
            crate::w_none()
        } else {
            PY_NULL
        },
        warned_unawaited: false,
        saved_exc_value: PY_NULL,
        previous_gen_or_coroutine: PY_NULL,
        hooks_inited: false,
        ag_running: false,
        w_finalizer: PY_NULL,
        lifeline: PY_NULL,
    };
    // A generator must be GC-managed, not immortal `malloc_typed`: the
    // collector never reaches an immortal object, so the registered
    // `generator_object_custom_trace` (which walks the SUSPENDED frame's
    // locals/cells/valuestack via `walk_suspended_generator_frame`) would
    // never run, and a value live only across a `yield` would be reclaimed by
    // a major collection — resuming the generator then dereferences freed
    // memory. Same `malloc_fixedsize` bump as `w_exception_new_empty_impl`.
    // `generator_object_custom_trace` already forwards `frame_ptr` as a
    // GCREF when the frame is GC-owned. Fall back to immortal only when
    // the GC is not installed.
    let raw =
        crate::gc_hook::try_gc_alloc_nursery_raw(W_GENERATOR_GC_TYPE_ID, W_GENERATOR_OBJECT_SIZE);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(raw as *mut GeneratorIterator, value);
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    crate::lltype::malloc_typed(value) as PyObjectRef
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GeneratorKind {
    Generator,
    Coroutine,
    AsyncGenerator,
}

pub fn w_generator_new(frame_ptr: crate::gc_hook::GCREF, pycode: PyObjectRef) -> PyObjectRef {
    w_generator_or_coroutine_new(frame_ptr, pycode, GeneratorKind::Generator)
}

pub fn w_coroutine_new(frame_ptr: crate::gc_hook::GCREF, pycode: PyObjectRef) -> PyObjectRef {
    w_generator_or_coroutine_new(frame_ptr, pycode, GeneratorKind::Coroutine)
}

pub fn w_async_generator_new(frame_ptr: crate::gc_hook::GCREF, pycode: PyObjectRef) -> PyObjectRef {
    w_generator_or_coroutine_new(frame_ptr, pycode, GeneratorKind::AsyncGenerator)
}

pub fn w_coroutine_wrapper_new(coroutine: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let coroutine = crate::gc_roots::pin_root(coroutine);
    CoroutineWrapper::allocate_stable(CoroutineWrapper {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        coroutine,
    })
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_generator(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &GENERATOR_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_coroutine(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &COROUTINE_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_async_generator(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ASYNC_GENERATOR_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_async_gen_asend(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ASYNC_GEN_ASEND_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_async_gen_athrow(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ASYNC_GEN_ATHROW_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_generator_or_coroutine(obj: PyObjectRef) -> bool {
    unsafe { is_generator(obj) || is_coroutine(obj) || is_async_generator(obj) }
}

pub fn w_async_gen_value_wrapper_new(w_value: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let w_value = crate::gc_roots::pin_root(w_value);
    AsyncGenValueWrapper::allocate_stable(AsyncGenValueWrapper {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_value,
    })
}

pub fn w_async_gen_asend_new(async_gen: PyObjectRef, w_value_to_send: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(async_gen);
    let _ = crate::gc_roots::pin_root(w_value_to_send);
    let async_gen = crate::gc_roots::shadow_stack_get(root_base);
    let w_value_to_send = crate::gc_roots::shadow_stack_get(root_base + 1);
    AsyncGenASend::allocate_stable(AsyncGenASend {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        async_gen,
        w_value_to_send,
        state: ASYNC_GEN_STATE_INIT,
    })
}

pub fn w_async_gen_athrow_new(
    async_gen: PyObjectRef,
    w_exc_type: PyObjectRef,
    w_exc_value: PyObjectRef,
    w_exc_tb: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    for value in [async_gen, w_exc_type, w_exc_value, w_exc_tb] {
        let _ = crate::gc_roots::pin_root(value);
    }
    let async_gen = crate::gc_roots::shadow_stack_get(root_base);
    let w_exc_type = crate::gc_roots::shadow_stack_get(root_base + 1);
    let w_exc_value = crate::gc_roots::shadow_stack_get(root_base + 2);
    let w_exc_tb = crate::gc_roots::shadow_stack_get(root_base + 3);
    AsyncGenAThrow::allocate_stable(AsyncGenAThrow {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        async_gen,
        w_exc_type,
        w_exc_value,
        w_exc_tb,
        state: ASYNC_GEN_STATE_INIT,
    })
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_coroutine_wrapper(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &COROUTINE_WRAPPER_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_coroutine_wrapper_get_coroutine(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const CoroutineWrapper)).coroutine }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_get_frame(obj: PyObjectRef) -> crate::gc_hook::GCREF {
    unsafe { (*(obj as *const GeneratorIterator)).frame_ptr }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_get_pycode(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).pycode }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_set_frame(obj: PyObjectRef, frame_ptr: crate::gc_hook::GCREF) {
    unsafe { (*(obj as *mut GeneratorIterator)).frame_ptr = frame_ptr };
    if !frame_ptr.is_null() {
        crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_is_exhausted(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const GeneratorIterator)).exhausted }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_set_exhausted(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut GeneratorIterator)).exhausted = true;
    }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_is_started(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const GeneratorIterator)).started }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_set_started(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut GeneratorIterator)).started = true;
    }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_coroutine_warned_unawaited(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const GeneratorIterator)).warned_unawaited }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_coroutine_set_warned_unawaited(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut GeneratorIterator)).warned_unawaited = true;
    }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_is_running(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const GeneratorIterator)).running }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_set_running(obj: PyObjectRef, val: bool) {
    unsafe {
        (*(obj as *mut GeneratorIterator)).running = val;
    }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_get_saved_exc_value(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).saved_exc_value }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_set_saved_exc_value(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GeneratorIterator)).saved_exc_value = value };
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_get_previous(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).previous_gen_or_coroutine }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_set_previous(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GeneratorIterator)).previous_gen_or_coroutine = value };
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_get_name(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).name }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_set_name(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GeneratorIterator)).name = value };
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_get_qualname(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).qualname }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_generator_set_qualname(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GeneratorIterator)).qualname = value };
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_coroutine_get_origin(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).cr_origin }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_coroutine_set_origin(obj: PyObjectRef, origin: PyObjectRef) {
    unsafe { (*(obj as *mut GeneratorIterator)).cr_origin = origin };
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_async_generator_hooks_inited(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const GeneratorIterator)).hooks_inited }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_async_generator_set_hooks_inited(obj: PyObjectRef) {
    unsafe { (*(obj as *mut GeneratorIterator)).hooks_inited = true };
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_async_generator_is_running(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const GeneratorIterator)).ag_running }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_async_generator_set_running(obj: PyObjectRef, value: bool) {
    unsafe { (*(obj as *mut GeneratorIterator)).ag_running = value };
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_async_generator_get_finalizer(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).w_finalizer }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_async_generator_set_finalizer(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GeneratorIterator)).w_finalizer = value };
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_generator_gc_type_id_matches_descr() {
        assert_eq!(W_GENERATOR_GC_TYPE_ID, 32);
        assert_eq!(
            <GeneratorIterator as crate::lltype::GcType>::type_id(),
            W_GENERATOR_GC_TYPE_ID
        );
        assert_eq!(
            <GeneratorIterator as crate::lltype::GcType>::SIZE,
            W_GENERATOR_OBJECT_SIZE
        );
    }
}
