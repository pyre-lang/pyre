//! `_queue` accelerator module.
//!
//! PRE-EXISTING-ADAPTATION: PyPy does NOT ship this module. `pypy/module/`
//! has no `_queue`, and `lib-python/3/queue.py` falls back to its own
//! `_PySimpleQueue` when the import fails, so upstream's `SimpleQueue` is
//! pure Python. The spec here is therefore CPython's
//! `Modules/_queuemodule.c`, not an RPython module.
//!
//! The `rthread` citations below are an ANALOGY for where the JIT boundary
//! belongs — a native blocking primitive is residual in RPython too — and
//! not a claim of provenance for this type.

use parking_lot::{Condvar, Mutex, MutexGuard};
use pyre_object::*;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// CPython 3.14 Modules/_queuemodule.c:_queue_exec uses
// PyType_FromModuleAndSpec with IMMUTABLETYPE.
#[pyre_interpreter::pyre_class("_queue.SimpleQueue", cpython_heaptype, weakrefable)]
#[derive(Default)]
pub struct W_SimpleQueue {
    pub map: pyre_object::mapdict::MapRef,
    pub storage: *mut pyre_object::object_array::ItemsBlock,
    queue: Mutex<VecDeque<PyObjectRef>>,
    not_empty: Condvar,
}

const _: () = assert!(
    std::mem::offset_of!(W_SimpleQueue, map)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, map),
    "W_SimpleQueue must keep W_ObjectObject's map offset"
);
const _: () = assert!(
    std::mem::offset_of!(W_SimpleQueue, storage)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, storage),
    "W_SimpleQueue must keep W_ObjectObject's storage offset"
);

/// Only the native acquire is opaque to the tracer; queue operations performed
/// while the guard is held stay look-inside.  This is the same split as
/// `pyre_object::listobject::w_list_lock` and PyPy's `rthread.Lock`: the
/// wrapper remains RPython while `c_thread_acquirelock{,_timed}` is an
/// `llexternal` (`rpython/rlib/rthread.py`).  Rust's
/// `Result<Guard, PoisonError<Guard>>` is consequently an implementation ABI
/// inside the residual native-lock leaf, not a trace-visible value shape.
#[majit_macros::dont_look_inside]
fn queue_lock<'a>(
    mutex: &'a Mutex<VecDeque<PyObjectRef>>,
) -> MutexGuard<'a, VecDeque<PyObjectRef>> {
    if let Some(guard) = mutex.try_lock() {
        return guard;
    }
    let blocked = pyre_interpreter::module::thread::before_external_block();
    let guard = mutex.lock();
    drop(blocked);
    guard
}

#[majit_macros::dont_look_inside]
fn queue_push_back(mutex: &Mutex<VecDeque<PyObjectRef>>, base: usize) {
    // Read the pinned slot after `queue_lock` returns: argument
    // evaluation would snapshot the pointer before the residual
    // acquire, and `before_external_block` can collect.
    queue_lock(mutex).push_back(pyre_object::gc_roots::shadow_stack_get(base));
}

#[majit_macros::dont_look_inside]
fn queue_pop_front(mutex: &Mutex<VecDeque<PyObjectRef>>) -> Option<PyObjectRef> {
    queue_lock(mutex).pop_front()
}

#[majit_macros::dont_look_inside]
fn queue_is_empty(mutex: &Mutex<VecDeque<PyObjectRef>>) -> bool {
    queue_lock(mutex).is_empty()
}

#[majit_macros::dont_look_inside]
fn queue_len(mutex: &Mutex<VecDeque<PyObjectRef>>) -> usize {
    queue_lock(mutex).len()
}

/// `_queue_SimpleQueue_get_impl` reads `timeout` only on the blocking path:
/// `block=False` is answered from the queue immediately, so the argument is
/// neither converted nor range-checked there.
///
/// `_PyTime_FromSecondsObject` runs before the sign check, so a value it cannot
/// represent as a nanosecond timestamp is refused rather than turned into a
/// wait: an infinity would otherwise block forever and a NaN would poll once.
/// The conversion is the one `_thread.lock.acquire` already performs
/// (`module/thread/mod.rs parse_acquire_args`).
fn parse_timeout(
    block: bool,
    timeout: PyObjectRef,
) -> Result<Option<f64>, pyre_interpreter::PyError> {
    if !block {
        return Ok(None);
    }
    if timeout.is_null() || unsafe { pyre_object::is_none(timeout) } {
        return Ok(None);
    }
    let seconds = pyre_interpreter::baseobjspace::float_w(timeout)?;
    if seconds.is_nan() {
        return Err(pyre_interpreter::PyError::value_error(
            "Invalid value NaN (not a number)",
        ));
    }
    // `rarithmetic.ovfcheck_float_to_longlong` bounds, as in `parse_acquire_args`.
    const NS_MIN: f64 = -9223372036854776832.0;
    const NS_MAX: f64 = 9223372036854775296.0;
    if !(NS_MIN..NS_MAX).contains(&(seconds * 1e9).ceil()) {
        return Err(pyre_interpreter::PyError::overflow_error(
            "timestamp out of range for platform time_t",
        ));
    }
    if seconds < 0.0 {
        return Err(pyre_interpreter::PyError::value_error(
            "'timeout' must be a non-negative number",
        ));
    }
    Ok(Some(seconds))
}

/// `parse_timeout` accepted only a finite, non-negative number of seconds, so
/// the deadline is always representable and `None` means "wait forever".
fn deadline_from_timeout(timeout: Option<f64>) -> Option<Instant> {
    timeout.map(|seconds| Instant::now() + Duration::from_secs_f64(seconds))
}

fn empty_error() -> pyre_interpreter::PyError {
    let Some(cls) = pyre_interpreter::builtins::lookup_exc_class("_queue.Empty") else {
        return pyre_interpreter::PyError::runtime_error("");
    };
    pyre_interpreter::PyError::from_type_and_value(cls, pyre_object::PY_NULL)
}

fn simplequeue_put(queue: &W_SimpleQueue, item: PyObjectRef) -> PyObjectRef {
    // `_PySimpleQueue.put` keeps `item` as a frame local across the append.
    // There is no `pypy/module/_queue`; this native `put` is the same live
    // word.  `queue_lock` can take `before_external_block` (`rffi.aroundstate
    // .before` / `_RPyGilRelease`), which leaves the census so another
    // mutator can collect before `push_back`.  Same-thread reachability
    // cannot see that collect, so the bracket is kept even when the
    // intra-function scan reports none.
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(item);
    queue_push_back(&queue.queue, base);
    // The deque is storage `w_simplequeue_custom_trace` walks. The queue is
    // old (`allocate_stable`), so the item it now holds reaches the next
    // minor only through the barrier's remembered set.
    pyre_object::gc_hook::try_gc_write_barrier(
        queue as *const W_SimpleQueue as pyre_object::gc_hook::GCREF,
    );
    queue.not_empty.notify_one();
    w_none()
}

/// Wait until the native queue condition has made an item available.
///
/// This is the synchronization primitive, corresponding to the external
/// acquire beneath PyPy's `rthread.Lock`/semaphore.  The caller keeps timeout
/// parsing, the non-blocking fast path, and the eventual deque pop visible to
/// the trace; only the host mutex/condvar wait and Rust poison ABI are opaque.
#[majit_macros::dont_look_inside]
fn simplequeue_wait_for_item(
    queue: &W_SimpleQueue,
    timeout: Option<f64>,
) -> Result<MutexGuard<'_, VecDeque<PyObjectRef>>, pyre_interpreter::PyError> {
    let mut guard = queue_lock(&queue.queue);
    let deadline = deadline_from_timeout(timeout);
    loop {
        if !guard.is_empty() {
            return Ok(guard);
        }
        let blocked = pyre_interpreter::module::thread::before_external_block();
        if let Some(deadline) = deadline {
            let now = Instant::now();
            if now >= deadline {
                drop(blocked);
                return Err(empty_error());
            }
            let result = queue.not_empty.wait_for(&mut guard, deadline - now);
            drop(blocked);
            if result.timed_out() && guard.is_empty() {
                return Err(empty_error());
            }
        } else {
            queue.not_empty.wait(&mut guard);
            drop(blocked);
        }
    }
}

fn simplequeue_get(
    queue: &W_SimpleQueue,
    block: bool,
    timeout: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let timeout = parse_timeout(block, timeout)?;
    if block {
        return simplequeue_wait_and_pop(queue, timeout);
    }
    queue_pop_front(&queue.queue).ok_or_else(empty_error)
}

/// Residual wait+pop so `MutexGuard` never returns into look-inside code.
#[majit_macros::dont_look_inside]
fn simplequeue_wait_and_pop(
    queue: &W_SimpleQueue,
    timeout: Option<f64>,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let mut guard = simplequeue_wait_for_item(queue, timeout)?;
    guard.pop_front().ok_or_else(empty_error)
}

mod simplequeue_methods {
    use super::*;

    #[pyre_interpreter::pyre_methods(weakrefable, unhashable)]
    impl W_SimpleQueue {
        #[staticmethod]
        fn __new__(
            cls: PyObjectRef,
            args: &[PyObjectRef],
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            if args.len() > 1 {
                return Err(pyre_interpreter::PyError::type_error(
                    "_queue.SimpleQueue() takes no arguments",
                ));
            }
            pyre_interpreter::typedef::check_user_subclass(type_object(), cls)?;
            let obj = Self::allocate_stable(Self::default());
            unsafe { (*obj).w_class = cls };
            // Old-gen shell may hold a young subclass; incminimark.py write_barrier records the old->young edge.
            pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF);
            Ok(obj)
        }

        /// `block` and `timeout` are accepted and ignored: the queue is
        /// unbounded, so a put never blocks.  They are named without a leading
        /// underscore because the keyword a caller may bind is taken from the
        /// parameter's own identifier, and `put(item, block=True,
        /// timeout=None)` is the signature.
        fn put(
            &self,
            item: PyObjectRef,
            #[default(true)] block: bool,
            #[default(w_none())] timeout: PyObjectRef,
        ) -> PyObjectRef {
            let _ = (block, timeout);
            simplequeue_put(self, item)
        }

        fn put_nowait(&self, item: PyObjectRef) -> PyObjectRef {
            simplequeue_put(self, item)
        }

        fn get(
            &self,
            #[default(true)] block: bool,
            #[default(w_none())] timeout: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            simplequeue_get(self, block, timeout)
        }

        fn get_nowait(&self) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            simplequeue_get(self, false, w_none())
        }

        fn empty(&self) -> bool {
            queue_is_empty(&self.queue)
        }

        fn qsize(&self) -> i64 {
            queue_len(&self.queue) as i64
        }

        #[classmethod]
        fn __class_getitem__(
            cls: PyObjectRef,
            item: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            pyre_interpreter::_pypy_generic_alias::generic_alias_class_getitem(&[cls, item])
        }
    }
}

/// Drop the Rust-owned queue storage.
///
/// # Safety
/// `obj` must be a GC-dead `W_SimpleQueue`.
pub unsafe fn w_simplequeue_dealloc(obj: PyObjectRef) {
    unsafe { std::ptr::drop_in_place(obj as *mut W_SimpleQueue) };
}

/// Walk the queued items owned by a `W_SimpleQueue`.
///
/// # Safety
/// `obj_addr` must point at a live `W_SimpleQueue`.
pub unsafe fn w_simplequeue_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let queue = unsafe { &mut *(obj_addr as *mut W_SimpleQueue) };
    // `get_mut` is sound because the deque is only ever mutated while holding
    // the GIL, so collection cannot overlap a mutation. A thread parked inside
    // `queue_lock` or `Condvar::wait` may hold the mutex, but it is not
    // touching the deque; it rejoins the RUNNING census before touching it
    // again.
    let items = queue.queue.get_mut();
    let (front, back) = items.as_mut_slices();
    for item in front.iter_mut().chain(back.iter_mut()) {
        f(item as *mut PyObjectRef as *mut majit_ir::GcRef);
    }
}

pyre_interpreter::py_module! {
    "_queue",
    interpleveldefs: {
        "SimpleQueue" => simplequeue_methods::type_object(),
    },
    exceptions: {
        "Empty" => pyre_interpreter::builtins::lookup_exc_class("Exception")
            .expect("Exception must be installed before _queue init"),
    },
}

/// `_queue.SimpleQueue` is subclassable and owns a FIFO of Python objects.
unsafe fn simplequeue_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    unsafe {
        pyre_interpreter::objspace::std::mapdict::mapdict_storage_custom_trace(obj_addr, f);
        w_simplequeue_custom_trace(obj_addr, f);
    }
}

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // `_queue.SimpleQueue` is unconditional, so it registers ahead of the
    // target-gated tail and keeps one id on every target.
    types.push(ModuleGcType {
        descriptor: <W_SimpleQueue as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::CustomTrace(simplequeue_custom_trace),
        destructor: Some(gc_destructor!(w_simplequeue_dealloc)),
    });
}
