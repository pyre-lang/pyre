//! _io module — PyPy: pypy/module/_io/
//!
//! Pyre stubs the bulk of the C IO classes: ctors return None / "" and
//! ABC base classes (`_IOBase` / `_RawIOBase` / `_BufferedIOBase` /
//! `_TextIOBase`) are exposed as plain types so io.py's class
//! inheritance succeeds.

use pyre_object::*;

mod interp_bufferedio;
pub use interp_bufferedio::{
    BUFFEREDREADER_USER_TYPE, W_BUFFEREDREADER_USER_GC_TYPE_ID_CELL,
    W_BUFFEREDREADER_USER_PYRE_CLASS_DESCRIPTOR, W_BufferedReader, W_BufferedReaderUser,
};
mod buffered_writer;
pub use buffered_writer::{
    BUFFEREDWRITER_USER_TYPE, W_BUFFEREDWRITER_USER_GC_TYPE_ID_CELL,
    W_BUFFEREDWRITER_USER_PYRE_CLASS_DESCRIPTOR, W_BufferedWriter, W_BufferedWriterUser,
};
mod buffered_rwpair;
pub use buffered_rwpair::{
    BUFFEREDRWPAIR_USER_TYPE, W_BUFFEREDRWPAIR_USER_GC_TYPE_ID_CELL,
    W_BUFFEREDRWPAIR_USER_PYRE_CLASS_DESCRIPTOR, W_BufferedRWPair, W_BufferedRWPairUser,
};
mod buffered_random;
pub use buffered_random::{
    BUFFEREDRANDOM_USER_TYPE, W_BUFFEREDRANDOM_USER_GC_TYPE_ID_CELL,
    W_BUFFEREDRANDOM_USER_PYRE_CLASS_DESCRIPTOR, W_BufferedRandom, W_BufferedRandomUser,
};
mod interp_fileio;
pub use interp_fileio::{
    FILEIO_USER_TYPE, W_FILEIO_USER_GC_TYPE_ID_CELL, W_FILEIO_USER_PYRE_CLASS_DESCRIPTOR, W_FileIO,
    W_FileIOUser,
};
mod interp_bytesio;
pub use interp_bytesio::{
    BYTESIO_USER_TYPE, W_BYTESIO_USER_GC_TYPE_ID_CELL, W_BYTESIO_USER_PYRE_CLASS_DESCRIPTOR,
    W_BytesIO, W_BytesIOUser,
};
mod interp_stringio;
pub use interp_stringio::{
    STRINGIO_USER_TYPE, W_STRINGIO_USER_GC_TYPE_ID_CELL, W_STRINGIO_USER_PYRE_CLASS_DESCRIPTOR,
    W_StringIO, W_StringIOUser,
};
mod interp_textio;
pub use interp_textio::{
    TEXTIOWRAPPER_USER_TYPE, W_TEXTIOWRAPPER_USER_GC_TYPE_ID_CELL,
    W_TEXTIOWRAPPER_USER_PYRE_CLASS_DESCRIPTOR, W_TextIOWrapper, W_TextIOWrapperUser,
};
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
pub(crate) mod interp_win32consoleio;
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
pub use interp_win32consoleio::{
    W_WINCONSOLEIO_USER_GC_TYPE_ID_CELL, W_WINCONSOLEIO_USER_PYRE_CLASS_DESCRIPTOR, W_WinConsoleIO,
    W_WinConsoleIOUser, WINCONSOLEIO_USER_TYPE,
};

pub fn text_io_wrapper_type() -> PyObjectRef {
    interp_textio::type_object()
}

/// Whether `obj` has the `W_BytesIO` layout, including a Python subclass.
pub(crate) fn is_bytesio(obj: PyObjectRef) -> bool {
    W_BytesIO::from_obj(obj).is_some()
}

/// Whether `obj` has the `W_StringIO` layout, including a Python subclass.
pub(crate) fn is_stringio(obj: PyObjectRef) -> bool {
    W_StringIO::from_obj(obj).is_some()
}

/// `interp_iobase.py W_IOBase.w_dict` when `obj` is one of the typed IO
/// payloads. `from_obj` accepts both the base and the `_getusercls` typeptr.
/// `None` is every other layout. `Some(PY_NULL)` is a payload whose
/// dictionary has not been created.
pub(crate) fn iobase_read_dict(obj: PyObjectRef) -> Option<PyObjectRef> {
    if obj.is_null() {
        return None;
    }
    if let Some(inst) = W_FileIO::from_obj(obj) {
        return Some(inst.w_dict);
    }
    if let Some(inst) = W_BytesIO::from_obj(obj) {
        return Some(inst.w_dict);
    }
    if let Some(inst) = W_StringIO::from_obj(obj) {
        return Some(inst.w_dict);
    }
    if let Some(inst) = W_TextIOWrapper::from_obj(obj) {
        return Some(inst.w_dict);
    }
    if let Some(inst) = W_BufferedReader::from_obj(obj) {
        return Some(inst.w_dict);
    }
    if let Some(inst) = W_BufferedWriter::from_obj(obj) {
        return Some(inst.w_dict);
    }
    if let Some(inst) = W_BufferedRandom::from_obj(obj) {
        return Some(inst.w_dict);
    }
    if let Some(inst) = W_BufferedRWPair::from_obj(obj) {
        return Some(inst.w_dict);
    }
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    if let Some(inst) = W_WinConsoleIO::from_obj(obj) {
        return Some(inst.w_dict);
    }
    None
}

/// `self.w_dict = w_dict` (`interp_iobase.py W_IOBase.getdict`).
pub(crate) fn iobase_write_dict(obj: PyObjectRef, w_dict: PyObjectRef) {
    if let Some(inst) = W_FileIO::from_obj(obj) {
        inst.w_dict = w_dict;
        return;
    }
    if let Some(inst) = W_BytesIO::from_obj(obj) {
        inst.w_dict = w_dict;
        return;
    }
    if let Some(inst) = W_StringIO::from_obj(obj) {
        inst.w_dict = w_dict;
        return;
    }
    if let Some(inst) = W_TextIOWrapper::from_obj(obj) {
        inst.w_dict = w_dict;
        return;
    }
    if let Some(inst) = W_BufferedReader::from_obj(obj) {
        inst.w_dict = w_dict;
        return;
    }
    if let Some(inst) = W_BufferedWriter::from_obj(obj) {
        inst.w_dict = w_dict;
        return;
    }
    if let Some(inst) = W_BufferedRandom::from_obj(obj) {
        inst.w_dict = w_dict;
        return;
    }
    if let Some(inst) = W_BufferedRWPair::from_obj(obj) {
        inst.w_dict = w_dict;
        return;
    }
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    if let Some(inst) = W_WinConsoleIO::from_obj(obj) {
        inst.w_dict = w_dict;
    }
}

/// `interp_iobase.py W_IOBase.getdict`. Allocates `w_dict` on the first call.
///
/// # Safety
/// `obj` must be a payload [`iobase_read_dict`] recognises.
pub(crate) unsafe fn iobase_getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let existing = iobase_read_dict(obj).expect("typed IO payload");
        if !existing.is_null() {
            return existing;
        }
        // The dict allocation can collect. Pin the receiver and write the
        // field on the forwarded object. The stored dict may be young, so
        // the old payload takes the write barrier.
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let w_dict = w_dict_new();
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        iobase_write_dict(obj, w_dict);
        pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF);
        w_dict
    }
}

/// `interp_iobase.py W_IOBase.getdictvalue`. A null `w_dict` is a miss and
/// does not allocate the dictionary.
///
/// # Safety
/// `obj` must be a payload [`iobase_read_dict`] recognises.
pub(crate) unsafe fn iobase_getdictvalue(
    obj: PyObjectRef,
    attr: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        let w_dict = iobase_read_dict(obj).expect("typed IO payload");
        if w_dict.is_null() {
            return Ok(None);
        }
        crate::baseobjspace::finditem_str(w_dict, attr)
    }
}

// CPython 3.14 raised the public and constructor default from 8 KiB to
// 128 KiB.  Keep one module-owned value shared by every buffered type.
pub(crate) const DEFAULT_BUFFER_SIZE: i64 = 128 * 1024;

/// `interp_bufferedio.py TryLock` — one native lock and its owning
/// thread per buffered stream. The stream stores the opaque handle directly,
/// never in a side table.
struct BufferedTryLock {
    lock: *mut crate::baseobjspace::Lock,
    owner: std::sync::atomic::AtomicI64,
}

#[majit_macros::dont_look_inside]
pub(crate) fn allocate_buffered_lock() -> usize {
    Box::into_raw(Box::new(BufferedTryLock {
        lock: crate::baseobjspace::allocate_lock(),
        owner: std::sync::atomic::AtomicI64::new(0),
    })) as usize
}

/// Acquire a buffered stream's `TryLock`. `false` is the same-thread
/// reentrant arm; contention from another thread blocks until it can acquire.
#[majit_macros::dont_look_inside]
pub(crate) fn acquire_buffered_lock(handle: usize) -> bool {
    let lock = unsafe { &*(handle as *const BufferedTryLock) };
    let current = crate::module::thread::current_ident();
    let native = unsafe { &*lock.lock };
    if !native.acquire(false) {
        if lock.owner.load(std::sync::atomic::Ordering::Acquire) == current {
            return false;
        }
        native.acquire(true);
    }
    lock.owner
        .store(current, std::sync::atomic::Ordering::Release);
    true
}

#[majit_macros::dont_look_inside]
pub(crate) fn release_buffered_lock(handle: usize) {
    let lock = unsafe { &*(handle as *const BufferedTryLock) };
    lock.owner.store(0, std::sync::atomic::Ordering::Release);
    unsafe { &*lock.lock }.release();
}

/// `TryLock.__exit__` as a destructor. `with self.lock:` always runs
/// `__exit__` when the body leaves, including an unwind; a sequential
/// `release_buffered_lock` after `body()` does not.
pub(crate) struct BufferedLockGuard(pub(crate) usize);

impl Drop for BufferedLockGuard {
    fn drop(&mut self) {
        release_buffered_lock(self.0);
    }
}

// The module-local exception class is process-global, like PyPy's module
// definition object.  Keep the immortal type pointer shared across threads;
// runtime semantic state must not be duplicated in TLS.
static UNSUPPORTED_OPERATION_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

fn type_method(mut ns: PyObjectRef, name: &str, mut function: PyObjectRef) {
    // Both words already exist. Sequential `pin_root(ns)` would be a
    // safepoint while `function` is still unpublished (`RootScope::pin_roots`).
    crate::__pyre_store!(ns, name, function);
}

fn io_closed(obj: PyObjectRef) -> bool {
    crate::baseobjspace::getattr_str(obj, "closed")
        .ok()
        .and_then(|value| crate::baseobjspace::is_true(value).ok())
        .unwrap_or(false)
}

fn iobase_internal_closed(obj: PyObjectRef) -> bool {
    crate::baseobjspace::getattr_str(obj, "__iobase_closed__")
        .ok()
        .is_some_and(|value| unsafe {
            pyre_object::is_bool(value) && pyre_object::w_bool_get_value(value)
        })
}

pub(crate) fn iobase_set_internal_closed(
    obj: PyObjectRef,
    closed: bool,
) -> Result<(), crate::PyError> {
    if crate::baseobjspace::setdictvalue(obj, "__iobase_closed__", w_bool_from(closed))? {
        Ok(())
    } else {
        Err(crate::PyError::runtime_error(
            "_IOBase instance has no state dictionary",
        ))
    }
}

/// Reads the `io.UnsupportedOperation` type installed into
/// `UNSUPPORTED_OPERATION_TYPE` at module init; null before then.  The value is
/// not a build-time constant, so the JIT residualises the read instead of
/// tracing into it (`@dont_look_inside`).
#[majit_macros::dont_look_inside]
pub(crate) fn unsupported_operation_type() -> PyObjectRef {
    UNSUPPORTED_OPERATION_TYPE
        .get()
        .unwrap_or(std::ptr::null_mut())
}

/// `space.getexecutioncontext().checksignals()` as the buffered writer's
/// partial-write loops call it (`interp_bufferedio.py:429`, `:914`).
///
/// wasm32 carries no `signal` module (`module/mod.rs`), and with no handler to
/// run the retry has nothing to check for.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn checksignals() -> Result<(), crate::PyError> {
    crate::module::signal::interp_signal::checksignals_now()
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn checksignals() -> Result<(), crate::PyError> {
    Ok(())
}

/// `interp_iobase.py:unsupported` — construct the module-local
/// `UnsupportedOperation`, preserving its OSError + ValueError MRO.
pub(crate) fn unsupported(message: &str) -> crate::PyError {
    let w_type = unsupported_operation_type();
    if w_type.is_null() {
        return crate::PyError::value_error(message);
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_type);
    let msg_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(message));
    match crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(type_slot),
        &[pyre_object::gc_roots::shadow_stack_get(msg_slot)],
    ) {
        Ok(exc) => {
            let exc_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(exc);
            unsafe {
                crate::PyError::from_exc_object(pyre_object::gc_roots::shadow_stack_get(exc_slot))
            }
        }
        Err(error) => error,
    }
}

/// A zero-filled `bytearray` of `size` bytes, for a `size` that came from
/// Python.
///
/// `w_bytearray_new` reaches Rust's infallible allocator, which aborts the
/// process instead of unwinding, so `f.read1(1 << 44)` or
/// `BufferedReader(raw, 1 << 44)` terminates the interpreter where
/// `PyBytes_FromStringAndSize(NULL, n)` and `_buffered_init`'s `PyMem_Malloc`
/// answer `MemoryError`.
pub(crate) fn try_new_buffer(size: usize) -> Result<PyObjectRef, crate::PyError> {
    pyre_object::bytearrayobject::w_bytearray_try_new(size)
        .ok_or_else(crate::builtins::reservation_failed)
}

pub(crate) fn iobase_close(args: &[PyObjectRef]) -> crate::PyResult {
    let mut self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("close() requires self"))?;
    if pyre_object::with_roots!(self_obj => iobase_internal_closed(self_obj)) {
        return Ok(w_none());
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_obj);
    // `close_w` closes the C `FILE` a cffi cast opened over this stream
    // before the flush below, the same order it takes upstream.
    if let Some(hooks) = crate::importing::optional_module_hooks() {
        (hooks.close_cffi_fileobj)(pyre_object::gc_roots::shadow_stack_get(self_slot));
    }
    // PyPy `close_w`: `flush()` runs while `closed` is still false, and the
    // internal flag is set in `finally`, even when the virtual flush raises.
    let flushed = match call_method_result(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        "flush",
        &[],
    ) {
        Ok(value) => {
            let value = pyre_object::gc_roots::pin_root(value);
            iobase_set_internal_closed(pyre_object::gc_roots::shadow_stack_get(self_slot), true)?;
            Ok(value)
        }
        Err(error) => {
            let mut error = error;
            let error_slot = error.pin(&_roots);
            iobase_set_internal_closed(pyre_object::gc_roots::shadow_stack_get(self_slot), true)?;
            error.reload(&_roots, error_slot);
            Err(error)
        }
    };
    flushed?;
    // `close_w` reaches its `maybe_unregister_rpython_finalizer_io` only past
    // the `try`, so a flush that raised does not get the hint even though the
    // `finally` above has already marked the object closed.
    maybe_unregister_rpython_finalizer_io(pyre_object::gc_roots::shadow_stack_get(self_slot));
    Ok(w_none())
}

/// `interp_iobase.py W_IOBase.maybe_unregister_rpython_finalizer_io`: once
/// `closed` is set, `iobase_del`'s own work is a no-op, so the object may be
/// collected without being delivered.
///
/// Withheld from an app-level subclass — upstream's `user_overridden_class`
/// guard.  Its `close` may do anything, and `iobase_del` reaches it through
/// the ordinary lookup.  `is_heaptype` is that predicate here: every
/// interp-level `_io` type is a builtin one, so a heap type in this position
/// is a class written in Python.
pub(crate) fn maybe_unregister_rpython_finalizer_io(self_obj: PyObjectRef) {
    let Some(w_type) = crate::typedef::r#type(self_obj) else {
        return;
    };
    if unsafe { pyre_object::w_type_is_heaptype(w_type.as_ptr()) } {
        return;
    }
    crate::executioncontext::may_ignore_finalizer(self_obj);
}

fn iobase_flush(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("flush() requires self"))?;
    if iobase_internal_closed(self_obj) {
        return Err(crate::PyError::value_error("I/O operation on closed file."));
    }
    Ok(w_none())
}

fn iobase_closed_get(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .get(1)
        .copied()
        .ok_or_else(|| crate::PyError::type_error("descriptor requires an instance"))?;
    Ok(w_bool_from(iobase_internal_closed(self_obj)))
}

/// `iobase.c iobase_check_closed`, reached as `_IOBase._checkClosed`.
///
/// The sentence ends in a period at this layer and in `textio.c`, and does not
/// in `fileio.c` or `stringio.c`; `flush` and `isatty` below carry the same
/// spelling this one does.
fn iobase_check_closed(args: &[PyObjectRef]) -> crate::PyResult {
    // CPython 3.14 `IOBase._checkClosed` no longer exposes PyPy's optional
    // `message` argument: Argument Clinic rejects every argument after self.
    if args.len() != 1 {
        return Err(crate::PyError::type_error(
            "_IOBase._checkClosed() takes no arguments",
        ));
    }
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("_checkClosed() requires self"))?;
    if io_closed(self_obj) {
        Err(crate::PyError::value_error("I/O operation on closed file."))
    } else {
        Ok(w_none())
    }
}

fn iobase_check_capability(args: &[PyObjectRef], method: &str, message: &str) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error(format!("{method}() requires self")))?;
    let capable = call_method_result(self_obj, method, &[])?;
    if crate::baseobjspace::is_true(capable)? {
        Ok(w_none())
    } else {
        Err(unsupported(message))
    }
}

fn iobase_unsupported(args: &[PyObjectRef], operation: &str) -> crate::PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "{operation}() requires self"
        )));
    }
    Err(unsupported(operation))
}

fn iobase_seek(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "seek")
}

fn iobase_truncate(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "truncate")
}

fn iobase_fileno(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "fileno")
}

fn iobase_tell(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("tell() requires self"))?;
    let _seek_roots = pyre_object::gc_roots::push_roots();
    let pos_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_int_new(0));
    let whence_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_int_new(1));
    call_method_result(
        self_obj,
        "seek",
        &[
            pyre_object::gc_roots::shadow_stack_get(pos_slot),
            pyre_object::gc_roots::shadow_stack_get(whence_slot),
        ],
    )
}

fn iobase_enter(args: &[PyObjectRef]) -> crate::PyResult {
    let mut self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("__enter__() requires self"))?;
    pyre_object::with_roots!(self_obj => iobase_check_closed(&[self_obj]))?;
    Ok(self_obj)
}

fn iobase_iter(args: &[PyObjectRef]) -> crate::PyResult {
    let mut self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("__iter__() requires self"))?;
    pyre_object::with_roots!(self_obj => iobase_check_closed(&[self_obj]))?;
    Ok(self_obj)
}

fn iobase_next(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("__next__() requires self"))?;
    let mut line = call_method_result(self_obj, "readline", &[])?;
    if pyre_object::with_roots!(line => crate::baseobjspace::len_w(line))? == 0 {
        Err(crate::PyError::stop_iteration())
    } else {
        Ok(line)
    }
}

// ── AutoFlusher ──────────────────────────────────────────────────────
//
// `interp_iobase.py` keeps one `AutoFlusher` per space, weakly holding
// every stream ever constructed, and `moduledef.py:37-40 Module.shutdown`
// flushes whatever is still alive.  Without it an unclosed buffered stream
// loses its writes whenever it outlives the only teardown pyre performs — the
// `__main__` globals — which is every stream reachable from another module,
// from `sys.modules`, or from a container.
//
// `rweaklist.py store_handle` holds each stream through `weakref.ref`; the
// `GcWeakrefBox` is pyre's rweakref, so the handle list is a `Vec` of boxes.
// A box is collector-managed and this list is its only referent, so
// [`walk_autoflusher_roots`] walks the slots as roots. The table is shared
// process state because `space.fromcache(AutoFlusher)` owns one instance for
// the object space, independent of the thread that constructs a stream.

/// `rweaklist.py INITIAL_SIZE`.
const AUTOFLUSHER_INITIAL_SIZE: usize = 4;

#[derive(Default)]
struct AutoFlusher {
    /// `rweaklist.py:18 self.handles` — index → `GcWeakrefBox`. A null slot is
    /// the `dead_ref` placeholder RPython pre-fills the list with.
    handles: Vec<usize>,
    /// `rweaklist.py:19 self.free_list`.
    free_list: Vec<usize>,
    /// Marks each `rweaklist.py initialize` of the handle table so an
    /// allocation can detect that shutdown discarded its reserved slot.
    generation: u64,
}

impl AutoFlusher {
    /// `rweaklist.py initialize`.
    fn initialize(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.handles = vec![0; AUTOFLUSHER_INITIAL_SIZE];
        self.free_list = (0..AUTOFLUSHER_INITIAL_SIZE).collect();
    }

    /// `rweaklist.py reserve_next_handle_index`.
    fn reserve_next_handle_index(&mut self) -> usize {
        if self.handles.is_empty() {
            self.initialize();
        }
        if let Some(index) = self.free_list.pop() {
            return index;
        }
        for (index, &handle) in self.handles.iter().enumerate() {
            if unsafe { pyre_object::weakref::w_gc_weakref_box_deref(handle as PyObjectRef) }
                .is_null()
            {
                self.free_list.push(index);
            }
        }
        if self.free_list.len() * 3 < self.handles.len() * 2 {
            let length = self.handles.len();
            self.free_list.extend(length..length * 2);
            self.handles.resize(length * 2, 0);
        }
        self.free_list
            .pop()
            .expect("the doubling above always frees an index")
    }
}

/// `interp_iobase.py get_autoflusher` — `space.fromcache(AutoFlusher)`.
static AUTOFLUSHER: std::sync::LazyLock<parking_lot::Mutex<AutoFlusher>> =
    std::sync::LazyLock::new(|| parking_lot::Mutex::new(AutoFlusher::default()));

/// Visit each stored `GcWeakrefBox` as a strong root.
///
/// The box is an ordinary collector-managed object and this list is the only
/// place that holds it, so without this walk a collection would sweep it — or
/// relocate it and leave the slot pointing at the old address — and
/// [`flush_all_streams`] would dereference a dangling handle.  Keeping the box
/// alive does not make the handle strong with respect to the stream: the box's
/// inner rweakref is still cleared by the collector's
/// `invalidate_young_weakrefs` / `invalidate_old_weakrefs` when the stream
/// itself dies, which is what turns the slot into `rweaklist.py`'s `dead_ref`.
///
pub fn walk_autoflusher_roots(mut visitor: impl FnMut(&mut PyObjectRef)) {
    let mut flusher = AUTOFLUSHER.lock();
    for slot in flusher.handles.iter_mut() {
        if *slot == 0 {
            continue;
        }
        let mut handle = *slot as PyObjectRef;
        visitor(&mut handle);
        *slot = handle as usize;
    }
}

/// `interp_iobase.py AutoFlusher.add`, reached from
/// `interp_iobase.py W_IOBase.__init__` for every stream that does not
/// opt out.
///
/// Returns `w_iobase`, which the rweakref allocation may have relocated.
///
/// Reads and rewrites the runtime-mutable process-global `AUTOFLUSHER` handle
/// table, not a build-time constant, so the JIT residualises the call instead
/// of tracing into it (`@dont_look_inside`, the `importing::sys_modules_dict`
/// shape).
#[majit_macros::dont_look_inside]
pub(crate) fn autoflusher_add(w_iobase: PyObjectRef) -> PyObjectRef {
    if w_iobase.is_null() {
        return w_iobase;
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let target_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_iobase);
    loop {
        let (index, generation, handle) = {
            let mut flusher = AUTOFLUSHER.lock();
            let index = flusher.reserve_next_handle_index();
            (index, flusher.generation, flusher.handles[index])
        };
        // `rweaklist.py store_handle` — reuse the slot's box when it already
        // holds one, so a long-lived process bounds the immortal boxes by its peak
        // number of open streams.
        let target = pyre_object::gc_roots::shadow_stack_get(target_root);
        let stored = unsafe {
            pyre_object::weakref::w_gc_weakref_box_retarget(handle as PyObjectRef, target)
        };
        let boxed = if stored {
            std::ptr::null_mut()
        } else {
            pyre_object::weakref::w_gc_weakref_box_new(pyre_object::gc_roots::shadow_stack_get(
                target_root,
            ))
        };
        let mut flusher = AUTOFLUSHER.lock();
        // `rweaklist.py initialize` replaces the table, invalidating an
        // index reserved from the previous generation.
        if flusher.generation != generation {
            continue;
        }
        if !stored {
            flusher.handles[index] = boxed as usize;
        }
        break;
    }
    pyre_object::gc_roots::shadow_stack_get(target_root)
}

/// `interp_iobase.py AutoFlusher.flush_all`, run from
/// `moduledef.py:37-40 Module.shutdown` — "at shutdown, flush all open streams.
/// Ignore I/O errors."
pub fn flush_all_streams() {
    loop {
        let handles = {
            let mut flusher = AUTOFLUSHER.lock();
            flusher.free_list.clear();
            // `rweaklist.py initialize` resets the state here, so a
            // stream created while flushing is picked up by the next round
            // instead of being flushed twice.
            flusher.generation = flusher.generation.wrapping_add(1);
            std::mem::take(&mut flusher.handles)
        };
        // `initialize` has rebound the list the walker reads, so the detached
        // boxes have no root left — in RPython the local `handles` list is
        // itself traced.  A `flush` runs Python code, so a collection between
        // two iterations would sweep the boxes this round has not reached yet.
        // Pin them for the round and read each one back, the way
        // `autoflusher_add` reads its pinned stream back.  The `dead_ref`
        // placeholder slots carry nothing to keep alive.
        let handles: Vec<PyObjectRef> = handles
            .into_iter()
            .filter(|slot| *slot != 0)
            .map(|slot| slot as PyObjectRef)
            .collect();
        let _handle_roots = pyre_object::gc_roots::push_roots();
        let handles_root = pyre_object::gc_roots::shadow_stack_len();
        for &handle in &handles {
            let _ = pyre_object::gc_roots::pin_root(handle);
        }
        let mut progress = false;
        for index in 0..handles.len() {
            let _roots = pyre_object::gc_roots::push_roots();
            let stream_root = pyre_object::gc_roots::shadow_stack_len();
            let handle = pyre_object::gc_roots::shadow_stack_get(handles_root + index);
            let stream = unsafe { pyre_object::weakref::w_gc_weakref_box_deref(handle) };
            if stream.is_null() {
                continue;
            }
            let _ = pyre_object::gc_roots::pin_root(stream);
            progress = true;
            // "Silencing all errors is bad, but getting randomly interrupted
            // here is equally as bad, and potentially more frequent (because of
            // shutdown issues)."
            let _ = call_method_result(
                pyre_object::gc_roots::shadow_stack_get(stream_root),
                "flush",
                &[],
            );
        }
        if !progress {
            break;
        }
    }
}

/// Tag a freshly allocated `_io` object with the class being constructed, then
/// put it on the finalizer queue — `objspace.py:485-487 allocate_instance`
/// followed by `interp_iobase.py W_IOBase.__init__`:
/// `if self.needs_finalizer(): self.register_finalizer(space)`.
///
/// An unclosed stream still has to flush and close once it is unreachable, and
/// `iobase_del` does that. It is a builtin method on the interp-level type, so
/// `hasuserdel` — set only for a class whose own dict carries `__del__` — is
/// false for the plain types and the allocation hook would skip them. A
/// subclass that does define `__del__` is registered by the hook and reaches
/// this call too — the case `baseobjspace.py` returns early on; here
/// the queue drops the repeat.
pub(crate) fn tag_io_instance(obj: PyObjectRef, cls: PyObjectRef) -> PyObjectRef {
    tag_io_instance_with_finalizer(obj, cls, true)
}

/// [`tag_io_instance`] for a type that overrides
/// `interp_iobase.py W_IOBase.needs_finalizer` — "can return False if we
/// know that the precise close() method of this class will have no effect".
/// Every override reads `type(self) is not <that class>`, so a subclass, whose
/// `close` may do anything, keeps the default answer.
pub(crate) fn tag_io_instance_with_finalizer(
    obj: PyObjectRef,
    cls: PyObjectRef,
    needs_finalizer: bool,
) -> PyObjectRef {
    tag_io_instance_impl(obj, cls, needs_finalizer, true)
}

/// `W_IOBase.__init__(add_to_autoflusher=False)` with the same subclass
/// finalizer rule as [`tag_io_instance_with_finalizer`].
pub(crate) fn tag_io_instance_without_autoflusher(
    obj: PyObjectRef,
    cls: PyObjectRef,
    needs_finalizer: bool,
) -> PyObjectRef {
    tag_io_instance_impl(obj, cls, needs_finalizer, false)
}

fn tag_io_instance_impl(
    obj: PyObjectRef,
    cls: PyObjectRef,
    needs_finalizer: bool,
    add_to_autoflusher: bool,
) -> PyObjectRef {
    if !cls.is_null() {
        crate::typedef::tag_subclass_instance(obj, cls);
    }
    let obj = if add_to_autoflusher {
        autoflusher_add(obj)
    } else {
        obj
    };
    if needs_finalizer {
        crate::executioncontext::register_finalizer(obj);
    }
    obj
}

fn iobase_del(args: &[PyObjectRef]) -> crate::PyResult {
    let Some(mut self_obj) = args.first().copied() else {
        return Ok(w_none());
    };
    // pypy/module/_io/interp_iobase.py `descr_del` and CPython 3.14
    // Modules/_io/iobase.c:275-310 `iobase_finalize`: failure to obtain or
    // truth-test `closed` means the partially initialized/detached object is
    // unusable and finalization stops quietly.  This check must not collapse
    // the error to `closed == false` and call close(), or tracing-GC latency
    // leaks a stale ValueError into a later `catch_unraisable_exception`.
    let closed = match pyre_object::with_roots!(self_obj => crate::baseobjspace::getattr_str(self_obj, "closed"))
    {
        Ok(value) => {
            match pyre_object::with_roots!(self_obj => crate::baseobjspace::is_true(value)) {
                Ok(closed) => closed,
                Err(_) => return Ok(w_none()),
            }
        }
        Err(_) => return Ok(w_none()),
    };
    if !closed {
        // CPython sets this best-effort marker before the dynamic close call;
        // FileIO uses it to distinguish implicit-close warnings.  PyPy's
        // `_dealloc_warn_w` followed by `space.call_method(self, "close")`
        // supplies the same call order.
        let _ = pyre_object::with_roots!(self_obj => crate::baseobjspace::setattr_str(
            self_obj,
            "_finalizing",
            w_bool_from(true)
        ));
        let _ = pyre_object::with_roots!(self_obj => call_method_result(self_obj, "_dealloc_warn", &[self_obj]));

        // CPython 3.14 reports a real close failure via unraisablehook
        // (test_io.py:test_error_through_destructor).  UserDelAction owns that
        // reporting boundary in pyre, so preserve the close error here.
        call_method_result(self_obj, "close", &[])?;
    }
    Ok(w_none())
}

fn iobase_getstate(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("__getstate__() requires self"))?;
    Err(crate::PyError::type_error(format!(
        "cannot pickle '{}' object",
        crate::type_methods::arg_type_name(self_obj)
    )))
}

fn iobase_isatty(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("isatty() requires self"))?;
    if io_closed(self_obj) {
        return Err(crate::PyError::value_error("I/O operation on closed file."));
    }
    Ok(w_bool_from(false))
}

/// `interp_iobase.py W_IOBase.writelines_w` — validate the stream,
/// obtain the input iterator, and call the receiver's (possibly overridden)
/// `write` method once for each line.  The iteration is deliberately lazy;
/// no list snapshot is introduced.
pub(crate) fn iobase_writelines(args: &[PyObjectRef]) -> crate::PyResult {
    let mut self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("writelines() requires self"))?;
    // The registered arity is only a fast-dispatch hint, so surplus
    // positionals still arrive here.  PyPy's interp2app gateway exposes one
    // argument after the bound receiver and rejects the call before it can
    // iterate or write anything.
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "{}.writelines() takes exactly one argument ({} given)",
            crate::type_methods::arg_type_name(self_obj),
            args.len() - 1,
        )));
    }
    let mut lines = args[1];
    if pyre_object::with_roots!(lines, self_obj => io_closed(self_obj)) {
        return Err(crate::PyError::value_error("I/O operation on closed file."));
    }

    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_obj);
    let _ = pyre_object::gc_roots::pin_root(lines);
    let iterator = crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(sp + 1))?;
    let iterator = pyre_object::gc_roots::pin_root(iterator);
    loop {
        let iterator = pyre_object::gc_roots::shadow_stack_get(sp + 2);
        let line = match crate::baseobjspace::next(iterator) {
            Ok(line) => line,
            Err(err) => {
                let (stop, err) = err.matches_stop_iteration_keep();
                if stop {
                    break;
                }
                return Err(err);
            }
        };
        let _line_root = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(line);
        let line =
            pyre_object::gc_roots::shadow_stack_get(pyre_object::gc_roots::shadow_stack_len() - 1);
        call_method_result(
            pyre_object::gc_roots::shadow_stack_get(sp),
            "write",
            &[line],
        )?;
    }
    Ok(w_none())
}

/// A null value means the argument was not supplied and produces the same
/// `-1` result as `None`.
fn iobase_convert_size(value: PyObjectRef) -> Result<i64, crate::PyError> {
    if value.is_null() || unsafe { pyre_object::is_none(value) } {
        Ok(-1)
    } else {
        crate::baseobjspace::int_w(crate::baseobjspace::space_index(value)?)
    }
}

/// `interp_iobase.py:W_IOBase.readline_w` — backwards-compatible mixin over
/// virtual `peek` and `read`, including the one-byte fallback.
fn iobase_readline(args: &[PyObjectRef]) -> crate::PyResult {
    let mut self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("readline() requires self"))?;
    if args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "readline() takes at most one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let limit = pyre_object::with_roots!(self_obj => iobase_convert_size(args.get(1).copied().unwrap_or(PY_NULL)))?;
    let peek = match pyre_object::with_roots!(self_obj => crate::baseobjspace::getattr_str(self_obj, "peek"))
    {
        Ok(method) => Some(method),
        Err(error) if error.kind == crate::PyErrorKind::AttributeError => None,
        Err(error) => return Err(error),
    };

    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_obj);
    let _ = pyre_object::gc_roots::pin_root(peek.unwrap_or(PY_NULL));
    let one_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_int_new(1));
    let n_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_int_new(1));
    let mut output = Vec::new();
    while limit < 0 || output.len() < limit as usize {
        let mut nreadahead = 1usize;
        let peek = pyre_object::gc_roots::shadow_stack_get(sp + 1);
        if !peek.is_null() {
            pyre_object::gc_roots::shadow_stack_set(one_slot, w_int_new(1));
            let readahead = crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(sp + 1),
                &[pyre_object::gc_roots::shadow_stack_get(one_slot)],
            )?;
            if !unsafe { pyre_object::bytesobject::is_bytes_like(readahead) } {
                return Err(crate::PyError::os_error(format!(
                    "peek() should have returned a bytes object, not '{}'",
                    crate::type_methods::arg_type_name(readahead)
                )));
            }
            let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(readahead) };
            if !bytes.is_empty() {
                let remaining = if limit < 0 {
                    bytes.len()
                } else {
                    (limit as usize - output.len()).min(bytes.len())
                };
                nreadahead = bytes[..remaining]
                    .iter()
                    .position(|byte| *byte == b'\n')
                    .map_or(remaining, |index| index + 1);
            }
        }

        pyre_object::gc_roots::shadow_stack_set(n_slot, w_int_new(nreadahead as i64));
        let read = call_method_result(
            pyre_object::gc_roots::shadow_stack_get(sp),
            "read",
            &[pyre_object::gc_roots::shadow_stack_get(n_slot)],
        )?;
        if !unsafe { pyre_object::bytesobject::is_bytes_like(read) } {
            return Err(crate::PyError::os_error(format!(
                "peek() should have returned a bytes object, not '{}'",
                crate::type_methods::arg_type_name(read)
            )));
        }
        let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(read) };
        if bytes.is_empty() {
            break;
        }
        output.extend_from_slice(bytes);
        if bytes.last() == Some(&b'\n') {
            break;
        }
    }
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&output))
}

/// `interp_iobase.py:W_IOBase.readlines_w` — consume the stream iterator,
/// stopping after the accumulated line lengths exceed a positive hint.
pub(super) fn iobase_readlines(args: &[PyObjectRef]) -> crate::PyResult {
    let mut self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("readlines() requires self"))?;
    if args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "readlines() takes at most one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let hint = pyre_object::with_roots!(self_obj => iobase_convert_size(args.get(1).copied().unwrap_or(PY_NULL)))?;
    let iterator = crate::baseobjspace::iter(self_obj)?;
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(iterator);
    let lines_sp = pyre_object::gc_roots::shadow_stack_len();
    let mut count = 0usize;
    let mut length = 0i64;
    loop {
        let line = match crate::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(sp)) {
            Ok(line) => line,
            Err(error) => {
                let (stop, error) = error.matches_stop_iteration_keep();
                if stop {
                    break;
                }
                return Err(error);
            }
        };
        // `len_w` can run `__len__`; the line is pinned before it.
        let line_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(line);
        length = length.saturating_add(crate::baseobjspace::len_w(
            pyre_object::gc_roots::shadow_stack_get(line_slot),
        )?);
        count += 1;
        if hint > 0 && length > hint {
            break;
        }
    }
    let lines = (0..count)
        .map(|index| pyre_object::gc_roots::shadow_stack_get(lines_sp + index))
        .collect();
    Ok(w_list_new(lines))
}

fn init_iobase_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // interp_iobase.py W_IOBase.typedef declares both descriptors
    // in the raw typedef.  They must be present before the type/layout is
    // built: setting only the hasdict/weakrefable flags afterwards leaves
    // `_IOBase()` without the observable `__dict__` descriptor.
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "__dict__",
        crate::typedef::dict_descr(),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "__weakref__",
        crate::typedef::weakref_descr(),
    );
    let closed_getter = crate::make_builtin_function_with_arity("closed", iobase_closed_get, 2);
    let closed_getter_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(closed_getter);
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "closed",
        crate::typedef::make_getset_property_named_doc(
            pyre_object::gc_roots::shadow_stack_get(closed_getter_slot),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            "True if the file is closed",
            "closed",
        ),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "close",
        crate::make_builtin_function_with_arity("close", iobase_close, 1),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "flush",
        crate::make_builtin_function_with_arity("flush", iobase_flush, 1),
    );
    for (name, function) in [
        ("seek", iobase_seek as crate::gateway::BuiltinCodeFn),
        ("truncate", iobase_truncate as crate::gateway::BuiltinCodeFn),
        ("fileno", iobase_fileno as crate::gateway::BuiltinCodeFn),
    ] {
        type_method(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            name,
            crate::make_builtin_function(name, function),
        );
    }
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "tell",
        crate::make_builtin_function_with_arity("tell", iobase_tell, 1),
    );
    for name in ["readable", "writable", "seekable"] {
        type_method(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            name,
            crate::make_builtin_function_with_arity(name, |_| Ok(w_bool_from(false)), 1),
        );
    }
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "_checkReadable",
        crate::make_builtin_function_with_arity(
            "_checkReadable",
            |args| iobase_check_capability(args, "readable", "File or stream is not readable"),
            1,
        ),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "_checkWritable",
        crate::make_builtin_function_with_arity(
            "_checkWritable",
            |args| iobase_check_capability(args, "writable", "File or stream is not writable"),
            1,
        ),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "_checkSeekable",
        crate::make_builtin_function_with_arity(
            "_checkSeekable",
            |args| iobase_check_capability(args, "seekable", "File or stream is not seekable"),
            1,
        ),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "_checkClosed",
        crate::make_builtin_function_with_arity("_checkClosed", iobase_check_closed, 1),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "isatty",
        crate::make_builtin_function_with_arity("isatty", iobase_isatty, 1),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "readline",
        crate::make_builtin_function("readline", iobase_readline),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "readlines",
        crate::make_builtin_function("readlines", iobase_readlines),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "writelines",
        crate::make_builtin_function_with_arity("writelines", iobase_writelines, 2),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "__enter__",
        crate::make_builtin_function_with_arity("__enter__", iobase_enter, 1),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "__exit__",
        crate::make_builtin_function("__exit__", |args| {
            // Dispatch `close` dynamically (`W_IOBase._exit` calls
            // `space.call_method(self, "close")`) so a Python subclass
            // override runs; a static `iobase_close` would mark the object
            // closed without ever running the override.
            call_method_result(args[0], "close", &[])?;
            Ok(w_none())
        }),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "__iter__",
        crate::make_builtin_function_with_arity("__iter__", iobase_iter, 1),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "__next__",
        crate::make_builtin_function_with_arity("__next__", iobase_next, 1),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "__del__",
        crate::make_builtin_function_with_arity("__del__", iobase_del, 1),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "__getstate__",
        crate::make_builtin_function_with_arity("__getstate__", iobase_getstate, 1),
    );
    type_method(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "_dealloc_warn",
        crate::make_builtin_function_with_arity("_dealloc_warn", |_| Ok(w_none()), 2),
    );
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__new__",
            crate::typedef::make_new_descr(iobase_new)
        )
    };
}

/// `interp_iobase.py __new__ = generic_new_descr(W_IOBase)`.
///
/// `typedef.py generic_new_descr` allocates the instance and then runs
/// the interp-level `W_IOBase.__init__`, whose only effects are
/// `interp_iobase.py` — hand the stream to the autoflusher, and put it on
/// the finalizer queue so an unclosed one still flushes and closes once it
/// becomes unreachable. Because
/// it sits in `__new__` rather than in the app-level `__init__`, a subclass
/// that overrides `__init__` without calling up is registered all the same.
///
/// Each exported type installs its own descriptor, while TypeDef identity
/// follows PyPy's TypeCache overrides: notably `_RawIOBase` reuses
/// `W_IOBase.typedef` through `applevel_subclasses_base = W_IOBase`.  Typed
/// payloads provide their own `__new__` and reach the same queue through
/// [`tag_io_instance`].
///
/// The extra arguments go to `__init__`; `generic_new_descr` ignores its
/// `__args__` in the same way.
fn allocate_iobase_instance(
    owner: PyObjectRef,
    owner_name: &str,
    args: &[PyObjectRef],
) -> crate::PyResult {
    let (positional, _) = crate::builtins::split_builtin_kwargs(args);
    let Some(&cls) = positional.first() else {
        return Err(crate::PyError::type_error(format!(
            "{owner_name}.__new__(): not enough arguments"
        )));
    };
    crate::typedef::check_user_subclass(owner, cls)?;
    Ok(tag_io_instance(pyre_object::w_instance_new(cls), cls))
}

fn iobase_new(args: &[PyObjectRef]) -> crate::PyResult {
    allocate_iobase_instance(io_base_type(), "_IOBase", args)
}

fn raw_iobase_new(args: &[PyObjectRef]) -> crate::PyResult {
    allocate_iobase_instance(raw_iobase_type(), "_RawIOBase", args)
}

fn buffered_iobase_new(args: &[PyObjectRef]) -> crate::PyResult {
    allocate_iobase_instance(buffered_iobase_type(), "_BufferedIOBase", args)
}

fn text_iobase_new(args: &[PyObjectRef]) -> crate::PyResult {
    allocate_iobase_instance(text_iobase_type(), "_TextIOBase", args)
}

/// `interp_iobase.py:rawiobase_read_w` — the default raw `read` is a
/// one-shot `readinto` over a freshly allocated bytearray.  A negative or
/// omitted size delegates to the virtual `readall` method.
fn rawiobase_read(args: &[PyObjectRef]) -> crate::PyResult {
    let mut self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("read() requires self"))?;
    if args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "read() takes at most one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let size = match args.get(1).copied() {
        None => -1,
        Some(value) if unsafe { pyre_object::is_none(value) } => -1,
        Some(value) => {
            let index =
                pyre_object::with_roots!(self_obj => crate::baseobjspace::space_index(value))?;
            pyre_object::with_roots!(self_obj => crate::baseobjspace::int_w(index))?
        }
    };
    if size < 0 {
        return call_method_result(self_obj, "readall", &[]);
    }
    let size = usize::try_from(size).map_err(|_| {
        crate::PyError::overflow_error("Python int too large to convert to C ssize_t")
    })?;

    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_obj);
    // `_io__RawIOBase_read_impl` sizes its buffer with
    // `PyByteArray_FromStringAndSize(NULL, n)`, whose failure is a
    // `MemoryError`.
    let _ = pyre_object::gc_roots::pin_root(try_new_buffer(size)?);
    let length = call_method_result(
        pyre_object::gc_roots::shadow_stack_get(sp),
        "readinto",
        &[pyre_object::gc_roots::shadow_stack_get(sp + 1)],
    )?;
    if unsafe { pyre_object::is_none(length) } {
        return Ok(length);
    }
    let length = crate::baseobjspace::int_w(length)?;
    if length < 0 || length as u128 > size as u128 {
        return Err(crate::PyError::value_error(format!(
            "readinto returned {length} outside buffer size {size}"
        )));
    }
    let buffer = pyre_object::gc_roots::shadow_stack_get(sp + 1);
    let data = unsafe { pyre_object::bytearrayobject::w_bytearray_data(buffer) };
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(
        &data[..length as usize],
    ))
}

/// `interp_iobase.py:rawiobase_readall_w` — repeatedly invoke the virtual
/// limited `read(DEFAULT_BUFFER_SIZE)` until EOF.  `None` is propagated only
/// when no bytes have yet been accumulated.
fn rawiobase_readall(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("readall() requires self"))?;
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "readall() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_obj);
    let n_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_int_new(DEFAULT_BUFFER_SIZE));
    let mut output = Vec::new();
    loop {
        pyre_object::gc_roots::shadow_stack_set(n_slot, w_int_new(DEFAULT_BUFFER_SIZE));
        let data = call_method_result(
            pyre_object::gc_roots::shadow_stack_get(sp),
            "read",
            &[pyre_object::gc_roots::shadow_stack_get(n_slot)],
        )?;
        if unsafe { pyre_object::is_none(data) } {
            if output.is_empty() {
                return Ok(data);
            }
            break;
        }
        if !unsafe { pyre_object::bytesobject::is_bytes_like(data) } {
            return Err(crate::PyError::type_error("read() should return bytes"));
        }
        let chunk = unsafe { pyre_object::bytesobject::bytes_like_data(data) };
        if chunk.is_empty() {
            break;
        }
        output.extend_from_slice(chunk);
    }
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&output))
}

fn init_rawiobase_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    crate::__pyre_store!(
        ns,
        "__new__",
        crate::typedef::make_new_descr(raw_iobase_new)
    );
    crate::__pyre_store!(
        ns,
        "read",
        crate::make_builtin_function("read", rawiobase_read)
    );
    crate::__pyre_store!(
        ns,
        "readall",
        crate::make_builtin_function_with_arity("readall", rawiobase_readall, 1)
    );
}

fn buffered_iobase_read(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "read")
}

fn buffered_iobase_read1(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "read1")
}

fn buffered_iobase_write(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "write")
}

fn buffered_iobase_detach(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "detach")
}

/// `interp_bufferedio.py:W_BufferedIOBase._readinto` — acquire one writable
/// view, call the virtual `read`/`read1` once, bounds-check the returned bytes,
/// then copy them into the exact exported window.
fn buffered_iobase_readinto_impl(args: &[PyObjectRef], read_once: bool) -> crate::PyResult {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "{}() takes exactly one argument ({} given)",
            if read_once { "readinto1" } else { "readinto" },
            args.len().saturating_sub(1)
        )));
    }
    let self_obj = args[0];
    let mut buffer = unsafe { crate::builtins::WritableBuffer::acquire(args[1]) }?;
    let target = unsafe { buffer.as_mut_slice() };
    let method = if read_once { "read1" } else { "read" };
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_obj);
    let size_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_int_new(target.len() as i64));
    let data = call_method_result(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        method,
        &[pyre_object::gc_roots::shadow_stack_get(size_slot)],
    )?;
    if !unsafe { pyre_object::bytesobject::is_bytes_like(data) } {
        return Err(crate::PyError::type_error(format!(
            "{method}() should return bytes"
        )));
    }
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(data) };
    if data.len() > target.len() {
        return Err(crate::PyError::value_error(format!(
            "{method}() returned too much data: {} bytes requested, {} returned",
            target.len(),
            data.len()
        )));
    }
    target[..data.len()].copy_from_slice(data);
    Ok(w_int_new(data.len() as i64))
}

fn buffered_iobase_readinto(args: &[PyObjectRef]) -> crate::PyResult {
    buffered_iobase_readinto_impl(args, false)
}

fn buffered_iobase_readinto1(args: &[PyObjectRef]) -> crate::PyResult {
    buffered_iobase_readinto_impl(args, true)
}

fn init_buffered_iobase_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    crate::__pyre_store!(
        ns,
        "__new__",
        crate::typedef::make_new_descr(buffered_iobase_new)
    );
    for (name, function) in [
        (
            "read",
            buffered_iobase_read as crate::gateway::BuiltinCodeFn,
        ),
        (
            "read1",
            buffered_iobase_read1 as crate::gateway::BuiltinCodeFn,
        ),
        (
            "write",
            buffered_iobase_write as crate::gateway::BuiltinCodeFn,
        ),
        (
            "detach",
            buffered_iobase_detach as crate::gateway::BuiltinCodeFn,
        ),
    ] {
        crate::__pyre_store!(ns, name, crate::make_builtin_function(name, function));
    }
    crate::__pyre_store!(
        ns,
        "readinto",
        crate::make_builtin_function_with_arity("readinto", buffered_iobase_readinto, 2)
    );
    crate::__pyre_store!(
        ns,
        "readinto1",
        crate::make_builtin_function_with_arity("readinto1", buffered_iobase_readinto1, 2)
    );
}

fn text_iobase_read(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "read")
}

fn text_iobase_readline(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "readline")
}

fn text_iobase_write(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "write")
}

fn text_iobase_detach(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "detach")
}

fn text_iobase_none_get(args: &[PyObjectRef]) -> crate::PyResult {
    if args.get(1).is_none() {
        return Err(crate::PyError::type_error(
            "descriptor requires an instance",
        ));
    }
    Ok(w_none())
}

fn init_text_iobase_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    crate::__pyre_store!(
        ns,
        "__new__",
        crate::typedef::make_new_descr(text_iobase_new)
    );
    for (name, function) in [
        ("read", text_iobase_read as crate::gateway::BuiltinCodeFn),
        (
            "readline",
            text_iobase_readline as crate::gateway::BuiltinCodeFn,
        ),
        ("write", text_iobase_write as crate::gateway::BuiltinCodeFn),
        (
            "detach",
            text_iobase_detach as crate::gateway::BuiltinCodeFn,
        ),
    ] {
        crate::__pyre_store!(ns, name, crate::make_builtin_function(name, function));
    }
    for name in ["encoding", "newlines", "errors"] {
        let getter = crate::make_builtin_function_with_arity(name, text_iobase_none_get, 2);
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        crate::__pyre_store!(
            ns,
            name,
            crate::typedef::make_getset_descriptor_named(
                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                name
            )
        );
    }
}

/// Process-global abstract IO type objects.  PyPy owns these as the module's
/// `TypeDef` instances; they are shared by every import and are the actual
/// bases used by the typed concrete stream payloads below.
pub fn io_base_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_base(
            "_io._IOBase",
            init_iobase_type,
            crate::typedef::w_object(),
        );
        // CPython 3.14 Modules/_io/_iomodule.c:ADD_TYPE creates the immutable
        // iobase_spec through PyType_FromModuleAndSpec.
        crate::typedef::mark_cpython_heap_type(tp, true);
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(tp, true);
            pyre_object::w_type_set_weakrefable(tp, true);
            pyre_object::typeobject::w_type_set_hasdict(tp, true);
        }
        tp
    })
}

pub(super) fn raw_iobase_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        let base = io_base_type();
        let layout = unsafe { pyre_object::w_type_get_layout_ptr(base) };
        // `W_RawIOBase.typedef.applevel_subclasses_base = W_IOBase`;
        // `TypeCache.build` therefore supplies W_IOBase's TypeDef as the
        // override instead of minting a sibling Layout.
        let tp = crate::typedef::make_builtin_type_with_overridetypedef(
            "_io._RawIOBase",
            init_rawiobase_type,
            base,
            unsafe { (*layout).typedef },
        );
        // `_iomodule.c:ADD_TYPE` creates rawiobase_spec as immutable heap.
        crate::typedef::mark_cpython_heap_type(tp, true);
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(tp, true);
            pyre_object::w_type_set_weakrefable(tp, true);
            pyre_object::typeobject::w_type_set_hasdict(tp, true);
        }
        tp
    })
}

/// PyPy `space.gettypefor(W_FileIO)`: one process-global concrete raw type.
///
/// `open()` needs the same type object exported by `_io`, rather than a
/// second wrapper-shaped allocation path.
pub(crate) fn fileio_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        let tp = interp_fileio::type_object();
        let type_ns = unsafe { pyre_object::w_type_get_dict_ptr(tp) } as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        // Both words already exist. Sequential `pin_root(type_ns)` would be a
        // safepoint while `tp` is still unpublished (`RootScope::pin_roots`).
        let tp_base = pyre_object::gc_roots::pin_roots(&[tp, type_ns]);
        let ns_slot = tp_base + 1;
        crate::builtins::init_file_wrapper_type(pyre_object::gc_roots::shadow_stack_get(ns_slot));
        crate::builtins::init_fileio_type(pyre_object::gc_roots::shadow_stack_get(ns_slot));
        let init_fn = crate::make_builtin_function("__init__", crate::builtins::fileio_init);
        let init_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(init_fn);
        type_method(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            "__init__",
            pyre_object::gc_roots::shadow_stack_get(init_slot),
        );
        pyre_object::gc_roots::shadow_stack_get(tp_base)
    })
}

#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
pub(crate) fn windows_console_io_type() -> PyObjectRef {
    interp_win32consoleio::type_object()
}

pub(crate) fn buffered_reader_type() -> PyObjectRef {
    interp_bufferedio::type_object()
}

pub(crate) fn buffered_writer_type() -> PyObjectRef {
    buffered_writer::type_object()
}

pub(crate) fn buffered_random_type() -> PyObjectRef {
    buffered_random::type_object()
}

pub(super) fn buffered_iobase_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_base(
            "_io._BufferedIOBase",
            init_buffered_iobase_type,
            io_base_type(),
        );
        // `_iomodule.c:ADD_TYPE` creates bufferediobase_spec as immutable heap.
        crate::typedef::mark_cpython_heap_type(tp, true);
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(tp, true);
            pyre_object::w_type_set_weakrefable(tp, true);
            pyre_object::typeobject::w_type_set_hasdict(tp, true);
        }
        tp
    })
}

fn text_iobase_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_base(
            "_io._TextIOBase",
            init_text_iobase_type,
            io_base_type(),
        );
        // `_iomodule.c:ADD_TYPE` creates textiobase_spec as immutable heap.
        crate::typedef::mark_cpython_heap_type(tp, true);
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(tp, true);
            pyre_object::w_type_set_weakrefable(tp, true);
            pyre_object::typeobject::w_type_set_hasdict(tp, true);
        }
        tp
    })
}

pub(crate) fn call_method_result(
    obj: PyObjectRef,
    name: &str,
    args: &[PyObjectRef],
) -> crate::PyResult {
    let result = crate::baseobjspace::call_method(obj, name, args);
    if result.is_null() {
        Err(crate::call::take_call_error()
            .unwrap_or_else(|| crate::PyError::runtime_error(format!("{name} failed"))))
    } else {
        Ok(result)
    }
}

/// CPython 3.14 `_io.text_encoding`: select the PEP 597 spelling and emit the
/// default-encoding warning at the caller-selected stack depth.
pub(crate) fn text_encoding(
    encoding: PyObjectRef,
    stacklevel: i64,
) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { !pyre_object::is_none(encoding) } {
        return Ok(encoding);
    }
    if crate::importing::warn_default_encoding_flag() {
        crate::warn::warn_category(
            "'encoding' argument not specified",
            "EncodingWarning",
            stacklevel + 1,
        )?;
    }
    Ok(w_str_new_managed(
        if crate::importing::utf8_mode_flag() != 0 {
            "utf-8"
        } else {
            "locale"
        },
    ))
}

crate::py_module! {
    "_io",
    interpleveldefs: {
        "DEFAULT_BUFFER_SIZE" => w_int_new(DEFAULT_BUFFER_SIZE),
    },
    functions: {
        "open"            / * = crate::builtins::builtin_open,
        // `io.open_code(path)` — `_PyIO_open_code` opens the path in binary
        // read mode ("rb"); pyre has no audit hooks so it is just `open`.
        "open_code"       / * = |args| {
            let path = args.first().copied().unwrap_or_else(w_none);
            crate::builtins::builtin_open(&[path, w_str_new("rb")])
        },
        "text_encoding"   / * = |args| {
            let encoding = args.first().copied().unwrap_or_else(w_none);
            let stacklevel = args.get(1).copied().map(crate::builtins::space_index_w)
                .transpose()?.unwrap_or(2);
            text_encoding(encoding, stacklevel)
        },
    },
    extra_init: |ns| {
        let mut ns = ns;
        let extra_ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        ns = pyre_object::gc_roots::shadow_stack_get(extra_ns_slot);
        // `Modules/_io/_iomodule.c`:
        //   UnsupportedOperation = class UnsupportedOperation(OSError, ValueError)
        // A real exception class so `raise`/`except` and io.py's
        // `UnsupportedOperation.__module__ = "io"` work.  Falls back to a
        // single OSError base if the builtin exceptions aren't registered.
        // A temporary `&[PyObjectRef]` is not a root. Name each base so
        // `with_roots!` can publish it (`ShadowStackFrameworkGCTransformer.push_roots`).
        let mut os_error = crate::builtins::lookup_exc_class("OSError")
            .expect("OSError must be registered before _io init");
        let unsupported = if let Some(mut value_error) =
            crate::builtins::lookup_exc_class("ValueError")
        {
            pyre_object::with_roots!(ns, os_error, value_error => {
                crate::builtins::make_exc_type_multi(
                    "io.UnsupportedOperation",
                    crate::builtins::exc_os_error_new,
                    &[os_error, value_error],
                )
            })
        } else {
            pyre_object::with_roots!(ns, os_error => {
                crate::builtins::make_exc_type_multi(
                    "io.UnsupportedOperation",
                    crate::builtins::exc_os_error_new,
                    &[os_error],
                )
            })
        };
        // Publish `ns` and the class together before the static takes the
        // word. `pin_root(ns)` alone is a safepoint while the class is still
        // unpublished (`RootScope::pin_roots`).
        let unsupported_base = pyre_object::gc_roots::pin_roots(&[ns, unsupported]);
        ns = pyre_object::gc_roots::shadow_stack_get(unsupported_base);
        let mut unsupported = pyre_object::gc_roots::shadow_stack_get(unsupported_base + 1);
        UNSUPPORTED_OPERATION_TYPE.set(unsupported);
        crate::__pyre_store!(ns, "UnsupportedOperation", unsupported);

        // `_io.BlockingIOError` aliases the builtin BlockingIOError.
        if let Some(mut blocking) = crate::builtins::lookup_exc_class("BlockingIOError") {
            crate::__pyre_store!(ns, "BlockingIOError", blocking);
        }

        // Abstract base classes as W_TypeObject (required for io.py class inheritance).
        // PyPy hierarchy: RawIOBase/BufferedIOBase/TextIOBase all derive IOBase.
        // Each type is published before the next `type()` call: those builds
        // allocate (`ShadowStackFrameworkGCTransformer.push_roots`).
        let io_base_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(io_base_type());
        let raw_base_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(raw_iobase_type());
        let buffered_base_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(buffered_iobase_type());
        let text_base_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(text_iobase_type());
        ns = pyre_object::gc_roots::shadow_stack_get(extra_ns_slot);
        for (name, slot) in [
            ("_IOBase", io_base_slot),
            ("_RawIOBase", raw_base_slot),
            ("_BufferedIOBase", buffered_base_slot),
            ("_TextIOBase", text_base_slot),
        ] {
            let mut typ = pyre_object::gc_roots::shadow_stack_get(slot);
            unsafe {
                pyre_object::w_type_set_acceptable_as_base_class(typ, true);
                pyre_object::w_type_set_weakrefable(typ, true);
                pyre_object::typeobject::w_type_set_hasdict(typ, true);
            };
            crate::__pyre_store!(ns, name, typ);
        }

        // Concrete stream classes as subclassable W_TypeObjects.  stdlib
        // modules derive from them at import (`class ExFileObject(
        // io.BufferedReader)` in tarfile, `class _MockRawIO(...)` in
        // test_io), so they must be real types, not function stubs.
        // `FileIO` derives from `_RawIOBase`; the buffered classes from
        // `_BufferedIOBase` (`Modules/_io/_iomodule.c` PyInit__io).
        let file_io_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(fileio_type());
        let bytes_io_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(interp_bytesio::type_object());
        let string_io_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(interp_stringio::type_object());
        let buffered_reader_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(interp_bufferedio::type_object());
        let buffered_writer_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(buffered_writer::type_object());
        let buffered_rwpair_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(buffered_rwpair::type_object());
        let buffered_random_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(buffered_random::type_object());
        ns = pyre_object::gc_roots::shadow_stack_get(extra_ns_slot);
        for (name, slot) in [
            ("FileIO", file_io_slot),
            ("BytesIO", bytes_io_slot),
            ("StringIO", string_io_slot),
            ("BufferedReader", buffered_reader_slot),
            ("BufferedWriter", buffered_writer_slot),
            ("BufferedRWPair", buffered_rwpair_slot),
            ("BufferedRandom", buffered_random_slot),
        ] {
            let mut t = pyre_object::gc_roots::shadow_stack_get(slot);
            unsafe {
                pyre_object::w_type_set_acceptable_as_base_class(t, true);
                pyre_object::typeobject::w_type_set_hasdict(t, true);
            }
            crate::__pyre_store!(ns, name, t);
        }

        // `PyInit__io` exposes the PEP 528 raw console stream only on
        // Windows.  PyPy's `W_WinConsoleIO` likewise derives directly from
        // `W_RawIOBase`; the descriptor, ownership flag and UTF-8 carry buffer
        // live on that one stream object.
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            let console_io_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(interp_win32consoleio::type_object());
            ns = pyre_object::gc_roots::shadow_stack_get(extra_ns_slot);
            let mut console_io = pyre_object::gc_roots::shadow_stack_get(console_io_slot);
            unsafe {
                pyre_object::w_type_set_acceptable_as_base_class(console_io, true);
                pyre_object::typeobject::w_type_set_hasdict(console_io, true);
            }
            crate::__pyre_store!(ns, "_WindowsConsoleIO", console_io);
        }

        // `TextIOWrapper` is a real (subclassable) type: stdlib modules such
        // as argparse / pickle / _android_support derive from it
        // (`class StdIOBuffer(io.TextIOWrapper)`).  Its `__init__` configures
        // the underlying buffer + encoding so `TextIOWrapper(buffer, ...)`
        // and a subclass's `super().__init__(...)` both work.
        let text_io_wrapper_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(crate::builtins::text_io_wrapper_type());
        ns = pyre_object::gc_roots::shadow_stack_get(extra_ns_slot);
        let mut text_io_wrapper = pyre_object::gc_roots::shadow_stack_get(text_io_wrapper_slot);
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(text_io_wrapper, true);
        }
        crate::__pyre_store!(ns, "TextIOWrapper", text_io_wrapper);

        // The remaining pure-Python newline decoder needs `_TextIOBase` bound
        // before this source runs; that is what puts
        // this install here rather than in the `appleveldefs:` table, which
        // the macro expands ahead of `extra_init`.
        pyre_object::with_roots!(ns => crate::importing::appleveldef_install_seeded(
            ns,
            include_str!("_io_app.py"),
            "_io_app.py",
            "_io",
            &["IncrementalNewlineDecoder"],
            &[(
                "_TextIOBase",
                pyre_object::gc_roots::shadow_stack_get(text_base_slot),
            )],
        ))?;
    }
}
