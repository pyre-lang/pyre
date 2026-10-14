//! `__pypy__.PickleBuffer` — `pypy/module/__pypy__/interp_buffer.py
//! W_PickleBuffer`. Wraps a bytes-like object so the `_pickle` accelerator
//! can serialize it either in-band or out-of-band (protocol 5). The
//! `_pickle` save path recognizes the wrapper via `from_obj` and reads its
//! contents through `buffer_view`.

use pyre_object::PyObjectRef;

use crate::PyError;

/// `interp_buffer.newmemoryview` — construct a formatted N-D view over an
/// existing memoryview.  This is an internal PyPy helper, not the public
/// `memoryview.cast`: structured format strings are accepted verbatim.
pub(crate) fn newmemoryview(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // Bound scope: required `buf`/`itemsize`/`format`, optional `shape`/`strides`
    // (`PY_NULL` omitted). Extra positionals are refused by the Signature
    // overlay; a direct test slice still names the count here.
    let given = args.iter().filter(|o| !o.is_null()).count();
    if given > 5 {
        return Err(PyError::type_error(format!(
            "newmemoryview() takes at most 5 arguments ({given} given)"
        )));
    }
    let arg = |index: usize| args.get(index).copied().filter(|o| !o.is_null());
    let w_obj = arg(0)
        .ok_or_else(|| PyError::type_error("newmemoryview() missing required argument 'buf'"))?;
    let itemsize_obj = arg(1).ok_or_else(|| {
        PyError::type_error("newmemoryview() missing required argument 'itemsize'")
    })?;
    let fmt_obj = arg(2)
        .ok_or_else(|| PyError::type_error("newmemoryview() missing required argument 'format'"))?;
    // `if w_shape` / `if w_strides` are RPython reference tests, not truth
    // tests: only an omitted argument selects the derive-it path.  An empty
    // sequence still counts as supplied, and so does an explicit `None`, which
    // then fails in `dimensions` with "'NoneType' object is not iterable".
    let shape_arg = arg(3);
    let strides_arg = arg(4);
    let has_shape = shape_arg.is_some();
    let has_strides = strides_arg.is_some();
    let shape_obj = shape_arg.unwrap_or_else(pyre_object::w_none);
    let strides_obj = strides_arg.unwrap_or_else(pyre_object::w_none);
    let _roots = pyre_object::gc_roots::push_roots();
    // One batch publish rather than five `pin_root`s: the first pin's forwarding
    // query is itself a safepoint, so the four raw locals behind it would be
    // read after a foreign collection could already have moved them.
    let sp =
        pyre_object::gc_roots::pin_roots(&[w_obj, itemsize_obj, fmt_obj, shape_obj, strides_obj]);
    let w_obj = pyre_object::gc_roots::shadow_stack_get(sp);
    if !unsafe { pyre_object::memoryview::is_w_memoryview(w_obj) } {
        return Err(PyError::value_error("memoryview expected"));
    }
    unsafe { crate::builtins::memoryview_check_released(w_obj)? };
    let itemsize = crate::baseobjspace::int_w(pyre_object::gc_roots::shadow_stack_get(sp + 1))?;
    let fmt_obj = pyre_object::gc_roots::shadow_stack_get(sp + 2);
    if !unsafe { pyre_object::is_str(fmt_obj) } {
        return Err(PyError::type_error("format must be a string"));
    }
    let fmt = unsafe { pyre_object::w_str_get_wtf8(fmt_obj) }.to_string();
    let shape_obj = pyre_object::gc_roots::shadow_stack_get(sp + 3);
    let shape = if has_shape {
        dimensions(shape_obj)?
    } else {
        Vec::new()
    };
    let strides_obj = pyre_object::gc_roots::shadow_stack_get(sp + 4);
    let strides = if has_strides {
        dimensions(strides_obj)?
    } else {
        Vec::new()
    };
    if !has_shape && has_strides && strides.len() != 1 {
        return Err(PyError::value_error(
            "strides must have a single value if shape not provided",
        ));
    }
    if has_shape && has_strides && shape.len() != strides.len() {
        // The reference raises before its shape accumulator is filled, so the
        // diagnostic names an empty shape rather than the supplied one.
        return Err(PyError::value_error(format!(
            "shape [] does not match strides {strides:?}"
        )));
    }
    let w_obj = pyre_object::gc_roots::shadow_stack_get(sp);
    // `lgt` is the source's own length — for an N-D view that is the outermost
    // dimension, not the element count — and the byte extent every check below
    // compares against follows from it.
    let lgt = crate::baseobjspace::len_w(w_obj)?;
    let w_obj = pyre_object::gc_roots::shadow_stack_get(sp);
    let old_size = unsafe { pyre_object::memoryview::w_memoryview_itemsize(w_obj) };
    let nbytes = lgt.saturating_mul(old_size);
    let shape = if !has_shape {
        if itemsize == 0 {
            return Err(PyError::value_error("cannot guess shape when itemsize==0"));
        }
        if nbytes % itemsize != 0 {
            return Err(PyError::value_error(format!(
                "itemsize {itemsize} does not match obj len/itemsize {lgt}/{old_size}"
            )));
        }
        vec![nbytes / itemsize]
    } else {
        shape
    };
    if shape.len() > 64 {
        return Err(PyError::value_error(
            "number of dimensions must not exceed 64",
        ));
    }
    let strides = if !has_strides {
        let mut result = vec![itemsize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            result[i] = result[i + 1].saturating_mul(shape[i + 1]);
        }
        result
    } else {
        if strides.len() != shape.len() {
            return Err(PyError::value_error(format!(
                "shape {shape:?} does not match strides {strides:?}"
            )));
        }
        strides
    };
    if has_strides {
        // PyPy's `FormatBufferViewND` validates the covered byte extent from
        // strides (a strided view may expose fewer elements than its source).
        let mut span = 1i64;
        for i in (0..shape.len()).rev() {
            if span != 0 && strides[i] % span != 0 {
                return Err(PyError::value_error(
                    "strides does not match shape, itemsize",
                ));
            }
            let step = if span == 0 { 0 } else { strides[i] / span };
            span = span
                .checked_mul(shape[i])
                .and_then(|v| v.checked_mul(step))
                .ok_or_else(|| PyError::value_error("shape exceeds buffer size"))?;
        }
        if span != nbytes {
            return Err(PyError::value_error(format!(
                "shape * strides / itemsize {shape:?} * {strides:?} / {itemsize} \
                 does not match obj data {lgt} * {old_size}"
            )));
        }
    } else {
        let mut product = 1i64;
        for &dim in &shape {
            product = product
                .checked_mul(dim)
                .ok_or_else(|| PyError::value_error("shape exceeds buffer size"))?;
        }
        if product.saturating_mul(itemsize) != nbytes {
            return Err(PyError::value_error(format!(
                "shape/itemsize {shape:?}/{itemsize} does not match obj len/itemsize {lgt}/{old_size}"
            )));
        }
    }
    if nbytes > 0 {
        for (&stride, &dim) in strides.iter().zip(&shape) {
            if stride.saturating_mul(dim) > nbytes {
                return Err(PyError::value_error(format!(
                    "shape {shape:?} and strides {strides:?} exceed object size {nbytes}"
                )));
            }
        }
    }
    Ok(unsafe {
        crate::builtins::w_memoryview_new_formatted_nd(w_obj, &fmt, itemsize, &shape, &strides)
    })
}

fn dimensions(obj: PyObjectRef) -> Result<Vec<i64>, PyError> {
    let items = crate::baseobjspace::unpackiterable(obj, -1)?;
    // Converting one element runs its `__index__`, which allocates and can
    // relocate every element still sitting in this plain `Vec`.  Publish the
    // whole batch first and read each one back through its root slot.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&items);
    let mut values = Vec::with_capacity(items.len());
    for index in 0..items.len() {
        values.push(crate::baseobjspace::int_w(
            pyre_object::gc_roots::shadow_stack_get(base + index),
        )?);
    }
    Ok(values)
}

/// `pypy/module/__pypy__/interp_buffer.py:W_Bufferable`.
///
/// This is deliberately a zero-payload, subclassable type.  PyPy's base
/// implementation forwards to a subclass's `__buffer__`; the exact base
/// instance is an error, so accidentally using the helper without an
/// override is diagnosed instead of recursing forever.
pub mod bufferable_impl {
    use super::*;

    #[crate::pyre_class("__pypy__.Bufferable", user_layout)]
    #[derive(Default)]
    pub struct W_Bufferable {}

    #[crate::pyre_methods(doc = "Base class for objects implementing the buffer protocol.")]
    impl W_Bufferable {
        /// `generic_new_descr(W_Bufferable)` accepts and ignores constructor
        /// arguments, while preserving a user-defined subtype on the instance.
        #[staticmethod]
        fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
            let _ = args;
            let base = type_object();
            crate::typedef::check_user_subclass(base, cls)?;
            // `objspace.py` `allocate_instance`: exact `Bufferable` stays the
            // base layout; a subtype is `typedef.py` `_getusercls`.
            Ok(W_Bufferable::allocate_instance(
                W_Bufferable {
                    ob: pyre_object::PyObject {
                        ob_type: std::ptr::null(),
                        w_class: std::ptr::null_mut(),
                    },
                },
                cls,
            ))
        }

        /// PyPy `W_Bufferable.descr_buffer`: subclasses provide the actual
        /// exporter and return a memoryview; the base class itself is abstract.
        fn __buffer__(&self, mut w_flags: PyObjectRef) -> Result<PyObjectRef, PyError> {
            let self_obj = self as *const W_Bufferable as PyObjectRef;
            let self_type = crate::typedef::r#type(self_obj)
                .map(|p| p.as_ptr())
                .unwrap_or(pyre_object::PY_NULL);
            if std::ptr::eq(self_type, type_object()) {
                return Err(PyError::value_error("override __buffer__ in a subclass"));
            }
            let method = pyre_object::with_roots!(w_flags => crate::baseobjspace::getattr_str(self_obj, "__buffer__"))?;
            let result = crate::baseobjspace::call_function(method, &[w_flags]);
            if result.is_null() {
                Err(crate::call::take_call_error()
                    .unwrap_or_else(|| PyError::runtime_error("__buffer__ call failed")))
            } else {
                Ok(result)
            }
        }
    }
}

#[crate::pyre_class("pickle.PickleBuffer", weakrefable)]
pub struct W_PickleBuffer {
    /// The wrapped buffer-supporting object, or `None` after `release()`.
    w_obj: PyObjectRef,
    /// `self.buf is not None and self.buf.needs_release()`: owns exactly one
    /// exporter release until `release()` or `_finalize_` takes it.
    export_active: bool,
    /// The generic `W_Root.__buffer_w` path returns a temporary memoryview.
    /// Keep and release that carrier as part of `self.buf`; a concrete
    /// exporter supplied directly by the caller is borrowed instead.
    release_memoryview: bool,
    /// The PEP 688 exporter paired with the temporary memoryview, or `None`
    /// for a native exporter.  This is the inline owner corresponding to the
    /// buffer protocol's `__release_buffer__(view)` obligation.
    w_release_exporter: PyObjectRef,
}

#[crate::pyre_methods(
    doc = "PickleBuffer(buffer) -> wrapper for potentially out-of-band serialization.",
    // interp_buffer.py make_weakref_descr(W_PickleBuffer)
    weakrefable
)]
impl W_PickleBuffer {
    #[staticmethod]
    fn __new__(_cls: PyObjectRef, w_obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
        // interp_buffer.py descr_new_picklebuffer — acquire the
        // export while constructing the object; PickleBuffer has no separate
        // __init__ phase.
        let (w_buffer, release_memoryview, w_release_exporter) = acquire_pickle_buffer(w_obj)?;
        let _roots = pyre_object::gc_roots::push_roots();
        // Both operands are Rust-stack copies out of `acquire_pickle_buffer`, so
        // they have to become visible in one publish: pinning the first alone
        // opens a safepoint the second copy would not survive.
        let sp = pyre_object::gc_roots::pin_roots(&[w_buffer, w_release_exporter]);
        // `allocate` may collect; store the post-collection exporter rather
        // than the stale Rust-stack copy.
        let r_obj = pyre_object::gc_roots::shadow_stack_get(sp);
        let export_active = unsafe { crate::builtins::buffer_export_incref(r_obj) };
        // Like the other self-mutating interp-level payloads, keep the
        // wrapper stationary.  More importantly for `register_finalizer`, an
        // acquired export must enter the old-object finalizer queue at its
        // definitive address; a nursery registration is first promoted as a
        // finalizer root and otherwise delays `_release_buf` by a collection.
        let w_pickle_buffer = W_PickleBuffer::allocate_stable(W_PickleBuffer {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_obj: r_obj,
            export_active,
            release_memoryview,
            w_release_exporter: pyre_object::gc_roots::shadow_stack_get(sp + 1),
            lifeline: pyre_object::PY_NULL,
        });
        // `_finalize_` releases the acquired export. Pyre also routes weakref
        // invalidation through this queue, so register immutable-buffer
        // wrappers too: PyPy's collector handles their weakrefs independently
        // even when `buf.needs_release()` is false. The idempotent release
        // body is a no-op for that export.
        crate::executioncontext::register_finalizer(w_pickle_buffer);
        Ok(w_pickle_buffer)
    }

    /// `raw()` — a memoryview of the raw bytes underlying the wrapped buffer.
    /// The result is a one-dimensional unsigned-byte view (`format='B'`,
    /// itemsize 1) that aliases the source and preserves its read-only flag,
    /// regardless of the source's element format; extracting it from a
    /// non-contiguous buffer raises `BufferError`.
    ///
    /// `picklebufobject.c:161` asks `PyBuffer_IsContiguous(&view, 'A')`, so a
    /// Fortran-contiguous buffer qualifies: the bytes are still one run, and
    /// what `raw()` promises is the physical order, not the logical one.
    fn raw(&self) -> Result<PyObjectRef, PyError> {
        let w_obj = self.w_obj;
        if unsafe { pyre_object::is_none(w_obj) } {
            return Err(released_error());
        }
        let mv_type = memoryview_type()
            .ok_or_else(|| PyError::runtime_error("memoryview type unavailable"))?;
        let mut mv = crate::baseobjspace::call_function(mv_type, &[w_obj]);
        if mv.is_null() {
            return Err(crate::call::take_call_error()
                .unwrap_or_else(|| PyError::runtime_error("memoryview() failed")));
        }
        let w_contig =
            pyre_object::with_roots!(mv => crate::baseobjspace::getattr_str(mv, "contiguous"))?;
        if !pyre_object::with_roots!(mv => crate::baseobjspace::is_true(w_contig))? {
            return Err(PyError::new(
                crate::PyErrorKind::BufferError,
                "cannot extract raw buffer from non-contiguous buffer",
            ));
        }
        // Normalize to the raw byte layout so an `array('i')` or other
        // non-`'B'` source still yields a byte view.
        Ok(crate::builtins::memoryview_raw_bytes(mv))
    }

    /// PEP 688 exporter slot exposed by CPython 3.14's `pickle.PickleBuffer`.
    /// Returning the normalized raw view keeps the wrapper's acquired export
    /// alive until the PickleBuffer itself is released.
    fn __buffer__(&self, _flags: PyObjectRef) -> Result<PyObjectRef, PyError> {
        self.raw()
    }

    /// The view passed to this callback owns only its temporary carrier; the
    /// PickleBuffer's own export remains active until `release()` or finalise.
    fn __release_buffer__(&mut self, _view: PyObjectRef) -> Result<(), PyError> {
        Ok(())
    }

    /// `release()` — drop the reference to the underlying buffer.
    fn release(&mut self) -> Result<(), PyError> {
        self.release_export_result()
    }
}

impl W_PickleBuffer {
    /// `_release_buf` (`interp_buffer.py`) — clear ownership before
    /// calling the exporter release, making repeated explicit/finalizer calls
    /// idempotent.
    pub(crate) fn release_export(&mut self) {
        let _ = self.release_export_result();
    }

    fn release_export_result(&mut self) -> Result<(), PyError> {
        if unsafe { pyre_object::is_none(self.w_obj) } {
            return Ok(());
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(self.w_obj);
        let _ = pyre_object::gc_roots::pin_root(self.w_release_exporter);
        let w_obj = pyre_object::gc_roots::shadow_stack_get(sp);
        if self.export_active {
            self.export_active = false;
            unsafe { crate::builtins::buffer_export_decref(w_obj) };
        }
        self.w_obj = pyre_object::w_none();
        self.w_release_exporter = pyre_object::w_none();
        if self.release_memoryview {
            self.release_memoryview = false;
            let w_exporter = pyre_object::gc_roots::shadow_stack_get(sp + 1);
            let callback_result = if let Some(w_release) =
                unsafe { crate::baseobjspace::lookup(w_exporter, "__release_buffer__") }
            {
                let w_type = crate::typedef::r#type(w_exporter).map_or(w_exporter, |p| p.as_ptr());
                unsafe {
                    crate::baseobjspace::get_and_call_function(
                        w_release,
                        w_exporter,
                        w_type,
                        &[pyre_object::gc_roots::shadow_stack_get(sp)],
                    )
                }
                .map(|_| ())
            } else {
                Ok(())
            };
            // The callback is advisory cleanup; the temporary carrier still
            // has to be released if it raises, matching a try/finally around
            // the acquired Py_buffer.
            match callback_result {
                Ok(()) => {
                    crate::builtins::memoryview_release(&[
                        pyre_object::gc_roots::shadow_stack_get(sp),
                    ])?;
                }
                Err(error) => {
                    let mut error = error;
                    let error_slot = error.pin(&_roots);
                    let _ = crate::builtins::memoryview_release(&[
                        pyre_object::gc_roots::shadow_stack_get(sp),
                    ]);
                    error.reload(&_roots, error_slot);
                    return Err(error);
                }
            }
        }
        Ok(())
    }
}

/// `W_PickleBuffer.typedef.acceptable_as_base_class = False` — return the
/// shared type after applying PyPy's explicit final-type flag. Both
/// `__pypy__.PickleBuffer` and `_pickle.PickleBuffer` call this accessor, so
/// the flag is installed regardless of which module is imported first.
pub fn picklebuffer_type_object() -> PyObjectRef {
    let tp = type_object();
    unsafe { pyre_object::w_type_set_acceptable_as_base_class(tp, false) };
    tp
}

impl W_PickleBuffer {
    /// The wrapped buffer object (`None` after `release()`), read by the
    /// `_pickle` save path.
    pub fn wrapped(&self) -> PyObjectRef {
        self.w_obj
    }
}

/// If `obj` is a `PickleBuffer`, the wrapped exporter its buffer protocol
/// forwards to — `buffer_w` delegates to the underlying object, so `bytes(pb)`
/// / `memoryview(pb)` operate on the wrapped `bytes`/`bytearray`/`array`/
/// `memoryview`. `Some(Err(..))` once the buffer was released; `None` when
/// `obj` is not a `PickleBuffer`.
pub(crate) fn forwarded_exporter(obj: PyObjectRef) -> Option<Result<PyObjectRef, PyError>> {
    W_PickleBuffer::from_obj(obj).map(|pb| {
        let w = pb.wrapped();
        if unsafe { pyre_object::is_none(w) } {
            Err(released_error())
        } else {
            Ok(w)
        }
    })
}

fn released_error() -> PyError {
    PyError::value_error("operation forbidden on released PickleBuffer object")
}

fn type_name(obj: PyObjectRef) -> String {
    match crate::typedef::r#type(obj) {
        Some(t) => unsafe { pyre_object::w_type_get_name(t.as_ptr()) }.to_string(),
        None => "object".to_string(),
    }
}

/// `space.buffer_w(w_obj, BUF_FULL_RO)` for PickleBuffer construction.
///
/// Concrete native exporters keep their identity.  The W_Root fallback is
/// the Python 3.14/PyPy PEP 688 path: look up and descriptor-bind
/// `__buffer__`, pass `BUF_FULL_RO`, and require a memoryview result.  The
/// returned boolean says that the result is the temporary carrier owned by
/// this acquisition and must be released with the PickleBuffer.
fn acquire_pickle_buffer(obj: PyObjectRef) -> Result<(PyObjectRef, bool, PyObjectRef), PyError> {
    unsafe {
        if pyre_object::is_bytes(obj)
            || pyre_object::is_bytearray(obj)
            || pyre_object::interp_array::is_array(obj)
            || is_memoryview(obj)
        {
            if is_memoryview(obj) {
                crate::builtins::memoryview_check_released(obj)?;
            }
            return Ok((obj, false, pyre_object::w_none()));
        }

        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(
            crate::baseobjspace::BUF_FULL_RO as i64,
        ));
        let r_obj = pyre_object::gc_roots::shadow_stack_get(sp);
        if let Some(w_impl) = crate::baseobjspace::lookup(r_obj, "__buffer__") {
            let _ = pyre_object::gc_roots::pin_root(w_impl);
            let w_type = crate::typedef::r#type(r_obj).map_or(r_obj, |p| p.as_ptr());
            let w_result = crate::baseobjspace::get_and_call_function(
                pyre_object::gc_roots::shadow_stack_get(sp + 2),
                r_obj,
                w_type,
                &[pyre_object::gc_roots::shadow_stack_get(sp + 1)],
            )?;
            if is_memoryview(w_result) {
                crate::builtins::memoryview_check_released(w_result)?;
                return Ok((w_result, true, pyre_object::gc_roots::shadow_stack_get(sp)));
            }
            return Err(PyError::type_error(format!(
                "a bytes-like object is required, not '{}'",
                type_name(pyre_object::gc_roots::shadow_stack_get(sp))
            )));
        }
        // Interp-level exporters such as mmap / ctypes expose their native
        // `buffer_w` implementation without a Python-visible `__buffer__`
        // descriptor.  Normalize those to the same owned memoryview carrier.
        let w_result =
            crate::builtins::w_memoryview_new(pyre_object::gc_roots::shadow_stack_get(sp))?;
        Ok((w_result, true, pyre_object::w_none()))
    }
}

fn is_memoryview(obj: PyObjectRef) -> bool {
    unsafe { pyre_object::memoryview::is_w_memoryview(obj) }
}

/// Extract `(contents, readonly)` from a buffer exporter: `bytes` is
/// read-only, `bytearray` and `array` are mutable, and a `memoryview` reports
/// both through its own contents and `readonly` flag.
///
/// The contents are the bytes in physical order, which is what the buffer a
/// `PickleBuffer` saves is defined to be.
pub fn buffer_view(mut obj: PyObjectRef) -> Result<(Vec<u8>, bool), PyError> {
    unsafe {
        if pyre_object::is_bytes(obj) {
            return Ok((pyre_object::bytesobject::w_bytes_data(obj).to_vec(), true));
        }
        if pyre_object::is_bytearray(obj) {
            return Ok((
                pyre_object::bytearrayobject::w_bytearray_data(obj).to_vec(),
                false,
            ));
        }
        if pyre_object::interp_array::is_array(obj) {
            return Ok((
                pyre_object::interp_array::w_array_bytes(obj).to_vec(),
                false,
            ));
        }
    }
    if is_memoryview(obj) {
        // `_pickle.c:2691` writes `view.buf` for `view.len` bytes — the
        // physical run.  `memoryview.tobytes()` walks the strides instead, and
        // the two orders differ for a Fortran-contiguous export, so the raw
        // 1-D reinterpretation is what goes into the stream.  The only caller
        // has already refused a buffer contiguous in neither order.
        let raw = crate::builtins::memoryview_raw_bytes(obj);
        let w_data =
            pyre_object::with_roots!(obj => crate::baseobjspace::call_method(raw, "tobytes", &[]));
        if w_data.is_null() {
            return Err(crate::call::take_call_error()
                .unwrap_or_else(|| PyError::runtime_error("tobytes() failed")));
        }
        let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_data) }.to_vec();
        let w_ro = crate::baseobjspace::getattr_str(obj, "readonly")?;
        return Ok((data, crate::baseobjspace::is_true(w_ro)?));
    }
    Err(PyError::type_error(format!(
        "a bytes-like object is required, not '{}'",
        type_name(obj)
    )))
}

/// Whether the wrapped exporter's buffer is contiguous in either order, which
/// is what `_pickle.c:2683` asks of the one it is about to save.
///
/// `bytes`/`bytearray`/`array` are one-dimensional and always contiguous; a
/// `memoryview` reports through its `contiguous` flag, the `'A'` order one.
pub fn is_contiguous(obj: PyObjectRef) -> Result<bool, PyError> {
    if is_memoryview(obj) {
        let w = crate::baseobjspace::getattr_str(obj, "contiguous")?;
        return crate::baseobjspace::is_true(w);
    }
    Ok(true)
}

/// The `memoryview` builtin type via the live execution context.
fn memoryview_type() -> Option<PyObjectRef> {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return None;
    }
    unsafe { (*ec).lookup_builtin("memoryview") }
}

#[cfg(test)]
mod tests {
    use super::newmemoryview;

    #[test]
    fn formatted_memoryview_keeps_shape_strides_and_format() {
        crate::typedef::init_typeobjects();
        let source = pyre_object::w_bytearray_new(12);
        let view = crate::builtins::w_memoryview_new(source).unwrap();
        let shape = pyre_object::w_list_new(vec![pyre_object::w_int_new(6)]);
        let strides = pyre_object::w_list_new(vec![pyre_object::w_int_new(2)]);
        let result = newmemoryview(&[
            view,
            pyre_object::w_int_new(1),
            pyre_object::w_str_new("T{B:x}"),
            shape,
            strides,
        ])
        .unwrap();
        unsafe {
            assert_eq!(
                pyre_object::memoryview::w_memoryview_native_shape(result),
                vec![6]
            );
            assert_eq!(
                pyre_object::memoryview::w_memoryview_native_strides(result),
                vec![2]
            );
            assert_eq!(
                pyre_object::memoryview::w_memoryview_format_str(result),
                "T{B:x}"
            );
            assert_eq!(pyre_object::memoryview::w_memoryview_itemsize(result), 1);
        }
    }

    #[test]
    fn empty_formatted_memoryview_accepts_zero_itemsize_with_shape() {
        crate::typedef::init_typeobjects();
        let source = pyre_object::w_bytearray_new(0);
        let view = crate::builtins::w_memoryview_new(source).unwrap();
        let shape = pyre_object::w_tuple_new(vec![pyre_object::w_int_new(42)]);
        let result = newmemoryview(&[
            view,
            pyre_object::w_int_new(0),
            pyre_object::w_str_new("B"),
            shape,
        ])
        .unwrap();
        unsafe {
            assert_eq!(
                pyre_object::memoryview::w_memoryview_native_shape(result),
                vec![42]
            );
            assert_eq!(
                pyre_object::memoryview::w_memoryview_native_strides(result),
                vec![0]
            );
            assert_eq!(pyre_object::memoryview::w_memoryview_length(result), 0);
        }
    }
}
