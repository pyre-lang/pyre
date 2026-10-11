//! The compact raw-memory buffer — PyPy:
//! `pypy/module/_cffi_backend/cbuffer.py`.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;
use std::cmp::Ordering;

use super::cdataobj;
use super::ctypeobj;

const DOC: &str = "ffi.buffer(cdata[, byte_size]):\nReturn a read-write buffer object that references the raw C data\npointed to by the given 'cdata'.  The 'cdata' must be a pointer or an\narray.  Can be passed to functions expecting a buffer, or directly\nmanipulated with:\n\n    buf[:]          get a copy of it in a regular string, or\n    buf[idx]        as a single character\n    buf[:] = ...\n    buf[idx] = ...  change the content\n";

/// `cbuffer.py MiniBuffer`.
#[pyre_interpreter::pyre_class("_cffi_backend.buffer", weakrefable)]
#[derive(Default)]
pub struct MiniBuffer {
    /// `LLBuffer.raw_cdata`.
    pub ptr: *mut u8,
    /// `LLBuffer.size`.
    pub size: i64,
    /// `MiniBuffer.keepalive`.
    pub w_keepalive: PyObjectRef,
}

/// The raw buffer parameters used by the object-space buffer protocol.
pub fn mini_buffer_params(w_obj: PyObjectRef) -> Option<(*mut u8, usize)> {
    MiniBuffer::from_obj(w_obj).map(|buffer| (buffer.ptr, buffer.size.max(0) as usize))
}

/// `cbuffer.py MiniBuffer___new__`.
fn mini_buffer_new(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let n_args = args.len();
    if n_args < 2 {
        return Err(PyError::type_error("buffer() missing cdata argument"));
    }
    let arg_roots = pyre_object::gc_roots::push_roots();
    let arg_base = arg_roots.pin_roots(args);
    let mut w_cdata = arg_roots.get(arg_base + 1);
    let cdata = pyre_object::with_roots!(w_cdata => cdataobj::cdata_arg(w_cdata))?;
    let ct = ctypeobj::ctype_at(cdata.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    // The signature binder pads an omitted optional argument with `PY_NULL`.
    // `WrappedDefault(-1)` in PyPy makes that indistinguishable from no third
    // app-level argument to `MiniBuffer___new__`.
    let explicit_size = n_args > 2 && !arg_roots.get(arg_base + 2).is_null();
    let mut size = if n_args > 2 {
        let w_size = arg_roots.get(arg_base + 2);
        if !w_size.is_null() {
            pyre_object::with_roots!(w_cdata => pyre_interpreter::baseobjspace::int_w(w_size))?
        } else {
            -1
        }
    } else {
        -1
    };
    match ct.kind {
        ctypeobj::KIND_POINTER => {
            if size < 0 {
                if let Some(structobj) = cdataobj::structobj_of(w_cdata)
                    && ctypeobj::ctype_at(structobj.ctype)
                        .is_some_and(|item| item.is_struct_or_union())
                {
                    size = pyre_object::with_roots!(w_cdata => cdataobj::cdata_sizeof(structobj as *mut _ as PyObjectRef))?;
                }
                if size < 0 {
                    size = super::ctypeptr::item_of(ct)?.size;
                }
            }
        }
        ctypeobj::KIND_ARRAY => {
            if size < 0 {
                size = pyre_object::with_roots!(w_cdata => cdataobj::cdata_sizeof(w_cdata))?;
            }
        }
        _ => {
            return Err(PyError::type_error(format!(
                "expected a pointer or array cdata, got '{}'",
                ct.name()
            )));
        }
    }
    if size < 0 {
        return Err(PyError::type_error(format!(
            "don't know the size pointed to by '{}'",
            ct.name()
        )));
    }
    if explicit_size {
        let max_size = pyre_object::with_roots!(w_cdata => cdataobj::maximum_buffer_size(w_cdata))?;
        if max_size >= 0 && size > max_size {
            pyre_object::with_roots!(w_cdata => pyre_interpreter::warn::warn_category(
                &format!(
                    "ffi.buffer(cdata, bytes): creating a buffer of {size} bytes over a cdata that owns only {max_size} bytes.  This will crash if you access the extra memory"
                ),
                "UserWarning",
                1,
            ))?;
        }
    }
    let roots = pyre_object::gc_roots::push_roots();
    let keepalive_slot = roots.base();
    let _ = roots.pin_root(w_cdata);
    let obj = MiniBuffer::allocate_stable(MiniBuffer {
        ptr: cdata.ptr as *mut u8,
        size,
        ..Default::default()
    });
    MiniBuffer::from_obj(obj)
        .expect("allocate_stable hands back this layout")
        .w_keepalive = roots.get(keepalive_slot);
    // The buffer is born old-gen and the cdata it keeps alive may be young.
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj as pyre_object::gc_hook::GCREF);
    Ok(obj)
}

fn buffer_arg(w_obj: PyObjectRef) -> Result<&'static mut MiniBuffer, PyError> {
    MiniBuffer::from_obj(w_obj).ok_or_else(|| PyError::type_error("expected a buffer object"))
}

/// `MiniBuffer.descr_len`.
fn mini_len(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(pyre_object::w_int_new(buffer_arg(args[0])?.size))
}

fn adjusted_index(w_index: PyObjectRef, length: i64) -> Result<i64, PyError> {
    let mut index = pyre_interpreter::baseobjspace::getindex_w(w_index)?;
    if index < 0 {
        index += length;
    }
    if index < 0 || index >= length {
        return Err(PyError::index_error("index out of range"));
    }
    Ok(index)
}

/// `MiniBuffer.descr_getitem`.
fn mini_getitem(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let buffer = buffer_arg(args[0])?;
    if unsafe { pyre_object::sliceobject::is_slice(args[1]) } {
        let (start, stop, step) = unsafe {
            pyre_interpreter::sliceobject::slice_unpack(
                pyre_object::sliceobject::w_slice_get_start(args[1]),
                pyre_object::sliceobject::w_slice_get_stop(args[1]),
                pyre_object::sliceobject::w_slice_get_step(args[1]),
            )?
        };
        let (start, _stop, step, length) =
            pyre_interpreter::sliceobject::slice_adjust_indices(start, stop, step, buffer.size);
        let mut data = Vec::with_capacity(length as usize);
        let mut at = start;
        for _ in 0..length {
            data.push(unsafe { buffer.ptr.offset(at as isize).read() });
            at += step;
        }
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data));
    }
    let index = adjusted_index(args[1], buffer.size)?;
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&[unsafe {
        buffer.ptr.offset(index as isize).read()
    }]))
}

/// `MiniBuffer.descr_setitem`.
fn mini_setitem(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let args_base = roots.pin_roots(&[args[0], args[1], args[2]]);
    let self_slot = args_base;
    let index_slot = args_base + 1;
    let value_slot = args_base + 2;
    let w_index = roots.get(index_slot);
    let (start, size) = if unsafe { pyre_object::sliceobject::is_slice(w_index) } {
        let (raw_start, raw_stop, step) = unsafe {
            pyre_interpreter::sliceobject::slice_unpack(
                pyre_object::sliceobject::w_slice_get_start(w_index),
                pyre_object::sliceobject::w_slice_get_stop(w_index),
                pyre_object::sliceobject::w_slice_get_step(w_index),
            )?
        };
        let length = buffer_arg(roots.get(self_slot))?.size;
        let (start, _, step, size) =
            pyre_interpreter::sliceobject::slice_adjust_indices(raw_start, raw_stop, step, length);
        if step != 1 {
            return Err(PyError::not_implemented(""));
        }
        (start, size)
    } else {
        let length = buffer_arg(roots.get(self_slot))?.size;
        let index = adjusted_index(roots.get(index_slot), length)?;
        (index, 1)
    };
    let Some(value) = pyre_interpreter::baseobjspace::simple_buffer_bytes(roots.get(value_slot))?
    else {
        return Err(PyError::type_error("a bytes-like object is required"));
    };
    if value.as_bytes().len() as i64 != size {
        value.release();
        return Err(PyError::value_error(
            "cannot modify size of memoryview object",
        ));
    }
    let buffer = buffer_arg(roots.get(self_slot))?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            value.as_bytes().as_ptr(),
            buffer.ptr.offset(start as isize),
            size as usize,
        )
    };
    value.release();
    Ok(pyre_object::w_none())
}

fn comparison(args: &[PyObjectRef], mode: fn(Ordering) -> bool) -> Result<PyObjectRef, PyError> {
    if unsafe { pyre_object::unicodeobject::is_str(args[1]) } {
        return Ok(pyre_object::special::w_not_implemented());
    }
    let mut w_self = args[0];
    let w_other = args[1];
    // The view owns a bracket of its own, above this one; this one stays
    // open until the view is released.
    let self_roots = pyre_object::gc_roots::push_roots();
    let self_base = self_roots.pin_roots(&[w_self]);
    let acquired = pyre_interpreter::baseobjspace::simple_buffer_bytes(w_other);
    w_self = self_roots.get(self_base);
    let Some(other) = acquired? else {
        return Ok(pyre_object::special::w_not_implemented());
    };
    let buffer = buffer_arg(w_self)?;
    // A buffer over a NULL cdata is empty, and no slice may be built from a
    // null address even at length zero.  `_comparison_helper` reads zero bytes.
    let mine = if buffer.ptr.is_null() {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(buffer.ptr, buffer.size as usize) }
    };
    let ordering = mine.cmp(other.as_bytes());
    other.release();
    Ok(pyre_object::boolobject::w_bool_from(mode(ordering)))
}

macro_rules! comparison {
    ($name:ident, $body:expr) => {
        fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
            comparison(args, $body)
        }
    };
}
comparison!(mini_eq, |o| o == Ordering::Equal);
comparison!(mini_ne, |o| o != Ordering::Equal);
comparison!(mini_lt, |o| o == Ordering::Less);
comparison!(mini_le, |o| o != Ordering::Greater);
comparison!(mini_gt, |o| o == Ordering::Greater);
comparison!(mini_ge, |o| o != Ordering::Less);

static BUFFER_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// `_cffi_backend.buffer`.
pub fn buffer_type() -> PyObjectRef {
    BUFFER_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "_cffi_backend.buffer",
            init_buffer_type,
            pyre_interpreter::typedef::w_object(),
            <MiniBuffer as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<MiniBuffer as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(tp, false);
            pyre_object::w_type_set_weakrefable(tp, true);
            pyre_object::w_type_set_typedef_buffer(tp, Some(pyre_object::TypeDefBuffer::ReadWrite));
        }
        tp
    })
}

fn init_buffer_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    let store =
        |name: &str, value: PyObjectRef| pyre_interpreter::__pyre_put_new!(ns_slot, name, value);
    store("__doc__", pyre_object::w_str_new(DOC));
    store(
        "__new__",
        pyre_interpreter::typedef::make_new_descr_with_signature(
            mini_buffer_new,
            pyre_interpreter::gateway::Signature::new(
                vec!["cls", "cdata", "size"],
                None,
                None,
                0,
                1,
            ),
        ),
    );
    for (name, f, arity) in [
        (
            "__len__",
            mini_len as pyre_interpreter::gateway::BuiltinCodeFn,
            1u16,
        ),
        ("__getitem__", mini_getitem, 2),
        ("__setitem__", mini_setitem, 3),
        ("__eq__", mini_eq, 2),
        ("__ne__", mini_ne, 2),
        ("__lt__", mini_lt, 2),
        ("__le__", mini_le, 2),
        ("__gt__", mini_gt, 2),
        ("__ge__", mini_ge, 2),
    ] {
        store(
            name,
            pyre_interpreter::make_builtin_function_with_arity(name, f, arity),
        );
    }
    store(
        "__weakref__",
        pyre_interpreter::typedef::make_weakref_descr(pyre_object::gc_roots::shadow_stack_get(
            ns_slot,
        )),
    );
}
