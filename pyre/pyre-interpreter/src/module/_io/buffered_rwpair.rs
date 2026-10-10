//! Paired buffered streams — PyPy `W_BufferedRWPair`.

use pyre_object::*;

use super::DEFAULT_BUFFER_SIZE;

// CPython 3.14 Modules/_io/_iomodule.c:ADD_TYPE creates the immutable
// BufferedRWPair heap spec.
#[crate::pyre_class("_io.BufferedRWPair", cpython_heaptype, user_layout, weakrefable)]
pub struct W_BufferedRWPair {
    w_reader: PyObjectRef,
    w_writer: PyObjectRef,
    // interp_iobase.py W_IOBase.w_dict — null until getdict.
    pub(crate) w_dict: PyObjectRef,
}

impl Default for W_BufferedRWPair {
    fn default() -> Self {
        Self {
            ob: PyObject::default(),
            w_reader: PY_NULL,
            w_writer: PY_NULL,
            w_dict: PY_NULL,
            lifeline: PY_NULL,
        }
    }
}

impl W_BufferedRWPair {
    fn self_obj(&self) -> PyObjectRef {
        self as *const Self as PyObjectRef
    }

    fn from_slot(slot: usize) -> &'static mut Self {
        super::from_pinned_slot(slot)
    }

    fn pin_self(&self) -> usize {
        super::pin_io_self(self.self_obj())
    }

    fn check_reader(&self) -> Result<PyObjectRef, crate::PyError> {
        if self.w_reader.is_null() {
            Err(crate::PyError::value_error(
                "I/O operation on uninitialized object",
            ))
        } else {
            Ok(self.w_reader)
        }
    }

    fn check_writer(&self) -> Result<PyObjectRef, crate::PyError> {
        if self.w_writer.is_null() {
            Err(crate::PyError::value_error(
                "I/O operation on uninitialized object",
            ))
        } else {
            Ok(self.w_writer)
        }
    }

    fn reader_call(&self, name: &str, args: &[PyObjectRef]) -> crate::PyResult {
        super::call_method_result(self.check_reader()?, name, args)
    }

    fn writer_call(&self, name: &str, args: &[PyObjectRef]) -> crate::PyResult {
        super::call_method_result(self.check_writer()?, name, args)
    }
}

#[crate::pyre_methods(
    base = super::buffered_iobase_type(),
    weakrefable,
    doc = "BufferedRWPair(reader, writer, buffer_size=DEFAULT_BUFFER_SIZE)"
)]
impl W_BufferedRWPair {
    #[staticmethod]
    fn __new__(cls: PyObjectRef, _args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // `check_user_subclass` can collect. The slot is the live word;
        // this pin's argument is not read again.
        let _roots = pyre_object::gc_roots::push_roots();
        let cls_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(cls);
        crate::typedef::check_user_subclass(
            type_object(),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        )?;
        let obj = W_BufferedRWPair::allocate_instance(
            W_BufferedRWPair::default(),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        );
        // interp_bufferedio.py `needs_finalizer`: `self.w_writer` and
        // `self.w_reader` have their own finalizer, so the pair itself needs
        // none. A subclass keeps one — its `close` may do anything.
        let cls = pyre_object::gc_roots::shadow_stack_get(cls_slot);
        let needs_finalizer = !cls.is_null() && !std::ptr::eq(cls, type_object());
        Ok(super::tag_io_instance_with_finalizer(
            obj,
            cls,
            needs_finalizer,
        ))
    }

    fn __init__(
        &mut self,
        w_reader: PyObjectRef,
        w_writer: PyObjectRef,
        #[default(DEFAULT_BUFFER_SIZE)] buffer_size: i64,
    ) -> Result<(), crate::PyError> {
        // Constructing the inner BufferedReader / BufferedWriter allocates.
        // Pin `self` first (`interp_bufferedio.py` `W_BufferedRWPair.descr_init`)
        // so the field stores land on the live nursery object.
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).w_reader = PY_NULL;
        Self::from_slot(slot).w_writer = PY_NULL;

        let _ = pyre_object::gc_roots::pin_root(w_reader);
        let _ = pyre_object::gc_roots::pin_root(w_writer);
        let input_sp = pyre_object::gc_roots::shadow_stack_len() - 2;
        let reader_size_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_int_new(buffer_size));
        let reader = crate::call::call_function_impl_result(
            super::interp_bufferedio::type_object(),
            &[
                pyre_object::gc_roots::shadow_stack_get(input_sp),
                pyre_object::gc_roots::shadow_stack_get(reader_size_slot),
            ],
        )?;
        let _ = pyre_object::gc_roots::pin_root(reader);
        let reader_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let writer_size_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_int_new(buffer_size));
        let writer = crate::call::call_function_impl_result(
            super::buffered_writer::type_object(),
            &[
                pyre_object::gc_roots::shadow_stack_get(input_sp + 1),
                pyre_object::gc_roots::shadow_stack_get(writer_size_slot),
            ],
        )?;
        let writer_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(writer);
        let this = Self::from_slot(slot);
        this.w_reader = pyre_object::gc_roots::shadow_stack_get(reader_slot);
        this.w_writer = pyre_object::gc_roots::shadow_stack_get(writer_slot);
        pyre_object::gc_hook::try_gc_write_barrier(
            this as *mut Self as pyre_object::gc_hook::GCREF,
        );
        Ok(())
    }

    fn read(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        self.reader_call("read", &args[1..])
    }

    fn peek(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        self.reader_call("peek", &args[1..])
    }

    fn read1(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        self.reader_call("read1", &args[1..])
    }

    fn readinto(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        self.reader_call("readinto", &args[1..])
    }

    fn readable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.reader_call("readable", &[])
    }

    fn write(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        self.writer_call("write", &args[1..])
    }

    fn flush(&self) -> Result<PyObjectRef, crate::PyError> {
        self.writer_call("flush", &[])
    }

    fn writable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.writer_call("writable", &[])
    }

    fn close(&self) -> Result<(), crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let writer = Self::from_slot(slot).check_writer()?;
        let writer_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(writer);
        let writer_close = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(writer_slot),
            "close",
            &[],
        );
        let (reader, mut writer_error) = match writer_close {
            Err(error) => {
                let _err_roots = pyre_object::gc_roots::push_roots();
                let mut error = error;
                let error_slot = error.pin(&_err_roots);
                let reader = Self::from_slot(slot).check_reader()?;
                error.reload(&_err_roots, error_slot);
                (reader, Some(error))
            }
            Ok(_) => (Self::from_slot(slot).check_reader()?, None),
        };
        let reader_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(reader);
        // The reader close can collect. The writer handle is a `PyObjectRef`
        // for that call so `pin_root`'s returned word is the live root.
        let writer_ptr = writer_error
            .take()
            .map(|err| err.as_raw() as pyre_object::PyObjectRef)
            .unwrap_or(pyre_object::PY_NULL);
        let (reader_close, writer_error) = if writer_ptr.is_null() {
            (
                super::call_method_result(
                    pyre_object::gc_roots::shadow_stack_get(reader_slot),
                    "close",
                    &[],
                ),
                None,
            )
        } else {
            let _roots = pyre_object::gc_roots::push_roots();
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let writer_ptr = pyre_object::gc_roots::pin_root(writer_ptr);
            let result = super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(reader_slot),
                "close",
                &[],
            );
            let writer_ptr = pyre_object::gc_roots::shadow_stack_get(slot);
            let _ = writer_ptr;
            (
                result,
                Some(crate::PyError::from_raw(
                    pyre_object::gc_roots::shadow_stack_get(slot),
                )),
            )
        };
        if let Err(reader_error) = reader_close {
            let reader_error = if let Some(context) = writer_error {
                let _roots = pyre_object::gc_roots::push_roots();
                let mut context = context;
                let mut reader_error = reader_error;
                let context_slot = context.pin(&_roots);
                let reader_slot = reader_error.pin(&_roots);
                let context_obj = context.to_exc_object();
                context.reload(&_roots, context_slot);
                let _ = pyre_object::gc_roots::pin_root(context_obj);
                let context_obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                reader_error.reload(&_roots, reader_slot);
                let reader_obj = reader_error.to_exc_object();
                reader_error.reload(&_roots, reader_slot);
                let _ = pyre_object::gc_roots::pin_root(reader_obj);
                let reader_obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                unsafe {
                    pyre_object::interp_exceptions::w_exception_set_context(
                        pyre_object::gc_roots::shadow_stack_get(reader_obj_slot),
                        pyre_object::gc_roots::shadow_stack_get(context_obj_slot),
                    )
                };
                reader_error.reload(&_roots, reader_slot);
                reader_error
                    .set_exc_object(pyre_object::gc_roots::shadow_stack_get(reader_obj_slot));
                return Err(reader_error);
            } else {
                reader_error
            };
            return Err(reader_error);
        }
        if let Some(error) = writer_error {
            return Err(error);
        }
        Ok(())
    }

    fn isatty(&self) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let writer = Self::from_slot(slot).writer_call("isatty", &[])?;
        if crate::baseobjspace::is_true(writer)? {
            Ok(w_bool_from(true))
        } else {
            Self::from_slot(slot).reader_call("isatty", &[])
        }
    }

    #[getter]
    fn closed(&self) -> Result<PyObjectRef, crate::PyError> {
        crate::baseobjspace::getattr_str(self.check_writer()?, "closed")
    }
}
