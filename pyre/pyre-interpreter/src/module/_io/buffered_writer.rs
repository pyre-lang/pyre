//! Buffered binary writer — PyPy `pypy/module/_io/interp_bufferedio.py`.

use pyre_object::*;

use super::DEFAULT_BUFFER_SIZE;
const STATE_ZERO: i64 = 0;
const STATE_OK: i64 = 1;
const STATE_DETACHED: i64 = 2;

pub(super) fn is_blocking_error(error: &crate::PyError) -> bool {
    let Some(blocking) = crate::builtins::lookup_exc_class("BlockingIOError") else {
        return false;
    };
    !error.exc_object.is_null()
        && unsafe { crate::baseobjspace::isinstance_w(error.exc_object, blocking) }
}

pub(super) fn make_write_blocking_error(written: usize) -> crate::PyError {
    let Some(blocking) = crate::builtins::lookup_exc_class("BlockingIOError") else {
        return crate::PyError::os_error("write could not complete without blocking");
    };
    // `interp_bufferedio.py make_write_blocking_error` reads the saved errno here.  It may be
    // nonsense when the raw `write` is a Python method that simply returned
    // None, but a caller that asked for a non-blocking fd needs EAGAIN, and
    // the only place it survives is the errno the failed syscall left.
    // Ints are old-gen; the message is a per-call nursery string.  Mint the
    // ints first so the only word that can move is already on the root stack
    // before the constructor allocates.
    let _roots = pyre_object::gc_roots::push_roots();
    let errno = w_int_new(majit_rlib::rposix::get_saved_errno() as i64);
    let written = w_int_new(written as i64);
    let msg_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(
        "write could not complete without blocking",
    ));
    match crate::call::call_function_impl_result(
        blocking,
        &[
            errno,
            pyre_object::gc_roots::shadow_stack_get(msg_slot),
            written,
        ],
    ) {
        Ok(value) => unsafe { crate::PyError::from_exc_object(value) },
        Err(error) => error,
    }
}

pub(super) fn input_bytes(obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    if unsafe { pyre_object::bytesobject::is_bytes_like(obj) } {
        return Ok(unsafe { pyre_object::bytesobject::bytes_like_data(obj) }.to_vec());
    }
    let view = crate::builtins::w_memoryview_new(obj)?;
    unsafe {
        crate::builtins::memoryview_check_released(view)?;
        Ok(crate::builtins::memoryview_gather_bytes(view))
    }
}

// CPython 3.14 Modules/_io/_iomodule.c:ADD_TYPE uses
// PyType_FromModuleAndSpec; the BufferedWriter spec is immutable.
#[crate::pyre_class("_io.BufferedWriter", cpython_heaptype, user_layout, weakrefable)]
pub struct W_BufferedWriter {
    state: i64,
    w_raw: PyObjectRef,
    buffer: PyObjectRef,
    buffer_size: i64,
    abs_pos: i64,
    pos: i64,
    raw_pos: i64,
    read_end: i64,
    write_pos: i64,
    write_end: i64,
    readable: bool,
    writable: bool,
    lock: usize,
    // interp_iobase.py W_IOBase.w_dict — null until getdict.
    pub(crate) w_dict: PyObjectRef,
}

impl Default for W_BufferedWriter {
    fn default() -> Self {
        Self {
            ob: PyObject::default(),
            state: STATE_ZERO,
            w_raw: PY_NULL,
            buffer: PY_NULL,
            buffer_size: 0,
            abs_pos: 0,
            pos: 0,
            raw_pos: 0,
            read_end: -1,
            write_pos: 0,
            write_end: -1,
            readable: false,
            writable: false,
            lock: 0,
            w_dict: PY_NULL,
            lifeline: PY_NULL,
        }
    }
}

impl W_BufferedWriter {
    fn self_obj(&self) -> PyObjectRef {
        self as *const Self as PyObjectRef
    }

    /// Re-read the receiver from `slot` after Python code ran.
    ///
    /// `writable()`, `try_new_buffer`, and `raw.write` each allocate; a
    /// collection inside one of those moves a nursery writer — leaving the
    /// `&mut self` the method was entered with behind the forwarding
    /// pointer, so later stores land in the abandoned body.
    /// RPython's GC transform keeps `self` live across the call
    /// (`interp_bufferedio.py` `W_BufferedWriter`); reload from the
    /// shadow-stack slot instead.
    fn from_slot(slot: usize) -> &'static mut Self {
        super::from_pinned_slot(slot)
    }

    /// Pin the receiver so [`Self::from_slot`] can recover it, and answer the
    /// slot it landed in.
    fn pin_self(&self) -> usize {
        super::pin_io_self(self.self_obj())
    }

    fn check_init(&self) -> Result<(), crate::PyError> {
        match self.state {
            STATE_ZERO => Err(crate::PyError::value_error(
                "I/O operation on uninitialized object",
            )),
            STATE_DETACHED => Err(crate::PyError::value_error("raw stream has been detached")),
            _ => Ok(()),
        }
    }

    fn raw_closed(&self) -> Result<bool, crate::PyError> {
        self.check_init()?;
        let closed = crate::baseobjspace::getattr_str(self.w_raw, "closed")?;
        crate::baseobjspace::is_true(closed)
    }

    fn check_closed(&self, message: &str) -> Result<(), crate::PyError> {
        if self.raw_closed()? {
            Err(crate::PyError::value_error(message))
        } else {
            Ok(())
        }
    }

    fn with_lock<T>(
        &mut self,
        body: impl FnOnce(&mut Self) -> Result<T, crate::PyError>,
    ) -> Result<T, crate::PyError> {
        // See `Buffered::with_lock`: bind the handle once so a re-`__init__`
        // inside `body` cannot redirect the release.
        let lock = self.lock;
        if !super::acquire_buffered_lock(lock) {
            return Err(crate::PyError::runtime_error("reentrant call"));
        }
        let _guard = super::BufferedLockGuard(lock);
        body(self)
    }

    fn writer_reset_buf(&mut self) {
        self.write_pos = 0;
        self.write_end = -1;
    }

    fn raw_offset(&self) -> i64 {
        if self.raw_pos >= 0
            && ((self.readable && self.read_end != -1) || (self.writable && self.write_end != -1))
        {
            self.raw_pos - self.pos
        } else {
            0
        }
    }

    fn raw_tell(&mut self) -> Result<i64, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let raw_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_raw);
        let result = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(raw_slot),
            "tell",
            &[],
        )?;
        let pos = crate::baseobjspace::int_w(result)?;
        if pos < 0 {
            return Err(crate::PyError::os_error(
                "raw stream returned invalid position",
            ));
        }
        Self::from_slot(slot).abs_pos = pos;
        Ok(pos)
    }

    fn raw_seek(&mut self, pos: i64, whence: i64) -> Result<i64, crate::PyError> {
        let _seek_roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let pos_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_int_new(pos));
        let whence_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_int_new(whence));
        let raw_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_raw);
        let result = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(raw_slot),
            "seek",
            &[
                pyre_object::gc_roots::shadow_stack_get(pos_slot),
                pyre_object::gc_roots::shadow_stack_get(whence_slot),
            ],
        )?;
        let pos = crate::baseobjspace::int_w(result)?;
        if pos < 0 {
            return Err(crate::PyError::os_error(
                "Raw stream returned invalid position",
            ));
        }
        Self::from_slot(slot).abs_pos = pos;
        Ok(pos)
    }

    fn raw_write(&mut self, data: &[u8]) -> Result<usize, crate::PyError> {
        // `w_bytes_from_bytes` and the raw `write` each allocate. Pin
        // before the first of those (`interp_bufferedio.py` `_write`) so
        // `abs_pos` is stored on the live object.
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let bytes = pyre_object::bytesobject::w_bytes_from_bytes(data);
        let bytes_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(bytes);
        let raw_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_raw);
        let result = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(raw_slot),
            "write",
            &[pyre_object::gc_roots::shadow_stack_get(bytes_slot)],
        )?;
        if unsafe { pyre_object::is_none(result) } {
            return Err(make_write_blocking_error(0));
        }
        let written = crate::builtins::space_index_w(result)?;
        if written < 0 || written as usize > data.len() {
            return Err(crate::PyError::os_error(
                "raw write() returned invalid length",
            ));
        }
        let this = Self::from_slot(slot);
        if this.abs_pos != -1 {
            this.abs_pos += written;
        }
        Ok(written as usize)
    }

    fn writer_flush_unlocked(&mut self) -> Result<(), crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let this = Self::from_slot(slot);
        if this.write_end == -1 || this.write_pos == this.write_end {
            this.writer_reset_buf();
            return Ok(());
        }
        let rewind = this.raw_offset() + (this.pos - this.write_pos);
        if rewind != 0 {
            this.raw_seek(-rewind, 1)?;
            let this = Self::from_slot(slot);
            this.raw_pos -= rewind;
        }
        loop {
            let this = Self::from_slot(slot);
            if this.write_pos >= this.write_end {
                break;
            }
            let start = this.write_pos as usize;
            let end = this.write_end as usize;
            let data = unsafe { pyre_object::bytearrayobject::w_bytearray_data(this.buffer) }
                [start..end]
                .to_vec();
            let written = match this.raw_write(&data) {
                Ok(written) => written,
                Err(error) if is_blocking_error(&error) => {
                    return Err(make_write_blocking_error(0));
                }
                Err(error) => return Err(error),
            };
            let this = Self::from_slot(slot);
            this.write_pos += written as i64;
            this.raw_pos = this.write_pos;
            // Partial writes can return successfully when interrupted by a
            // signal (see write(2)).  Run signal handlers before blocking
            // another time, possibly indefinitely.
            super::checksignals()?;
        }
        Self::from_slot(slot).writer_reset_buf();
        Ok(())
    }

    fn flush_unlocked(&mut self) -> Result<(), crate::PyError> {
        self.writer_flush_unlocked()
    }

    fn write_bytes(&mut self, data: &[u8]) -> Result<i64, crate::PyError> {
        self.with_lock(|this| {
            let _roots = pyre_object::gc_roots::push_roots();
            let slot = this.pin_self();
            Self::from_slot(slot).check_closed("write to closed file")?;
            let this = Self::from_slot(slot);
            if !((this.readable && this.read_end != -1) || (this.writable && this.write_end != -1))
            {
                this.pos = 0;
                this.raw_pos = 0;
            }
            let available = this.buffer_size as usize - this.pos as usize;
            if data.len() <= available {
                let start = this.pos as usize;
                let buffer =
                    unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(this.buffer) };
                buffer[start..start + data.len()].copy_from_slice(data);
                if this.write_end == -1 || this.write_pos > this.pos {
                    this.write_pos = this.pos;
                }
                this.pos += data.len() as i64;
                this.write_end = this.write_end.max(this.pos);
                return Ok(data.len() as i64);
            }

            if let Err(error) = this.writer_flush_unlocked() {
                let this = Self::from_slot(slot);
                if !is_blocking_error(&error) {
                    return Err(error);
                }
                let old_start = this.write_pos as usize;
                let old_end = this.write_end as usize;
                let pending = unsafe {
                    pyre_object::bytearrayobject::w_bytearray_data(this.buffer)[old_start..old_end]
                        .to_vec()
                };
                let buffer =
                    unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(this.buffer) };
                buffer[..pending.len()].copy_from_slice(&pending);
                this.write_end -= this.write_pos;
                this.raw_pos -= this.write_pos;
                this.pos -= this.write_pos;
                this.write_pos = 0;
                let available = this.buffer_size as usize - this.write_end as usize;
                if data.len() <= available {
                    let start = this.write_end as usize;
                    buffer[start..start + data.len()].copy_from_slice(data);
                    this.write_end += data.len() as i64;
                    this.pos += data.len() as i64;
                    return Ok(data.len() as i64);
                }
                buffer[this.write_end as usize..this.write_end as usize + available]
                    .copy_from_slice(&data[..available]);
                this.write_end += available as i64;
                this.pos += available as i64;
                return Err(make_write_blocking_error(available));
            }

            let mut written = 0usize;
            loop {
                let this = Self::from_slot(slot);
                if data.len() - written <= this.buffer_size as usize {
                    break;
                }
                match this.raw_write(&data[written..]) {
                    Ok(0) => return Err(make_write_blocking_error(written)),
                    Ok(size) => written += size,
                    Err(error) if is_blocking_error(&error) => {
                        let this = Self::from_slot(slot);
                        let remaining = data.len() - written;
                        let take = remaining.min(this.buffer_size as usize);
                        let buffer = unsafe {
                            pyre_object::bytearrayobject::w_bytearray_data_mut(this.buffer)
                        };
                        buffer[..take].copy_from_slice(&data[written..written + take]);
                        this.raw_pos = 0;
                        this.pos = take as i64;
                        this.write_pos = 0;
                        this.write_end = take as i64;
                        written += take;
                        return Err(make_write_blocking_error(written));
                    }
                    Err(error) => return Err(error),
                }
                // Same partial-write rule as `writer_flush_unlocked`: run the
                // handlers before the next call can block indefinitely.
                super::checksignals()?;
            }

            let this = Self::from_slot(slot);
            let remaining = data.len() - written;
            if remaining > 0 {
                let buffer =
                    unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(this.buffer) };
                buffer[..remaining].copy_from_slice(&data[written..]);
                written += remaining;
            }
            this.write_pos = 0;
            this.write_end = remaining as i64;
            this.pos = remaining as i64;
            this.raw_pos = 0;
            Ok(written as i64)
        })
    }
}

#[crate::pyre_methods(
    base = super::buffered_iobase_type(),
    weakrefable,
    doc = "BufferedWriter(raw, buffer_size=DEFAULT_BUFFER_SIZE)"
)]
impl W_BufferedWriter {
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
        let obj = W_BufferedWriter::allocate_instance(
            W_BufferedWriter::default(),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        );
        Ok(super::tag_io_instance(
            obj,
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        ))
    }

    fn __init__(
        &mut self,
        w_raw: PyObjectRef,
        #[default(DEFAULT_BUFFER_SIZE)] buffer_size: i64,
    ) -> Result<(), crate::PyError> {
        // `writable()` and `try_new_buffer` each allocate. Pin `self` and
        // `w_raw` first (`interp_bufferedio.py` `W_BufferedWriter.descr_init`)
        // so the field stores land on the live nursery object.
        let _raw_roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let raw_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_raw);
        Self::from_slot(slot).state = STATE_ZERO;
        let writable = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(raw_slot),
            "writable",
            &[],
        )?;
        if !crate::baseobjspace::is_true(writable)? {
            return Err(super::unsupported("File or stream is not writable."));
        }
        if buffer_size <= 0 {
            return Err(crate::PyError::value_error(
                "buffer size must be strictly positive",
            ));
        }
        // `_buffered_init` reports a buffer it cannot allocate as
        // `MemoryError`, and `buffer_size` came from Python.
        let buffer = super::try_new_buffer(buffer_size as usize)?;
        let this = Self::from_slot(slot);
        this.w_raw = pyre_object::gc_roots::shadow_stack_get(raw_slot);
        this.buffer = buffer;
        this.buffer_size = buffer_size;
        this.readable = false;
        this.writable = true;
        this.pos = 0;
        this.raw_pos = 0;
        this.read_end = -1;
        this.writer_reset_buf();
        this.lock = super::allocate_buffered_lock();
        // Unknown rather than asked for, as in `buffered.rs`: `tell` always
        // asks the raw stream and `flush` only adjusts a position it already
        // knows, so construction owes no `lseek`.
        this.abs_pos = -1;
        this.state = STATE_OK;
        pyre_object::gc_hook::try_gc_write_barrier(
            this as *mut Self as pyre_object::gc_hook::GCREF,
        );
        Ok(())
    }

    fn __getstate__(&self) -> Result<PyObjectRef, crate::PyError> {
        Err(crate::PyError::type_error(format!(
            "cannot pickle '{}' object",
            crate::type_methods::arg_type_name(self.self_obj())
        )))
    }

    fn write(&mut self, w_data: PyObjectRef) -> Result<i64, crate::PyError> {
        // PyPy `BufferedMixin.write_w`: initialization is checked before
        // acquiring `self.lock` (which does not exist until `__init__`).
        // `input_bytes` allocates; pin first so `write_bytes` sees the live
        // receiver.
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let data_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_data);
        Self::from_slot(slot).check_init()?;
        let data = input_bytes(pyre_object::gc_roots::shadow_stack_get(data_slot))?;
        Self::from_slot(slot).write_bytes(&data)
    }

    fn flush(&mut self) -> Result<(), crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_closed("flush of closed file")?;
        Self::from_slot(slot).with_lock(Self::flush_unlocked)
    }

    #[getter]
    fn raw(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        Ok(self.w_raw)
    }

    #[getter]
    fn closed(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        crate::baseobjspace::getattr_str(self.w_raw, "closed")
    }

    #[getter]
    fn name(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        crate::baseobjspace::getattr_str(self.w_raw, "name")
    }

    #[getter]
    fn mode(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        crate::baseobjspace::getattr_str(self.w_raw, "mode")
    }

    fn readable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        // A writer never reports readable, even over a bidirectional raw.
        Ok(pyre_object::w_bool_from(false))
    }

    fn writable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        super::call_method_result(self.w_raw, "writable", &[])
    }

    fn seekable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        super::call_method_result(self.w_raw, "seekable", &[])
    }

    fn isatty(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        super::call_method_result(self.w_raw, "isatty", &[])
    }

    fn fileno(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        super::call_method_result(self.w_raw, "fileno", &[])
    }

    fn tell(&mut self) -> Result<i64, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_init()?;
        let pos = Self::from_slot(slot).raw_tell()?;
        Ok((pos - Self::from_slot(slot).raw_offset()).max(0))
    }

    fn seek(
        &mut self,
        mut w_pos: PyObjectRef,
        #[default(pyre_object::w_int_new(0))] mut w_whence: PyObjectRef,
    ) -> Result<i64, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        pyre_object::with_roots!(w_pos, w_whence => Self::from_slot(slot).check_closed("seek of closed file"))?;
        let pos = pyre_object::with_roots!(w_whence => crate::builtins::space_index_w(w_pos))?;
        let whence = crate::builtins::space_index_w(w_whence)?;
        if !(0..=2).contains(&whence) {
            return Err(crate::PyError::value_error(format!(
                "whence must be between 0 and 2, not {whence}"
            )));
        }
        let raw_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_raw);
        let seekable = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(raw_slot),
            "seekable",
            &[],
        )?;
        if !crate::baseobjspace::is_true(seekable)? {
            return Err(super::unsupported("File or stream is not seekable."));
        }
        Self::from_slot(slot).with_lock(|this| {
            let _lock_roots = pyre_object::gc_roots::push_roots();
            let lock_slot = this.pin_self();
            Self::from_slot(lock_slot).writer_flush_unlocked()?;
            let this = Self::from_slot(lock_slot);
            this.writer_reset_buf();
            let adjusted = if whence == 1 {
                pos - this.raw_offset()
            } else {
                pos
            };
            let result = this.raw_seek(adjusted, whence)?;
            Self::from_slot(lock_slot).raw_pos = -1;
            Ok(result)
        })
    }

    fn close(&mut self) -> Result<(), crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_init()?;
        if Self::from_slot(slot).with_lock(|this| this.raw_closed())? {
            return Ok(());
        }
        let self_obj = Self::from_slot(slot).self_obj();
        let flushed = super::call_method_result(self_obj, "flush", &[]);
        let (close_result, flush_error) = match flushed {
            Err(error) => {
                let _err_roots = pyre_object::gc_roots::push_roots();
                let mut error = error;
                let error_slot = error.pin(&_err_roots);
                let close_result = Self::from_slot(slot).with_lock(|this| {
                    super::call_method_result(this.w_raw, "close", &[]).map(|_| ())
                });
                error.reload(&_err_roots, error_slot);
                (close_result, Some(error))
            }
            Ok(_) => (
                Self::from_slot(slot).with_lock(|this| {
                    super::call_method_result(this.w_raw, "close", &[]).map(|_| ())
                }),
                None,
            ),
        };
        if let Err(close_error) = close_result {
            let close_error = if let Some(flush_error) = flush_error {
                let _err_roots = pyre_object::gc_roots::push_roots();
                let mut close_error = close_error;
                let mut flush_error = flush_error;
                let close_slot = close_error.pin(&_err_roots);
                let flush_slot = flush_error.pin(&_err_roots);
                let flush_obj = flush_error.to_exc_object();
                flush_error.reload(&_err_roots, flush_slot);
                let _ = pyre_object::gc_roots::pin_root(flush_obj);
                let flush_obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                close_error.reload(&_err_roots, close_slot);
                let close_obj = close_error.to_exc_object();
                close_error.reload(&_err_roots, close_slot);
                let _ = pyre_object::gc_roots::pin_root(close_obj);
                let close_obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                unsafe {
                    pyre_object::interp_exceptions::w_exception_set_context(
                        pyre_object::gc_roots::shadow_stack_get(close_obj_slot),
                        pyre_object::gc_roots::shadow_stack_get(flush_obj_slot),
                    )
                };
                close_error.reload(&_err_roots, close_slot);
                close_error.set_exc_object(pyre_object::gc_roots::shadow_stack_get(close_obj_slot));
                close_error
            } else {
                close_error
            };
            return Err(close_error);
        }
        Self::from_slot(slot).buffer = PY_NULL;
        match flush_error {
            Some(error) => {
                let _err_roots = pyre_object::gc_roots::push_roots();
                let mut error = error;
                let error_slot = error.pin(&_err_roots);
                pyre_object::gc_hook::try_gc_write_barrier(
                    Self::from_slot(slot) as *mut Self as pyre_object::gc_hook::GCREF
                );
                error.reload(&_err_roots, error_slot);
                Err(error)
            }
            None => {
                pyre_object::gc_hook::try_gc_write_barrier(
                    Self::from_slot(slot) as *mut Self as pyre_object::gc_hook::GCREF
                );
                Ok(())
            }
        }
    }

    fn detach(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_init()?;
        super::call_method_result(Self::from_slot(slot).self_obj(), "flush", &[])?;
        let this = Self::from_slot(slot);
        let raw = this.w_raw;
        this.w_raw = PY_NULL;
        this.buffer = PY_NULL;
        this.state = STATE_DETACHED;
        pyre_object::gc_hook::try_gc_write_barrier(
            this as *mut Self as pyre_object::gc_hook::GCREF,
        );
        Ok(raw)
    }

    fn truncate(
        &mut self,
        #[default(pyre_object::w_none())] mut w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        pyre_object::with_roots!(w_size => Self::from_slot(slot).check_closed("truncate of closed file"))?;
        Self::from_slot(slot).with_lock(|this| {
            let _lock_roots = pyre_object::gc_roots::push_roots();
            let lock_slot = this.pin_self();
            Self::from_slot(lock_slot).flush_unlocked()?;
            let this = Self::from_slot(lock_slot);
            this.abs_pos = -1;
            super::call_method_result(this.w_raw, "truncate", &[w_size])
        })
    }

    fn _dealloc_warn(&self, w_source: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        super::call_method_result(self.w_raw, "_dealloc_warn", &[w_source])
    }

    fn __repr__(&self) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let mut self_obj = Self::from_slot(slot).self_obj();
        let Some(_guard) = crate::display::ReprGuard::enter(self_obj) else {
            return Err(crate::PyError::runtime_error(
                "reentrant call inside BufferedWriter.__repr__",
            ));
        };
        let typename = crate::type_methods::arg_type_name(self_obj);
        match pyre_object::with_roots!(self_obj => crate::baseobjspace::getattr_str(self_obj, "name"))
        {
            Ok(mut name) => Ok(crate::display::wtf8_format!(
                format!("<{typename} name="),
                unsafe {
                    pyre_object::with_roots!(self_obj, name => crate::display::py_repr_wtf8(name))?
                },
                ">"
            )),
            Err(_) => Ok(rustpython_wtf8::Wtf8Buf::from_string(format!(
                "<{typename}>"
            ))),
        }
    }
}
