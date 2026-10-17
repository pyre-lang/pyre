//! In-memory binary stream — PyPy `pypy/module/_io/interp_bytesio.py`.

use pyre_object::*;

const AT_END: i64 = -1;

// CPython 3.14 Modules/_io/_iomodule.c:ADD_TYPE creates the immutable
// BytesIO heap spec.
#[crate::pyre_class("_io.BytesIO", cpython_heaptype, user_layout, weakrefable)]
pub struct W_BytesIO {
    // rpython/rlib/rStringIO.py:16-23 splits immutable strings between an
    // append-optimized builder and a mutable character list. A bytearray is
    // already mutable and appendable, so this is their single storage object.
    buffer: PyObjectRef,
    pos: i64,
    closed: bool,
    // interp_iobase.py W_IOBase.w_dict — null until getdict.
    pub(crate) w_dict: PyObjectRef,
}

impl Default for W_BytesIO {
    fn default() -> Self {
        Self {
            ob: PyObject::default(),
            buffer: PY_NULL,
            pos: AT_END,
            closed: false,
            w_dict: PY_NULL,
            lifeline: PY_NULL,
        }
    }
}

impl W_BytesIO {
    fn self_obj(&self) -> PyObjectRef {
        self as *const Self as PyObjectRef
    }

    fn check_closed(&self) -> Result<(), crate::PyError> {
        if self.closed {
            Err(crate::PyError::value_error("I/O operation on closed file."))
        } else {
            Ok(())
        }
    }

    fn check_exports(&mut self) -> Result<(), crate::PyError> {
        if self.buffer.is_null() {
            return Ok(());
        }
        // `W_BytesIO._check_exports`. `export_count` lives on the bytearray
        // rather than beside `pos`: `getbuffer` hands out a view of that
        // object, so its own exporter lock already counts the live views and
        // releases them, where `BytesIOView.releasebuffer` has to decrement a
        // counter of its own.
        unsafe { crate::builtins::bytearray_check_exports(self.buffer) }
    }

    fn getsize(&self) -> i64 {
        if self.buffer.is_null() {
            0
        } else {
            unsafe { pyre_object::bytearrayobject::w_bytearray_len(self.buffer) as i64 }
        }
    }

    fn tell_pos(&self) -> i64 {
        if self.pos == AT_END {
            self.getsize()
        } else {
            self.pos
        }
    }

    fn seek_pos(&mut self, mut position: i64, mode: i64) {
        // rpython/rlib/rStringIO.py:103-119 — preserve AT_END rather than
        // materializing the numeric end position.
        if mode == 0 {
            if position == self.getsize() {
                self.pos = AT_END;
                return;
            }
        } else if mode == 1 {
            if self.pos == AT_END {
                self.pos = self.getsize();
            }
            position += self.pos;
        } else if mode == 2 {
            if position == 0 {
                self.pos = AT_END;
                return;
            }
            position += self.getsize();
        }
        if position < 0 {
            position = 0;
        }
        self.pos = position;
    }

    fn read_bytes(&mut self, size: i64) -> Vec<u8> {
        // rpython/rlib/rStringIO.py:129-149.
        let p = self.pos;
        if p == 0 && size < 0 {
            self.pos = AT_END;
            return unsafe { pyre_object::bytearrayobject::w_bytearray_data(self.buffer).to_vec() };
        }
        if p == AT_END || size == 0 {
            return Vec::new();
        }
        let mysize = self.getsize();
        let mut count = mysize - p;
        if size >= 0 {
            count = count.min(size);
        }
        if count <= 0 {
            return Vec::new();
        }
        if p == 0 && count == mysize {
            self.pos = AT_END;
        } else {
            self.pos = p + count;
        }
        unsafe {
            pyre_object::bytearrayobject::w_bytearray_data(self.buffer)
                [p as usize..(p + count) as usize]
                .to_vec()
        }
    }

    fn readline_bytes(&mut self, size: i64) -> Vec<u8> {
        // rpython/rlib/rStringIO.py:151-176.
        let p = self.pos;
        if p == AT_END || size == 0 {
            return Vec::new();
        }
        let length = self.getsize();
        let count = length - p;
        if count <= 0 {
            return Vec::new();
        }
        let mut end = length;
        if size >= 0 && size < count {
            end = p + size;
        }
        let newline = unsafe {
            pyre_object::bytearrayobject::w_bytearray_find(self.buffer, b'\n', p as usize)
        };
        if newline >= 0 && newline < end {
            end = newline + 1;
        }
        self.pos = end;
        unsafe {
            pyre_object::bytearrayobject::w_bytearray_data(self.buffer)[p as usize..end as usize]
                .to_vec()
        }
    }

    fn write_bytes(&mut self, data: &[u8]) -> Result<i64, crate::PyError> {
        if data.is_empty() {
            return Ok(0);
        }
        if self.pos == AT_END {
            let vec = unsafe { pyre_object::bytearrayobject::w_bytearray_vec_mut(self.buffer) };
            let old_len = vec.len();
            // rStringIO.py:69-72 `__fast_write` appends to a StringBuilder,
            // whose growth is amortized across the run.  An exact reservation
            // reallocates and copies the whole buffer per write, making a
            // chunked stream quadratic in its length.
            vec.try_reserve(data.len())
                .map_err(|_| crate::PyError::memory_error(""))?;
            vec.extend_from_slice(data);
            unsafe { pyre_object::bytearrayobject::w_bytearray_sync_alloc(self.buffer, old_len) };
            return Ok(data.len() as i64);
        }

        // rpython/rlib/rStringIO.py:72-101 `__slow_write` overwrites in place,
        // extends past EOF, and fills an overseeked gap with NUL bytes.
        let p = self.pos as usize;
        let end = p
            .checked_add(data.len())
            .ok_or_else(|| crate::PyError::overflow_error("new position too large"))?;
        if end > i64::MAX as usize {
            return Err(crate::PyError::overflow_error("new position too large"));
        }
        let vec = unsafe { pyre_object::bytearrayobject::w_bytearray_vec_mut(self.buffer) };
        let old_len = vec.len();
        if end > old_len {
            // `__slow_write` extends `self.__bigbuffer` with `+=`, amortized
            // for the same reason the fast path's builder is.
            vec.try_reserve(end - old_len)
                .map_err(|_| crate::PyError::memory_error(""))?;
            if p > vec.len() {
                vec.resize(p, 0);
            }
            vec.resize(end, 0);
            unsafe { pyre_object::bytearrayobject::w_bytearray_sync_alloc(self.buffer, old_len) };
        }
        vec[p..end].copy_from_slice(data);
        self.pos = if end > old_len { AT_END } else { end as i64 };
        Ok(data.len() as i64)
    }

    fn truncate_to(&mut self, size: i64) {
        // rpython/rlib/rStringIO.py:178-200 never enlarges and always seeks
        // to the resulting end using the AT_END sentinel.
        let vec = unsafe { pyre_object::bytearrayobject::w_bytearray_vec_mut(self.buffer) };
        let old_len = vec.len();
        if size < vec.len() as i64 {
            vec.truncate(size as usize);
        }
        unsafe { pyre_object::bytearrayobject::w_bytearray_sync_alloc(self.buffer, old_len) };
        self.pos = AT_END;
    }

    /// `space.buffer_w(w_data, space.BUF_CONTIG_RO)` — the contiguous
    /// read-only bytes `descr_init` and `write_w` both copy from.
    fn contiguous_bytes(mut w_data: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
        let Some(data) = pyre_object::with_roots!(w_data =>
            crate::baseobjspace::simple_buffer_bytes(w_data)
                .map(|input| input.map(crate::baseobjspace::SimpleBufferBytes::into_bytes)))?
        else {
            return Err(crate::PyError::type_error(format!(
                "a bytes-like object is required, not '{}'",
                crate::type_methods::arg_type_name(w_data)
            )));
        };
        Ok(data)
    }

    /// Re-read the receiver from `slot` after Python code ran.
    ///
    /// `space.buffer_w` / `space.acquire_writebuf` / `space.r_longlong_w`
    /// each reach a method a Python class may define (`__buffer__`,
    /// `__index__`), and a collection inside one of those moves the stream —
    /// leaving the `&mut self` the method was entered with behind the
    /// forwarding pointer, so a `closed` set by the callback is invisible and
    /// the copy lands in the abandoned body.  Upstream has no counterpart:
    /// RPython's GC transform keeps `self` live across the call for it.
    fn from_slot(slot: usize) -> &'static mut Self {
        super::from_pinned_slot(slot)
    }

    /// Pin the receiver so [`Self::from_slot`] can recover it, and answer the
    /// slot it landed in.
    fn pin_self(&self) -> usize {
        super::pin_io_self(self.self_obj())
    }

    fn reset_buffer(&mut self) {
        self.buffer = pyre_object::bytearrayobject::w_bytearray_new(0);
        self.pos = AT_END;
        self.closed = false;
        pyre_object::gc_hook::try_gc_write_barrier(
            self as *mut Self as pyre_object::gc_hook::GCREF,
        );
    }
}

#[crate::pyre_methods(
    base = super::buffered_iobase_type(),
    weakrefable,
    doc = "read-write"
)]
impl W_BytesIO {
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
        let buffer = pyre_object::bytearrayobject::w_bytearray_new(0);
        let _ = pyre_object::gc_roots::pin_root(buffer);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let obj = W_BytesIO::allocate_instance(
            W_BytesIO {
                buffer: pyre_object::gc_roots::shadow_stack_get(slot),
                ..W_BytesIO::default()
            },
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        );
        // interp_bytesio.py needs_finalizer: only a subclass needs finalization; line
        // 70 also opts this in-memory stream out of the autoflusher.
        let cls = pyre_object::gc_roots::shadow_stack_get(cls_slot);
        let needs_finalizer = !cls.is_null() && !std::ptr::eq(cls, type_object());
        Ok(super::tag_io_instance_without_autoflusher(
            obj,
            cls,
            needs_finalizer,
        ))
    }

    fn __init__(
        &mut self,
        #[default(pyre_object::w_none())] w_initial_bytes: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        // `W_BytesIO.descr_init`. The initial bytes stay live across
        // `reset_buffer` (allocation) and `write`.
        let _roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::pin_roots(&[self.self_obj(), w_initial_bytes]);
        Self::from_slot(base).check_exports()?;
        let this = Self::from_slot(base);
        this.reset_buffer();
        let w_initial_bytes = pyre_object::gc_roots::shadow_stack_get(base + 1);
        if !unsafe { pyre_object::is_none(w_initial_bytes) } {
            this.write(w_initial_bytes)?;
            Self::from_slot(base).seek_pos(0, 0);
        }
        Ok(())
    }

    fn read(
        &mut self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        // interp_bytesio.py read_w plus interp_iobase.py `convert_size`.
        self.check_closed()?;
        let size = super::iobase_convert_size(w_size)?;
        Ok(pyre_object::bytesobject::w_bytes_from_bytes(
            &self.read_bytes(size),
        ))
    }

    fn read1(
        &mut self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        // interp_bytesio.py delegates to read_w.
        self.read(w_size)
    }

    fn readline(
        &mut self,
        #[default(pyre_object::w_none())] w_limit: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        // interp_bytesio.py:105-108 plus interp_iobase.py `convert_size`.
        self.check_closed()?;
        let limit = super::iobase_convert_size(w_limit)?;
        Ok(pyre_object::bytesobject::w_bytes_from_bytes(
            &self.readline_bytes(limit),
        ))
    }

    fn readinto(&mut self, w_buffer: PyObjectRef) -> Result<i64, crate::PyError> {
        // interp_bytesio.py readinto_w: hold the writable export through copy.
        self.check_closed()?;
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let mut output = unsafe { crate::builtins::WritableBuffer::acquire(w_buffer)? };
        let output = unsafe { output.as_mut_slice() };
        let data = Self::from_slot(slot).read_bytes(output.len() as i64);
        output[..data.len()].copy_from_slice(&data);
        Ok(data.len() as i64)
    }

    fn readinto1(&mut self, w_buffer: PyObjectRef) -> Result<i64, crate::PyError> {
        self.readinto(w_buffer)
    }

    fn write(&mut self, w_data: PyObjectRef) -> Result<i64, crate::PyError> {
        // `W_BytesIO.write_w`: check state before acquiring one contiguous
        // read-only buffer, then copy its bytes once.
        self.check_closed()?;
        let _roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::pin_roots(&[self.self_obj(), w_data]);
        Self::from_slot(base).check_exports()?;
        let w_data = pyre_object::gc_roots::shadow_stack_get(base + 1);
        let data = Self::contiguous_bytes(w_data)?;
        // A `__buffer__` written in Python may have closed or exported the
        // stream, so repeat both checks — against the receiver as it stands
        // now, which that callback may also have moved.
        let this = Self::from_slot(base);
        this.check_closed()?;
        this.check_exports()?;
        this.write_bytes(&data)
    }

    fn truncate(
        &mut self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<i64, crate::PyError> {
        // `W_BytesIO.truncate_w`. Pin across `index_int_w_preserve_negative`
        // (`__index__`).
        self.check_closed()?;
        let _roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::pin_roots(&[self.self_obj(), w_size]);
        let this = Self::from_slot(base);
        this.check_exports()?;
        let pos = this.tell_pos();
        let w_size = pyre_object::gc_roots::shadow_stack_get(base + 1);
        let size = if unsafe { pyre_object::is_none(w_size) } {
            pos
        } else {
            crate::baseobjspace::index_int_w_preserve_negative(w_size)?
        };
        let this = Self::from_slot(base);
        if size < 0 {
            return Err(crate::PyError::value_error(format!(
                "negative size value {size:?}"
            )));
        }
        this.truncate_to(size);
        if size == pos {
            this.seek_pos(0, 2);
        } else {
            this.seek_pos(pos, 0);
        }
        Ok(size)
    }

    fn getbuffer(&mut self) -> Result<PyObjectRef, crate::PyError> {
        // interp_bytesio.py getbuffer_w. The bytearray exporter owns the release
        // accounting, while the BytesIO remains the view's reported owner.
        self.check_closed()?;
        Ok(crate::builtins::w_memoryview_new_simple_with_owner(
            self.buffer,
            self.self_obj(),
        ))
    }

    fn getvalue(&self) -> Result<PyObjectRef, crate::PyError> {
        // interp_bytesio.py getvalue_w.
        self.check_closed()?;
        Ok(pyre_object::bytesobject::w_bytes_from_bytes(unsafe {
            pyre_object::bytearrayobject::w_bytearray_data(self.buffer)
        }))
    }

    fn seek(
        &mut self,
        pos: PyIndexInt,
        #[default(0)] whence: PyIndexCInt,
    ) -> Result<i64, crate::PyError> {
        // interp_bytesio.py:162-180 validation followed by RStringIO.seek.
        // `@unwrap_spec(pos=r_longlong, whence=int)` (:162): the position is a
        // long long but the whence is a C int, so one that does not fit is an
        // OverflowError from the converter rather than a value the arms below
        // ever see. Both come through `__index__` alone — 3.14 rejects a whence
        // carrying only `__int__`, and reads one carrying both from `__index__`.
        self.check_closed()?;
        match whence {
            0 if pos < 0 => {
                return Err(crate::PyError::value_error(format!(
                    "negative seek value {pos:?}"
                )));
            }
            0 => {}
            1 => {
                if pos > i64::MAX - self.tell_pos() {
                    return Err(crate::PyError::overflow_error("new position too large"));
                }
            }
            2 => {
                if pos > i64::MAX - self.getsize() {
                    return Err(crate::PyError::overflow_error("new position too large"));
                }
            }
            _ => {
                return Err(crate::PyError::value_error(format!(
                    "invalid whence ({whence:?}, should be 0, 1 or 2)"
                )));
            }
        }
        self.seek_pos(pos, whence as i64);
        Ok(self.tell_pos())
    }

    fn tell(&self) -> Result<i64, crate::PyError> {
        self.check_closed()?;
        Ok(self.tell_pos())
    }

    fn readable(&self) -> Result<bool, crate::PyError> {
        self.check_closed()?;
        Ok(true)
    }

    fn writable(&self) -> Result<bool, crate::PyError> {
        self.check_closed()?;
        Ok(true)
    }

    fn seekable(&self) -> Result<bool, crate::PyError> {
        self.check_closed()?;
        Ok(true)
    }

    // `W_BytesIO.typedef` has no `flush`; `W_IOBase.flush_w` uses `_CLOSED()`
    // (`__IOBase_closed`), so a closed BytesIO still flushes. `BytesIO_flush`
    // lives on the BytesIO type and raises `ValueError` once `.closed` is true
    // (measured on 3.14.6; extra_tests `stdlib_io_bytesio.py` asserts both the
    // type-dict key and the closed raise). `_immutable_` on BytesIOBuffer /
    // BytesIOView does not govern this.
    fn flush(&self) -> Result<(), crate::PyError> {
        self.check_closed()?;
        Ok(())
    }

    fn close(&mut self) -> Result<(), crate::PyError> {
        // Any replacement of the exported bytearray would invalidate the
        // view, so it takes the same resize lock as write/truncate/__init__.
        // `interp_bytesio.py` `close_w` omits the check and drops the
        // storage from under a live `getbuffer()` result; closing an exported
        // buffer has to raise `BufferError: Existing exports of data: object
        // cannot be re-sized`, so the check runs ahead of the store.
        if self.closed {
            return Ok(());
        }
        self.check_exports()?;
        self.buffer = pyre_object::bytearrayobject::w_bytearray_new(0);
        self.pos = AT_END;
        self.closed = true;
        pyre_object::gc_hook::try_gc_write_barrier(
            self as *mut Self as pyre_object::gc_hook::GCREF,
        );
        Ok(())
    }

    #[getter]
    fn closed(&self) -> bool {
        self.closed
    }

    fn __getstate__(&self) -> Result<PyObjectRef, crate::PyError> {
        // interp_bytesio.py getstate_w, including the instance dictionary.
        self.check_closed()?;
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(self.self_obj());
        let _ = pyre_object::gc_roots::pin_root(self.getvalue()?);
        let _ = pyre_object::gc_roots::pin_root(w_int_new(self.tell_pos()));
        let dict = crate::baseobjspace::getdict_native(pyre_object::gc_roots::shadow_stack_get(sp));
        let _ = pyre_object::gc_roots::pin_root(if dict.is_null() { w_none() } else { dict });
        Ok(w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(sp + 1),
            pyre_object::gc_roots::shadow_stack_get(sp + 2),
            pyre_object::gc_roots::shadow_stack_get(sp + 3),
        ]))
    }

    fn __setstate__(&mut self, mut w_state: PyObjectRef) -> Result<(), crate::PyError> {
        // interp_bytesio.py:212-227.
        self.check_closed()?;
        let length = pyre_object::with_roots!(w_state => crate::baseobjspace::len_w(w_state))?;
        if length != 3 {
            return Err(crate::PyError::type_error(format!(
                "{}.__setstate__ argument should be 3-tuple, got {}",
                crate::type_methods::arg_type_name(self.self_obj()),
                crate::type_methods::arg_type_name(w_state)
            )));
        }
        let state = crate::baseobjspace::unpackiterable(w_state, 3)?;
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::pin_roots(&state);
        let slot = self.pin_self();
        self.check_exports()?;
        self.truncate_to(0);
        let content = pyre_object::gc_roots::shadow_stack_get(sp);
        self.write(content)?;
        let pos = crate::baseobjspace::index_int_w_preserve_negative(
            pyre_object::gc_roots::shadow_stack_get(sp + 1),
        )?;
        if pos < 0 {
            return Err(crate::PyError::value_error(
                "position value cannot be negative",
            ));
        }
        let this = Self::from_slot(slot);
        this.seek_pos(pos, 0);
        let w_dict = pyre_object::gc_roots::shadow_stack_get(sp + 2);
        if !unsafe { pyre_object::is_none(w_dict) } {
            let own_dict = crate::baseobjspace::getdict_native(this.self_obj());
            super::call_method_result(own_dict, "update", &[w_dict])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytesio_iobase_w_dict_skips_instance_dict() {
        crate::test_hooks::install_hash_hook();
        unsafe {
            assert!(
                W_BYTESIO_GC_PTR_OFFSETS.contains(&std::mem::offset_of!(W_BytesIO, w_dict)),
                "pyre_class ptr_offsets must include w_dict"
            );
            let obj = W_BytesIO::allocate_stable(W_BytesIO::default());
            assert!(!crate::objspace::std::mapdict::has_mapdict_storage(obj));
            assert!(crate::baseobjspace::getdictvalue_native(obj, "x").is_none());
            assert!(W_BytesIO::from_obj(obj).unwrap().w_dict.is_null());

            let value = pyre_object::w_int_new(1);
            assert!(crate::baseobjspace::setdictvalue(obj, "x", value).unwrap());
            let inst = W_BytesIO::from_obj(obj).unwrap();
            assert!(!inst.w_dict.is_null());
            assert_eq!(
                pyre_object::w_dict_getitem_str(inst.w_dict, "x"),
                Some(value)
            );
            assert_eq!(
                crate::baseobjspace::getdictvalue_native(obj, "x"),
                Some(value)
            );
            let w_dict = crate::baseobjspace::getdict(obj).unwrap();
            assert_eq!(w_dict, W_BytesIO::from_obj(obj).unwrap().w_dict);
            assert_eq!(crate::baseobjspace::getdict(obj).unwrap(), w_dict);

            // baseobjspace.py W_Root.setdict — TypeError, %T is the receiver.
            // This fixture's allocate_stable carrier has no initialized
            // W_TypeObject, so %T is `object`. A live BytesIO prints
            // `_io.BytesIO` (baseobjspace.py W_Root.setdict).
            let refused = "attribute '__dict__' of object objects is not writable";
            for replacement in [pyre_object::w_dict_new(), pyre_object::w_int_new(5)] {
                let err = crate::baseobjspace::setdict(obj, replacement).unwrap_err();
                assert_eq!(err.kind, crate::PyErrorKind::TypeError);
                assert_eq!(err.message_text(), refused);
            }
            assert_eq!(W_BytesIO::from_obj(obj).unwrap().w_dict, w_dict);

            crate::baseobjspace::object_delattr(obj, "x").unwrap();
            assert!(crate::baseobjspace::getdictvalue_native(obj, "x").is_none());
            assert_eq!(W_BytesIO::from_obj(obj).unwrap().w_dict, w_dict);
        }
    }
}
