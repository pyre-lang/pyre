//! In-memory text stream — PyPy `pypy/module/_io/interp_stringio.py`.

use pyre_object::*;
use rustpython_wtf8::{CodePoint, Wtf8Buf};

// CPython 3.14 Modules/_io/_iomodule.c:ADD_TYPE creates the immutable
// StringIO heap spec.
#[crate::pyre_class("_io.StringIO", cpython_heaptype, user_layout, weakrefable)]
pub struct W_StringIO {
    // interp_stringio.py stores UnicodeIO.data as a list of r_int32.
    // `array('w')` is the existing GC object whose raw payload is a mutable
    // sequence of 32-bit code points: it preserves O(1) indexing/overwrite,
    // and keeps the dropping Vec out of this GC-allocated class header.
    buffer: PyObjectRef,
    pos: i64,
    closed: bool,
    readnl: PyObjectRef,
    writenl: PyObjectRef,
    readuniversal: bool,
    readtranslate: bool,
    w_decoder: PyObjectRef,
    // interp_iobase.py W_IOBase.w_dict — null until getdict.
    pub(crate) w_dict: PyObjectRef,
}

impl Default for W_StringIO {
    fn default() -> Self {
        Self {
            ob: PyObject::default(),
            buffer: PY_NULL,
            pos: 0,
            closed: false,
            readnl: PY_NULL,
            writenl: PY_NULL,
            readuniversal: false,
            readtranslate: false,
            w_decoder: PY_NULL,
            w_dict: PY_NULL,
            lifeline: PY_NULL,
        }
    }
}

impl W_StringIO {
    fn self_obj(&self) -> PyObjectRef {
        self as *const Self as PyObjectRef
    }

    fn from_slot(slot: usize) -> &'static mut Self {
        super::from_pinned_slot(slot)
    }

    fn pin_self(&self) -> usize {
        super::pin_io_self(self.self_obj())
    }

    fn publish_refs(&mut self) {
        pyre_object::gc_hook::try_gc_write_barrier(
            self as *mut Self as pyre_object::gc_hook::GCREF,
        );
    }

    fn check_closed(&self) -> Result<(), crate::PyError> {
        if self.closed {
            // interp_stringio.py _check_closed.
            Err(crate::PyError::value_error("I/O operation on closed file"))
        } else {
            Ok(())
        }
    }

    fn data(&self) -> &'static [u8] {
        unsafe { pyre_object::interp_array::w_array_bytes(self.buffer) }
    }

    fn data_mut(&mut self) -> &'static mut Vec<u8> {
        unsafe { pyre_object::interp_array::w_array_vec_mut(self.buffer) }
    }

    fn len(&self) -> usize {
        self.data().len() / 4
    }

    fn codepoint(&self, index: usize) -> u32 {
        let offset = index * 4;
        u32::from_ne_bytes(self.data()[offset..offset + 4].try_into().unwrap())
    }

    fn codepoints(w_obj: PyObjectRef) -> Vec<u32> {
        unsafe {
            pyre_object::w_str_get_wtf8(w_obj)
                .code_points()
                .map(CodePoint::to_u32)
                .collect()
        }
    }

    fn string_from_range(&self, start: usize, end: usize) -> PyObjectRef {
        // interp_stringio.py `UnicodeIO.getdata_slice`.
        let mut result = Wtf8Buf::new();
        for index in start..end {
            if let Some(cp) = CodePoint::from_u32(self.codepoint(index)) {
                result.push(cp);
            }
        }
        pyre_object::w_str_from_wtf8_managed(result)
    }

    fn reset_buffer_from(slot: usize, w_value: PyObjectRef) -> Result<(), crate::PyError> {
        if !unsafe { crate::baseobjspace::isinstance_str_w(w_value) } {
            return Err(crate::PyError::type_error(format!(
                "unicode argument expected, got '{}'",
                crate::type_methods::arg_type_name(w_value)
            )));
        }
        let codepoints = Self::codepoints(w_value);
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(codepoints.len().saturating_mul(4))
            .map_err(|_| crate::PyError::memory_error(""))?;
        for cp in codepoints {
            bytes.extend_from_slice(&cp.to_ne_bytes());
        }
        let buffer = pyre_object::interp_array::w_array_from_bytes(b'w', 4, bytes);
        let this = Self::from_slot(slot);
        this.buffer = buffer;
        this.publish_refs();
        Ok(())
    }

    fn init_newline(slot: usize, w_newline: PyObjectRef) -> Result<(), crate::PyError> {
        // interp_stringio.py _init_newline.
        let newline = if unsafe { pyre_object::is_none(w_newline) } {
            None
        } else if unsafe { crate::baseobjspace::isinstance_str_w(w_newline) } {
            Some(unsafe { pyre_object::w_str_get_wtf8(w_newline) })
        } else {
            return Err(crate::PyError::type_error(format!(
                "newline must be str or None, not {}",
                crate::type_methods::arg_type_name(w_newline)
            )));
        };
        if let Some(value) = newline
            && !matches!(value.as_bytes(), b"" | b"\n" | b"\r" | b"\r\n")
        {
            let shown = unsafe { crate::display::py_repr_wtf8(w_newline) }?;
            return Err(crate::PyError::value_error(crate::display::wtf8_format!(
                "illegal newline value: ",
                shown
            )));
        }

        let this = Self::from_slot(slot);
        this.readnl = w_newline;
        this.writenl = PY_NULL;
        this.readuniversal = newline.is_none_or(|value| value.as_bytes().is_empty());
        this.readtranslate = newline.is_none();
        this.w_decoder = PY_NULL;
        if newline.is_some_and(|value| value.as_bytes().starts_with(b"\r")) {
            this.writenl = w_newline;
        }
        this.publish_refs();

        if this.readuniversal {
            let io = crate::importing::get_builtin_module("_io")
                .ok_or_else(|| crate::PyError::runtime_error("_io module is not initialized"))?;
            let decoder_type = crate::baseobjspace::getattr_str(io, "IncrementalNewlineDecoder")?;
            let decoder = crate::call::call_function_impl_result(
                decoder_type,
                &[w_none(), w_bool_from(this.readtranslate)],
            )?;
            let this = Self::from_slot(slot);
            this.w_decoder = decoder;
            this.publish_refs();
        }
        Ok(())
    }

    fn decode_string(slot: usize, w_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        // interp_stringio.py _decode_string. Calls are kept at object level so the
        // app-level IncrementalNewlineDecoder owns translation and seennl.
        if !unsafe { crate::baseobjspace::isinstance_str_w(w_obj) } {
            return Err(crate::PyError::type_error(format!(
                "unicode argument expected, got '{}'",
                crate::type_methods::arg_type_name(w_obj)
            )));
        }
        let this = Self::from_slot(slot);
        this.check_closed()?;
        let mut decoded = if this.w_decoder.is_null() {
            w_obj
        } else {
            super::call_method_result(this.w_decoder, "decode", &[w_obj, w_bool_from(true)])?
        };
        let _roots = pyre_object::gc_roots::push_roots();
        let decoded_slot = pyre_object::gc_roots::shadow_stack_len();
        let mut decoded = pyre_object::gc_roots::pin_root(decoded);
        let this = Self::from_slot(slot);
        if !this.writenl.is_null() {
            let writenl_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(this.writenl);
            let nl_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_str_new_managed("\n"));
            decoded = super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(decoded_slot),
                "replace",
                &[
                    pyre_object::gc_roots::shadow_stack_get(nl_slot),
                    pyre_object::gc_roots::shadow_stack_get(writenl_slot),
                ],
            )?;
        }
        if !unsafe { crate::baseobjspace::isinstance_str_w(decoded) } {
            return Err(crate::PyError::type_error(
                "decoder should return a string result",
            ));
        }
        Ok(decoded)
    }

    fn write_codepoints(&mut self, codepoints: &[u32]) -> Result<(), crate::PyError> {
        // interp_stringio.py `UnicodeIO.write`.
        let start = usize::try_from(self.pos)
            .map_err(|_| crate::PyError::overflow_error("new position too large"))?;
        let end = start
            .checked_add(codepoints.len())
            .ok_or_else(|| crate::PyError::overflow_error("new position too large"))?;
        if end > i64::MAX as usize {
            return Err(crate::PyError::overflow_error("new position too large"));
        }
        let data = self.data_mut();
        let byte_end = end
            .checked_mul(4)
            .ok_or_else(|| crate::PyError::overflow_error("new position too large"))?;
        if byte_end > data.len() {
            // `UnicodeIO.resize` extends a list, so a run of writes pays the
            // list's amortized growth.  Reserving the exact delta instead
            // reallocates and copies the whole buffer on every write, which
            // makes feeding a stream in chunks quadratic in its length.
            data.try_reserve(byte_end - data.len())
                .map_err(|_| crate::PyError::memory_error(""))?;
            data.resize(byte_end, 0);
        }
        for (index, cp) in codepoints.iter().enumerate() {
            let offset = (start + index) * 4;
            data[offset..offset + 4].copy_from_slice(&cp.to_ne_bytes());
        }
        self.pos = end as i64;
        Ok(())
    }
}

#[crate::pyre_methods(base = super::text_iobase_type(), weakrefable, doc = "In-memory text stream")]
impl W_StringIO {
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
        let buffer = pyre_object::interp_array::w_array_new(b'w', 4);
        let _ = pyre_object::gc_roots::pin_root(buffer);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let obj = W_StringIO::allocate_instance(
            W_StringIO {
                buffer: pyre_object::gc_roots::shadow_stack_get(slot),
                ..W_StringIO::default()
            },
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        );
        // interp_stringio.py:465-467: only a subclass needs finalization;
        // W_TextIOBase's default autoflusher membership is retained.
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
        #[default(pyre_object::w_none())] w_initvalue: PyObjectRef,
        #[default(pyre_object::w_str_new("\n"))] w_newline: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        // interp_stringio.py:177-188.
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let _ = pyre_object::gc_roots::pin_root(w_initvalue);
        let initvalue_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        Self::init_newline(slot, w_newline)?;
        let decoded = if unsafe {
            pyre_object::is_none(pyre_object::gc_roots::shadow_stack_get(initvalue_slot))
        } {
            w_str_new("")
        } else {
            Self::decode_string(
                slot,
                pyre_object::gc_roots::shadow_stack_get(initvalue_slot),
            )?
        };
        let _ = pyre_object::gc_roots::pin_root(decoded);
        let decoded =
            pyre_object::gc_roots::shadow_stack_get(pyre_object::gc_roots::shadow_stack_len() - 1);
        Self::reset_buffer_from(slot, decoded)?;
        let this = Self::from_slot(slot);
        this.pos = 0;
        this.closed = false;
        Ok(())
    }

    fn write(&mut self, w_obj: PyObjectRef) -> Result<i64, crate::PyError> {
        // interp_stringio.py write_w.
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let _ = pyre_object::gc_roots::pin_root(w_obj);
        let input_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let decoded =
            Self::decode_string(slot, pyre_object::gc_roots::shadow_stack_get(input_slot))?;
        let decoded = pyre_object::gc_roots::pin_root(decoded);
        let original_size = unsafe {
            pyre_object::w_str_len(pyre_object::gc_roots::shadow_stack_get(input_slot)) as i64
        };
        let codepoints = Self::codepoints(decoded);
        let this = Self::from_slot(slot);
        this.check_closed()?;
        if codepoints.is_empty() {
            return Ok(original_size);
        }
        this.write_codepoints(&codepoints)?;
        Ok(original_size)
    }

    fn read(
        &mut self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        // interp_stringio.py read_w plus interp_iobase.py `convert_size`.
        self.check_closed()?;
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let size = super::iobase_convert_size(w_size)?;
        let this = Self::from_slot(slot);
        this.check_closed()?;
        if this.pos >= this.len() as i64 {
            return Ok(w_str_new(""));
        }
        let start = this.pos as usize;
        let available = this.len() - start;
        let count = if size >= 0 {
            available.min(size as usize)
        } else {
            available
        };
        let end = start + count;
        let result = this.string_from_range(start, end);
        Self::from_slot(slot).pos = end as i64;
        Ok(result)
    }

    fn readline(
        &mut self,
        #[default(pyre_object::w_none())] w_limit: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        // interp_stringio.py readline_w.
        self.check_closed()?;
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let limit = super::iobase_convert_size(w_limit)?;
        let this = Self::from_slot(slot);
        this.check_closed()?;
        if this.pos >= this.len() as i64 {
            return Ok(w_str_new(""));
        }
        let start = this.pos as usize;
        let available = this.len() - start;
        let count = if limit >= 0 {
            available.min(limit as usize)
        } else {
            available
        };
        let bound = start + count;
        let mut end = bound;
        if this.readuniversal {
            let mut cursor = start;
            while cursor < bound {
                let cp = this.codepoint(cursor);
                cursor += 1;
                if cp == b'\n' as u32 {
                    end = cursor;
                    break;
                }
                if cp == b'\r' as u32 {
                    if cursor < bound && this.codepoint(cursor) == b'\n' as u32 {
                        cursor += 1;
                    }
                    end = cursor;
                    break;
                }
            }
        } else {
            let marker = Self::codepoints(this.readnl);
            for cursor in start..bound {
                if cursor + marker.len() <= bound
                    && (0..marker.len()).all(|i| this.codepoint(cursor + i) == marker[i])
                {
                    end = cursor + marker.len();
                    break;
                }
            }
        }
        let result = this.string_from_range(start, end);
        Self::from_slot(slot).pos = end as i64;
        Ok(result)
    }

    fn seek(
        &mut self,
        w_pos: PyObjectRef,
        #[default(pyre_object::w_int_new(0))] w_whence: PyObjectRef,
    ) -> Result<i64, crate::PyError> {
        // W_StringIO.seek_w `@unwrap_spec(pos=int, mode=int)`. Conversion
        // stays inside the pinned region because either argument may execute
        // Python through `__index__`. Overflow of pos is OverflowError from
        // `index_int_w` before the negative-seek ValueError arm. The whence
        // is a C int, so one that does not fit is OverflowError from the
        // converter rather than a value the range check below ever sees.
        // The position conversion runs `__index__`, so whence is reloaded
        // from this pin afterwards.
        self.check_closed()?;
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let arg_base = pyre_object::gc_roots::pin_roots(&[w_pos, w_whence]);
        let pos =
            crate::baseobjspace::index_int_w(pyre_object::gc_roots::shadow_stack_get(arg_base))?;
        let whence = crate::baseobjspace::index_c_int_w(pyre_object::gc_roots::shadow_stack_get(
            arg_base + 1,
        ))?;
        let this = Self::from_slot(slot);
        this.check_closed()?;
        if !(0..=2).contains(&whence) {
            return Err(crate::PyError::value_error(format!(
                "Invalid whence ({whence}, should be 0, 1 or 2)"
            )));
        }
        if whence == 0 && pos < 0 {
            return Err(crate::PyError::value_error(format!(
                "Negative seek position {pos}"
            )));
        }
        if whence != 0 && pos != 0 {
            return Err(crate::PyError::os_error(
                "Can't do nonzero cur-relative seeks",
            ));
        }
        let new_pos = match whence {
            1 => this.pos,
            2 => this.len() as i64,
            _ => pos,
        };
        this.pos = new_pos;
        Ok(new_pos)
    }

    fn truncate(
        &mut self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<i64, crate::PyError> {
        // interp_stringio.py truncate_w plus interp_iobase.py `convert_size`.
        self.check_closed()?;
        let current = self.pos;
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let size = if unsafe { pyre_object::is_none(w_size) } {
            current
        } else {
            super::iobase_convert_size(w_size)?
        };
        let this = Self::from_slot(slot);
        this.check_closed()?;
        if size < 0 {
            return Err(crate::PyError::value_error(format!(
                "Negative size value {size}"
            )));
        }
        if size < this.len() as i64 {
            this.data_mut().truncate(size as usize * 4);
        }
        Ok(size)
    }

    fn getvalue(&self) -> Result<PyObjectRef, crate::PyError> {
        // interp_stringio.py getvalue_w.
        self.check_closed()?;
        Ok(self.string_from_range(0, self.len()))
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

    fn close(&mut self) {
        // interp_stringio.py close_w.
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let buffer = pyre_object::interp_array::w_array_new(b'w', 4);
        let this = Self::from_slot(slot);
        this.buffer = buffer;
        this.closed = true;
        this.publish_refs();
    }

    #[getter]
    fn closed(&self) -> bool {
        self.closed
    }

    #[getter]
    fn line_buffering(&self) -> Result<bool, crate::PyError> {
        // `_io_StringIO_line_buffering_get_impl` opens at `CHECK_CLOSED`, so
        // the constant is only the answer while the stream is open.
        self.check_closed()?;
        Ok(false)
    }

    #[getter]
    fn newlines(&self) -> Result<PyObjectRef, crate::PyError> {
        // interp_stringio.py:477-480, with the `CHECK_CLOSED` that
        // `_io_StringIO_newlines_get_impl` states ahead of the decoder test.
        self.check_closed()?;
        if self.w_decoder.is_null() {
            Ok(w_none())
        } else {
            crate::baseobjspace::getattr_str(self.w_decoder, "newlines")
        }
    }

    fn __getstate__(&self) -> Result<PyObjectRef, crate::PyError> {
        // interp_stringio.py descr_getstate.
        self.check_closed()?;
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(self.self_obj());
        let _ = pyre_object::gc_roots::pin_root(self.getvalue()?);
        let own_dict =
            crate::baseobjspace::getdict_native(pyre_object::gc_roots::shadow_stack_get(sp));
        let own_dict = pyre_object::gc_roots::pin_root(own_dict);
        let copied = super::call_method_result(own_dict, "copy", &[])?;
        let _ = pyre_object::gc_roots::pin_root(copied);
        let this = Self::from_slot(sp);
        let readnl = if unsafe { pyre_object::is_none(this.readnl) } {
            w_none()
        } else {
            pyre_object::w_str_from_wtf8_managed(unsafe {
                pyre_object::w_str_get_wtf8(this.readnl).to_wtf8_buf()
            })
        };
        let _ = pyre_object::gc_roots::pin_root(readnl);
        let pos = Self::from_slot(sp).pos;
        let _ = pyre_object::gc_roots::pin_root(w_int_new(pos));
        Ok(w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(sp + 1),
            pyre_object::gc_roots::shadow_stack_get(sp + 4),
            pyre_object::gc_roots::shadow_stack_get(sp + 5),
            pyre_object::gc_roots::shadow_stack_get(sp + 3),
        ]))
    }

    fn __setstate__(&mut self, w_state: PyObjectRef) -> Result<(), crate::PyError> {
        // interp_stringio.py descr_setstate, including acceptance of future state
        // tuples longer than four items.
        self.check_closed()?;
        if !unsafe { pyre_object::is_tuple(w_state) }
            || unsafe { pyre_object::w_tuple_len(w_state) } < 4
        {
            return Err(crate::PyError::type_error(format!(
                "{}.__setstate__ argument should be a 4-tuple, got {}",
                crate::type_methods::arg_type_name(self.self_obj()),
                crate::type_methods::arg_type_name(w_state)
            )));
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(w_state);
        let slot = self.pin_self();
        let state_slot = slot - 1;
        let state = pyre_object::gc_roots::shadow_stack_get(state_slot);
        let w_value = unsafe { pyre_object::w_tuple_getitem(state, 0).unwrap() };
        let w_readnl = unsafe { pyre_object::w_tuple_getitem(state, 1).unwrap() };
        let w_pos = unsafe { pyre_object::w_tuple_getitem(state, 2).unwrap() };
        let w_dict = unsafe { pyre_object::w_tuple_getitem(state, 3).unwrap() };
        pyre_object::gc_roots::pin_roots(&[w_value, w_readnl, w_pos, w_dict]);
        Self::reset_buffer_from(slot, w_value)?;
        Self::init_newline(slot, pyre_object::gc_roots::shadow_stack_get(slot + 2))?;
        let pos = crate::baseobjspace::index_int_w_preserve_negative(
            pyre_object::gc_roots::shadow_stack_get(slot + 3),
        )?;
        if pos < 0 {
            return Err(crate::PyError::value_error(
                "position value cannot be negative",
            ));
        }
        let this = Self::from_slot(slot);
        this.pos = pos;
        let w_dict = pyre_object::gc_roots::shadow_stack_get(slot + 4);
        if !unsafe { pyre_object::is_none(w_dict) } {
            let dict_type = crate::typedef::gettypeobject(&pyre_object::DICT_TYPE);
            if !unsafe { crate::baseobjspace::isinstance_w(w_dict, dict_type) } {
                return Err(crate::PyError::type_error(format!(
                    "fourth item of state should be a dict, got a {}",
                    crate::type_methods::arg_type_name(w_dict)
                )));
            }
            let own_dict = crate::baseobjspace::getdict_native(this.self_obj());
            super::call_method_result(own_dict, "update", &[w_dict])?;
        }
        Ok(())
    }
}
