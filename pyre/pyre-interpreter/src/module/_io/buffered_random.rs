//! Seekable buffered binary streams — PyPy `W_BufferedRandom`.

use pyre_object::*;

use super::DEFAULT_BUFFER_SIZE;

const STATE_ZERO: i64 = 0;
const STATE_OK: i64 = 1;
const STATE_DETACHED: i64 = 2;

// CPython 3.14 Modules/_io/_iomodule.c:ADD_TYPE creates the immutable
// BufferedRandom heap spec.
#[crate::pyre_class("_io.BufferedRandom", cpython_heaptype, user_layout, weakrefable)]
pub struct W_BufferedRandom {
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

impl Default for W_BufferedRandom {
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

impl W_BufferedRandom {
    fn self_obj(&self) -> PyObjectRef {
        self as *const Self as PyObjectRef
    }

    /// Re-read the receiver from `slot` after Python code ran.
    ///
    /// `readable()` / `writable()` / `try_new_buffer` / `raw.read` each
    /// allocate; a collection inside one of those moves a nursery random
    /// stream — leaving the `&mut self` the method was entered with behind
    /// the forwarding pointer. RPython's GC transform keeps `self` live
    /// (`interp_bufferedio.py` `W_BufferedRandom`); reload from the
    /// shadow-stack slot instead.
    fn from_slot(slot: usize) -> &'static mut Self {
        super::from_pinned_slot(slot)
    }

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
        self.check_init()?;
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

    fn reader_reset_buf(&mut self) {
        self.read_end = -1;
    }

    fn writer_reset_buf(&mut self) {
        self.write_pos = 0;
        self.write_end = -1;
    }

    fn readahead(&self) -> usize {
        if self.readable && self.read_end != -1 {
            debug_assert!(self.read_end >= self.pos);
            (self.read_end - self.pos) as usize
        } else {
            0
        }
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

    fn adjust_position(&mut self, new_pos: i64) {
        debug_assert!(new_pos >= 0);
        self.pos = new_pos;
        if self.readable && self.read_end != -1 && self.read_end < new_pos {
            self.read_end = new_pos;
        }
    }

    fn buffer_bytes(&self, start: usize, end: usize) -> Vec<u8> {
        unsafe { pyre_object::bytearrayobject::w_bytearray_data(self.buffer)[start..end].to_vec() }
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

    fn raw_read(
        &mut self,
        w_owner: PyObjectRef,
        dest: *mut u8,
        length: usize,
    ) -> Result<usize, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        // `SubBuffer(buffer, start, length)` holds the destination buffer
        // itself, so the storage `dest` names outlives the `readinto` even
        // when the call reassigns the field it was read from.  `dest` is a
        // bare address, so the owner is rooted here instead.
        let _ = pyre_object::gc_roots::pin_root(w_owner);
        let window_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(crate::builtins::w_memoryview_new_raw_window(
            dest as usize,
            length,
        ));
        let raw_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_raw);
        let outcome = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(raw_slot),
            "readinto",
            &[pyre_object::gc_roots::shadow_stack_get(window_slot)],
        );
        // The window names memory this call borrowed, so it is closed whatever
        // the outcome rather than left writable to a raw stream that kept it.
        // The outcome word is the pin result across that release.
        let outcome = match outcome {
            Ok(value) => {
                let value = pyre_object::gc_roots::pin_root(value);
                let _ = crate::builtins::memoryview_release(&[
                    pyre_object::gc_roots::shadow_stack_get(window_slot),
                ]);
                Ok(value)
            }
            Err(error) => {
                let mut error = error;
                let error_slot = error.pin(&_roots);
                let _ = crate::builtins::memoryview_release(&[
                    pyre_object::gc_roots::shadow_stack_get(window_slot),
                ]);
                error.reload(&_roots, error_slot);
                Err(error)
            }
        };
        let result = outcome?;
        if unsafe { pyre_object::is_none(result) } {
            return Err(super::interp_bufferedio::make_blocking_error());
        }
        let size = super::interp_bufferedio::raw_readinto_size(result, length)?;
        let this = Self::from_slot(slot);
        if this.abs_pos != -1 {
            this.abs_pos += size as i64;
        }
        Ok(size)
    }

    fn fill_buffer(&mut self) -> Result<usize, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_closed("I/O operation on closed file")?;
        let this = Self::from_slot(slot);
        let start = if this.read_end == -1 {
            0
        } else {
            this.read_end as usize
        };
        let length = this.buffer_size as usize - start;
        // `_bufferedreader_fill_buffer` reads into `self->buffer + start`.
        let dest = unsafe {
            pyre_object::bytearrayobject::w_bytearray_data_mut(this.buffer)[start..].as_mut_ptr()
        };
        let size = this.raw_read(this.buffer, dest, length)?;
        if size > 0 {
            let this = Self::from_slot(slot);
            this.read_end = (start + size) as i64;
            this.raw_pos = this.read_end;
        }
        Ok(size)
    }

    fn raw_write(&mut self, data: &[u8]) -> Result<usize, crate::PyError> {
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
            return Err(super::buffered_writer::make_write_blocking_error(0));
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
            let data = this.buffer_bytes(this.write_pos as usize, this.write_end as usize);
            let written = match this.raw_write(&data) {
                Ok(written) => written,
                Err(error) if super::buffered_writer::is_blocking_error(&error) => {
                    return Err(super::buffered_writer::make_write_blocking_error(0));
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

    fn flush_and_rewind_unlocked(&mut self) -> Result<(), crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).writer_flush_unlocked()?;
        if Self::from_slot(slot).readable {
            let offset = Self::from_slot(slot).raw_offset();
            let result = Self::from_slot(slot).raw_seek(-offset, 1).map(|_| ());
            Self::from_slot(slot).reader_reset_buf();
            result?;
        }
        Ok(())
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
                this.adjust_position(this.pos + data.len() as i64);
                this.write_end = this.write_end.max(this.pos);
                return Ok(data.len() as i64);
            }

            if let Err(error) = this.writer_flush_unlocked() {
                let this = Self::from_slot(slot);
                if !super::buffered_writer::is_blocking_error(&error) {
                    return Err(error);
                }
                if this.readable {
                    this.reader_reset_buf();
                }
                let pending = this.buffer_bytes(this.write_pos as usize, this.write_end as usize);
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
                return Err(super::buffered_writer::make_write_blocking_error(available));
            }

            let this = Self::from_slot(slot);
            let offset = this.raw_offset();
            if offset != 0 {
                this.raw_seek(-offset, 1)?;
                Self::from_slot(slot).raw_pos -= offset;
            }

            let mut written = 0usize;
            loop {
                let this = Self::from_slot(slot);
                if data.len() - written <= this.buffer_size as usize {
                    break;
                }
                match this.raw_write(&data[written..]) {
                    Ok(0) => {
                        return Err(super::buffered_writer::make_write_blocking_error(written));
                    }
                    Ok(size) => written += size,
                    Err(error) if super::buffered_writer::is_blocking_error(&error) => {
                        let this = Self::from_slot(slot);
                        let take = (data.len() - written).min(this.buffer_size as usize);
                        let buffer = unsafe {
                            pyre_object::bytearrayobject::w_bytearray_data_mut(this.buffer)
                        };
                        buffer[..take].copy_from_slice(&data[written..written + take]);
                        this.raw_pos = 0;
                        this.adjust_position(take as i64);
                        this.write_pos = 0;
                        this.write_end = take as i64;
                        written += take;
                        return Err(super::buffered_writer::make_write_blocking_error(written));
                    }
                    Err(error) => return Err(error),
                }
                // Same partial-write rule as `writer_flush_unlocked`: run the
                // handlers before the next call can block indefinitely.
                super::checksignals()?;
            }

            let this = Self::from_slot(slot);
            if this.readable {
                this.reader_reset_buf();
            }
            let remaining = data.len() - written;
            if remaining > 0 {
                let buffer =
                    unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(this.buffer) };
                buffer[..remaining].copy_from_slice(&data[written..]);
                written += remaining;
            }
            this.write_pos = 0;
            this.write_end = remaining as i64;
            this.adjust_position(remaining as i64);
            this.raw_pos = 0;
            Ok(written as i64)
        })
    }

    fn read_fast(&mut self, n: usize) -> Option<Vec<u8>> {
        if n <= self.readahead() {
            let start = self.pos as usize;
            let end = start + n;
            let result = self.buffer_bytes(start, end);
            self.pos = end as i64;
            Some(result)
        } else {
            None
        }
    }

    fn read_all_unlocked(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let current_size = Self::from_slot(slot).readahead();
        let mut output = Vec::new();
        if current_size > 0 {
            let this = Self::from_slot(slot);
            let start = this.pos as usize;
            output.extend_from_slice(&this.buffer_bytes(start, start + current_size));
            this.pos += current_size as i64;
        }
        if Self::from_slot(slot).writable {
            Self::from_slot(slot).flush_and_rewind_unlocked()?;
        }
        Self::from_slot(slot).reader_reset_buf();
        let raw_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_raw);
        loop {
            let data = super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(raw_slot),
                "read",
                &[],
            )?;
            if unsafe { pyre_object::is_none(data) } {
                if output.is_empty() {
                    return Ok(data);
                }
                break;
            }
            if !unsafe { crate::baseobjspace::isinstance_bytes_w(data) } {
                return Err(crate::PyError::type_error(format!(
                    "expected bytes, got {} object",
                    crate::type_methods::arg_type_name(data)
                )));
            }
            let chunk = unsafe { pyre_object::bytesobject::bytes_like_data(data) };
            if chunk.is_empty() {
                break;
            }
            output.extend_from_slice(chunk);
            let this = Self::from_slot(slot);
            if this.abs_pos != -1 {
                this.abs_pos += chunk.len() as i64;
            }
        }
        Ok(pyre_object::bytesobject::w_bytes_from_bytes(&output))
    }

    fn read_generic_unlocked(&mut self, n: usize) -> Result<Option<Vec<u8>>, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let current_size = Self::from_slot(slot).readahead();
        if n <= current_size {
            return Ok(Self::from_slot(slot).read_fast(n));
        }
        // `_bufferedreader_read_generic` sizes its result with
        // `PyBytes_FromStringAndSize(NULL, n)`: one fallible reservation of the
        // requested length, whose failure is a `MemoryError`, and every whole
        // block the raw stream still owes lands directly in it.  Reserving
        // without filling is what lets a request the machine can only promise
        // -- `read(1 << 36)` on a short file -- answer at all.
        let mut output = crate::builtins::try_vec_with_capacity::<u8>(n)?;
        if current_size > 0 {
            let this = Self::from_slot(slot);
            let start = this.pos as usize;
            output.extend_from_slice(&this.buffer_bytes(start, start + current_size));
            this.pos += current_size as i64;
        }
        if Self::from_slot(slot).writable {
            Self::from_slot(slot).flush_and_rewind_unlocked()?;
        }
        Self::from_slot(slot).reader_reset_buf();

        let mut remaining = n - output.len();
        while remaining >= Self::from_slot(slot).buffer_size as usize {
            // `MINUS_LAST_BLOCK`: every whole block still wanted bypasses the
            // stream's buffer and is read in one go, leaving the last partial
            // block to the buffer below.
            let this = Self::from_slot(slot);
            let block = this.buffer_size as usize * (remaining / this.buffer_size as usize);
            // `out + written`: the whole reservation above is the destination,
            // so the block lands in it with nothing copied afterwards.  The
            // reservation is exact and never exceeded, so the pointer stays
            // valid for every pass, and no Python object names it for the
            // raw stream to drop.
            let written = output.len();
            let dest = unsafe { output.as_mut_ptr().add(written) };
            let size = match this.raw_read(pyre_object::w_none(), dest, block) {
                Ok(size) => size,
                Err(error) if super::interp_bufferedio::is_blocking_error(&error) => {
                    return if output.is_empty() {
                        Ok(None)
                    } else {
                        Ok(Some(output))
                    };
                }
                Err(error) => return Err(error),
            };
            if size == 0 {
                return Ok(Some(output));
            }
            unsafe { output.set_len(written + size) };
            remaining -= size;
        }

        {
            let this = Self::from_slot(slot);
            this.pos = 0;
            this.raw_pos = 0;
            this.read_end = 0;
        }
        while remaining > 0 && Self::from_slot(slot).read_end < Self::from_slot(slot).buffer_size {
            let size = match Self::from_slot(slot).fill_buffer() {
                Ok(size) => size,
                Err(error) if super::interp_bufferedio::is_blocking_error(&error) => {
                    if output.is_empty() {
                        return Ok(None);
                    }
                    0
                }
                Err(error) => return Err(error),
            };
            if size == 0 {
                break;
            }
            let this = Self::from_slot(slot);
            let take = size.min(remaining);
            output
                .extend_from_slice(&this.buffer_bytes(this.pos as usize, this.pos as usize + take));
            this.pos += take as i64;
            remaining -= take;
        }
        Ok(Some(output))
    }

    fn read_size(&mut self, size: i64) -> Result<PyObjectRef, crate::PyError> {
        // `_io__Buffered_read_impl` refuses the length before it looks at the
        // stream, so a closed file still answers a bad length with the length's
        // own message.
        if size < -1 {
            // [3.14-spec] zero is a valid length there; PyPy `read_w` still
            // says "positive".
            return Err(crate::PyError::value_error(
                "read length must be non-negative or -1",
            ));
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_closed("read of closed file")?;
        if size == -1 {
            return Self::from_slot(slot).with_lock(Self::read_all_unlocked);
        }
        let size = size as usize;
        // See `Buffered::read_size`: the unlocked `_read_fast` races with a
        // `with_lock` body once another thread has dropped the GIL in
        // `raw_read`.
        Self::from_slot(slot).with_lock(|this| {
            let _lock_roots = pyre_object::gc_roots::push_roots();
            let lock_slot = this.pin_self();
            if let Some(result) = Self::from_slot(lock_slot).read_fast(size) {
                return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&result));
            }
            Ok(
                match Self::from_slot(lock_slot).read_generic_unlocked(size)? {
                    Some(data) => pyre_object::bytesobject::w_bytes_from_bytes(&data),
                    None => w_none(),
                },
            )
        })
    }

    fn read1_size(&mut self, mut size: i64) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_closed("read of closed file")?;
        if size < 0 {
            size = Self::from_slot(slot).buffer_size;
        }
        if size == 0 {
            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&[]));
        }
        Self::from_slot(slot).with_lock(|this| {
            let _lock_roots = pyre_object::gc_roots::push_roots();
            let lock_slot = this.pin_self();
            let mut have = Self::from_slot(lock_slot).readahead();
            if have == 0 {
                if Self::from_slot(lock_slot).writable {
                    Self::from_slot(lock_slot).flush_and_rewind_unlocked()?;
                }
                let this = Self::from_slot(lock_slot);
                if size > this.buffer_size {
                    this.reader_reset_buf();
                    let requested = size as usize;
                    // `_io__Buffered_read1_impl` allocates the whole request
                    // with `PyBytes_FromStringAndSize(NULL, n)` and fills it
                    // from a single raw read.  The reservation is local, so
                    // there is no owner for the raw stream to drop.
                    let mut out = crate::builtins::try_vec_with_capacity::<u8>(requested)?;
                    let read =
                        match this.raw_read(pyre_object::w_none(), out.as_mut_ptr(), requested) {
                            Ok(read) => read,
                            // `r == -2` there -- the raw stream would have blocked
                            // -- is read as zero bytes, not as an error.
                            Err(error) if super::interp_bufferedio::is_blocking_error(&error) => 0,
                            Err(error) => return Err(error),
                        };
                    unsafe { out.set_len(read) };
                    return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&out));
                }
                this.reader_reset_buf();
                this.pos = 0;
                have = match this.fill_buffer() {
                    Ok(size) => size,
                    Err(error) if super::interp_bufferedio::is_blocking_error(&error) => 0,
                    Err(error) => return Err(error),
                };
            }
            let this = Self::from_slot(lock_slot);
            let take = (size as usize).min(have);
            let start = this.pos as usize;
            let data = this.buffer_bytes(start, start + take);
            this.pos += take as i64;
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data))
        })
    }

    fn readinto_impl(
        &mut self,
        mut w_buffer: PyObjectRef,
        read_once: bool,
    ) -> Result<i64, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_init()?;
        pyre_object::with_roots!(w_buffer => Self::from_slot(slot).check_closed("readinto of closed file"))?;
        let mut target = unsafe { crate::builtins::WritableBuffer::acquire(w_buffer) }?;
        let requested = unsafe { target.as_mut_slice() }.len();
        if requested == 0 {
            return Ok(0);
        }
        Self::from_slot(slot).with_lock(|this| {
            let _lock_roots = pyre_object::gc_roots::push_roots();
            let lock_slot = this.pin_self();
            let mut this = Self::from_slot(lock_slot);
            let mut written = 0usize;
            let have = this.readahead();
            if have > 0 {
                let take = have.min(requested);
                let start = this.pos as usize;
                let data = this.buffer_bytes(start, start + take);
                (unsafe { target.as_mut_slice() })[..take].copy_from_slice(&data);
                this.pos += take as i64;
                written = take;
                if written == requested {
                    return Ok(written as i64);
                }
            }
            if this.writable {
                this.flush_and_rewind_unlocked()?;
                this = Self::from_slot(lock_slot);
            }
            this.reader_reset_buf();
            while written < requested {
                this = Self::from_slot(lock_slot);
                let remaining = requested - written;
                if remaining > this.buffer_size as usize {
                    // `_buffered_readinto_generic` reads into
                    // `buf.buf + written`, so the raw stream fills the caller's
                    // own buffer and nothing is copied afterwards.  `target`
                    // roots that buffer's owner for its whole lease.
                    let dest = (unsafe { target.as_mut_slice() })[written..].as_mut_ptr();
                    let size = match this.raw_read(pyre_object::w_none(), dest, remaining) {
                        Ok(size) => size,
                        Err(error) if super::interp_bufferedio::is_blocking_error(&error) => break,
                        Err(error) => return Err(error),
                    };
                    if size == 0 {
                        break;
                    }
                    written += size;
                    if read_once {
                        break;
                    }
                    continue;
                }
                if read_once && written > 0 {
                    break;
                }
                this.pos = 0;
                this.raw_pos = 0;
                this.read_end = 0;
                let size = match this.fill_buffer() {
                    Ok(size) => size,
                    Err(error) if super::interp_bufferedio::is_blocking_error(&error) => 0,
                    Err(error) => return Err(error),
                };
                if size == 0 {
                    break;
                }
                this = Self::from_slot(lock_slot);
                let take = size.min(remaining);
                let data = this.buffer_bytes(0, take);
                (unsafe { target.as_mut_slice() })[written..written + take].copy_from_slice(&data);
                this.pos = take as i64;
                written += take;
                if read_once {
                    break;
                }
            }
            Ok(written as i64)
        })
    }
}

#[crate::pyre_methods(
    base = super::buffered_iobase_type(),
    weakrefable,
    doc = "BufferedRandom(raw, buffer_size=DEFAULT_BUFFER_SIZE)"
)]
impl W_BufferedRandom {
    #[staticmethod]
    fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let given = args.len().saturating_sub(1);
        if given > 2 {
            return Err(crate::PyError::type_error(format!(
                "BufferedRandom() takes at most 2 arguments ({} given)",
                given
            )));
        }
        // `check_user_subclass` can collect. The slot is the live word;
        // this pin's argument is not read again.
        let _roots = pyre_object::gc_roots::push_roots();
        let cls_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(cls);
        crate::typedef::check_user_subclass(
            type_object(),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        )?;
        let obj = W_BufferedRandom::allocate_instance(
            W_BufferedRandom::default(),
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
        // `readable()` / `writable()` / `seekable()` / `try_new_buffer` each
        // allocate. Pin `self` and `w_raw` first
        // (`interp_bufferedio.py` `W_BufferedRandom.descr_init`) so the field
        // stores land on the live nursery object.
        let _raw_roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let raw_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_raw);
        Self::from_slot(slot).state = STATE_ZERO;
        let w_raw = || pyre_object::gc_roots::shadow_stack_get(raw_slot);
        let readable = super::call_method_result(w_raw(), "readable", &[])?;
        if !crate::baseobjspace::is_true(readable)? {
            return Err(super::unsupported("File or stream is not readable."));
        }
        let writable = super::call_method_result(w_raw(), "writable", &[])?;
        if !crate::baseobjspace::is_true(writable)? {
            return Err(super::unsupported("File or stream is not writable."));
        }
        let seekable = super::call_method_result(w_raw(), "seekable", &[])?;
        if !crate::baseobjspace::is_true(seekable)? {
            return Err(super::unsupported("File or stream is not seekable."));
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
        this.w_raw = w_raw();
        this.buffer = buffer;
        this.buffer_size = buffer_size;
        this.readable = true;
        this.writable = true;
        this.pos = 0;
        this.raw_pos = 0;
        this.reader_reset_buf();
        this.writer_reset_buf();
        this.lock = super::allocate_buffered_lock();
        // Unknown rather than asked for, as in `buffered.rs`: `seek` refreshes
        // it before its fast path reads it and `tell` always asks, so
        // construction owes no `lseek`.
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

    fn read(
        &mut self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let size = super::iobase_convert_size(w_size)?;
        Self::from_slot(slot).read_size(size)
    }

    fn read1(&mut self, #[default(-1)] size: i64) -> Result<PyObjectRef, crate::PyError> {
        self.read1_size(size)
    }

    fn peek(&mut self, #[default(0)] _size: i64) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_closed("peek of closed file")?;
        Self::from_slot(slot).with_lock(|this| {
            let _lock_roots = pyre_object::gc_roots::push_roots();
            let lock_slot = this.pin_self();
            if Self::from_slot(lock_slot).writable {
                Self::from_slot(lock_slot).flush_and_rewind_unlocked()?;
            }
            let mut have = Self::from_slot(lock_slot).readahead();
            if have == 0 {
                Self::from_slot(lock_slot).reader_reset_buf();
                have = match Self::from_slot(lock_slot).fill_buffer() {
                    Ok(size) => size,
                    Err(error) if super::interp_bufferedio::is_blocking_error(&error) => 0,
                    Err(error) => return Err(error),
                };
                Self::from_slot(lock_slot).pos = 0;
            }
            let this = Self::from_slot(lock_slot);
            let data = this.buffer_bytes(this.pos as usize, this.pos as usize + have);
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data))
        })
    }

    fn readline(
        &mut self,
        #[default(pyre_object::w_none())] mut w_limit: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        pyre_object::with_roots!(w_limit => Self::from_slot(slot).check_closed("readline of closed file"))?;
        let mut limit = super::iobase_convert_size(w_limit)?;
        let mut have = Self::from_slot(slot).readahead();
        if limit >= 0 {
            have = have.min(limit as usize);
        }
        let buffered = {
            let this = Self::from_slot(slot);
            let start = this.pos as usize;
            let buffered = this.buffer_bytes(start, start + have);
            if let Some(index) = buffered.iter().position(|byte| *byte == b'\n') {
                let end = index + 1;
                this.pos += end as i64;
                return Ok(pyre_object::bytesobject::w_bytes_from_bytes(
                    &buffered[..end],
                ));
            }
            if limit >= 0 && have == limit as usize {
                this.pos += have as i64;
                return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&buffered));
            }
            buffered
        };
        Self::from_slot(slot).with_lock(|this| {
            let _lock_roots = pyre_object::gc_roots::push_roots();
            let lock_slot = this.pin_self();
            let mut output = buffered;
            Self::from_slot(lock_slot).pos += have as i64;
            if limit >= 0 {
                limit -= have as i64;
            }
            if Self::from_slot(lock_slot).writable {
                Self::from_slot(lock_slot).flush_and_rewind_unlocked()?;
            }
            loop {
                Self::from_slot(lock_slot).reader_reset_buf();
                let filled = match Self::from_slot(lock_slot).fill_buffer() {
                    Ok(size) => size,
                    Err(error) if super::interp_bufferedio::is_blocking_error(&error) => 0,
                    Err(error) => return Err(error),
                };
                if filled == 0 {
                    break;
                }
                let this = Self::from_slot(lock_slot);
                let take = if limit >= 0 {
                    filled.min(limit as usize)
                } else {
                    filled
                };
                let chunk = this.buffer_bytes(0, take);
                let newline = chunk.iter().position(|byte| *byte == b'\n');
                let end = newline.map_or(take, |index| index + 1);
                output.extend_from_slice(&chunk[..end]);
                this.pos = end as i64;
                if newline.is_some() || (limit >= 0 && end == limit as usize) {
                    break;
                }
                if limit >= 0 {
                    limit -= end as i64;
                }
            }
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&output))
        })
    }

    fn readinto(&mut self, w_buffer: PyObjectRef) -> Result<i64, crate::PyError> {
        self.readinto_impl(w_buffer, false)
    }

    fn readinto1(&mut self, w_buffer: PyObjectRef) -> Result<i64, crate::PyError> {
        self.readinto_impl(w_buffer, true)
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
        let data = super::buffered_writer::input_bytes(pyre_object::gc_roots::shadow_stack_get(
            data_slot,
        ))?;
        Self::from_slot(slot).write_bytes(&data)
    }

    fn flush(&mut self) -> Result<(), crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_closed("flush of closed file")?;
        Self::from_slot(slot).with_lock(Self::flush_and_rewind_unlocked)
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
        super::call_method_result(self.w_raw, "readable", &[])
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
        if whence != 2 && Self::from_slot(slot).readable {
            if Self::from_slot(slot).abs_pos == -1 {
                let _ = Self::from_slot(slot).raw_tell()?;
            }
            let this = Self::from_slot(slot);
            let current = this.abs_pos;
            let available = this.readahead() as i64;
            if available > 0 {
                let offset = if whence == 0 {
                    pos - (current - this.raw_offset())
                } else {
                    pos
                };
                if -this.pos <= offset && offset <= available {
                    this.pos += offset;
                    return Ok((current - available + offset).max(0));
                }
            }
        }
        Self::from_slot(slot).with_lock(|this| {
            let _lock_roots = pyre_object::gc_roots::push_roots();
            let lock_slot = this.pin_self();
            if Self::from_slot(lock_slot).writable {
                Self::from_slot(lock_slot).writer_flush_unlocked()?;
                Self::from_slot(lock_slot).writer_reset_buf();
            }
            let this = Self::from_slot(lock_slot);
            let adjusted = if whence == 1 {
                pos - this.raw_offset()
            } else {
                pos
            };
            let result = this.raw_seek(adjusted, whence)?;
            let this = Self::from_slot(lock_slot);
            this.raw_pos = -1;
            if this.readable {
                this.reader_reset_buf();
            }
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
                super::maybe_unregister_rpython_finalizer_io(Self::from_slot(slot).self_obj());
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
            if Self::from_slot(lock_slot).writable {
                Self::from_slot(lock_slot).flush_and_rewind_unlocked()?;
            }
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
                "reentrant call inside BufferedRandom.__repr__",
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
