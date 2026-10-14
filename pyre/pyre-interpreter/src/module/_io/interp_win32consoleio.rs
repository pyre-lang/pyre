//! Windows console raw stream — `_io._WindowsConsoleIO` (PEP 528),
//! ported from PyPy `W_WinConsoleIO`.

use pyre_object::*;
use rustpython_host_env::{io as host_io, nt as host_nt};

const SMALLBUF: usize = 4;
const BUFMAX: usize = 32 * 1024 * 1024;

#[crate::pyre_class("_io._WindowsConsoleIO", user_layout, weakrefable)]
pub struct W_WinConsoleIO {
    // PyPy `W_WinConsoleIO.__init__`: all state belongs to the raw stream.
    // In particular the incomplete UTF-8 character is not a process-global or
    // thread-local side table; it resumes with this exact console frame.
    fd: i32,
    readable: bool,
    writable: bool,
    closefd: bool,
    // The flag a buffered layer sets before dropping its raw stream during
    // interpreter finalization, so the `ResourceWarning` it would otherwise
    // emit is suppressed.  Written by Python, never read from here.
    finalizing: bool,
    blksize: i64,
    smallbuf: [u8; SMALLBUF],
    // interp_iobase.py W_IOBase.w_dict — null until getdict.
    pub(crate) w_dict: PyObjectRef,
}

impl Default for W_WinConsoleIO {
    fn default() -> Self {
        Self {
            ob: PyObject::default(),
            fd: -1,
            readable: false,
            writable: false,
            closefd: false,
            finalizing: false,
            blksize: super::DEFAULT_BUFFER_SIZE,
            smallbuf: [0; SMALLBUF],
            w_dict: PY_NULL,
            lifeline: PY_NULL,
        }
    }
}

fn io_error(error: std::io::Error, filename: PyObjectRef) -> crate::PyError {
    match error.raw_os_error() {
        Some(code) => crate::PyError::os_error_win32_syscall2(code, filename, PY_NULL),
        None => crate::PyError::os_error(error.to_string()),
    }
}

fn read_console_error(error: host_nt::ReadConsoleError) -> crate::PyError {
    match error {
        host_nt::ReadConsoleError::Io(error) => io_error(error, PY_NULL),
        host_nt::ReadConsoleError::BufferTooSmall {
            available,
            required,
        } => crate::PyError::system_error(format!(
            "Buffer had room for {available} bytes but {required} bytes required"
        )),
    }
}

/// `_PyIO_get_console_type` / RustPython `pyio_get_console_type`.
/// This is deliberately only a probe: an invalid fd or ordinary path stays a
/// normal FileIO candidate and its constructor reports the authoritative error.
pub(crate) fn pyio_get_console_type(path_or_fd: PyObjectRef) -> char {
    if unsafe { pyre_object::is_int(path_or_fd) } {
        return crate::baseobjspace::c_int_w(path_or_fd)
            .ok()
            .map_or('\0', host_nt::console_type_from_fd);
    }
    let Ok(path) = crate::gateway::fsencode_path_w(path_or_fd) else {
        return '\0';
    };
    let os_name = crate::gateway::os_string_from_fs_bytes(&path.as_bytes);
    host_nt::console_type_from_name(&os_name.to_string_lossy())
}

impl W_WinConsoleIO {
    fn self_obj(&self) -> PyObjectRef {
        self as *const Self as PyObjectRef
    }

    fn pin_self(&self) -> usize {
        super::pin_io_self(self.self_obj())
    }

    fn from_slot(slot: usize) -> &'static mut Self {
        super::from_pinned_slot(slot)
    }

    fn check_closed(&self) -> Result<(), crate::PyError> {
        if self.fd < 0 {
            Err(crate::PyError::value_error("I/O operation on closed file"))
        } else {
            Ok(())
        }
    }

    fn close_descriptor(&mut self) -> Result<(), crate::PyError> {
        let fd = std::mem::replace(&mut self.fd, -1);
        if fd >= 0 && self.closefd {
            host_io::close_owned_fd(unsafe { rustpython_host_env::crt_fd::Owned::from_raw(fd) })
                .map_err(|error| io_error(error, PY_NULL))?;
        }
        Ok(())
    }

    fn mode_string(&self) -> &'static str {
        if self.readable { "rb" } else { "wb" }
    }

    fn console_handle(&self) -> Result<host_nt::Handle, crate::PyError> {
        self.check_closed()?;
        let handle = host_nt::handle_from_fd(self.fd);
        if host_nt::is_invalid_handle(handle) {
            // `_get_osfhandle` rejects a bad descriptor through `errno`, not
            // `GetLastError`, so this is an `EBADF` `OSError` with no
            // `winerror` slot - `_Py_get_osfhandle` raises it with
            // `PyErr_SetFromErrno`.
            return Err(crate::PyError::os_error_syscall(
                crate::builtins::crt_errno(),
                PY_NULL,
            ));
        }
        Ok(handle)
    }

    fn read_native(
        handle: host_nt::Handle,
        length: usize,
        smallbuf: &mut [u8; SMALLBUF],
    ) -> Result<Vec<u8>, crate::PyError> {
        let mut data = vec![0; length];
        let read = {
            // `ReadConsoleW` may block.  The destination and UTF-8 carry are
            // native stack/Vec storage, never pointers into movable GC objects.
            let _blocked = crate::module::thread::before_external_block();
            host_nt::read_console_into(handle, &mut data, smallbuf)
        }
        .map_err(read_console_error)?;
        data.truncate(read);
        Ok(data)
    }
}

#[crate::pyre_methods(base = super::raw_iobase_type(), weakrefable)]
impl W_WinConsoleIO {
    #[staticmethod]
    fn __new__(cls: PyObjectRef, _args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        crate::typedef::check_user_subclass(type_object(), cls)?;
        let _roots = pyre_object::gc_roots::push_roots();
        let cls_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(cls);
        let obj = W_WinConsoleIO::allocate_instance(
            W_WinConsoleIO::default(),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        );
        Ok(super::tag_io_instance(
            obj,
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        ))
    }

    fn __init__(
        &mut self,
        w_name: PyObjectRef,
        #[default(pyre_object::w_str_new("r"))] w_mode: PyObjectRef,
        #[default(pyre_object::w_bool_from(true))] w_closefd: PyObjectRef,
        #[default(pyre_object::w_none())] _w_opener: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        self.close_descriptor()?;
        self.readable = false;
        self.writable = false;
        self.closefd = false;
        self.smallbuf = [0; SMALLBUF];

        let _roots = pyre_object::gc_roots::push_roots();
        let w_name = pyre_object::gc_roots::pin_root(w_name);
        let name_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let _ = pyre_object::gc_roots::pin_root(w_mode);
        let mode_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let _ = pyre_object::gc_roots::pin_root(w_closefd);
        let closefd_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let self_slot = self.pin_self();

        if unsafe { pyre_object::is_bool(w_name) } {
            crate::warn::warn_category("bool is used as a file descriptor", "RuntimeWarning", 1)?;
        }

        let mode_obj = pyre_object::gc_roots::shadow_stack_get(mode_slot);
        if unsafe { !pyre_object::is_str(mode_obj) } {
            return Err(crate::PyError::type_error(format!(
                "_WindowsConsoleIO() argument 'mode' must be str, not {}",
                crate::type_methods::arg_type_name(mode_obj)
            )));
        }
        let mode = crate::baseobjspace::str_utf8_w(mode_obj)?.to_string();
        let mut rwa = false;
        let mut readable = false;
        let mut writable = false;
        for flag in mode.chars() {
            match flag {
                '+' | 'a' | 'b' | 'x' => {}
                'r' => {
                    if rwa {
                        return Err(crate::PyError::value_error(format!("invalid mode: {mode}")));
                    }
                    rwa = true;
                    readable = true;
                }
                'w' => {
                    if rwa {
                        return Err(crate::PyError::value_error(format!("invalid mode: {mode}")));
                    }
                    rwa = true;
                    writable = true;
                }
                _ => {
                    return Err(crate::PyError::value_error(format!("invalid mode: {mode}")));
                }
            }
        }
        if !rwa {
            return Err(crate::PyError::value_error(
                "Must have exactly one of read or write mode",
            ));
        }

        let name_obj = pyre_object::gc_roots::shadow_stack_get(name_slot);
        let (fd, closefd, mut console_type) = if unsafe { pyre_object::is_int(name_obj) } {
            let fd = crate::baseobjspace::c_int_w(name_obj)?;
            if fd < 0 {
                return Err(crate::PyError::value_error("negative file descriptor"));
            }
            // `winconsoleio.c:417-419` sets `self->closefd = 0` for every
            // `fd >= 0` whatever the argument said, so a descriptor is never
            // owned here and `close()` leaves it open.
            (fd, false, host_nt::console_type_from_fd(fd))
        } else {
            let closefd = crate::baseobjspace::is_true(pyre_object::gc_roots::shadow_stack_get(
                closefd_slot,
            ))?;
            if !closefd {
                return Err(crate::PyError::value_error(
                    "Cannot use closefd=False with file name",
                ));
            }
            // `closefd` is whatever the caller passed, so `is_true` can run a
            // `__bool__` and a moving collection with it: the name comes back
            // out of its slot rather than from the borrow taken above.
            let path = crate::gateway::fsencode_path_w(pyre_object::gc_roots::shadow_stack_get(
                name_slot,
            ))?;
            let os_name = crate::gateway::os_string_from_fs_bytes(&path.as_bytes);
            let mut kind = host_nt::console_type_from_name(&os_name.to_string_lossy());
            if kind == 'x' {
                kind = if writable { 'w' } else { 'r' };
            }
            let wide = widestring::WideCString::from_os_str(&os_name)
                .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
            let opened = {
                let _blocked = crate::module::thread::before_external_block();
                host_nt::open_console_path_fd(&wide, writable)
            }
            // Read the rooted slot after the block, not before: releasing the
            // GIL lets a moving collection run, and a `PyObjectRef` snapshotted
            // ahead of it names the old address.
            .map_err(|error| io_error(error, path.w_path()))?;
            (opened, true, kind)
        };

        if console_type == '\0' {
            console_type = host_nt::console_type(host_nt::handle_from_fd(fd));
        }
        if (writable && console_type != 'w')
            || (readable && console_type != 'r')
            || console_type == '\0'
        {
            if closefd {
                let _ = host_io::close_owned_fd(unsafe {
                    rustpython_host_env::crt_fd::Owned::from_raw(fd)
                });
            }
            // Mode parsing has already settled exactly one of the two, so a
            // target that is no console at all is reported through whichever
            // direction was asked for rather than by its own name.
            return Err(crate::PyError::value_error(if writable {
                "Cannot open console input buffer for writing"
            } else if readable {
                "Cannot open console output buffer for reading"
            } else {
                "Cannot open non-console file"
            }));
        }

        let this = Self::from_slot(self_slot);
        this.fd = fd;
        this.readable = readable;
        this.writable = writable;
        this.closefd = closefd;
        this.blksize = super::DEFAULT_BUFFER_SIZE;
        this.smallbuf = [0; SMALLBUF];
        crate::baseobjspace::setattr_str(
            this.self_obj(),
            "name",
            pyre_object::gc_roots::shadow_stack_get(name_slot),
        )?;
        // `W_IOBase.__init__` installs the per-instance closed flag.  A
        // successful second `__init__` reopens the same object, so its base
        // flush/close methods must observe the new live state as well as this
        // payload's fd-backed `closed` property.  Storing `name` allocates
        // the instance dict, so the receiver is read back out of its slot
        // rather than taken from the borrow held across that store.
        super::iobase_set_internal_closed(Self::from_slot(self_slot).self_obj(), false)?;
        Ok(())
    }

    #[getter]
    fn closed(&self) -> bool {
        self.fd < 0
    }

    #[getter]
    fn closefd(&self) -> bool {
        self.closefd
    }

    #[getter]
    fn _blksize(&self) -> i64 {
        self.blksize
    }

    #[getter]
    fn mode(&self) -> PyObjectRef {
        w_str_new_managed(self.mode_string())
    }

    fn fileno(&self) -> Result<i64, crate::PyError> {
        self.check_closed()?;
        Ok(self.fd as i64)
    }

    fn readable(&self) -> Result<bool, crate::PyError> {
        self.check_closed()?;
        Ok(self.readable)
    }

    fn writable(&self) -> Result<bool, crate::PyError> {
        self.check_closed()?;
        Ok(self.writable)
    }

    fn isatty(&self) -> Result<bool, crate::PyError> {
        self.check_closed()?;
        Ok(true)
    }

    /// The variant `open()` asks for right after construction.  A console
    /// handle is a terminal by definition, so this is `isatty` with no other
    /// state left to consult.
    fn _isatty_open_only(&self) -> Result<bool, crate::PyError> {
        self.check_closed()?;
        Ok(true)
    }

    #[getter]
    fn _finalizing(&self) -> bool {
        self.finalizing
    }

    #[setter]
    fn set_finalizing(&mut self, value: PyObjectRef) -> Result<(), crate::PyError> {
        if unsafe { !pyre_object::is_bool(value) } {
            return Err(crate::PyError::type_error(
                "attribute value type must be bool",
            ));
        }
        self.finalizing = crate::baseobjspace::is_true(value)?;
        Ok(())
    }

    fn close(&mut self) -> Result<(), crate::PyError> {
        if self.fd < 0 {
            return Ok(());
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = self.pin_self();
        let flush_error = super::iobase_close(&[self.self_obj()]).err();
        let close_error = Self::from_slot(self_slot).close_descriptor().err();
        match (flush_error, close_error) {
            (Some(mut flush), Some(mut close)) => {
                let flush_slot = pyre_object::gc_roots::shadow_stack_len();
                let flush_obj = flush.to_exc_object();
                let _ = pyre_object::gc_roots::pin_root(flush_obj);
                // The second `to_exc_object` allocates, so a moving collection
                // can run between the pin and the read: it forwards the slot,
                // never the local copy above.
                let close_obj = close.to_exc_object();
                unsafe {
                    pyre_object::interp_exceptions::w_exception_set_context(
                        close_obj,
                        pyre_object::gc_roots::shadow_stack_get(flush_slot),
                    )
                };
                close.set_exc_object(close_obj);
                Err(close)
            }
            (Some(error), None) | (None, Some(error)) => Err(error),
            (None, None) => Ok(()),
        }
    }

    fn readinto(&mut self, w_buffer: PyObjectRef) -> Result<i64, crate::PyError> {
        self.check_closed()?;
        if !self.readable {
            return Err(super::unsupported(
                "Console buffer does not support reading",
            ));
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = self.pin_self();
        let mut output = unsafe { crate::builtins::WritableBuffer::acquire(w_buffer)? };
        let length = unsafe { output.as_mut_slice().len() };
        if length > BUFMAX {
            return Err(crate::PyError::value_error(format!(
                "cannot read more than {BUFMAX} bytes"
            )));
        }
        if length == 0 {
            return Ok(0);
        }
        let this = Self::from_slot(self_slot);
        let handle = this.console_handle()?;
        let mut smallbuf = this.smallbuf;
        let data = Self::read_native(handle, length, &mut smallbuf)?;
        let this = Self::from_slot(self_slot);
        this.smallbuf = smallbuf;
        let target = unsafe { output.as_mut_slice() };
        target[..data.len()].copy_from_slice(&data);
        Ok(data.len() as i64)
    }

    fn read(
        &mut self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        if !self.readable {
            return Err(super::unsupported(
                "Console buffer does not support reading",
            ));
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = self.pin_self();
        let size = super::iobase_convert_size(w_size)?;
        if size < 0 {
            return Self::from_slot(self_slot).readall();
        }
        let length = usize::try_from(size)
            .map_err(|_| crate::PyError::overflow_error("Python int too large"))?;
        if length > BUFMAX {
            return Err(crate::PyError::value_error(format!(
                "cannot read more than {BUFMAX} bytes"
            )));
        }
        let this = Self::from_slot(self_slot);
        let handle = this.console_handle()?;
        let mut smallbuf = this.smallbuf;
        let data = Self::read_native(handle, length, &mut smallbuf)?;
        Self::from_slot(self_slot).smallbuf = smallbuf;
        Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data))
    }

    fn readall(&mut self) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        // No readable check here: a write handle reaches `ReadConsole` and
        // fails with the invalid-handle error the console API reports, which
        // is a different answer from the one `read` gives for the same stream.
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = self.pin_self();
        let handle = self.console_handle()?;
        let mut smallbuf = self.smallbuf;
        let data = {
            let _blocked = crate::module::thread::before_external_block();
            host_nt::read_console_all(handle, &mut smallbuf)
        }
        .map_err(|error| io_error(error, PY_NULL))?;
        Self::from_slot(self_slot).smallbuf = smallbuf;
        Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data))
    }

    fn write(&mut self, w_data: PyObjectRef) -> Result<i64, crate::PyError> {
        self.check_closed()?;
        if !self.writable {
            return Err(super::unsupported(
                "Console buffer does not support writing",
            ));
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = self.pin_self();
        let Some(input) = crate::baseobjspace::simple_buffer_bytes(w_data)? else {
            return Err(crate::PyError::type_error(format!(
                "a bytes-like object is required, not '{}'",
                crate::type_methods::arg_type_name(w_data)
            )));
        };
        let data = input.as_bytes().to_vec();
        input.release();
        if data.is_empty() {
            return Ok(0);
        }
        let handle = Self::from_slot(self_slot).console_handle()?;
        let written = {
            let _blocked = crate::module::thread::before_external_block();
            host_nt::write_console_utf8(handle, &data, BUFMAX)
        }
        .map_err(|error| io_error(error, PY_NULL))?;
        Ok(written as i64)
    }

    fn __reduce__(&self) -> Result<PyObjectRef, crate::PyError> {
        Err(crate::PyError::type_error(
            "cannot pickle '_WindowsConsoleIO' object",
        ))
    }

    fn __repr__(&self) -> rustpython_wtf8::Wtf8Buf {
        let typename = crate::type_methods::arg_type_name(self.self_obj());
        if self.fd < 0 {
            rustpython_wtf8::Wtf8Buf::from_string(format!("<{typename} [closed]>"))
        } else {
            rustpython_wtf8::Wtf8Buf::from_string(format!(
                "<{typename} mode='{}' closefd={}>",
                self.mode_string(),
                if self.closefd { "True" } else { "False" }
            ))
        }
    }
}
