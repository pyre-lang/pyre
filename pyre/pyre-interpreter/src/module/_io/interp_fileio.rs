//! Raw file stream — PyPy `pypy/module/_io/interp_fileio.py` `W_FileIO`.

use pyre_object::*;

#[crate::pyre_class("_io.FileIO", cpython_heaptype, user_layout, weakrefable)]
pub struct W_FileIO {
    fd: i32,
    readable: bool,
    writable: bool,
    created: bool,
    appending: bool,
    /// `-1` until `seekable_w` probes; then `0` or `1`.
    seekable: i32,
    closefd: bool,
    w_name: PyObjectRef,
    blksize: i64,
    // interp_iobase.py W_IOBase.w_dict — null until getdict.
    pub(crate) w_dict: PyObjectRef,
}

impl Default for W_FileIO {
    fn default() -> Self {
        Self {
            ob: PyObject::default(),
            fd: -1,
            readable: false,
            writable: false,
            created: false,
            appending: false,
            seekable: -1,
            closefd: true,
            w_name: PY_NULL,
            blksize: super::DEFAULT_BUFFER_SIZE,
            w_dict: PY_NULL,
            lifeline: PY_NULL,
        }
    }
}

impl W_FileIO {
    pub(crate) fn self_obj(&self) -> PyObjectRef {
        self as *const Self as PyObjectRef
    }

    pub(crate) fn from_slot(slot: usize) -> &'static mut Self {
        super::from_pinned_slot(slot)
    }

    pub(crate) fn pin_self(&self) -> usize {
        super::pin_io_self(self.self_obj())
    }

    pub(crate) fn fd(&self) -> i32 {
        self.fd
    }

    pub(crate) fn set_fd(&mut self, fd: i32) {
        self.fd = fd;
    }

    /// `_closed`: a negative fd is the closed / not-yet-opened sentinel.
    pub(crate) fn closed(&self) -> bool {
        self.fd < 0
    }

    pub(crate) fn closefd(&self) -> bool {
        self.closefd
    }

    pub(crate) fn set_closefd(&mut self, closefd: bool) {
        self.closefd = closefd;
    }

    pub(crate) fn set_mode_flags(
        &mut self,
        readable: bool,
        writable: bool,
        created: bool,
        appending: bool,
    ) {
        self.readable = readable;
        self.writable = writable;
        self.created = created;
        self.appending = appending;
    }

    /// `W_FileIO.readable`.
    pub(crate) fn readable(&self) -> bool {
        self.readable
    }

    /// `W_FileIO.writable`.
    pub(crate) fn writable(&self) -> bool {
        self.writable
    }

    /// `W_FileIO.appending`.
    pub(crate) fn appending(&self) -> bool {
        self.appending
    }

    /// `W_FileIO._mode`.
    pub(crate) fn mode_str(&self) -> &'static str {
        if self.created {
            if self.readable { "xb+" } else { "xb" }
        } else if self.appending {
            if self.readable { "ab+" } else { "ab" }
        } else if self.readable {
            if self.writable { "rb+" } else { "rb" }
        } else {
            "wb"
        }
    }

    pub(crate) fn name(&self) -> PyObjectRef {
        self.w_name
    }

    pub(crate) fn set_name(&mut self, w_name: PyObjectRef) {
        // framework.py transform_generic_set: barrier, then the store.
        pyre_object::gc_hook::try_gc_write_barrier(
            self as *mut Self as pyre_object::gc_hook::GCREF,
        );
        self.w_name = w_name;
    }

    pub(crate) fn blksize(&self) -> i64 {
        self.blksize
    }

    pub(crate) fn set_blksize(&mut self, blksize: i64) {
        self.blksize = blksize;
    }

    pub(crate) fn seekable_flag(&self) -> i32 {
        self.seekable
    }

    pub(crate) fn set_seekable_flag(&mut self, seekable: i32) {
        self.seekable = seekable;
    }
}

#[crate::pyre_methods(
    base = super::raw_iobase_type(),
    weakrefable,
    doc = "Open a file."
)]
impl W_FileIO {
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
        let obj = W_FileIO::allocate_instance(
            W_FileIO::default(),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        );
        let obj = super::tag_io_instance(obj, pyre_object::gc_roots::shadow_stack_get(cls_slot));
        Ok(obj)
    }
}
