//! mmap class + module-level helpers — PyPy: pypy/module/mmap/interp_mmap.py
//!
//! Verbatim move of the inline block previously in importing.rs.  The
//! `init_mmap` entry point has been renamed to `register_module` so that
//! moduledef.rs can call it directly; `init_mmap_type` remains private.

// POSIX `mmap.mmap` calls `rmmap.c_mmap` / `c_msync` / `c_munmap_safe` /
// `c_madvise_safe` (`rpython/rlib/rmmap.py`). Windows keeps `winexternal`
// behind `host_env::mmap` (`CreateFileMappingW` / `MapViewOfFile`). Each
// object owns the mapping and the descriptor it duplicated — the fd on
// POSIX, the file handle on Windows — so close()/GC release exactly that
// object's native resources.

#[cfg(unix)]
use majit_rlib::rmmap;
#[cfg(windows)]
use rustpython_host_env::mmap as host_mmap;

/// POSIX mapping from `c_mmap`. `Drop` calls `c_munmap_safe` (`__del__`).
#[cfg(unix)]
struct PosixMap {
    ptr: *mut u8,
    len: usize,
}

#[cfg(unix)]
impl PosixMap {
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

#[cfg(unix)]
impl Drop for PosixMap {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        unsafe {
            rmmap::c_munmap_safe(self.ptr.cast(), self.len);
        }
        self.ptr = std::ptr::null_mut();
    }
}

/// `os.dup` result kept by `rmmap.mmap` when `trackfd` is set. `Drop` is
/// `os.close` with `releasegil=False` (`c_close`).
#[cfg(unix)]
struct OwnedFd(i32);

#[cfg(unix)]
impl OwnedFd {
    fn as_raw(&self) -> i32 {
        self.0
    }
}

#[cfg(unix)]
impl Drop for OwnedFd {
    fn drop(&mut self) {
        if self.0 >= 0 {
            unsafe {
                rmmap::c_close(self.0);
            }
            self.0 = -1;
        }
    }
}

/// The live mapping one object owns. A Windows `mmap(…, tagname=…)` goes
/// through `CreateFileMappingW`/`MapViewOfFile` (`rmmap.py`).
#[cfg(any(unix, windows))]
enum MappedObj {
    #[cfg(unix)]
    Mapped(PosixMap),
    #[cfg(windows)]
    Mapped(host_mmap::MappedFile),
    #[cfg(windows)]
    Named(host_mmap::NamedMmap),
}

#[cfg(any(unix, windows))]
impl MappedObj {
    fn as_ptr(&self) -> *const u8 {
        match self {
            #[cfg(unix)]
            Self::Mapped(m) => m.as_ptr(),
            #[cfg(windows)]
            Self::Mapped(m) => m.as_ptr(),
            #[cfg(windows)]
            Self::Named(m) => m.as_slice().as_ptr(),
        }
    }

    fn len(&self) -> usize {
        match self {
            #[cfg(unix)]
            Self::Mapped(m) => m.len(),
            #[cfg(windows)]
            Self::Mapped(m) => m.as_slice().len(),
            #[cfg(windows)]
            Self::Named(m) => m.as_slice().len(),
        }
    }

    #[cfg(windows)]
    fn flush_range(&self, offset: usize, size: usize) -> std::io::Result<()> {
        match self {
            Self::Mapped(m) => m.flush_range(offset, size),
            Self::Named(m) => m.flush_range(offset, size),
        }
    }

    #[cfg(windows)]
    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Self::Mapped(m) => m.as_mut_slice(),
            Self::Named(m) => m.as_mut_slice(),
        }
    }
}

/// `rmmap.MMap`: the mapping plus the descriptor the object duplicated for it.
///
/// `mapped` is `None` only while a resize holds no mapping: Windows'
/// `SetEndOfFile` fails with ERROR_USER_MAPPED_FILE while any view of the file
/// is open, so the old view must go before the file grows.  `rmmap.py`
/// clears `self.data` across the same window.
#[cfg(any(unix, windows))]
struct NativeMMap {
    mapped: Option<MappedObj>,
    /// `rmmap.mmap`'s `m.fd` — `os.dup` of the caller's descriptor, kept so
    /// `resize` can `ftruncate` and `size` can `fstat`. Absent for an
    /// anonymous map and when `trackfd` is false. Windows tracks `handle`.
    #[cfg(unix)]
    fd: Option<OwnedFd>,
    /// The `flags` the mapping was created with, as `mmap(2)` received them.
    /// `mode` cannot stand in for it: a `MAP_PRIVATE` mapping that is not
    /// writable resolves to the same `AccessMode::Read` as a shared one.
    /// `resize` needs the bit to reject expanding a shared anonymous mapping.
    #[cfg(unix)]
    #[allow(dead_code)]
    flags: libc::c_int,
    /// `rmmap.py:953-970` — the file handle the object duplicated at
    /// construction, or `INVALID_HANDLE` for a mapping backed by no file of
    /// its own.  Leaving it open would keep the file locked after `close()`.
    #[cfg(windows)]
    handle: host_mmap::Handle,
    /// `rmmap.py:986-987 m.tagname` — the name `CreateFileMappingW` created the
    /// mapping object under, empty for an untagged mapping.  `resize` recreates
    /// the mapping object under the same name.
    #[cfg(windows)]
    tagname: String,
    trackfd: bool,
}

#[cfg(windows)]
impl Drop for NativeMMap {
    fn drop(&mut self) {
        if !host_mmap::is_invalid_handle_value(self.handle as isize) {
            host_mmap::close_handle(self.handle);
            self.handle = host_mmap::INVALID_HANDLE;
        }
    }
}

/// PyPy `interp_mmap.py W_MMap`: cursor/access/offset/export state is stored
/// on the object and the low-level `rmmap.MMap` is owned by that same object.
/// A user subclass is `typedef.py` `_getusercls` (`MapdictStorageMixin`).
#[cfg(any(unix, windows))]
#[pyre_interpreter::pyre_class("mmap.mmap", cpython_heaptype, user_layout, weakrefable)]
#[derive(Default)]
pub struct W_MMap {
    backend: *mut NativeMMap,
    pos: i64,
    access: i64,
    /// The protection the mapping was actually created with, which `access`
    /// alone does not determine: `_ACCESS_DEFAULT` resolves against the
    /// caller's `prot` (`rmmap.mmap`). `resize` remaps with it, where
    /// `mremap` would have preserved it. Numbering matches `MMAP_ACCESS_*`
    /// (`ACCESS_READ` 1, `ACCESS_WRITE` 2, `ACCESS_COPY` 3).
    mode: i64,
    offset: i64,
    exports: i64,
}

#[cfg(any(unix, windows))]
fn mmap_this(
    obj: pyre_object::PyObjectRef,
) -> Result<&'static mut W_MMap, pyre_interpreter::PyError> {
    W_MMap::from_obj(obj)
        .ok_or_else(|| pyre_interpreter::PyError::type_error("expected mmap object"))
}

#[cfg(any(unix, windows))]
fn mmap_native(
    obj: pyre_object::PyObjectRef,
) -> Result<&'static mut NativeMMap, pyre_interpreter::PyError> {
    let this = mmap_this(obj)?;
    if this.backend.is_null() {
        return Err(pyre_interpreter::PyError::value_error(
            "mmap closed or invalid",
        ));
    }
    Ok(unsafe { &mut *this.backend })
}

#[cfg(any(unix, windows))]
fn mmap_close_native(obj: pyre_object::PyObjectRef) {
    if let Some(this) = W_MMap::from_obj(obj)
        && !this.backend.is_null()
    {
        unsafe { drop(Box::from_raw(this.backend)) };
        this.backend = std::ptr::null_mut();
        this.pos = 0;
    }
}

/// The live mapping, or `EBADF` when the object no longer holds one.
#[cfg(any(unix, windows))]
fn mmap_mapped(obj: pyre_object::PyObjectRef) -> std::io::Result<&'static MappedObj> {
    mmap_native(obj)
        .ok()
        .and_then(|native| native.mapped.as_ref())
        .ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))
}

/// `rmmap.py flush` — `c_msync(self.getptr(offset), size, MS_SYNC)`.
/// The pointer goes to `msync` exactly as computed: Linux rejects a start that
/// is not page-aligned with EINVAL, and that refusal is observable
/// (`test_flush_return_value`).  `MappedFile::flush_range` rounds the start
/// down to a page boundary first, which turns the error into a success.
#[cfg(unix)]
fn mmap_flush(
    mut obj: pyre_object::PyObjectRef,
    offset: usize,
    size: usize,
) -> std::io::Result<()> {
    let mapped = pyre_object::with_roots!(obj => mmap_mapped(obj));
    let start = unsafe { mapped?.as_ptr().add(offset) };
    // `c_msync` saves errno and releases the GIL. The pointer is the mapping,
    // not a GC object; `obj` is pinned so a move during the call is reloaded.
    let res = pyre_object::with_roots!(obj => unsafe {
        rmmap::c_msync(start as *mut libc::c_void, size, rmmap::MS_SYNC)
    });
    if res == -1 {
        return Err(std::io::Error::from_raw_os_error(
            majit_rlib::rposix::get_saved_errno(),
        ));
    }
    Ok(())
}

#[cfg(windows)]
fn mmap_flush(obj: pyre_object::PyObjectRef, offset: usize, size: usize) -> std::io::Result<()> {
    mmap_mapped(obj)?.flush_range(offset, size)
}

#[cfg(all(unix, not(target_os = "redox")))]
fn mmap_madvise(
    obj: pyre_object::PyObjectRef,
    start: usize,
    length: usize,
    advice: i32,
) -> std::io::Result<()> {
    // `c_madvise_safe` is `_nowrapper`, so errno stays live on the call.
    let ptr = unsafe { mmap_mapped(obj)?.as_ptr().add(start) };
    let res = unsafe { rmmap::c_madvise_safe(ptr as *mut libc::c_void, length, advice) };
    if res == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
fn mmap_io_err(e: std::io::Error, ctx: &str) -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::os_error_with_errno(e.raw_os_error().unwrap_or(0), ctx)
}

/// The host layer reaches the mapping through Win32, so the code an
/// `io::Error` carries here is a Win32 error: it belongs in `.winerror`, with
/// `.errno` and the OSError subclass derived from it (`rmmap.py:1010`
/// `lastSavedWindowsError`).
#[cfg(windows)]
fn mmap_io_err(e: std::io::Error, ctx: &str) -> pyre_interpreter::PyError {
    match e.raw_os_error() {
        Some(code) => pyre_interpreter::PyError::os_error_win32_syscall2(
            code,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ),
        None => pyre_interpreter::PyError::os_error(format!("{ctx}: {e}")),
    }
}

#[cfg(any(unix, windows))]
static MMAP_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// Reads (and lazily installs) the runtime-assigned `mmap` type object, not a
/// build-time constant, so the JIT residualizes the call instead of tracing
/// into it (`@dont_look_inside`, the `gc_interp::enabled` shape). The
/// Whether `obj` has the `W_MMap` layout, including a Python subclass, where
/// [`is_mmap`] tests the Python class and so declines one.
#[cfg(any(unix, windows))]
pub(crate) fn has_mmap_layout(obj: pyre_object::PyObjectRef) -> bool {
    W_MMap::from_obj(obj).is_some()
}

/// `-> PyObjectRef` return fits a single word and it cannot raise.
#[cfg(any(unix, windows))]
#[majit_macros::dont_look_inside]
pub(crate) fn mmap_type() -> pyre_object::PyObjectRef {
    MMAP_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "mmap.mmap",
            init_mmap_type,
            pyre_interpreter::typedef::w_object(),
            <W_MMap as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        // CPython 3.14 Modules/mmapmodule.c:mmap_exec uses
        // PyType_FromModuleAndSpec; mmap_object_type_spec is immutable.
        pyre_interpreter::typedef::mark_cpython_heap_type(tp, true);
        unsafe {
            pyre_object::w_type_set_typedef_buffer(tp, Some(pyre_object::TypeDefBuffer::ReadWrite));
        }
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_MMap as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        unsafe { pyre_object::w_type_set_weakrefable(tp, true) };
        // A view dropped by the collector never reaches `__release_buffer__`,
        // so the buffer layer needs a way back here to drop the count.
        unsafe {
            pyre_object::buffer::set_external_buffer_exporter(
                mmap_external_buffer_view,
                mmap_exports_incref,
                mmap_exports_decref,
                has_mmap_layout,
            );
        }
        tp
    })
}

/// interp2app wrapper for `mmap.flush`. The integer checks are spelled out
/// so this body has no loop: a trace looks through to `c_msync`.
#[cfg(any(unix, windows))]
pub fn __majit_wrap_mmap_flush(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "flush() missing self",
        ));
    }
    let n_args = args.len();
    let mut obj = args[0];
    let mut w_off = if n_args > 1 {
        args[1]
    } else {
        pyre_object::PY_NULL
    };
    let mut w_size = if n_args > 2 {
        args[2]
    } else {
        pyre_object::PY_NULL
    };
    let mapped = pyre_object::with_roots!(obj, w_off, w_size => mmap_ptr(obj));
    let (p, len) = mapped?;
    if n_args > 1 && unsafe { !pyre_object::is_int(w_off) } {
        return Err(pyre_interpreter::PyError::type_error(
            "flush: offset must be an integer",
        ));
    }
    if n_args > 2 && unsafe { !pyre_object::is_int(w_size) } {
        return Err(pyre_interpreter::PyError::type_error(
            "flush: size must be an integer",
        ));
    }
    // Read as signed so negative user input does not wrap into
    // a huge `usize` and underflow the `len - off` subtraction
    // below (Critical: previously panicked / arbitrary length).
    let off_raw = if n_args >= 2 {
        unsafe { pyre_object::w_int_get_value(w_off) }
    } else {
        0
    };
    let raw_size_raw = if n_args >= 3 {
        unsafe { pyre_object::w_int_get_value(w_size) }
    } else {
        0
    };
    if off_raw < 0 || raw_size_raw < 0 {
        return Err(pyre_interpreter::PyError::value_error(
            "flush values out of range",
        ));
    }
    let off = off_raw as usize;
    let raw_size = raw_size_raw as usize;
    if off > len {
        return Err(pyre_interpreter::PyError::value_error(
            "flush values out of range",
        ));
    }
    let n = if raw_size == 0 { len - off } else { raw_size };
    if off.checked_add(n).map(|s| s > len).unwrap_or(true) {
        return Err(pyre_interpreter::PyError::value_error(
            "flush values out of range",
        ));
    }
    let _ = p;
    mmap_flush(obj, off, n).map_err(|e| mmap_io_err(e, "msync"))?;
    Ok(pyre_object::w_none())
}

#[cfg(any(unix, windows))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_mmap_flush,
    __majit_wrap_mmap_flush
);

/// interp2app wrapper for `mmap.size`. Loop-free; the unix body calls `c_fstat`.
#[cfg(any(unix, windows))]
pub fn __majit_wrap_mmap_size(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_ptr")) == 0 {
        return Err(pyre_interpreter::PyError::value_error(
            "mmap closed or invalid",
        ));
    }
    Ok(pyre_object::w_int_new(mmap_file_size(obj)?))
}

#[cfg(any(unix, windows))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_mmap_size,
    __majit_wrap_mmap_size
);

#[cfg(any(unix, windows))]
fn mmap_get_attr_i64(obj: pyre_object::PyObjectRef, key: &str) -> i64 {
    if let Some(this) = W_MMap::from_obj(obj) {
        return match key {
            "_ptr" => (!this.backend.is_null())
                .then(|| unsafe { (&*this.backend).mapped.as_ref() })
                .flatten()
                .map_or(0, |m| m.as_ptr() as usize as i64),
            "_len" => (!this.backend.is_null())
                .then(|| unsafe { (&*this.backend).mapped.as_ref() })
                .flatten()
                .map_or(0, |m| m.len() as i64),
            "_pos" => this.pos,
            "_access" => this.access,
            "_mode" => this.mode,
            #[cfg(unix)]
            "_fd" => (!this.backend.is_null())
                .then(|| unsafe { (&*this.backend).fd.as_ref().map(|fd| fd.as_raw() as i64) })
                .flatten()
                .unwrap_or(-1),
            #[cfg(windows)]
            "_handle" => (!this.backend.is_null())
                .then(|| unsafe { (&*this.backend).handle as isize as i64 })
                .unwrap_or(host_mmap::INVALID_HANDLE as isize as i64),
            "_offset" => this.offset,
            "_exports" => this.exports,
            _ => 0,
        };
    }
    let d = pyre_interpreter::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return 0;
    }
    if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(d, key) }
        && unsafe { pyre_object::is_int(v) }
    {
        return unsafe { pyre_object::w_int_get_value(v) };
    }
    0
}

#[cfg(any(unix, windows))]
fn mmap_set_attr(obj: pyre_object::PyObjectRef, key: &str, mut v: pyre_object::PyObjectRef) {
    if let Some(this) = W_MMap::from_obj(obj) {
        let value = if unsafe { pyre_object::is_int(v) } {
            unsafe { pyre_object::w_int_get_value(v) }
        } else {
            0
        };
        match key {
            "_pos" => this.pos = value,
            "_access" => this.access = value,
            "_mode" => this.mode = value,
            "_offset" => this.offset = value,
            "_exports" => this.exports = value,
            _ => {}
        }
        return;
    }
    let d = pyre_object::with_roots!(v => pyre_interpreter::baseobjspace::getdict_native(obj));
    if d.is_null() {
        return;
    }
    unsafe {
        pyre_object::w_dict_setitem_str(d, key, v);
    }
}

#[cfg(any(unix, windows))]
fn mmap_ptr(obj: pyre_object::PyObjectRef) -> Result<(*mut u8, usize), pyre_interpreter::PyError> {
    let mapped = mmap_native(obj)?
        .mapped
        .as_ref()
        .ok_or_else(|| pyre_interpreter::PyError::value_error("mmap closed or invalid"))?;
    Ok((mapped.as_ptr() as *mut u8, mapped.len()))
}

/// The file handle the object owns, or `None` for a mapping backed by no file
/// of its own — an anonymous one, and the pagefile-backed tagged mappings.
#[cfg(windows)]
fn mmap_handle(obj: pyre_object::PyObjectRef) -> Option<host_mmap::Handle> {
    let handle = mmap_get_attr_i64(obj, "_handle") as isize;
    if handle == 0 || host_mmap::is_invalid_handle_value(handle) {
        return None;
    }
    Some(handle as host_mmap::Handle)
}

/// `rmmap.py MMap.file_size` — the backing file's current size, which
/// diverges from the mapped length after `resize()`.  An anonymous map has no
/// file to stat, and fstat on the `-1` descriptor is the OSError that reports
/// it.
#[cfg(unix)]
fn mmap_file_size(mut obj: pyre_object::PyObjectRef) -> Result<i64, pyre_interpreter::PyError> {
    let fd = pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_fd")) as libc::c_int;
    if fd < 0 {
        return Err(pyre_interpreter::PyError::os_error_with_errno(
            libc::EBADF,
            "mmap: cannot find file size for anonymous map",
        ));
    }
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    // `os.fstat` via `c_fstat`. Failure is the saved errno, not a live one:
    // the wrapper clears errno before the call and releases the GIL.
    let rc = pyre_object::with_roots!(obj => unsafe { rmmap::c_fstat(fd, &mut st) });
    if rc < 0 {
        return Err(pyre_interpreter::PyError::os_error_with_errno(
            majit_rlib::rposix::get_saved_errno(),
            "mmap.size: fstat failed",
        ));
    }
    Ok(st.st_size as i64)
}

/// `rmmap.py:511-520` — `GetFileSize` on the handle the map owns.  An
/// anonymous map has no handle and reports the mapped length instead.
#[cfg(windows)]
fn mmap_file_size(obj: pyre_object::PyObjectRef) -> Result<i64, pyre_interpreter::PyError> {
    match mmap_handle(obj) {
        Some(handle) => host_mmap::get_file_len(handle).map_err(|e| mmap_io_err(e, "GetFileSize")),
        None => Ok(mmap_get_attr_i64(obj, "_len")),
    }
}

/// Integer unwrapping may run arbitrary `__index__`; PyPy checks the mapping
/// again afterwards because that callback may have closed it (gh-103987).
#[cfg(any(unix, windows))]
fn mmap_index_w(
    mut obj: pyre_object::PyObjectRef,
    value: pyre_object::PyObjectRef,
) -> Result<i64, pyre_interpreter::PyError> {
    let w_index =
        pyre_object::with_roots!(obj => pyre_interpreter::baseobjspace::space_index(value))?;
    let index = pyre_object::with_roots!(obj => pyre_interpreter::baseobjspace::int_w(w_index))?;
    let _ = mmap_native(obj)?;
    Ok(index)
}

/// The search pattern of `find` and `rfind`, held the way the `view: Py_buffer`
/// converter holds it: the export is acquired before `start` and `end` are
/// converted and released once the search is over.
///
/// `mmap_gfind_lock_held` reads `view->buf` and `view->len` after
/// `_As_Py_ssize_t` has run both conversions, so a `bytearray` pattern that one
/// of their `__index__` hooks edits in place is what the search compares
/// against, and one such a hook resizes is refused with `BufferError`.  Reading
/// the bytes back out of the object is what makes the edit visible; the export
/// is what makes the resize impossible, so the address cannot go stale between
/// the two.
#[cfg(any(unix, windows))]
struct MmapPattern {
    /// Shadow-stack slot the pattern is pinned in.  The bytes live in storage
    /// the object owns, so the object has to outlive every read of them.
    slot: usize,
    /// Whether the acquisition took an export to pair with a release; an
    /// immutable `bytes` pattern has no count to keep.
    held: bool,
}

#[cfg(any(unix, windows))]
impl MmapPattern {
    /// The root scope the pin lands in belongs to the caller, as it does for
    /// `SocketWritableBuffer::acquire`.
    ///
    /// # Safety
    /// The caller must uphold every validity, runtime-type, aliasing, and
    /// lifetime invariant required by the object arguments for the entire call.
    unsafe fn acquire(obj: pyre_object::PyObjectRef) -> Result<Self, pyre_interpreter::PyError> {
        // `PyObject_GetBuffer(args[0], &view, PyBUF_SIMPLE)` both admits the
        // pattern and reports why it is turned down, so every exporter
        // `readbuf_w` answers is a pattern here — a `memoryview`, an
        // `array.array` and another mapping alongside `bytes` and `bytearray`.
        unsafe { pyre_interpreter::builtins::acquire_readbuf(obj) }.map_err(|err| {
            match err.kind {
                // [3.14-spec] CPython `PyObject_GetBuffer` quotes `tp_name` in
                // this gateway's refusal; PyPy `ObjSpace._getarg_error`, reached
                // by `W_MMap.find` / `rfind`, reports it without quotes (and
                // spells its None case simply `None`).
                pyre_interpreter::PyErrorKind::TypeError => {
                    pyre_interpreter::PyError::type_error(format!(
                        "a bytes-like object is required, not '{}'",
                        pyre_interpreter::error::type_name_of(obj)
                    ))
                }
                _ => err,
            }
        })?;
        let _ = pyre_object::gc_roots::pin_root(obj);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let held = unsafe {
            pyre_interpreter::builtins::buffer_export_incref(
                pyre_object::gc_roots::shadow_stack_get(slot),
            )
        };
        Ok(Self { slot, held })
    }

    /// The pattern's bytes as they stand now.  Read this after everything that
    /// can run Python: the export refuses a resize, but an in-place edit is
    /// allowed and the search has to see it.
    ///
    /// # Safety
    /// The caller must uphold every validity, runtime-type, aliasing, and
    /// lifetime invariant required by the object arguments for the entire call.
    unsafe fn data(&self) -> Result<&'static [u8], pyre_interpreter::PyError> {
        unsafe {
            pyre_interpreter::builtins::acquire_readbuf(pyre_object::gc_roots::shadow_stack_get(
                self.slot,
            ))
        }
    }
}

#[cfg(any(unix, windows))]
impl Drop for MmapPattern {
    fn drop(&mut self) {
        if self.held {
            let obj = pyre_object::gc_roots::shadow_stack_get(self.slot);
            unsafe { pyre_interpreter::builtins::buffer_export_decref(obj) };
        }
    }
}

/// `mmap_gfind_lock_held` — the search `find` and `rfind` share, `reverse`
/// apart.
///
/// [3.14-spec] CPython's generated `mmap_mmap_find` / `mmap_mmap_rfind`
/// gateways count the arguments and acquire the `view: Py_buffer` export
/// before `mmap_gfind_lock_held`, so a wrong argument count is reported ahead
/// of a wrong pattern type and both ahead of the closed-mapping check.  PyPy's
/// gateway counts first too, but `W_MMap.find` / `rfind` check the mapping
/// before `ObjSpace.bufferstr_w` acquires the pattern; its argument-count and
/// rejected-pattern wording differs as well.
///
/// [3.14-spec] `end` is converted only when `start` is not `None`, so
/// `m.find(b"Z", None, 5)` searches to the end of the mapping and never runs
/// the `__index__` of `5`.  `interp_mmap.py find` converts each argument on its
/// own and has no `None` to pass through, so it rejects an explicit one.
#[cfg(any(unix, windows))]
fn mmap_gfind(
    who: &str,
    args: &[pyre_object::PyObjectRef],
    reverse: bool,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    // `_PyArg_CheckPositional(who, nargs, 1, 3)` over the arguments the
    // gateway is given, the receiver the descriptor bound aside.
    let nargs = args.len().saturating_sub(1);
    if !(1..=3).contains(&nargs) {
        let (bound, limit) = if nargs < 1 {
            ("at least", 1)
        } else {
            ("at most", 3)
        };
        let plural = if limit == 1 { "" } else { "s" };
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{who} expected {bound} {limit} argument{plural}, got {nargs}"
        )));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    // `acquire` pins the pattern into this scope, and the conversions below
    // run `__index__`. The mapping and those arguments are reloaded from this
    // pin; the argument slice itself is not read again after it.
    let args_base = _roots.pin_roots(args);
    // Taken before the conversions below, which run `__index__`, and released
    // when this returns — the `Py_buffer` converter's export brackets the whole
    // call.  `interp_mmap.py find` snapshots `space.bufferstr_w(w_tofind)` here
    // instead, which neither sees an in-place edit made by one of those hooks
    // nor refuses a resize.
    let pattern = unsafe { MmapPattern::acquire(_roots.get(args_base + 1)) }?;
    // `mmap_gfind_lock_held` takes the `self->pos` and `self->size` defaults
    // and performs its first `CHECK_VALID` before either conversion.
    let (_, len_at_entry) = mmap_ptr(_roots.get(args_base))?;
    let cur = mmap_get_attr_i64(_roots.get(args_base), "_pos") as usize;
    // `end` is converted only when `start` is present and not `None`.
    let (start_raw, end_raw) = if nargs >= 2 {
        let start_obj = _roots.get(args_base + 2);
        if unsafe { pyre_object::is_none(start_obj) } {
            (None, None)
        } else {
            // Both upstreams finish converting the optional arguments before
            // checking the mapping again.  In particular, an end conversion
            // still runs after the start conversion closes the mapping.
            let start = pyre_interpreter::baseobjspace::int_w(
                pyre_interpreter::baseobjspace::space_index(start_obj)?,
            )?;
            let end = if nargs >= 3 {
                let end_obj = _roots.get(args_base + 3);
                if unsafe { pyre_object::is_none(end_obj) } {
                    None
                } else {
                    Some(pyre_interpreter::baseobjspace::int_w(
                        pyre_interpreter::baseobjspace::space_index(end_obj)?,
                    )?)
                }
            } else {
                None
            };
            (Some(start), end)
        }
    } else {
        (None, None)
    };
    // `rmmap.find` reads `self.data` and `self.size` at this point, after the
    // `check_valid()` that follows `getindex_w`: an `__index__` above is free
    // to `resize()` the mapping, which installs a new view at a new address.
    // Both the pointer and every bound are therefore taken from the mapping as
    // it is now.
    let (p, len) = mmap_ptr(_roots.get(args_base))?;
    let clamp = |v: i64| {
        if v < 0 {
            ((v + len as i64).max(0)) as usize
        } else {
            (v as usize).min(len)
        }
    };
    let start = start_raw.map_or_else(|| cur.min(len), clamp);
    // [3.14-spec] the clamp below the `end = self->size` default only ever
    // lowers the value, so a mapping grown by one of the conversions above
    // keeps the size it had on the way in.  `interp_mmap.py find` reads
    // `self.mmap.size` after the start conversion instead, which reports the
    // new one.
    let end = end_raw.map_or_else(|| len_at_entry.min(len), clamp);
    if start > end {
        return Ok(pyre_object::w_int_new(-1));
    }
    let needle = unsafe { pattern.data() }?;
    if needle.is_empty() {
        let empty = if reverse { end } else { start };
        return Ok(pyre_object::w_int_new(empty as i64));
    }
    let hay = unsafe { std::slice::from_raw_parts(p.add(start), end - start) };
    if needle.len() > hay.len() {
        return Ok(pyre_object::w_int_new(-1));
    }
    let last = hay.len() - needle.len();
    let matches = |i: &usize| &hay[*i..*i + needle.len()] == needle;
    let pos = if reverse {
        (0..=last).rev().find(matches)
    } else {
        (0..=last).find(matches)
    };
    Ok(pyre_object::w_int_new(
        pos.map_or(-1, |i| (start + i) as i64),
    ))
}

/// Sweep-time counterpart of PyPy `W_MMap.__del__` / `rmmap.MMap.close`.
#[cfg(any(unix, windows))]
pub unsafe fn w_mmap_dealloc(obj: pyre_object::PyObjectRef) {
    mmap_close_native(obj);
}

/// True when `obj` is an `mmap` instance.
#[cfg(any(unix, windows))]
#[allow(dead_code)] // rmmap.MMap type test; lowered by name from LLBC
pub(crate) fn is_mmap(obj: pyre_object::PyObjectRef) -> bool {
    match pyre_interpreter::typedef::r#type(obj) {
        Some(tp) => std::ptr::eq(tp.as_ptr(), mmap_type()),
        None => false,
    }
}

/// `_exports` — how many buffers are currently exported from the mapping.
/// The mapping is raw foreign memory, so unmapping it under a live view is a
/// use-after-free rather than a stale-but-owned read; `close` and `resize`
/// refuse while this is non-zero.
#[cfg(any(unix, windows))]
pub(crate) fn mmap_exports_incref(obj: pyre_object::PyObjectRef) {
    // PyPy's buffer exporters keep `_exports` as an untranslated Signed
    // field and mutate it directly (`bytearrayobject.py:68-80`).  In
    // particular releasebuffer can run from a GC finalizer, where allocating
    // a boxed Python integer would recursively enter the collector.
    if let Some(this) = W_MMap::from_obj(obj) {
        this.exports += 1;
    }
}

/// Paired with [`mmap_exports_incref`]; saturates at zero so a double release
/// cannot wrap the count and strand the mapping.
#[cfg(any(unix, windows))]
pub(crate) unsafe fn mmap_exports_decref(obj: pyre_object::PyObjectRef) {
    // Keep this allocation-free: memoryview destruction calls it while the
    // collector is already running.  This is the same raw-field decrement as
    // PyPy `W_BytearrayObject.bf_releasebuffer` / `ByteBuffer.releasebuffer`.
    if let Some(this) = W_MMap::from_obj(obj) {
        this.exports = (this.exports - 1).max(0);
    }
}

/// Reject unmapping while a view still points into the mapping.
#[cfg(any(unix, windows))]
fn mmap_check_exports(
    obj: pyre_object::PyObjectRef,
    message: &str,
) -> Result<(), pyre_interpreter::PyError> {
    if mmap_get_attr_i64(obj, "_exports") > 0 {
        return Err(pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::BufferError,
            message,
        ));
    }
    Ok(())
}

/// `W_MMap.readbuf_w` / `writebuf_w` — expose the live mapping to the
/// object-space buffer protocol.  `None` means the object is not an mmap;
/// the inner error preserves the closed-mapping failure.
#[cfg(any(unix, windows))]
fn mmap_external_buffer_view(
    obj: pyre_object::PyObjectRef,
) -> Option<Result<(usize, usize, bool), &'static str>> {
    match mmap_buffer_view(obj) {
        None => None,
        Some(Ok(view)) => Some(Ok(view)),
        Some(Err(_)) => Some(Err("mmap closed or invalid")),
    }
}

/// `W_MMap.readbuf_w` / `writebuf_w` — expose the live mapping to the
/// object-space buffer protocol.  `None` means the object does not have
/// the mmap layout (a Python subclass still does); the inner error
/// preserves the closed-mapping failure.
#[cfg(any(unix, windows))]
pub(crate) fn mmap_buffer_view(
    obj: pyre_object::PyObjectRef,
) -> Option<Result<(usize, usize, bool), pyre_interpreter::PyError>> {
    if !has_mmap_layout(obj) {
        return None;
    }
    Some(mmap_ptr(obj).map(|(ptr, len)| {
        let readonly = mmap_get_attr_i64(obj, "_access") == MMAP_ACCESS_READ;
        (ptr as usize, len, readonly)
    }))
}

#[cfg(any(unix, windows))]
fn mmap_get_attr_obj(obj: pyre_object::PyObjectRef, key: &str) -> pyre_object::PyObjectRef {
    let d = pyre_interpreter::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return pyre_object::PY_NULL;
    }
    unsafe { pyre_object::w_dict_getitem_str(d, key) }.unwrap_or(pyre_object::PY_NULL)
}

// `interp_mmap.py descr_iter` / `:256 descr_reversed` return a
// generator yielding the 1-byte slices `m[i:i+1]`.  pyre models that
// generator as a dedicated iterator object holding the source mmap, a
// cursor, and a step (`+1` forwards, `-1` for `reversed`).
#[cfg(any(unix, windows))]
static MMAP_ITER_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

#[cfg(any(unix, windows))]
fn mmap_iterator_type() -> pyre_object::PyObjectRef {
    MMAP_ITER_TYPE_OBJ.get_or_init(|| {
        let tp =
            pyre_interpreter::typedef::make_builtin_type("mmap_iterator", init_mmap_iterator_type);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp
    })
}

#[cfg(any(unix, windows))]
fn make_mmap_iterator(
    m: pyre_object::PyObjectRef,
    start: i64,
    step: i64,
) -> pyre_object::PyObjectRef {
    let mut it = pyre_object::w_instance_new(mmap_iterator_type());
    pyre_object::with_roots!(it => mmap_set_attr(it, "_m", m));
    pyre_object::with_roots!(it => mmap_set_attr(it, "_i", pyre_object::w_int_new(start)));
    pyre_object::with_roots!(it => mmap_set_attr(it, "_step", pyre_object::w_int_new(step)));
    it
}

#[cfg(any(unix, windows))]
fn init_mmap_iterator_type(ns: pyre_object::PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__iter__",
        pyre_interpreter::make_builtin_function_with_arity(
            "__iter__",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        )
    );
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "__next__",
            pyre_interpreter::make_builtin_function_with_arity(
                "__next__",
                |args| {
                    let mut it = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                    let mut m = pyre_object::with_roots!(it => mmap_get_attr_obj(it, "_m"));
                    let i = pyre_object::with_roots!(it, m => mmap_get_attr_i64(it, "_i"));
                    let step = pyre_object::with_roots!(it, m => mmap_get_attr_i64(it, "_step"));
                    let (p, len) = mmap_ptr(m)?;
                    if i < 0 || i >= len as i64 {
                        return Err(pyre_interpreter::PyError::stop_iteration());
                    }
                    let b = { *p.add(i as usize) };
                    mmap_set_attr(it, "_i", pyre_object::w_int_new(i + step));
                    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&[b]))
                },
                1,
            )
        )
    };
}

#[cfg(any(unix, windows))]
fn init_mmap_type(ns: pyre_object::PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // `interp_mmap.py __new__ = interp2app(mmap)` — the class call
    // `mmap.mmap(fileno, length, ...)` lands here.  The common builtin
    // argument binder below supplies PyPy's named/defaulted gateway shape.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__new__",
        pyre_interpreter::typedef::make_new_descr(|args| {
            if args.is_empty() {
                return Err(pyre_interpreter::PyError::type_error(
                    "mmap() requires fileno + length",
                ));
            }
            mmap_construct(args[0], &args[1..])
        })
    );

    // close() — munmap and zero the pointer.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "close",
        pyre_interpreter::make_builtin_function_with_arity(
            "close",
            |args| {
                let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let p = pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_ptr")) as usize;
                if p != 0 {
                    pyre_object::with_roots!(obj => mmap_check_exports(
                        obj,
                        "cannot close exported pointers exist"
                    ))?;
                    mmap_close_native(obj);
                }
                Ok(pyre_object::w_none())
            },
            1,
        )
    );

    // `interp_mmap.py closed = GetSetProperty(W_MMap.closed_get)` —
    // bare attribute access (`m.closed`) returns the bool directly via
    // descriptor lookup, not a bound method.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "closed",
        pyre_interpreter::typedef::make_getset_descriptor_named(
            pyre_interpreter::make_builtin_function_with_arity(
                "closed",
                |args| {
                    // GetSetProperty fget callbacks receive (descriptor_self,
                    // w_obj); the mmap instance is the second argument.
                    let obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                    Ok(pyre_object::w_bool_from(
                        mmap_get_attr_i64(obj, "_ptr") == 0,
                    ))
                },
                2,
            ),
            "closed",
        )
    );

    // `interp_mmap.py descr_size` returns `mmap.file_size()` —
    // the underlying file's current size via fstat, not the mapped
    // length.  The two diverge after `resize()`, and an anonymous mmap
    // (no fd) raises ValueError per rmmap.py:MMap.file_size.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "size",
        pyre_interpreter::make_builtin_function_with_arity("size", __majit_wrap_mmap_size, 1,)
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "tell",
        pyre_interpreter::make_builtin_function_with_arity(
            "tell",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_int_new(mmap_get_attr_i64(obj, "_pos")))
            },
            1,
        )
    );

    // CPython's mmap object advertises the seekable stream capability.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "seekable",
        pyre_interpreter::make_builtin_function_with_arity(
            "seekable",
            |_args| Ok(pyre_object::w_bool_from(true)),
            1,
        )
    );

    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "seek",
            pyre_interpreter::make_builtin_function("seek", |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "seek() missing argument",
                    ));
                }
                let mut obj = args[0];
                let (_, len) = mmap_ptr(obj)?;
                if !{ pyre_object::is_int(args[1]) } {
                    return Err(pyre_interpreter::PyError::type_error(
                        "seek: offset must be an integer",
                    ));
                }
                let off = { pyre_object::w_int_get_value(args[1]) };
                let whence = if args.len() >= 3 {
                    if !{ pyre_object::is_int(args[2]) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "seek: whence must be an integer",
                        ));
                    }
                    pyre_object::w_int_get_value(args[2])
                } else {
                    0
                };
                let cur = pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_pos"));
                let new_pos = match whence {
                    0 => off,
                    1 => cur + off,
                    2 => len as i64 + off,
                    _ => {
                        return Err(pyre_interpreter::PyError::value_error("invalid whence"));
                    }
                };
                if new_pos < 0 || (new_pos as usize) > len {
                    return Err(pyre_interpreter::PyError::value_error("seek out of range"));
                }
                mmap_set_attr(obj, "_pos", pyre_object::w_int_new(new_pos));
                Ok(pyre_object::w_int_new(new_pos))
            })
        )
    };

    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "read",
            pyre_interpreter::make_builtin_function("read", |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error("read() missing self"));
                }
                let mut obj = args[0];
                // `interp_mmap.py read(num=-1)` — None or -1 reads to
                // end; positive value caps at remaining bytes.
                let requested = if args.len() >= 2 && !{ pyre_object::is_none(args[1]) } {
                    Some(pyre_object::with_roots!(obj => mmap_index_w(obj, args[1]))?)
                } else {
                    None
                };
                let (p, len) = mmap_ptr(obj)?;
                let pos = pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_pos")) as usize;
                let remaining = len.saturating_sub(pos);
                let n = if let Some(req) = requested {
                    if req < 0 {
                        remaining
                    } else {
                        (req as usize).min(remaining)
                    }
                } else {
                    remaining
                };
                let slice = { std::slice::from_raw_parts(p.add(pos), n) };
                let data: Vec<u8> = slice.to_vec();
                mmap_set_attr(obj, "_pos", pyre_object::w_int_new((pos + n) as i64));
                Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data))
            })
        )
    };

    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "read_byte",
            pyre_interpreter::make_builtin_function_with_arity(
                "read_byte",
                |args| {
                    let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                    let (p, len) = mmap_ptr(obj)?;
                    let pos =
                        pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_pos")) as usize;
                    if pos >= len {
                        return Err(pyre_interpreter::PyError::value_error(
                            "read byte out of range",
                        ));
                    }
                    let b = { *p.add(pos) };
                    mmap_set_attr(obj, "_pos", pyre_object::w_int_new((pos + 1) as i64));
                    Ok(pyre_object::w_int_new(b as i64))
                },
                1,
            )
        )
    };

    // `interp_mmap.py readline` — read bytes from current pos until
    // the first '\n' (inclusive); if absent, read to end.  Mirrors
    // `rmmap.py`.
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "readline",
            pyre_interpreter::make_builtin_function_with_arity(
                "readline",
                |args| {
                    let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                    let (p, len) = mmap_ptr(obj)?;
                    let pos =
                        pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_pos")) as usize;
                    if pos >= len {
                        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&[]));
                    }
                    let tail = { std::slice::from_raw_parts(p.add(pos), len - pos) };
                    let eol = tail
                        .iter()
                        .position(|&b| b == b'\n')
                        .map_or(len, |i| pos + i + 1);
                    let data = { std::slice::from_raw_parts(p.add(pos), eol - pos) }.to_vec();
                    mmap_set_attr(obj, "_pos", pyre_object::w_int_new(eol as i64));
                    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data))
                },
                1,
            )
        )
    };

    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "write",
            pyre_interpreter::make_builtin_function_with_arity(
                "write",
                |args| {
                    if args.len() < 2 {
                        return Err(pyre_interpreter::PyError::type_error(
                            "write() missing buffer",
                        ));
                    }
                    let roots = pyre_object::gc_roots::push_roots();
                    let args_base = roots.pin_roots(args);
                    let access = mmap_get_attr_i64(roots.get(args_base), "_access");
                    if access == MMAP_ACCESS_READ {
                        return Err(pyre_interpreter::PyError::type_error("mmap is read-only"));
                    }
                    let arg1 = roots.get(args_base + 1);
                    let mut obj = roots.get(args_base);
                    drop(roots);
                    let buf = {
                        if !pyre_object::bytesobject::is_bytes_like(arg1) {
                            return Err(pyre_interpreter::PyError::type_error(
                                "write: buffer must be bytes-like",
                            ));
                        }
                        pyre_object::bytesobject::bytes_like_data(arg1)
                    };
                    let (p, len) = mmap_ptr(obj)?;
                    let pos =
                        pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_pos")) as usize;
                    if pos + buf.len() > len {
                        return Err(pyre_interpreter::PyError::value_error("data out of range"));
                    }
                    {
                        std::ptr::copy_nonoverlapping(buf.as_ptr(), p.add(pos), buf.len())
                    };
                    mmap_set_attr(
                        obj,
                        "_pos",
                        pyre_object::w_int_new((pos + buf.len()) as i64),
                    );
                    Ok(pyre_object::w_int_new(buf.len() as i64))
                },
                2,
            )
        )
    };

    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "write_byte",
            pyre_interpreter::make_builtin_function_with_arity(
                "write_byte",
                |args| {
                    if args.len() < 2 {
                        return Err(pyre_interpreter::PyError::type_error(
                            "write_byte() missing arg",
                        ));
                    }
                    let roots = pyre_object::gc_roots::push_roots();
                    let args_base = roots.pin_roots(args);
                    let access = mmap_get_attr_i64(roots.get(args_base), "_access");
                    if access == MMAP_ACCESS_READ {
                        return Err(pyre_interpreter::PyError::type_error("mmap is read-only"));
                    }
                    let arg1 = roots.get(args_base + 1);
                    let mut obj = roots.get(args_base);
                    drop(roots);
                    // `interp_mmap.py write_byte(byte=int)` —
                    // `@unwrap_spec(byte=int)` rejects non-ints, then
                    // `chr(byte)` raises ValueError on values outside
                    // 0..256.
                    let raw = pyre_object::with_roots!(obj => mmap_index_w(obj, arg1))?;
                    if !(0..=255).contains(&raw) {
                        return Err(pyre_interpreter::PyError::value_error(
                            "byte must be in range(0, 256)",
                        ));
                    }
                    let (p, len) = mmap_ptr(obj)?;
                    let pos =
                        pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_pos")) as usize;
                    if pos >= len {
                        return Err(pyre_interpreter::PyError::value_error(
                            "write_byte out of range",
                        ));
                    }
                    {
                        *p.add(pos) = raw as u8
                    };
                    mmap_set_attr(obj, "_pos", pyre_object::w_int_new((pos + 1) as i64));
                    Ok(pyre_object::w_none())
                },
                2,
            )
        )
    };

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        // `interp_mmap.py flush(offset=0, size=0)` —
        // `@unwrap_spec(offset=int, size=int)` then `mmap.flush(offset,
        // size)`.  rmmap.flush passes size==0 through as "whole map",
        // which we mirror via `len - off`.
        "flush",
        pyre_interpreter::make_builtin_function("flush", __majit_wrap_mmap_flush)
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "find",
        pyre_interpreter::make_builtin_function("find", |args| { mmap_gfind("find", args, false) })
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "rfind",
        pyre_interpreter::make_builtin_function("rfind", |args| {
            mmap_gfind("rfind", args, true)
        })
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__enter__",
        pyre_interpreter::make_builtin_function_with_arity(
            "__enter__",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        )
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__exit__",
        pyre_interpreter::make_builtin_function("__exit__", |args| {
            if let Some(mut obj) = args.first().copied() {
                let p = pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_ptr")) as usize;
                if p != 0 {
                    pyre_object::with_roots!(obj => mmap_check_exports(obj, "cannot close exported pointers exist"))?;
                    mmap_close_native(obj);
                }
            }
            Ok(pyre_object::w_bool_from(false))
        })
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__len__",
        pyre_interpreter::make_builtin_function_with_arity(
            "__len__",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_int_new(mmap_get_attr_i64(obj, "_len")))
            },
            1,
        )
    );

    // `interp_mmap.py descr_getitem` — integer index returns a
    // single int byte; slice returns bytes (contiguous fast path for
    // step=1, stepped extraction otherwise).
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "__getitem__",
            pyre_interpreter::make_builtin_function_with_arity(
                "__getitem__",
                |args| {
                    if args.len() < 2 {
                        return Err(pyre_interpreter::PyError::type_error(
                            "__getitem__() requires index",
                        ));
                    }
                    let mut obj = args[0];
                    let index = args[1];
                    let _ = mmap_ptr(obj)?;
                    if pyre_object::is_slice(index) {
                        // `mmap_subscript` runs `PySlice_Unpack` (which may call
                        // `__index__`), then `CHECK_VALID`, and only then
                        // `PySlice_AdjustIndices(self->size, ...)` — so the size
                        // the bounds are clamped against is the one the mapping
                        // has after that call, not before it.
                        let (start_raw, stop_raw, step) = pyre_object::with_roots!(obj =>
                        pyre_interpreter::sliceobject::slice_unpack(
                            pyre_object::sliceobject::w_slice_get_start(index),
                            pyre_object::sliceobject::w_slice_get_stop(index),
                            pyre_object::sliceobject::w_slice_get_step(index),
                        ))?;
                        let (p, len) = mmap_ptr(obj)?;
                        let (start, stop, step, _) =
                            pyre_interpreter::sliceobject::slice_adjust_indices(
                                start_raw, stop_raw, step, len as i64,
                            );
                        if step == 1 {
                            if stop <= start {
                                return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&[]));
                            }
                            let n = (stop - start) as usize;
                            let data = { std::slice::from_raw_parts(p.add(start as usize), n) };
                            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(data));
                        }
                        let mut out = Vec::new();
                        let mut i = start;
                        while (step > 0 && i < stop) || (step < 0 && i > stop) {
                            out.push(*p.add(i as usize));
                            let Some(next) = i.checked_add(step) else {
                                break;
                            };
                            i = next;
                        }
                        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&out));
                    }
                    let mut idx = pyre_object::with_roots!(obj => mmap_index_w(obj, index))?;
                    // `mmap_subscript`'s integer arm re-checks validity after
                    // `PyNumber_AsSsize_t` and then reads `self->size`, so a
                    // `resize()` from that `__index__` moves the bound too.
                    let (p, len) = mmap_ptr(obj)?;
                    let len_i64 = len as i64;
                    if idx < 0 {
                        idx += len_i64;
                    }
                    if idx < 0 || idx >= len_i64 {
                        return Err(pyre_interpreter::PyError::index_error(
                            "mmap index out of range",
                        ));
                    }
                    let b = { *p.add(idx as usize) };
                    Ok(pyre_object::w_int_new(b as i64))
                },
                2,
            )
        )
    };

    // `interp_mmap.py descr_setitem` — integer index writes a
    // single byte (0..256); slice writes a buffer whose length matches
    // the slice length.  Read-only mmaps raise TypeError.
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "__setitem__",
            pyre_interpreter::make_builtin_function_with_arity(
                "__setitem__",
                |args| {
                    if args.len() < 3 {
                        return Err(pyre_interpreter::PyError::type_error(
                            "__setitem__() requires index and value",
                        ));
                    }
                    let mut obj = args[0];
                    let mut index = args[1];
                    let mut value = args[2];
                    let access = pyre_object::with_roots!(index, obj, value => mmap_get_attr_i64(obj, "_access"));
                    if access == MMAP_ACCESS_READ {
                        return Err(pyre_interpreter::PyError::type_error("mmap is read-only"));
                    }
                    let _ = mmap_ptr(obj)?;
                    if pyre_object::is_slice(index) {
                        // `mmap_ass_subscript` splits the same way `mmap_subscript`
                        // does: unpack, re-check, then adjust against the size the
                        // mapping has now.
                        let (start_raw, stop_raw, step) = pyre_object::with_roots!(obj, value =>
                        pyre_interpreter::sliceobject::slice_unpack(
                            pyre_object::sliceobject::w_slice_get_start(index),
                            pyre_object::sliceobject::w_slice_get_stop(index),
                            pyre_object::sliceobject::w_slice_get_step(index),
                        ))?;
                        let (p, len) = mmap_ptr(obj)?;
                        let (start, stop, step, _) =
                            pyre_interpreter::sliceobject::slice_adjust_indices(
                                start_raw, stop_raw, step, len as i64,
                            );
                        let length = if step > 0 && stop > start {
                            (1 + (i128::from(stop) - i128::from(start) - 1) / i128::from(step))
                                as i64
                        } else if step < 0 && start > stop {
                            (1 + (i128::from(start) - i128::from(stop) - 1) / -i128::from(step))
                                as i64
                        } else {
                            0
                        };
                        if !{ pyre_object::bytesobject::is_bytes_like(value) } {
                            return Err(pyre_interpreter::PyError::type_error(
                                "mmap slice assignment must be bytes-like",
                            ));
                        }
                        let buf = { pyre_object::bytesobject::bytes_like_data(value) };
                        if (buf.len() as i64) != length {
                            return Err(pyre_interpreter::PyError::value_error(
                                "mmap slice assignment is wrong size",
                            ));
                        }
                        if step == 1 {
                            if length > 0 {
                                {
                                    std::ptr::copy_nonoverlapping(
                                        buf.as_ptr(),
                                        p.add(start as usize),
                                        length as usize,
                                    );
                                }
                            }
                        } else {
                            let mut i = start;
                            let mut k = 0usize;
                            while (step > 0 && i < stop) || (step < 0 && i > stop) {
                                {
                                    *p.add(i as usize) = buf[k]
                                };
                                let Some(next) = i.checked_add(step) else {
                                    break;
                                };
                                i = next;
                                k += 1;
                            }
                        }
                        return Ok(pyre_object::w_none());
                    }
                    let mut idx = pyre_object::with_roots!(obj, value => mmap_index_w(obj, index))?;
                    let (p, len) = mmap_ptr(obj)?;
                    let len_i64 = len as i64;
                    if idx < 0 {
                        idx += len_i64;
                    }
                    if idx < 0 || idx >= len_i64 {
                        return Err(pyre_interpreter::PyError::index_error(
                            "mmap index out of range",
                        ));
                    }
                    if !{ pyre_object::is_int(value) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "mmap item value must be an integer",
                        ));
                    }
                    let v = { pyre_object::w_int_get_value(value) };
                    if !(0..256).contains(&v) {
                        return Err(pyre_interpreter::PyError::value_error(
                            "mmap item value must be in range(0, 256)",
                        ));
                    }
                    {
                        *p.add(idx as usize) = v as u8
                    };
                    Ok(pyre_object::w_none())
                },
                3,
            )
        )
    };

    // `interp_mmap.py descr_iter` — iterate the 1-byte slices
    // `m[i:i+1]` forwards.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__iter__",
        pyre_interpreter::make_builtin_function_with_arity(
            "__iter__",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(make_mmap_iterator(obj, 0, 1))
            },
            1,
        )
    );

    // `interp_mmap.py descr_reversed` — iterate the 1-byte slices
    // `m[i:i+1]` from `len(m) - 1` down to `0`.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__reversed__",
        pyre_interpreter::make_builtin_function_with_arity(
            "__reversed__",
            |args| {
                let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let len = pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_len"));
                Ok(make_mmap_iterator(obj, len - 1, -1))
            },
            1,
        )
    );

    // `interp_mmap.py` — `madvise` is registered from `optional[]`,
    // i.e. only where `rmmap` has the call.  Windows has no counterpart, so
    // the method is absent there rather than present and inert.
    //
    // `interp_mmap.py:descr_madvise` — call madvise(addr+start, length,
    // advice).  Defaults: start=0, length=remaining bytes.
    #[cfg(unix)]
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "madvise",
        pyre_interpreter::make_builtin_function("madvise", |args| {
            let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            let nargs = args.len();
            let roots = pyre_object::gc_roots::push_roots();
            let args_base = if args.is_empty() {
                roots.pin_roots(&[obj])
            } else {
                roots.pin_roots(args)
            };
            let obj_slot = args_base;
            let _ = mmap_ptr(roots.get(obj_slot))?;
            if nargs < 2 {
                return Err(pyre_interpreter::PyError::type_error(
                    "madvise() requires option",
                ));
            }
            let option = mmap_index_w(roots.get(obj_slot), roots.get(args_base + 1))? as i32;
            let start_raw = if nargs > 2 {
                mmap_index_w(roots.get(obj_slot), roots.get(args_base + 2))?
            } else {
                0
            };
            let length_arg = if nargs > 3 {
                Some(mmap_index_w(roots.get(obj_slot), roots.get(args_base + 3))?)
            } else {
                None
            };
            obj = roots.get(obj_slot);
            drop(roots);
            // Every `__index__` above could have resized the mapping, so the
            // extent the advice is applied to is read once they are all done.
            let (_, total) = mmap_ptr(obj)?;
            if start_raw < 0 || usize::try_from(start_raw).map_or(true, |start| start >= total) {
                return Err(pyre_interpreter::PyError::value_error(
                    "madvise start out of bounds",
                ));
            }
            let start = start_raw as usize;
            let length_raw = length_arg.unwrap_or((total - start) as i64);
            if length_raw < 0 {
                return Err(pyre_interpreter::PyError::value_error(
                    "madvise length invalid",
                ));
            }
            if i128::from(length_raw) + start as i128 > isize::MAX as i128 {
                return Err(pyre_interpreter::PyError::overflow_error(
                    "madvise length too large",
                ));
            }
            let length = (length_raw as usize).min(total - start);
            #[cfg(not(target_os = "redox"))]
            {
                mmap_madvise(obj, start, length, option).map_err(|e| mmap_io_err(e, "madvise"))?;
            }
            #[cfg(target_os = "redox")]
            {
                let _ = (length, option);
            }
            Ok(pyre_object::w_none())
        })
    );

    // `interp_mmap.py:descr_move` — copy `length` bytes from source
    // offset to dest offset within the mapping (memmove semantics).
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "move",
            pyre_interpreter::make_builtin_function_with_arity(
                "move",
                |args| {
                    if args.len() < 4 {
                        return Err(pyre_interpreter::PyError::type_error(
                            "move() requires dest, src, count",
                        ));
                    }
                    // `interp_mmap.py move(dest, src, count)` —
                    // `@unwrap_spec(dest=int, src=int, count=int)` plus
                    // `self.check_writeable()` upfront.  We require all
                    // three args use the index protocol and reject ACCESS_READ.
                    let roots = pyre_object::gc_roots::push_roots();
                    let args_base = roots.pin_roots(args);
                    if mmap_get_attr_i64(roots.get(args_base), "_access") == MMAP_ACCESS_READ {
                        return Err(pyre_interpreter::PyError::type_error("mmap is read-only"));
                    }
                    let dest =
                        mmap_index_w(roots.get(args_base), roots.get(args_base + 1))? as usize;
                    let src =
                        mmap_index_w(roots.get(args_base), roots.get(args_base + 2))? as usize;
                    let count =
                        mmap_index_w(roots.get(args_base), roots.get(args_base + 3))? as usize;
                    let obj = roots.get(args_base);
                    drop(roots);
                    let (p, total) = mmap_ptr(obj)?;
                    if dest.saturating_add(count) > total || src.saturating_add(count) > total {
                        return Err(pyre_interpreter::PyError::value_error(
                            "source or destination out of range",
                        ));
                    }
                    {
                        std::ptr::copy(p.add(src), p.add(dest), count);
                    }
                    Ok(pyre_object::w_none())
                },
                4,
            )
        )
    };

    // `interp_mmap.py resize` → `rmmap.py`.  POSIX path:
    // ftruncate the backing fd (if any) to `offset + newsize`, then
    // mremap(MREMAP_MAYMOVE).  Platforms without mremap (e.g. macOS)
    // raise SystemError to match PyPy's RValueError→SystemError
    // translation at `interp_mmap.py`.  Read-only / copy
    // mappings reject with TypeError.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "resize",
        pyre_interpreter::make_builtin_function_with_arity(
            "resize",
            |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "resize() requires newsize",
                    ));
                }
                let mut obj = args[0];
                // `__index__` runs user code, so convert before anything
                // is read from the mapping: it can close or resize it.
                let newsize = pyre_object::with_roots!(obj => mmap_index_w(obj, args[1]))?;
                let access = pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_access"));
                if !(access == MMAP_ACCESS_WRITE || access == MMAP_ACCESS_DEFAULT) {
                    return Err(pyre_interpreter::PyError::type_error(
                        "mmap can't resize a readonly or copy-on-write memory map.",
                    ));
                }
                pyre_object::with_roots!(obj => mmap_check_exports(obj, "mmap can't resize with extant buffers exported."))?;
                let (p, old_len) = mmap_ptr(obj)?;
                if newsize < 0 {
                    return Err(pyre_interpreter::PyError::value_error(
                        "new_size must be positive",
                    ));
                }
                let newsize = newsize as usize;
                if !mmap_native(obj)?.trackfd {
                    return Err(pyre_interpreter::PyError::value_error(
                        "mmap can't resize with trackfd=False.",
                    ));
                }
                mmap_resize_mapping(obj, p, old_len, newsize)?;
                Ok(pyre_object::w_none())
            },
            2,
        )
    );

    // `interp_mmap.py:descr_repr` — `<mmap.mmap closed=False, access=...>`.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__repr__",
        pyre_interpreter::make_builtin_function_with_arity(
            "__repr__",
            |args| {
                let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                // `interp_mmap.py descr_repr`: closed mmaps
                // suppress the inner fields and emit just
                // `<mmap.mmap closed=True>`; otherwise dump
                // access / length / pos / offset.  Capitalised True /
                // False matches CPython's bool repr.
                if pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_ptr")) == 0 {
                    return Ok(pyre_object::w_str_new_managed("<mmap.mmap closed=True>"));
                }
                let len = pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_len"));
                let pos = pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_pos"));
                let offset = pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_offset"));
                let access = mmap_get_attr_i64(obj, "_access");
                let access_str = match access {
                    x if x == MMAP_ACCESS_READ => "ACCESS_READ",
                    x if x == MMAP_ACCESS_WRITE => "ACCESS_WRITE",
                    x if x == MMAP_ACCESS_COPY => "ACCESS_COPY",
                    _ => "ACCESS_DEFAULT",
                };
                // W_MMap.descr_repr returns `space.newtext(...)` for both
                // live and closed mappings.
                Ok(pyre_object::w_str_new_managed(&format!(
                    "<mmap.mmap closed=False, access={access_str}, length={len}, pos={pos}, offset={offset}>"
                )))
            },
            1,
        )
    );
}

/// `rmmap.MMap.resize` — `os.ftruncate` the backing fd (if any) to
/// `offset + newsize`, then map again. There is no `mremap` here, so the
/// mapping is re-created at the new size. A file-backed map is re-mapped
/// from the fd already owned (no second `dup`); an anonymous map is remade
/// and the surviving bytes copied. The new address is not exposed to Python.
///
/// Re-creating the mapping would grow a shared anonymous mapping, which
/// `mremap` refuses on Linux (kernel bug 8691). The guard below keeps that
/// refusal.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn mmap_resize_mapping(
    mut obj: pyre_object::PyObjectRef,
    p: *mut u8,
    old_len: usize,
    newsize: usize,
) -> Result<(), pyre_interpreter::PyError> {
    let fd = pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_fd")) as libc::c_int;
    let offset = pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_offset"));
    // `mmap_resize_method`: shared anonymous maps cannot grow.
    //   if (self->fd == -1 && !(self->flags & MAP_PRIVATE) && new_size > self->size)
    if fd < 0 && mmap_native(obj)?.flags & rmmap::MAP_PRIVATE == 0 && newsize > old_len {
        return Err(pyre_interpreter::PyError::value_error(
            "mmap: can't expand a shared anonymous mapping on Linux",
        ));
    }
    let mapped = if fd >= 0 {
        let rc = pyre_object::with_roots!(obj => unsafe {
            rmmap::c_ftruncate(fd, (offset as libc::off_t) + newsize as libc::off_t)
        });
        if rc < 0 {
            return Err(pyre_interpreter::PyError::os_error_with_errno(
                majit_rlib::rposix::get_saved_errno(),
                "ftruncate",
            ));
        }
        // `mremap` keeps the mapping's protection. Remap from the stored mode
        // (`rmmap.mmap` access→prot), not from `_access`: a PROT_READ map and
        // a shared writable map are different, and `MAP_FIXED` is not replayed.
        let mode = pyre_object::with_roots!(obj => mmap_get_attr_i64(obj, "_mode"));
        let (flags, prot) = if mode == MMAP_ACCESS_READ {
            (rmmap::MAP_SHARED, rmmap::PROT_READ)
        } else if mode == MMAP_ACCESS_COPY {
            (rmmap::MAP_PRIVATE, rmmap::PROT_READ | rmmap::PROT_WRITE)
        } else {
            (rmmap::MAP_SHARED, rmmap::PROT_READ | rmmap::PROT_WRITE)
        };
        let ptr = pyre_object::with_roots!(obj => unsafe {
            rmmap::c_mmap(
                std::ptr::null_mut(),
                newsize,
                prot,
                flags,
                fd,
                offset as libc::off_t,
            )
        });
        if ptr == libc::MAP_FAILED {
            return Err(pyre_interpreter::PyError::os_error_with_errno(
                majit_rlib::rposix::get_saved_errno(),
                "mmap",
            ));
        }
        PosixMap {
            ptr: ptr.cast(),
            len: newsize,
        }
    } else {
        pyre_object::with_roots!(obj => mmap_remake_anon(p, old_len, newsize))?
    };
    mmap_native(obj)?.mapped = Some(MappedObj::Mapped(mapped));
    Ok(())
}

/// `rmmap.py` — unmap the view and close the mapping object, move the
/// file's EOF to `offset + newsize`, then map the file again.  Dropping the
/// old mapping first is required, not just tidy: `SetEndOfFile` fails with
/// ERROR_USER_MAPPED_FILE while any view of the file is open.  An anonymous
/// mapping has no file to extend, so it is remade and the surviving bytes
/// copied, as on Linux.
#[cfg(windows)]
fn mmap_resize_mapping(
    obj: pyre_object::PyObjectRef,
    p: *mut u8,
    old_len: usize,
    newsize: usize,
) -> Result<(), pyre_interpreter::PyError> {
    let tagname = mmap_native(obj)?.tagname.clone();
    let named = !tagname.is_empty();
    let handle = mmap_handle(obj);
    let offset = mmap_get_attr_i64(obj, "_offset");
    let access = mmap_access_mode(mmap_get_attr_i64(obj, "_mode"));
    // A tagged mapping with no file behind it is pagefile-backed: there is no
    // EOF to move, so the bytes that survive the resize are carried by hand.
    let old_bytes = (named && handle.is_none())
        .then(|| unsafe { std::slice::from_raw_parts(p, old_len) }.to_vec());

    let mapped = if let Some(handle) = handle {
        // The view has to go before the file can grow, so `_ptr` reads 0 from
        // here until one of the two remaps below puts a mapping back.
        mmap_native(obj)?.mapped = None;
        let remap = |size: usize, reject_existing: bool| {
            mmap_remap_named(handle, offset, access, size, &tagname, reject_existing)
        };
        match host_mmap::extend_file(handle, offset + newsize as i64)
            .and_then(|()| remap(newsize, true))
        {
            Ok(mapped) => mapped,
            Err(e) => {
                // Put the mapping back at its old size; if even that fails the
                // object stays mapping-less, which reads as closed.
                if let Ok(old) = remap(old_len, false) {
                    mmap_native(obj)?.mapped = Some(old);
                }
                return Err(mmap_io_err(e, "mmap"));
            }
        }
    } else if named {
        // A pagefile-backed tagged mapping has no file EOF to move, but the
        // same close/recreate sequence applies.  ERROR_ALREADY_EXISTS comes
        // from `CreateFileMapping` when another live view still owns the name.
        mmap_native(obj)?.mapped = None;
        let remap = |size: usize, reject_existing: bool| {
            mmap_remap_named(
                host_mmap::INVALID_HANDLE,
                offset,
                access,
                size,
                &tagname,
                reject_existing,
            )
        };
        match remap(newsize, true) {
            Ok(mut mapped) => {
                let old = old_bytes.as_deref().unwrap_or_default();
                let keep = old.len().min(newsize);
                mapped.as_mut_slice()[..keep].copy_from_slice(&old[..keep]);
                mapped
            }
            Err(e) => {
                if let Ok(mut old_mapping) = remap(old_len, false) {
                    if let Some(old) = old_bytes.as_deref() {
                        old_mapping.as_mut_slice().copy_from_slice(old);
                    }
                    mmap_native(obj)?.mapped = Some(old_mapping);
                }
                return Err(mmap_io_err(e, "CreateFileMapping"));
            }
        }
    } else {
        MappedObj::Mapped(mmap_remake_anon(p, old_len, newsize)?)
    };
    mmap_native(obj)?.mapped = Some(mapped);
    Ok(())
}

/// Recreate the Windows mapping object under the name it already had, matching
/// `rmmap.py:635-636`, which passes `self.tagname` whether or not one was
/// given.  `CreateFileMappingW` reports ERROR_ALREADY_EXISTS when another
/// mapping still owns the name; a resize surfaces that OS error, while the
/// recovery remap is allowed to reopen the existing object.  An untagged
/// mapping cannot reach it — a zero-length name is not a name.
#[cfg(windows)]
fn mmap_remap_named(
    handle: host_mmap::Handle,
    offset: i64,
    access: host_mmap::AccessMode,
    size: usize,
    tagname: &str,
    reject_existing: bool,
) -> std::io::Result<MappedObj> {
    let named = host_mmap::create_named_mapping(handle, tagname, access, offset, size)?;
    const ERROR_ALREADY_EXISTS: i32 = 183;
    if reject_existing && host_mmap::last_error() as i32 == ERROR_ALREADY_EXISTS {
        drop(named);
        return Err(std::io::Error::from_raw_os_error(ERROR_ALREADY_EXISTS));
    }
    Ok(MappedObj::Named(named))
}

/// A resized anonymous mapping is a new `MAP_PRIVATE|MAP_ANONYMOUS` map
/// holding the bytes that fit. Caller pins `obj` across this: `c_mmap`
/// releases the GIL.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn mmap_remake_anon(
    p: *mut u8,
    old_len: usize,
    newsize: usize,
) -> Result<PosixMap, pyre_interpreter::PyError> {
    let keep = old_len.min(newsize);
    let old = unsafe { std::slice::from_raw_parts(p, keep) }.to_vec();
    let ptr = unsafe {
        rmmap::c_mmap(
            std::ptr::null_mut(),
            newsize,
            rmmap::PROT_READ | rmmap::PROT_WRITE,
            rmmap::MAP_PRIVATE | rmmap::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(pyre_interpreter::PyError::os_error_with_errno(
            majit_rlib::rposix::get_saved_errno(),
            "mmap",
        ));
    }
    let mut mapped = PosixMap {
        ptr: ptr.cast(),
        len: newsize,
    };
    mapped.as_mut_slice()[..keep].copy_from_slice(&old);
    Ok(mapped)
}

/// A resized anonymous mapping is a new mapping holding the bytes that fit.
#[cfg(windows)]
fn mmap_remake_anon(
    p: *mut u8,
    old_len: usize,
    newsize: usize,
) -> Result<host_mmap::MappedFile, pyre_interpreter::PyError> {
    let keep = old_len.min(newsize);
    let old = unsafe { std::slice::from_raw_parts(p, keep) }.to_vec();
    let mut mapped = host_mmap::map_anon(newsize).map_err(|e| mmap_io_err(e, "mmap"))?;
    mapped.as_mut_slice()[..keep].copy_from_slice(&old);
    Ok(mapped)
}

/// Platforms without mremap raise SystemError, matching PyPy's
/// RValueError→SystemError translation at `interp_mmap.py:155-157`.
#[cfg(not(any(target_os = "linux", target_os = "android", windows)))]
fn mmap_resize_mapping(
    _obj: pyre_object::PyObjectRef,
    _p: *mut u8,
    _old_len: usize,
    _newsize: usize,
) -> Result<(), pyre_interpreter::PyError> {
    Err(pyre_interpreter::PyError::new(
        pyre_interpreter::error::PyErrorKind::SystemError,
        "mmap: resizing not available--no mremap()",
    ))
}

/// The mapping mode an `ACCESS_*` argument asks for. Windows expresses a
/// mapping as an `AccessMode` rather than the raw `flProtect` /
/// `dwDesiredAccess` pair `rmmap.mmap` derives.
#[cfg(windows)]
fn mmap_access_mode(access: i64) -> host_mmap::AccessMode {
    match access {
        x if x == MMAP_ACCESS_READ => host_mmap::AccessMode::Read,
        x if x == MMAP_ACCESS_WRITE => host_mmap::AccessMode::Write,
        x if x == MMAP_ACCESS_COPY => host_mmap::AccessMode::Copy,
        _ => host_mmap::AccessMode::Default,
    }
}

#[cfg(any(unix, windows))]
const MMAP_ACCESS_DEFAULT: i64 = 0;
#[cfg(any(unix, windows))]
const MMAP_ACCESS_READ: i64 = 1;
#[cfg(any(unix, windows))]
const MMAP_ACCESS_WRITE: i64 = 2;
#[cfg(any(unix, windows))]
const MMAP_ACCESS_COPY: i64 = 3;

/// `interp_mmap.py W_MMap.__init__` — hand the finished mapping to a
/// fresh instance of `cls`. `objspace.py` `allocate_instance` builds the
/// builtin layout, or `typedef.py` `_getusercls` for a subclass. The sweep
/// destructor still closes the mapping.
#[cfg(any(unix, windows))]
fn mmap_new_object(
    cls: pyre_object::PyObjectRef,
    backend: NativeMMap,
    access: i64,
    mode: i64,
    offset: i64,
) -> pyre_object::PyObjectRef {
    let backend = Box::into_raw(Box::new(backend));
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    W_MMap::allocate_instance(
        W_MMap {
            ob: pyre_object::PyObject::default(),
            backend,
            pos: 0,
            access,
            mode,
            offset,
            exports: 0,
            lifeline: pyre_object::PY_NULL,
        },
        pyre_object::gc_roots::shadow_stack_get(cls_slot),
    )
}

// `interp_mmap.py:55-130 mmap_new` / CPython 3.14's `trackfd` addition.
#[cfg(unix)]
fn mmap_construct(
    mut cls: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let bound = pyre_interpreter::builtins::bind_builtin_kwargs(
        args,
        &[
            "fileno", "length", "flags", "prot", "access", "offset", "trackfd",
        ],
        &[true, true, false, false, false, false, false],
        "mmap",
    )?;
    let index_i64 = |obj: pyre_object::PyObjectRef, label: &str| {
        if obj.is_null() {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "mmap() {label} must be an integer"
            )));
        }
        pyre_interpreter::baseobjspace::int_w(pyre_interpreter::baseobjspace::space_index(obj)?)
    };
    let fd_raw = pyre_object::with_roots!(cls => index_i64(bound[0], "fileno"))?;
    let fd = libc::c_int::try_from(fd_raw).map_err(|_| {
        pyre_interpreter::PyError::overflow_error("Python int too large to convert to C int")
    })?;
    let length_raw = pyre_object::with_roots!(cls => index_i64(bound[1], "length"))?;
    let flags_arg = if bound[2].is_null() {
        rmmap::MAP_SHARED
    } else {
        pyre_object::with_roots!(cls => index_i64(bound[2], "flags"))? as libc::c_int
    };
    let prot_arg = if bound[3].is_null() {
        rmmap::PROT_READ | rmmap::PROT_WRITE
    } else {
        pyre_object::with_roots!(cls => index_i64(bound[3], "prot"))? as libc::c_int
    };
    let mut access = if bound[4].is_null() {
        MMAP_ACCESS_DEFAULT
    } else {
        pyre_object::with_roots!(cls => index_i64(bound[4], "access"))?
    };
    let offset = if bound[5].is_null() {
        0
    } else {
        pyre_object::with_roots!(cls => index_i64(bound[5], "offset"))?
    };
    let trackfd = if bound[6].is_null() {
        true
    } else {
        pyre_object::with_roots!(cls => pyre_interpreter::baseobjspace::is_true(bound[6]))?
    };
    // `rmmap.mmap` performs these guards in this order, and before the
    // size_t cast.
    if access != MMAP_ACCESS_DEFAULT
        && (flags_arg != rmmap::MAP_SHARED || prot_arg != rmmap::PROT_READ | rmmap::PROT_WRITE)
    {
        return Err(pyre_interpreter::PyError::value_error(
            "mmap can't specify both access and flags, prot.",
        ));
    }
    // `rmmap.py _check_map_size`.
    if length_raw < 0 {
        return Err(pyre_interpreter::PyError::type_error(
            "memory mapped size must be positive",
        ));
    }
    if offset < 0 {
        return Err(pyre_interpreter::PyError::value_error("negative offset"));
    }
    let mut length = usize::try_from(length_raw).map_err(|_| {
        pyre_interpreter::PyError::overflow_error("memory mapped length must be positive")
    })?;
    let (mut flags, prot) = match access {
        x if x == MMAP_ACCESS_READ => (rmmap::MAP_SHARED, rmmap::PROT_READ),
        x if x == MMAP_ACCESS_WRITE => (rmmap::MAP_SHARED, rmmap::PROT_READ | rmmap::PROT_WRITE),
        x if x == MMAP_ACCESS_COPY => (rmmap::MAP_PRIVATE, rmmap::PROT_READ | rmmap::PROT_WRITE),
        x if x == MMAP_ACCESS_DEFAULT => {
            if prot_arg & rmmap::PROT_WRITE != 0 && prot_arg & rmmap::PROT_READ == 0 {
                access = MMAP_ACCESS_WRITE;
            } else if prot_arg & rmmap::PROT_WRITE == 0 {
                access = MMAP_ACCESS_READ;
            }
            (flags_arg, prot_arg)
        }
        _ => {
            return Err(pyre_interpreter::PyError::value_error(
                "mmap invalid access parameter.",
            ));
        }
    };
    // `rmmap.mmap` stores a mode derived from the prot/flags actually passed
    // to `c_mmap`: write+private is ACCESS_COPY, write+shared is ACCESS_WRITE,
    // otherwise ACCESS_READ. `_access` still records the caller's argument.
    let real_fd = fd;
    if real_fd != -1 {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        let rc = pyre_object::with_roots!(cls => unsafe { rmmap::c_fstat(real_fd, &mut st) });
        // `os.fstat` failure is ignored; the caller's length is trusted.
        if rc == 0 && st.st_mode as libc::mode_t & libc::S_IFMT == libc::S_IFREG {
            let file_size = usize::try_from(st.st_size).unwrap_or(usize::MAX);
            let offset_usize = usize::try_from(offset).unwrap_or(usize::MAX);
            if length == 0 {
                if file_size == 0 {
                    return Err(pyre_interpreter::PyError::value_error(
                        "cannot mmap an empty file",
                    ));
                }
                // `rmmap.mmap` rejects only `offset > size`, which leaves an
                // offset landing exactly on EOF asking `mmap(2)` for a
                // zero-length mapping. `new_mmap_object` refuses
                // `offset >= st_size`. Keep that ValueError.
                if offset_usize >= file_size {
                    return Err(pyre_interpreter::PyError::value_error(
                        "mmap offset is greater than file size",
                    ));
                }
                length = file_size - offset_usize;
            } else if offset_usize.saturating_add(length) > file_size {
                return Err(pyre_interpreter::PyError::value_error(
                    "mmap length is greater than file size",
                ));
            }
        }
    }
    // `fileno == -1` maps anonymous memory and ORs `MAP_ANONYMOUS` onto the
    // caller's flags. The fd passed to `c_mmap` stays the original one;
    // `os.dup` is only what the object keeps.
    let mut owned_fd = None;
    if real_fd == -1 {
        flags |= rmmap::MAP_ANONYMOUS;
    } else {
        let duped = pyre_object::with_roots!(cls => unsafe { rmmap::c_dup(real_fd) });
        if duped < 0 {
            return Err(pyre_interpreter::PyError::os_error_with_errno(
                majit_rlib::rposix::get_saved_errno(),
                "dup",
            ));
        }
        owned_fd = Some(OwnedFd(duped));
    }
    let mode = if prot & rmmap::PROT_WRITE != 0 {
        if flags & rmmap::MAP_PRIVATE != 0 {
            MMAP_ACCESS_COPY
        } else {
            MMAP_ACCESS_WRITE
        }
    } else {
        MMAP_ACCESS_READ
    };
    let ptr = pyre_object::with_roots!(cls => unsafe {
        rmmap::c_mmap(
            std::ptr::null_mut(),
            length,
            prot,
            flags,
            real_fd,
            offset as libc::off_t,
        )
    });
    if ptr == libc::MAP_FAILED {
        return Err(pyre_interpreter::PyError::os_error_with_errno(
            majit_rlib::rposix::get_saved_errno(),
            "mmap",
        ));
    }
    let stored_fd = if trackfd { owned_fd } else { None };
    Ok(mmap_new_object(
        cls,
        NativeMMap {
            mapped: Some(MappedObj::Mapped(PosixMap {
                ptr: ptr.cast(),
                len: length,
            })),
            fd: stored_fd,
            flags,
            trackfd,
        },
        access,
        mode,
        offset,
    ))
}

// `interp_mmap.py mmap(fileno, length, tagname, access, offset)` —
// the Windows constructor names a mapping object where POSIX passes
// flags/prot, and the mapping's protection comes from `access` alone
// (`rmmap.py`).
#[cfg(windows)]
fn mmap_construct(
    cls: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let bound = pyre_interpreter::builtins::bind_builtin_kwargs(
        args,
        &["fileno", "length", "tagname", "access", "offset"],
        &[true, true, false, false, false],
        "mmap",
    )?;
    let index_i64 = |obj: pyre_object::PyObjectRef, label: &str| {
        if obj.is_null() {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "mmap() {label} must be an integer"
            )));
        }
        pyre_interpreter::baseobjspace::int_w(pyre_interpreter::baseobjspace::space_index(obj)?)
    };
    let fileno = index_i64(bound[0], "fileno")? as i32;
    let length = index_i64(bound[1], "length")?;
    // `rmmap.py _check_map_size`.
    if length < 0 {
        return Err(pyre_interpreter::PyError::type_error(
            "memory mapped size must be positive",
        ));
    }
    let tagname = match bound[2] {
        a if a.is_null() || unsafe { pyre_object::is_none(a) } => "",
        a => {
            if !unsafe { pyre_object::is_str(a) } {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "expected str or None for 'tagname', not {}",
                    pyre_interpreter::type_methods::arg_type_name(a)
                )));
            }
            // The name reaches Win32 as UTF-16, which host_env builds from a
            // `&str`, so a tag carrying a lone surrogate has no route through.
            let Some(tag) = (unsafe { pyre_object::w_str_get_value_opt(a) }) else {
                return Err(pyre_interpreter::PyError::value_error(
                    "mmap() tagname must not contain lone surrogates",
                ));
            };
            if tag.contains('\0') {
                return Err(pyre_interpreter::PyError::value_error(
                    "embedded null character",
                ));
            }
            tag
        }
    };
    let access = if bound[3].is_null() {
        MMAP_ACCESS_DEFAULT
    } else {
        index_i64(bound[3], "access")?
    };
    if !(MMAP_ACCESS_DEFAULT..=MMAP_ACCESS_COPY).contains(&access) {
        return Err(pyre_interpreter::PyError::value_error(
            "mmap invalid access parameter.",
        ));
    }
    let offset = if bound[4].is_null() {
        0
    } else {
        index_i64(bound[4], "offset")?
    };
    // `rmmap.py:897-898`.
    if offset < 0 {
        return Err(pyre_interpreter::PyError::value_error("negative offset"));
    }
    let mut map_size = length as usize;

    // `rmmap.py:916-918` — "assume -1 and 0 both mean invalid file descriptor
    // to 'anonymously' map memory".  The handle is duplicated so Python code
    // may close the file it came from while the mapping lives on; the guard
    // hands it back to the OS on every path that does not reach the object.
    let mut file_handle = None;
    if fileno != -1 && fileno != 0 {
        let fh = rustpython_host_env::nt::handle_from_fd(fileno);
        if host_mmap::is_invalid_handle_value(fh as isize) {
            return Err(pyre_interpreter::PyError::os_error_with_errno(
                libc::EBADF,
                "mmap: bad file descriptor",
            ));
        }
        let guard = MmapHandleGuard(
            host_mmap::duplicate_handle(fh).map_err(|e| mmap_io_err(e, "DuplicateHandle"))?,
        );
        // `rmmap.py` trusts map_size when the size cannot be read
        // (a non-seeking file); only a readable size constrains it.
        if let Ok(file_len) = host_mmap::get_file_len(fh) {
            if map_size == 0 {
                if file_len == 0 {
                    return Err(pyre_interpreter::PyError::value_error(
                        "cannot mmap an empty file",
                    ));
                }
                // `rmmap.py:943-945` rejects only offset > size, so an
                // offset landing exactly on EOF resolves to a zero-length
                // mapping and asks `MapViewOfFile` for an empty view, which
                // fails with a Win32 error instead.  `mmapmodule.c
                // new_mmap_object` refuses the offset one step earlier, `if
                // (offset >= size)`, in its Windows size block (`:2087`; the
                // POSIX one spells the same test at `:1872`) — read at v3.14.6
                // in the checkout at Z:/cpython.
                if offset >= file_len {
                    return Err(pyre_interpreter::PyError::value_error(
                        "mmap offset is greater than file size",
                    ));
                }
                let remaining = file_len - offset;
                map_size = remaining as usize;
                if map_size as i64 != remaining {
                    return Err(pyre_interpreter::PyError::value_error(
                        "mmap length is too large",
                    ));
                }
            }
            // A given length is not measured against the file at all here.
            // `CreateFileMapping` is asked for `offset + length` and grows the
            // file to it, so a length past EOF is how the mapping is extended
            // on this platform.  `rmmap.py` rejects it with "mmap
            // length is greater than file size" instead; the run fails on that
            // at `lib-python/3/test/test_mmap.py`, which calls
            // `mmap(fileno(), mapsize + 1)` and, on `sys.platform` starting
            // with "win", turns a ValueError into
            // `self.fail("Opening mmap with size+1 should work on Windows.")`.
            // `pypy/module/mmap/test/test_mmap.py:747-749` concedes the same
            // point from the other side, skipping its own
            // `raises(ValueError, ...)` under `os.name == "nt"` with the note
            // "this should work under windows".
        }
        file_handle = Some(guard);
    }

    let handle = file_handle
        .as_ref()
        .map_or(host_mmap::INVALID_HANDLE, |guard| guard.0);
    // `rmmap.py:998-1000` reaches every file-backed mapping through one
    // `CreateFileMapping(m.file_handle, NULL, flProtect, size_hi, size_lo,
    // m.tagname)`, where the maximum size is `offset + map_size` and the name
    // is whatever `tagname` holds — the empty string when none was given.  A
    // zero-length name is not a name: two mappings created with one do not
    // report ERROR_ALREADY_EXISTS to each other, so the untagged mapping needs
    // no second route.  memmap2 is the wrong one to give it, because it asks
    // for a maximum size of 0, which means "the file as it stands" and cannot
    // extend it, and because it rounds the offset down to the allocation
    // granularity instead of letting `MapViewOfFile` reject an unaligned one.
    let (mapped, owned_handle) = if file_handle.is_some() || !tagname.is_empty() {
        let named = host_mmap::create_named_mapping(
            handle,
            tagname,
            mmap_access_mode(access),
            offset,
            map_size,
        )
        .map_err(|e| mmap_io_err(e, "CreateFileMapping"))?;
        // The mapping object holds its own reference to the file, but the
        // handle stays with the mmap so `size()` can still stat that file.
        (
            MappedObj::Named(named),
            file_handle.map_or(host_mmap::INVALID_HANDLE, |guard| guard.release()),
        )
    } else {
        let mapped = host_mmap::map_anon(map_size).map_err(|e| mmap_io_err(e, "mmap"))?;
        (MappedObj::Mapped(mapped), host_mmap::INVALID_HANDLE)
    };
    Ok(mmap_new_object(
        cls,
        NativeMMap {
            mapped: Some(mapped),
            handle: owned_handle,
            tagname: tagname.to_owned(),
            // No `trackfd` parameter on Windows; the object always owns the
            // handle it duplicated, so `resize()` is never disarmed.
            trackfd: true,
        },
        access,
        mmap_access_mode(access) as i64,
        offset,
    ))
}

/// Closes the duplicated file handle unless it is released into the finished
/// mmap object, so a constructor that fails after `DuplicateHandle` does not
/// keep the file open.
#[cfg(windows)]
struct MmapHandleGuard(host_mmap::Handle);

#[cfg(windows)]
impl MmapHandleGuard {
    fn release(mut self) -> host_mmap::Handle {
        let handle = self.0;
        self.0 = host_mmap::INVALID_HANDLE;
        handle
    }
}

#[cfg(windows)]
impl Drop for MmapHandleGuard {
    fn drop(&mut self) {
        if !host_mmap::is_invalid_handle_value(self.0 as isize) {
            host_mmap::close_handle(self.0);
        }
    }
}

pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    #[cfg(any(unix, windows))]
    {
        // `interp_mmap.py:42 error = OSError` alias.
        let w_os_error = pyre_interpreter::builtins::lookup_exc_class("OSError")
            .expect("OSError must be installed before init_mmap");
        pyre_interpreter::__pyre_put_new!(ns_slot, "error", w_os_error);

        #[cfg(unix)]
        register_posix_constants(pyre_object::gc_roots::shadow_stack_get(ns_slot));

        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "ACCESS_DEFAULT",
            pyre_object::w_int_new(MMAP_ACCESS_DEFAULT)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "ACCESS_READ",
            pyre_object::w_int_new(MMAP_ACCESS_READ)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "ACCESS_WRITE",
            pyre_object::w_int_new(MMAP_ACCESS_WRITE)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "ACCESS_COPY",
            pyre_object::w_int_new(MMAP_ACCESS_COPY)
        );

        // `getpagesize` / `GetSystemInfo`. POSIX allocation granularity is the
        // page size; Windows' is `SYSTEM_INFO.dwAllocationGranularity`.
        #[cfg(unix)]
        let (page, gran) = {
            let page = rmmap::page_size() as i64;
            (page, page)
        };
        #[cfg(windows)]
        let (page, gran) = (
            rustpython_host_env::os::page_size() as i64,
            rustpython_host_env::os::alloc_granularity() as i64,
        );
        pyre_interpreter::__pyre_put_new!(ns_slot, "PAGESIZE", pyre_object::w_int_new(page));
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "ALLOCATIONGRANULARITY",
            pyre_object::w_int_new(gran)
        );

        // Register the type itself.
        pyre_interpreter::__pyre_put_new!(ns_slot, "mmap", mmap_type());
    }
    Ok(())
}

/// `rmmap.py` — the mapping flags and advice values a POSIX `mmap(2)`
/// takes. The portable subset and the Darwin literals `rmmap` publishes come
/// from there; MAP_FIXED and the Linux-only MAP_* flags stay on libc.
/// Windows has none of them: the mapping's protection comes from `access`
/// alone there, and its module carries only the ACCESS_* and page constants.
#[cfg(unix)]
fn register_posix_constants(ns: pyre_object::PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "MAP_SHARED",
            pyre_object::w_int_new(rmmap::MAP_SHARED as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "MAP_PRIVATE",
            pyre_object::w_int_new(rmmap::MAP_PRIVATE as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "MAP_ANON",
            pyre_object::w_int_new(rmmap::MAP_ANON as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "MAP_ANONYMOUS",
            pyre_object::w_int_new(rmmap::MAP_ANONYMOUS as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "MAP_FIXED",
            pyre_object::w_int_new(libc::MAP_FIXED as i64)
        );
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            pyre_interpreter::__pyre_put_new!(
                ns_slot,
                "MAP_POPULATE",
                pyre_object::w_int_new(libc::MAP_POPULATE as i64)
            );
            pyre_interpreter::__pyre_put_new!(
                ns_slot,
                "MAP_STACK",
                pyre_object::w_int_new(libc::MAP_STACK as i64)
            );
            pyre_interpreter::__pyre_put_new!(
                ns_slot,
                "MAP_HUGETLB",
                pyre_object::w_int_new(libc::MAP_HUGETLB as i64)
            );
            pyre_interpreter::__pyre_put_new!(
                ns_slot,
                "MAP_NORESERVE",
                pyre_object::w_int_new(libc::MAP_NORESERVE as i64)
            );
            pyre_interpreter::__pyre_put_new!(
                ns_slot,
                "MAP_LOCKED",
                pyre_object::w_int_new(libc::MAP_LOCKED as i64)
            );
            pyre_interpreter::__pyre_put_new!(
                ns_slot,
                "MAP_NONBLOCK",
                pyre_object::w_int_new(libc::MAP_NONBLOCK as i64)
            );
        }
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "PROT_READ",
            pyre_object::w_int_new(rmmap::PROT_READ as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "PROT_WRITE",
            pyre_object::w_int_new(rmmap::PROT_WRITE as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "PROT_EXEC",
            pyre_object::w_int_new(rmmap::PROT_EXEC as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "MADV_NORMAL",
            pyre_object::w_int_new(rmmap::MADV_NORMAL as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "MADV_RANDOM",
            pyre_object::w_int_new(rmmap::MADV_RANDOM as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "MADV_SEQUENTIAL",
            pyre_object::w_int_new(rmmap::MADV_SEQUENTIAL as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "MADV_WILLNEED",
            pyre_object::w_int_new(rmmap::MADV_WILLNEED as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "MADV_DONTNEED",
            pyre_object::w_int_new(rmmap::MADV_DONTNEED as i64)
        );
        // The `MAP_*` / `MADV_*` names `<sys/mman.h>` defines only on darwin.
        // `rmmap.py` reaches them through `DefinedConstantInteger`, which
        // yields None wherever the header is silent; the cfg does the same
        // job.
        #[cfg(target_vendor = "apple")]
        {
            macro_rules! cst {
                ($name:literal, $val:expr) => {
                    pyre_interpreter::__pyre_put_new!(
                        ns_slot,
                        $name,
                        pyre_object::w_int_new($val as i64)
                    );
                };
            }
            cst!("MADV_FREE", rmmap::MADV_FREE);
            cst!("MADV_FREE_REUSABLE", libc::MADV_FREE_REUSABLE);
            cst!("MADV_FREE_REUSE", libc::MADV_FREE_REUSE);
            cst!("MAP_32BIT", rmmap::MAP_32BIT);
            cst!("MAP_HASSEMAPHORE", rmmap::MAP_HASSEMAPHORE);
            cst!("MAP_JIT", rmmap::MAP_JIT);
            cst!("MAP_NOCACHE", rmmap::MAP_NOCACHE);
            cst!("MAP_NOEXTEND", rmmap::MAP_NOEXTEND);
            cst!("MAP_NORESERVE", rmmap::MAP_NORESERVE);
            cst!("MAP_RESILIENT_CODESIGN", rmmap::MAP_RESILIENT_CODESIGN);
            cst!("MAP_RESILIENT_MEDIA", rmmap::MAP_RESILIENT_MEDIA);
            cst!("MAP_TPRO", rmmap::MAP_TPRO);
            cst!(
                "MAP_TRANSLATED_ALLOW_EXECUTE",
                rmmap::MAP_TRANSLATED_ALLOW_EXECUTE
            );
            cst!("MAP_UNIX03", rmmap::MAP_UNIX03);
        }
    }
}
