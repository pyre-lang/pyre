//! Windows overlapped I/O — PyPy: `lib_pypy/_overlapped.py`.
//!
//! PyPy keeps the `OVERLAPPED`, native buffers, operation tag and handle on
//! each `Overlapped` instance.  This port preserves that ownership exactly:
//! the W_Root owns one stable native state box and its writable-buffer lease.
//! The actual Win32/WinSock calls are the shared, pinned
//! `rustpython_host_env::overlapped` implementation.

use parking_lot::Mutex;
use pyre_object::{PY_NULL, PyObject, PyObjectRef};
use rustpython_host_env::{
    overlapped as host_overlapped, winapi as host_winapi, windows as host_windows,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OverlappedType {
    None,
    NotStarted,
    Read,
    ReadInto,
    Write,
    Accept,
    Connect,
    Disconnect,
    ConnectNamedPipe,
    TransmitFile,
    ReadFrom,
    WriteTo,
    ReadFromInto,
}

struct NativeOverlapped {
    overlapped: host_overlapped::OverlappedIo,
    handle: host_overlapped::Handle,
    error: u32,
    kind: OverlappedType,
    buffer: Vec<u8>,
    address: Vec<u8>,
    recv_address: host_overlapped::SocketAddrV6,
    recv_address_length: i32,
    buffer_export_held: bool,
}

// The OS may complete an operation on a Windows worker thread.  Access from
// Python remains serialized through the per-object mutex; the raw pointers
// handed to Windows all name fields/buffers owned by the stable Box.
unsafe impl Send for NativeOverlapped {}

// CPython 3.14 Modules/overlapped.c creates overlapped_type_spec through
// PyType_FromModuleAndSpec; its spec is immutable.
#[pyre_interpreter::pyre_class("_overlapped.Overlapped", cpython_heaptype)]
#[derive(Default)]
pub struct W_Overlapped {
    backend: *mut Mutex<NativeOverlapped>,
    /// CPython's/PyPy's retained Py_buffer owner for Read*Into.  Keeping it
    /// inline makes it an ordinary traced edge, not a process side table.
    w_buffer: PyObjectRef,
    /// Concrete exporter whose resize lock was acquired.
    w_buffer_owner: PyObjectRef,
    /// Cached `(buffer, address)` / `(count, address)` result for recvfrom.
    w_result: PyObjectRef,
}

fn this(obj: PyObjectRef) -> Result<&'static mut W_Overlapped, pyre_interpreter::PyError> {
    W_Overlapped::from_obj(obj).ok_or_else(|| {
        pyre_interpreter::PyError::type_error("expected _overlapped.Overlapped object")
    })
}

fn native(obj: PyObjectRef) -> Result<&'static Mutex<NativeOverlapped>, pyre_interpreter::PyError> {
    let this = this(obj)?;
    if this.backend.is_null() {
        return Err(pyre_interpreter::PyError::value_error(
            "invalid overlapped object",
        ));
    }
    Ok(unsafe { &*this.backend })
}

fn arg(
    args: &[PyObjectRef],
    index: usize,
    name: &str,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    args.get(index).copied().ok_or_else(|| {
        pyre_interpreter::PyError::type_error(format!("{name}() missing required argument"))
    })
}

fn method_arity(
    args: &[PyObjectRef],
    name: &str,
    min: usize,
    max: usize,
) -> Result<(), pyre_interpreter::PyError> {
    if (min..=max).contains(&args.len()) {
        Ok(())
    } else if min == max {
        Err(pyre_interpreter::PyError::type_error(format!(
            "{name}() takes exactly {} argument{}",
            min - 1,
            if min == 2 { "" } else { "s" }
        )))
    } else {
        Err(pyre_interpreter::PyError::type_error(format!(
            "{name}() takes from {} to {} arguments",
            min - 1,
            max - 1
        )))
    }
}

/// The `TypeError` `PyLong_AsVoidPtr` raises for something that is not an
/// integer at all, checked before the conversion so that whatever an
/// `__index__` of its own raises still comes through unchanged.
fn require_int(obj: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    if !unsafe { pyre_object::pyobject::is_int_or_long(obj) }
        && unsafe { pyre_interpreter::baseobjspace::lookup(obj, "__index__") }.is_none()
    {
        return Err(pyre_interpreter::PyError::type_error(
            "an integer is required",
        ));
    }
    Ok(())
}

/// A handle argument (`_Py_PARSE_UINTPTR`): the value modulo the pointer
/// width, so a handle may be written as the negative it is or as the unsigned
/// value it prints as.
fn handle_w(obj: PyObjectRef) -> Result<host_overlapped::Handle, pyre_interpreter::PyError> {
    require_int(obj)?;
    Ok(pyre_interpreter::baseobjspace::truncatedint_w(obj)? as isize as host_overlapped::Handle)
}

/// The integer a pointer-sized value comes back as (`PyLong_FromVoidPtr`):
/// the unsigned value, which is how a pseudo handle and a completion key that
/// is one print.
fn w_uintptr(value: usize) -> PyObjectRef {
    match i64::try_from(value as u64) {
        Ok(fits) => pyre_object::w_int_new(fits),
        Err(_) => pyre_object::longobject::w_long_new(majit_rlib::rbigint::RBigInt::from(
            value as u64 as i128,
        )),
    }
}

fn isize_w(obj: PyObjectRef) -> Result<isize, pyre_interpreter::PyError> {
    require_int(obj)?;
    Ok(pyre_interpreter::baseobjspace::truncatedint_w(obj)? as isize)
}

fn usize_w(obj: PyObjectRef) -> Result<usize, pyre_interpreter::PyError> {
    require_int(obj)?;
    Ok(pyre_interpreter::baseobjspace::truncatedint_w(obj)? as usize)
}

fn u32_w(obj: PyObjectRef) -> Result<u32, pyre_interpreter::PyError> {
    pyre_interpreter::baseobjspace::c_uint_w(obj)
}

/// `_winapi.py RaiseFromWindowsErr`: the two connection failures raise their
/// own OSError subclasses, which the errno the code maps to would not pick.
fn win32_err_code(code: u32) -> pyre_interpreter::PyError {
    use windows_sys::Win32::Foundation::{ERROR_CONNECTION_ABORTED, ERROR_CONNECTION_REFUSED};
    let class_name = match code {
        ERROR_CONNECTION_REFUSED => "ConnectionRefusedError",
        ERROR_CONNECTION_ABORTED => "ConnectionAbortedError",
        _ => {
            return pyre_interpreter::PyError::os_error_win32_syscall2(
                code as i32,
                PY_NULL,
                PY_NULL,
            );
        }
    };
    pyre_interpreter::PyError::os_error_win32_with_class(class_name, code as i32)
}

fn win32_err(err: std::io::Error) -> pyre_interpreter::PyError {
    win32_err_code(err.raw_os_error().unwrap_or(0) as u32)
}

fn set_object_refs(
    obj: PyObjectRef,
    w_buffer: PyObjectRef,
    w_owner: PyObjectRef,
    w_result: PyObjectRef,
) -> Result<(), pyre_interpreter::PyError> {
    let this = this(obj)?;
    this.w_buffer = w_buffer;
    this.w_buffer_owner = w_owner;
    this.w_result = w_result;
    pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF);
    Ok(())
}

/// PyPy `ffi.from_buffer(bufobj)` / CPython `PyBUF_WRITABLE`: resolve the
/// contiguous writable window and the concrete exporter whose resize lock is
/// held until the Overlapped object is swept.
fn writable_buffer(
    obj: PyObjectRef,
) -> Result<(&'static mut [u8], PyObjectRef), pyre_interpreter::PyError> {
    if unsafe { pyre_object::bytearrayobject::is_bytearray(obj) } {
        return Ok((
            unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(obj) },
            obj,
        ));
    }
    if unsafe { pyre_object::interp_array::is_array(obj) } {
        return Ok((
            unsafe { pyre_object::interp_array::w_array_vec_mut(obj).as_mut_slice() },
            obj,
        ));
    }
    if unsafe { pyre_object::memoryview::is_w_memoryview(obj) } {
        unsafe { pyre_interpreter::builtins::memoryview_check_released(obj) }?;
        if unsafe { pyre_object::memoryview::w_memoryview_readonly(obj) }
            || !unsafe { pyre_interpreter::builtins::memoryview_contiguity(obj).0 }
        {
            return Err(pyre_interpreter::PyError::new(
                pyre_interpreter::PyErrorKind::BufferError,
                "buffer is not contiguous and writable",
            ));
        }
        let view = unsafe { pyre_object::memoryview::w_memoryview_view(obj) };
        let Some(full) = (unsafe { view.backing().as_bytes_mut() }) else {
            return Err(pyre_interpreter::PyError::new(
                pyre_interpreter::PyErrorKind::BufferError,
                "buffer is not writable",
            ));
        };
        let offset = unsafe { view.offset() } as usize;
        let length = unsafe { pyre_object::memoryview::w_memoryview_length(obj) } as usize;
        if offset
            .checked_add(length)
            .is_none_or(|end| end > full.len())
        {
            return Err(pyre_interpreter::PyError::value_error(
                "memoryview buffer is no longer valid",
            ));
        }
        return Ok((&mut full[offset..offset + length], obj));
    }
    Err(pyre_interpreter::PyError::type_error(
        "a writable bytes-like object is required",
    ))
}

fn retain_writable_buffer(
    obj: PyObjectRef,
    w_buffer: PyObjectRef,
) -> Result<usize, pyre_interpreter::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_roots(&[obj, w_buffer]);
    let base = pyre_object::gc_roots::shadow_stack_len() - 2;
    let r_buffer = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let (slice, owner) = writable_buffer(r_buffer)?;
    let length = slice.len();
    let _ = pyre_object::gc_roots::pin_root(owner);
    let owner_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let held = unsafe {
        pyre_interpreter::builtins::buffer_export_incref(pyre_object::gc_roots::shadow_stack_get(
            owner_slot,
        ))
    };
    let r_obj = pyre_object::gc_roots::shadow_stack_get(base);
    let r_buffer = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let r_owner = pyre_object::gc_roots::shadow_stack_get(owner_slot);
    set_object_refs(r_obj, r_buffer, r_owner, PY_NULL)?;
    let mut state = native(r_obj)?.lock();
    state.buffer_export_held = held;
    drop(roots);
    Ok(length)
}

fn release_writable_buffer(obj: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let owner = this(obj)?.w_buffer_owner;
    let mut state = native(obj)?.lock();
    if state.buffer_export_held && !owner.is_null() {
        unsafe { pyre_interpreter::builtins::buffer_export_decref(owner) };
        state.buffer_export_held = false;
    }
    drop(state);
    set_object_refs(obj, PY_NULL, PY_NULL, PY_NULL)
}

fn copied_read_buffer(obj: PyObjectRef, name: &str) -> Result<Vec<u8>, pyre_interpreter::PyError> {
    let Some(buffer) = pyre_interpreter::baseobjspace::simple_buffer_bytes(obj)? else {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{name}() argument must be a bytes-like object"
        )));
    };
    let bytes = buffer.as_bytes().to_vec();
    buffer.release();
    Ok(bytes)
}

fn parse_address(obj: PyObjectRef) -> Result<(Vec<u8>, i32), pyre_interpreter::PyError> {
    let items = pyre_interpreter::baseobjspace::unpackiterable(obj, -1)?;
    match items.as_slice() {
        [host, port] => {
            let host = pyre_interpreter::baseobjspace::text_w(*host)?;
            let port = u32_w(*port)?;
            let port = u16::try_from(port)
                .map_err(|_| pyre_interpreter::PyError::overflow_error("port must be 0-65535"))?;
            host_overlapped::parse_address_v4(host, port).map_err(win32_err)
        }
        [host, port, flowinfo, scope_id] => {
            let host = pyre_interpreter::baseobjspace::text_w(*host)?;
            let port = u32_w(*port)?;
            let port = u16::try_from(port)
                .map_err(|_| pyre_interpreter::PyError::overflow_error("port must be 0-65535"))?;
            host_overlapped::parse_address_v6(host, port, u32_w(*flowinfo)?, u32_w(*scope_id)?)
                .map_err(win32_err)
        }
        _ => Err(pyre_interpreter::PyError::value_error(
            "expected tuple of length 2 or 4",
        )),
    }
}

fn unparse_address(
    address: &host_overlapped::SocketAddrV6,
    length: i32,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    match host_overlapped::unparse_address(
        address as *const _ as *const host_overlapped::SocketAddrRaw,
        length,
    )
    .map_err(|_| {
        pyre_interpreter::PyError::value_error("recvfrom returned unsupported address family")
    })? {
        host_overlapped::SocketAddress::V4 { host, port } => {
            let mut fields = pyre_object::gc_roots::RootedItems::new();
            fields.push(pyre_object::w_str_new_managed(&host));
            fields.push(pyre_object::w_int_new(port as i64));
            Ok(pyre_object::w_tuple_new(fields.take()))
        }
        host_overlapped::SocketAddress::V6 {
            host,
            port,
            flowinfo,
            scope_id,
        } => {
            let mut fields = pyre_object::gc_roots::RootedItems::new();
            fields.push(pyre_object::w_str_new_managed(&host));
            fields.push(pyre_object::w_int_new(port as i64));
            fields.push(pyre_object::w_int_new(flowinfo as i64));
            fields.push(pyre_object::w_int_new(scope_id as i64));
            Ok(pyre_object::w_tuple_new(fields.take()))
        }
    }
}

fn operation_already_attempted(state: &NativeOverlapped) -> Result<(), pyre_interpreter::PyError> {
    if state.kind != OverlappedType::None {
        Err(pyre_interpreter::PyError::value_error(
            "operation already attempted",
        ))
    } else {
        Ok(())
    }
}

fn accept_read_start_error(
    state: &mut NativeOverlapped,
    error: u32,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    use host_winapi::{ERROR_BROKEN_PIPE, ERROR_IO_PENDING, ERROR_MORE_DATA, ERROR_SUCCESS};
    state.error = error;
    match error {
        ERROR_BROKEN_PIPE => {
            host_overlapped::mark_as_completed(&mut state.overlapped);
            Err(win32_err_code(error))
        }
        ERROR_SUCCESS | ERROR_MORE_DATA | ERROR_IO_PENDING => Ok(pyre_object::w_none()),
        _ => {
            state.kind = OverlappedType::NotStarted;
            Err(win32_err_code(error))
        }
    }
}

fn accept_write_start_error(
    state: &mut NativeOverlapped,
    error: u32,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    state.error = error;
    match error {
        host_winapi::ERROR_SUCCESS | host_winapi::ERROR_IO_PENDING => Ok(pyre_object::w_none()),
        _ => {
            state.kind = OverlappedType::NotStarted;
            Err(win32_err_code(error))
        }
    }
}

fn overlapped_new(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let cls = args.first().copied().unwrap_or_else(overlapped_type);
    if args.len() > 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "Overlapped() takes at most one argument",
        ));
    }
    let mut event = match args.get(1).copied() {
        Some(w) => isize_w(w)?,
        None => host_overlapped::INVALID_HANDLE_VALUE_ISIZE,
    };
    if event == host_overlapped::INVALID_HANDLE_VALUE_ISIZE {
        event = host_winapi::create_event_w(true, false, None).map_err(win32_err)? as isize;
    }
    let mut overlapped: host_overlapped::OverlappedIo = unsafe { core::mem::zeroed() };
    if event != 0 {
        overlapped.hEvent = event as host_overlapped::Handle;
    }
    let backend = Box::into_raw(Box::new(Mutex::new(NativeOverlapped {
        overlapped,
        handle: std::ptr::null_mut(),
        error: 0,
        kind: OverlappedType::None,
        buffer: Vec::new(),
        address: Vec::new(),
        recv_address: unsafe { core::mem::zeroed() },
        recv_address_length: 0,
        buffer_export_held: false,
    })));
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let obj = W_Overlapped::allocate_stable(W_Overlapped {
        ob: PyObject::default(),
        backend,
        w_buffer: PY_NULL,
        w_buffer_owner: PY_NULL,
        w_result: PY_NULL,
    });
    Ok(pyre_interpreter::typedef::tag_subclass_instance(
        obj,
        unsafe { pyre_object::gc_roots::shadow_stack_get(cls_slot) },
    ))
}

fn overlapped_cancel(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "cancel", 1, 1)?;
    let obj = arg(args, 0, "cancel")?;
    let state = native(obj)?.lock();
    if matches!(state.kind, OverlappedType::NotStarted) {
        return Ok(pyre_object::w_none());
    }
    if !host_overlapped::has_overlapped_io_completed(&state.overlapped) {
        host_overlapped::cancel_overlapped(state.handle, &state.overlapped).map_err(win32_err)?;
    }
    Ok(pyre_object::w_none())
}

fn overlapped_getresult(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    use host_winapi::{ERROR_BROKEN_PIPE, ERROR_MORE_DATA, ERROR_SUCCESS};
    method_arity(args, "getresult", 1, 2)?;
    let obj = arg(args, 0, "getresult")?;
    let wait = match args.get(1).copied() {
        Some(w) => pyre_interpreter::baseobjspace::is_true(w)?,
        None => false,
    };
    let record = native(obj)?;
    // The wait runs with the GIL released, as `Overlapped_getresult` wraps
    // `GetOverlappedResult` in `Py_BEGIN_ALLOW_THREADS`, and with the record's
    // lock dropped: `cancel`, the one call that can end the operation early,
    // takes that lock.  The record is owned by the object and never moves.
    let (handle, overlapped) = {
        let state = record.lock();
        match state.kind {
            OverlappedType::None => {
                return Err(pyre_interpreter::PyError::value_error(
                    "operation not yet attempted",
                ));
            }
            OverlappedType::NotStarted => {
                return Err(pyre_interpreter::PyError::value_error(
                    "operation failed to start",
                ));
            }
            _ => {}
        }
        (state.handle, std::ptr::addr_of!(state.overlapped))
    };
    let result = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        host_overlapped::get_overlapped_result(handle, unsafe { &*overlapped }, wait)
    };
    let mut state = record.lock();
    let transferred = result.transferred as usize;
    state.error = result.error;
    let broken_pipe_ok = matches!(
        state.kind,
        OverlappedType::Read
            | OverlappedType::ReadInto
            | OverlappedType::ReadFrom
            | OverlappedType::ReadFromInto
    );
    match result.error {
        ERROR_SUCCESS | ERROR_MORE_DATA => {}
        ERROR_BROKEN_PIPE if broken_pipe_ok => {}
        error => return Err(win32_err_code(error)),
    }

    match state.kind {
        OverlappedType::Read => {
            let count = transferred.min(state.buffer.len());
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(
                &state.buffer[..count],
            ))
        }
        OverlappedType::ReadInto => {
            let count = transferred.min(state.buffer.len());
            let destination = writable_buffer(this(obj)?.w_buffer)?.0;
            let count = count.min(destination.len());
            destination[..count].copy_from_slice(&state.buffer[..count]);
            Ok(pyre_object::w_int_new(count as i64))
        }
        OverlappedType::ReadFrom | OverlappedType::ReadFromInto => {
            let cached = this(obj)?.w_result;
            if !cached.is_null() {
                return Ok(cached);
            }
            let address = unparse_address(&state.recv_address, state.recv_address_length)?;
            let first = if state.kind == OverlappedType::ReadFrom {
                let count = transferred.min(state.buffer.len());
                pyre_object::bytesobject::w_bytes_from_bytes(&state.buffer[..count])
            } else {
                let count = transferred.min(state.buffer.len());
                let destination = writable_buffer(this(obj)?.w_buffer)?.0;
                let count = count.min(destination.len());
                destination[..count].copy_from_slice(&state.buffer[..count]);
                pyre_object::w_int_new(count as i64)
            };
            let mut fields = pyre_object::gc_roots::RootedItems::new();
            fields.push(first);
            fields.push(address);
            let result = pyre_object::w_tuple_new(fields.take());
            this(obj)?.w_result = result;
            pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF);
            Ok(result)
        }
        _ => Ok(pyre_object::w_int_new(transferred as i64)),
    }
}

fn overlapped_read_file(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "ReadFile", 3, 3)?;
    let obj = arg(args, 0, "ReadFile")?;
    let handle = handle_w(arg(args, 1, "ReadFile")?)?;
    let size = u32_w(arg(args, 2, "ReadFile")?)?;
    let mut state = native(obj)?.lock();
    operation_already_attempted(&state)?;
    state.handle = handle;
    state.kind = OverlappedType::Read;
    state.buffer = vec![0; usize::max(size as usize, 1)];
    let error = host_overlapped::start_read_file(
        handle,
        state.buffer.as_mut_ptr(),
        size,
        &mut state.overlapped,
    );
    accept_read_start_error(&mut state, error)
}

fn overlapped_read_file_into(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "ReadFileInto", 3, 3)?;
    let obj = arg(args, 0, "ReadFileInto")?;
    let handle = handle_w(arg(args, 1, "ReadFileInto")?)?;
    let w_buffer = arg(args, 2, "ReadFileInto")?;
    {
        let state = native(obj)?.lock();
        operation_already_attempted(&state)?;
    }
    let length = retain_writable_buffer(obj, w_buffer)?;
    let size = u32::try_from(length)
        .map_err(|_| pyre_interpreter::PyError::value_error("buffer too large"))?;
    let mut state = native(obj)?.lock();
    state.handle = handle;
    state.kind = OverlappedType::ReadInto;
    state.buffer = vec![0; usize::max(length, 1)];
    let error = host_overlapped::start_read_file(
        handle,
        state.buffer.as_mut_ptr(),
        size,
        &mut state.overlapped,
    );
    accept_read_start_error(&mut state, error)
}

fn overlapped_write_file(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "WriteFile", 3, 3)?;
    let obj = arg(args, 0, "WriteFile")?;
    let handle = handle_w(arg(args, 1, "WriteFile")?)?;
    let buffer = copied_read_buffer(arg(args, 2, "WriteFile")?, "WriteFile")?;
    let size = u32::try_from(buffer.len())
        .map_err(|_| pyre_interpreter::PyError::value_error("buffer too large"))?;
    let mut state = native(obj)?.lock();
    operation_already_attempted(&state)?;
    state.handle = handle;
    state.kind = OverlappedType::Write;
    state.buffer = buffer;
    let error = host_overlapped::start_write_file(
        handle,
        state.buffer.as_ptr(),
        size,
        &mut state.overlapped,
    );
    accept_write_start_error(&mut state, error)
}

fn overlapped_wsa_recv(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "WSARecv", 3, 4)?;
    let obj = arg(args, 0, "WSARecv")?;
    let handle = isize_w(arg(args, 1, "WSARecv")?)?;
    let size = u32_w(arg(args, 2, "WSARecv")?)?;
    let mut flags = match args.get(3).copied() {
        Some(w) => u32_w(w)?,
        None => 0,
    };
    let mut state = native(obj)?.lock();
    operation_already_attempted(&state)?;
    state.handle = handle as host_overlapped::Handle;
    state.kind = OverlappedType::Read;
    state.buffer = vec![0; usize::max(size as usize, 1)];
    let error = host_overlapped::start_wsa_recv(
        handle as usize,
        state.buffer.as_mut_ptr(),
        size,
        &mut flags,
        &mut state.overlapped,
    );
    accept_read_start_error(&mut state, error)
}

fn overlapped_wsa_recv_into(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "WSARecvInto", 4, 4)?;
    let obj = arg(args, 0, "WSARecvInto")?;
    let handle = isize_w(arg(args, 1, "WSARecvInto")?)?;
    let w_buffer = arg(args, 2, "WSARecvInto")?;
    let mut flags = u32_w(arg(args, 3, "WSARecvInto")?)?;
    {
        let state = native(obj)?.lock();
        operation_already_attempted(&state)?;
    }
    let length = retain_writable_buffer(obj, w_buffer)?;
    let size = u32::try_from(length)
        .map_err(|_| pyre_interpreter::PyError::value_error("buffer too large"))?;
    let mut state = native(obj)?.lock();
    state.handle = handle as host_overlapped::Handle;
    state.kind = OverlappedType::ReadInto;
    state.buffer = vec![0; usize::max(length, 1)];
    let error = host_overlapped::start_wsa_recv(
        handle as usize,
        state.buffer.as_mut_ptr(),
        size,
        &mut flags,
        &mut state.overlapped,
    );
    accept_read_start_error(&mut state, error)
}

fn overlapped_wsa_send(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "WSASend", 4, 4)?;
    let obj = arg(args, 0, "WSASend")?;
    let handle = isize_w(arg(args, 1, "WSASend")?)?;
    let buffer = copied_read_buffer(arg(args, 2, "WSASend")?, "WSASend")?;
    let flags = u32_w(arg(args, 3, "WSASend")?)?;
    let size = u32::try_from(buffer.len())
        .map_err(|_| pyre_interpreter::PyError::value_error("buffer too large"))?;
    let mut state = native(obj)?.lock();
    operation_already_attempted(&state)?;
    state.handle = handle as host_overlapped::Handle;
    state.kind = OverlappedType::Write;
    state.buffer = buffer;
    let error = host_overlapped::start_wsa_send(
        handle as usize,
        state.buffer.as_ptr(),
        size,
        flags,
        &mut state.overlapped,
    );
    accept_write_start_error(&mut state, error)
}

fn overlapped_accept_ex(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "AcceptEx", 3, 3)?;
    let obj = arg(args, 0, "AcceptEx")?;
    let listen = isize_w(arg(args, 1, "AcceptEx")?)?;
    let accept = isize_w(arg(args, 2, "AcceptEx")?)?;
    let mut state = native(obj)?.lock();
    operation_already_attempted(&state)?;
    let address_size = core::mem::size_of::<host_overlapped::SocketAddrV6>() + 16;
    state.handle = listen as host_overlapped::Handle;
    state.kind = OverlappedType::Accept;
    state.buffer = vec![0; address_size * 2];
    let error = host_overlapped::start_accept_ex(
        listen as usize,
        accept as usize,
        state.buffer.as_mut_ptr(),
        address_size as u32,
        &mut state.overlapped,
    );
    accept_write_start_error(&mut state, error)
}

fn overlapped_connect_ex(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "ConnectEx", 3, 3)?;
    let obj = arg(args, 0, "ConnectEx")?;
    let socket = isize_w(arg(args, 1, "ConnectEx")?)?;
    let (address, length) = parse_address(arg(args, 2, "ConnectEx")?)?;
    let mut state = native(obj)?.lock();
    operation_already_attempted(&state)?;
    state.handle = socket as host_overlapped::Handle;
    state.kind = OverlappedType::Connect;
    state.address = address;
    let error = host_overlapped::start_connect_ex(
        socket as usize,
        state.address.as_ptr() as *const host_overlapped::SocketAddrRaw,
        length,
        &mut state.overlapped,
    );
    accept_write_start_error(&mut state, error)
}

fn overlapped_disconnect_ex(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "DisconnectEx", 3, 3)?;
    let obj = arg(args, 0, "DisconnectEx")?;
    let socket = isize_w(arg(args, 1, "DisconnectEx")?)?;
    let flags = u32_w(arg(args, 2, "DisconnectEx")?)?;
    let mut state = native(obj)?.lock();
    operation_already_attempted(&state)?;
    state.handle = socket as host_overlapped::Handle;
    state.kind = OverlappedType::Disconnect;
    let error = host_overlapped::start_disconnect_ex(socket as usize, flags, &mut state.overlapped);
    accept_write_start_error(&mut state, error)
}

fn overlapped_transmit_file(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "TransmitFile", 8, 8)?;
    let obj = arg(args, 0, "TransmitFile")?;
    let socket = isize_w(arg(args, 1, "TransmitFile")?)?;
    let file = handle_w(arg(args, 2, "TransmitFile")?)?;
    let offset = u32_w(arg(args, 3, "TransmitFile")?)?;
    let offset_high = u32_w(arg(args, 4, "TransmitFile")?)?;
    let count = u32_w(arg(args, 5, "TransmitFile")?)?;
    let per_send = u32_w(arg(args, 6, "TransmitFile")?)?;
    let flags = u32_w(arg(args, 7, "TransmitFile")?)?;
    let mut state = native(obj)?.lock();
    operation_already_attempted(&state)?;
    state.handle = socket as host_overlapped::Handle;
    state.kind = OverlappedType::TransmitFile;
    let error = host_overlapped::start_transmit_file(
        socket as usize,
        file,
        count,
        per_send,
        flags,
        offset,
        offset_high,
        &mut state.overlapped,
    );
    accept_write_start_error(&mut state, error)
}

fn overlapped_connect_named_pipe(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "ConnectNamedPipe", 2, 2)?;
    let obj = arg(args, 0, "ConnectNamedPipe")?;
    let pipe = handle_w(arg(args, 1, "ConnectNamedPipe")?)?;
    let mut state = native(obj)?.lock();
    operation_already_attempted(&state)?;
    state.handle = pipe;
    state.kind = OverlappedType::ConnectNamedPipe;
    let error = host_overlapped::start_connect_named_pipe(pipe, &mut state.overlapped);
    state.error = error;
    match error {
        host_winapi::ERROR_PIPE_CONNECTED => {
            host_overlapped::mark_as_completed(&mut state.overlapped);
            Ok(pyre_object::w_bool_from(true))
        }
        host_winapi::ERROR_SUCCESS | host_winapi::ERROR_IO_PENDING => {
            Ok(pyre_object::w_bool_from(false))
        }
        _ => {
            state.kind = OverlappedType::NotStarted;
            Err(win32_err_code(error))
        }
    }
}

fn start_recv_from(
    obj: PyObjectRef,
    handle: isize,
    size: u32,
    mut flags: u32,
    kind: OverlappedType,
) -> pyre_interpreter::PyResult {
    let mut state = native(obj)?.lock();
    operation_already_attempted(&state)?;
    state.handle = handle as host_overlapped::Handle;
    state.kind = kind;
    state.buffer = vec![0; usize::max(size as usize, 1)];
    state.recv_address = unsafe { core::mem::zeroed() };
    state.recv_address_length = core::mem::size_of::<host_overlapped::SocketAddrV6>() as i32;
    let state_ptr: *mut NativeOverlapped = &mut *state;
    let error = unsafe {
        host_overlapped::start_wsa_recv_from(
            handle as usize,
            (*state_ptr).buffer.as_mut_ptr(),
            size,
            &mut flags,
            &mut (*state_ptr).recv_address as *mut _ as *mut host_overlapped::SocketAddrRaw,
            &mut (*state_ptr).recv_address_length,
            &mut (*state_ptr).overlapped,
        )
    };
    accept_read_start_error(&mut state, error)
}

fn overlapped_wsa_recv_from(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "WSARecvFrom", 3, 4)?;
    let obj = arg(args, 0, "WSARecvFrom")?;
    let handle = isize_w(arg(args, 1, "WSARecvFrom")?)?;
    let size = u32_w(arg(args, 2, "WSARecvFrom")?)?;
    let flags = match args.get(3).copied() {
        Some(w) => u32_w(w)?,
        None => 0,
    };
    start_recv_from(obj, handle, size, flags, OverlappedType::ReadFrom)
}

fn overlapped_wsa_recv_from_into(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "WSARecvFromInto", 4, 5)?;
    let obj = arg(args, 0, "WSARecvFromInto")?;
    let handle = isize_w(arg(args, 1, "WSARecvFromInto")?)?;
    let w_buffer = arg(args, 2, "WSARecvFromInto")?;
    let requested = u32_w(arg(args, 3, "WSARecvFromInto")?)?;
    let flags = match args.get(4).copied() {
        Some(w) => u32_w(w)?,
        None => 0,
    };
    {
        let state = native(obj)?.lock();
        operation_already_attempted(&state)?;
    }
    let length = retain_writable_buffer(obj, w_buffer)?;
    if requested as usize > length {
        release_writable_buffer(obj)?;
        return Err(pyre_interpreter::PyError::value_error("buffer too small"));
    }
    start_recv_from(obj, handle, requested, flags, OverlappedType::ReadFromInto)
}

fn overlapped_wsa_send_to(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    method_arity(args, "WSASendTo", 5, 5)?;
    let obj = arg(args, 0, "WSASendTo")?;
    let handle = isize_w(arg(args, 1, "WSASendTo")?)?;
    let buffer = copied_read_buffer(arg(args, 2, "WSASendTo")?, "WSASendTo")?;
    let flags = u32_w(arg(args, 3, "WSASendTo")?)?;
    let (address, address_length) = parse_address(arg(args, 4, "WSASendTo")?)?;
    let size = u32::try_from(buffer.len())
        .map_err(|_| pyre_interpreter::PyError::value_error("buffer too large"))?;
    let mut state = native(obj)?.lock();
    operation_already_attempted(&state)?;
    state.handle = handle as host_overlapped::Handle;
    state.kind = OverlappedType::WriteTo;
    state.buffer = buffer;
    state.address = address;
    let error = host_overlapped::start_wsa_send_to(
        handle as usize,
        state.buffer.as_ptr(),
        size,
        flags,
        state.address.as_ptr() as *const host_overlapped::SocketAddrRaw,
        address_length,
        &mut state.overlapped,
    );
    accept_write_start_error(&mut state, error)
}

fn property(ns: PyObjectRef, name: &'static str, getter: pyre_interpreter::BuiltinCodeFn) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        name,
        pyre_interpreter::typedef::make_getset_descriptor_named(
            pyre_interpreter::make_builtin_function_with_arity(name, getter, 2),
            name,
        )
    );
}

fn init_overlapped_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__new__",
        pyre_interpreter::typedef::make_new_descr(overlapped_new)
    );
    for (name, function) in [
        (
            "cancel",
            overlapped_cancel as pyre_interpreter::BuiltinCodeFn,
        ),
        ("getresult", overlapped_getresult),
        ("ReadFile", overlapped_read_file),
        ("ReadFileInto", overlapped_read_file_into),
        ("WriteFile", overlapped_write_file),
        ("WSARecv", overlapped_wsa_recv),
        ("WSARecvInto", overlapped_wsa_recv_into),
        ("WSASend", overlapped_wsa_send),
        ("AcceptEx", overlapped_accept_ex),
        ("ConnectEx", overlapped_connect_ex),
        ("DisconnectEx", overlapped_disconnect_ex),
        ("TransmitFile", overlapped_transmit_file),
        ("ConnectNamedPipe", overlapped_connect_named_pipe),
        ("WSARecvFrom", overlapped_wsa_recv_from),
        ("WSARecvFromInto", overlapped_wsa_recv_from_into),
        ("WSASendTo", overlapped_wsa_send_to),
    ] {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            name,
            pyre_interpreter::make_builtin_function(name, function)
        );
    }
    property(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "address",
        |args| {
            let state = native(arg(args, 1, "address")?)?.lock();
            Ok(pyre_object::w_int_new(
                &state.overlapped as *const _ as usize as i64,
            ))
        },
    );
    property(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "pending",
        |args| {
            let state = native(arg(args, 1, "pending")?)?.lock();
            Ok(pyre_object::w_bool_from(
                !host_overlapped::has_overlapped_io_completed(&state.overlapped)
                    && state.kind != OverlappedType::NotStarted,
            ))
        },
    );
    property(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "error",
        |args| {
            let state = native(arg(args, 1, "error")?)?.lock();
            Ok(pyre_object::w_int_new(state.error as i64))
        },
    );
    property(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "event",
        |args| {
            let state = native(arg(args, 1, "event")?)?.lock();
            Ok(w_uintptr(state.overlapped.hEvent as usize))
        },
    );
}

static OVERLAPPED_RUNTIME_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

pub fn overlapped_type() -> PyObjectRef {
    OVERLAPPED_RUNTIME_TYPE.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "_overlapped.Overlapped",
            init_overlapped_type,
            pyre_interpreter::typedef::w_object(),
            <W_Overlapped as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_interpreter::typedef::mark_cpython_heap_type(tp, true);
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_Overlapped as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(tp, false) };
        tp
    })
}

/// PyPy `Overlapped.__del__`: wait/cancel before native buffers are released,
/// then close the event and restore the thread's last-error value.
pub unsafe fn w_overlapped_dealloc(obj: PyObjectRef) {
    let Some(this) = W_Overlapped::from_obj(obj) else {
        return;
    };
    if this.backend.is_null() {
        return;
    }
    let backend = unsafe { Box::from_raw(this.backend) };
    this.backend = std::ptr::null_mut();
    let mut state = backend.lock();
    let old_error = host_winapi::get_last_error();
    if !host_overlapped::has_overlapped_io_completed(&state.overlapped)
        && state.kind != OverlappedType::NotStarted
    {
        let _ = host_overlapped::cancel_overlapped_for_drop(state.handle, &state.overlapped);
    }
    if !state.overlapped.hEvent.is_null() {
        let _ = host_winapi::close_handle(state.overlapped.hEvent);
        state.overlapped.hEvent = std::ptr::null_mut();
    }
    if state.buffer_export_held && !this.w_buffer_owner.is_null() {
        unsafe { pyre_interpreter::builtins::buffer_export_decref(this.w_buffer_owner) };
        state.buffer_export_held = false;
    }
    host_windows::set_last_error(old_error);
}

fn connect_pipe(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let address = pyre_interpreter::baseobjspace::text_w(arg(args, 0, "ConnectPipe")?)?;
    let handle = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        host_overlapped::connect_pipe(address)
    };
    handle
        .map(|handle| w_uintptr(handle as usize))
        .map_err(win32_err)
}

fn create_iocp(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    host_overlapped::create_io_completion_port(
        isize_w(arg(args, 0, "CreateIoCompletionPort")?)?,
        isize_w(arg(args, 1, "CreateIoCompletionPort")?)?,
        usize_w(arg(args, 2, "CreateIoCompletionPort")?)?,
        u32_w(arg(args, 3, "CreateIoCompletionPort")?)?,
    )
    .map(|handle| w_uintptr(handle as usize))
    .map_err(win32_err)
}

fn get_queued_completion_status(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let port = isize_w(arg(args, 0, "GetQueuedCompletionStatus")?)?;
    let msecs = u32_w(arg(args, 1, "GetQueuedCompletionStatus")?)?;
    // The wait is where an event loop sleeps, so the other threads -- the
    // ones `call_soon_threadsafe` wakes it from -- run meanwhile.
    let status = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        host_overlapped::get_queued_completion_status(port, msecs)
    };
    match status.map_err(win32_err)? {
        host_overlapped::WaitResult::Timeout => Ok(pyre_object::w_none()),
        host_overlapped::WaitResult::Queued(status) => {
            let mut fields = pyre_object::gc_roots::RootedItems::new();
            fields.push(pyre_object::w_int_new(status.error as i64));
            fields.push(pyre_object::w_int_new(status.bytes_transferred as i64));
            fields.push(w_uintptr(status.completion_key));
            fields.push(w_uintptr(status.overlapped as usize));
            Ok(pyre_object::w_tuple_new(fields.take()))
        }
    }
}

fn post_queued_completion_status(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    host_overlapped::post_queued_completion_status(
        isize_w(arg(args, 0, "PostQueuedCompletionStatus")?)?,
        u32_w(arg(args, 1, "PostQueuedCompletionStatus")?)?,
        usize_w(arg(args, 2, "PostQueuedCompletionStatus")?)?,
        usize_w(arg(args, 3, "PostQueuedCompletionStatus")?)?,
    )
    .map_err(win32_err)?;
    Ok(pyre_object::w_none())
}

fn register_wait_with_queue(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    host_overlapped::register_wait_with_queue(
        isize_w(arg(args, 0, "RegisterWaitWithQueue")?)?,
        isize_w(arg(args, 1, "RegisterWaitWithQueue")?)?,
        usize_w(arg(args, 2, "RegisterWaitWithQueue")?)?,
        u32_w(arg(args, 3, "RegisterWaitWithQueue")?)?,
    )
    .map(|handle| w_uintptr(handle as usize))
    .map_err(win32_err)
}

fn unregister_wait(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    host_overlapped::unregister_wait(isize_w(arg(args, 0, "UnregisterWait")?)?)
        .map_err(win32_err)?;
    Ok(pyre_object::w_none())
}

fn unregister_wait_ex(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let wait_handle = isize_w(arg(args, 0, "UnregisterWaitEx")?)?;
    let event = isize_w(arg(args, 1, "UnregisterWaitEx")?)?;
    // An event handle makes the call wait for the running callbacks.
    let unregistered = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        host_overlapped::unregister_wait_ex(wait_handle, event)
    };
    unregistered.map_err(win32_err)?;
    Ok(pyre_object::w_none())
}

fn bind_local(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let socket = isize_w(arg(args, 0, "BindLocal")?)?;
    let family = pyre_interpreter::baseobjspace::c_int_w(arg(args, 1, "BindLocal")?)?;
    if family != host_overlapped::AF_INET_FAMILY && family != host_overlapped::AF_INET6_FAMILY {
        return Err(pyre_interpreter::PyError::value_error(
            "expected tuple of length 2 or 4",
        ));
    }
    host_overlapped::bind_local(socket, family).map_err(win32_err)?;
    Ok(pyre_object::w_none())
}

fn format_message(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    Ok(pyre_object::w_str_new_managed(
        &host_overlapped::format_message(u32_w(arg(args, 0, "FormatMessage")?)?),
    ))
}

fn wsa_connect(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let socket = isize_w(arg(args, 0, "WSAConnect")?)?;
    let (address, length) = parse_address(arg(args, 1, "WSAConnect")?)?;
    host_overlapped::wsa_connect(
        socket,
        address.as_ptr() as *const host_overlapped::SocketAddrRaw,
        length,
    )
    .map_err(win32_err)?;
    Ok(pyre_object::w_none())
}

fn create_event(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let attributes = arg(args, 0, "CreateEvent")?;
    if !unsafe { pyre_object::is_none(attributes) } {
        return Err(pyre_interpreter::PyError::value_error(
            "EventAttributes must be None",
        ));
    }
    let manual_reset = pyre_interpreter::baseobjspace::is_true(arg(args, 1, "CreateEvent")?)?;
    let initial_state = pyre_interpreter::baseobjspace::is_true(arg(args, 2, "CreateEvent")?)?;
    let w_name = arg(args, 3, "CreateEvent")?;
    let name = if unsafe { pyre_object::is_none(w_name) } {
        None
    } else {
        Some(
            widestring::WideCString::from_str(pyre_interpreter::baseobjspace::text_w(w_name)?)
                .map_err(|_| pyre_interpreter::PyError::value_error("embedded null character"))?,
        )
    };
    host_winapi::create_event_w(manual_reset, initial_state, name.as_deref())
        .map(|handle| w_uintptr(handle as usize))
        .map_err(win32_err)
}

fn set_event(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    host_winapi::set_event(handle_w(arg(args, 0, "SetEvent")?)?).map_err(win32_err)?;
    Ok(pyre_object::w_none())
}

fn reset_event(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    host_winapi::reset_event(handle_w(arg(args, 0, "ResetEvent")?)?).map_err(win32_err)?;
    Ok(pyre_object::w_none())
}

pub fn init(ns: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // PyPy imports `_socket` before resolving the extension-function GUIDs.
    // The host layer exposes the same process-global WSAStartup owner, so the
    // builtin can establish that prerequisite without creating a second
    // module-local or thread-local socket state.
    host_windows::init_winsock();
    if let Err(error) = host_overlapped::initialize_winsock_extensions() {
        // Module initialization cannot return a PyResult in the current mixed
        // module ABI.  `_socket` repeats WSA startup and each operation reports
        // its own exact error; retain the importable module surface here.
        let _ = error;
    }
    for (name, value) in [
        ("ERROR_IO_PENDING", host_winapi::ERROR_IO_PENDING as i64),
        (
            "ERROR_NETNAME_DELETED",
            host_winapi::ERROR_NETNAME_DELETED as i64,
        ),
        (
            "ERROR_OPERATION_ABORTED",
            host_winapi::ERROR_OPERATION_ABORTED as i64,
        ),
        ("ERROR_PIPE_BUSY", host_winapi::ERROR_PIPE_BUSY as i64),
        (
            "ERROR_PORT_UNREACHABLE",
            host_winapi::ERROR_PORT_UNREACHABLE as i64,
        ),
        ("ERROR_SEM_TIMEOUT", host_winapi::ERROR_SEM_TIMEOUT as i64),
        (
            "SO_UPDATE_ACCEPT_CONTEXT",
            host_overlapped::SO_UPDATE_ACCEPT_CONTEXT_VALUE as i64,
        ),
        (
            "SO_UPDATE_CONNECT_CONTEXT",
            host_overlapped::SO_UPDATE_CONNECT_CONTEXT_VALUE as i64,
        ),
        (
            "TF_REUSE_SOCKET",
            host_overlapped::TF_REUSE_SOCKET_FLAG as i64,
        ),
        ("INFINITE", host_winapi::INFINITE_TIMEOUT as i64),
        ("NULL", 0),
    ] {
        pyre_interpreter::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(value));
    }
    // The handle sentinel is `(HANDLE)-1`, which prints as the unsigned value.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "INVALID_HANDLE_VALUE",
        w_uintptr(host_overlapped::INVALID_HANDLE_VALUE_ISIZE as usize)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "Overlapped", overlapped_type());
    for (name, arity, function) in [
        (
            "ConnectPipe",
            1,
            connect_pipe as pyre_interpreter::BuiltinCodeFn,
        ),
        ("CreateIoCompletionPort", 4, create_iocp),
        ("GetQueuedCompletionStatus", 2, get_queued_completion_status),
        (
            "PostQueuedCompletionStatus",
            4,
            post_queued_completion_status,
        ),
        ("RegisterWaitWithQueue", 4, register_wait_with_queue),
        ("UnregisterWait", 1, unregister_wait),
        ("UnregisterWaitEx", 2, unregister_wait_ex),
        ("BindLocal", 2, bind_local),
        ("FormatMessage", 1, format_message),
        ("WSAConnect", 2, wsa_connect),
        ("CreateEvent", 4, create_event),
        ("SetEvent", 1, set_event),
        ("ResetEvent", 1, reset_event),
    ] {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            name,
            pyre_interpreter::gateway::with_module(
                "_overlapped",
                pyre_interpreter::make_module_builtin_function_with_arity(name, function, arity),
            )
        );
    }
    Ok(())
}

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // `lib_pypy/_overlapped.py Overlapped`: the object owns its native
    // OVERLAPPED record and pending buffers until cancellation/completion.
    // Its three Python fields are ordinary generated trace offsets; sweep
    // performs PyPy's cancel/wait-before-free ordering and closes hEvent.
    types.push(ModuleGcType {
        descriptor: <W_Overlapped as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: Some(gc_destructor!(w_overlapped_dealloc)),
    });
}
