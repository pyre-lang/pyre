//! The half of `_winapi` that `rustpython_host_env::winapi` backs.
//!
//! Split off from [`super`] because it needs the `host_env` layer, which the
//! module as a whole does not: without it `_winapi` still answers with its
//! constants and the handful of calls made straight against `windows-sys`.
//!
//! Each function is registered by hand at the end rather than through
//! `py_module!`'s `inline_functions`, which cannot be compiled conditionally.
//! The registration keeps the two shapes the module gives its calls: the ones
//! whose parameters are all positional-only get no `Signature`, so a keyword
//! call is turned away the way Argument Clinic's trailing `/` turns it away,
//! and the rest keep the derived one and bind by name.

use pyre_object::{PY_NULL, PyObjectRef, w_int_new, w_none, w_tuple_new};
use rustpython_host_env::winapi as host_winapi;
use std::os::windows::ffi::OsStringExt;
use widestring::WideCString;
use windows_sys::Win32::Foundation::HANDLE;

use super::{IntArg, handle_w, w_handle, win32_code};

/// [`super::win32_err`] for the two file-mapping calls, which name the mapping
/// in the error the way a failed open names its file.
fn win32_err_named(error: std::io::Error, w_name: PyObjectRef) -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::os_error_win32_syscall2(
        error.raw_os_error().unwrap_or(0),
        w_name,
        PY_NULL,
    )
}

/// How a call names a wide-string argument in the `TypeError` it raises for a
/// non-`str`.  Most of them do not name it at all; the two path calls name it,
/// and two more give its position.
pub(super) enum WideArg<'a> {
    Unnamed,
    Named {
        function: &'a str,
        argument: &'a str,
    },
    Numbered {
        function: &'a str,
        position: usize,
    },
}

/// A `str` argument as the NUL-terminated wide string Win32 takes.
///
/// The text goes to the host as spelled, unpaired surrogates included: these
/// calls take UTF-16, and a lossy re-encode would name a different object.
pub(super) fn wide(
    w_text: PyObjectRef,
    argument: WideArg<'_>,
) -> Result<WideCString, pyre_interpreter::PyError> {
    if !unsafe { pyre_object::is_str(w_text) } {
        return Err(wide_type_error(&argument, "str", w_text));
    }
    // `PyUnicode_AsWideCharString` refuses every NUL the text holds, the last
    // character included; `WideCString::from_vec` would read that one as the
    // terminator it appends itself and accept the string.
    let units = wide_units(w_text);
    if units.contains(&0) {
        return Err(pyre_interpreter::PyError::value_error(
            "embedded null character",
        ));
    }
    WideCString::from_vec(units)
        .map_err(|_| pyre_interpreter::PyError::value_error("embedded null character"))
}

/// The `TypeError` a wide-string argument raises for a value it cannot take.
fn wide_type_error(
    argument: &WideArg<'_>,
    accepted: &str,
    w_text: PyObjectRef,
) -> pyre_interpreter::PyError {
    let got = pyre_interpreter::gateway::short_type_name(w_text);
    pyre_interpreter::PyError::type_error(match argument {
        WideArg::Unnamed => format!("argument must be {accepted}, not {got}"),
        WideArg::Named { function, argument } => {
            format!("{function}() argument '{argument}' must be {accepted}, not {got}")
        }
        WideArg::Numbered { function, position } => {
            format!("{function}() argument {position} must be {accepted}, not {got}")
        }
    })
}

/// [`wide`] where `None` — or an argument that is not there at all — names the
/// Win32 default instead of a string.
pub(super) fn wide_or_none(
    w_text: PyObjectRef,
    argument: WideArg<'_>,
) -> Result<Option<WideCString>, pyre_interpreter::PyError> {
    if w_text.is_null() || unsafe { pyre_object::is_none(w_text) } {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_str(w_text) } {
        return Err(wide_type_error(&argument, "str or None", w_text));
    }
    wide(w_text, argument).map(Some)
}

/// The UTF-16 units of a `str`, without a terminator.
///
/// This is what `PyUnicode_AsWideCharString` hands back: the text as spelled,
/// so an unpaired surrogate stays the code unit it is instead of being refused
/// the way a strict utf-8 encode refuses it.
pub(super) fn wide_units(w_text: PyObjectRef) -> Vec<u16> {
    unsafe { pyre_object::w_str_get_wtf8(w_text) }
        .encode_wide()
        .collect()
}

/// Wide characters back as a `str`.
fn w_wide(units: &[u16]) -> PyObjectRef {
    pyre_object::w_str_from_wtf8_managed(rustpython_wtf8::Wtf8Buf::from_wide(units))
}

/// A `DWORD` argument `None` leaves alone.
fn c_uint_or_none(w_value: PyObjectRef) -> Result<Option<u32>, pyre_interpreter::PyError> {
    if unsafe { pyre_object::is_none(w_value) } {
        return Ok(None);
    }
    pyre_interpreter::baseobjspace::c_uint_w(w_value).map(Some)
}

/// The handles a `handle_seq` argument names.
///
/// `PySequence_Check` is what guards these two calls, so a mapping is turned
/// away along with everything that has no subscript at all.
fn sequence_handles(w_seq: PyObjectRef) -> Result<Vec<HANDLE>, pyre_interpreter::PyError> {
    let is_sequence = !unsafe { pyre_object::is_dict(w_seq) }
        && unsafe { pyre_interpreter::baseobjspace::lookup(w_seq, "__getitem__") }.is_some();
    if !is_sequence {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "sequence type expected, got '{}'",
            pyre_interpreter::gateway::short_type_name(w_seq)
        )));
    }
    let items = pyre_interpreter::baseobjspace::unpackiterable(w_seq, -1)?;
    let mut handles = Vec::with_capacity(items.len());
    for w_item in items {
        handles.push(handle_w(w_item, IntArg::Element)?);
    }
    Ok(handles)
}

// ── locale and version ──

/// `_winapi.GetACP()` — the process's ANSI code page.
#[pyre_interpreter::pyre_function]
pub fn GetACP() -> i64 {
    host_winapi::get_acp() as i64
}

/// `_winapi.GetVersion()` — the packed version word `GetVersion` reports.
#[pyre_interpreter::pyre_function]
pub fn GetVersion() -> i64 {
    host_winapi::get_version() as i64
}

/// `_winapi.LCMapStringEx(locale, flags, src)` — the mapped text, which is
/// what `ntpath.normcase` lowercases through.
///
/// The four flags that make the call answer with a sort key or a hash rather
/// than text are rejected: the result would not be a string.  An empty `src`
/// is not special-cased — `LCMapStringEx` reports `ERROR_INVALID_PARAMETER`
/// for it, and that is the error the caller sees.
#[pyre_interpreter::pyre_function]
pub fn LCMapStringEx(
    locale: PyObjectRef,
    flags: PyCUInt,
    src: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    const UNSUPPORTED: u32 = host_winapi::LCMAP_SORTHANDLE_FLAG
        | host_winapi::LCMAP_HASH_FLAG
        | host_winapi::LCMAP_BYTEREV_FLAG
        | host_winapi::LCMAP_SORTKEY_FLAG;
    if flags & UNSUPPORTED != 0 {
        return Err(pyre_interpreter::PyError::value_error("unsupported flags"));
    }
    let locale = wide(locale, WideArg::Unnamed)?;
    if !unsafe { pyre_object::is_str(src) } {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "LCMapStringEx() argument 3 must be str, not {}",
            pyre_interpreter::gateway::short_type_name(src)
        )));
    }
    // The source is passed by length, so an embedded NUL is a character like
    // any other rather than a terminator.
    let src = wide_units(src);
    host_winapi::lc_map_string_ex(&locale, flags, &src)
        .map(|mapped| w_wide(&mapped))
        .map_err(super::win32_err)
}

// ── paths ──

/// `_winapi.GetLongPathName(path)`
#[pyre_interpreter::pyre_function]
pub fn GetLongPathName(path: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let path = wide(
        path,
        WideArg::Named {
            function: "GetLongPathName",
            argument: "path",
        },
    )?;
    host_winapi::get_long_path_name_w(&path)
        .map(|name| w_wide(&name))
        .map_err(super::win32_err)
}

/// `_winapi.GetShortPathName(path)`
#[pyre_interpreter::pyre_function]
pub fn GetShortPathName(path: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let path = wide(
        path,
        WideArg::Named {
            function: "GetShortPathName",
            argument: "path",
        },
    )?;
    host_winapi::get_short_path_name_w(&path)
        .map(|name| w_wide(&name))
        .map_err(super::win32_err)
}

/// `_winapi.CreateJunction(src_path, dst_path)` — a directory junction, the
/// reparse point `os.symlink` falls back to for a directory when the process
/// may not create a real symlink.
#[pyre_interpreter::pyre_function]
pub fn CreateJunction(
    src_path: PyObjectRef,
    dst_path: PyObjectRef,
) -> Result<(), pyre_interpreter::PyError> {
    let src = wide(
        src_path,
        WideArg::Numbered {
            function: "CreateJunction",
            position: 1,
        },
    )?;
    let dst = wide(
        dst_path,
        WideArg::Numbered {
            function: "CreateJunction",
            position: 2,
        },
    )?;
    let src = std::ffi::OsString::from_wide(src.as_slice());
    let dst = std::ffi::OsString::from_wide(dst.as_slice());
    // Scoped to the call alone, as `releasegil=True` is — see `CreateFile`.
    let result = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        host_winapi::create_junction(std::path::Path::new(&src), std::path::Path::new(&dst))
    };
    result.map_err(super::win32_err)
}

/// `_winapi.CopyFile2(existing_file_name, new_file_name, flags,
/// progress_routine=None)`
///
/// `progress_routine` is accepted and never called, which is what
/// `_winapi.CopyFile2`'s own documentation states: "reserved for future use,
/// but is currently not implemented.  Its value is ignored."  The parameter
/// exists so a caller only has to test for `CopyFile2` itself, and the code
/// that would put the callback in the extended parameters is commented out
/// upstream.
#[pyre_interpreter::pyre_function]
pub fn CopyFile2(
    existing_file_name: PyObjectRef,
    new_file_name: PyObjectRef,
    flags: PyCUInt,
    #[default(pyre_object::w_none())] progress_routine: PyObjectRef,
) -> Result<(), pyre_interpreter::PyError> {
    let _ = progress_routine;
    let existing = wide(existing_file_name, WideArg::Unnamed)?;
    let new = wide(new_file_name, WideArg::Unnamed)?;
    let result = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        host_winapi::copy_file2(&existing, &new, flags)
    };
    result.map_err(super::win32_err)
}

// ── processes ──

/// `_winapi.OpenProcess(desired_access, inherit_handle, process_id)`
#[pyre_interpreter::pyre_function]
pub fn OpenProcess(
    desired_access: PyCUInt,
    inherit_handle: PyIndexCInt,
    process_id: PyCUInt,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    host_winapi::open_process(desired_access, inherit_handle != 0, process_id)
        .map(w_handle)
        .map_err(super::win32_err)
}

/// `_winapi.ExitProcess(ExitCode)` — the process ends inside the call, so
/// nothing after it runs and no buffered stream is flushed.
#[pyre_interpreter::pyre_function]
pub fn ExitProcess(ExitCode: PyCUInt) -> Result<(), pyre_interpreter::PyError> {
    host_winapi::exit_process(ExitCode)
}

// ── events and mutexes ──

/// `_winapi.CreateEventW(security_attributes, manual_reset, initial_state,
/// name)`
///
/// The security descriptor is required as an int and then left at the call's
/// default, the way `CreateProcess` treats its two attribute arguments.
#[pyre_interpreter::pyre_function]
pub fn CreateEventW(
    security_attributes: PyObjectRef,
    manual_reset: PyIndexCInt,
    initial_state: PyIndexCInt,
    name: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _ = handle_w(
        security_attributes,
        IntArg::At {
            function: "CreateEventW",
            position: 1,
        },
    )?;
    let name = wide_or_none(name, WideArg::Unnamed)?;
    host_winapi::create_event_w(manual_reset != 0, initial_state != 0, name.as_deref())
        .map(w_handle)
        .map_err(super::win32_err)
}

/// `_winapi.OpenEventW(desired_access, inherit_handle, name)` — `None` when
/// there is no event of that name, rather than an error.
#[pyre_interpreter::pyre_function]
pub fn OpenEventW(
    desired_access: PyCUInt,
    inherit_handle: PyIndexCInt,
    name: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let name = wide(name, WideArg::Unnamed)?;
    Ok(
        match host_winapi::open_event_w(desired_access, inherit_handle != 0, &name) {
            Ok(opened) => w_handle(opened),
            Err(_) => w_none(),
        },
    )
}

/// `_winapi.SetEvent(event)`
#[pyre_interpreter::pyre_function]
pub fn SetEvent(event: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let event = handle_w(
        event,
        IntArg::At {
            function: "SetEvent",
            position: 1,
        },
    )?;
    host_winapi::set_event(event).map_err(super::win32_err)
}

/// `_winapi.ResetEvent(event)`
#[pyre_interpreter::pyre_function]
pub fn ResetEvent(event: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let event = handle_w(
        event,
        IntArg::At {
            function: "ResetEvent",
            position: 1,
        },
    )?;
    host_winapi::reset_event(event).map_err(super::win32_err)
}

/// `_winapi.CreateMutexW(security_attributes, initial_owner, name)`
#[pyre_interpreter::pyre_function]
pub fn CreateMutexW(
    security_attributes: PyObjectRef,
    initial_owner: PyIndexCInt,
    name: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _ = handle_w(
        security_attributes,
        IntArg::At {
            function: "CreateMutexW",
            position: 1,
        },
    )?;
    let name = wide_or_none(name, WideArg::Unnamed)?;
    host_winapi::create_mutex_w(initial_owner != 0, name.as_deref())
        .map(w_handle)
        .map_err(super::win32_err)
}

/// `_winapi.OpenMutexW(desired_access, inherit_handle, name)` — `None` when
/// there is no mutex of that name, as with [`OpenEventW`].
#[pyre_interpreter::pyre_function]
pub fn OpenMutexW(
    desired_access: PyCUInt,
    inherit_handle: PyIndexCInt,
    name: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let name = wide(name, WideArg::Unnamed)?;
    Ok(
        match host_winapi::open_mutex_w(desired_access, inherit_handle != 0, &name) {
            Ok(opened) => w_handle(opened),
            Err(_) => w_none(),
        },
    )
}

/// `_winapi.ReleaseMutex(mutex)`
#[pyre_interpreter::pyre_function]
pub fn ReleaseMutex(mutex: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let mutex = handle_w(
        mutex,
        IntArg::At {
            function: "ReleaseMutex",
            position: 1,
        },
    )?;
    if host_winapi::release_mutex(mutex) == 0 {
        return Err(super::last_os_error());
    }
    Ok(())
}

// ── waiting ──

/// The most handles a wait may name.  `MAXIMUM_WAIT_OBJECTS` is 64 and the
/// last slot is reserved, so the sequence may fill 63 of them.
const MAXIMUM_HANDLES: usize = 63;

/// `WAIT_TIMEOUT`, which `BatchedWaitForMultipleObjects` reports as an error
/// rather than as a return value.
const WAIT_TIMEOUT: u32 = 258;

/// `_winapi.WaitForMultipleObjects(handle_seq, wait_flag, milliseconds)`
///
/// The reserved slot is where `_PyOS_SigintEvent()` goes upstream.  pyre
/// installs no OS signal handler on Windows at all — `pypysig_setflag` is a
/// `#[cfg(unix)]` body and its Windows twin declines — so there is no SIGINT
/// flag for an event to stand beside, the slot stays empty, and a wait runs to
/// its timeout instead of ending in `InterruptedError`.  Wiring one in belongs
/// with Windows signal delivery, not here; `lib_pypy/_winapi.py` leaves
/// the same gap.
#[pyre_interpreter::pyre_function]
pub fn WaitForMultipleObjects(
    handle_seq: PyObjectRef,
    wait_flag: PyIndexCInt,
    #[default(host_winapi::INFINITE_TIMEOUT)] milliseconds: PyCUInt,
) -> Result<i64, pyre_interpreter::PyError> {
    let handles = sequence_handles(handle_seq)?;
    if handles.len() > MAXIMUM_HANDLES {
        return Err(pyre_interpreter::PyError::value_error(format!(
            "need at most {MAXIMUM_HANDLES} handles, got a sequence of length {}",
            handles.len()
        )));
    }
    let result = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        host_winapi::wait_for_multiple_objects(&handles, wait_flag != 0, milliseconds)
    };
    result.map(i64::from).map_err(super::win32_err)
}

/// `_winapi.BatchedWaitForMultipleObjects(handle_seq, wait_all, milliseconds)`
/// -> the indices of the handles that were signalled, one per batch, or `None`
/// under `wait_all`, which has nothing to single out.
///
/// Unlike the call above this one takes any number of handles, splitting them
/// across worker threads that each wait on a batch of 63, and reports a
/// timeout as `TimeoutError` rather than as a return value.
#[pyre_interpreter::pyre_function]
pub fn BatchedWaitForMultipleObjects(
    handle_seq: PyObjectRef,
    wait_all: PyIndexCInt,
    #[default(host_winapi::INFINITE_TIMEOUT)] milliseconds: PyCUInt,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let handles = sequence_handles(handle_seq)?;
    // Nothing to wait for: `wait_all` is satisfied at once and the other form
    // has no handle to name, neither of which is a timeout.
    if handles.is_empty() {
        return Ok(batched_wait_answer(wait_all != 0, Vec::new()));
    }
    let result = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        host_winapi::batched_wait_for_multiple_objects(&handles, wait_all != 0, milliseconds, None)
    };
    let indices = match result {
        Ok(host_winapi::BatchedWaitResult::All) => {
            return Ok(batched_wait_answer(true, Vec::new()));
        }
        Ok(host_winapi::BatchedWaitResult::Indices(indices)) => indices,
        Err(host_winapi::BatchedWaitError::Timeout) => return Err(win32_code(WAIT_TIMEOUT)),
        // Only reachable once a SIGINT event is passed, which pyre has none
        // of; the wait that saw one reports it as the interrupted syscall it
        // is.
        Err(host_winapi::BatchedWaitError::Interrupted) => {
            return Err(pyre_interpreter::PyError::os_error_syscall(
                libc::EINTR,
                PY_NULL,
            ));
        }
        Err(host_winapi::BatchedWaitError::Os(code)) => return Err(win32_code(code)),
    };
    Ok(batched_wait_answer(wait_all != 0, indices))
}

/// What a completed batched wait answers with: a `wait_all` names no handle
/// in particular, so it has no list to give back.
fn batched_wait_answer(wait_all: bool, indices: Vec<usize>) -> PyObjectRef {
    if wait_all {
        return w_none();
    }
    pyre_object::listobject::w_list_new(indices.into_iter().map(|i| w_int_new(i as i64)).collect())
}

// ── files and named pipes ──

/// `_winapi.CreateFile(file_name, desired_access, share_mode,
/// security_attributes, creation_disposition, flags_and_attributes,
/// template_file)`
///
/// The security descriptor and the template file are required and then left
/// at the call's defaults; every caller passes `NULL` for both.
#[allow(clippy::too_many_arguments)]
#[pyre_interpreter::pyre_function]
pub fn CreateFile(
    file_name: PyObjectRef,
    desired_access: PyCUInt,
    share_mode: PyCUInt,
    security_attributes: PyObjectRef,
    creation_disposition: PyCUInt,
    flags_and_attributes: PyCUInt,
    template_file: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let at = |position| IntArg::At {
        function: "CreateFile",
        position,
    };
    let _ = (
        handle_w(security_attributes, at(4))?,
        handle_w(template_file, at(7))?,
    );
    let file_name = wide(file_name, WideArg::Unnamed)?;
    // `llexternal(..., releasegil=True)` releases around the C call alone and
    // is back inside before the caller's next RPython instruction, so nothing
    // that allocates or raises runs outside.  The guard has to be scoped the
    // same way: `w_handle` allocates the answer, and a thread that allocates
    // while it is out of the RUNNING census can have it collected under it —
    // `_winapi.CreateFile` then answers 0, and the `SetNamedPipeHandleState`
    // that `multiprocessing.connection.PipeClient` runs next reports
    // `ERROR_INVALID_HANDLE`.
    let result = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        host_winapi::create_file_w(
            &file_name,
            desired_access,
            share_mode,
            creation_disposition,
            flags_and_attributes,
        )
    };
    result.map(w_handle).map_err(super::win32_err)
}

/// `_winapi.CreateNamedPipe(name, open_mode, pipe_mode, max_instances,
/// out_buffer_size, in_buffer_size, default_timeout, security_attributes)`
#[allow(clippy::too_many_arguments)]
#[pyre_interpreter::pyre_function]
pub fn CreateNamedPipe(
    name: PyObjectRef,
    open_mode: PyCUInt,
    pipe_mode: PyCUInt,
    max_instances: PyCUInt,
    out_buffer_size: PyCUInt,
    in_buffer_size: PyCUInt,
    default_timeout: PyCUInt,
    security_attributes: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _ = handle_w(
        security_attributes,
        IntArg::At {
            function: "CreateNamedPipe",
            position: 8,
        },
    )?;
    let name = wide(name, WideArg::Unnamed)?;
    host_winapi::create_named_pipe_w(
        &name,
        open_mode,
        pipe_mode,
        max_instances,
        out_buffer_size,
        in_buffer_size,
        default_timeout,
    )
    .map(w_handle)
    .map_err(super::win32_err)
}

/// `_winapi.SetNamedPipeHandleState(named_pipe, mode, max_collection_count,
/// collect_data_timeout)` — each `None` leaves that part of the state alone.
#[pyre_interpreter::pyre_function]
pub fn SetNamedPipeHandleState(
    named_pipe: PyObjectRef,
    mode: PyObjectRef,
    max_collection_count: PyObjectRef,
    collect_data_timeout: PyObjectRef,
) -> Result<(), pyre_interpreter::PyError> {
    let named_pipe = handle_w(
        named_pipe,
        IntArg::At {
            function: "SetNamedPipeHandleState",
            position: 1,
        },
    )?;
    host_winapi::set_named_pipe_handle_state(
        named_pipe,
        c_uint_or_none(mode)?,
        c_uint_or_none(max_collection_count)?,
        c_uint_or_none(collect_data_timeout)?,
    )
    .map_err(super::win32_err)
}

/// `_winapi.WaitNamedPipe(name, timeout)`
#[pyre_interpreter::pyre_function]
pub fn WaitNamedPipe(name: PyObjectRef, timeout: PyCUInt) -> Result<(), pyre_interpreter::PyError> {
    let name = wide(name, WideArg::Unnamed)?;
    let result = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        host_winapi::wait_named_pipe_w(&name, timeout)
    };
    result.map_err(super::win32_err)
}

// ── the asynchronous half, which a sandbox build leaves out ──

/// `_winapi.ConnectNamedPipe(handle, overlapped=False)` — `None` once the
/// connection is made, or the `Overlapped` to wait on.
#[cfg(not(feature = "sandbox"))]
#[pyre_interpreter::pyre_function]
pub fn ConnectNamedPipe(
    handle_: PyObjectRef,
    #[default(pyre_object::w_bool_from(false))] overlapped: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let pipe = handle_w(
        handle_,
        IntArg::At {
            function: "ConnectNamedPipe",
            position: 1,
        },
    )?;
    let use_overlapped = pyre_interpreter::baseobjspace::is_true(overlapped)?;
    super::overlapped::connect_named_pipe(pipe, use_overlapped)
}

/// `_winapi.ReadFile(handle, size, overlapped=False)`
#[cfg(not(feature = "sandbox"))]
#[pyre_interpreter::pyre_function]
pub fn ReadFile(
    handle_: PyObjectRef,
    size: PyCUInt,
    #[default(pyre_object::w_bool_from(false))] overlapped: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let file = handle_w(
        handle_,
        IntArg::At {
            function: "ReadFile",
            position: 1,
        },
    )?;
    let use_overlapped = pyre_interpreter::baseobjspace::is_true(overlapped)?;
    super::overlapped::read_file(file, size, use_overlapped)
}

/// `_winapi.WriteFile(handle, buffer, overlapped=False)`
#[cfg(not(feature = "sandbox"))]
#[pyre_interpreter::pyre_function]
pub fn WriteFile(
    handle_: PyObjectRef,
    buffer: PyObjectRef,
    #[default(pyre_object::w_bool_from(false))] overlapped: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let file = handle_w(
        handle_,
        IntArg::At {
            function: "WriteFile",
            position: 1,
        },
    )?;
    let use_overlapped = pyre_interpreter::baseobjspace::is_true(overlapped)?;
    let Some(data) = pyre_interpreter::baseobjspace::simple_buffer_bytes(buffer)? else {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "a bytes-like object is required, not '{}'",
            pyre_interpreter::gateway::short_type_name(buffer)
        )));
    };
    // The write reads from the buffer for as long as it is in flight, and the
    // exporter is only borrowed for the length of this call, so the bytes are
    // taken out of it before either can move.
    let bytes = data.as_bytes().to_vec();
    data.release();
    super::overlapped::write_file(file, &bytes, use_overlapped)
}

/// `_winapi.PeekNamedPipe(handle, size=0)` -> `(available, left_this_message)`,
/// and `(data, available, left_this_message)` once a size is asked for.
#[pyre_interpreter::pyre_function]
pub fn PeekNamedPipe(
    handle_: PyObjectRef,
    #[default(0u32)] size: PyCUInt,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let pipe = handle_w(
        handle_,
        IntArg::At {
            function: "PeekNamedPipe",
            position: 1,
        },
    )?;
    let peeked = host_winapi::peek_named_pipe(pipe, (size != 0).then_some(size))
        .map_err(super::win32_err)?;
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    if let Some(data) = peeked.data {
        fields.push(pyre_object::bytesobject::w_bytes_from_bytes(&data));
    }
    fields.push(w_int_new(i64::from(peeked.available)));
    fields.push(w_int_new(i64::from(peeked.left_this_message)));
    Ok(w_tuple_new(fields.take()))
}

// ── file mappings ──

/// `_winapi.CreateFileMapping(file_handle, security_attributes, protect,
/// max_size_high, max_size_low, name)`
#[pyre_interpreter::pyre_function]
pub fn CreateFileMapping(
    file_handle: PyObjectRef,
    security_attributes: PyObjectRef,
    protect: PyCUInt,
    max_size_high: PyCUInt,
    max_size_low: PyCUInt,
    name: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let at = |position| IntArg::At {
        function: "CreateFileMapping",
        position,
    };
    let file = handle_w(file_handle, at(1))?;
    let _ = handle_w(security_attributes, at(2))?;
    let wide_name = wide(name, WideArg::Unnamed)?;
    host_winapi::create_file_mapping_w(file, protect, max_size_high, max_size_low, Some(&wide_name))
        .map(w_handle)
        .map_err(|error| win32_err_named(error, name))
}

/// `_winapi.OpenFileMapping(desired_access, inherit_handle, name)`
#[pyre_interpreter::pyre_function]
pub fn OpenFileMapping(
    desired_access: PyCUInt,
    inherit_handle: PyIndexCInt,
    name: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let wide_name = wide(name, WideArg::Unnamed)?;
    host_winapi::open_file_mapping_w(desired_access, inherit_handle != 0, &wide_name)
        .map(w_handle)
        .map_err(|error| win32_err_named(error, name))
}

/// `_winapi.MapViewOfFile(file_map, desired_access, file_offset_high,
/// file_offset_low, number_bytes)` -> the address of the mapped view.
#[pyre_interpreter::pyre_function]
pub fn MapViewOfFile(
    file_map: PyObjectRef,
    desired_access: PyCUInt,
    file_offset_high: PyCUInt,
    file_offset_low: PyCUInt,
    number_bytes: PyIndexInt,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let mapping = handle_w(
        file_map,
        IntArg::At {
            function: "MapViewOfFile",
            position: 1,
        },
    )?;
    host_winapi::map_view_of_file(
        mapping,
        desired_access,
        file_offset_high,
        file_offset_low,
        number_bytes as usize,
    )
    .map(|address| w_int_new(address as i64))
    .map_err(super::win32_err)
}

/// `_winapi.UnmapViewOfFile(address)`
#[pyre_interpreter::pyre_function]
pub fn UnmapViewOfFile(address: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let address = handle_w(address, IntArg::Only("UnmapViewOfFile"))? as isize;
    host_winapi::unmap_view_of_file(address).map_err(super::win32_err)
}

/// `_winapi.VirtualQuerySize(address)` -> the size of the region the address
/// falls in, which is how a mapped view's length is recovered from the address
/// alone.
#[pyre_interpreter::pyre_function]
pub fn VirtualQuerySize(address: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let address = handle_w(address, IntArg::Only("VirtualQuerySize"))? as isize;
    host_winapi::virtual_query_size(address)
        .map(|size| w_int_new(size as i64))
        .map_err(super::win32_err)
}

// ── the mimetypes registry reader ──

/// `_winapi._mimetypes_read_windows_registry(on_type_read)` — calls
/// `on_type_read(type, ext)` for every `HKEY_CLASSES_ROOT` extension key
/// carrying a `Content Type`.
///
/// The whole walk lives behind one call because doing it a key at a time
/// through `winreg` is what made `mimetypes.init()` slow enough to matter.
#[pyre_interpreter::pyre_function]
pub fn _mimetypes_read_windows_registry(
    on_type_read: PyObjectRef,
) -> Result<(), pyre_interpreter::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let callback_slot = roots.base();
    let _ = roots.pin_root(on_type_read);
    host_winapi::read_windows_mimetype_registry_in_batches(
        |entries: &mut Vec<(String, String)>| -> Result<(), pyre_interpreter::PyError> {
            // The host refills the batch it hands over, so it has to be
            // drained here rather than merely read.
            for (mime_type, extension) in entries.drain(..) {
                // The second string is allocated while the first is only in a
                // Rust local, so each is published as it is made and read back
                // once both exist.
                let _entry_roots = pyre_object::gc_roots::push_roots();
                let type_slot =
                    pyre_object::gc_roots::pin_roots(&[pyre_object::w_str_new_managed(&mime_type)]);
                let extension_slot =
                    pyre_object::gc_roots::pin_roots(&[pyre_object::w_str_new_managed(&extension)]);
                pyre_interpreter::call::call_function_impl_result(
                    roots.get(callback_slot),
                    &[
                        pyre_object::gc_roots::shadow_stack_get(type_slot),
                        pyre_object::gc_roots::shadow_stack_get(extension_slot),
                    ],
                )?;
            }
            Ok(())
        },
    )
    .map_err(|error| match error {
        host_winapi::MimeRegistryReadError::Os(code) => win32_code(code),
        host_winapi::MimeRegistryReadError::Callback(error) => error,
    })
}

/// Bind the functions above into the module dict under their own names.
///
/// `keywords` keeps the `Signature` `#[pyre_interpreter::pyre_function]` derives, so a
/// caller may name those parameters; `positional` drops it, which is what
/// makes the gateway refuse a keyword call for the definitions whose
/// parameters are all positional-only.
///
/// `positional` is only sound where every parameter is required.  Without a
/// `Signature` a keyword binds to nothing, and the call is refused only
/// because the positional count then falls short — which it does not when the
/// parameter the caller named had a default.  So the two positional-only
/// definitions that do have one keep their signature and merely accept a
/// keyword call the upstream module turns away; silently substituting the
/// default for what the caller wrote would be the worse answer.
macro_rules! install {
    ($ns_slot:expr, keywords: [$($kw:ident),* $(,)?], positional: [$($pos:ident),* $(,)?]) => {
        $(
            pyre_interpreter::__pyre_put_new!(
                $ns_slot,
                stringify!($kw),
                pyre_interpreter::gateway::with_module(
                    "_winapi",
                    pyre_interpreter::make_module_builtin_function_with_arity_and_maybe_sig(
                        stringify!($kw),
                        $kw,
                        ::paste::paste! { [<$kw _pyre_arity>]() },
                        ::paste::paste! { [<$kw _pyre_sig>]() },
                    ),
                )
            );
        )*
        $(
            pyre_interpreter::__pyre_put_new!(
                $ns_slot,
                stringify!($pos),
                pyre_interpreter::gateway::with_module(
                    "_winapi",
                    pyre_interpreter::make_module_builtin_function_with_arity_and_maybe_sig(
                        stringify!($pos),
                        $pos,
                        ::paste::paste! { [<$pos _pyre_arity>]() },
                        ::std::option::Option::None,
                    ),
                )
            );
        )*
    };
}

pub fn install(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // `LOCALE_NAME_USER_DEFAULT` is the null locale name, which is `None`
    // rather than a string; the other two are the reserved names themselves.
    pyre_interpreter::__pyre_put_new!(ns_slot, "LOCALE_NAME_INVARIANT", pyre_object::w_str_new(""));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "LOCALE_NAME_SYSTEM_DEFAULT",
        pyre_object::w_str_new("!x-sys-default-locale")
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "LOCALE_NAME_USER_DEFAULT", w_none());
    #[cfg(not(feature = "sandbox"))]
    {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "Overlapped",
            super::overlapped::overlapped_type()
        );
        install!(ns_slot, keywords: [ConnectNamedPipe, ReadFile, WriteFile], positional: []);
    }
    install!(
        ns_slot,
        keywords: [
            BatchedWaitForMultipleObjects,
            CopyFile2,
            CreateEventW,
            CreateMutexW,
            GetLongPathName,
            GetShortPathName,
            LCMapStringEx,
            OpenEventW,
            OpenMutexW,
            // Positional-only upstream, but with an optional parameter, so it
            // keeps its `Signature` here instead of joining `positional`
            // (`install!`'s doc gives the soundness rule).
            PeekNamedPipe,
            WaitForMultipleObjects,
            ReleaseMutex,
            ResetEvent,
            SetEvent,
            _mimetypes_read_windows_registry,
        ],
        positional: [
            CreateFile,
            CreateFileMapping,
            CreateJunction,
            CreateNamedPipe,
            ExitProcess,
            GetACP,
            GetVersion,
            MapViewOfFile,
            OpenFileMapping,
            OpenProcess,
            SetNamedPipeHandleState,
            UnmapViewOfFile,
            VirtualQuerySize,
            WaitNamedPipe,
        ]
    );
}
