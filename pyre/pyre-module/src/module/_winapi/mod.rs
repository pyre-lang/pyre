//! _winapi module — the Win32 calls the stdlib's Windows branches reach for.
//!
//! The Windows build reports `sys.platform == "win32"` and installs the posix
//! module under `os.name == "nt"`, so both the stdlib's `os.name == "nt"` and
//! its `sys.platform == "win32"` branches are live.  The latter reach for
//! `_winapi`, and without the module `import shutil` — hence `tempfile`, and
//! everything downstream — fails outright.
//!
//! `lib_pypy/_winapi.py` predates the stdlib revision pyre ships and stops
//! well short of it, so the shapes here follow `PC/_winapi.c` and the Win32
//! headers instead.  The module comes in three parts: the constants and the
//! few calls made straight against `windows-sys` are below; [`process`] is
//! the launch half `subprocess.Popen` spawns through — `CreatePipe`,
//! `DuplicateHandle`, `GetStdHandle`, `CreateProcess`, `TerminateProcess`,
//! `GetFileType` — and [`host`] is everything
//! `rustpython_host_env::winapi` backs, which is the named-pipe, event,
//! mutex, file-mapping, path and locale surface `multiprocessing`,
//! `ntpath.normcase`, `mimetypes` and `shutil.copy2` walk.
//!
//! `subprocess` picks its Windows implementation on the presence of `msvcrt`,
//! so its module body imports the process/priority constants and captures
//! `CloseHandle`/`WaitForSingleObject`/`GetExitCodeProcess` as default
//! arguments.
//!
//! [`overlapped`] holds the `Overlapped` object and the asynchronous form of
//! `ConnectNamedPipe`, `ReadFile` and `WriteFile` that produces one, which is
//! what `multiprocessing.connection`'s `PipeConnection` is written against.

use windows_sys::Win32::Foundation::HANDLE;

#[cfg(feature = "host_env")]
use rustpython_host_env::winapi as host_winapi;

use pyre_interpreter::PyError;

/// Map the current thread's last OS error to an `OSError`
/// (`PyErr_SetFromWindowsErr`): the code is the one `winerror` reports.
fn last_os_error() -> pyre_interpreter::PyError {
    win32_err(std::io::Error::last_os_error())
}

/// The `OSError` a failed Win32 call reports (`PyErr_SetFromWindowsErr`).
fn win32_err(error: std::io::Error) -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::os_error_win32_syscall2(
        error.raw_os_error().unwrap_or(0),
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
    )
}

/// [`win32_err`] for a call that reports its error code rather than setting
/// the thread's.
fn win32_code(code: u32) -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::os_error_win32_syscall2(
        code as i32,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
    )
}

/// How a call names an integer argument in the `TypeError` it raises for one
/// that is not an integer at all.
enum IntArg<'a> {
    /// The only parameter of `function`.
    Only(&'a str),
    /// Parameter `position` of `function`, counted from one.
    At { function: &'a str, position: usize },
    /// An element of a sequence argument, which the message does not name.
    Element,
}

/// The value an integer parameter carries: read through `__index__`, then
/// taken modulo the width the call passes it in.  Both `_Py_PARSE_UINTPTR`
/// and the `DWORD` converter work that way, so a handle may be written as the
/// negative it is or as the unsigned value it prints as, and a flag word may
/// be written either way round too.
fn masked_int_w(w_value: pyre_object::PyObjectRef, argument: IntArg<'_>) -> Result<i64, PyError> {
    if !unsafe { pyre_object::pyobject::is_int_or_long(w_value) }
        && unsafe { pyre_interpreter::baseobjspace::lookup(w_value, "__index__") }.is_none()
    {
        let got = pyre_interpreter::gateway::short_type_name(w_value);
        return Err(PyError::type_error(match argument {
            IntArg::Only(function) => format!("{function}() argument must be int, not {got}"),
            IntArg::At { function, position } => {
                format!("{function}() argument {position} must be int, not {got}")
            }
            IntArg::Element => format!("argument must be int, not {got}"),
        }));
    }
    // `PyLong_AsNativeBytes` reads the value under
    // `Py_ASNATIVEBYTES_ALLOW_INDEX`, so `__index__` decides it and `__int__`
    // is never asked for one.
    pyre_interpreter::baseobjspace::truncatedint_w(pyre_interpreter::baseobjspace::space_index(
        w_value,
    )?)
}

/// [`masked_int_w`] for a `HANDLE` parameter.
fn handle_w(w_handle: pyre_object::PyObjectRef, argument: IntArg<'_>) -> Result<HANDLE, PyError> {
    Ok(masked_int_w(w_handle, argument)? as isize as HANDLE)
}

/// [`masked_int_w`] for a `DWORD` parameter.
fn dword_w(w_value: pyre_object::PyObjectRef, argument: IntArg<'_>) -> Result<u32, PyError> {
    Ok(masked_int_w(w_value, argument)? as u32)
}

/// The integer a handle comes back as (`PyLong_FromVoidPtr`): the unsigned
/// value, which is how the pseudo handles print.
fn w_handle(handle: HANDLE) -> pyre_object::PyObjectRef {
    let value = handle as usize as u64;
    match i64::try_from(value) {
        Ok(fits) => pyre_object::w_int_new(fits),
        Err(_) => {
            pyre_object::longobject::w_long_new(majit_rlib::rbigint::RBigInt::from(value as i128))
        }
    }
}

#[cfg(feature = "host_env")]
mod host;
// The asynchronous half reaches the host's overlapped I/O directly, so a
// sandbox build leaves it out along with `_overlapped`.
#[cfg(all(feature = "host_env", not(feature = "sandbox")))]
pub mod overlapped;

/// The process-launch half of the module, which `subprocess.Popen` walks in
/// order: `CreatePipe` for each redirected stream, `DuplicateHandle` to make
/// the child's end of it inheritable, `CreateProcess`, then
/// `WaitForSingleObject` / `GetExitCodeProcess` / `TerminateProcess` on what
/// comes back.  Backed by `rustpython_host_env::winapi`, so it is compiled
/// only where that is — which is also the only build that has `msvcrt`, the
/// module `subprocess` picks its Windows implementation from.
#[cfg(feature = "host_env")]
mod process {
    use pyre_object::{PyObjectRef, w_int_new, w_none, w_tuple_new};
    use rustpython_host_env::winapi as host_winapi;
    use rustpython_host_env::winapi::{CSTR_EQUAL, HANDLE, LCMAP_UPPERCASE};
    use rustpython_wtf8::Wtf8Buf;
    use widestring::WideCString;

    use super::{IntArg, handle_w, w_handle, win32_err};

    fn arg(
        args: &[PyObjectRef],
        index: usize,
        name: &str,
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        args.get(index).copied().ok_or_else(|| {
            pyre_interpreter::PyError::type_error(format!("{name}() missing required argument"))
        })
    }

    /// `_winapi.GetStdHandle(std_handle)` — `None` for a stream the process
    /// does not have, which is what the caller tests for before making a pipe
    /// of its own.
    pub fn get_std_handle(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let id = super::dword_w(arg(args, 0, "GetStdHandle")?, IntArg::Only("GetStdHandle"))?;
        match host_winapi::get_std_handle(id) {
            Ok(Some(handle)) => Ok(w_handle(handle)),
            Ok(None) => Ok(w_none()),
            Err(e) => Err(win32_err(e)),
        }
    }

    /// `_winapi.GetCurrentProcess()` — the pseudo handle, which names this
    /// process to `DuplicateHandle` without being a handle to close.
    pub fn get_current_process(
        _args: &[PyObjectRef],
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        Ok(w_handle(host_winapi::get_current_process()))
    }

    /// `_winapi.GetFileType(handle)` — a console handle is the one kind that
    /// cannot be passed in an inherited handle list.
    pub fn get_file_type(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let handle = handle_w(arg(args, 0, "GetFileType")?, IntArg::Only("GetFileType"))?;
        host_winapi::get_file_type(handle)
            .map(|file_type| w_int_new(file_type as i64))
            .map_err(win32_err)
    }

    /// `_winapi.GetLastError()`
    pub fn get_last_error(_args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        Ok(w_int_new(host_winapi::get_last_error() as i64))
    }

    /// `_winapi.GetModuleFileName(module_handle)` — the path the module was
    /// loaded from, or the executable's own path for handle 0.
    ///
    /// `sysconfig._init_non_posix` calls it on `sys.dllhandle` to locate the
    /// install prefix, so it is on the path of every `import sysconfig` here.
    ///
    /// The `MAX_PATH` buffer is the interface, not a shortcut: this call is
    /// specified to hand back one fixed-size buffer, so a longer path comes
    /// back truncated rather than retried in a growing loop.  `initpath.py:308-315
    /// _get_module_file_name` allocates exactly `_MAX_PATH` and gives up when
    /// the result does not fit, and the module this shadows declares
    /// `WCHAR filename[MAX_PATH]` and forces `filename[MAX_PATH - 1] = '\0'`
    /// before returning it.  Growing the buffer here would be the deviation.
    ///
    /// The length is then the NUL scan rather than the count the call reported,
    /// because on a truncation the call reports the whole buffer while the
    /// loader has already terminated the string inside it — taking the reported
    /// count would append that terminator to the path.
    pub fn get_module_file_name(
        args: &[PyObjectRef],
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        const MAX_PATH: usize = 260;
        let module = handle_w(
            arg(args, 0, "GetModuleFileName")?,
            IntArg::Only("GetModuleFileName"),
        )? as *mut core::ffi::c_void;
        let mut buffer = [0u16; MAX_PATH];
        let length = host_winapi::get_module_file_name(module, &mut buffer);
        if length == 0 {
            return Err(super::last_os_error());
        }
        buffer[MAX_PATH - 1] = 0;
        let filename = &buffer[..buffer.iter().position(|&u| u == 0).unwrap_or(MAX_PATH)];
        Ok(pyre_object::w_str_from_wtf8_managed(
            rustpython_wtf8::Wtf8Buf::from_wide(filename),
        ))
    }

    /// `_winapi.TerminateProcess(handle, exit_code)`
    pub fn terminate_process(
        args: &[PyObjectRef],
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let handle = handle_w(
            arg(args, 0, "TerminateProcess")?,
            IntArg::At {
                function: "TerminateProcess",
                position: 1,
            },
        )?;
        let exit_code = super::dword_w(
            arg(args, 1, "TerminateProcess")?,
            IntArg::At {
                function: "TerminateProcess",
                position: 2,
            },
        )?;
        if host_winapi::terminate_process(handle, exit_code) == 0 {
            return Err(super::last_os_error());
        }
        Ok(w_none())
    }

    /// `_winapi.CreatePipe(pipe_attrs, size)` -> (read, write).  The
    /// attributes argument is the security descriptor, which the call takes
    /// the default of; neither end is inheritable until one is duplicated
    /// into an inheritable copy.
    pub fn create_pipe(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let size = super::dword_w(
            arg(args, 1, "CreatePipe")?,
            IntArg::At {
                function: "CreatePipe",
                position: 2,
            },
        )?;
        let (read, write) = host_winapi::create_pipe(size).map_err(win32_err)?;
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(w_handle(read));
        fields.push(w_handle(write));
        Ok(w_tuple_new(fields.take()))
    }

    /// `_winapi.DuplicateHandle(source_process, source, target_process,
    /// desired_access, inherit_handle, options=0)` -> handle
    pub fn duplicate_handle(
        args: &[PyObjectRef],
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        const NAME: &str = "DuplicateHandle";
        let at = |position| IntArg::At {
            function: NAME,
            position,
        };
        let source_process = handle_w(arg(args, 0, NAME)?, at(1))?;
        let source = handle_w(arg(args, 1, NAME)?, at(2))?;
        let target_process = handle_w(arg(args, 2, NAME)?, at(3))?;
        let access = super::dword_w(arg(args, 3, NAME)?, at(4))?;
        let inherit = pyre_interpreter::baseobjspace::index_c_int_w(arg(args, 4, NAME)?)?;
        let options = match args.get(5) {
            Some(&w) => super::dword_w(w, at(6))?,
            None => 0,
        };
        host_winapi::duplicate_handle(
            source_process,
            source,
            target_process,
            access,
            inherit,
            options,
        )
        .map(w_handle)
        .map_err(win32_err)
    }

    /// The invariant locale's `LCMAP_UPPERCASE` mapping of an environment
    /// name, which is the key the block is ordered by
    /// (`sortenvironmentkey`).  The mapped names are compared the way two
    /// `str` are, which is code point order, so the mapping comes back as
    /// WTF-8: byte order over that is exactly code point order.
    ///
    /// `str::to_uppercase` cannot stand in for it.  `ß` → `SS`, `ı` → `I` and
    /// `ﬁ` → `FI` fold two names Win32 keeps apart into one, and the block
    /// would then carry only the later of the pair.
    ///
    /// `LCMapStringEx` reports `ERROR_INVALID_PARAMETER` for an empty name,
    /// and that is the error an empty name raises.
    fn environment_sort_key(name: &[u16]) -> Result<Wtf8Buf, pyre_interpreter::PyError> {
        // `LOCALE_NAME_INVARIANT` is the empty locale name.
        let mapped = host_winapi::lc_map_string_ex(&WideCString::new(), LCMAP_UPPERCASE, name)
            .map_err(win32_err)?;
        Ok(Wtf8Buf::from_wide(&mapped))
    }

    /// Whether two environment names are one name, which is the ordinal
    /// case-insensitive match `CompareStringOrdinal` makes.  Both are passed
    /// by length, so neither has to be terminated.
    fn environment_names_equal(left: &[u16], right: &[u16]) -> bool {
        host_winapi::compare_string_ordinal(left, right, true) == CSTR_EQUAL
    }

    /// The environment block `CreateProcess` takes: every `KEY=value` in
    /// order, each terminated, the whole thing terminated again
    /// (`getenvironment`).
    ///
    /// Read through the mapping protocol, which is what `getenvironment`
    /// takes: `subprocess` hands `os.environ` straight on, and that is a
    /// `MutableMapping` rather than a dict.  `keys()` names the variables and
    /// each one is subscripted for its value, so a mapping that computes its
    /// values gets to.
    ///
    /// The names are ordered and folded before any of them is subscripted
    /// (`normalize_environment`): the block Win32 reads has to be sorted, and
    /// a name spelled twice under different casing is one variable, whose
    /// last spelling — and that spelling's value — is the one carried.  Names
    /// and values travel as UTF-16 throughout, so an unpaired surrogate
    /// reaches the child as the code unit it is.
    fn environment_block(w_env: PyObjectRef) -> Result<Vec<u16>, pyre_interpreter::PyError> {
        const EQUALS: u16 = b'=' as u16;

        // What a subscript cannot be taken from is no mapping.
        if unsafe { pyre_interpreter::baseobjspace::lookup(w_env, "__getitem__") }.is_none() {
            return Err(pyre_interpreter::PyError::type_error(
                "environment must be dictionary or None",
            ));
        }
        let w_keys = pyre_interpreter::baseobjspace::call_method(w_env, "keys", &[]);
        if w_keys.is_null() {
            return Err(
                pyre_interpreter::call::take_call_error().unwrap_or_else(|| {
                    pyre_interpreter::PyError::type_error("environment must be dictionary or None")
                }),
            );
        }
        let keys = pyre_interpreter::baseobjspace::unpackiterable(w_keys, -1)?;
        // `getitem` runs the mapping's own `__getitem__`, which allocates, so
        // the mapping and every key are published and read back per iteration
        // rather than kept in plain locals.
        let _env_roots = pyre_object::gc_roots::push_roots();
        let env_slot = pyre_object::gc_roots::pin_roots(&[w_env]);
        let keys_base = pyre_object::gc_roots::pin_roots(&keys);

        // Each name with its sort key and the slot its key object sits in.
        // Nothing here holds an object, so the subscripts below are free to
        // collect.
        let mut names = Vec::with_capacity(keys.len());
        for i in 0..keys.len() {
            let w_key = pyre_object::gc_roots::shadow_stack_get(keys_base + i);
            if !unsafe { pyre_object::is_str(w_key) } {
                // The sort maps the name through `PyUnicode_AsWideCharString`,
                // which answers a name that is no string with
                // `PyErr_BadArgument` rather than a message of its own.
                return Err(pyre_interpreter::PyError::type_error(
                    "bad argument type for built-in operation",
                ));
            }
            let name = super::host::wide_units(w_key);
            let sort_key = environment_sort_key(&name)?;
            names.push((sort_key, i, name));
        }
        // Stable, so names that map alike keep the order the mapping gave
        // them and the last of the run is the one kept below.
        names.sort_by(|left, right| left.0.cmp(&right.0));

        let mut entries = Vec::with_capacity(names.len());
        let mut names = names.into_iter().peekable();
        while let Some((_, index, name)) = names.next() {
            // A name the next one matches is shadowed by it
            // (`dedup_environment_keys`), and a shadowed name is never
            // subscripted at all.
            if let Some((_, _, next)) = names.peek() {
                // `compare_string_ordinal` spells both operands with
                // `PyUnicode_AsWideCharString(str, NULL)`, which refuses a
                // name carrying a NUL before the comparison happens.  The
                // refusal therefore reaches names the per-entry scan below
                // never gets to; the last name enters no comparison and so is
                // spelled only there.
                if name.contains(&0) || next.contains(&0) {
                    return Err(pyre_interpreter::PyError::value_error(
                        "embedded null character",
                    ));
                }
                if environment_names_equal(&name, next) {
                    continue;
                }
            }
            let _entry_roots = pyre_object::gc_roots::push_roots();
            let w_value = pyre_interpreter::baseobjspace::getitem(
                pyre_object::gc_roots::shadow_stack_get(env_slot),
                pyre_object::gc_roots::shadow_stack_get(keys_base + index),
            )?;
            let value_slot = pyre_object::gc_roots::pin_roots(&[w_value]);
            let w_value = pyre_object::gc_roots::shadow_stack_get(value_slot);
            // A value that is no string is reported only once the whole
            // mapping has been read, which is where `getenvironment` reports
            // it — after `normalize_environment` has subscripted every name.
            let value =
                unsafe { pyre_object::is_str(w_value) }.then(|| super::host::wide_units(w_value));
            entries.push((name, value));
        }

        let mut block = Vec::new();
        for (name, value) in entries {
            let Some(value) = value else {
                return Err(pyre_interpreter::PyError::type_error(
                    "environment can only contain strings",
                ));
            };
            if name.contains(&0) || value.contains(&0) {
                return Err(pyre_interpreter::PyError::value_error(
                    "embedded null character",
                ));
            }
            // The `=` is looked for from the second unit on, because a name
            // that starts with one is a hidden per-drive variable and the
            // block is allowed to carry it.  A code unit search is the code
            // point search `PyUnicode_FindChar` makes: `=` is in the basic
            // plane, and the second unit of an astral name is a low surrogate
            // rather than any character.
            if name.is_empty() || name[1..].contains(&EQUALS) {
                return Err(pyre_interpreter::PyError::value_error(
                    "illegal environment variable name",
                ));
            }
            block.extend_from_slice(&name);
            block.push(EQUALS);
            block.extend_from_slice(&value);
            block.push(0);
        }
        // A block ends with two NULs — one for the last string and one for
        // the block — so an empty one is the two on their own.
        if block.is_empty() {
            block.push(0);
        }
        block.push(0);
        Ok(block)
    }

    /// Read one of `STARTUPINFO`'s fields.  A field left as `None` is the
    /// zero the structure is built with (`getulong` / `gethandle`).
    fn startup_info_field(
        w_info: PyObjectRef,
        name: &str,
    ) -> Result<i64, pyre_interpreter::PyError> {
        let w_value = pyre_interpreter::baseobjspace::getattr_str(w_info, name)?;
        if unsafe { pyre_object::is_none(w_value) } {
            return Ok(0);
        }
        pyre_interpreter::baseobjspace::int_w(w_value)
    }

    /// The handles `lpAttributeList["handle_list"]` names, which are the ones
    /// the child inherits and the only ones it does.
    ///
    /// An empty list is no list: an attribute list has to carry at least one
    /// handle, and `getattributelist` drops it rather than building one that
    /// `CreateProcess` answers `ERROR_BAD_LENGTH` to.  `subprocess.STARTUPINFO`
    /// starts out with exactly that empty list.
    fn handle_list(w_info: PyObjectRef) -> Result<Option<Vec<usize>>, pyre_interpreter::PyError> {
        let w_attrs = pyre_interpreter::baseobjspace::getattr_str(w_info, "lpAttributeList")?;
        if w_attrs.is_null() || !unsafe { pyre_object::is_dict(w_attrs) } {
            return Ok(None);
        }
        let Some(w_handles) = (unsafe { pyre_object::w_dict_getitem_str(w_attrs, "handle_list") })
        else {
            return Ok(None);
        };
        let items = pyre_interpreter::baseobjspace::unpackiterable(w_handles, -1)?;
        if items.is_empty() {
            return Ok(None);
        }
        let mut handles = Vec::with_capacity(items.len());
        for w_handle in items {
            handles.push(pyre_interpreter::baseobjspace::int_w(w_handle)? as usize);
        }
        Ok(Some(handles))
    }

    /// `_winapi.CreateProcess(application_name, command_line, proc_attrs,
    /// thread_attrs, inherit_handles, creation_flags, env_mapping,
    /// current_directory, startup_info)` -> (process, thread, pid, tid)
    ///
    /// The two attribute arguments are security descriptors, which the call
    /// takes the defaults of.
    pub fn create_process(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        use super::host::{WideArg, wide_or_none};

        let application_name = wide_or_none(arg(args, 0, "CreateProcess")?, WideArg::Unnamed)?;
        // `CreateProcessW` writes into the command line it is given, so it
        // takes a buffer rather than the string itself.
        let mut command_line = wide_or_none(
            arg(args, 1, "CreateProcess")?,
            WideArg::Numbered {
                function: "CreateProcess",
                position: 2,
            },
        )?
        .map(|line| line.into_vec_with_nul());
        let inherit_handles =
            pyre_interpreter::baseobjspace::int_w(arg(args, 4, "CreateProcess")?)?;
        let creation_flags =
            pyre_interpreter::baseobjspace::c_uint_w(arg(args, 5, "CreateProcess")?)?;
        let w_env = arg(args, 6, "CreateProcess")?;
        let environment = if w_env.is_null() || unsafe { pyre_object::is_none(w_env) } {
            None
        } else {
            Some(environment_block(w_env)?)
        };
        let current_directory = wide_or_none(arg(args, 7, "CreateProcess")?, WideArg::Unnamed)?;
        let w_startup_info = arg(args, 8, "CreateProcess")?;
        let startup_info = host_winapi::StartupInfoData {
            flags: startup_info_field(w_startup_info, "dwFlags")? as u32,
            show_window: startup_info_field(w_startup_info, "wShowWindow")? as u16,
            std_input: startup_info_field(w_startup_info, "hStdInput")? as isize as HANDLE,
            std_output: startup_info_field(w_startup_info, "hStdOutput")? as isize as HANDLE,
            std_error: startup_info_field(w_startup_info, "hStdError")? as isize as HANDLE,
        };
        let handles = handle_list(w_startup_info)?;

        let info = host_winapi::create_process(
            application_name.as_deref(),
            command_line.as_deref_mut(),
            inherit_handles as i32,
            creation_flags,
            environment.as_deref(),
            current_directory.as_deref(),
            startup_info,
            handles,
        )
        .map_err(win32_err)?;
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(w_handle(info.process));
        fields.push(w_handle(info.thread));
        fields.push(w_int_new(info.process_id as i64));
        fields.push(w_int_new(info.thread_id as i64));
        Ok(w_tuple_new(fields.take()))
    }
}

#[cfg(feature = "host_env")]
macro_rules! win_const {
    (INFINITE) => {
        host_winapi::INFINITE_TIMEOUT as i64
    };
    ($name:ident) => {
        host_winapi::$name as i64
    };
}

#[cfg(not(feature = "host_env"))]
macro_rules! win_const {
    (INFINITE) => {
        0xFFFF_FFFFu32 as i64
    };
    (COPY_FILE_ALLOW_DECRYPTED_DESTINATION) => {
        0x0000_0008 as i64
    };
    (COPY_FILE_COPY_SYMLINK) => {
        0x0000_0800 as i64
    };
    (COPY_FILE_DIRECTORY) => {
        0x0000_0080 as i64
    };
    (COPY_FILE_FAIL_IF_EXISTS) => {
        0x0000_0001 as i64
    };
    (COPY_FILE_NO_BUFFERING) => {
        0x0000_1000 as i64
    };
    (COPY_FILE_NO_OFFLOAD) => {
        0x0004_0000 as i64
    };
    (COPY_FILE_OPEN_SOURCE_FOR_WRITE) => {
        0x0000_0004 as i64
    };
    (COPY_FILE_REQUEST_COMPRESSED_TRAFFIC) => {
        0x1000_0000 as i64
    };
    (COPY_FILE_REQUEST_SECURITY_PRIVILEGES) => {
        0x0000_2000 as i64
    };
    (COPY_FILE_RESTARTABLE) => {
        0x0000_0002 as i64
    };
    (COPY_FILE_RESUME_FROM_PAUSE) => {
        0x0000_4000 as i64
    };
    (COPYFILE2_CALLBACK_CHUNK_STARTED) => {
        1 as i64
    };
    (COPYFILE2_CALLBACK_CHUNK_FINISHED) => {
        2 as i64
    };
    (COPYFILE2_CALLBACK_STREAM_STARTED) => {
        3 as i64
    };
    (COPYFILE2_CALLBACK_STREAM_FINISHED) => {
        4 as i64
    };
    (COPYFILE2_CALLBACK_POLL_CONTINUE) => {
        5 as i64
    };
    (COPYFILE2_CALLBACK_ERROR) => {
        6 as i64
    };
    (COPYFILE2_PROGRESS_CONTINUE) => {
        0 as i64
    };
    (COPYFILE2_PROGRESS_CANCEL) => {
        1 as i64
    };
    (COPYFILE2_PROGRESS_STOP) => {
        2 as i64
    };
    (COPYFILE2_PROGRESS_QUIET) => {
        3 as i64
    };
    (COPYFILE2_PROGRESS_PAUSE) => {
        4 as i64
    };
    (ERROR_ACCESS_DENIED) => {
        5 as i64
    };
    (ERROR_PRIVILEGE_NOT_HELD) => {
        1314 as i64
    };
    (ERROR_ALREADY_EXISTS) => {
        183 as i64
    };
    (ERROR_BROKEN_PIPE) => {
        109 as i64
    };
    (ERROR_IO_PENDING) => {
        997 as i64
    };
    (ERROR_MORE_DATA) => {
        234 as i64
    };
    (ERROR_NETNAME_DELETED) => {
        64 as i64
    };
    (ERROR_NO_DATA) => {
        232 as i64
    };
    (ERROR_NO_SYSTEM_RESOURCES) => {
        1450 as i64
    };
    (ERROR_OPERATION_ABORTED) => {
        995 as i64
    };
    (ERROR_PIPE_BUSY) => {
        231 as i64
    };
    (ERROR_PIPE_CONNECTED) => {
        535 as i64
    };
    (ERROR_SEM_TIMEOUT) => {
        121 as i64
    };
    (STD_INPUT_HANDLE) => {
        0xFFFF_FFF6u32 as i64
    };
    (STD_OUTPUT_HANDLE) => {
        0xFFFF_FFF5u32 as i64
    };
    (STD_ERROR_HANDLE) => {
        0xFFFF_FFF4u32 as i64
    };
    (SW_HIDE) => {
        0 as i64
    };
    (STARTF_USESHOWWINDOW) => {
        0x0000_0001 as i64
    };
    (STARTF_USESTDHANDLES) => {
        0x0000_0100 as i64
    };
    (STARTF_FORCEONFEEDBACK) => {
        0x0000_0040 as i64
    };
    (STARTF_FORCEOFFFEEDBACK) => {
        0x0000_0080 as i64
    };
    (STARTF_USESIZE) => {
        0x0000_0002 as i64
    };
    (STARTF_USEPOSITION) => {
        0x0000_0004 as i64
    };
    (STARTF_USECOUNTCHARS) => {
        0x0000_0008 as i64
    };
    (STARTF_USEFILLATTRIBUTE) => {
        0x0000_0010 as i64
    };
    (STARTF_RUNFULLSCREEN) => {
        0x0000_0020 as i64
    };
    (STARTF_USEHOTKEY) => {
        0x0000_0200 as i64
    };
    (STARTF_TITLEISLINKNAME) => {
        0x0000_0800 as i64
    };
    (STARTF_TITLEISAPPID) => {
        0x0000_1000 as i64
    };
    (STARTF_PREVENTPINNING) => {
        0x0000_2000 as i64
    };
    (STARTF_UNTRUSTEDSOURCE) => {
        0x0000_8000 as i64
    };
    (CREATE_NEW_CONSOLE) => {
        0x0000_0010 as i64
    };
    (CREATE_NEW_PROCESS_GROUP) => {
        0x0000_0200 as i64
    };
    (CREATE_NO_WINDOW) => {
        0x0800_0000 as i64
    };
    (DETACHED_PROCESS) => {
        0x0000_0008 as i64
    };
    (CREATE_DEFAULT_ERROR_MODE) => {
        0x0400_0000 as i64
    };
    (CREATE_BREAKAWAY_FROM_JOB) => {
        0x0100_0000 as i64
    };
    (ABOVE_NORMAL_PRIORITY_CLASS) => {
        0x0000_8000 as i64
    };
    (BELOW_NORMAL_PRIORITY_CLASS) => {
        0x0000_4000 as i64
    };
    (HIGH_PRIORITY_CLASS) => {
        0x0000_0080 as i64
    };
    (IDLE_PRIORITY_CLASS) => {
        0x0000_0040 as i64
    };
    (NORMAL_PRIORITY_CLASS) => {
        0x0000_0020 as i64
    };
    (REALTIME_PRIORITY_CLASS) => {
        0x0000_0100 as i64
    };
    (WAIT_OBJECT_0) => {
        0 as i64
    };
    (WAIT_ABANDONED_0) => {
        0x0000_0080 as i64
    };
    (WAIT_TIMEOUT) => {
        0x0000_0102 as i64
    };
    (STILL_ACTIVE) => {
        259 as i64
    };
    (DUPLICATE_SAME_ACCESS) => {
        0x0000_0002 as i64
    };
    (DUPLICATE_CLOSE_SOURCE) => {
        0x0000_0001 as i64
    };
    (FILE_TYPE_UNKNOWN) => {
        0x0000 as i64
    };
    (FILE_TYPE_DISK) => {
        0x0001 as i64
    };
    (FILE_TYPE_CHAR) => {
        0x0002 as i64
    };
    (FILE_TYPE_PIPE) => {
        0x0003 as i64
    };
    (FILE_TYPE_REMOTE) => {
        0x8000 as i64
    };
    (PROCESS_ALL_ACCESS) => {
        0x001F_FFFF as i64
    };
    (PROCESS_DUP_HANDLE) => {
        0x0000_0040 as i64
    };
    (SYNCHRONIZE) => {
        0x0010_0000 as i64
    };
    (GENERIC_READ) => {
        0x8000_0000u32 as i64
    };
    (GENERIC_WRITE) => {
        0x4000_0000 as i64
    };
    (FILE_GENERIC_READ) => {
        0x0012_0089 as i64
    };
    (FILE_GENERIC_WRITE) => {
        0x0012_0116 as i64
    };
    (OPEN_EXISTING) => {
        3 as i64
    };
    (FILE_FLAG_OVERLAPPED) => {
        0x4000_0000 as i64
    };
    (FILE_FLAG_FIRST_PIPE_INSTANCE) => {
        0x0008_0000 as i64
    };
    (PIPE_ACCESS_INBOUND) => {
        0x0000_0001 as i64
    };
    (PIPE_ACCESS_DUPLEX) => {
        0x0000_0003 as i64
    };
    (PIPE_READMODE_MESSAGE) => {
        0x0000_0002 as i64
    };
    (PIPE_TYPE_MESSAGE) => {
        0x0000_0004 as i64
    };
    (PIPE_WAIT) => {
        0x0000_0000 as i64
    };
    (PIPE_UNLIMITED_INSTANCES) => {
        255 as i64
    };
    (NMPWAIT_WAIT_FOREVER) => {
        0xFFFF_FFFFu32 as i64
    };
    (PAGE_NOACCESS) => {
        0x0000_0001 as i64
    };
    (PAGE_READONLY) => {
        0x0000_0002 as i64
    };
    (PAGE_READWRITE) => {
        0x0000_0004 as i64
    };
    (PAGE_WRITECOPY) => {
        0x0000_0008 as i64
    };
    (PAGE_EXECUTE) => {
        0x0000_0010 as i64
    };
    (PAGE_EXECUTE_READ) => {
        0x0000_0020 as i64
    };
    (PAGE_EXECUTE_READWRITE) => {
        0x0000_0040 as i64
    };
    (PAGE_EXECUTE_WRITECOPY) => {
        0x0000_0080 as i64
    };
    (PAGE_GUARD) => {
        0x0000_0100 as i64
    };
    (PAGE_NOCACHE) => {
        0x0000_0200 as i64
    };
    (PAGE_WRITECOMBINE) => {
        0x0000_0400 as i64
    };
    (FILE_MAP_COPY) => {
        0x0000_0001 as i64
    };
    (FILE_MAP_WRITE) => {
        0x0000_0002 as i64
    };
    (FILE_MAP_READ) => {
        0x0000_0004 as i64
    };
    (FILE_MAP_EXECUTE) => {
        0x0000_0020 as i64
    };
    (FILE_MAP_ALL_ACCESS) => {
        0x000F_001F as i64
    };
    (SEC_IMAGE) => {
        0x0100_0000 as i64
    };
    (SEC_RESERVE) => {
        0x0400_0000 as i64
    };
    (SEC_COMMIT) => {
        0x0800_0000 as i64
    };
    (SEC_NOCACHE) => {
        0x1000_0000 as i64
    };
    (SEC_WRITECOMBINE) => {
        0x4000_0000 as i64
    };
    (SEC_LARGE_PAGES) => {
        0x8000_0000u32 as i64
    };
    (MEM_COMMIT) => {
        0x0000_1000 as i64
    };
    (MEM_RESERVE) => {
        0x0000_2000 as i64
    };
    (MEM_FREE) => {
        0x0001_0000 as i64
    };
    (MEM_PRIVATE) => {
        0x0002_0000 as i64
    };
    (MEM_MAPPED) => {
        0x0004_0000 as i64
    };
    (MEM_IMAGE) => {
        0x0100_0000 as i64
    };
    (LCMAP_LOWERCASE) => {
        0x0000_0100 as i64
    };
    (LCMAP_UPPERCASE) => {
        0x0000_0200 as i64
    };
    (LCMAP_TITLECASE) => {
        0x0000_0300 as i64
    };
    (LCMAP_HIRAGANA) => {
        0x0010_0000 as i64
    };
    (LCMAP_KATAKANA) => {
        0x0020_0000 as i64
    };
    (LCMAP_HALFWIDTH) => {
        0x0040_0000 as i64
    };
    (LCMAP_FULLWIDTH) => {
        0x0080_0000 as i64
    };
    (LCMAP_LINGUISTIC_CASING) => {
        0x0100_0000 as i64
    };
    (LCMAP_SIMPLIFIED_CHINESE) => {
        0x0200_0000 as i64
    };
    (LCMAP_TRADITIONAL_CHINESE) => {
        0x0400_0000 as i64
    };
    (LOCALE_NAME_MAX_LENGTH) => {
        85 as i64
    };
}

pyre_interpreter::py_module! {
    "_winapi",
    int_constants: {
        // CopyFileEx / CopyFile2 flags (winbase.h).
        "COPY_FILE_ALLOW_DECRYPTED_DESTINATION" => win_const!(COPY_FILE_ALLOW_DECRYPTED_DESTINATION),
        "COPY_FILE_COPY_SYMLINK" => win_const!(COPY_FILE_COPY_SYMLINK),
        "COPY_FILE_DIRECTORY" => win_const!(COPY_FILE_DIRECTORY),
        "COPY_FILE_FAIL_IF_EXISTS" => win_const!(COPY_FILE_FAIL_IF_EXISTS),
        "COPY_FILE_NO_BUFFERING" => win_const!(COPY_FILE_NO_BUFFERING),
        "COPY_FILE_NO_OFFLOAD" => win_const!(COPY_FILE_NO_OFFLOAD),
        "COPY_FILE_OPEN_SOURCE_FOR_WRITE" => win_const!(COPY_FILE_OPEN_SOURCE_FOR_WRITE),
        "COPY_FILE_REQUEST_COMPRESSED_TRAFFIC" => win_const!(COPY_FILE_REQUEST_COMPRESSED_TRAFFIC),
        "COPY_FILE_REQUEST_SECURITY_PRIVILEGES" => win_const!(COPY_FILE_REQUEST_SECURITY_PRIVILEGES),
        "COPY_FILE_RESTARTABLE" => win_const!(COPY_FILE_RESTARTABLE),
        "COPY_FILE_RESUME_FROM_PAUSE" => win_const!(COPY_FILE_RESUME_FROM_PAUSE),
        // The reasons a `CopyFile2` progress routine is called and the
        // answers it may give (winbase.h).  Nothing here calls one — the
        // extended parameters leave the callback field unset — but the
        // constants are part of the module's surface.
        "COPYFILE2_CALLBACK_CHUNK_STARTED" => win_const!(COPYFILE2_CALLBACK_CHUNK_STARTED),
        "COPYFILE2_CALLBACK_CHUNK_FINISHED" => win_const!(COPYFILE2_CALLBACK_CHUNK_FINISHED),
        "COPYFILE2_CALLBACK_STREAM_STARTED" => win_const!(COPYFILE2_CALLBACK_STREAM_STARTED),
        "COPYFILE2_CALLBACK_STREAM_FINISHED" => win_const!(COPYFILE2_CALLBACK_STREAM_FINISHED),
        "COPYFILE2_CALLBACK_POLL_CONTINUE" => win_const!(COPYFILE2_CALLBACK_POLL_CONTINUE),
        "COPYFILE2_CALLBACK_ERROR" => win_const!(COPYFILE2_CALLBACK_ERROR),
        "COPYFILE2_PROGRESS_CONTINUE" => win_const!(COPYFILE2_PROGRESS_CONTINUE),
        "COPYFILE2_PROGRESS_CANCEL" => win_const!(COPYFILE2_PROGRESS_CANCEL),
        "COPYFILE2_PROGRESS_STOP" => win_const!(COPYFILE2_PROGRESS_STOP),
        "COPYFILE2_PROGRESS_QUIET" => win_const!(COPYFILE2_PROGRESS_QUIET),
        "COPYFILE2_PROGRESS_PAUSE" => win_const!(COPYFILE2_PROGRESS_PAUSE),
        // System error codes (winerror.h) a caller compares
        // `OSError.winerror` against to decide whether to retry.
        "ERROR_ACCESS_DENIED" => win_const!(ERROR_ACCESS_DENIED),
        "ERROR_PRIVILEGE_NOT_HELD" => win_const!(ERROR_PRIVILEGE_NOT_HELD),
        "ERROR_ALREADY_EXISTS" => win_const!(ERROR_ALREADY_EXISTS),
        "ERROR_BROKEN_PIPE" => win_const!(ERROR_BROKEN_PIPE),
        "ERROR_IO_PENDING" => win_const!(ERROR_IO_PENDING),
        "ERROR_MORE_DATA" => win_const!(ERROR_MORE_DATA),
        "ERROR_NETNAME_DELETED" => win_const!(ERROR_NETNAME_DELETED),
        "ERROR_NO_DATA" => win_const!(ERROR_NO_DATA),
        "ERROR_NO_SYSTEM_RESOURCES" => win_const!(ERROR_NO_SYSTEM_RESOURCES),
        "ERROR_OPERATION_ABORTED" => win_const!(ERROR_OPERATION_ABORTED),
        "ERROR_PIPE_BUSY" => win_const!(ERROR_PIPE_BUSY),
        "ERROR_PIPE_CONNECTED" => win_const!(ERROR_PIPE_CONNECTED),
        "ERROR_SEM_TIMEOUT" => win_const!(ERROR_SEM_TIMEOUT),
        // `subprocess` imports these at module load (its Windows branch, taken
        // once `msvcrt` exists) — GetStdHandle ids, ShowWindow/STARTUPINFO
        // flags, and CreateProcess creation/priority flags (winbase.h,
        // processthreadsapi.h).  Exposed as the unsigned DWORD values.
        "STD_INPUT_HANDLE" => win_const!(STD_INPUT_HANDLE),
        "STD_OUTPUT_HANDLE" => win_const!(STD_OUTPUT_HANDLE),
        "STD_ERROR_HANDLE" => win_const!(STD_ERROR_HANDLE),
        "SW_HIDE" => win_const!(SW_HIDE),
        "STARTF_USESHOWWINDOW" => win_const!(STARTF_USESHOWWINDOW),
        "STARTF_USESTDHANDLES" => win_const!(STARTF_USESTDHANDLES),
        "STARTF_FORCEONFEEDBACK" => win_const!(STARTF_FORCEONFEEDBACK),
        "STARTF_FORCEOFFFEEDBACK" => win_const!(STARTF_FORCEOFFFEEDBACK),
        "STARTF_USESIZE" => win_const!(STARTF_USESIZE),
        "STARTF_USEPOSITION" => win_const!(STARTF_USEPOSITION),
        "STARTF_USECOUNTCHARS" => win_const!(STARTF_USECOUNTCHARS),
        "STARTF_USEFILLATTRIBUTE" => win_const!(STARTF_USEFILLATTRIBUTE),
        "STARTF_RUNFULLSCREEN" => win_const!(STARTF_RUNFULLSCREEN),
        "STARTF_USEHOTKEY" => win_const!(STARTF_USEHOTKEY),
        "STARTF_TITLEISLINKNAME" => win_const!(STARTF_TITLEISLINKNAME),
        "STARTF_TITLEISAPPID" => win_const!(STARTF_TITLEISAPPID),
        "STARTF_PREVENTPINNING" => win_const!(STARTF_PREVENTPINNING),
        "STARTF_UNTRUSTEDSOURCE" => win_const!(STARTF_UNTRUSTEDSOURCE),
        "CREATE_NEW_CONSOLE" => win_const!(CREATE_NEW_CONSOLE),
        "CREATE_NEW_PROCESS_GROUP" => win_const!(CREATE_NEW_PROCESS_GROUP),
        "CREATE_NO_WINDOW" => win_const!(CREATE_NO_WINDOW),
        "DETACHED_PROCESS" => win_const!(DETACHED_PROCESS),
        "CREATE_DEFAULT_ERROR_MODE" => win_const!(CREATE_DEFAULT_ERROR_MODE),
        "CREATE_BREAKAWAY_FROM_JOB" => win_const!(CREATE_BREAKAWAY_FROM_JOB),
        "ABOVE_NORMAL_PRIORITY_CLASS" => win_const!(ABOVE_NORMAL_PRIORITY_CLASS),
        "BELOW_NORMAL_PRIORITY_CLASS" => win_const!(BELOW_NORMAL_PRIORITY_CLASS),
        "HIGH_PRIORITY_CLASS" => win_const!(HIGH_PRIORITY_CLASS),
        "IDLE_PRIORITY_CLASS" => win_const!(IDLE_PRIORITY_CLASS),
        "NORMAL_PRIORITY_CLASS" => win_const!(NORMAL_PRIORITY_CLASS),
        "REALTIME_PRIORITY_CLASS" => win_const!(REALTIME_PRIORITY_CLASS),
        // WaitForSingleObject results / GetExitCodeProcess sentinel that
        // `subprocess.Popen._wait`/`poll` capture (winbase.h, ntstatus.h).
        "WAIT_OBJECT_0" => win_const!(WAIT_OBJECT_0),
        "WAIT_ABANDONED_0" => win_const!(WAIT_ABANDONED_0),
        "WAIT_TIMEOUT" => win_const!(WAIT_TIMEOUT),
        "INFINITE" => win_const!(INFINITE),
        "STILL_ACTIVE" => win_const!(STILL_ACTIVE),
        // `DuplicateHandle` options and the handle sentinel (handleapi.h).
        "DUPLICATE_SAME_ACCESS" => win_const!(DUPLICATE_SAME_ACCESS),
        "DUPLICATE_CLOSE_SOURCE" => win_const!(DUPLICATE_CLOSE_SOURCE),
        "NULL" => 0,
        // `GetFileType` answers.  A console handle is the one a caller drops
        // from an inherited handle list (`Popen._filter_handle_list`).
        "FILE_TYPE_UNKNOWN" => win_const!(FILE_TYPE_UNKNOWN),
        "FILE_TYPE_DISK" => win_const!(FILE_TYPE_DISK),
        "FILE_TYPE_CHAR" => win_const!(FILE_TYPE_CHAR),
        "FILE_TYPE_PIPE" => win_const!(FILE_TYPE_PIPE),
        "FILE_TYPE_REMOTE" => win_const!(FILE_TYPE_REMOTE),
        // `OpenProcess`/`DuplicateHandle` access rights (processthreadsapi.h),
        // and the one right every waitable object grants (winnt.h).
        "PROCESS_ALL_ACCESS" => win_const!(PROCESS_ALL_ACCESS),
        "PROCESS_DUP_HANDLE" => win_const!(PROCESS_DUP_HANDLE),
        "SYNCHRONIZE" => win_const!(SYNCHRONIZE),
        // `CreateFile` access rights, share modes and creation dispositions
        // (winnt.h, fileapi.h).
        "GENERIC_READ" => win_const!(GENERIC_READ),
        "GENERIC_WRITE" => win_const!(GENERIC_WRITE),
        "FILE_GENERIC_READ" => win_const!(FILE_GENERIC_READ),
        "FILE_GENERIC_WRITE" => win_const!(FILE_GENERIC_WRITE),
        "OPEN_EXISTING" => win_const!(OPEN_EXISTING),
        "FILE_FLAG_OVERLAPPED" => win_const!(FILE_FLAG_OVERLAPPED),
        "FILE_FLAG_FIRST_PIPE_INSTANCE" => win_const!(FILE_FLAG_FIRST_PIPE_INSTANCE),
        // Named pipe open modes, pipe modes and waits (winbase.h).
        "PIPE_ACCESS_INBOUND" => win_const!(PIPE_ACCESS_INBOUND),
        "PIPE_ACCESS_DUPLEX" => win_const!(PIPE_ACCESS_DUPLEX),
        "PIPE_READMODE_MESSAGE" => win_const!(PIPE_READMODE_MESSAGE),
        "PIPE_TYPE_MESSAGE" => win_const!(PIPE_TYPE_MESSAGE),
        "PIPE_WAIT" => win_const!(PIPE_WAIT),
        "PIPE_UNLIMITED_INSTANCES" => win_const!(PIPE_UNLIMITED_INSTANCES),
        "NMPWAIT_WAIT_FOREVER" => win_const!(NMPWAIT_WAIT_FOREVER),
        // File-mapping protections, view access rights, section attributes
        // and the region states `VirtualQuery` reports (winnt.h, memoryapi.h).
        "PAGE_NOACCESS" => win_const!(PAGE_NOACCESS),
        "PAGE_READONLY" => win_const!(PAGE_READONLY),
        "PAGE_READWRITE" => win_const!(PAGE_READWRITE),
        "PAGE_WRITECOPY" => win_const!(PAGE_WRITECOPY),
        "PAGE_EXECUTE" => win_const!(PAGE_EXECUTE),
        "PAGE_EXECUTE_READ" => win_const!(PAGE_EXECUTE_READ),
        "PAGE_EXECUTE_READWRITE" => win_const!(PAGE_EXECUTE_READWRITE),
        "PAGE_EXECUTE_WRITECOPY" => win_const!(PAGE_EXECUTE_WRITECOPY),
        "PAGE_GUARD" => win_const!(PAGE_GUARD),
        "PAGE_NOCACHE" => win_const!(PAGE_NOCACHE),
        "PAGE_WRITECOMBINE" => win_const!(PAGE_WRITECOMBINE),
        "FILE_MAP_COPY" => win_const!(FILE_MAP_COPY),
        "FILE_MAP_WRITE" => win_const!(FILE_MAP_WRITE),
        "FILE_MAP_READ" => win_const!(FILE_MAP_READ),
        "FILE_MAP_EXECUTE" => win_const!(FILE_MAP_EXECUTE),
        "FILE_MAP_ALL_ACCESS" => win_const!(FILE_MAP_ALL_ACCESS),
        "SEC_IMAGE" => win_const!(SEC_IMAGE),
        "SEC_RESERVE" => win_const!(SEC_RESERVE),
        "SEC_COMMIT" => win_const!(SEC_COMMIT),
        "SEC_NOCACHE" => win_const!(SEC_NOCACHE),
        "SEC_WRITECOMBINE" => win_const!(SEC_WRITECOMBINE),
        "SEC_LARGE_PAGES" => win_const!(SEC_LARGE_PAGES),
        "MEM_COMMIT" => win_const!(MEM_COMMIT),
        "MEM_RESERVE" => win_const!(MEM_RESERVE),
        "MEM_FREE" => win_const!(MEM_FREE),
        "MEM_PRIVATE" => win_const!(MEM_PRIVATE),
        "MEM_MAPPED" => win_const!(MEM_MAPPED),
        "MEM_IMAGE" => win_const!(MEM_IMAGE),
        // `LCMapStringEx` transforms and the longest locale name it takes
        // (winnls.h).  The four that answer with a sort key or a hash rather
        // than text are not among them — `LCMapStringEx` rejects those.
        "LCMAP_LOWERCASE" => win_const!(LCMAP_LOWERCASE),
        "LCMAP_UPPERCASE" => win_const!(LCMAP_UPPERCASE),
        "LCMAP_TITLECASE" => win_const!(LCMAP_TITLECASE),
        "LCMAP_HIRAGANA" => win_const!(LCMAP_HIRAGANA),
        "LCMAP_KATAKANA" => win_const!(LCMAP_KATAKANA),
        "LCMAP_HALFWIDTH" => win_const!(LCMAP_HALFWIDTH),
        "LCMAP_FULLWIDTH" => win_const!(LCMAP_FULLWIDTH),
        "LCMAP_LINGUISTIC_CASING" => win_const!(LCMAP_LINGUISTIC_CASING),
        "LCMAP_SIMPLIFIED_CHINESE" => win_const!(LCMAP_SIMPLIFIED_CHINESE),
        "LCMAP_TRADITIONAL_CHINESE" => win_const!(LCMAP_TRADITIONAL_CHINESE),
        "LOCALE_NAME_MAX_LENGTH" => win_const!(LOCALE_NAME_MAX_LENGTH),
    },
    inline_functions: {
        fn NeedCurrentDirectoryForExePath(
            exe_name: &str,
        ) -> Result<bool, pyre_interpreter::PyError> {
            // The wide string ends at its first NUL, so a name carrying one
            // would be answered for a shorter name than was passed.
            let mut exe_name_w: Vec<u16> = exe_name.encode_utf16().collect();
            if exe_name_w.contains(&0) {
                return Err(pyre_interpreter::PyError::value_error("embedded null character"));
            }
            exe_name_w.push(0);
            #[cfg(not(feature = "host_env"))]
            {
                let _ = exe_name_w;
                return Err(pyre_interpreter::PyError::not_implemented(
                    "_winapi.NeedCurrentDirectoryForExePath requires host_env",
                ));
            }
            #[cfg(feature = "host_env")]
            {
                let exe_name = widestring::WideCStr::from_slice(&exe_name_w)
                    .expect("validated NUL-terminated executable name");
                return Ok(host_winapi::need_current_directory_for_exe_path_w(exe_name));
            }
        }
        // `subprocess.Handle.Close` captures `_winapi.CloseHandle` as a default
        // argument at class-definition time, so the attribute must exist for
        // `import subprocess` to succeed.
        fn CloseHandle(handle: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            let handle = handle_w(handle, IntArg::Only("CloseHandle"))?;
            #[cfg(not(feature = "host_env"))]
            {
                let _ = handle;
                return Err(pyre_interpreter::PyError::not_implemented(
                    "_winapi.CloseHandle requires host_env",
                ));
            }
            #[cfg(feature = "host_env")]
            let ok = host_winapi::close_handle(handle);
            #[cfg(feature = "host_env")]
            if ok == 0 {
                return Err(last_os_error());
            }
            #[cfg(feature = "host_env")]
            Ok(())
        }
        // `subprocess.Popen._wait`/`poll` also capture these as default
        // arguments at import time, and reach them once a launch has a
        // process to wait on.
        fn WaitForSingleObject(
            handle: pyre_object::PyObjectRef,
            milliseconds: pyre_object::PyObjectRef,
        ) -> Result<i64, pyre_interpreter::PyError> {
            const NAME: &str = "WaitForSingleObject";
            let handle = handle_w(handle, IntArg::At { function: NAME, position: 1 })?;
            let milliseconds =
                dword_w(milliseconds, IntArg::At { function: NAME, position: 2 })?;
            // `rpython/rlib/rwin32.py` declares this through `winexternal`,
            // whose default `releasegil='auto'` runs the native call between
            // the rffi around-handlers.  In particular, an infinite wait must
            // let the Python thread which will signal the handle run.
            // The failure code is read inside the released region, which is
            // what `save_err=rffi.RFFI_SAVE_LASTERROR` does — reacquiring goes
            // through `mutex2_lock_timeout`, whose timed wait leaves
            // `ERROR_TIMEOUT` behind on a poll that expired.
            #[cfg(not(feature = "host_env"))]
            {
                let _ = (handle, milliseconds);
                return Err(pyre_interpreter::PyError::not_implemented(
                    "_winapi.WaitForSingleObject requires host_env",
                ));
            }
            #[cfg(feature = "host_env")]
            let (result, error) = {
                let _blocked = pyre_interpreter::module::thread::before_external_block();
                let result = host_winapi::wait_for_single_object(handle, milliseconds);
                match result {
                    Ok(result) => (result, 0),
                    Err(error) => (
                        host_winapi::WAIT_FAILED,
                        error.raw_os_error().unwrap_or(0) as u32,
                    ),
                }
            };
            if result == host_winapi::WAIT_FAILED {
                return Err(win32_code(error));
            }
            Ok(i64::from(result))
        }
        fn GetExitCodeProcess(handle: pyre_object::PyObjectRef) -> Result<i64, pyre_interpreter::PyError> {
            let handle = handle_w(handle, IntArg::Only("GetExitCodeProcess"))?;
            #[cfg(not(feature = "host_env"))]
            {
                let _ = handle;
                return Err(pyre_interpreter::PyError::not_implemented(
                    "_winapi.GetExitCodeProcess requires host_env",
                ));
            }
            #[cfg(feature = "host_env")]
            host_winapi::get_exit_code_process(handle)
                .map(i64::from)
                .map_err(win32_err)
        }
    },
    extra_init: |ns| {
        let mut ns = ns;
        // The handle sentinel is `(HANDLE)-1` (handleapi.h), which prints as
        // the unsigned value and so does not fit the `int_constants` table.
        pyre_interpreter::__pyre_store!(ns, "INVALID_HANDLE_VALUE", w_handle(windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE));
        // The launch half, registered by hand: the module is built without
        // `host_env` too, and there it stops at the constants and the calls
        // above.
        #[cfg(feature = "host_env")]
        {
            // `py_module!` pins the module dict in `ns_slot`, which this body
            // cannot name. Publish that live word into a slot `host::install`
            // can reload (`ShadowStackFrameworkGCTransformer.pop_roots`).
            let ns_here = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            host::install(pyre_object::gc_roots::shadow_stack_get(ns_here));
            ns = pyre_object::gc_roots::shadow_stack_get(ns_here);
            for (name, arity, function) in [
                ("GetStdHandle", 1, process::get_std_handle as pyre_interpreter::BuiltinCodeFn),
                ("GetCurrentProcess", 0, process::get_current_process),
                ("GetFileType", 1, process::get_file_type),
                ("GetLastError", 0, process::get_last_error),
                ("GetModuleFileName", 1, process::get_module_file_name),
                ("TerminateProcess", 2, process::terminate_process),
                ("CreatePipe", 2, process::create_pipe),
            ] {
                pyre_interpreter::__pyre_store!(ns, name, pyre_interpreter::gateway::with_module(
                        "_winapi",
                        pyre_interpreter::make_module_builtin_function_with_arity(name, function, arity),
                    ));
            }
            // Fixed-arity builtin fast paths only cover zero through four
            // arguments (`gateway.py BuiltinCode0..BuiltinCode4`), and
            // CreateProcess takes nine. It still needs an exact-arity check, so
            // register it through the general call path with a checked body.
            pyre_interpreter::__pyre_store!(ns, "CreateProcess", pyre_interpreter::gateway::with_module(
                    "_winapi",
                    pyre_interpreter::make_module_builtin_function(
                        "CreateProcess",
                        pyre_interpreter::py_checked_arity_fn!(
                            "CreateProcess",
                            9,
                            process::create_process
                        ),
                    ),
                ));
            // `options` is the one argument with a default (`0`), so the
            // count is not fixed.
            pyre_interpreter::__pyre_store!(ns, "DuplicateHandle", pyre_interpreter::gateway::with_module(
                    "_winapi",
                    pyre_interpreter::make_module_builtin_function(
                        "DuplicateHandle",
                        process::duplicate_handle,
                    ),
                ));
        }
    },
}

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // `_winapi.Overlapped` owns the same kind of native record, waited on
    // through its own event rather than a completion port. Sweep cancels and
    // waits out an operation still in flight before the buffer it writes into
    // is freed, then closes that event.
    #[cfg(not(feature = "sandbox"))]
    types.push(ModuleGcType {
        descriptor: <overlapped::W_Overlapped as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: Some(gc_destructor!(overlapped::w_overlapped_dealloc)),
    });
}
