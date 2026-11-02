//! msvcrt module — Windows Microsoft Visual C runtime services.
//!
//! `subprocess` selects its Windows implementation from the presence of this
//! module (`try: import msvcrt` → `_mswindows = True`), and `getpass` uses the
//! console read/write calls. The native boundary follows `lib_pypy/msvcrt.py`'s
//! CRT declarations, using `rustpython_host_env` where its argument and result
//! types preserve them, and raw wchar_t calls where Rust char/String cannot.
//! The app-level owner transition still requires `_rawffi.get_libc().name`.

use rustpython_host_env::msvcrt as host_msvcrt;

// `lib_pypy/_pypy_winbase_build.py`'s CRT declarations. The host adapter's
// Rust char/String boundary cannot carry a Windows wchar_t surrogate.
// Keep the same linked CRT (and its console pushback state), using u16
// rather than routing a code unit through Unicode scalar values.
unsafe extern "C" {
    #[link_name = "_getwch"]
    fn crt_getwch() -> u16;
    #[link_name = "_getwche"]
    fn crt_getwche() -> u16;
    #[link_name = "_putwch"]
    fn crt_putwch(ch: u16) -> u16;
    #[link_name = "_ungetwch"]
    fn crt_ungetwch(ch: u16) -> u16;
    #[link_name = "_setmode"]
    fn crt_setmode(fd: i32, mode: i32) -> i32;
}

/// `lib_pypy/msvcrt.py putwch` / `ungetwch`: one Python character, including
/// a lone surrogate. [3.15-spec] The v3.15.0 clinic wrappers of the same
/// names accept one code point and report its actual string length; their
/// CRT calls narrow it to wchar_t, including supplementary code points.
fn wchar_w(obj: pyre_object::PyObjectRef, name: &str) -> Result<u16, pyre_interpreter::PyError> {
    if !unsafe { pyre_object::is_str(obj) } {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{name}() argument must be a unicode character, not {}",
            pyre_interpreter::type_methods::clinic_arg_type_name(obj),
        )));
    }
    let text = pyre_interpreter::baseobjspace::text_wtf8_w(obj)?;
    let length = text.code_points().count();
    if length != 1 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{name}(): argument must be a unicode character, not a string of length {length}",
        )));
    }
    Ok(text.code_points().next().unwrap().to_u32() as u16)
}

/// Windows `long` at `lib_pypy/msvcrt.py locking`'s CFFI boundary.
/// [3.15-spec] CFFI's `misc.as_long` accepts `__int__` and reports the C
/// type's name; `PC/clinic/msvcrtmodule.c.h msvcrt_locking` at v3.15.0
/// uses `PyLong_AsLong`, which accepts only the index protocol. Its overflow
/// text is in `Objects/longobject.c PyLong_AsLong` at the same tag.
fn c_long_w(obj: pyre_object::PyObjectRef) -> Result<i32, pyre_interpreter::PyError> {
    let w_index = pyre_interpreter::baseobjspace::space_index(obj)?;
    let too_large =
        || pyre_interpreter::PyError::overflow_error("Python int too large to convert to C long");
    // Only the conversion of the already-indexed integer is translated here:
    // an OverflowError raised by __index__ itself must keep its own message.
    let value = pyre_interpreter::baseobjspace::int_w(w_index).map_err(|error| {
        if error.kind == pyre_interpreter::PyErrorKind::OverflowError {
            too_large()
        } else {
            error
        }
    })?;
    i32::try_from(value).map_err(|_| too_large())
}

/// Convert a host `io::Error` into an `OSError` carrying its errno.
fn msvcrt_err(error: std::io::Error) -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::os_error_with_errno(
        pyre_interpreter::builtins::io_error_posix_errno(&error, 0),
        format!("{error}"),
    )
}

/// [3.15-spec] `lib_pypy/msvcrt.py` has no audit calls. At v3.15.0,
/// `PC/msvcrtmodule.c msvcrt_locking_impl`, `msvcrt_open_osfhandle_impl`
/// and `msvcrt_get_osfhandle_impl` audit converted arguments before the CRT.
/// Use the interpreter-owned audit dispatcher, as PyPy's winreg wrappers do.
fn audit_crt(event: &str, values: &[i128]) -> Result<(), pyre_interpreter::PyError> {
    if !pyre_interpreter::module::sys::vm::audit_hooks_armed() {
        return Ok(());
    }
    let mut args = pyre_object::gc_roots::RootedItems::new();
    for &value in values {
        args.push(match i64::try_from(value) {
            Ok(value) => pyre_object::w_int_new(value),
            Err(_) => {
                pyre_object::longobject::w_long_new(majit_rlib::rbigint::RBigInt::from(value))
            }
        });
    }
    pyre_interpreter::module::sys::vm::audit(event, &args.take())
}

/// `lib_pypy/msvcrt.py putch` / `ungetch` at the byte-character boundary.
/// [3.15-spec] v3.15.0 `PC/clinic/msvcrtmodule.c.h msvcrt_putch` and
/// `msvcrt_ungetch` accept bytes and bytearray, without the general buffer
/// protocol; PyPy's ord/bytes checks accept different argument shapes.
fn byte_w(obj: pyre_object::PyObjectRef, name: &str) -> Result<u8, pyre_interpreter::PyError> {
    let (bytes, kind) = unsafe {
        if pyre_object::bytesobject::is_bytes(obj) {
            (pyre_object::bytesobject::w_bytes_data(obj), "bytes")
        } else if pyre_object::bytearrayobject::is_bytearray(obj) {
            (
                pyre_object::bytearrayobject::w_bytearray_data(obj),
                "bytearray",
            )
        } else {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "{name}() argument must be a byte string of length 1, not {}",
                pyre_interpreter::type_methods::clinic_arg_type_name(obj),
            )));
        }
    };
    if bytes.len() != 1 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{name}(): argument must be a byte string of length 1, not a {kind} object of length {}",
            bytes.len(),
        )));
    }
    Ok(bytes[0])
}

pyre_interpreter::py_module! {
    "msvcrt",
    int_constants: {
        "LK_UNLCK" => host_msvcrt::LK_UNLCK,
        "LK_LOCK" => host_msvcrt::LK_LOCK,
        "LK_NBLCK" => host_msvcrt::LK_NBLCK,
        "LK_RLCK" => host_msvcrt::LK_RLCK,
        "LK_NBRLCK" => host_msvcrt::LK_NBRLCK,
        "SEM_FAILCRITICALERRORS" => host_msvcrt::SEM_FAILCRITICALERRORS,
        "SEM_NOALIGNMENTFAULTEXCEPT" => host_msvcrt::SEM_NOALIGNMENTFAULTEXCEPT,
        "SEM_NOGPFAULTERRORBOX" => host_msvcrt::SEM_NOGPFAULTERRORBOX,
        "SEM_NOOPENFILEERRORBOX" => host_msvcrt::SEM_NOOPENFILEERRORBOX,
    },
    inline_functions: {
        fn getch() -> Vec<u8> {
            let _blocked = pyre_interpreter::module::thread::before_external_block();
            host_msvcrt::getch()
        }
        fn getwch() -> pyre_object::PyObjectRef {
            let ch = {
                // PyPy's CFFI external call releases the execution context.
                let _blocked = pyre_interpreter::module::thread::before_external_block();
                unsafe { crt_getwch() }
            };
            pyre_object::unicodeobject::w_str_from_codepoint(u32::from(ch))
        }
        fn getche() -> Vec<u8> {
            let _blocked = pyre_interpreter::module::thread::before_external_block();
            host_msvcrt::getche()
        }
        fn getwche() -> pyre_object::PyObjectRef {
            let ch = {
                let _blocked = pyre_interpreter::module::thread::before_external_block();
                unsafe { crt_getwche() }
            };
            pyre_object::unicodeobject::w_str_from_codepoint(u32::from(ch))
        }
        fn putch(char_: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            let byte = byte_w(char_, "putch")?;
            host_msvcrt::putch(byte);
            Ok(())
        }
        fn putwch(unicode_char: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            let ch = wchar_w(unicode_char, "putwch")?;
            unsafe { rustpython_host_env::suppress_iph!(crt_putwch(ch)) };
            Ok(())
        }
        fn ungetch(char_: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            let byte = byte_w(char_, "ungetch")?;
            host_msvcrt::ungetch(byte).map_err(msvcrt_err)
        }
        fn ungetwch(unicode_char: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            let ch = wchar_w(unicode_char, "ungetwch")?;
            let result = unsafe { rustpython_host_env::suppress_iph!(crt_ungetwch(ch)) };
            if result == u16::MAX {
                return Err(pyre_interpreter::PyError::os_error_syscall(
                    majit_rlib::rposix::_get_errno(), pyre_object::PY_NULL,
                ));
            }
            Ok(())
        }
        // `kbhit = _lib._kbhit`: the C int, unchanged.
        fn kbhit() -> i32 {
            host_msvcrt::kbhit()
        }
        fn locking(fd: pyre_object::PyObjectRef, mut mode: pyre_object::PyObjectRef, mut nbytes: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            // `lib_pypy/msvcrt.py locking` calls `_lib._locking`; its CFFI
            // declaration in `_pypy_winbase_build.py` is (int, int, long).
            // Windows long is 32 bits even on a 64-bit host. Check each
            // argument before the CRT call instead of truncating a Rust cast.
            // [3.15-spec] CFFI accepts __int__; v3.15.0's
            // `PC/clinic/msvcrtmodule.c.h msvcrt_locking` uses PyLong_AsInt
            // for fd/mode, hence the index-only conversion and C int message.
            let fd = pyre_object::with_roots!(mode, nbytes => pyre_interpreter::baseobjspace::index_c_int_w(fd))?;
            let mode = pyre_object::with_roots!(nbytes => pyre_interpreter::baseobjspace::index_c_int_w(mode))?;
            let nbytes = c_long_w(nbytes)?;
            audit_crt("msvcrt.locking", &[i128::from(fd), i128::from(mode), i128::from(nbytes)])?;
            let result = {
                let _blocked = pyre_interpreter::module::thread::before_external_block();
                host_msvcrt::locking(fd, mode, i64::from(nbytes))
            };
            result.map_err(msvcrt_err)
        }
        fn setmode(fd: pyre_object::PyObjectRef, mut flags: pyre_object::PyObjectRef) -> Result<i32, pyre_interpreter::PyError> {
            // `lib_pypy/msvcrt.py setmode`: `_setmode(int fd, int mode)`.
            // [3.15-spec] `PC/clinic/msvcrtmodule.c.h msvcrt_setmode` at
            // v3.15.0 uses PyLong_AsInt for both, rather than CFFI's __int__.
            let fd = pyre_object::with_roots!(flags => pyre_interpreter::baseobjspace::index_c_int_w(fd))?;
            let flags = pyre_interpreter::baseobjspace::index_c_int_w(flags)?;
            // `lib_pypy/msvcrt.py setmode` passes both converted arguments
            // directly to the CRT. Prevalidating a Borrowed descriptor here
            // changes the CRT's mode-before-descriptor error order.
            let previous = unsafe { rustpython_host_env::suppress_iph!(crt_setmode(fd, flags)) };
            if previous == -1 {
                return Err(pyre_interpreter::PyError::os_error_syscall(
                    majit_rlib::rposix::_get_errno(), pyre_object::PY_NULL,
                ));
            }
            Ok(previous)
        }
        fn open_osfhandle(mut handle: pyre_object::PyObjectRef, mut flags: pyre_object::PyObjectRef) -> Result<i32, pyre_interpreter::PyError> {
            // `lib_pypy/msvcrt.py open_osfhandle` takes an intptr_t handle
            // and an int flags argument; only the latter is 32 bits.
            // [3.15-spec] CFFI accepts __int__ and a signed intptr_t.
            // `PC/clinic/msvcrtmodule.c.h msvcrt_open_osfhandle` at v3.15.0
            // uses `Objects/longobject.c PyLong_AsVoidPtr`: only real ints,
            // signed negative values, and the full unsigned positive range.
            // Validate this before flags so their error cannot mask this one.
            if !unsafe { pyre_object::pyobject::is_int_or_long(handle) } {
                return Err(pyre_interpreter::PyError::type_error("an integer is required"));
            }
            let too_large = || pyre_interpreter::PyError::overflow_error("int too big to convert");
            let signed = pyre_object::with_roots!(handle, flags => pyre_interpreter::baseobjspace::int_w(handle));
            let handle = match signed {
                Ok(value) if value < 0 => isize::try_from(value).map_err(|_| too_large())?,
                Ok(value) => usize::try_from(value as u64).map_err(|_| too_large())? as isize,
                Err(error) if error.kind == pyre_interpreter::PyErrorKind::OverflowError => {
                    let value = pyre_object::with_roots!(flags => pyre_interpreter::baseobjspace::uint_w(handle))
                        .map_err(|_| too_large())?;
                    usize::try_from(value).map_err(|_| too_large())? as isize
                }
                Err(error) => return Err(error),
            };
            // [3.15-spec] `PC/clinic/msvcrtmodule.c.h msvcrt_open_osfhandle`
            // at v3.15.0 converts flags with PyLong_AsInt, not CFFI's __int__.
            let flags = pyre_interpreter::baseobjspace::index_c_int_w(flags)?;
            audit_crt("msvcrt.open_osfhandle", &[handle as usize as i128, i128::from(flags)])?;
            host_msvcrt::open_osfhandle(handle, flags).map_err(msvcrt_err)
        }
        fn get_osfhandle(fd: pyre_object::PyObjectRef) -> Result<i64, pyre_interpreter::PyError> {
            // [3.15-spec] `lib_pypy/msvcrt.py get_osfhandle` forwards through
            // `interp_magic.get_osfhandle`'s machine-word unwrap to a C int,
            // which truncates. `PC/clinic/msvcrtmodule.c.h msvcrt_get_osfhandle`
            // at v3.15.0 checks PyLong_AsInt before looking up the descriptor.
            let fd = pyre_interpreter::baseobjspace::index_c_int_w(fd)?;
            audit_crt("msvcrt.get_osfhandle", &[i128::from(fd)])?;
            let handle = rustpython_host_env::nt::handle_from_fd(fd);
            if handle as isize == -1 {
                return Err(pyre_interpreter::PyError::os_error_syscall(libc::EBADF, pyre_object::PY_NULL));
            }
            Ok(handle as isize as i64)
        }
        // [3.14-spec] `lib_pypy/msvcrt.py` binds no `heapmin`;
        // `lib-python/3/test/test_msvcrt.py` `TestOther.test_heap_min` calls
        // `msvcrt.heapmin()` and tolerates only `OSError`, so the attribute
        // must exist (`msvcrtmodule.c` `msvcrt_heapmin_impl`).
        fn heapmin() -> Result<(), pyre_interpreter::PyError> {
            host_msvcrt::heapmin().map_err(msvcrt_err)
        }
        fn SetErrorMode(mode: u32) -> u32 {
            host_msvcrt::set_error_mode(mode)
        }
        // The mode `SetErrorMode` replaces, which is the only way to read it
        // back: the setter reports the previous value, so a caller that wants
        // to leave it alone has nothing to restore without this.
        fn GetErrorMode() -> u32 {
            host_msvcrt::get_error_mode()
        }
    },
}
