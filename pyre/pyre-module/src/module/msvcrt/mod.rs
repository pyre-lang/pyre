//! msvcrt module — Windows Microsoft Visual C runtime services.
//!
//! `subprocess` selects its Windows implementation from the presence of this
//! module (`try: import msvcrt` → `_mswindows = True`), and `getpass` uses the
//! console read/write calls; the whole surface delegates to
//! `rustpython_host_env::msvcrt`, which wraps the CRT `_getch`/`_setmode`/…
//! family, plus `get_osfhandle` via `nt::handle_from_fd`.

use rustpython_host_env::msvcrt as host_msvcrt;

/// Convert a host `io::Error` into an `OSError` carrying its errno.
fn msvcrt_err(error: std::io::Error) -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::os_error_with_errno(
        pyre_interpreter::builtins::io_error_posix_errno(&error, 0),
        format!("{error}"),
    )
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
            host_msvcrt::getch()
        }
        fn getwch() -> String {
            host_msvcrt::getwch()
        }
        fn getche() -> Vec<u8> {
            host_msvcrt::getche()
        }
        fn getwche() -> String {
            host_msvcrt::getwche()
        }
        fn putch(char_: &[u8]) -> Result<(), pyre_interpreter::PyError> {
            let byte = *char_.first().ok_or_else(|| {
                pyre_interpreter::PyError::type_error("putch() argument must be a byte string of length 1")
            })?;
            host_msvcrt::putch(byte);
            Ok(())
        }
        fn putwch(unicode_char: &str) -> Result<(), pyre_interpreter::PyError> {
            let ch = unicode_char.chars().next().ok_or_else(|| {
                pyre_interpreter::PyError::type_error("putwch() argument must be a unicode character")
            })?;
            host_msvcrt::putwch(ch);
            Ok(())
        }
        fn ungetch(char_: &[u8]) -> Result<(), pyre_interpreter::PyError> {
            let byte = *char_.first().ok_or_else(|| {
                pyre_interpreter::PyError::type_error("ungetch() argument must be a byte string of length 1")
            })?;
            host_msvcrt::ungetch(byte).map_err(msvcrt_err)
        }
        fn ungetwch(unicode_char: &str) -> Result<(), pyre_interpreter::PyError> {
            let ch = unicode_char.chars().next().ok_or_else(|| {
                pyre_interpreter::PyError::type_error("ungetwch() argument must be a unicode character")
            })?;
            host_msvcrt::ungetwch(ch).map_err(msvcrt_err)
        }
        fn kbhit() -> bool {
            host_msvcrt::kbhit() != 0
        }
        fn locking(fd: i32, mode: i32, nbytes: i64) -> Result<(), pyre_interpreter::PyError> {
            host_msvcrt::locking(fd, mode, nbytes).map_err(msvcrt_err)
        }
        fn setmode(fd: i32, flags: i32) -> Result<i32, pyre_interpreter::PyError> {
            let borrowed = unsafe { rustpython_host_env::crt_fd::Borrowed::try_borrow_raw(fd) }
                .map_err(msvcrt_err)?;
            host_msvcrt::setmode(borrowed, flags).map_err(msvcrt_err)
        }
        fn open_osfhandle(handle: i64, flags: i32) -> Result<i32, pyre_interpreter::PyError> {
            host_msvcrt::open_osfhandle(handle as isize, flags).map_err(msvcrt_err)
        }
        fn get_osfhandle(fd: i32) -> Result<i64, pyre_interpreter::PyError> {
            let handle = rustpython_host_env::nt::handle_from_fd(fd);
            if handle as isize == -1 {
                return Err(pyre_interpreter::PyError::os_error_syscall(libc::EBADF, pyre_object::PY_NULL));
            }
            Ok(handle as isize as i64)
        }
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
    extra_init: |ns| {
        // The C runtime assembly version the build's own toolset named; the
        // build script leaves it unset when no toolset header answered.
        if let Some(version) = option_env!("PYRE_CRT_ASSEMBLY_VERSION") {
            pyre_interpreter::__pyre_store!(ns, "CRT_ASSEMBLY_VERSION", pyre_object::w_str_new(version));
        }
    }
}
