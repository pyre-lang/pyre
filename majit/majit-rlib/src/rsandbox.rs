//! Runtime half of `rpython/translator/sandbox/rsandbox.py` plus the error
//! taxonomy shared with `sandlib.py` `EXCEPTION_TABLE`.
//!
//! [`make_sandbox_trampoline`](run_trampoline) dumps `fnname` and the
//! marshalled argument tuple to fd 1, reads the reply from fd 0, and either
//! returns the loaded result or lets a [`SandboxError`] escape the wrapper
//! (`std::panic::panic_any`), matching `rsandbox.py` `reraise_error`.

use std::os::raw::c_void;
use std::sync::Mutex;

use crate::rmarshal::{
    IntFlavor, Loader, MarshalValue, NeedMore, StatResult, dump_string, dump_tuple, load_bool,
    load_float, load_int, load_int_nonneg, load_longlong, load_statresult, load_string, load_value,
};

/// An error raised across the sandbox boundary.
///
/// The numeric `Os` payload is the errno carried as a second marshalled int
/// after the exception code (`rsandbox.py` `reraise_error`). `Protocol` is
/// local to this port: malformed marshal data / EOF on the pipe, which
/// RPython surfaces as an `IOError`/`ValueError` from the loader.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SandboxError {
    /// `OSError(errno, ...)` — code 1, followed by the errno int.
    Os(i32),
    /// `IOError` — code 2.
    Io,
    /// `OverflowError` — code 3.
    Overflow,
    /// `ValueError` — code 4.
    Value,
    /// `ZeroDivisionError` — code 5.
    ZeroDivision,
    /// `MemoryError` — code 6.
    Memory,
    /// `KeyError` — code 7.
    Key,
    /// `IndexError` — code 8.
    Index,
    /// `RuntimeError` — code 9 (and the catch-all).
    Runtime,
    /// Malformed wire data / unexpected EOF (loader-level failure).
    Protocol(String),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxError::Os(e) => write!(f, "OSError({e})"),
            SandboxError::Io => f.write_str("IOError"),
            SandboxError::Overflow => f.write_str("OverflowError"),
            SandboxError::Value => f.write_str("ValueError"),
            SandboxError::ZeroDivision => f.write_str("ZeroDivisionError"),
            SandboxError::Memory => f.write_str("MemoryError"),
            SandboxError::Key => f.write_str("KeyError"),
            SandboxError::Index => f.write_str("IndexError"),
            SandboxError::Runtime => f.write_str("RuntimeError"),
            SandboxError::Protocol(msg) => write!(f, "protocol error: {msg}"),
        }
    }
}

impl std::error::Error for SandboxError {}

impl SandboxError {
    /// RPython exception class name `rsandbox.py reraise_error` raises for
    /// this variant. `Protocol` is a loader-level `ValueError`
    /// (`rmarshal.Loader.need_more_data` / `check_finished` / the `load_*`
    /// `ValueError`s); `FdLoader.need_more_data` is `IOError` and maps to
    /// [`SandboxError::Io`].
    pub fn rpython_exception_class_name(&self) -> &'static str {
        match self {
            SandboxError::Os(_) => "OSError",
            SandboxError::Io => "IOError",
            SandboxError::Overflow => "OverflowError",
            SandboxError::Value => "ValueError",
            SandboxError::ZeroDivision => "ZeroDivisionError",
            SandboxError::Memory => "MemoryError",
            SandboxError::Key => "KeyError",
            SandboxError::Index => "IndexError",
            SandboxError::Runtime => "RuntimeError",
            SandboxError::Protocol(_) => "ValueError",
        }
    }
}

pub type SandboxResult<T> = Result<T, SandboxError>;

/// Process-global last-raised sandbox traceback (`debug_traceback.h`
/// `pypy_debug_tracebacks` / `PYPY_DEBUG_RECORD_TRACEBACK`).
static RECORDED_TRACEBACK: Mutex<Option<std::backtrace::Backtrace>> = Mutex::new(None);

fn record_debug_traceback() {
    if let Ok(mut slot) = RECORDED_TRACEBACK.lock() {
        *slot = Some(std::backtrace::Backtrace::force_capture());
    }
}

/// `debug_traceback.h` `pypy_debug_traceback_print` / `debug_print_traceback`:
/// take the traceback recorded at the raise so a later error never reprints it.
pub fn take_recorded_traceback() -> Option<std::backtrace::Backtrace> {
    RECORDED_TRACEBACK
        .lock()
        .ok()
        .and_then(|mut slot| slot.take())
}

/// Catch an `OSError` escaping a non-`Result` trampoline, matching an interp
/// caller of `except OSError as e: raise wrap_oserror(...)`.
///
/// `SandboxError::Os(errno)` becomes `Err(errno)`; any other payload is
/// resumed so it reaches `error.py get_converted_unexpected_exception`.
#[cfg(feature = "sandbox")]
pub fn catch_os_error<T>(f: impl FnOnce() -> T) -> Result<T, i32> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => Ok(value),
        Err(payload) => match payload.downcast::<SandboxError>() {
            Ok(err) => match *err {
                SandboxError::Os(errno) => Err(errno),
                other => std::panic::resume_unwind(Box::new(other)),
            },
            Err(payload) => std::panic::resume_unwind(payload),
        },
    }
}

#[cfg(not(feature = "sandbox"))]
#[inline(always)]
pub fn catch_os_error<T>(f: impl FnOnce() -> T) -> Result<T, i32> {
    Ok(f())
}

#[cfg(feature = "sandbox")]
impl From<crate::rffi::MemoryError> for SandboxError {
    fn from(_: crate::rffi::MemoryError) -> Self {
        SandboxError::Memory
    }
}

/// Re-raise an RPython exception that a typed result carried out of a trampoline,
/// for a caller whose upstream does not catch it: it unwinds to the
/// `BuiltinCode.handle_exception` boundary (`catch_sandbox_stub`).
///
/// `debug_traceback.h` `PYPY_DEBUG_RECORD_TRACEBACK` then `panic_any`, matching
/// `rsandbox.py` `reraise_error`.
pub fn reraise(err: impl Into<SandboxError>) -> ! {
    record_debug_traceback();
    std::panic::panic_any(err.into())
}

/// Map an exception code (the first int of a reply) to a [`SandboxError`].
///
/// For code 1 (`OSError`) the caller must subsequently read the errno int and
/// patch it into the returned `Os(0)`. Mirrors `rsandbox.reraise_error`.
pub fn error_from_code(code: i64) -> SandboxError {
    match code {
        1 => SandboxError::Os(0),
        2 => SandboxError::Io,
        3 => SandboxError::Overflow,
        4 => SandboxError::Value,
        5 => SandboxError::ZeroDivision,
        6 => SandboxError::Memory,
        7 => SandboxError::Key,
        8 => SandboxError::Index,
        _ => SandboxError::Runtime,
    }
}

/// Map a [`SandboxError`] to its wire code. Mirrors `sandlib.EXCEPTION_TABLE`.
pub fn code_for_error(err: &SandboxError) -> i64 {
    match err {
        SandboxError::Os(_) => 1,
        SandboxError::Io => 2,
        SandboxError::Overflow => 3,
        SandboxError::Value => 4,
        SandboxError::ZeroDivision => 5,
        SandboxError::Memory => 6,
        SandboxError::Key => 7,
        SandboxError::Index => 8,
        SandboxError::Runtime | SandboxError::Protocol(_) => 9,
    }
}

/// The static shape of a reply value, so the client knows which loader to run
/// after the success code. Each variant corresponds to a `load_result`
/// specialization in `rsandbox.make_sandbox_trampoline`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResultKind {
    /// `None` reply (e.g. `ll_os_close`).
    None,
    /// A plain int (`'i'`/`'I'`).
    Int,
    /// A forced 64-bit int (`RESULTTYPE_LONGLONG`, e.g. `ll_os_lseek`).
    LongLong,
    /// A bool (`'T'`/`'F'`).
    Bool,
    /// A byte string (`ll_os_read`, `ll_os_getcwd`, ...).
    Str,
    /// `str | None` (`ll_os_getenv`).
    OptStr,
    /// A list of byte strings (`ll_os_listdir`).
    ListStr,
    /// A list of `(str, str)` pairs (`ll_os_envitems`).
    EnvItems,
    /// A hand-packed stat result (`RESULTTYPE_STATRESULT`).
    StatResult,
    /// A float (`ll_time_time` / `ll_time_clock`).
    Float,
}

/// A decoded reply payload — the typed Rust value a trampoline returns, keyed
/// by the [`ResultKind`] requested. (The wire `MarshalValue` cannot represent
/// a stat result, so decoding lands in this richer enum rather than
/// `MarshalValue`.)
#[derive(Clone, Debug, PartialEq)]
pub enum SyscallResult {
    None,
    Int(i64),
    Bool(bool),
    Str(Vec<u8>),
    OptStr(Option<Vec<u8>>),
    ListStr(Vec<Vec<u8>>),
    EnvItems(Vec<(Vec<u8>, Vec<u8>)>),
    Stat(StatResult),
    Float(f64),
}

/// Result side of `rmarshal.get_loader(s_result)`. Wire error code 1
/// (`OSError`) becomes `Err(errno)` on [`Result<T, i32>`]; any other error
/// (or an error on a non-`Result` return) is `std::panic::panic_any` of
/// [`SandboxError`], the RPython exception escaping the wrapper.
pub trait SandboxLoad: Sized {
    fn load<N: NeedMore>(loader: &mut Loader<N>) -> SandboxResult<Self>;

    fn from_error(err: SandboxError) -> Self {
        reraise(err)
    }
}

/// `rmarshal.find_loader` over an integer result: a signed result
/// (`SomeInteger()` contains it) takes `load_int`; an unsigned one unions
/// only with `SomeInteger(nonneg=True)` and takes `load_int_nonneg`. The
/// value is then cast to the result type.
macro_rules! sandbox_load_int {
    ($load:ident: $($t:ty),+ $(,)?) => {$(
        impl SandboxLoad for $t {
            fn load<N: NeedMore>(loader: &mut Loader<N>) -> SandboxResult<Self> {
                Ok($load(loader)? as $t)
            }
        }
    )+};
}

sandbox_load_int!(load_int: i8, i16, i32, i64, isize);
sandbox_load_int!(load_int_nonneg: u8, u16, u32, u64, usize);

impl SandboxLoad for bool {
    fn load<N: NeedMore>(loader: &mut Loader<N>) -> SandboxResult<Self> {
        load_bool(loader)
    }
}

impl SandboxLoad for f64 {
    fn load<N: NeedMore>(loader: &mut Loader<N>) -> SandboxResult<Self> {
        load_float(loader)
    }
}

impl SandboxLoad for () {
    fn load<N: NeedMore>(loader: &mut Loader<N>) -> SandboxResult<Self> {
        match load_value(loader)? {
            MarshalValue::None => Ok(()),
            _ => Err(SandboxError::Protocol("expected None reply".into())),
        }
    }
}

impl SandboxLoad for Vec<u8> {
    fn load<N: NeedMore>(loader: &mut Loader<N>) -> SandboxResult<Self> {
        load_string(loader)
    }
}

impl SandboxLoad for Option<Vec<u8>> {
    fn load<N: NeedMore>(loader: &mut Loader<N>) -> SandboxResult<Self> {
        match load_value(loader)? {
            MarshalValue::None => Ok(None),
            MarshalValue::Str(s) => Ok(Some(s)),
            _ => Err(SandboxError::Protocol("expected str|None reply".into())),
        }
    }
}

impl SandboxLoad for Vec<Vec<u8>> {
    fn load<N: NeedMore>(loader: &mut Loader<N>) -> SandboxResult<Self> {
        match load_value(loader)? {
            MarshalValue::List(items) => items.into_iter().map(as_bytes).collect(),
            _ => Err(SandboxError::Protocol("expected list[str] reply".into())),
        }
    }
}

impl SandboxLoad for Vec<(Vec<u8>, Vec<u8>)> {
    fn load<N: NeedMore>(loader: &mut Loader<N>) -> SandboxResult<Self> {
        match load_value(loader)? {
            MarshalValue::List(items) => {
                let mut pairs = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        MarshalValue::Tuple(mut kv) if kv.len() == 2 => {
                            let value = as_bytes(kv.pop().unwrap())?;
                            let key = as_bytes(kv.pop().unwrap())?;
                            pairs.push((key, value));
                        }
                        _ => {
                            return Err(SandboxError::Protocol(
                                "expected (str,str) env item".into(),
                            ));
                        }
                    }
                }
                Ok(pairs)
            }
            _ => Err(SandboxError::Protocol(
                "expected list[(str,str)] reply".into(),
            )),
        }
    }
}

impl<T: SandboxLoad> SandboxLoad for Result<T, i32> {
    fn load<N: NeedMore>(loader: &mut Loader<N>) -> SandboxResult<Self> {
        Ok(Ok(T::load(loader)?))
    }

    fn from_error(err: SandboxError) -> Self {
        match err {
            SandboxError::Os(errno) => Err(errno),
            other => reraise(other),
        }
    }
}

impl<T: SandboxLoad> SandboxLoad for Result<T, crate::rffi::MemoryError> {
    fn load<N: NeedMore>(loader: &mut Loader<N>) -> SandboxResult<Self> {
        Ok(Ok(T::load(loader)?))
    }

    fn from_error(err: SandboxError) -> Self {
        match err {
            SandboxError::Memory => Err(crate::rffi::MemoryError),
            other => Ok(T::from_error(other)),
        }
    }
}

/// `rposix.strerror` returns `Err(())` for the host null that
/// `interp_posix.strerror` turns into `ValueError`. Other trampoline
/// errors keep the panic path, matching [`Result<T, i32>`] for non-Os.
impl<T: SandboxLoad> SandboxLoad for Result<T, ()> {
    fn load<N: NeedMore>(loader: &mut Loader<N>) -> SandboxResult<Self> {
        Ok(Ok(T::load(loader)?))
    }

    fn from_error(err: SandboxError) -> Self {
        match err {
            SandboxError::Value => Err(()),
            other => reraise(other),
        }
    }
}

/// Wire `RESULTTYPE_STATRESULT` into `rposix_stat.STAT_FIELDS`. Extra
/// unix fields (blksize/blocks/rdev/flags/nsec) are not on the wire.
impl SandboxLoad for crate::rposix_stat::StatResult {
    fn load<N: NeedMore>(loader: &mut Loader<N>) -> SandboxResult<Self> {
        let st = load_statresult(loader)?;
        Ok(crate::rposix_stat::StatResult {
            st_mode: st.st_mode as i64,
            st_ino: st.st_ino as i64,
            st_dev: st.st_dev as i64,
            st_nlink: st.st_nlink as i64,
            st_uid: st.st_uid as i64,
            st_gid: st.st_gid as i64,
            st_size: st.st_size as i64,
            st_atime: st.st_atime,
            st_mtime: st.st_mtime,
            st_ctime: st.st_ctime,
            st_blksize: 0,
            st_blocks: 0,
            st_rdev: 0,
            #[cfg(any(
                target_os = "macos",
                target_os = "ios",
                target_os = "freebsd",
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "dragonfly"
            ))]
            st_flags: 0,
            nsec_atime: 0,
            nsec_mtime: 0,
            nsec_ctime: 0,
        })
    }
}

/// `rsandbox.py` `make_sandbox_trampoline` / `execute`.
pub fn run_trampoline<R: SandboxLoad>(fnname: &str, args: &[MarshalValue]) -> R {
    let buf = encode_request(fnname, args);
    match sandboxed_io(&buf) {
        Ok(mut loader) => {
            let value = match R::load(&mut loader) {
                Ok(v) => v,
                Err(err) => return R::from_error(err),
            };
            if let Err(err) = loader.check_finished() {
                return R::from_error(err);
            }
            value
        }
        Err(err) => R::from_error(err),
    }
}

// rsandbox.py — the protocol talks over the raw process fds.
const STDIN_FD: i32 = 0;
const STDOUT_FD: i32 = 1;

/// A [`NeedMore`] that refills from a raw fd via the un-sandboxed `read(2)` —
/// the reply pipe (STDIN). `buflen` starts at 4096 and doubles on every
/// refill, matching `rsandbox.FdLoader.need_more_data`. A `FdLoader` is
/// therefore `Loader<FdNeedMore>`.
pub struct FdNeedMore {
    fd: i32,
    buflen: usize,
}

impl FdNeedMore {
    pub fn new(fd: i32) -> Self {
        FdNeedMore { fd, buflen: 4096 }
    }
}

impl NeedMore for FdNeedMore {
    // rsandbox.py `need_more_data`.
    fn need_more(&mut self) -> SandboxResult<Vec<u8>> {
        let mut buf = vec![0u8; self.buflen];
        // SAFETY: read(2) into a buffer we own and sized; raw un-sandboxed read.
        // `rsandbox.FdLoader.need_more_data` calls `ll_read_not_sandboxed` once.
        let count = unsafe { libc::read(self.fd, buf.as_mut_ptr() as *mut c_void, self.buflen) };
        if count <= 0 {
            return Err(SandboxError::Io);
        }
        buf.truncate(count as usize);
        self.buflen *= 2;
        Ok(buf)
    }
}

/// `FdLoader(fd)` — a marshal loader fed by the raw fd.
pub fn fd_loader(fd: i32) -> Loader<FdNeedMore> {
    Loader::new(Vec::new(), FdNeedMore::new(fd))
}

/// rsandbox.py `writeall_not_sandboxed` — write the whole buffer with the raw
/// un-sandboxed `write(2)`, looping over partial writes, IOError on `count <= 0`.
pub fn writeall_not_sandboxed(fd: i32, mut buf: &[u8]) -> SandboxResult<()> {
    while !buf.is_empty() {
        // SAFETY: write(2) from a slice we hold for the duration of the call.
        // `rsandbox.writeall_not_sandboxed` calls `ll_write_not_sandboxed` once
        // per partial write; `count <= 0` is `IOError`.
        let count = unsafe { libc::write(fd, buf.as_ptr() as *const c_void, buf.len()) };
        if count <= 0 {
            return Err(SandboxError::Io);
        }
        buf = &buf[count as usize..];
    }
    Ok(())
}

/// rsandbox.py `reraise_error` — map the leading exception code to a
/// [`SandboxError`]. Code 1 (`OSError`) reads a second int as the errno;
/// 2..=8 map via [`error_from_code`]; anything else is `RuntimeError`.
pub fn reraise_error<N: NeedMore>(error: i64, loader: &mut Loader<N>) -> SandboxError {
    if error == 1 {
        match load_int(loader) {
            Ok(errno) => SandboxError::Os(errno as i32),
            Err(e) => e,
        }
    } else {
        error_from_code(error)
    }
}

/// `rsandbox.py not_implemented_stub` — write `msg + '\n'` to fd 2 with
/// `writeall_not_sandboxed`; a write failure is that `IOError`, otherwise
/// `RuntimeError` (the RPython `msg` is ignored).
pub fn not_implemented_stub(msg: &str) -> SandboxError {
    let mut line = String::with_capacity(msg.len() + 1);
    line.push_str(msg);
    line.push('\n');
    match writeall_not_sandboxed(2, line.as_bytes()) {
        Ok(()) => SandboxError::Runtime,
        Err(err) => err,
    }
}

/// `rsandbox.py` `execute` — marshal `fnname` (TYPE_STRING) followed by the
/// argument TUPLE, with the client int convention (`IntFlavor::Rmarshal` =
/// every int as TYPE_INT64). Frozen by `rmarshal::tests::golden_request_open`.
pub fn encode_request(fnname: &str, args: &[MarshalValue]) -> Vec<u8> {
    let mut buf = Vec::new();
    dump_string(&mut buf, fnname.as_bytes());
    dump_tuple(&mut buf, args, IntFlavor::Rmarshal);
    buf
}

fn as_bytes(v: MarshalValue) -> SandboxResult<Vec<u8>> {
    match v {
        MarshalValue::Str(s) => Ok(s),
        _ => Err(SandboxError::Protocol("expected string element".into())),
    }
}

/// rsandbox.py `load_result = rmarshal.get_loader(s_result)` — decode the
/// typed reply payload by dispatching on [`ResultKind`]. The reply shapes are
/// exactly what `sandlib::encode_reply` emits.
pub fn load_result<N: NeedMore>(
    loader: &mut Loader<N>,
    kind: ResultKind,
) -> SandboxResult<SyscallResult> {
    match kind {
        ResultKind::None => match load_value(loader)? {
            MarshalValue::None => Ok(SyscallResult::None),
            _ => Err(SandboxError::Protocol("expected None reply".into())),
        },
        ResultKind::Int => Ok(SyscallResult::Int(load_int(loader)?)),
        ResultKind::LongLong => Ok(SyscallResult::Int(load_longlong(loader)?)),
        ResultKind::Bool => Ok(SyscallResult::Bool(load_bool(loader)?)),
        ResultKind::Str => Ok(SyscallResult::Str(load_string(loader)?)),
        ResultKind::OptStr => match load_value(loader)? {
            MarshalValue::None => Ok(SyscallResult::OptStr(None)),
            MarshalValue::Str(s) => Ok(SyscallResult::OptStr(Some(s))),
            _ => Err(SandboxError::Protocol("expected str|None reply".into())),
        },
        ResultKind::ListStr => match load_value(loader)? {
            MarshalValue::List(items) => Ok(SyscallResult::ListStr(
                items
                    .into_iter()
                    .map(as_bytes)
                    .collect::<SandboxResult<_>>()?,
            )),
            _ => Err(SandboxError::Protocol("expected list[str] reply".into())),
        },
        ResultKind::EnvItems => Ok(SyscallResult::EnvItems(Vec::<(Vec<u8>, Vec<u8>)>::load(
            loader,
        )?)),
        ResultKind::StatResult => Ok(SyscallResult::Stat(load_statresult(loader)?)),
        ResultKind::Float => Ok(SyscallResult::Float(load_float(loader)?)),
    }
}

/// rsandbox.py `sandboxed_io` — write the request, build a loader on the
/// reply pipe, read the leading code, and either raise or return the loader
/// positioned at the result payload.
pub fn sandboxed_io(buf: &[u8]) -> SandboxResult<Loader<FdNeedMore>> {
    writeall_not_sandboxed(STDOUT_FD, buf)?;
    let mut loader = fd_loader(STDIN_FD);
    let error = load_int(&mut loader)?;
    if error != 0 {
        return Err(reraise_error(error, &mut loader));
    }
    Ok(loader)
}

/// The generic trampoline body (`rsandbox.py` `execute`). Marshals the
/// request, performs the round-trip, decodes the typed result, and asserts the
/// reply was fully consumed.
pub fn syscall(
    fnname: &str,
    args: &[MarshalValue],
    kind: ResultKind,
) -> SandboxResult<SyscallResult> {
    let buf = encode_request(fnname, args);
    let mut loader = sandboxed_io(&buf)?;
    let result = load_result(&mut loader, kind)?;
    loader.check_finished()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rmarshal::{IntFlavor, dump_string, dump_tuple};

    #[test]
    fn getuid_trampoline_request_bytes_match_encode_request() {
        let written = crate::rposix::getuid_sandbox_request_bytes();
        assert_eq!(written, encode_request("ll_os.ll_os_getuid", &[]));
        let mut expected = Vec::new();
        dump_string(&mut expected, b"ll_os.ll_os_getuid");
        dump_tuple(&mut expected, &[], IntFlavor::Rmarshal);
        assert_eq!(written, expected);
        assert_eq!(
            crate::rposix::geteuid_sandbox_request_bytes(),
            encode_request("ll_os.ll_os_geteuid", &[])
        );
        assert_eq!(
            crate::rposix::getgid_sandbox_request_bytes(),
            encode_request("ll_os.ll_os_getgid", &[])
        );
        assert_eq!(
            crate::rposix::getegid_sandbox_request_bytes(),
            encode_request("ll_os.ll_os_getegid", &[])
        );
        assert_eq!(
            crate::rposix::isatty_sandbox_request_bytes(0),
            encode_request("ll_os.ll_os_isatty", &[MarshalValue::Int(0)])
        );
        assert_eq!(
            crate::rposix::close_sandbox_request_bytes(3),
            encode_request("ll_os.ll_os_close", &[MarshalValue::Int(3)])
        );
        assert_eq!(
            crate::rposix::lseek_sandbox_request_bytes(3, 7, 0),
            encode_request(
                "ll_os.ll_os_lseek",
                &[
                    MarshalValue::Int(3),
                    MarshalValue::Int(7),
                    MarshalValue::Int(0)
                ]
            )
        );
        assert_eq!(
            crate::rposix::open_sandbox_request_bytes(b"/tmp/x", 0, 0o666),
            encode_request(
                "ll_os.ll_os_open",
                &[
                    MarshalValue::Str(b"/tmp/x".to_vec()),
                    MarshalValue::Int(0),
                    MarshalValue::Int(0o666)
                ]
            )
        );
        assert_eq!(
            crate::rposix::unlink_sandbox_request_bytes(b"/tmp/x"),
            encode_request(
                "ll_os.ll_os_unlink",
                &[MarshalValue::Str(b"/tmp/x".to_vec())]
            )
        );
        assert_eq!(
            crate::rposix::mkdir_sandbox_request_bytes(b"/tmp/x", 0o777),
            encode_request(
                "ll_os.ll_os_mkdir",
                &[
                    MarshalValue::Str(b"/tmp/x".to_vec()),
                    MarshalValue::Int(0o777)
                ]
            )
        );
        assert_eq!(
            crate::rposix::access_sandbox_request_bytes(b"/tmp/x", 0),
            encode_request(
                "ll_os.ll_os_access",
                &[MarshalValue::Str(b"/tmp/x".to_vec()), MarshalValue::Int(0)]
            )
        );
        assert_eq!(
            crate::rposix_stat::stat_sandbox_request_bytes(b"/tmp/x"),
            encode_request("ll_os.ll_os_stat", &[MarshalValue::Str(b"/tmp/x".to_vec())])
        );
        assert_eq!(
            crate::rposix_stat::lstat_sandbox_request_bytes(b"/tmp/x"),
            encode_request(
                "ll_os.ll_os_lstat",
                &[MarshalValue::Str(b"/tmp/x".to_vec())]
            )
        );
        assert_eq!(
            crate::rposix_stat::fstat_sandbox_request_bytes(3),
            encode_request("ll_os.ll_os_fstat", &[MarshalValue::Int(3)])
        );
        assert_eq!(
            crate::rposix::read_sandbox_request_bytes(3, 16),
            encode_request(
                "ll_os.ll_os_read",
                &[MarshalValue::Int(3), MarshalValue::Int(16)]
            )
        );
        let hi = crate::lltypesystem::rstr::alloc_immortal_str(b"hi");
        assert_eq!(
            crate::rposix::write_sandbox_request_bytes(1, hi),
            encode_request(
                "ll_os.ll_os_write",
                &[MarshalValue::Int(1), MarshalValue::Str(b"hi".to_vec())]
            )
        );
        unsafe { crate::lltypesystem::rstr::free_immortal_str(hi, 2) };
        assert_eq!(
            crate::rposix::getcwd_sandbox_request_bytes(),
            encode_request("ll_os.ll_os_getcwd", &[])
        );
        assert_eq!(
            crate::rposix::listdir_sandbox_request_bytes(b"/tmp"),
            encode_request(
                "ll_os.ll_os_listdir",
                &[MarshalValue::Str(b"/tmp".to_vec())]
            )
        );
        assert_eq!(
            crate::rposix::strerror_sandbox_request_bytes(2),
            encode_request("ll_os.ll_os_strerror", &[MarshalValue::Int(2)])
        );
        assert_eq!(
            crate::rposix_environ::getenv_llimpl_sandbox_request_bytes(b"PATH"),
            encode_request("ll_os.ll_os_getenv", &[MarshalValue::Str(b"PATH".to_vec())])
        );
        assert_eq!(
            crate::rposix_environ::envitems_llimpl_sandbox_request_bytes(),
            encode_request("ll_os.ll_os_envitems", &[])
        );
        assert_eq!(
            crate::rtime::time_sandbox_request_bytes(),
            encode_request("ll_time.ll_time_time", &[])
        );
        assert_eq!(
            crate::rtime::clock_sandbox_request_bytes(),
            encode_request("ll_time.ll_time_clock", &[])
        );
        assert_eq!(
            crate::rtime::sleep_sandbox_request_bytes(0.25),
            encode_request("ll_time.ll_time_sleep", &[MarshalValue::Float(0.25)])
        );
        assert_eq!(
            crate::rurandom::urandom_os_sandbox_request_bytes(16),
            encode_request("ll_os.ll_os_urandom", &[MarshalValue::Int(16)])
        );
    }

    #[test]
    fn integer_sandbox_loads_follow_find_loader() {
        let mut buf = Vec::new();
        crate::rmarshal::dump_longlong_result(&mut buf, 0x12380000007);
        let mut loader = crate::rmarshal::Loader::from_bytes(buf);
        let v = i64::load(&mut loader).unwrap();
        assert_eq!(v, 0x12380000007);
        loader.check_finished().unwrap();
        // An unsigned result takes `load_int_nonneg`.
        let mut buf = Vec::new();
        crate::rmarshal::dump_int(&mut buf, -1, IntFlavor::Marshal);
        assert!(u64::load(&mut crate::rmarshal::Loader::from_bytes(buf.clone())).is_err());
        assert_eq!(
            i64::load(&mut crate::rmarshal::Loader::from_bytes(buf)).unwrap(),
            -1
        );
    }

    #[test]
    fn catch_os_error_returns_the_closure_value() {
        assert_eq!(catch_os_error(|| 7), Ok(7));
    }

    #[cfg(feature = "sandbox")]
    #[test]
    fn catch_os_error_maps_os_and_resumes_any_other_payload() {
        assert_eq!(
            catch_os_error(|| -> i32 { reraise(SandboxError::Os(libc::ENOENT)) }),
            Err(libc::ENOENT)
        );
        let resumed = std::panic::catch_unwind(|| {
            let _ = catch_os_error(|| -> i32 { reraise(SandboxError::Value) });
        });
        let payload = resumed.expect_err("non-Os SandboxError resumes");
        assert_eq!(
            payload.downcast_ref::<SandboxError>(),
            Some(&SandboxError::Value)
        );
    }
}
