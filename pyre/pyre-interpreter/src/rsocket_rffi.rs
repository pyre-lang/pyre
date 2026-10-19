//! Host socket layer — PyPy: `rpython/rlib/_rsocket_rffi.py`.
//!
//! `_rsocket_rffi.py` is the single place where rsocket's POSIX and `_MSVC`
//! spellings of the same call meet; everything above it names one API.  This
//! module plays that role for `interp_socket`: one set of names, a libc body
//! on unix and a WinSock body on Windows, so the module's own code stays
//! platform-neutral.  Anything with no counterpart on the other side
//! (`recvmsg`, `AF_UNIX`, the POSIX-only options) stays gated at the call
//! site instead of being faked here.

#[cfg(unix)]
pub use libc::{addrinfo, sockaddr, sockaddr_in, sockaddr_in6, sockaddr_storage};

/// `rsocket.py`'s `_c`: the WinSock half of `_rsocket_rffi`.
#[cfg(windows)]
use majit_rlib::_rsocket_rffi as _c;

#[cfg(windows)]
pub use majit_rlib::_rsocket_rffi::{
    addrinfo, sockaddr, sockaddr_in, sockaddr_in6, sockaddr_storage,
};

/// The descriptor a socket call takes.  POSIX numbers sockets alongside every
/// other file descriptor; WinSock hands out an unsigned kernel handle, so the
/// two differ in both width and signedness and only this alias may be assumed.
#[cfg(unix)]
pub type Socket = libc::c_int;
#[cfg(windows)]
pub type Socket = _c::socketfd_type;

/// `getsockname`'s length argument.  `socklen_t` on POSIX, a plain `int` in
/// WinSock's prototypes.
#[cfg(unix)]
pub type SockLen = libc::socklen_t;
#[cfg(windows)]
pub type SockLen = _c::socklen_t;

/// The width `sockaddr.sa_family` is written through.
#[cfg(unix)]
pub type SaFamily = libc::sa_family_t;
#[cfg(windows)]
pub type SaFamily = u16;

/// Whether a descriptor names no socket.  POSIX reserves every negative value.
/// Windows reserves only `INVALID_SOCKET` — `(SOCKET)~0` — and every handle it
/// hands out is unsigned and may exceed `i32::MAX`, so a socket there must
/// never be range-tested with `< 0`.
#[cfg(unix)]
pub fn is_invalid(s: Socket) -> bool {
    s < 0
}
#[cfg(windows)]
pub fn is_invalid(s: Socket) -> bool {
    _c::invalid_socket(s)
}

/// `fileno()` reports the descriptor as a Python int, and `_fd` stores it as
/// one.  `PyLong_FromSocket_t` widens the Windows handle through a signed
/// 64-bit integer, so `INVALID_SOCKET` reads back as `-1` there too and the
/// closed-socket sentinel is the same value on both platforms.
pub fn socket_to_i64(s: Socket) -> i64 {
    // Through the pointer-width signed type, so a 32-bit `SOCKET` reaches the
    // widening as `-1` rather than as `4294967295`.
    s as isize as i64
}

pub fn socket_from_i64(v: i64) -> Socket {
    v as Socket
}

// ── constants the shared body compares against ──

#[cfg(unix)]
pub use libc::{
    AF_INET, AF_INET6, AF_UNSPEC, SO_ERROR, SO_TYPE, SOCK_DGRAM, SOCK_STREAM, SOL_SOCKET,
};
#[cfg(unix)]
pub const NI_MAXHOST: usize = libc::NI_MAXHOST as usize;
#[cfg(unix)]
pub const INADDR_ANY: u32 = libc::INADDR_ANY;
#[cfg(unix)]
pub const AI_NUMERICHOST: libc::c_int = libc::AI_NUMERICHOST;
#[cfg(unix)]
pub const AI_PASSIVE: libc::c_int = libc::AI_PASSIVE;
#[cfg(unix)]
pub const INADDR_BROADCAST: u32 = libc::INADDR_BROADCAST;

#[cfg(windows)]
pub use majit_rlib::_rsocket_rffi::{
    AF_INET, AF_INET6, AF_UNSPEC, INADDR_ANY, INADDR_BROADCAST, SO_ERROR, SO_TYPE, SOCK_DGRAM,
    SOCK_STREAM, SOL_SOCKET,
};
/// The code a call about a descriptor that is not a socket comes back with.
#[cfg(windows)]
pub const WSAENOTSOCK: i32 = _c::WSAENOTSOCK;
/// The code a send that would have blocked comes back with — a buffer with no
/// room left, which is the one failure a caller may choose to drop.
#[cfg(windows)]
pub const WSAEWOULDBLOCK: i32 = _c::WSAEWOULDBLOCK;
#[cfg(windows)]
pub const NI_MAXHOST: usize = _c::NI_MAXHOST as usize;
#[cfg(windows)]
pub const AI_NUMERICHOST: libc::c_int = _c::AI_NUMERICHOST;
#[cfg(windows)]
pub const AI_PASSIVE: libc::c_int = _c::AI_PASSIVE;

// ── WinSock initialisation ──

/// `rsocket.rsocket_startup` (`WSAStartup`).  Every WinSock entry point fails
/// with `WSANOTINITIALISED` until the process has made this call, and the
/// only other things that make it are the standard library's own socket code
/// and socket2 — neither of which this module goes through.  Idempotent, so
/// the entry points below can each open with it rather than depend on an
/// initialisation order.  The process keeps the library loaded for its whole
/// life, so no `WSACleanup` pairs with it.
#[cfg(windows)]
pub fn init() {
    static STARTUP: std::sync::Once = std::sync::Once::new();
    STARTUP.call_once(majit_rlib::rsocket::rsocket_startup);
}

#[cfg(unix)]
pub fn init() {}

// ── error reporting ──

/// `_rsocket_rffi.geterrno`: the code the last socket call failed with.  The
/// `llexternal`s with `save_err` store it before the interpreter is taken
/// back, so this reads that slot — `errno` on POSIX, `WSAGetLastError` on
/// Windows (`RFFI_SAVE_WSALASTERROR`), which never touches the C runtime's
/// `errno`.
#[cfg(unix)]
pub fn last_error_code() -> i32 {
    majit_rlib::rposix::get_saved_errno()
}
#[cfg(windows)]
pub fn last_error_code() -> i32 {
    _c::geterrno()
}

pub fn last_error() -> std::io::Error {
    std::io::Error::from_raw_os_error(last_error_code())
}

/// Whether a code means "a signal arrived, nothing happened" and the call is
/// to be retried.
#[cfg(unix)]
pub fn error_is_interrupted(code: i32) -> bool {
    code == libc::EINTR
}
#[cfg(windows)]
pub fn error_is_interrupted(code: i32) -> bool {
    code == _c::WSAEINTR
}

/// Whether a code means the operation would have blocked.  PyPy keeps every
/// socket with a non-negative timeout in non-blocking mode and brackets the
/// actual operation with `RSocket._select`, so this is the result of a raced
/// readiness notification rather than a kernel socket-option timeout.
#[cfg(unix)]
pub fn error_is_would_block(code: i32) -> bool {
    code == libc::EAGAIN || code == libc::EWOULDBLOCK
}
#[cfg(windows)]
pub fn error_is_would_block(code: i32) -> bool {
    code == _c::WSAEWOULDBLOCK
}

/// `rsocket.py HAVE_SOCK_CLOEXEC` / `HAVE_SOCK_NONBLOCK`: the two creation
/// flags Linux carries inside `socket()`'s type argument.  A host without them
/// gets zero, which makes both the mask and the test that read them the
/// no-ops the `HAVE_*` guards stand for.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub const SOCK_CLOEXEC: libc::c_int = libc::SOCK_CLOEXEC;
#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub const SOCK_CLOEXEC: libc::c_int = 0;
#[cfg(any(target_os = "linux", target_os = "android"))]
pub const SOCK_NONBLOCK: libc::c_int = libc::SOCK_NONBLOCK;
#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub const SOCK_NONBLOCK: libc::c_int = 0;

/// The non-blocking `connect` result for which `RSocket._connect` waits for
/// writability before reading `SO_ERROR`.
#[cfg(unix)]
pub fn error_is_connect_in_progress(code: i32) -> bool {
    code == libc::EINPROGRESS
}
#[cfg(windows)]
pub fn error_is_connect_in_progress(code: i32) -> bool {
    code == _c::WSAEWOULDBLOCK
}

/// `gai_strerror`.  Unix only: the symbol is a header-level inline over the
/// system message table on Windows, `HAVE_GAI_STRERROR` is not defined for
/// that build, and `set_gaierror` spells the message itself there, so nothing
/// asks for it.
#[cfg(unix)]
pub fn gai_strerror(code: libc::c_int) -> String {
    unsafe { std::ffi::CStr::from_ptr(majit_rlib::_rsocket_rffi::gai_strerror(code)) }
        .to_string_lossy()
        .into_owned()
}

// ── name lookups ──

/// `interp_func.py gethostname` — the host's own name, as the opaque OS
/// string it is (`sethostname(2)` takes a plain `const char *`), so a byte or
/// code unit with no text spelling survives the caller's filesystem decode.
#[cfg(unix)]
pub fn hostname() -> std::io::Result<std::ffi::OsString> {
    use std::os::unix::ffi::OsStringExt;
    majit_rlib::rsocket::gethostname()
        .map(std::ffi::OsString::from_vec)
        .map_err(|error| std::io::Error::from_raw_os_error(error.errno))
}

/// `sethostname`. The name is the raw bytes the caller already encoded.
/// The syscall is `rsocket.sethostname`.
#[cfg(unix)]
pub fn sethostname(name: &[u8]) -> std::io::Result<()> {
    majit_rlib::rsocket::sethostname(name)
        .map_err(|error| std::io::Error::from_raw_os_error(error.errno))
}
/// `GetComputerNameExW`, which the Windows `hostname` asks instead of
/// `_c.gethostname`.
#[cfg(all(windows, feature = "host_env"))]
mod computer_name {
    use majit_rlib::rffi::{CWCHARP, INT, RFFI_SAVE_LASTERROR};
    use majit_rlib::rwin32::{BOOL, LPDWORD};

    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["windows.h"],
            libraries: ["kernel32"],
        };
    }

    majit_rlib::rffi::llexternal!(
        pub(super) GetComputerNameExW = "GetComputerNameExW",
        [INT, CWCHARP, LPDWORD],
        BOOL,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_LASTERROR
    );

    /// `ComputerNamePhysicalDnsHostname` (`COMPUTER_NAME_FORMAT`).
    pub(super) const COMPUTER_NAME_PHYSICAL_DNS_HOSTNAME: INT = 5;
    /// `MAX_COMPUTERNAME_LENGTH`.
    pub(super) const MAX_COMPUTERNAME_LENGTH: usize = 15;
    /// `ERROR_MORE_DATA`.
    pub(super) const ERROR_MORE_DATA: i64 = 234;
}

/// [3.14-spec] `GetComputerNameExW(ComputerNamePhysicalDnsHostname)` <->
/// PyPy's `rsocket.gethostname` (`_c.gethostname`, the ANSI code page's
/// bytes) — a host name the code page cannot spell comes back as its UTF-16
/// text instead of as replacement bytes; evidence: `socketmodule.c
/// socket_gethostname`'s `MS_WINDOWS` branch at v3.14.6.
#[cfg(all(windows, feature = "host_env"))]
pub fn hostname() -> std::io::Result<std::ffi::OsString> {
    use computer_name::*;
    use std::os::windows::ffi::OsStringExt;
    let last_error =
        || std::io::Error::from_raw_os_error(majit_rlib::rwin32::GetLastError_saved() as i32);
    let mut buf = [0u16; MAX_COMPUTERNAME_LENGTH + 1];
    let mut size = buf.len() as u32;
    if unsafe {
        GetComputerNameExW(
            COMPUTER_NAME_PHYSICAL_DNS_HOSTNAME,
            buf.as_mut_ptr(),
            &raw mut size,
        )
    } != 0
    {
        return Ok(std::ffi::OsString::from_wide(&buf[..size as usize]));
    }
    if majit_rlib::rwin32::GetLastError_saved() != ERROR_MORE_DATA {
        return Err(last_error());
    }
    if size == 0 {
        return Ok(std::ffi::OsString::new());
    }
    // DNS allows a longer name than the fixed buffer holds.
    let mut name = vec![0u16; size as usize];
    if unsafe {
        GetComputerNameExW(
            COMPUTER_NAME_PHYSICAL_DNS_HOSTNAME,
            name.as_mut_ptr(),
            &raw mut size,
        )
    } == 0
    {
        return Err(last_error());
    }
    name.truncate(size as usize);
    Ok(std::ffi::OsString::from_wide(&name))
}
#[cfg(all(windows, not(feature = "host_env")))]
pub fn hostname() -> std::io::Result<std::ffi::OsString> {
    Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
}

/// `interp_func.py getprotobyname` — the `IPPROTO_*` number a protocol
/// name stands for, or `None` when the database does not name it.
#[cfg(unix)]
pub fn protocol_by_name(name: &std::ffi::CStr) -> Option<libc::c_int> {
    majit_rlib::rsocket::getprotobyname(name)
        .ok()
        .and_then(|proto| libc::c_int::try_from(proto).ok())
}
#[cfg(windows)]
pub fn protocol_by_name(name: &std::ffi::CStr) -> Option<libc::c_int> {
    init();
    let entry = unsafe { _c::getprotobyname(name.as_ptr()) };
    (!entry.is_null()).then(|| unsafe { (*entry).p_proto as libc::c_int })
}

/// `inet_aton` — four address bytes in network order. The body is
/// `rsocket.inet_aton`.
#[cfg(unix)]
pub fn inet_aton(text: &std::ffi::CStr) -> Option<[u8; 4]> {
    majit_rlib::rsocket::inet_aton(text).ok()
}
#[cfg(windows)]
pub fn inet_aton(text: &std::ffi::CStr) -> Option<[u8; 4]> {
    init();
    majit_rlib::rsocket::inet_aton(text).ok()
}

/// `inet_ntoa` — the dotted-quad spelling of four address bytes. The body is
/// `rsocket.inet_ntoa`.
#[cfg(unix)]
pub fn inet_ntoa(packed: [u8; 4]) -> Option<String> {
    majit_rlib::rsocket::inet_ntoa(&packed).ok()
}
#[cfg(windows)]
pub fn inet_ntoa(packed: [u8; 4]) -> Option<String> {
    init();
    majit_rlib::rsocket::inet_ntoa(&packed).ok()
}

// ── descriptor inheritance ──

/// `rsocket.py:RSocket.__init__` closes every socket it creates over an exec
/// (PEP 446).  POSIX spells that as `FD_CLOEXEC`, Windows as the handle's
/// inherit flag.  Best-effort, like the `fcntl` call it replaces.
#[cfg(unix)]
pub fn set_cloexec(s: Socket) {
    unsafe { majit_rlib::_rsocket_rffi::fcntl(s, libc::F_SETFD, libc::FD_CLOEXEC) };
}
#[cfg(all(windows, feature = "host_env"))]
pub fn set_cloexec(s: Socket) {
    let _ = sock_set_inheritable(s, false);
}

/// `rsocket.sock_set_inheritable` / `sock_get_inheritable` on Windows: the
/// socket is a kernel handle, so `rwin32` answers for it directly.
#[cfg(all(windows, feature = "host_env"))]
fn sock_set_inheritable(fd: Socket, inheritable: bool) -> std::io::Result<()> {
    let handle = core::ptr::without_provenance_mut(fd);
    majit_rlib::rwin32::set_handle_inheritable(handle, inheritable).map_err(winerror_to_io)
}
#[cfg(all(windows, feature = "host_env"))]
fn sock_get_inheritable(fd: Socket) -> std::io::Result<bool> {
    let handle = core::ptr::without_provenance_mut(fd);
    majit_rlib::rwin32::get_handle_inheritable(handle).map_err(winerror_to_io)
}
#[cfg(all(windows, feature = "host_env"))]
fn winerror_to_io(e: majit_rlib::rwin32::WindowsError) -> std::io::Error {
    std::io::Error::from_raw_os_error(e.winerror as i32)
}
#[cfg(all(windows, not(feature = "host_env")))]
pub fn set_cloexec(_s: Socket) {}

/// `interp_socket.py set_inheritable_w` — whether an exec'd child keeps this
/// socket.  POSIX inverts `FD_CLOEXEC`, Windows reads the handle's own flag.
#[cfg(unix)]
pub fn get_inheritable(s: Socket) -> std::io::Result<bool> {
    let flags = unsafe { majit_rlib::_rsocket_rffi::fcntl(s, libc::F_GETFD, 0) };
    if flags < 0 {
        return Err(std::io::Error::from_raw_os_error(
            majit_rlib::rposix::get_saved_errno(),
        ));
    }
    Ok((flags & libc::FD_CLOEXEC) == 0)
}
#[cfg(all(windows, feature = "host_env"))]
pub fn get_inheritable(s: Socket) -> std::io::Result<bool> {
    sock_get_inheritable(s)
}
#[cfg(all(windows, not(feature = "host_env")))]
pub fn get_inheritable(_s: Socket) -> std::io::Result<bool> {
    Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
}

#[cfg(unix)]
pub fn set_inheritable(s: Socket, inheritable: bool) -> std::io::Result<()> {
    let flags = unsafe { majit_rlib::_rsocket_rffi::fcntl(s, libc::F_GETFD, 0) };
    if flags < 0 {
        return Err(std::io::Error::from_raw_os_error(
            majit_rlib::rposix::get_saved_errno(),
        ));
    }
    let wanted = if inheritable {
        flags & !libc::FD_CLOEXEC
    } else {
        flags | libc::FD_CLOEXEC
    };
    if wanted != flags && unsafe { majit_rlib::_rsocket_rffi::fcntl(s, libc::F_SETFD, wanted) } < 0
    {
        return Err(std::io::Error::from_raw_os_error(
            majit_rlib::rposix::get_saved_errno(),
        ));
    }
    Ok(())
}
#[cfg(all(windows, feature = "host_env"))]
pub fn set_inheritable(s: Socket, inheritable: bool) -> std::io::Result<()> {
    sock_set_inheritable(s, inheritable)
}
#[cfg(all(windows, not(feature = "host_env")))]
pub fn set_inheritable(_s: Socket, _inheritable: bool) -> std::io::Result<()> {
    Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
}

// ── blocking mode and timeouts ──

/// `rsocket.py:RSocket.settimeout` — put a live socket into the blocking mode
/// a Python timeout asks for.  `RSocket.settimeout` calls
/// `_setblocking(self.timeout < 0.0)`: a negative `timeout` is the `None`
/// sentinel and blocks indefinitely, while zero and positive timeouts both
/// leave the descriptor non-blocking.  Positive waits belong to `_select`,
/// not to `SO_RCVTIMEO`/`SO_SNDTIMEO`.
#[cfg(unix)]
pub fn apply_timeout(s: Socket, timeout: f64) -> std::io::Result<()> {
    let flags = unsafe { majit_rlib::_rsocket_rffi::fcntl(s, libc::F_GETFL, 0) };
    if flags < 0 {
        return Err(std::io::Error::from_raw_os_error(
            majit_rlib::rposix::get_saved_errno(),
        ));
    }
    // Bit-clear without unary `!` so the static analyzer accepts the helper
    // (the analyzer rejects bitwise-not on signed `c_int`).
    let new_flags = if timeout >= 0.0 {
        flags | libc::O_NONBLOCK
    } else if (flags & libc::O_NONBLOCK) != 0 {
        flags - libc::O_NONBLOCK
    } else {
        flags
    };
    if new_flags != flags
        && unsafe { majit_rlib::_rsocket_rffi::fcntl(s, libc::F_SETFL, new_flags) } < 0
    {
        return Err(std::io::Error::from_raw_os_error(
            majit_rlib::rposix::get_saved_errno(),
        ));
    }
    Ok(())
}

/// `RSocket._setblocking` on Windows (`ioctlsocket(FIONBIO)`).
#[cfg(windows)]
pub fn apply_timeout(s: Socket, timeout: f64) -> std::io::Result<()> {
    let mut nonblocking: std::ffi::c_ulong = std::ffi::c_ulong::from(timeout >= 0.0);
    if unsafe { _c::ioctlsocket(s, _c::FIONBIO, &raw mut nonblocking) } == _c::SOCKET_ERROR {
        return Err(last_error());
    }
    Ok(())
}

/// `RSocket._select(False)`: wait until a socket has something to read, for at
/// most `timeout_ms`.  Returns the call's result — positive when ready, `0` on
/// expiry, negative on failure — paired with the code a failure reported.
#[cfg(unix)]
pub fn poll_readable(s: Socket, timeout_ms: libc::c_int) -> (libc::c_int, i32) {
    let mut pollfd = libc::pollfd {
        fd: s,
        events: libc::POLLIN,
        revents: 0,
    };
    let ready = unsafe { majit_rlib::_rsocket_rffi::poll(&mut pollfd, 1, timeout_ms) };
    (ready, majit_rlib::rposix::get_saved_errno())
}

/// Writable half of `RSocket._select(True)`.  Keep this beside
/// [`poll_readable`]: both are the host spelling of `wait_for_data`, and a
/// finite `sendall` deadline must survive handled signals instead of starting
/// a fresh kernel timeout after every EINTR.
#[cfg(unix)]
pub fn poll_writable(s: Socket, timeout_ms: libc::c_int) -> (libc::c_int, i32) {
    let mut pollfd = libc::pollfd {
        fd: s,
        events: libc::POLLOUT,
        revents: 0,
    };
    let ready = unsafe { majit_rlib::_rsocket_rffi::poll(&mut pollfd, 1, timeout_ms) };
    (ready, majit_rlib::rposix::get_saved_errno())
}

/// The wait in `RSocket._connect`, which off win32 is `_select(True)`.
#[cfg(unix)]
pub fn poll_connect(s: Socket, timeout_ms: libc::c_int) -> (libc::c_int, i32) {
    poll_writable(s, timeout_ms)
}

/// `RSocket._select(False)` without `poll()`: `select` over a read set.
#[cfg(windows)]
pub fn poll_readable(s: Socket, timeout_ms: libc::c_int) -> (libc::c_int, i32) {
    let mut readfds = one_socket_set(s);
    let mut timeout = select_timeout(timeout_ms);
    let timeout = timeout_ptr(timeout_ms, &mut timeout);
    // WinSock ignores `nfds`: its fd_set is a counted SOCKET array, not a
    // descriptor-indexed bitmap.
    let ready = unsafe {
        _c::select(
            0,
            &raw mut readfds,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            timeout,
        )
    };
    (ready, last_error_code())
}

/// `RSocket._select(True)` without `poll()`: `select` over a write set.
#[cfg(windows)]
pub fn poll_writable(s: Socket, timeout_ms: libc::c_int) -> (libc::c_int, i32) {
    let mut writefds = one_socket_set(s);
    let mut timeout = select_timeout(timeout_ms);
    let timeout = timeout_ptr(timeout_ms, &mut timeout);
    let ready = unsafe {
        _c::select(
            0,
            core::ptr::null_mut(),
            &raw mut writefds,
            core::ptr::null_mut(),
            timeout,
        )
    };
    (ready, last_error_code())
}

/// The wait in `RSocket._connect`.  The win32 `_connect` selects with the
/// socket in `fds_exc` as well as the write set: a refused connection makes
/// the socket exceptional rather than writable, and `SO_ERROR` then carries
/// the error.
#[cfg(windows)]
pub fn poll_connect(s: Socket, timeout_ms: libc::c_int) -> (libc::c_int, i32) {
    let mut writefds = one_socket_set(s);
    let mut exceptfds = one_socket_set(s);
    let mut timeout = select_timeout(timeout_ms);
    let timeout = timeout_ptr(timeout_ms, &mut timeout);
    let ready = unsafe {
        _c::select(
            0,
            core::ptr::null_mut(),
            &raw mut writefds,
            &raw mut exceptfds,
            timeout,
        )
    };
    (ready, last_error_code())
}

/// `FD_ZERO` + `FD_SET(fd)`.  The handle is stored at its full width: a
/// WinSock `SOCKET` may exceed the `int` `FD_SET` takes.
#[cfg(windows)]
fn one_socket_set(s: Socket) -> _c::fd_set {
    let mut set = _c::fd_set {
        fd_count: 1,
        fd_array: [0; _c::FD_SETSIZE],
    };
    set.fd_array[0] = s;
    set
}

/// A negative `timeout_ms` waits indefinitely, as `poll()` reads it on the
/// unix side: `select` takes a null timeout for that.
#[cfg(windows)]
fn timeout_ptr(timeout_ms: libc::c_int, timeout: &mut _c::timeval) -> *mut _c::timeval {
    if timeout_ms < 0 {
        core::ptr::null_mut()
    } else {
        timeout
    }
}

#[cfg(windows)]
fn select_timeout(timeout_ms: libc::c_int) -> _c::timeval {
    _c::timeval {
        tv_sec: timeout_ms / 1000,
        tv_usec: (timeout_ms % 1000) * 1000,
    }
}

// ── address-structure accessors ──
//
// `_rsocket_rffi.py` `CConfig.in_addr` is `s_addr`, `CConfig.in6_addr` is
// `s6_addr`, and `CConfig.sockaddr_in6` carries `sin6_scope_id`, as plain
// fields on both platforms.

pub fn sockaddr_in_get_addr(sin: &sockaddr_in) -> u32 {
    sin.sin_addr.s_addr
}

/// The IPv4 address in network byte order, as `inet_pton` writes it.
pub fn sockaddr_in_set_addr(sin: &mut sockaddr_in, addr: u32) {
    sin.sin_addr.s_addr = addr;
}

pub fn sockaddr_in6_get_addr(sin6: &sockaddr_in6) -> [u8; 16] {
    sin6.sin6_addr.s6_addr
}

pub fn sockaddr_in6_set_addr(sin6: &mut sockaddr_in6, addr: [u8; 16]) {
    sin6.sin6_addr.s6_addr = addr;
}

pub fn sockaddr_in6_get_scope_id(sin6: &sockaddr_in6) -> u32 {
    sin6.sin6_scope_id
}

pub fn sockaddr_in6_set_scope_id(sin6: &mut sockaddr_in6, scope_id: u32) {
    sin6.sin6_scope_id = scope_id;
}

// ── the calls themselves ──
//
// One signature per call, taking the widths the shared body already works in:
// byte counts as `usize`, transfer results as `isize`.  WinSock's `int`-shaped
// prototypes are narrowed here, at the single place that knows the ceiling.

/// WinSock counts a transfer in an `int`, so a single call can move at most
/// `i32::MAX` bytes; a longer request is served over several calls, exactly as
/// a short read leaves the caller looping today.
#[cfg(windows)]
fn clamp_transfer_len(len: usize) -> i32 {
    len.min(i32::MAX as usize) as i32
}

#[cfg(unix)]
pub unsafe fn socket(family: libc::c_int, ty: libc::c_int, proto: libc::c_int) -> Socket {
    unsafe { majit_rlib::_rsocket_rffi::socket(family, ty, proto) }
}
#[cfg(windows)]
pub unsafe fn socket(family: libc::c_int, ty: libc::c_int, proto: libc::c_int) -> Socket {
    init();
    unsafe { _c::socket(family, ty, proto) }
}

pub unsafe fn close(s: Socket) -> libc::c_int {
    unsafe { majit_rlib::_rsocket_rffi::socketclose(s) }
}

pub unsafe fn bind(s: Socket, addr: *const sockaddr, len: SockLen) -> libc::c_int {
    unsafe { majit_rlib::_rsocket_rffi::socketbind(s, addr, len) }
}

pub unsafe fn connect(s: Socket, addr: *const sockaddr, len: SockLen) -> libc::c_int {
    unsafe { majit_rlib::_rsocket_rffi::socketconnect(s, addr, len) }
}

pub unsafe fn listen(s: Socket, backlog: libc::c_int) -> libc::c_int {
    unsafe { majit_rlib::_rsocket_rffi::socketlisten(s, backlog) }
}

pub unsafe fn accept(s: Socket, addr: *mut sockaddr, len: *mut SockLen) -> Socket {
    unsafe { majit_rlib::_rsocket_rffi::socketaccept(s, addr, len) }
}

#[cfg(unix)]
pub unsafe fn send(
    s: Socket,
    buf: *const core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
) -> isize {
    unsafe { majit_rlib::_rsocket_rffi::send(s, buf, len, flags) }
}
#[cfg(windows)]
pub unsafe fn send(
    s: Socket,
    buf: *const core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
) -> isize {
    unsafe { _c::send(s, buf, clamp_transfer_len(len), flags) as isize }
}

#[cfg(unix)]
pub unsafe fn recv(
    s: Socket,
    buf: *mut core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
) -> isize {
    unsafe { majit_rlib::_rsocket_rffi::socketrecv(s, buf, len, flags) }
}
#[cfg(windows)]
pub unsafe fn recv(
    s: Socket,
    buf: *mut core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
) -> isize {
    unsafe { _c::socketrecv(s, buf, clamp_transfer_len(len), flags) as isize }
}

#[cfg(unix)]
pub unsafe fn sendto(
    s: Socket,
    buf: *const core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
    addr: *const sockaddr,
    addrlen: SockLen,
) -> isize {
    unsafe { majit_rlib::_rsocket_rffi::sendto(s, buf, len, flags, addr, addrlen) }
}
#[cfg(windows)]
pub unsafe fn sendto(
    s: Socket,
    buf: *const core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
    addr: *const sockaddr,
    addrlen: SockLen,
) -> isize {
    unsafe { _c::sendto(s, buf, clamp_transfer_len(len), flags, addr, addrlen) as isize }
}

#[cfg(unix)]
pub unsafe fn recvfrom(
    s: Socket,
    buf: *mut core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
    addr: *mut sockaddr,
    addrlen: *mut SockLen,
) -> isize {
    unsafe { majit_rlib::_rsocket_rffi::recvfrom(s, buf, len, flags, addr, addrlen) }
}
#[cfg(windows)]
pub unsafe fn recvfrom(
    s: Socket,
    buf: *mut core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
    addr: *mut sockaddr,
    addrlen: *mut SockLen,
) -> isize {
    unsafe { _c::recvfrom(s, buf, clamp_transfer_len(len), flags, addr, addrlen) as isize }
}

pub unsafe fn getsockname(s: Socket, addr: *mut sockaddr, len: *mut SockLen) -> libc::c_int {
    unsafe { majit_rlib::_rsocket_rffi::socketgetsockname(s, addr, len) }
}

pub unsafe fn getpeername(s: Socket, addr: *mut sockaddr, len: *mut SockLen) -> libc::c_int {
    unsafe { majit_rlib::_rsocket_rffi::socketgetpeername(s, addr, len) }
}

pub unsafe fn getsockopt(
    s: Socket,
    level: libc::c_int,
    option: libc::c_int,
    value: *mut core::ffi::c_void,
    len: *mut SockLen,
) -> libc::c_int {
    unsafe { majit_rlib::_rsocket_rffi::socketgetsockopt(s, level, option, value, len) }
}

/// `WSAIoctl` with an input value and no output buffer, which is the shape
/// every command `sock_ioctl` takes is issued in.  Answers the `DWORD` the
/// call reports as returned, or `None` with `WSAGetLastError` set.
///
/// # Safety
/// `value` must address `len` initialised bytes for the duration of the call.
#[cfg(windows)]
pub unsafe fn wsa_ioctl(
    s: Socket,
    code: u32,
    value: *const core::ffi::c_void,
    len: u32,
) -> Option<u32> {
    let mut returned: u32 = 0;
    let result = unsafe {
        _c::WSAIoctl(
            s,
            code,
            value.cast_mut(),
            len,
            core::ptr::null_mut(),
            0,
            &raw mut returned,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        )
    };
    (result != _c::SOCKET_ERROR).then_some(returned)
}

/// `SIO_TCP_SET_ACK_FREQUENCY`, which is what `TCP_QUICKACK` names here.
/// It is an ioctl rather than a socket option, so `setsockopt` answers
/// `WSAENOPROTOOPT` for it and the flag has to travel through `WSAIoctl`.
/// Nothing reads it back: WinSock exposes no query for the current setting.
#[cfg(windows)]
pub unsafe fn set_ack_frequency(s: Socket, flag: libc::c_int) -> libc::c_int {
    let code = _c::SIO_TCP_SET_ACK_FREQUENCY;
    let sent = unsafe {
        wsa_ioctl(
            s,
            code,
            (&raw const flag).cast(),
            core::mem::size_of::<libc::c_int>() as u32,
        )
    };
    if sent.is_some() { 0 } else { _c::SOCKET_ERROR }
}

pub unsafe fn setsockopt(
    s: Socket,
    level: libc::c_int,
    option: libc::c_int,
    value: *const core::ffi::c_void,
    len: SockLen,
) -> libc::c_int {
    unsafe { majit_rlib::_rsocket_rffi::socketsetsockopt(s, level, option, value, len) }
}

pub unsafe fn shutdown(s: Socket, how: libc::c_int) -> libc::c_int {
    unsafe { majit_rlib::_rsocket_rffi::socketshutdown(s, how) }
}

#[cfg(unix)]
pub unsafe fn getaddrinfo(
    node: *const libc::c_char,
    service: *const libc::c_char,
    hints: *const addrinfo,
    res: *mut *mut addrinfo,
) -> libc::c_int {
    unsafe { majit_rlib::_rsocket_rffi::getaddrinfo(node, service, hints, res) }
}
#[cfg(windows)]
pub unsafe fn getaddrinfo(
    node: *const libc::c_char,
    service: *const libc::c_char,
    hints: *const addrinfo,
    res: *mut *mut addrinfo,
) -> libc::c_int {
    init();
    unsafe { _c::getaddrinfo(node, service, hints, res) }
}

pub unsafe fn freeaddrinfo(res: *mut addrinfo) {
    unsafe { majit_rlib::_rsocket_rffi::freeaddrinfo(res) }
}

#[cfg(unix)]
pub unsafe fn getnameinfo(
    addr: *const sockaddr,
    addrlen: SockLen,
    host: *mut libc::c_char,
    hostlen: SockLen,
    service: *mut libc::c_char,
    servicelen: SockLen,
    flags: libc::c_int,
) -> libc::c_int {
    unsafe {
        majit_rlib::_rsocket_rffi::getnameinfo(
            addr, addrlen, host, hostlen, service, servicelen, flags,
        )
    }
}
#[cfg(windows)]
pub unsafe fn getnameinfo(
    addr: *const sockaddr,
    addrlen: SockLen,
    host: *mut libc::c_char,
    hostlen: SockLen,
    service: *mut libc::c_char,
    servicelen: SockLen,
    flags: libc::c_int,
) -> libc::c_int {
    init();
    unsafe {
        _c::getnameinfo(
            addr,
            addrlen,
            host,
            hostlen as u32,
            service,
            servicelen as u32,
            flags,
        )
    }
}

#[cfg(unix)]
pub unsafe fn inet_pton(
    family: libc::c_int,
    src: *const libc::c_char,
    dst: *mut core::ffi::c_void,
) -> libc::c_int {
    unsafe { majit_rlib::_rsocket_rffi::inet_pton(family, src, dst) }
}
#[cfg(windows)]
pub unsafe fn inet_pton(
    family: libc::c_int,
    src: *const libc::c_char,
    dst: *mut core::ffi::c_void,
) -> libc::c_int {
    init();
    unsafe { _c::inet_pton(family, src, dst) }
}

#[cfg(unix)]
pub unsafe fn inet_ntop(
    family: libc::c_int,
    src: *const core::ffi::c_void,
    dst: *mut libc::c_char,
    size: SockLen,
) -> *const libc::c_char {
    unsafe { majit_rlib::_rsocket_rffi::inet_ntop(family, src, dst, size) }
}
#[cfg(windows)]
pub unsafe fn inet_ntop(
    family: libc::c_int,
    src: *const core::ffi::c_void,
    dst: *mut libc::c_char,
    size: SockLen,
) -> *const libc::c_char {
    init();
    unsafe { _c::inet_ntop(family, src, dst, size as usize) }
}

/// `inet_pton`'s two failures, which it reports through different channels: a
/// negative return leaves `EAFNOSUPPORT` behind, a zero says the string was
/// read and rejected.
pub enum PtonError {
    /// A family with no parser behind it.
    Family(i32),
    /// An address this family's parser refused.
    Address,
}

/// `inet_pton` over the buffer it fills, sized by the family it was given.
pub fn pton(family: libc::c_int, text: &std::ffi::CStr) -> Result<Vec<u8>, PtonError> {
    #[cfg(windows)]
    init();
    match majit_rlib::rsocket::inet_pton(family, text) {
        Ok(packed) => Ok(packed),
        Err(majit_rlib::rsocket::PtonError::Family(code)) => Err(PtonError::Family(code)),
        Err(majit_rlib::rsocket::PtonError::Address) => Err(PtonError::Address),
    }
}

/// `inet_ntop` over the text buffer it fills.
pub fn ntop(family: libc::c_int, packed: &[u8]) -> Option<String> {
    #[cfg(windows)]
    init();
    majit_rlib::rsocket::inet_ntop(family, packed).ok()
}

// The legacy `<netdb.h>` resolvers.
//
// `libc` 0.2.186 declares none of them, and the two records they answer with
// are not the same type on both platforms: `hostent`'s `h_addrtype` and
// `h_length` are `int` on POSIX and `short` in WinSock, and `servent` orders
// `s_proto` before `s_port` on Win64 and the other way round everywhere else.
// A single `#[repr(C)]` mirror would therefore be wrong on one side, so the
// record stays opaque and every field is read through an accessor.

/// The resolver's `hostent` record. Never constructed here — only the pointer
/// the resolver returns is ever held, and it points into process-global
/// storage that the next lookup overwrites.
#[cfg(unix)]
#[repr(C)]
#[allow(non_snake_case)]
pub struct Hostent {
    h_name: *const libc::c_char,
    h_aliases: *mut *mut libc::c_char,
    h_addrtype: libc::c_int,
    h_length: libc::c_int,
    h_addr_list: *mut *mut libc::c_char,
}
#[cfg(windows)]
pub type Hostent = _c::hostent;

/// The resolver's `servent` record, held on the same terms as [`Hostent`].
#[cfg(unix)]
#[repr(C)]
#[allow(non_snake_case)]
pub struct Servent {
    s_name: *const libc::c_char,
    s_aliases: *mut *mut libc::c_char,
    s_port: libc::c_int,
    s_proto: *const libc::c_char,
}
#[cfg(windows)]
pub type Servent = _c::servent;

/// `rsocket._get_netdb_lock_thread`: the lookups below answer with a pointer
/// into one process-global record, so a second lookup on another thread
/// invalidates the first one's answer. Hold this across both the lookup and
/// the copying out of everything the caller needs.
pub fn netdb_lock() -> parking_lot::MutexGuard<'static, ()> {
    static NETDB_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
    NETDB_LOCK.lock()
}

/// Read one slot of a `char **` array the resolver owns.
///
/// Darwin packs these arrays at byte-aligned addresses — immediately after a C
/// string, for instance. C permits the resulting unaligned pointer load and
/// Rust references do not, so the slot is read as raw storage, matching
/// `rsocket.gethost_common`'s rffi access rather than manufacturing an aligned
/// reference.
pub unsafe fn pointer_at(array: *mut *mut libc::c_char, index: usize) -> *mut libc::c_char {
    unsafe { std::ptr::read_unaligned(array.add(index)) }
}

/// # Safety
/// `name` must be a valid NUL-terminated C string. The returned pointer is
/// borrowed from process-global storage — see [`netdb_lock`].
pub unsafe fn host_by_name(name: *const libc::c_char) -> *mut Hostent {
    #[cfg(unix)]
    {
        unsafe { majit_rlib::_rsocket_rffi::gethostbyname(name).cast() }
    }
    #[cfg(windows)]
    {
        init();
        unsafe { _c::gethostbyname(name) }
    }
}

/// # Safety
/// `addr` must point to `len` readable bytes, and the answer is borrowed —
/// see [`host_by_name`].
pub unsafe fn host_by_addr(
    addr: *const libc::c_void,
    len: SockLen,
    family: libc::c_int,
) -> *mut Hostent {
    #[cfg(unix)]
    {
        unsafe { majit_rlib::_rsocket_rffi::gethostbyaddr(addr, len, family).cast() }
    }
    #[cfg(windows)]
    {
        init();
        unsafe { _c::gethostbyaddr(addr.cast(), len, family) }
    }
}

/// The code the resolver reports a failed name or address lookup with, and the
/// platform's message for it.
///
/// `gethostbyname` / `gethostbyaddr` leave it in `h_errno` rather than `errno`,
/// and `set_herror` raises `herror(code, hstrerror(code))` from the pair.  The
/// variable is spelled three ways: glibc and musl make it a thread-local behind
/// `__h_errno_location`, the BSDs export the plain global `<netdb.h>` declares,
/// and winsock defines it as `WSAGetLastError()`.
pub fn host_error() -> (libc::c_int, String) {
    #[cfg(all(unix, any(target_os = "linux", target_os = "android")))]
    let code = {
        unsafe extern "C" {
            fn __h_errno_location() -> *mut libc::c_int;
        }
        unsafe { *__h_errno_location() }
    };
    #[cfg(all(unix, not(any(target_os = "linux", target_os = "android"))))]
    let code = {
        unsafe extern "C" {
            static h_errno: libc::c_int;
        }
        unsafe { h_errno }
    };
    #[cfg(windows)]
    let code = last_error_code();

    #[cfg(unix)]
    let message = {
        let p = unsafe { libc::hstrerror(code) };
        if p.is_null() {
            "host not found".to_string()
        } else {
            unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned()
        }
    };
    // No `hstrerror` on Windows, where `set_herror` falls back to the literal.
    #[cfg(windows)]
    let message = "host not found".to_string();

    (code, message)
}

/// # Safety
/// Both arguments must be valid NUL-terminated C strings or null, and the
/// answer is borrowed — see [`host_by_name`].
pub unsafe fn serv_by_name(name: *const libc::c_char, proto: *const libc::c_char) -> *mut Servent {
    #[cfg(unix)]
    {
        unsafe { majit_rlib::_rsocket_rffi::getservbyname(name, proto).cast() }
    }
    #[cfg(windows)]
    {
        init();
        unsafe { _c::getservbyname(name, proto) }
    }
}

/// # Safety
/// `proto` must be a valid NUL-terminated C string or null, and the answer is
/// borrowed — see [`host_by_name`].
pub unsafe fn serv_by_port(port: libc::c_int, proto: *const libc::c_char) -> *mut Servent {
    #[cfg(unix)]
    {
        unsafe { majit_rlib::_rsocket_rffi::getservbyport(port, proto).cast() }
    }
    #[cfg(windows)]
    {
        init();
        unsafe { _c::getservbyport(port, proto) }
    }
}

/// # Safety
/// `h` must be a live answer from one of the host lookups above.
pub unsafe fn hostent_name(h: *mut Hostent) -> *const libc::c_char {
    unsafe { (*h).h_name as *const libc::c_char }
}

/// # Safety
/// See [`hostent_name`].
pub unsafe fn hostent_aliases(h: *mut Hostent) -> *mut *mut libc::c_char {
    unsafe { (*h).h_aliases as *mut *mut libc::c_char }
}

/// # Safety
/// See [`hostent_name`].
pub unsafe fn hostent_addr_type(h: *mut Hostent) -> libc::c_int {
    unsafe { (*h).h_addrtype as libc::c_int }
}

/// # Safety
/// See [`hostent_name`].
pub unsafe fn hostent_length(h: *mut Hostent) -> libc::c_int {
    unsafe { (*h).h_length as libc::c_int }
}

/// # Safety
/// See [`hostent_name`].
pub unsafe fn hostent_addr_list(h: *mut Hostent) -> *mut *mut libc::c_char {
    unsafe { (*h).h_addr_list as *mut *mut libc::c_char }
}

/// # Safety
/// `s` must be a live answer from one of the service lookups above.
pub unsafe fn servent_name(s: *mut Servent) -> *const libc::c_char {
    unsafe { (*s).s_name as *const libc::c_char }
}

/// The port as the record stores it: network byte order, in the low 16 bits.
///
/// # Safety
/// See [`servent_name`].
pub unsafe fn servent_port(s: *mut Servent) -> u16 {
    unsafe { (*s).s_port as u16 }
}
