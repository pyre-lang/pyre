//! `rpython/rlib/rsocket.py` — socket calls above `_rsocket_rffi`.
//!
//! `look_inside_function` rejects this module exactly. `_rsocket_rffi` stays
//! visible to the tracer.

use crate::rffi::{INT, SIGNED};

/// `CSocketError`. `errno` is the code `last_error` read.
#[derive(Debug)]
pub struct CSocketError {
    pub errno: i32,
}

/// `last_error` — `CSocketError(geterrno())`.
pub fn last_error() -> CSocketError {
    CSocketError {
        errno: crate::_rsocket_rffi::geterrno(),
    }
}

/// `rsocket_startup`.
#[cfg(windows)]
pub fn rsocket_startup() {
    let mut wsadata: crate::_rsocket_rffi::WSAData = unsafe { std::mem::zeroed() };
    let res = unsafe { crate::_rsocket_rffi::WSAStartup(0x0101, &raw mut wsadata) };
    assert_eq!(res, 0);
}

/// `rsocket_startup` off Windows.
#[cfg(not(windows))]
pub fn rsocket_startup() {}

#[cfg(unix)]
type Fd = INT;
#[cfg(windows)]
type Fd = usize;

/// `getsockopt_int` — one `int` socket option.
///
/// A WinSock boolean option may write a single byte. The value starts at
/// zero so the unread bytes stay zero.
#[majit_macros::dont_look_inside]
pub fn getsockopt_int(fd: Fd, level: INT, option: INT) -> Result<SIGNED, CSocketError> {
    let mut flag: INT = 0;
    #[cfg(unix)]
    let mut flagsize: libc::socklen_t = std::mem::size_of::<INT>() as libc::socklen_t;
    #[cfg(windows)]
    let mut flagsize: INT = std::mem::size_of::<INT>() as INT;
    let res = unsafe {
        crate::_rsocket_rffi::socketgetsockopt(
            fd,
            level,
            option,
            (&raw mut flag).cast(),
            &raw mut flagsize,
        )
    };
    if res < 0 {
        return Err(last_error());
    }
    Ok(flag as SIGNED)
}

/// `ntohs` — `_c.ntohs` as a host integer.
pub fn ntohs(x: crate::rffi::USHORT) -> i64 {
    unsafe { crate::_rsocket_rffi::ntohs(x) as i64 }
}

/// `ntohl` — `_c.ntohl` as a host integer.
pub fn ntohl(x: crate::rffi::UINT) -> i64 {
    unsafe { crate::_rsocket_rffi::ntohl(x) as i64 }
}

/// `htons` — `_c.htons` as a host integer.
pub fn htons(x: crate::rffi::USHORT) -> i64 {
    unsafe { crate::_rsocket_rffi::htons(x) as i64 }
}

/// `htonl` — `_c.htonl` as a host integer.
pub fn htonl(x: crate::rffi::UINT) -> i64 {
    unsafe { crate::_rsocket_rffi::htonl(x) as i64 }
}

/// `Defaults.timeout`. `-1.0` blocks. Every thread reads the same cell.
static DEFAULT_TIMEOUT_BITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new((-1.0f64).to_bits());

/// `getdefaulttimeout`.
pub fn getdefaulttimeout() -> f64 {
    f64::from_bits(DEFAULT_TIMEOUT_BITS.load(std::sync::atomic::Ordering::Relaxed))
}

/// `setdefaulttimeout`. A negative value is stored as `-1.0`.
pub fn setdefaulttimeout(timeout: f64) {
    let timeout = if timeout < 0.0 { -1.0 } else { timeout };
    DEFAULT_TIMEOUT_BITS.store(timeout.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// Why [`wait_for_data`] stopped without reporting readiness.
#[derive(Debug)]
pub enum WaitError<E> {
    /// The deadline passed before the socket was ready.
    Timeout,
    /// `poll` failed with an errno other than an interruption.
    Os(CSocketError),
    /// The signal check refused to continue.
    Check(E),
}

/// One `_select` inside [`wait_for_data`].
///
/// Interrupted is the platform's "a signal arrived" result (`EINTR`,
/// `WSAEINTR`). The caller classifies it; this module retries either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PollWait {
    /// The socket is ready.
    Ready,
    /// The slice elapsed with nothing ready.
    TimedOut,
    /// The call was interrupted and should be retried.
    Interrupted,
    /// The call failed. `errno` is the code it reported.
    Failed(i32),
}

/// `RSocket.wait_for_data` against one absolute deadline.
///
/// `poll(fd, for_writing, timeout_ms)` is one `_select`. [`PollWait::Ready`]
/// answers `Ok(true)`. [`PollWait::TimedOut`] and [`PollWait::Interrupted`]
/// both run `check_signals` and wait again until `deadline`. The millisecond
/// argument is at least 1 and at most `i32::MAX`; on macOS it is also at most
/// 50 so a restarted `poll` still returns to the signal check.
pub fn wait_for_data<E>(
    fd: Fd,
    for_writing: bool,
    deadline: std::time::Instant,
    mut poll: impl FnMut(Fd, bool, i32) -> PollWait,
    mut check_signals: impl FnMut() -> Result<(), E>,
) -> Result<bool, WaitError<E>> {
    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return Err(WaitError::Timeout);
        }
        let timeout_ms = remaining.as_millis().max(1).min(i32::MAX as u128) as i32;
        // Darwin may restart poll after a signal instead of returning EINTR.
        #[cfg(target_os = "macos")]
        let timeout_ms = timeout_ms.min(50);
        match poll(fd, for_writing, timeout_ms) {
            PollWait::Ready => return Ok(true),
            PollWait::TimedOut | PollWait::Interrupted => {
                check_signals().map_err(WaitError::Check)?;
            }
            PollWait::Failed(errno) => return Err(WaitError::Os(CSocketError { errno })),
        }
    }
}

/// `RSocketError`. `message` is `get_msg`.
#[derive(Debug)]
pub struct RSocketError {
    pub message: &'static str,
}

fn rsocket_error(message: &'static str) -> RSocketError {
    RSocketError { message }
}

/// `inet_aton`. Four address bytes, network order.
///
/// Windows has no `inet_aton`. `inet_addr` reports failure as `INADDR_NONE`,
/// which is also the broadcast address, so that spelling is answered first.
#[cfg(unix)]
pub fn inet_aton(ip: &std::ffi::CStr) -> Result<[u8; 4], RSocketError> {
    let mut addr: libc::in_addr = unsafe { std::mem::zeroed() };
    let ok = unsafe { crate::_rsocket_rffi::inet_aton(ip.as_ptr(), &raw mut addr) };
    if ok == 0 {
        return Err(rsocket_error(
            "illegal IP address string passed to inet_aton",
        ));
    }
    Ok(addr.s_addr.to_ne_bytes())
}

#[cfg(windows)]
pub fn inet_aton(ip: &std::ffi::CStr) -> Result<[u8; 4], RSocketError> {
    if ip.to_bytes() == b"255.255.255.255" {
        return Ok([0xff; 4]);
    }
    let packed = unsafe { crate::_rsocket_rffi::inet_addr(ip.as_ptr()) };
    if packed == crate::rffi::UINT::MAX {
        return Err(rsocket_error(
            "illegal IP address string passed to inet_aton",
        ));
    }
    Ok((packed as u32).to_ne_bytes())
}

/// `inet_ntoa`. The packed buffer is `sizeof(in_addr)` bytes.
pub fn inet_ntoa(packed: &[u8]) -> Result<String, RSocketError> {
    #[cfg(unix)]
    let width = std::mem::size_of::<libc::in_addr>();
    #[cfg(windows)]
    let width = std::mem::size_of::<crate::_rsocket_rffi::in_addr>();
    if packed.len() != width {
        return Err(rsocket_error("packed IP wrong length for inet_ntoa"));
    }
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(packed);
    #[cfg(unix)]
    let addr = libc::in_addr {
        s_addr: u32::from_ne_bytes(bytes),
    };
    #[cfg(windows)]
    let addr = crate::_rsocket_rffi::in_addr {
        s_addr: u32::from_ne_bytes(bytes),
    };
    let text = unsafe { crate::_rsocket_rffi::inet_ntoa(addr) };
    if text.is_null() {
        return Err(rsocket_error("inet_ntoa failed"));
    }
    Ok(unsafe { std::ffi::CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned())
}

/// `inet_pton` failures. A negative return leaves an errno; zero rejects the text.
#[derive(Debug)]
pub enum PtonError {
    /// `res < 0`. `errno` is the code `last_error` read.
    Family(i32),
    /// `res == 0`.
    Address,
}

fn af_inet() -> INT {
    #[cfg(unix)]
    {
        libc::AF_INET
    }
    #[cfg(windows)]
    {
        crate::_rsocket_rffi::AF_INET
    }
}

/// `inet_pton`. Four bytes for `AF_INET`, sixteen otherwise.
pub fn inet_pton(family: INT, ip: &std::ffi::CStr) -> Result<Vec<u8>, PtonError> {
    let mut buf = [0u8; 16];
    let res =
        unsafe { crate::_rsocket_rffi::inet_pton(family, ip.as_ptr(), buf.as_mut_ptr().cast()) };
    if res < 0 {
        return Err(PtonError::Family(last_error().errno));
    }
    if res == 0 {
        return Err(PtonError::Address);
    }
    let width = if family == af_inet() { 4 } else { 16 };
    Ok(buf[..width].to_vec())
}

/// `inet_ntop`. The packed bytes are the family width the caller checked.
pub fn inet_ntop(family: INT, packed: &[u8]) -> Result<String, CSocketError> {
    let mut buf = [0u8; 64];
    #[cfg(unix)]
    let text = unsafe {
        crate::_rsocket_rffi::inet_ntop(
            family,
            packed.as_ptr().cast(),
            buf.as_mut_ptr().cast(),
            buf.len() as libc::socklen_t,
        )
    };
    #[cfg(windows)]
    let text = unsafe {
        crate::_rsocket_rffi::inet_ntop(
            family,
            packed.as_ptr().cast(),
            buf.as_mut_ptr().cast(),
            buf.len(),
        )
    };
    if text.is_null() {
        return Err(last_error());
    }
    Ok(unsafe { std::ffi::CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned())
}

/// `gethostname`. The buffer is 1024 bytes. The result stops at the first NUL.
#[cfg(unix)]
pub fn gethostname() -> Result<Vec<u8>, CSocketError> {
    let mut buf = [0u8; 1024];
    let res = unsafe { crate::_rsocket_rffi::gethostname(buf.as_mut_ptr().cast(), buf.len()) };
    if res < 0 {
        return Err(last_error());
    }
    let end = buf.iter().position(|&byte| byte == 0).unwrap_or(buf.len());
    Ok(buf[..end].to_vec())
}

#[cfg(unix)]
fn proto_ptr(proto: Option<&std::ffi::CStr>) -> *const std::ffi::c_char {
    proto
        .map(std::ffi::CStr::as_ptr)
        .unwrap_or(std::ptr::null())
}

/// `getservbyname`. `proto` null matches any protocol. The port is host order.
#[cfg(unix)]
pub fn getservbyname(
    name: &std::ffi::CStr,
    proto: Option<&std::ffi::CStr>,
) -> Result<i64, RSocketError> {
    let servent = unsafe { crate::_rsocket_rffi::getservbyname(name.as_ptr(), proto_ptr(proto)) };
    if servent.is_null() {
        return Err(rsocket_error("service/proto not found"));
    }
    let port = unsafe { (*servent).s_port } as crate::rffi::USHORT;
    Ok(ntohs(port))
}

/// `getservbyport`. `port` is host order. The name is copied off the record.
#[cfg(unix)]
pub fn getservbyport(port: i32, proto: Option<&std::ffi::CStr>) -> Result<String, RSocketError> {
    let net = htons(port as crate::rffi::USHORT) as crate::rffi::INT;
    let servent = unsafe { crate::_rsocket_rffi::getservbyport(net, proto_ptr(proto)) };
    if servent.is_null() {
        return Err(rsocket_error("port/proto not found"));
    }
    let name = unsafe { (*servent).s_name };
    if name.is_null() {
        return Err(rsocket_error("port/proto not found"));
    }
    Ok(unsafe { std::ffi::CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned())
}

/// `if_nameindex`. Names are copied out before `if_freenameindex`.
#[cfg(unix)]
pub fn if_nameindex() -> Result<Vec<(u32, Vec<u8>)>, CSocketError> {
    let head = unsafe { crate::_rsocket_rffi::if_nameindex() };
    if head.is_null() {
        return Err(last_error());
    }
    let mut out = Vec::new();
    unsafe {
        let mut entry = head;
        while (*entry).if_index != 0 && !(*entry).if_name.is_null() {
            let name = std::ffi::CStr::from_ptr((*entry).if_name)
                .to_bytes()
                .to_vec();
            out.push(((*entry).if_index, name));
            entry = entry.add(1);
        }
        crate::_rsocket_rffi::if_freenameindex(head);
    }
    Ok(out)
}

/// `getprotobyname`. The number is `p_proto`.
#[cfg(unix)]
pub fn getprotobyname(name: &std::ffi::CStr) -> Result<i64, RSocketError> {
    let entry = unsafe { crate::_rsocket_rffi::getprotobyname(name.as_ptr()) };
    if entry.is_null() {
        return Err(rsocket_error("protocol not found"));
    }
    Ok(i64::from(unsafe { (*entry).p_proto }))
}

/// `GAIError`. `errno` is the `getaddrinfo` / `getnameinfo` code.
#[cfg(unix)]
#[derive(Debug)]
pub struct GAIError {
    pub errno: i32,
}

/// One `getaddrinfo` result. `addr` is the `sockaddr` `make_address` copies.
#[cfg(unix)]
#[derive(Debug)]
pub struct Addrinfo {
    pub family: i64,
    pub socktype: i64,
    pub protocol: i64,
    pub canonname: String,
    pub addr: Vec<u8>,
}

#[cfg(unix)]
struct FreeAddrinfo(*mut libc::addrinfo);

#[cfg(unix)]
impl Drop for FreeAddrinfo {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { crate::_rsocket_rffi::freeaddrinfo(self.0) };
        }
    }
}

/// `getaddrinfo`. `port_or_service` is text or absent. On macOS a
/// numeric-service lookup of an absent port or `"0"` is passed as `"00"`.
#[cfg(unix)]
pub fn getaddrinfo(
    host: Option<&std::ffi::CStr>,
    port_or_service: Option<&std::ffi::CStr>,
    family: INT,
    socktype: INT,
    proto: INT,
    flags: INT,
) -> Result<Vec<Addrinfo>, GAIError> {
    #[cfg(target_os = "macos")]
    let port_or_service = if (flags & libc::AI_NUMERICSERV) != 0
        && port_or_service.is_none_or(|port| port.to_bytes() == b"0")
    {
        Some(c"00")
    } else {
        port_or_service
    };
    let mut hints: libc::addrinfo = unsafe { std::mem::zeroed() };
    hints.ai_family = family;
    hints.ai_socktype = socktype;
    hints.ai_protocol = proto;
    hints.ai_flags = flags;
    let mut res: *mut libc::addrinfo = std::ptr::null_mut();
    let host_ptr = host.map(std::ffi::CStr::as_ptr).unwrap_or(std::ptr::null());
    let port_ptr = port_or_service
        .map(std::ffi::CStr::as_ptr)
        .unwrap_or(std::ptr::null());
    let error = unsafe { crate::_rsocket_rffi::getaddrinfo(host_ptr, port_ptr, &hints, &mut res) };
    if error != 0 {
        return Err(GAIError { errno: error });
    }
    let guard = FreeAddrinfo(res);
    let mut out = Vec::new();
    let mut info = guard.0;
    while !info.is_null() {
        let ai = unsafe { &*info };
        let addrlen = ai.ai_addrlen as usize;
        let addr = if ai.ai_addr.is_null() || addrlen == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(ai.ai_addr.cast::<u8>(), addrlen).to_vec() }
        };
        let canonname = if ai.ai_canonname.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(ai.ai_canonname) }
                .to_string_lossy()
                .into_owned()
        };
        out.push(Addrinfo {
            family: i64::from(ai.ai_family),
            socktype: i64::from(ai.ai_socktype),
            protocol: i64::from(ai.ai_protocol),
            canonname,
            addr,
        });
        info = ai.ai_next;
    }
    Ok(out)
}

/// `NI_MAXSERV`. The libc crate omits it on linux-gnu. The header value is 32.
#[cfg(unix)]
const NI_MAXSERV: usize = 32;

/// `getnameinfo`. `addr` is the `sockaddr` bytes and its length is `addrlen`.
#[cfg(unix)]
pub fn getnameinfo(addr: &[u8], flags: INT) -> Result<(String, String), GAIError> {
    let mut host = vec![0u8; libc::NI_MAXHOST as usize];
    let mut serv = vec![0u8; NI_MAXSERV];
    let error = unsafe {
        crate::_rsocket_rffi::getnameinfo(
            addr.as_ptr().cast(),
            addr.len() as libc::socklen_t,
            host.as_mut_ptr().cast(),
            host.len() as libc::socklen_t,
            serv.as_mut_ptr().cast(),
            serv.len() as libc::socklen_t,
            flags,
        )
    };
    if error != 0 {
        return Err(GAIError { errno: error });
    }
    let host_s = unsafe { std::ffi::CStr::from_ptr(host.as_ptr().cast()) }
        .to_string_lossy()
        .into_owned();
    let serv_s = unsafe { std::ffi::CStr::from_ptr(serv.as_ptr().cast()) }
        .to_string_lossy()
        .into_owned();
    Ok((host_s, serv_s))
}

/// `sethostname`. `hostname` is the raw byte count passed to the syscall.
#[cfg(unix)]
pub fn sethostname(hostname: &[u8]) -> Result<(), CSocketError> {
    let ptr = if hostname.is_empty() {
        c"".as_ptr()
    } else {
        hostname.as_ptr().cast()
    };
    let res = unsafe { crate::_rsocket_rffi::sethostname(ptr, hostname.len()) };
    if res < 0 {
        return Err(last_error());
    }
    Ok(())
}

/// `socket`. The new descriptor has `FD_CLOEXEC` set.
#[cfg(unix)]
pub fn socket(family: INT, ty: INT, proto: INT) -> Result<INT, CSocketError> {
    let fd = unsafe { crate::_rsocket_rffi::socket(family, ty, proto) };
    if fd < 0 {
        return Err(last_error());
    }
    unsafe {
        crate::_rsocket_rffi::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
    }
    Ok(fd)
}

/// `dup`. The new descriptor has `FD_CLOEXEC` set.
#[cfg(unix)]
pub fn dup(fd: INT) -> Result<INT, CSocketError> {
    let n = unsafe { crate::_rsocket_rffi::dup(fd) };
    if n < 0 {
        return Err(last_error());
    }
    unsafe {
        crate::_rsocket_rffi::fcntl(n, libc::F_SETFD, libc::FD_CLOEXEC);
    }
    Ok(n)
}

/// `socketpair`. Both descriptors have `FD_CLOEXEC` set.
#[cfg(unix)]
pub fn socketpair(family: INT, ty: INT, proto: INT) -> Result<(INT, INT), CSocketError> {
    let mut fds = [0; 2];
    let res = unsafe { crate::_rsocket_rffi::socketpair(family, ty, proto, fds.as_mut_ptr()) };
    if res < 0 {
        return Err(last_error());
    }
    unsafe {
        crate::_rsocket_rffi::fcntl(fds[0], libc::F_SETFD, libc::FD_CLOEXEC);
        crate::_rsocket_rffi::fcntl(fds[1], libc::F_SETFD, libc::FD_CLOEXEC);
    }
    Ok((fds[0], fds[1]))
}

/// `getsockname` / `getpeername`. The bytes are a `sockaddr_storage`.
/// `addrlen` is the length the call wrote.
#[cfg(unix)]
fn read_socket_address(
    call: impl FnOnce(*mut libc::sockaddr, *mut libc::socklen_t) -> INT,
) -> Result<(Vec<u8>, i32), CSocketError> {
    let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut addrlen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let res = call((&raw mut storage).cast(), &raw mut addrlen);
    if res < 0 {
        return Err(last_error());
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (&raw const storage).cast::<u8>(),
            std::mem::size_of::<libc::sockaddr_storage>(),
        )
    };
    Ok((bytes.to_vec(), addrlen as i32))
}

/// `getsockname`.
#[cfg(unix)]
pub fn getsockname(fd: INT) -> Result<(Vec<u8>, i32), CSocketError> {
    read_socket_address(|addr, addrlen| unsafe {
        crate::_rsocket_rffi::socketgetsockname(fd, addr, addrlen)
    })
}

/// `getpeername`.
#[cfg(unix)]
pub fn getpeername(fd: INT) -> Result<(Vec<u8>, i32), CSocketError> {
    read_socket_address(|addr, addrlen| unsafe {
        crate::_rsocket_rffi::socketgetpeername(fd, addr, addrlen)
    })
}

/// `bind`. `addr` is the `sockaddr` bytes, and its length is `addrlen`.
#[cfg(unix)]
pub fn bind(fd: INT, addr: &[u8]) -> Result<(), CSocketError> {
    let res = unsafe {
        crate::_rsocket_rffi::socketbind(fd, addr.as_ptr().cast(), addr.len() as libc::socklen_t)
    };
    if res < 0 {
        return Err(last_error());
    }
    Ok(())
}

/// `listen`.
#[cfg(unix)]
pub fn listen(fd: INT, backlog: INT) -> Result<(), CSocketError> {
    // `RSocket.listen`: "The backlog argument must be at least 1".
    let backlog = backlog.max(1);
    let res = unsafe { crate::_rsocket_rffi::socketlisten(fd, backlog) };
    if res < 0 {
        return Err(last_error());
    }
    Ok(())
}

/// `shutdown`.
#[cfg(unix)]
pub fn shutdown(fd: INT, how: INT) -> Result<(), CSocketError> {
    let res = unsafe { crate::_rsocket_rffi::socketshutdown(fd, how) };
    if res < 0 {
        return Err(last_error());
    }
    Ok(())
}

/// `close`. A non-zero return is `CSocketError`.
#[cfg(unix)]
pub fn close(fd: INT) -> Result<(), CSocketError> {
    let res = unsafe { crate::_rsocket_rffi::socketclose(fd) };
    if res != 0 {
        return Err(last_error());
    }
    Ok(())
}

/// `accept`. The bytes are a `sockaddr_storage`. The new descriptor has
/// `FD_CLOEXEC` set. `addrlen` is the length the call wrote.
#[cfg(unix)]
pub fn accept(fd: INT) -> Result<(INT, Vec<u8>, i32), CSocketError> {
    let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut addrlen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let newfd = unsafe {
        crate::_rsocket_rffi::socketaccept(fd, (&raw mut storage).cast(), &raw mut addrlen)
    };
    if newfd < 0 {
        return Err(last_error());
    }
    unsafe {
        crate::_rsocket_rffi::fcntl(newfd, libc::F_SETFD, libc::FD_CLOEXEC);
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (&raw const storage).cast::<u8>(),
            std::mem::size_of::<libc::sockaddr_storage>(),
        )
    };
    Ok((newfd, bytes.to_vec(), addrlen as i32))
}

/// `connect`. `addr` is the `sockaddr` bytes, and its length is `addrlen`.
#[cfg(unix)]
pub fn connect(fd: INT, addr: &[u8]) -> Result<(), CSocketError> {
    let res = unsafe {
        crate::_rsocket_rffi::socketconnect(fd, addr.as_ptr().cast(), addr.len() as libc::socklen_t)
    };
    if res < 0 {
        return Err(last_error());
    }
    Ok(())
}

/// `send`. A negative return is `CSocketError`.
#[cfg(unix)]
pub fn send(fd: INT, buf: &[u8], flags: INT) -> Result<isize, CSocketError> {
    let sent = unsafe { crate::_rsocket_rffi::send(fd, buf.as_ptr().cast(), buf.len(), flags) };
    if sent < 0 {
        return Err(last_error());
    }
    Ok(sent as isize)
}

/// `recv`. A negative return is `CSocketError`. The count is the number of
/// bytes written into `buf`.
#[cfg(unix)]
pub fn recv(fd: INT, buf: &mut [u8], flags: INT) -> Result<usize, CSocketError> {
    let read =
        unsafe { crate::_rsocket_rffi::socketrecv(fd, buf.as_mut_ptr().cast(), buf.len(), flags) };
    if read < 0 {
        return Err(last_error());
    }
    Ok(read as usize)
}

/// `sendto`. `addr` is the `sockaddr` bytes, and its length is `addrlen`.
/// A negative return is `CSocketError`.
#[cfg(unix)]
pub fn sendto(fd: INT, buf: &[u8], flags: INT, addr: &[u8]) -> Result<isize, CSocketError> {
    let sent = unsafe {
        crate::_rsocket_rffi::sendto(
            fd,
            buf.as_ptr().cast(),
            buf.len(),
            flags,
            addr.as_ptr().cast(),
            addr.len() as libc::socklen_t,
        )
    };
    if sent < 0 {
        return Err(last_error());
    }
    Ok(sent as isize)
}

/// `recvfrom`. The address bytes are a `sockaddr_storage`. `addrlen` is the
/// length the call wrote. A negative return is `CSocketError`.
#[cfg(unix)]
pub fn recvfrom(
    fd: INT,
    buf: &mut [u8],
    flags: INT,
) -> Result<(usize, Vec<u8>, i32), CSocketError> {
    let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut addrlen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let read = unsafe {
        crate::_rsocket_rffi::recvfrom(
            fd,
            buf.as_mut_ptr().cast(),
            buf.len(),
            flags,
            (&raw mut storage).cast(),
            &raw mut addrlen,
        )
    };
    if read < 0 {
        return Err(last_error());
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (&raw const storage).cast::<u8>(),
            std::mem::size_of::<libc::sockaddr_storage>(),
        )
    };
    Ok((read as usize, bytes.to_vec(), addrlen as i32))
}

/// `getsockopt`. The returned bytes are the prefix the call wrote.
#[cfg(unix)]
pub fn getsockopt(
    fd: INT,
    level: INT,
    option: INT,
    maxlen: usize,
) -> Result<Vec<u8>, CSocketError> {
    let mut buf = vec![0u8; maxlen];
    let mut size = maxlen as libc::socklen_t;
    let res = unsafe {
        crate::_rsocket_rffi::socketgetsockopt(
            fd,
            level,
            option,
            buf.as_mut_ptr().cast(),
            &raw mut size,
        )
    };
    if res < 0 {
        return Err(last_error());
    }
    buf.truncate(size as usize);
    Ok(buf)
}

/// `setsockopt`. `value` is the option bytes, and its length is `optlen`.
#[cfg(unix)]
pub fn setsockopt(fd: INT, level: INT, option: INT, value: &[u8]) -> Result<(), CSocketError> {
    let res = unsafe {
        crate::_rsocket_rffi::socketsetsockopt(
            fd,
            level,
            option,
            value.as_ptr().cast(),
            value.len() as libc::socklen_t,
        )
    };
    if res < 0 {
        return Err(last_error());
    }
    Ok(())
}

/// `setsockopt_None` — a NULL option value of `optlen` bytes.
#[cfg(unix)]
pub fn setsockopt_none(fd: INT, level: INT, option: INT, optlen: u32) -> Result<(), CSocketError> {
    let res = unsafe {
        crate::_rsocket_rffi::socketsetsockopt(
            fd,
            level,
            option,
            core::ptr::null(),
            optlen as libc::socklen_t,
        )
    };
    if res < 0 {
        return Err(last_error());
    }
    Ok(())
}

/// `get_socket_family` — `sa_family` from `getsockname`.
#[majit_macros::dont_look_inside]
pub fn get_socket_family(fd: Fd) -> Result<SIGNED, CSocketError> {
    // [3.14-spec] a `sockaddr_storage` buffer <-> PyPy's `_c.sockaddr` —
    // Winsock fails `getsockname` with WSAEFAULT when the address does not
    // fit, so PyPy's `socket(fileno=fd)` on an AF_INET6 socket raises
    // OSError 10014 on Windows, where `sock_initobj` passes a `sock_addr_t`
    // and succeeds.  POSIX truncates the address and answers either way.
    #[cfg(unix)]
    type Storage = libc::sockaddr_storage;
    #[cfg(unix)]
    type Sockaddr = libc::sockaddr;
    #[cfg(windows)]
    type Storage = crate::_rsocket_rffi::sockaddr_storage;
    #[cfg(windows)]
    type Sockaddr = crate::_rsocket_rffi::sockaddr;
    let mut storage: Storage = unsafe { std::mem::zeroed() };
    #[cfg(unix)]
    let mut addrlen: libc::socklen_t = std::mem::size_of::<Storage>() as libc::socklen_t;
    #[cfg(windows)]
    let mut addrlen: INT = std::mem::size_of::<Storage>() as INT;
    let addr = (&raw mut storage).cast::<Sockaddr>();
    let res = unsafe { crate::_rsocket_rffi::socketgetsockname(fd, addr, &raw mut addrlen) };
    // The length comes back in the same slot the call was given.
    let _addrlen = addrlen;
    let result = unsafe { (*addr).sa_family } as SIGNED;
    if res < 0 {
        return Err(last_error());
    }
    Ok(result)
}

#[cfg(all(test, unix, feature = "host_env", not(feature = "sandbox")))]
fn defining_module() -> &'static str {
    module_path!()
}

#[cfg(all(test, unix, feature = "host_env", not(feature = "sandbox")))]
mod tests {
    use super::*;

    #[test]
    fn stream_socket_option_and_family() {
        unsafe {
            let fd = crate::_rsocket_rffi::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
            assert!(fd >= 0, "socket errno {}", crate::rposix::get_saved_errno());
            let ty = getsockopt_int(fd, libc::SOL_SOCKET, libc::SO_TYPE)
                .unwrap_or_else(|error| panic!("getsockopt errno {}", error.errno));
            assert_eq!(ty, libc::SOCK_STREAM as SIGNED);
            let family = get_socket_family(fd)
                .unwrap_or_else(|error| panic!("getsockname errno {}", error.errno));
            assert_eq!(family, libc::AF_INET as SIGNED);
            assert_eq!(crate::_rsocket_rffi::socketclose(fd), 0);

            let inet6 = crate::_rsocket_rffi::socket(libc::AF_INET6, libc::SOCK_STREAM, 0);
            assert!(
                inet6 >= 0,
                "inet6 errno {}",
                crate::rposix::get_saved_errno()
            );
            let family6 = get_socket_family(inet6)
                .unwrap_or_else(|error| panic!("inet6 getsockname errno {}", error.errno));
            assert_eq!(family6, libc::AF_INET6 as SIGNED);
            assert_eq!(crate::_rsocket_rffi::socketclose(inet6), 0);

            let unix = crate::_rsocket_rffi::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0);
            assert!(unix >= 0, "unix errno {}", crate::rposix::get_saved_errno());
            let family_u = get_socket_family(unix)
                .unwrap_or_else(|error| panic!("unix getsockname errno {}", error.errno));
            assert_eq!(family_u, libc::AF_UNIX as SIGNED);
            assert_eq!(crate::_rsocket_rffi::socketclose(unix), 0);
            let missing = getsockopt_int(-1, libc::SOL_SOCKET, libc::SO_TYPE).unwrap_err();
            assert_eq!(missing.errno, libc::EBADF);
        }
    }

    #[test]
    fn network_byte_order_matches_libc() {
        assert_eq!(htons(1), i64::from(libc::htons(1)));
        assert_eq!(ntohs(0x0201), i64::from(libc::ntohs(0x0201)));
        assert_eq!(htonl(0x0102_0304), i64::from(libc::htonl(0x0102_0304)));
        assert_eq!(ntohl(0x0403_0201), i64::from(libc::ntohl(0x0403_0201)));
    }

    #[test]
    fn default_timeout_blocks_until_set() {
        let saved = getdefaulttimeout();
        setdefaulttimeout(-1.0);
        assert_eq!(getdefaulttimeout(), -1.0);
        setdefaulttimeout(1.5);
        assert_eq!(getdefaulttimeout(), 1.5);
        setdefaulttimeout(-4.0);
        assert_eq!(getdefaulttimeout(), -1.0);
        setdefaulttimeout(saved);
    }

    #[test]
    fn inet_aton_and_ntoa_round_trip() {
        let packed = inet_aton(c"127.0.0.1").expect("loopback");
        assert_eq!(packed, [127, 0, 0, 1]);
        assert_eq!(inet_ntoa(&packed).expect("ntoa"), "127.0.0.1");
        let broadcast = inet_aton(c"255.255.255.255").expect("broadcast");
        assert_eq!(broadcast, [255, 255, 255, 255]);
        assert_eq!(
            inet_ntoa(&broadcast).expect("broadcast text"),
            "255.255.255.255"
        );
        assert_eq!(
            inet_aton(c"nope").expect_err("bad address").message,
            "illegal IP address string passed to inet_aton"
        );
        assert_eq!(
            inet_ntoa(&[1, 2, 3]).expect_err("short").message,
            "packed IP wrong length for inet_ntoa"
        );
    }

    #[test]
    fn gethostname_matches_libc() {
        let ours = gethostname().expect("gethostname");
        let mut buf = [0u8; 1024];
        let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) };
        assert_eq!(rc, 0);
        let end = buf.iter().position(|&byte| byte == 0).unwrap_or(buf.len());
        assert_eq!(ours, buf[..end]);
    }

    #[test]
    fn service_and_protocol_lookups() {
        assert_eq!(getprotobyname(c"tcp").expect("tcp"), 6);
        assert_eq!(getprotobyname(c"udp").expect("udp"), 17);
        assert_eq!(
            getprotobyname(c"not-a-proto")
                .expect_err("missing proto")
                .message,
            "protocol not found"
        );
        assert_eq!(getservbyname(c"http", Some(c"tcp")).expect("http"), 80);
        assert_eq!(getservbyport(80, Some(c"tcp")).expect("port 80"), "http");
        assert_eq!(
            getservbyname(c"not-a-service", Some(c"tcp"))
                .expect_err("missing service")
                .message,
            "service/proto not found"
        );
    }

    #[test]
    fn if_nameindex_copies_names() {
        let list = if_nameindex().expect("if_nameindex");
        assert!(!list.is_empty());
        assert!(
            list.iter()
                .all(|(_, name)| !name.is_empty() && !name.contains(&0))
        );
    }

    #[test]
    fn inet_pton_and_ntop_round_trip() {
        let packed = inet_pton(libc::AF_INET, c"127.0.0.1").expect("loopback");
        assert_eq!(packed, [127, 0, 0, 1]);
        assert_eq!(
            inet_ntop(libc::AF_INET, &packed).expect("ntop"),
            "127.0.0.1"
        );
        let v6 = inet_pton(libc::AF_INET6, c"::1").expect("v6");
        assert_eq!(v6.len(), 16);
        assert_eq!(*v6.last().unwrap(), 1);
        assert_eq!(inet_ntop(libc::AF_INET6, &v6).expect("v6 text"), "::1");
        assert!(matches!(
            inet_pton(libc::AF_INET, c"nope"),
            Err(PtonError::Address)
        ));
        match inet_pton(1234, c"1.2.3.4") {
            Err(PtonError::Family(code)) => assert_eq!(code, libc::EAFNOSUPPORT),
            other => panic!("family 1234 returned {other:?}"),
        }
    }

    #[test]
    fn dup_and_socketpair_set_cloexec() {
        unsafe {
            let fd = crate::_rsocket_rffi::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
            assert!(fd >= 0, "socket errno {}", crate::rposix::get_saved_errno());
            let n = dup(fd).expect("dup");
            assert_ne!(n, fd);
            let flags = crate::_rsocket_rffi::fcntl(n, libc::F_GETFD, 0);
            assert!(flags >= 0 && (flags & libc::FD_CLOEXEC) != 0);
            assert_eq!(dup(-1).expect_err("bad fd").errno, libc::EBADF);
            assert_eq!(crate::_rsocket_rffi::socketclose(fd), 0);
            assert_eq!(crate::_rsocket_rffi::socketclose(n), 0);

            let (a, b) = socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0).expect("pair");
            assert_ne!(a, b);
            for fd in [a, b] {
                let flags = crate::_rsocket_rffi::fcntl(fd, libc::F_GETFD, 0);
                assert!(flags >= 0 && (flags & libc::FD_CLOEXEC) != 0);
                assert_eq!(crate::_rsocket_rffi::socketclose(fd), 0);
            }
        }
    }

    #[test]
    fn socket_names_match_libc() {
        unsafe {
            let fd = crate::_rsocket_rffi::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
            assert!(fd >= 0, "socket errno {}", crate::rposix::get_saved_errno());
            let (bytes, len) = getsockname(fd).expect("getsockname");
            let mut storage: libc::sockaddr_storage = std::mem::zeroed();
            let mut addrlen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
            assert_eq!(
                libc::getsockname(fd, (&raw mut storage).cast(), &raw mut addrlen),
                0
            );
            assert_eq!(len, addrlen as i32);
            let libc_bytes = std::slice::from_raw_parts(
                (&raw const storage).cast::<u8>(),
                std::mem::size_of::<libc::sockaddr_storage>(),
            );
            assert_eq!(bytes, libc_bytes);
            assert_eq!(
                getpeername(fd).expect_err("unconnected").errno,
                libc::ENOTCONN
            );
            assert_eq!(crate::_rsocket_rffi::socketclose(fd), 0);

            let (a, b) = socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0).expect("pair");
            let (local, local_len) = getsockname(a).expect("local");
            let (peer, peer_len) = getpeername(b).expect("peer");
            assert_eq!(local_len, peer_len);
            assert_eq!(&local[..local_len as usize], &peer[..peer_len as usize]);
            assert_eq!(crate::_rsocket_rffi::socketclose(a), 0);
            assert_eq!(crate::_rsocket_rffi::socketclose(b), 0);
        }
    }

    #[test]
    fn bind_listen_shutdown_and_close() {
        unsafe {
            let fd = crate::_rsocket_rffi::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
            assert!(fd >= 0, "socket errno {}", crate::rposix::get_saved_errno());
            let mut addr: libc::sockaddr_in = std::mem::zeroed();
            addr.sin_family = libc::AF_INET as libc::sa_family_t;
            addr.sin_addr.s_addr = u32::from_ne_bytes([127, 0, 0, 1]);
            let bytes = std::slice::from_raw_parts(
                (&raw const addr).cast::<u8>(),
                std::mem::size_of::<libc::sockaddr_in>(),
            );
            bind(fd, bytes).expect("bind");
            listen(fd, 1).expect("listen");
            close(fd).expect("close");
            // A sibling thread can be handed the closed number. -1 is never allocated.
            assert_eq!(close(-1).expect_err("closed").errno, libc::EBADF);

            let (a, b) = socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0).expect("pair");
            shutdown(a, libc::SHUT_WR).expect("shutdown");
            close(a).expect("close a");
            close(b).expect("close b");
        }
    }

    fn loopback_listener() -> (INT, Vec<u8>) {
        unsafe {
            let fd = crate::_rsocket_rffi::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
            assert!(fd >= 0, "socket errno {}", crate::rposix::get_saved_errno());
            let mut addr: libc::sockaddr_in = std::mem::zeroed();
            addr.sin_family = libc::AF_INET as libc::sa_family_t;
            addr.sin_addr.s_addr = u32::from_ne_bytes([127, 0, 0, 1]);
            let bytes = std::slice::from_raw_parts(
                (&raw const addr).cast::<u8>(),
                std::mem::size_of::<libc::sockaddr_in>(),
            );
            bind(fd, bytes).expect("bind");
            listen(fd, 1).expect("listen");
            let (stored, nlen) = getsockname(fd).expect("getsockname");
            assert!(nlen > 0);
            (fd, stored[..nlen as usize].to_vec())
        }
    }

    #[test]
    fn accept_sets_cloexec_and_connects() {
        let (listener, addr) = loopback_listener();
        let connect_addr = addr.clone();
        let peer = std::thread::spawn(move || unsafe {
            let client = libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
            assert!(client >= 0, "client socket");
            let rc = libc::connect(
                client,
                connect_addr.as_ptr().cast(),
                connect_addr.len() as libc::socklen_t,
            );
            assert_eq!(rc, 0, "peer connect {}", std::io::Error::last_os_error());
            assert_eq!(libc::close(client), 0);
        });
        unsafe {
            let mut pfd = libc::pollfd {
                fd: listener,
                events: libc::POLLIN,
                revents: 0,
            };
            assert!(
                libc::poll(&raw mut pfd, 1, 2000) > 0,
                "listener not readable"
            );
            let (newfd, bytes, addrlen) = accept(listener).expect("accept");
            assert!(newfd >= 0);
            assert!(addrlen > 0);
            assert_eq!(bytes.len(), std::mem::size_of::<libc::sockaddr_storage>());
            let flags = crate::_rsocket_rffi::fcntl(newfd, libc::F_GETFD, 0);
            assert!(flags >= 0 && (flags & libc::FD_CLOEXEC) != 0);
            assert_eq!(accept(-1).expect_err("bad fd").errno, libc::EBADF);
            close(newfd).expect("close accepted");
            close(listener).expect("close listener");
        }
        peer.join().expect("peer");

        let (listener, addr) = loopback_listener();
        let accepted = std::thread::spawn(move || unsafe {
            let mut pfd = libc::pollfd {
                fd: listener,
                events: libc::POLLIN,
                revents: 0,
            };
            assert!(
                libc::poll(&raw mut pfd, 1, 2000) > 0,
                "connect listener not readable"
            );
            let mut storage: libc::sockaddr_storage = std::mem::zeroed();
            let mut addrlen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
            let cfd = libc::accept(listener, (&raw mut storage).cast(), &raw mut addrlen);
            assert!(cfd >= 0, "libc accept");
            assert_eq!(libc::close(cfd), 0);
            assert_eq!(libc::close(listener), 0);
        });
        unsafe {
            let client = crate::_rsocket_rffi::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
            assert!(
                client >= 0,
                "client errno {}",
                crate::rposix::get_saved_errno()
            );
            connect(client, &addr).expect("connect");
            close(client).expect("close client");
            assert_eq!(connect(-1, &addr).expect_err("bad fd").errno, libc::EBADF);
        }
        accepted.join().expect("accepted");
    }

    #[test]
    fn send_and_recv_round_trip() {
        let (a, b) = socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0).expect("pair");
        assert_eq!(send(a, b"hello", 0).expect("send"), 5);
        let mut buf = [0u8; 8];
        assert_eq!(recv(b, &mut buf[..2], 0).expect("partial"), 2);
        assert_eq!(&buf[..2], b"he");
        assert_eq!(recv(b, &mut buf, 0).expect("rest"), 3);
        assert_eq!(&buf[..3], b"llo");
        let mut bad = [0u8; 1];
        assert_eq!(send(-1, b"x", 0).expect_err("send").errno, libc::EBADF);
        assert_eq!(recv(-1, &mut bad, 0).expect_err("recv").errno, libc::EBADF);
        close(a).expect("close a");
        close(b).expect("close b");
    }

    fn udp_bound() -> (INT, Vec<u8>) {
        unsafe {
            let fd = crate::_rsocket_rffi::socket(libc::AF_INET, libc::SOCK_DGRAM, 0);
            assert!(fd >= 0, "socket errno {}", crate::rposix::get_saved_errno());
            let mut addr: libc::sockaddr_in = std::mem::zeroed();
            addr.sin_family = libc::AF_INET as libc::sa_family_t;
            addr.sin_addr.s_addr = u32::from_ne_bytes([127, 0, 0, 1]);
            let bytes = std::slice::from_raw_parts(
                (&raw const addr).cast::<u8>(),
                std::mem::size_of::<libc::sockaddr_in>(),
            );
            bind(fd, bytes).expect("bind");
            let (stored, nlen) = getsockname(fd).expect("getsockname");
            assert!(nlen > 0);
            (fd, stored[..nlen as usize].to_vec())
        }
    }

    #[test]
    fn sendto_and_recvfrom_round_trip() {
        let (a, a_addr) = udp_bound();
        let (b, b_addr) = udp_bound();
        unsafe {
            let tv = libc::timeval {
                tv_sec: 2,
                tv_usec: 0,
            };
            assert_eq!(
                libc::setsockopt(
                    b,
                    libc::SOL_SOCKET,
                    libc::SO_RCVTIMEO,
                    (&raw const tv).cast(),
                    std::mem::size_of::<libc::timeval>() as libc::socklen_t,
                ),
                0
            );
        }
        assert_eq!(sendto(a, b"ping", 0, &b_addr).expect("sendto"), 4);
        let mut buf = [0u8; 8];
        let (n, from, from_len) = recvfrom(b, &mut buf, 0).expect("recvfrom");
        assert_eq!(n, 4);
        assert_eq!(&buf[..4], b"ping");
        assert_eq!(from.len(), std::mem::size_of::<libc::sockaddr_storage>());
        assert_eq!(from_len as usize, a_addr.len());
        assert_eq!(&from[..a_addr.len()], a_addr.as_slice());
        assert_eq!(
            sendto(-1, b"x", 0, &b_addr).expect_err("sendto").errno,
            libc::EBADF
        );
        assert_eq!(
            recvfrom(-1, &mut buf, 0).expect_err("recvfrom").errno,
            libc::EBADF
        );
        close(a).expect("close a");
        close(b).expect("close b");
    }

    #[test]
    fn setsockopt_and_getsockopt_round_trip() {
        unsafe {
            let fd = crate::_rsocket_rffi::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
            assert!(fd >= 0, "socket errno {}", crate::rposix::get_saved_errno());
            let on = 1i32.to_ne_bytes();
            setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEADDR, &on).expect("setsockopt");
            let reuse = getsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEADDR, 4).expect("reuse");
            assert_eq!(reuse.len(), 4);
            assert_ne!(i32::from_ne_bytes(reuse.try_into().unwrap()), 0);
            let ty = getsockopt(fd, libc::SOL_SOCKET, libc::SO_TYPE, 4).expect("type");
            assert_eq!(
                i32::from_ne_bytes(ty.try_into().unwrap()),
                libc::SOCK_STREAM
            );
            assert_eq!(
                setsockopt(-1, libc::SOL_SOCKET, libc::SO_REUSEADDR, &on)
                    .expect_err("setsockopt")
                    .errno,
                libc::EBADF
            );
            assert_eq!(
                getsockopt(-1, libc::SOL_SOCKET, libc::SO_TYPE, 4)
                    .expect_err("getsockopt")
                    .errno,
                libc::EBADF
            );
            close(fd).expect("close");
        }
    }

    #[test]
    fn defining_module_is_the_policy_key() {
        assert_eq!(defining_module(), "majit_rlib::rsocket");
    }

    #[test]
    fn wait_for_data_returns_when_the_poll_is_ready() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        let mut polls = 0;
        let ready = wait_for_data(
            0,
            false,
            deadline,
            |_, _, timeout_ms| {
                polls += 1;
                assert!(timeout_ms >= 1);
                PollWait::Ready
            },
            || -> Result<(), ()> { panic!("ready poll does not check signals") },
        )
        .expect("ready");
        assert!(ready);
        assert_eq!(polls, 1);
    }

    #[test]
    fn wait_for_data_times_out_when_the_deadline_has_passed() {
        let deadline = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let err = wait_for_data(
            0,
            true,
            deadline,
            |_, _, _| panic!("expired deadline does not poll"),
            || Ok::<(), ()>(()),
        );
        assert!(matches!(err, Err(WaitError::Timeout)));
    }

    #[test]
    fn wait_for_data_retries_an_interruption_then_surfaces_the_os_error() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        let mut polls = 0;
        let err = wait_for_data(
            0,
            false,
            deadline,
            |_, _, _| {
                polls += 1;
                if polls == 1 {
                    PollWait::Interrupted
                } else {
                    PollWait::Failed(libc::EBADF)
                }
            },
            || Ok::<(), ()>(()),
        );
        match err {
            Err(WaitError::Os(error)) => assert_eq!(error.errno, libc::EBADF),
            other => panic!("expected EBADF, got {other:?}"),
        }
        assert_eq!(polls, 2);
    }

    #[test]
    fn wait_for_data_returns_the_signal_check_error() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        let err = wait_for_data(
            0,
            false,
            deadline,
            |_, _, _| PollWait::TimedOut,
            || Err("stopped"),
        );
        assert!(matches!(err, Err(WaitError::Check("stopped"))));
    }

    #[test]
    fn getaddrinfo_numeric_and_getnameinfo() {
        let list = getaddrinfo(
            Some(c"127.0.0.1"),
            Some(c"80"),
            libc::AF_INET,
            libc::SOCK_STREAM,
            0,
            libc::AI_NUMERICHOST,
        )
        .expect("numeric getaddrinfo");
        assert!(!list.is_empty());
        assert_eq!(list[0].family, i64::from(libc::AF_INET));
        assert_eq!(list[0].socktype, i64::from(libc::SOCK_STREAM));
        assert_eq!(list[0].addr.len(), std::mem::size_of::<libc::sockaddr_in>());
        let (host, serv) = getnameinfo(&list[0].addr, libc::NI_NUMERICHOST | libc::NI_NUMERICSERV)
            .expect("getnameinfo");
        assert_eq!(host, "127.0.0.1");
        assert_eq!(serv, "80");

        let err = getaddrinfo(
            Some(c"not-numeric"),
            None,
            libc::AF_INET,
            0,
            0,
            libc::AI_NUMERICHOST,
        )
        .expect_err("numeric host");
        assert_eq!(err.errno, libc::EAI_NONAME);
        let err = getaddrinfo(
            Some(c"127.0.0.1"),
            Some(c"http"),
            libc::AF_INET,
            0,
            0,
            libc::AI_NUMERICHOST | libc::AI_NUMERICSERV,
        )
        .expect_err("numeric service");
        assert_eq!(err.errno, libc::EAI_NONAME);

        // Absent port and `"0"` with `AI_NUMERICSERV` stay inside getaddrinfo.
        let _ = getaddrinfo(
            Some(c"127.0.0.1"),
            None,
            0,
            0,
            0,
            libc::AI_NUMERICHOST | libc::AI_NUMERICSERV,
        );
        let _ = getaddrinfo(
            Some(c"127.0.0.1"),
            Some(c"0"),
            0,
            0,
            0,
            libc::AI_NUMERICHOST | libc::AI_NUMERICSERV,
        );
    }

    #[test]
    fn socket_sets_cloexec() {
        let fd = socket(libc::AF_INET, libc::SOCK_STREAM, 0).expect("socket");
        let flags = unsafe { crate::_rsocket_rffi::fcntl(fd, libc::F_GETFD, 0) };
        assert!(flags >= 0 && (flags & libc::FD_CLOEXEC) != 0);
        assert_eq!(
            socket(-1, libc::SOCK_STREAM, 0).expect_err("family").errno,
            libc::EAFNOSUPPORT
        );
        close(fd).expect("close");
    }
}
