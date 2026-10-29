//! `rpython/rlib/_rsocket_rffi.py` — socket C calls.
//!
//! POSIX covers `select` / `poll` / `FD_*` and the `external` socket calls
//! (`socket`, `connect`, `send`, `recv`, the resolvers, `fcntl`, `dup`,
//! `socketpair`, `if_nameindex`). `recvmsg_implementation`,
//! `sendmsg_implementation`, `CMSG_SPACE_wrapper` and `CMSG_LEN_wrapper`
//! stay out: each one is a separate C source. Windows here is `FD_*`,
//! `select`, `getsockname`, `getsockopt`, the byte-order conversions,
//! `inet_addr`, `inet_ntoa`, `inet_pton`, `inet_ntop`, and `_WSAGetLastError`.

#![allow(non_snake_case, non_camel_case_types)]

#[cfg(unix)]
mod posix {
    use crate::rffi::{INT, RFFI_SAVE_ERRNO, UINT, USHORT};

    // `_rsocket_rffi.eci` includes. Linux-only headers (`netpacket/packet.h`,
    // `linux/netlink.h`) stay out with the packet-socket slice.
    crate::rffi::external_compilation_info! {
        const ECI = {
            includes: [
                "sys/types.h",
                "sys/socket.h",
                "sys/un.h",
                "poll.h",
                "sys/select.h",
                "sys/time.h",
                "netinet/in.h",
                "netinet/tcp.h",
                "unistd.h",
                "fcntl.h",
                "stdio.h",
                "netdb.h",
                "arpa/inet.h",
                "stdint.h",
                "errno.h",
                "limits.h",
                "net/if.h",
            ],
        };
    }

    /// `fd_set` (`COpaquePtr`). Callers pass a pointer at the struct.
    pub type fd_set = *mut libc::fd_set;

    pub const FD_SETSIZE: usize = libc::FD_SETSIZE;
    /// `MAX_FD_SIZE`. POSIX `select` indexes a bitmap by descriptor.
    pub const MAX_FD_SIZE: Option<i32> = Some(FD_SETSIZE as i32);

    /// `_rsocket_rffi.geterrno` on POSIX (`rposix.get_saved_errno`).
    pub fn geterrno() -> i32 {
        crate::rposix::get_saved_errno()
    }

    // libc private build-script cfgs (`gnu_time_bits64`, `gnu_file_offset_bits64`,
    // `musl_redir_time64`) select 32-bit redirects (`__select64`, `__fcntl_time64`,
    // `__getsockopt64`, `__setsockopt64`). Native targets are 64-bit, so those
    // `link_name`s are not copied.
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86_64"),
            link_name = "select$1050"
        )]
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "select$UNIX2003"
        )]
        #[cfg_attr(target_os = "netbsd", link_name = "__select50")]
        #[cfg_attr(target_os = "aix", link_name = "__fd_select")]
        pub select = "select",
        [INT, fd_set, fd_set, fd_set, *mut libc::timeval],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "poll$UNIX2003"
        )]
        pub poll = "poll",
        [*mut libc::pollfd, libc::nfds_t, INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );

    // `external_c` `FD_CLR` / `FD_ISSET` / `FD_SET` / `FD_ZERO` is `macro=True`.
    // `libc`'s `FD_ISSET` returns `bool`; upstream's result is `rffi.INT`.
    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    unsafe fn fd_isset(fd: INT, set: fd_set) -> INT {
        unsafe { libc::FD_ISSET(fd, set) as INT }
    }

    crate::rffi::llexternal!(
        pub FD_CLR = "FD_CLR",
        [INT, fd_set],
        (),
        compilation_info = ECI,
        calling_conv = "c",
        macro = libc::FD_CLR
    );
    crate::rffi::llexternal!(
        pub FD_ISSET = "FD_ISSET",
        [INT, fd_set],
        INT,
        compilation_info = ECI,
        calling_conv = "c",
        macro = fd_isset
    );
    crate::rffi::llexternal!(
        pub FD_SET = "FD_SET",
        [INT, fd_set],
        (),
        compilation_info = ECI,
        calling_conv = "c",
        macro = libc::FD_SET
    );
    crate::rffi::llexternal!(
        pub FD_ZERO = "FD_ZERO",
        [fd_set],
        (),
        compilation_info = ECI,
        calling_conv = "c",
        macro = libc::FD_ZERO
    );

    // `socketclose_no_errno`: `close`, `releasegil=False`, no `save_err`.
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "close$NOCANCEL$UNIX2003"
        )]
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86_64"),
            link_name = "close$NOCANCEL"
        )]
        pub socketclose_no_errno = "close",
        [INT],
        INT,
        compilation_info = ECI,
        releasegil = false
    );

    crate::rffi::llexternal!(
        pub dup = "dup",
        [INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(target_os = "netbsd", link_name = "__socket30")]
        #[cfg_attr(target_os = "illumos", link_name = "__xnet_socket")]
        #[cfg_attr(target_os = "solaris", link_name = "__xnet7_socket")]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_socket")]
        pub socket = "socket",
        [INT, INT, INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    // `socketclose`: `close`, `releasegil=False`, `save_err`. Distinct from
    // `socketclose_no_errno`, which does not record errno.
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "close$NOCANCEL$UNIX2003"
        )]
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86_64"),
            link_name = "close$NOCANCEL"
        )]
        pub socketclose = "close",
        [INT],
        INT,
        compilation_info = ECI,
        releasegil = false,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "connect$UNIX2003"
        )]
        #[cfg_attr(
            any(target_os = "illumos", target_os = "solaris"),
            link_name = "__xnet_connect"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_connect")]
        pub socketconnect = "connect",
        [INT, *const libc::sockaddr, libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(target_os = "espidf", link_name = "lwip_bind")]
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "bind$UNIX2003"
        )]
        #[cfg_attr(
            any(target_os = "solaris", target_os = "illumos"),
            link_name = "__xnet_bind"
        )]
        pub socketbind = "bind",
        [INT, *const libc::sockaddr, libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "listen$UNIX2003"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_listen")]
        pub socketlisten = "listen",
        [INT, INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "accept$UNIX2003"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_accept")]
        #[cfg_attr(target_os = "aix", link_name = "naccept")]
        pub socketaccept = "accept",
        [INT, *mut libc::sockaddr, *mut libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "getpeername$UNIX2003"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_getpeername")]
        #[cfg_attr(target_os = "aix", link_name = "ngetpeername")]
        pub socketgetpeername = "getpeername",
        [INT, *mut libc::sockaddr, *mut libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "getsockname$UNIX2003"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_getsockname")]
        #[cfg_attr(target_os = "aix", link_name = "ngetsockname")]
        pub socketgetsockname = "getsockname",
        [INT, *mut libc::sockaddr, *mut libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            any(target_os = "illumos", target_os = "solaris"),
            link_name = "__xnet_getsockopt"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_getsockopt")]
        pub socketgetsockopt = "getsockopt",
        [INT, INT, INT, *mut libc::c_void, *mut libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    // `htons` / `ntohs` / `htonl` / `ntohl` (`releasegil=False`). Darwin and
    // OpenBSD publish these as macros; the libc crate exposes the same
    // functions on every unix target.
    crate::rffi::llexternal!(
        pub htons = "htons",
        [USHORT],
        USHORT,
        compilation_info = ECI,
        releasegil = false,
        macro = libc::htons
    );
    crate::rffi::llexternal!(
        pub ntohs = "ntohs",
        [USHORT],
        USHORT,
        compilation_info = ECI,
        releasegil = false,
        macro = libc::ntohs
    );
    crate::rffi::llexternal!(
        pub htonl = "htonl",
        [UINT],
        UINT,
        compilation_info = ECI,
        releasegil = false,
        macro = libc::htonl
    );
    crate::rffi::llexternal!(
        pub ntohl = "ntohl",
        [UINT],
        UINT,
        compilation_info = ECI,
        releasegil = false,
        macro = libc::ntohl
    );
    crate::rffi::llexternal!(
        #[cfg_attr(target_os = "espidf", link_name = "lwip_setsockopt")]
        pub socketsetsockopt = "setsockopt",
        [INT, INT, INT, *const libc::c_void, libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "socketpair$UNIX2003"
        )]
        #[cfg_attr(
            any(target_os = "illumos", target_os = "solaris"),
            link_name = "__xnet_socketpair"
        )]
        pub socketpair = "socketpair",
        [INT, INT, INT, *mut INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "send$UNIX2003"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_send")]
        pub send = "send",
        [INT, *const libc::c_void, libc::size_t, INT],
        libc::ssize_t,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "recv$UNIX2003"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_recv")]
        pub socketrecv = "recv",
        [INT, *mut libc::c_void, libc::size_t, INT],
        libc::ssize_t,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "sendto$UNIX2003"
        )]
        #[cfg_attr(
            any(target_os = "illumos", target_os = "solaris"),
            link_name = "__xnet_sendto"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_sendto")]
        pub sendto = "sendto",
        [
            INT,
            *const libc::c_void,
            libc::size_t,
            INT,
            *const libc::sockaddr,
            libc::socklen_t
        ],
        libc::ssize_t,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(target_os = "espidf", link_name = "lwip_recvfrom")]
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "recvfrom$UNIX2003"
        )]
        #[cfg_attr(target_os = "aix", link_name = "nrecvfrom")]
        pub recvfrom = "recvfrom",
        [
            INT,
            *mut libc::c_void,
            libc::size_t,
            INT,
            *mut libc::sockaddr,
            *mut libc::socklen_t
        ],
        libc::ssize_t,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(target_os = "espidf", link_name = "lwip_shutdown")]
        pub socketshutdown = "shutdown",
        [INT, INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        pub gethostname = "gethostname",
        [*mut libc::c_char, libc::size_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    // `fcntl` is variadic. `_rsocket_rffi.fcntl` uses `natural_arity = 2`.
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "fcntl$UNIX2003"
        )]
        pub fcntl = "fcntl",
        [INT, INT, INT],
        INT,
        compilation_info = ECI,
        natural_arity = 2,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        pub gai_strerror = "gai_strerror",
        [INT],
        *const libc::c_char,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            any(target_os = "illumos", target_os = "solaris"),
            link_name = "__xnet_getaddrinfo"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_getaddrinfo")]
        pub getaddrinfo = "getaddrinfo",
        [
            *const libc::c_char,
            *const libc::c_char,
            *const libc::addrinfo,
            *mut *mut libc::addrinfo
        ],
        INT,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        #[cfg_attr(target_os = "espidf", link_name = "lwip_freeaddrinfo")]
        pub freeaddrinfo = "freeaddrinfo",
        [*mut libc::addrinfo],
        (),
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub getnameinfo = "getnameinfo",
        [
            *const libc::sockaddr,
            libc::socklen_t,
            *mut libc::c_char,
            libc::socklen_t,
            *mut libc::c_char,
            libc::socklen_t,
            INT
        ],
        INT,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub getservbyname = "getservbyname",
        [*const libc::c_char, *const libc::c_char],
        *mut libc::servent,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub getservbyport = "getservbyport",
        [INT, *const libc::c_char],
        *mut libc::servent,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub getprotobyname = "getprotobyname",
        [*const libc::c_char],
        *mut libc::protoent,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub inet_aton = "inet_aton",
        [*const libc::c_char, *mut libc::in_addr],
        INT,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub inet_ntoa = "inet_ntoa",
        [libc::in_addr],
        *mut libc::c_char,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub inet_pton = "inet_pton",
        [INT, *const libc::c_char, *mut libc::c_void],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        pub inet_ntop = "inet_ntop",
        [INT, *const libc::c_void, *mut libc::c_char, libc::socklen_t],
        *const libc::c_char,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    // `gethostbyaddr`'s length is `socklen_t`, matching the call pyre already
    // makes. The resolver records failure in `h_errno`, not `errno`.
    crate::rffi::llexternal!(
        pub gethostbyname = "gethostbyname",
        [*const libc::c_char],
        *mut libc::c_void,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub gethostbyaddr = "gethostbyaddr",
        [*const libc::c_void, libc::socklen_t, INT],
        *mut libc::c_void,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub if_nameindex = "if_nameindex",
        [],
        *mut libc::if_nameindex,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        pub if_freenameindex = "if_freenameindex",
        [*mut libc::if_nameindex],
        (),
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );

    // `libc::sethostname`'s length is `c_int` on Apple, FreeBSD, DragonFly,
    // AIX and Solaris, and `size_t` on Linux and the other BSDs.
    #[cfg(any(
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "aix",
        target_os = "solaris",
        target_os = "illumos",
    ))]
    crate::rffi::llexternal!(
        pub(super) c_sethostname = "sethostname",
        [*const libc::c_char, libc::c_int],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    #[cfg(not(any(
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "aix",
        target_os = "solaris",
        target_os = "illumos",
    )))]
    crate::rffi::llexternal!(
        pub(super) c_sethostname = "sethostname",
        [*const libc::c_char, libc::size_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );

    /// `sethostname`. `len` is the byte count, narrowed to `c_int` where
    /// `libc::sethostname` takes one.
    pub unsafe fn sethostname(name: *const libc::c_char, len: usize) -> libc::c_int {
        unsafe {
            #[cfg(any(
                target_vendor = "apple",
                target_os = "freebsd",
                target_os = "dragonfly",
                target_os = "aix",
                target_os = "solaris",
                target_os = "illumos",
            ))]
            {
                c_sethostname(name, len as libc::c_int)
            }
            #[cfg(not(any(
                target_vendor = "apple",
                target_os = "freebsd",
                target_os = "dragonfly",
                target_os = "aix",
                target_os = "solaris",
                target_os = "illumos",
            )))]
            {
                c_sethostname(name, len)
            }
        }
    }

    #[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
    mod tests {
        use super::*;

        #[test]
        fn stream_socket_dup_fcntl_and_close() {
            unsafe {
                let fd = socket(libc::AF_INET, libc::SOCK_STREAM, 0);
                assert!(fd >= 0, "socket errno {}", crate::rposix::get_saved_errno());
                let flags = fcntl(fd, libc::F_GETFD, 0);
                assert!(
                    flags >= 0,
                    "fcntl errno {}",
                    crate::rposix::get_saved_errno()
                );
                let duped = dup(fd);
                assert!(duped >= 0, "dup errno {}", crate::rposix::get_saved_errno());
                assert_eq!(socketclose(duped), 0);
                assert_eq!(socketclose(fd), 0);
            }
        }

        #[test]
        fn inet_pton_loopback() {
            let text = c"127.0.0.1";
            let mut buf = [0u8; 4];
            let rc = unsafe { inet_pton(libc::AF_INET, text.as_ptr(), buf.as_mut_ptr().cast()) };
            assert_eq!(rc, 1);
            assert_eq!(buf, [127, 0, 0, 1]);
        }
    }
}

#[cfg(unix)]
pub use posix::*;

#[cfg(windows)]
mod winsock {
    use crate::rffi::{INT, RFFI_SAVE_WSALASTERROR, UINT, USHORT};
    use crate::rwin32::{DWORD, LPDWORD, WORD};
    use std::ffi::{c_char, c_long, c_ulong, c_void};

    /// `socketfd_type`: `lltype.Unsigned` under `_MSVC` (`SOCKET`).
    pub type socketfd_type = usize;
    /// `socklen_t`. `ws2tcpip.h` typedefs it as `int`.
    pub type socklen_t = INT;

    /// `INVALID_SOCKET` (`(SOCKET)(~0)`).
    pub const INVALID_SOCKET: socketfd_type = !0;
    /// `SOCKET_ERROR`.
    pub const SOCKET_ERROR: INT = -1;

    /// `_rsocket_rffi.invalid_socket` under `_MSVC`.
    pub fn invalid_socket(fd: socketfd_type) -> bool {
        fd == INVALID_SOCKET
    }

    /// WinSock `fd_set` capacity. `_rsocket_rffi.py CConfig` measures the
    /// SDK default of 64, which silently drops later sockets in PyPy.
    /// Windows `Modules/selectmodule.c` at v3.15.0 defines FD_SETSIZE as 512
    /// before including WinSock (also in the free-threaded build). Preserve
    /// the count/array representation, with the capacity its public select
    /// contract requires.
    pub const FD_SETSIZE: usize = 512;
    /// `MAX_FD_SIZE` is `None` on Windows: the set is a count, not a bitmap.
    pub const MAX_FD_SIZE: Option<i32> = None;

    /// `struct fd_set` (`fd_count` plus `fd_array[FD_SETSIZE]`).
    #[repr(C)]
    pub struct fd_set {
        pub fd_count: u32,
        pub fd_array: [usize; FD_SETSIZE],
    }

    /// `struct timeval` (`tv_sec` / `tv_usec` are `long`).
    #[repr(C)]
    pub struct timeval {
        pub tv_sec: std::ffi::c_long,
        pub tv_usec: std::ffi::c_long,
    }

    /// `struct sockaddr` (`sockaddr`). `sa_family` is `ADDRESS_FAMILY`.
    #[repr(C)]
    pub struct sockaddr {
        pub sa_family: u16,
        pub sa_data: [i8; 14],
    }

    /// `struct sockaddr_storage` (`SOCKADDR_STORAGE`): 128 bytes, 8-aligned.
    #[repr(C)]
    pub struct sockaddr_storage {
        pub ss_family: u16,
        pub __ss_pad1: [u8; 6],
        pub __ss_align: i64,
        pub __ss_pad2: [u8; 112],
    }

    /// `struct in6_addr` (`s6_addr`).
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct in6_addr {
        pub s6_addr: [u8; 16],
    }

    /// `struct sockaddr_in`. `sin_addr` is the `S_un` union, read as its
    /// `S_addr` member.
    #[repr(C)]
    pub struct sockaddr_in {
        pub sin_family: u16,
        pub sin_port: u16,
        pub sin_addr: in_addr,
        pub sin_zero: [c_char; 8],
    }

    /// `struct sockaddr_in6`. `sin6_scope_id` is the union's `ULONG` member.
    #[repr(C)]
    pub struct sockaddr_in6 {
        pub sin6_family: u16,
        pub sin6_port: u16,
        pub sin6_flowinfo: u32,
        pub sin6_addr: in6_addr,
        pub sin6_scope_id: u32,
    }

    /// `struct addrinfo` (`ADDRINFOA`). WinSock orders `ai_canonname` before
    /// `ai_addr` and counts `ai_addrlen` in a `size_t`.
    #[repr(C)]
    pub struct addrinfo {
        pub ai_flags: INT,
        pub ai_family: INT,
        pub ai_socktype: INT,
        pub ai_protocol: INT,
        pub ai_addrlen: usize,
        pub ai_canonname: *mut c_char,
        pub ai_addr: *mut sockaddr,
        pub ai_next: *mut addrinfo,
    }

    /// `struct hostent`. `h_addrtype` and `h_length` are `short`.
    #[repr(C)]
    pub struct hostent {
        pub h_name: *mut c_char,
        pub h_aliases: *mut *mut c_char,
        pub h_addrtype: i16,
        pub h_length: i16,
        pub h_addr_list: *mut *mut c_char,
    }

    /// `struct servent`. Win64 orders `s_proto` before `s_port`.
    #[repr(C)]
    pub struct servent {
        pub s_name: *mut c_char,
        pub s_aliases: *mut *mut c_char,
        #[cfg(target_pointer_width = "64")]
        pub s_proto: *mut c_char,
        pub s_port: i16,
        #[cfg(not(target_pointer_width = "64"))]
        pub s_proto: *mut c_char,
    }

    /// `struct protoent`.
    #[repr(C)]
    pub struct protoent {
        pub p_name: *mut c_char,
        pub p_aliases: *mut *mut c_char,
        pub p_proto: i16,
    }

    /// `WSADESCRIPTION_LEN + 1` / `WSASYS_STATUS_LEN + 1`.
    const WSADESCRIPTION_LEN_1: usize = 257;
    const WSASYS_STATUS_LEN_1: usize = 129;

    /// `struct WSAData`. Win64 moves the two counters and `lpVendorInfo`
    /// ahead of the strings.
    #[repr(C)]
    pub struct WSAData {
        pub wVersion: WORD,
        pub wHighVersion: WORD,
        #[cfg(target_pointer_width = "64")]
        pub iMaxSockets: USHORT,
        #[cfg(target_pointer_width = "64")]
        pub iMaxUdpDg: USHORT,
        #[cfg(target_pointer_width = "64")]
        pub lpVendorInfo: *mut c_char,
        pub szDescription: [c_char; WSADESCRIPTION_LEN_1],
        pub szSystemStatus: [c_char; WSASYS_STATUS_LEN_1],
        #[cfg(not(target_pointer_width = "64"))]
        pub iMaxSockets: USHORT,
        #[cfg(not(target_pointer_width = "64"))]
        pub iMaxUdpDg: USHORT,
        #[cfg(not(target_pointer_width = "64"))]
        pub lpVendorInfo: *mut c_char,
    }

    // `constant_names` / `constants_w_defaults`, at their `winsock2.h` /
    // `ws2def.h` / `ws2tcpip.h` values.
    pub const AF_UNSPEC: INT = 0;
    pub const SOCK_STREAM: INT = 1;
    pub const SOCK_DGRAM: INT = 2;
    pub const SOL_SOCKET: INT = 0xffff;
    pub const SO_ERROR: INT = 0x1007;
    pub const SO_TYPE: INT = 0x1008;
    pub const INADDR_ANY: u32 = 0;
    pub const INADDR_BROADCAST: u32 = 0xffff_ffff;
    pub const AI_PASSIVE: INT = 0x01;
    pub const AI_NUMERICHOST: INT = 0x04;
    /// `cConfig.NI_MAXHOST`.
    pub const NI_MAXHOST: INT = 1025;
    /// `cConfig.FIONBIO` (`_IOW('f', 126, u_long)`).
    pub const FIONBIO: c_long = 0x8004_667E_u32 as c_long;
    /// `SIO_TCP_SET_ACK_FREQUENCY` (`_WSAIOW(IOC_VENDOR, 23)`, `mstcpip.h`).
    pub const SIO_TCP_SET_ACK_FREQUENCY: DWORD = 0x9800_0017;
    /// `cConfig.WSAEINTR` / `WSAEWOULDBLOCK` / `WSAENOTSOCK`.
    pub const WSAEINTR: INT = 10004;
    pub const WSAEWOULDBLOCK: INT = 10035;
    pub const WSAENOTSOCK: INT = 10038;

    pub type fd_set_p = *mut fd_set;

    crate::rffi::external_compilation_info! {
        const ECI = {
            includes: ["winsock2.h", "ws2tcpip.h"],
            libraries: ["ws2_32"],
        };
    }

    crate::rffi::llexternal!(
        pub _WSAGetLastError = "WSAGetLastError",
        [],
        crate::rwin32::DWORD,
        compilation_info = ECI,
        calling_conv = "win",
        _nowrapper = true,
        sandboxsafe = true
    );

    /// `_rsocket_rffi.geterrno` on Windows (`rwin32.GetLastError_saved`).
    /// `select` saves `WSAGetLastError` into that slot.
    pub fn geterrno() -> i32 {
        crate::rwin32::GetLastError_saved() as i32
    }

    // `winsock2.h` spells `FD_ISSET` as this call.
    crate::rffi::llexternal!(
        __WSAFDIsSet = "__WSAFDIsSet",
        [usize, fd_set_p],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        _nowrapper = true
    );

    // `external_c` `FD_ZERO` / `FD_SET` / `FD_ISSET` / `FD_CLR` is `macro=True`.
    // The bodies match
    // the WinSock headers: scan `fd_array`, append under `FD_SETSIZE`.
    // `FD_*` take `rffi.INT` on Windows too (`external_c`). The WinSock
    // array stores `SOCKET` (`uintptr`), so the body sign-extends that int.
    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    unsafe fn fd_zero(set: fd_set_p) {
        unsafe { (*set).fd_count = 0 };
    }
    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    unsafe fn fd_set_insert(fd: INT, set: fd_set_p) {
        let fd = fd as usize;
        unsafe {
            let count = (*set).fd_count as usize;
            for slot in 0..count {
                if (*set).fd_array[slot] == fd {
                    return;
                }
            }
            if count < FD_SETSIZE {
                (*set).fd_array[count] = fd;
                (*set).fd_count += 1;
            }
        }
    }
    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    unsafe fn fd_isset(fd: INT, set: fd_set_p) -> INT {
        unsafe { __WSAFDIsSet(fd as usize, set) }
    }
    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    unsafe fn fd_clr(fd: INT, set: fd_set_p) {
        let fd = fd as usize;
        unsafe {
            let count = (*set).fd_count as usize;
            let mut kept = 0usize;
            for slot in 0..count {
                let cur = (*set).fd_array[slot];
                if cur != fd {
                    (*set).fd_array[kept] = cur;
                    kept += 1;
                }
            }
            (*set).fd_count = kept as u32;
        }
    }

    crate::rffi::llexternal!(
        pub FD_ZERO = "FD_ZERO",
        [fd_set_p],
        (),
        compilation_info = ECI,
        calling_conv = "c",
        macro = fd_zero
    );
    crate::rffi::llexternal!(
        pub FD_SET = "FD_SET",
        [INT, fd_set_p],
        (),
        compilation_info = ECI,
        calling_conv = "c",
        macro = fd_set_insert
    );
    crate::rffi::llexternal!(
        pub FD_ISSET = "FD_ISSET",
        [INT, fd_set_p],
        INT,
        compilation_info = ECI,
        calling_conv = "c",
        macro = fd_isset
    );
    crate::rffi::llexternal!(
        pub FD_CLR = "FD_CLR",
        [INT, fd_set_p],
        (),
        compilation_info = ECI,
        calling_conv = "c",
        macro = fd_clr
    );

    // `external('select', ..., save_err=RFFI_SAVE_WSALASTERROR)` with
    // `calling_conv='win'`.
    crate::rffi::llexternal!(
        pub select = "select",
        [INT, fd_set_p, fd_set_p, fd_set_p, *mut timeval],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    // `socketgetsockname` / `socketgetsockopt` (`external`, `save_err=SAVE_ERR`).
    // The descriptor is `lltype.Unsigned` (`socketfd_type` on Windows).
    crate::rffi::llexternal!(
        pub socketgetsockname = "getsockname",
        [usize, *mut sockaddr, *mut INT],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub socketgetsockopt = "getsockopt",
        [usize, INT, INT, *mut core::ffi::c_void, *mut INT],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub socket = "socket",
        [INT, INT, INT],
        socketfd_type,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    // `socketclosename` is `closesocket` on Windows.
    crate::rffi::llexternal!(
        pub socketclose = "closesocket",
        [socketfd_type],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        releasegil = false,
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub socketclose_no_errno = "closesocket",
        [socketfd_type],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        releasegil = false
    );
    crate::rffi::llexternal!(
        pub socketconnect = "connect",
        [socketfd_type, *const sockaddr, socklen_t],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub getaddrinfo = "getaddrinfo",
        [*const c_char, *const c_char, *const addrinfo, *mut *mut addrinfo],
        INT,
        compilation_info = ECI,
        calling_conv = "win"
    );
    crate::rffi::llexternal!(
        pub freeaddrinfo = "freeaddrinfo",
        [*mut addrinfo],
        (),
        compilation_info = ECI,
        calling_conv = "win"
    );
    // The two buffer lengths are `size_t` upstream; the WinSock prototype
    // takes `DWORD`.
    crate::rffi::llexternal!(
        pub getnameinfo = "getnameinfo",
        [
            *const sockaddr,
            socklen_t,
            *mut c_char,
            DWORD,
            *mut c_char,
            DWORD,
            INT
        ],
        INT,
        compilation_info = ECI,
        calling_conv = "win"
    );
    crate::rffi::llexternal!(
        pub socketaccept = "accept",
        [socketfd_type, *mut sockaddr, *mut socklen_t],
        socketfd_type,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub socketbind = "bind",
        [socketfd_type, *const sockaddr, socklen_t],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub socketlisten = "listen",
        [socketfd_type, INT],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub socketgetpeername = "getpeername",
        [socketfd_type, *mut sockaddr, *mut socklen_t],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub socketsetsockopt = "setsockopt",
        [socketfd_type, INT, INT, *const c_void, socklen_t],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    // `if WIN32: socketrecv = external('recv', [..., rffi.INT, rffi.INT],
    // rffi.INT)`.  `send` / `sendto` / `recvfrom` take the same `int` length
    // and answer an `int` in the WinSock prototypes.
    crate::rffi::llexternal!(
        pub socketrecv = "recv",
        [socketfd_type, *mut c_void, INT, INT],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub recvfrom = "recvfrom",
        [
            socketfd_type,
            *mut c_void,
            INT,
            INT,
            *mut sockaddr,
            *mut socklen_t
        ],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub send = "send",
        [socketfd_type, *const c_void, INT, INT],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub sendto = "sendto",
        [
            socketfd_type,
            *const c_void,
            INT,
            INT,
            *const sockaddr,
            socklen_t
        ],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub socketshutdown = "shutdown",
        [socketfd_type, INT],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub gethostname = "gethostname",
        [*mut c_char, INT],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    // [3.14-spec] `_rsocket_rffi.py` saves nothing for the two lookups, and
    // `rsocket.gethost_common` raises `HSocketError(host)`;
    // `socketmodule.c gethost_common` at v3.14.6 raises
    // `set_herror(state, h_errno)`, and WinSock's `h_errno` is
    // `WSAGetLastError()`, so `socket.herror` carries that code.  The callers
    // read it from the saved slot.  Windows only; evidence is the C source at
    // the tag.
    crate::rffi::llexternal!(
        pub gethostbyname = "gethostbyname",
        [*const c_char],
        *mut hostent,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub gethostbyaddr = "gethostbyaddr",
        [*const c_char, INT, INT],
        *mut hostent,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub getservbyname = "getservbyname",
        [*const c_char, *const c_char],
        *mut servent,
        compilation_info = ECI,
        calling_conv = "win"
    );
    crate::rffi::llexternal!(
        pub getservbyport = "getservbyport",
        [INT, *const c_char],
        *mut servent,
        compilation_info = ECI,
        calling_conv = "win"
    );
    crate::rffi::llexternal!(
        pub getprotobyname = "getprotobyname",
        [*const c_char],
        *mut protoent,
        compilation_info = ECI,
        calling_conv = "win"
    );
    crate::rffi::llexternal!(
        pub ioctlsocket = "ioctlsocket",
        [socketfd_type, c_long, *mut c_ulong],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub WSAIoctl = "WSAIoctl",
        [
            socketfd_type,
            DWORD,
            *mut c_void,
            DWORD,
            *mut c_void,
            DWORD,
            LPDWORD,
            *mut c_void,
            *mut c_void
        ],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub WSAStartup = "WSAStartup",
        [WORD, *mut WSAData],
        INT,
        compilation_info = ECI,
        calling_conv = "win"
    );
    // `htons` / `ntohs` / `htonl` / `ntohl` (`external`, `releasegil=False`,
    // not the Darwin macro).
    crate::rffi::llexternal!(
        pub htons = "htons",
        [USHORT],
        USHORT,
        compilation_info = ECI,
        calling_conv = "win",
        releasegil = false
    );
    crate::rffi::llexternal!(
        pub ntohs = "ntohs",
        [USHORT],
        USHORT,
        compilation_info = ECI,
        calling_conv = "win",
        releasegil = false
    );
    crate::rffi::llexternal!(
        pub htonl = "htonl",
        [UINT],
        UINT,
        compilation_info = ECI,
        calling_conv = "win",
        releasegil = false
    );
    crate::rffi::llexternal!(
        pub ntohl = "ntohl",
        [UINT],
        UINT,
        compilation_info = ECI,
        calling_conv = "win",
        releasegil = false
    );

    /// `struct in_addr`. `S_addr` is the first field of `S_un`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct in_addr {
        pub s_addr: u32,
    }

    // `inet_addr` / `inet_ntoa`. Windows has no `inet_aton`.
    crate::rffi::llexternal!(
        pub inet_addr = "inet_addr",
        [*const std::ffi::c_char],
        UINT,
        compilation_info = ECI,
        calling_conv = "win"
    );
    crate::rffi::llexternal!(
        pub inet_ntoa = "inet_ntoa",
        [in_addr],
        *mut std::ffi::c_char,
        compilation_info = ECI,
        calling_conv = "win"
    );

    /// `AF_INET` / `AF_INET6` in WinSock.
    pub const AF_INET: INT = 2;
    pub const AF_INET6: INT = 23;

    crate::rffi::llexternal!(
        pub inet_pton = "inet_pton",
        [INT, *const std::ffi::c_char, *mut core::ffi::c_void],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub inet_ntop = "inet_ntop",
        [INT, *const core::ffi::c_void, *mut std::ffi::c_char, usize],
        *const std::ffi::c_char,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );

    // The `CConfig` structs at their SDK sizes (`platform.configure` measures
    // them; these are written out by hand).
    #[cfg(target_pointer_width = "64")]
    const _: () = {
        use std::mem::{offset_of, size_of};
        assert!(size_of::<sockaddr>() == 16);
        assert!(size_of::<sockaddr_in>() == 16);
        assert!(size_of::<sockaddr_in6>() == 28);
        assert!(offset_of!(sockaddr_in6, sin6_scope_id) == 24);
        assert!(size_of::<sockaddr_storage>() == 128);
        assert!(size_of::<addrinfo>() == 48);
        assert!(offset_of!(addrinfo, ai_canonname) == 24);
        assert!(size_of::<hostent>() == 32);
        assert!(offset_of!(servent, s_port) == 24);
        assert!(size_of::<protoent>() == 24);
        assert!(size_of::<WSAData>() == 408);
        assert!(offset_of!(WSAData, szDescription) == 16);
        assert!(size_of::<fd_set>() == 8 + 8 * FD_SETSIZE);
    };

    #[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
    mod tests {
        use super::*;

        #[test]
        fn loopback_bind_getsockname_close() {
            crate::rsocket::rsocket_startup();
            unsafe {
                let fd = socket(AF_INET, SOCK_STREAM, 0);
                assert!(!invalid_socket(fd), "socket error {}", geterrno());
                let mut addr: sockaddr_in = std::mem::zeroed();
                addr.sin_family = AF_INET as u16;
                addr.sin_addr.s_addr = htonl(0x7f00_0001);
                let len = std::mem::size_of::<sockaddr_in>() as socklen_t;
                let rc = socketbind(fd, (&raw const addr).cast(), len);
                assert_eq!(rc, 0, "bind error {}", geterrno());
                let mut bound: sockaddr_in = std::mem::zeroed();
                let mut bound_len = len;
                let rc = socketgetsockname(fd, (&raw mut bound).cast(), &raw mut bound_len);
                assert_eq!(rc, 0, "getsockname error {}", geterrno());
                assert_eq!(ntohl(bound.sin_addr.s_addr), 0x7f00_0001);
                assert_ne!(bound.sin_port, 0);
                assert_eq!(socketclose(fd), 0);
                // A handle that is no socket answers WSAENOTSOCK through the
                // saved slot.  INVALID_SOCKET rather than the closed `fd`: a
                // parallel test may already have been handed that value.
                assert_eq!(socketclose(INVALID_SOCKET), SOCKET_ERROR);
                assert_eq!(geterrno(), WSAENOTSOCK);
            }
        }

        #[test]
        fn getaddrinfo_numeric_host() {
            crate::rsocket::rsocket_startup();
            unsafe {
                let mut hints: addrinfo = std::mem::zeroed();
                hints.ai_family = AF_INET;
                hints.ai_flags = AI_NUMERICHOST;
                let mut res: *mut addrinfo = std::ptr::null_mut();
                let rc = getaddrinfo(c"127.0.0.1".as_ptr(), std::ptr::null(), &hints, &mut res);
                assert_eq!(rc, 0);
                assert!(!res.is_null());
                assert_eq!((*res).ai_family, AF_INET);
                let sin = (*res).ai_addr.cast::<sockaddr_in>();
                assert_eq!(ntohl((*sin).sin_addr.s_addr), 0x7f00_0001);
                freeaddrinfo(res);
            }
        }
    }
}

#[cfg(windows)]
pub use winsock::*;
