//! _socket module — PyPy: pypy/module/_socket/interp_socket.py.
//!
//! Carries the W_Socket class implementation plus the shared address
//! conversion / IDNA / error-mapping helpers.  `register_module` is the
//! single entry point invoked by `moduledef::init`; it populates the
//! module namespace with constants, error classes, module-level
//! functions and the `socket` type definition.

/// The same names again where there is no host layer to reach them through:
/// the four address converters are arithmetic over the spelling, so they are
/// written out rather than called, and the entry points below cannot tell.
#[cfg(not(any(unix, windows)))]
use super::interp_socket_wasm as rffi;
/// The host socket layer, in the role `_rsocket_rffi.py` plays for rsocket:
/// one set of names over libc and WinSock, so the bodies below name a single
/// API.
#[cfg(any(unix, windows))]
use pyre_interpreter::rsocket_rffi as rffi;

/// _socket module — PyPy: pypy/module/_socket/.
///
/// **Slice S1: constants + name resolution helpers.**
///
/// Provides the AF_* / SOCK_* / IPPROTO_* / SOL_* / SO_* / SHUT_* /
/// AI_* / NI_* / IPV4-IPV6 constants plus the small "lookup" helpers
/// gethostname / sethostname / inet_aton / inet_ntoa / inet_pton /
/// inet_ntop / htons / htonl / ntohs / ntohl / getservbyname /
/// getservbyport / gethostbyname.
///
/// Does NOT yet provide the `socket` class itself — that requires
/// per-instance heap state (the OwnedFd + family/type/proto triple) and
/// is the next slice (S2).  Until then `import socket` succeeds and the
/// constants/helpers above are usable, but `socket.socket(...)` raises
/// the C-extension stub error.

/// `interp_socket.py idna_converter` — turn the host of an address tuple
/// into the bytes a sockaddr is built from.  Accepts str / bytes /
/// bytearray, refuses an embedded null, and — unlike `socket_encode_idna`
/// — hands an ASCII host straight through, so a numeric address or a
/// plain ASCII name never enters the codec.
///
/// Three details follow `getsockaddrarg`'s own converter rather than
/// `idna_converter`, which spells each of them differently.  The fast
/// path is gated on the argument being an exact `str`, matching the
/// compact-ASCII representation no `str` subclass instance has, where
/// `idna_converter` gates on an ascii *encode* succeeding.  A codec
/// failure becomes a bare TypeError rather than the codec's own error.
/// And the wrong-type message names the three accepted types.  MEASURED
/// 2026-08-27 with `class S(str): pass`: `connect((S("a" * 70), 80))`
/// answers `TypeError("encoding of hostname failed")` under CPython
/// 3.14.2 and `gaierror(11001)` under PyPy 7.3.22, and a host whose
/// first label is empty answers that same TypeError against
/// `UnicodeError("encoding with 'idna' codec failed ...")`.
#[cfg(any(unix, windows))]
fn socket_idna_converter(
    w_host: pyre_object::PyObjectRef,
) -> Result<Vec<u8>, pyre_interpreter::PyError> {
    let wrong_type = |obj: pyre_object::PyObjectRef| {
        pyre_interpreter::PyError::type_error(format!(
            "str, bytes or bytearray expected, not {}",
            pyre_interpreter::type_methods::arg_type_name(obj)
        ))
    };
    if w_host.is_null() {
        return Err(wrong_type(w_host));
    }
    let bytes: Vec<u8> = unsafe {
        if pyre_object::is_str(w_host) {
            // The ASCII attempt reads the raw buffer, so a host that has no
            // utf-8 view — one carrying a lone surrogate — reaches the `idna`
            // fallback the same way any other non-ASCII host does.
            let s = pyre_object::unicodeobject::w_str_get_wtf8(w_host);
            if pyre_object::is_exact_type(w_host, &pyre_object::STR_TYPE) && s.as_bytes().is_ascii()
            {
                s.as_bytes().to_vec()
            } else {
                // `space.encode_unicode_object(w_host, 'idna', None)`: the
                // codec runs on the string value itself rather than through
                // an `encode` attribute lookup, so a `str` subclass cannot
                // decide how its hostname reaches the resolver.  Whatever it
                // raised is dropped for the one message that covers every way
                // an address host can fail to encode, which also drops a
                // failure the codec did not cause.
                pyre_interpreter::type_methods::encode_object(w_host, "idna", "strict").map_err(
                    |_| pyre_interpreter::PyError::type_error("encoding of hostname failed"),
                )?
            }
        } else if pyre_object::bytesobject::is_bytes_like(w_host) {
            // The guard admits a bytearray, so the read has to dispatch on
            // the storage type: `w_bytes_data` would take a
            // `W_BytearrayObject`'s `*mut Vec<u8>` for a `*const BytesBlock`
            // and slice from the capacity word.
            pyre_object::bytesobject::bytes_like_data(w_host).to_vec()
        } else {
            return Err(wrong_type(w_host));
        }
    };
    if bytes.contains(&0) {
        return Err(pyre_interpreter::PyError::type_error(
            "host name must not contain null character",
        ));
    }
    Ok(bytes)
}

/// `interp_func.py encode_idna` — run the `idna` codec over a host
/// argument with no ASCII shortcut, so an empty label or one 64 bytes or
/// longer is refused before any resolver sees it and an
/// internationalized name reaches the resolver in its ACE spelling.
/// `caller` names the function the argument was passed to.
///
/// `encode_idna` hands the argument to the unbound `unicode.encode`, so a
/// `bytes` or `bytearray` host is a TypeError from the gateway there.  The
/// `et` argument converter these three entry points parse their host with
/// accepts both and passes them through untouched, and that is the 3.14
/// answer.  MEASURED 2026-08-27: `gethostbyname(b"localhost")` and
/// `gethostbyname(bytearray(b"localhost"))` both answer `"127.0.0.1"`
/// under CPython 3.14.2, where PyPy 7.3.22 raises
/// `TypeError("'str' object expected, got 'bytes' instead")`.  Only a
/// `str` is ever encoded: the bytes forms already name an encoded host.
#[cfg(any(unix, windows))]
fn socket_encode_idna(
    caller: &str,
    w_host: pyre_object::PyObjectRef,
) -> Result<Vec<u8>, pyre_interpreter::PyError> {
    unsafe {
        if pyre_object::is_str(w_host) {
            pyre_interpreter::type_methods::encode_object(w_host, "idna", "strict")
        } else if pyre_object::bytesobject::is_bytes_like(w_host) {
            Ok(pyre_object::bytesobject::bytes_like_data(w_host).to_vec())
        } else {
            Err(pyre_interpreter::PyError::type_error(format!(
                "{caller}() argument 1 must be str, bytes or bytearray, not {}",
                pyre_interpreter::type_methods::clinic_arg_type_name(w_host)
            )))
        }
    }
}

/// The host argument of `gethostbyname`, `gethostbyname_ex` and
/// `gethostbyaddr`: `socket_encode_idna` followed by the `et` converter's
/// own `strlen(buf) != size` scan, which refuses a host carrying an
/// embedded null.  `encode_idna` has no such scan — MEASURED 2026-08-27,
/// PyPy 7.3.22 resolves a host whose name is followed by a null and more
/// text as if the text were not there, because the resolver stops at the
/// null — so the refusal is the 3.14 answer.
///
/// The scan runs after the codec, so a host that is both bad idna and
/// null-bearing reports the codec's error.  The type name the message
/// wants is read first: the codec runs Python, and `w_host` is a bare
/// reference that a collection can leave behind.
#[cfg(any(unix, windows))]
fn socket_idna_host_arg(
    caller: &str,
    w_host: pyre_object::PyObjectRef,
) -> Result<std::ffi::CString, pyre_interpreter::PyError> {
    let type_name = pyre_interpreter::type_methods::clinic_arg_type_name(w_host);
    let bytes = socket_encode_idna(caller, w_host)?;
    std::ffi::CString::new(bytes).map_err(|_| {
        pyre_interpreter::PyError::type_error(format!(
            "{caller}() argument 1 must be encoded string without null bytes, not {type_name}"
        ))
    })
}

/// The C string a `getaddrinfo` argument becomes.  Neither
/// `interp_func.py getaddrinfo` nor the 3.14 entry point scans its host or
/// its service for an embedded null: the encoded bytes are handed to
/// `getaddrinfo` as a C string, so the name simply stops at the first one.
/// MEASURED 2026-08-27: a host of `"localhost"` followed by a null and
/// more text resolves to what `"localhost"` alone resolves to under
/// CPython 3.14.2 and PyPy 7.3.22 alike.
#[cfg(any(unix, windows))]
fn socket_cstring_at_nul(mut bytes: Vec<u8>) -> std::ffi::CString {
    if let Some(nul) = bytes.iter().position(|&b| b == 0) {
        bytes.truncate(nul);
    }
    // The truncation removed every interior null.
    std::ffi::CString::new(bytes).unwrap()
}

/// `interp_socket.py converted_error` — turn an rsocket
/// `SocketError` subclass into the matching python-level exception.
///
/// `applevelerrcls` matches the field defined on each rsocket error class
/// (`rsocket.py` `SocketError`, `GAIError`, `HSocketError`,
/// `SocketTimeout`):
///   "error"    → builtin `OSError`
///   "gaierror" → `_socket.gaierror` (OSError subclass)
///   "herror"   → `_socket.herror`   (OSError subclass)
///   "timeout"  → builtin `TimeoutError` (`get_error` answers
///                `space.w_TimeoutError` for that name, NOT the
///                `_socket.timeout` attribute, which is a separate
///                OSError subclass exposed for `isinstance` use)
///
/// When `errno` is `Some`, builds the exception with `(errno, message)`
/// the way `converted_error` does for a `SocketErrorWithErrno`; otherwise
/// only `(message,)`, like the plain `SocketError`.
#[cfg(any(unix, windows))]
fn socket_converted_error(
    applevelerrcls: &str,
    errno: Option<i32>,
    message: &str,
) -> pyre_interpreter::PyError {
    let cls = match applevelerrcls {
        "timeout" => pyre_interpreter::builtins::lookup_exc_class("TimeoutError"),
        "gaierror" => pyre_interpreter::builtins::lookup_exc_class("socket.gaierror"),
        "herror" => pyre_interpreter::builtins::lookup_exc_class("socket.herror"),
        _ => pyre_interpreter::builtins::lookup_exc_class("OSError"),
    }
    .or_else(|| pyre_interpreter::builtins::lookup_exc_class("OSError"))
    .expect("OSError must be installed");

    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let w_value = if let Some(e) = errno {
        let errno_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(e as i64));
        let msg_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(message));
        pyre_object::w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(errno_slot),
            pyre_object::gc_roots::shadow_stack_get(msg_slot),
        ])
    } else {
        pyre_object::w_str_new_managed(message)
    };
    pyre_interpreter::PyError::from_type_and_value(
        pyre_object::gc_roots::shadow_stack_get(cls_slot),
        w_value,
    )
}

#[cfg(all(windows, feature = "host_env"))]
fn interface_io_error(error: std::io::Error) -> pyre_interpreter::PyError {
    use rustpython_host_env::os::ErrorExt;

    // The two scalar conversions fail through the C runtime `errno` the IP
    // Helper API sets, which `raw_os_error` leaves empty on purpose so that no
    // `winerror` is attached to it; `posix_errno` is what recovers it.  The
    // enumeration path instead fails through a Win32 status, and that one is
    // reported as the Win32 flavour it is.
    let Some(winerror) = error.raw_os_error() else {
        let errno = error.posix_errno();
        let message = match errno {
            libc::ENODEV => "No such device".to_string(),
            libc::ENXIO => "No such device or address".to_string(),
            _ => error.to_string(),
        };
        return pyre_interpreter::PyError::os_error_with_errno(errno, message);
    };
    pyre_interpreter::PyError::os_error_win32_syscall2(
        winerror,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
    )
}

/// One `space.acquire_writebuf` export held for the whole of a `recv_into`
/// family call, the way `with rwbuffer:` keeps it across `c_recv`.
///
/// The syscall runs with the GIL released, so without the export another
/// thread could resize the exporter and reallocate the storage the kernel is
/// writing through; the count is what makes that resize raise instead.  The
/// requested object and the concrete storage owner are both rooted, because
/// the release in `Drop` has to name the owner after whatever Python ran in
/// between.
#[cfg(any(unix, windows))]
struct SocketWritableBuffer {
    owner_slot: usize,
    held: bool,
    address: *mut u8,
    length: usize,
}

#[cfg(any(unix, windows))]
impl SocketWritableBuffer {
    /// The root scope the pins land in belongs to the caller.  A scope per
    /// buffer would be released in the order the buffers are dropped, and a
    /// `Vec` of them drops front to back, so the first release would truncate
    /// the stack that the slots the later ones still name live in —
    /// `recvmsg_into` with two buffers is the case that reaches it.
    unsafe fn acquire(obj: pyre_object::PyObjectRef) -> Result<Self, pyre_interpreter::PyError> {
        let _ = pyre_object::gc_roots::pin_root(obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let (data, owner) = socket_writebuf(pyre_object::gc_roots::shadow_stack_get(obj_slot))?;
        let _ = pyre_object::gc_roots::pin_root(owner);
        let owner_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let owner = pyre_object::gc_roots::shadow_stack_get(owner_slot);
        let held = unsafe { pyre_interpreter::builtins::buffer_export_incref(owner) };
        Ok(Self {
            owner_slot,
            held,
            address: data.as_mut_ptr(),
            length: data.len(),
        })
    }

    unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.address, self.length) }
    }
}

#[cfg(any(unix, windows))]
impl Drop for SocketWritableBuffer {
    fn drop(&mut self) {
        if self.held {
            let owner = pyre_object::gc_roots::shadow_stack_get(self.owner_slot);
            unsafe { pyre_interpreter::builtins::buffer_export_decref(owner) };
        }
    }
}

/// baseobjspace.py `writebuf_w` — the writable byte slice backing a
/// scatter/gather buffer argument, with the object the export count lives on.
/// PyPy accepts any object exporting a writable buffer; pyre's writable byte
/// stores are `bytearray` and a `memoryview` over one, so those are resolved
/// here and anything else is rejected as `writebuf_w` does.
#[cfg(any(unix, windows))]
fn socket_writebuf(
    obj: pyre_object::PyObjectRef,
) -> Result<(&'static mut [u8], pyre_object::PyObjectRef), pyre_interpreter::PyError> {
    if unsafe { pyre_object::bytearrayobject::is_bytearray(obj) } {
        return Ok((
            unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(obj) },
            obj,
        ));
    }
    if unsafe { pyre_object::interp_array::is_array(obj) } {
        // `space.writebuf_w` accepts any writable buffer exporter; an
        // `array.array` exposes its element bytes as one writable window
        // regardless of typecode (recv writes raw bytes into them).
        return Ok((
            unsafe { pyre_object::interp_array::w_array_vec_mut(obj).as_mut_slice() },
            obj,
        ));
    }
    if unsafe { pyre_object::memoryview::is_w_memoryview(obj) } {
        // `space.buffer_w` rejects a released view before exposing its storage.
        unsafe { pyre_interpreter::builtins::memoryview_check_released(obj) }?;
        // A read-write buffer is required; a read-only view cannot back recv_into.
        if unsafe { pyre_object::memoryview::w_memoryview_readonly(obj) } {
            return Err(pyre_interpreter::PyError::type_error(
                "a read-write bytes-like object is required, not 'memoryview'",
            ));
        }
        // Only C-contiguous views are accepted; a strided slice (`m[::2]`,
        // `m[::-1]`) would need a scatter writer pyre does not have.  A
        // contiguous N-D view (`memoryview(ba).cast('B', shape=(2, 2))`)
        // exposes its window as one flat byte range, so it qualifies even
        // though its outermost stride is a row stride, not the itemsize.
        if !unsafe { pyre_interpreter::builtins::memoryview_contiguity(obj).0 } {
            return Err(pyre_interpreter::PyError::type_error(
                "a read-write bytes-like object is required, not 'memoryview'",
            ));
        }
        let view = unsafe { pyre_object::memoryview::w_memoryview_view(obj) };
        // The view itself carries the export, as it does for `readinto`. Its
        // backing is already held by the view's own construction, so nothing
        // can resize the storage underneath while this one is alive either.
        let owner = obj;
        // A writable view's backing is a `bytearray` or an `array.array`; both
        // expose a mutable byte store.  Honour the view window: write only into
        // `[offset, offset+length)` of the backing storage (itself already the
        // `Buffer::Sub` window for a zero-copy slice), not the whole buffer.
        let Some(full) = (unsafe { view.backing().as_bytes_mut() }) else {
            return Err(pyre_interpreter::PyError::type_error(
                "cannot modify read-only memory",
            ));
        };
        let off = unsafe { view.offset() } as usize;
        let len = unsafe { pyre_object::memoryview::w_memoryview_length(obj) } as usize;
        // The backing may have been resized after the view was taken; reject a
        // window that no longer fits rather than panic.
        if off.checked_add(len).is_none_or(|end| end > full.len()) {
            return Err(pyre_interpreter::PyError::value_error(
                "memoryview buffer is no longer valid",
            ));
        }
        return Ok((&mut full[off..off + len], owner));
    }
    Err(pyre_interpreter::PyError::type_error(
        "a writable bytes-like object is required",
    ))
}

pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // `_rsocket_rffi.py:1150 rwin32.get_wsa_error`'s companion: WinSock has to
    // be started before any of its entry points answers, so the module takes
    // that cost at import rather than leaving the first call to fail with
    // WSANOTINITIALISED.  The individual entry points repeat it, since nothing
    // guarantees `_socket` is the first importer.
    #[cfg(any(unix, windows))]
    rffi::init();

    // `_rsocket_rffi.py constant_names` + `:234-262
    // constants_w_defaults` — populated through the libc crate where
    // available, hardcoded for platform-specific constants the crate
    // does not expose.  Mirrors PyPy's
    // `for constant, value in rsocket.constants.iteritems(): wrap(value)`
    // loop in `_socket/moduledef.py:48-50`.
    #[cfg(unix)]
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
        // ── Address families ──
        cst!("AF_UNSPEC", libc::AF_UNSPEC);
        cst!("AF_UNIX", libc::AF_UNIX);
        cst!("AF_INET", libc::AF_INET);
        cst!("AF_INET6", libc::AF_INET6);
        cst!("AF_ROUTE", libc::AF_ROUTE);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            cst!("AF_PACKET", libc::AF_PACKET);
            cst!("AF_NETLINK", libc::AF_NETLINK);
            cst!("AF_VSOCK", libc::AF_VSOCK);
        }
        // ── Socket types ──
        cst!("SOCK_STREAM", libc::SOCK_STREAM);
        cst!("SOCK_DGRAM", libc::SOCK_DGRAM);
        cst!("SOCK_RAW", libc::SOCK_RAW);
        cst!("SOCK_RDM", libc::SOCK_RDM);
        cst!("SOCK_SEQPACKET", libc::SOCK_SEQPACKET);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            cst!("SOCK_CLOEXEC", libc::SOCK_CLOEXEC);
            cst!("SOCK_NONBLOCK", libc::SOCK_NONBLOCK);
        }
        // ── Protocols ──
        cst!("IPPROTO_IP", libc::IPPROTO_IP);
        cst!("IPPROTO_HOPOPTS", libc::IPPROTO_HOPOPTS);
        cst!("IPPROTO_ICMP", libc::IPPROTO_ICMP);
        cst!("IPPROTO_IGMP", libc::IPPROTO_IGMP);
        cst!("IPPROTO_IPIP", libc::IPPROTO_IPIP);
        cst!("IPPROTO_TCP", libc::IPPROTO_TCP);
        cst!("IPPROTO_EGP", libc::IPPROTO_EGP);
        cst!("IPPROTO_PUP", libc::IPPROTO_PUP);
        cst!("IPPROTO_UDP", libc::IPPROTO_UDP);
        cst!("IPPROTO_IDP", libc::IPPROTO_IDP);
        cst!("IPPROTO_TP", libc::IPPROTO_TP);
        cst!("IPPROTO_IPV6", libc::IPPROTO_IPV6);
        cst!("IPPROTO_ROUTING", libc::IPPROTO_ROUTING);
        cst!("IPPROTO_FRAGMENT", libc::IPPROTO_FRAGMENT);
        cst!("IPPROTO_ESP", libc::IPPROTO_ESP);
        cst!("IPPROTO_AH", libc::IPPROTO_AH);
        cst!("IPPROTO_ICMPV6", libc::IPPROTO_ICMPV6);
        cst!("IPPROTO_NONE", libc::IPPROTO_NONE);
        cst!("IPPROTO_DSTOPTS", libc::IPPROTO_DSTOPTS);
        cst!("IPPROTO_PIM", libc::IPPROTO_PIM);
        cst!("IPPROTO_SCTP", libc::IPPROTO_SCTP);
        cst!("IPPROTO_RAW", libc::IPPROTO_RAW);
        // libc deprecates this: the kernel raised the value and libc will
        // follow upstream in a later release (rust-lang/libc#1896). Keep
        // reading it from libc rather than freezing a literal, so the new
        // value arrives with the dependency; `socketmodule.c` likewise takes
        // whatever the platform header defines.
        #[allow(deprecated)]
        {
            cst!("IPPROTO_MAX", libc::IPPROTO_MAX);
        }
        cst!("IPPROTO_GRE", libc::IPPROTO_GRE);
        cst!("IPPROTO_RSVP", libc::IPPROTO_RSVP);
        // `_rsocket_rffi.py constants_w_defaults` — SOL_IP/TCP/UDP
        // and IPPROTO_* duplicates kept for PyPy compatibility.
        cst!("SOL_IP", 0);
        cst!("SOL_TCP", 6);
        cst!("SOL_UDP", 17);
        // ── INADDR_* (host byte order) ──
        cst!("INADDR_ANY", libc::INADDR_ANY);
        cst!("INADDR_LOOPBACK", libc::INADDR_LOOPBACK);
        cst!("INADDR_BROADCAST", libc::INADDR_BROADCAST);
        cst!("INADDR_NONE", libc::INADDR_NONE);
        cst!("INADDR_ALLHOSTS_GROUP", 0xe0000001u32);
        cst!("INADDR_UNSPEC_GROUP", 0xe0000000u32);
        cst!("INADDR_MAX_LOCAL_GROUP", 0xe00000ffu32);
        cst!("IPPORT_RESERVED", 1024);
        cst!("IPPORT_USERRESERVED", 5000);
        // ── SOL_* / SO_* (socket level) ──
        cst!("SOL_SOCKET", libc::SOL_SOCKET);
        cst!("SO_REUSEADDR", libc::SO_REUSEADDR);
        cst!("SO_REUSEPORT", libc::SO_REUSEPORT);
        cst!("SO_KEEPALIVE", libc::SO_KEEPALIVE);
        cst!("SO_BROADCAST", libc::SO_BROADCAST);
        cst!("SO_DEBUG", libc::SO_DEBUG);
        cst!("SO_DONTROUTE", libc::SO_DONTROUTE);
        cst!("SO_LINGER", libc::SO_LINGER);
        cst!("SO_OOBINLINE", libc::SO_OOBINLINE);
        cst!("SO_RCVBUF", libc::SO_RCVBUF);
        cst!("SO_SNDBUF", libc::SO_SNDBUF);
        cst!("SO_RCVTIMEO", libc::SO_RCVTIMEO);
        cst!("SO_SNDTIMEO", libc::SO_SNDTIMEO);
        cst!("SO_RCVLOWAT", libc::SO_RCVLOWAT);
        cst!("SO_SNDLOWAT", libc::SO_SNDLOWAT);
        cst!("SO_ERROR", libc::SO_ERROR);
        cst!("SO_TYPE", libc::SO_TYPE);
        cst!("SO_ACCEPTCONN", libc::SO_ACCEPTCONN);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            cst!("SO_DOMAIN", libc::SO_DOMAIN);
            cst!("SO_PROTOCOL", libc::SO_PROTOCOL);
            cst!("SO_PEERCRED", libc::SO_PEERCRED);
            cst!("SO_PASSCRED", libc::SO_PASSCRED);
            cst!("SO_PEERSEC", libc::SO_PEERSEC);
            cst!("SO_PASSSEC", libc::SO_PASSSEC);
        }
        // ── TCP-level ──
        cst!("TCP_NODELAY", libc::TCP_NODELAY);
        cst!("TCP_MAXSEG", libc::TCP_MAXSEG);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            cst!("TCP_KEEPIDLE", libc::TCP_KEEPIDLE);
            cst!("TCP_KEEPINTVL", libc::TCP_KEEPINTVL);
            cst!("TCP_KEEPCNT", libc::TCP_KEEPCNT);
            cst!("TCP_CORK", libc::TCP_CORK);
            cst!("TCP_DEFER_ACCEPT", libc::TCP_DEFER_ACCEPT);
            cst!("TCP_INFO", libc::TCP_INFO);
            cst!("TCP_LINGER2", libc::TCP_LINGER2);
            cst!("TCP_QUICKACK", libc::TCP_QUICKACK);
            cst!("TCP_SYNCNT", libc::TCP_SYNCNT);
            cst!("TCP_WINDOW_CLAMP", libc::TCP_WINDOW_CLAMP);
            cst!("TCP_USER_TIMEOUT", libc::TCP_USER_TIMEOUT);
            cst!("TCP_CONGESTION", libc::TCP_CONGESTION);
            cst!("TCP_FASTOPEN", libc::TCP_FASTOPEN);
            cst!("TCP_NOTSENT_LOWAT", libc::TCP_NOTSENT_LOWAT);
        }
        #[cfg(target_os = "macos")]
        {
            cst!("TCP_KEEPALIVE", libc::TCP_KEEPALIVE);
        }
        // ── IP-level ──
        cst!("IP_TTL", libc::IP_TTL);
        cst!("IP_TOS", libc::IP_TOS);
        cst!("IP_MULTICAST_TTL", libc::IP_MULTICAST_TTL);
        cst!("IP_MULTICAST_LOOP", libc::IP_MULTICAST_LOOP);
        cst!("IP_MULTICAST_IF", libc::IP_MULTICAST_IF);
        cst!("IP_ADD_MEMBERSHIP", libc::IP_ADD_MEMBERSHIP);
        cst!("IP_DROP_MEMBERSHIP", libc::IP_DROP_MEMBERSHIP);
        cst!("IP_HDRINCL", libc::IP_HDRINCL);
        // IP_OPTIONS / IP_RECVOPTS / IP_RECVRETOPTS / IP_RETOPTS are
        // POSIX but the libc crate does not expose them; `_rsocket_rffi.py`
        // lists them in `constant_names`, and
        // `platform.DefinedConstantInteger` drops them when the header does
        // not define them.  The darwin values are spelled out below; on the
        // other unices they stay unexposed.
        cst!("IP_DEFAULT_MULTICAST_LOOP", 1);
        cst!("IP_DEFAULT_MULTICAST_TTL", 1);
        // `<netinet/in.h>` sizes the membership table per platform: 4095 on
        // darwin, 20 on linux.
        #[cfg(target_vendor = "apple")]
        cst!("IP_MAX_MEMBERSHIPS", 4095);
        #[cfg(not(target_vendor = "apple"))]
        cst!("IP_MAX_MEMBERSHIPS", 20);
        // ── IPv6 ──
        cst!("IPV6_V6ONLY", libc::IPV6_V6ONLY);
        cst!("IPV6_MULTICAST_HOPS", libc::IPV6_MULTICAST_HOPS);
        cst!("IPV6_MULTICAST_LOOP", libc::IPV6_MULTICAST_LOOP);
        cst!("IPV6_MULTICAST_IF", libc::IPV6_MULTICAST_IF);
        cst!("IPV6_UNICAST_HOPS", libc::IPV6_UNICAST_HOPS);
        cst!("IPV6_CHECKSUM", libc::IPV6_CHECKSUM);
        // `<netinet/in.h>` IPV6_JOIN_GROUP=20 / IPV6_LEAVE_GROUP=21 on Linux;
        // libc crate omits the symbols on linux-gnu though the kernel headers
        // define them.  Apple / BSD expose them with the BSD numbering (12 /
        // 13) — keep using `libc::*` there for header parity.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            cst!("IPV6_JOIN_GROUP", 20);
            cst!("IPV6_LEAVE_GROUP", 21);
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            cst!("IPV6_JOIN_GROUP", libc::IPV6_JOIN_GROUP);
            cst!("IPV6_LEAVE_GROUP", libc::IPV6_LEAVE_GROUP);
        }
        cst!("IPV6_RECVTCLASS", libc::IPV6_RECVTCLASS);
        cst!("IPV6_TCLASS", libc::IPV6_TCLASS);
        cst!("IPV6_RECVPKTINFO", libc::IPV6_RECVPKTINFO);
        cst!("IPV6_PKTINFO", libc::IPV6_PKTINFO);
        cst!("IPV6_RECVHOPLIMIT", libc::IPV6_RECVHOPLIMIT);
        cst!("IPV6_HOPLIMIT", libc::IPV6_HOPLIMIT);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            cst!("IPV6_DSTOPTS", libc::IPV6_DSTOPTS);
            cst!("IPV6_HOPOPTS", libc::IPV6_HOPOPTS);
            cst!("IPV6_NEXTHOP", libc::IPV6_NEXTHOP);
            cst!("IPV6_RECVDSTOPTS", libc::IPV6_RECVDSTOPTS);
            cst!("IPV6_RECVHOPOPTS", libc::IPV6_RECVHOPOPTS);
            cst!("IPV6_RECVRTHDR", libc::IPV6_RECVRTHDR);
            cst!("IPV6_RTHDR", libc::IPV6_RTHDR);
            cst!("IPV6_RTHDRDSTOPTS", libc::IPV6_RTHDRDSTOPTS);
            // `<netinet/in.h>` IPV6_RTHDR_TYPE_0=0; symbol omitted from
            // libc crate on linux-gnu but the kernel header defines it.
            cst!("IPV6_RTHDR_TYPE_0", 0);
        }
        // ── shutdown how ──
        cst!("SHUT_RD", libc::SHUT_RD);
        cst!("SHUT_WR", libc::SHUT_WR);
        cst!("SHUT_RDWR", libc::SHUT_RDWR);
        // ── Message flags ──
        cst!("MSG_OOB", libc::MSG_OOB);
        cst!("MSG_PEEK", libc::MSG_PEEK);
        cst!("MSG_DONTROUTE", libc::MSG_DONTROUTE);
        cst!("MSG_DONTWAIT", libc::MSG_DONTWAIT);
        cst!("MSG_WAITALL", libc::MSG_WAITALL);
        cst!("MSG_CTRUNC", libc::MSG_CTRUNC);
        cst!("MSG_TRUNC", libc::MSG_TRUNC);
        cst!("MSG_EOR", libc::MSG_EOR);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        cst!("MSG_ERRQUEUE", libc::MSG_ERRQUEUE);
        // ── Address-info flags ──
        cst!("AI_PASSIVE", libc::AI_PASSIVE);
        cst!("AI_CANONNAME", libc::AI_CANONNAME);
        cst!("AI_NUMERICHOST", libc::AI_NUMERICHOST);
        cst!("AI_NUMERICSERV", libc::AI_NUMERICSERV);
        cst!("AI_ADDRCONFIG", libc::AI_ADDRCONFIG);
        cst!("AI_V4MAPPED", libc::AI_V4MAPPED);
        cst!("AI_ALL", libc::AI_ALL);
        #[cfg(target_os = "macos")]
        {
            cst!("AI_DEFAULT", libc::AI_DEFAULT);
            cst!("AI_MASK", libc::AI_MASK);
            cst!("AI_V4MAPPED_CFG", libc::AI_V4MAPPED_CFG);
        }
        // ── Name-info flags ──
        cst!("NI_NUMERICHOST", libc::NI_NUMERICHOST);
        cst!("NI_NUMERICSERV", libc::NI_NUMERICSERV);
        cst!("NI_NOFQDN", libc::NI_NOFQDN);
        cst!("NI_NAMEREQD", libc::NI_NAMEREQD);
        cst!("NI_DGRAM", libc::NI_DGRAM);
        cst!("NI_MAXHOST", libc::NI_MAXHOST);
        // POSIX <netdb.h> NI_MAXSERV = 32; libc crate omits it on linux-gnu
        cst!("NI_MAXSERV", 32);
        // ── EAI_* (gai_strerror codes) ──
        cst!("EAI_AGAIN", libc::EAI_AGAIN);
        cst!("EAI_BADFLAGS", libc::EAI_BADFLAGS);
        cst!("EAI_FAIL", libc::EAI_FAIL);
        cst!("EAI_FAMILY", libc::EAI_FAMILY);
        cst!("EAI_MEMORY", libc::EAI_MEMORY);
        cst!("EAI_NODATA", libc::EAI_NODATA);
        cst!("EAI_NONAME", libc::EAI_NONAME);
        cst!("EAI_OVERFLOW", libc::EAI_OVERFLOW);
        cst!("EAI_SERVICE", libc::EAI_SERVICE);
        cst!("EAI_SOCKTYPE", libc::EAI_SOCKTYPE);
        cst!("EAI_SYSTEM", libc::EAI_SYSTEM);
        // `_rsocket_rffi.py` names these four alongside the rest and
        // `platform.DefinedConstantInteger` keeps whichever the platform's
        // `<netdb.h>` defines.  Darwin defines all four; the libc crate does
        // not re-export them, so they are spelled out here the way
        // `NI_MAXSERV` above is.
        #[cfg(target_vendor = "apple")]
        {
            cst!("EAI_ADDRFAMILY", 1);
            cst!("EAI_BADHINTS", 12);
            cst!("EAI_PROTOCOL", 13);
            cst!("EAI_MAX", 15);
        }
        // ── SCM_* (ancillary data types) ──
        cst!("SCM_RIGHTS", libc::SCM_RIGHTS);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        cst!("SCM_CREDENTIALS", libc::SCM_CREDENTIALS);
        // ── socket-level cap ──
        cst!("SOMAXCONN", libc::SOMAXCONN);

        // Names `<sys/socket.h>`, `<netinet/in.h>`, `<netinet/tcp.h>`,
        // `<net/ethernet.h>` and `<sys/sys_domain.h>` define only on darwin,
        // or define there with a different value than on linux.  Every value
        // was read back from the headers themselves.  `socketmodule.c` guards
        // the RFC 3542 half with `#define __APPLE_USE_RFC_3542 1`, which is
        // why those names exist on this platform at all.
        #[cfg(target_vendor = "apple")]
        {
            // ── Address / protocol families ──
            cst!("AF_APPLETALK", libc::AF_APPLETALK);
            cst!("AF_DECnet", libc::AF_DECnet);
            cst!("AF_IPX", libc::AF_IPX);
            cst!("AF_LINK", libc::AF_LINK);
            cst!("AF_SNA", libc::AF_SNA);
            cst!("AF_SYSTEM", libc::AF_SYSTEM);
            cst!("PF_SYSTEM", libc::PF_SYSTEM);
            cst!("SYSPROTO_CONTROL", libc::SYSPROTO_CONTROL);
            // ── IPPROTO_* ──
            cst!("IPPROTO_EON", libc::IPPROTO_EON);
            cst!("IPPROTO_GGP", libc::IPPROTO_GGP);
            cst!("IPPROTO_HELLO", libc::IPPROTO_HELLO);
            cst!("IPPROTO_IPCOMP", libc::IPPROTO_IPCOMP);
            cst!("IPPROTO_IPV4", 4);
            cst!("IPPROTO_ND", libc::IPPROTO_ND);
            cst!("IPPROTO_XTP", libc::IPPROTO_XTP);
            // ── IPv4 socket options ──
            cst!("IP_ADD_SOURCE_MEMBERSHIP", libc::IP_ADD_SOURCE_MEMBERSHIP);
            cst!("IP_BLOCK_SOURCE", libc::IP_BLOCK_SOURCE);
            cst!("IP_DROP_SOURCE_MEMBERSHIP", libc::IP_DROP_SOURCE_MEMBERSHIP);
            cst!("IP_UNBLOCK_SOURCE", libc::IP_UNBLOCK_SOURCE);
            cst!("IP_OPTIONS", 1);
            cst!("IP_RECVOPTS", 5);
            cst!("IP_RECVRETOPTS", 6);
            cst!("IP_RETOPTS", 8);
            cst!("IP_PKTINFO", libc::IP_PKTINFO);
            cst!("IP_RECVDSTADDR", libc::IP_RECVDSTADDR);
            cst!("IP_RECVTOS", libc::IP_RECVTOS);
            cst!("IP_RECVTTL", libc::IP_RECVTTL);
            // ── IPv6 socket options ──
            cst!("IPV6_DONTFRAG", libc::IPV6_DONTFRAG);
            cst!("IPV6_DSTOPTS", 50);
            cst!("IPV6_HOPOPTS", 49);
            cst!("IPV6_NEXTHOP", 48);
            cst!("IPV6_PATHMTU", 44);
            cst!("IPV6_RECVDSTOPTS", 40);
            cst!("IPV6_RECVHOPOPTS", 39);
            cst!("IPV6_RECVPATHMTU", 43);
            cst!("IPV6_RECVRTHDR", 38);
            cst!("IPV6_RTHDR", 51);
            cst!("IPV6_RTHDRDSTOPTS", 57);
            cst!("IPV6_RTHDR_TYPE_0", 0);
            cst!("IPV6_USE_MIN_MTU", 42);
            // ── TCP ──
            cst!("TCP_CONNECTION_INFO", libc::TCP_CONNECTION_INFO);
            cst!("TCP_FASTOPEN", libc::TCP_FASTOPEN);
            cst!("TCP_KEEPCNT", libc::TCP_KEEPCNT);
            cst!("TCP_KEEPINTVL", libc::TCP_KEEPINTVL);
            cst!("TCP_NOTSENT_LOWAT", 513);
            // ── SO_* / MSG_* / SCM_* ──
            cst!("SO_BINDTODEVICE", 4404);
            cst!("SO_USELOOPBACK", libc::SO_USELOOPBACK);
            cst!("LOCAL_PEERCRED", libc::LOCAL_PEERCRED);
            cst!("MSG_EOF", libc::MSG_EOF);
            cst!("MSG_NOSIGNAL", libc::MSG_NOSIGNAL);
            cst!("SCM_CREDS", libc::SCM_CREDS);
            // ── `<net/ethernet.h>` ──
            cst!("ETHERTYPE_ARP", 2054);
            cst!("ETHERTYPE_IP", 2048);
            cst!("ETHERTYPE_IPV6", 34525);
            cst!("ETHERTYPE_VLAN", 33024);
        }
    }

    // `_rsocket_rffi.py` keeps a separate `_MSVC` constant list because the
    // two headers name overlapping but different sets; this is that list.
    // The values are the host_env names of `<winsock2.h>` / `<ws2tcpip.h>` /
    // `<hvsocket.h>`. What is absent here is absent from the platform.
    #[cfg(windows)]
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
        // `$host as i32` before the widening cast: `PyModule_AddIntConstant`
        // takes a 32-bit `long`, so a top-bit word is the negative it spells.
        macro_rules! hc {
            ($py:literal, $host:ident, $fallback:expr) => {
                #[cfg(feature = "host_env")]
                cst!($py, rustpython_host_env::socket::$host);
                #[cfg(not(feature = "host_env"))]
                cst!($py, $fallback);
            };
            (signed $py:literal, $host:ident, $fallback:expr) => {
                #[cfg(feature = "host_env")]
                cst!($py, rustpython_host_env::socket::$host as i32);
                #[cfg(not(feature = "host_env"))]
                cst!($py, $fallback);
            };
        }
        #[cfg(not(feature = "host_env"))]
        use windows_sys::Win32::Devices::Bluetooth as bt;
        #[cfg(not(feature = "host_env"))]
        use windows_sys::Win32::Networking::WinSock as ws;
        // ── Address families ──
        hc!("AF_UNSPEC", AF_UNSPEC, ws::AF_UNSPEC);
        hc!("AF_INET", AF_INET, ws::AF_INET);
        hc!("AF_INET6", AF_INET6, ws::AF_INET6);
        hc!("AF_APPLETALK", AF_APPLETALK, ws::AF_APPLETALK);
        hc!("AF_DECnet", AF_DECnet, ws::AF_DECnet);
        hc!("AF_IPX", AF_IPX, ws::AF_IPX);
        hc!("AF_LINK", AF_LINK, ws::AF_LINK);
        hc!("AF_SNA", AF_SNA, 11);
        hc!("AF_IRDA", AF_IRDA, 26);
        hc!("AF_BLUETOOTH", AF_BLUETOOTH, bt::AF_BTH);
        hc!("AF_HYPERV", AF_HYPERV, ws::AF_HYPERV);
        // ── Socket types ──
        hc!("SOCK_STREAM", SOCK_STREAM, ws::SOCK_STREAM);
        hc!("SOCK_DGRAM", SOCK_DGRAM, ws::SOCK_DGRAM);
        hc!("SOCK_RAW", SOCK_RAW, ws::SOCK_RAW);
        hc!("SOCK_RDM", SOCK_RDM, ws::SOCK_RDM);
        hc!("SOCK_SEQPACKET", SOCK_SEQPACKET, ws::SOCK_SEQPACKET);
        // ── Protocols ──
        hc!("IPPROTO_IP", IPPROTO_IP, ws::IPPROTO_IP);
        hc!("IPPROTO_HOPOPTS", IPPROTO_HOPOPTS, ws::IPPROTO_HOPOPTS);
        hc!("IPPROTO_ICMP", IPPROTO_ICMP, ws::IPPROTO_ICMP);
        hc!("IPPROTO_IGMP", IPPROTO_IGMP, ws::IPPROTO_IGMP);
        hc!("IPPROTO_GGP", IPPROTO_GGP, ws::IPPROTO_GGP);
        hc!("IPPROTO_ST", IPPROTO_ST, 5);
        hc!("IPPROTO_CBT", IPPROTO_CBT, 7);
        hc!("IPPROTO_IGP", IPPROTO_IGP, 9);
        hc!("IPPROTO_IPV4", IPPROTO_IPV4, ws::IPPROTO_IPV4);
        hc!("IPPROTO_TCP", IPPROTO_TCP, ws::IPPROTO_TCP);
        hc!("IPPROTO_EGP", IPPROTO_EGP, ws::IPPROTO_EGP);
        hc!("IPPROTO_PUP", IPPROTO_PUP, ws::IPPROTO_PUP);
        hc!("IPPROTO_UDP", IPPROTO_UDP, ws::IPPROTO_UDP);
        hc!("IPPROTO_IDP", IPPROTO_IDP, ws::IPPROTO_IDP);
        hc!("IPPROTO_ICLFXBM", IPPROTO_ICLFXBM, 78);
        hc!("IPPROTO_IPV6", IPPROTO_IPV6, ws::IPPROTO_IPV6);
        hc!("IPPROTO_ROUTING", IPPROTO_ROUTING, ws::IPPROTO_ROUTING);
        hc!("IPPROTO_FRAGMENT", IPPROTO_FRAGMENT, ws::IPPROTO_FRAGMENT);
        hc!("IPPROTO_ESP", IPPROTO_ESP, ws::IPPROTO_ESP);
        hc!("IPPROTO_AH", IPPROTO_AH, ws::IPPROTO_AH);
        hc!("IPPROTO_ICMPV6", IPPROTO_ICMPV6, ws::IPPROTO_ICMPV6);
        hc!("IPPROTO_NONE", IPPROTO_NONE, ws::IPPROTO_NONE);
        hc!("IPPROTO_DSTOPTS", IPPROTO_DSTOPTS, ws::IPPROTO_DSTOPTS);
        hc!("IPPROTO_ND", IPPROTO_ND, ws::IPPROTO_ND);
        hc!("IPPROTO_PIM", IPPROTO_PIM, ws::IPPROTO_PIM);
        hc!("IPPROTO_PGM", IPPROTO_PGM, ws::IPPROTO_PGM);
        hc!("IPPROTO_RDP", IPPROTO_RDP, ws::IPPROTO_RDP);
        hc!("IPPROTO_L2TP", IPPROTO_L2TP, 115);
        hc!("IPPROTO_SCTP", IPPROTO_SCTP, ws::IPPROTO_SCTP);
        hc!("IPPROTO_RAW", IPPROTO_RAW, ws::IPPROTO_RAW);
        hc!("IPPROTO_MAX", IPPROTO_MAX, ws::IPPROTO_MAX);
        // SOL_TCP/UDP are kept for the older constant list. SOL_IP is the
        // platform level, not the zero placeholder the other arm uses.
        hc!("SOL_IP", SOL_IP, ws::SOL_IP);
        hc!("SOL_TCP", SOL_TCP, 6);
        hc!("SOL_UDP", SOL_UDP, 17);
        // ── INADDR_* (host byte order), signed 32-bit ──
        hc!(signed "INADDR_ANY", INADDR_ANY, ws::INADDR_ANY as i32);
        hc!(signed "INADDR_LOOPBACK", INADDR_LOOPBACK, ws::INADDR_LOOPBACK as i32);
        hc!(signed "INADDR_BROADCAST", INADDR_BROADCAST, ws::INADDR_BROADCAST as i32);
        hc!(signed "INADDR_NONE", INADDR_NONE, ws::INADDR_NONE as i32);
        hc!(signed "INADDR_ALLHOSTS_GROUP", INADDR_ALLHOSTS_GROUP, 0xe0000001u32 as i32);
        hc!(signed "INADDR_UNSPEC_GROUP", INADDR_UNSPEC_GROUP, 0xe0000000u32 as i32);
        hc!(signed "INADDR_MAX_LOCAL_GROUP", INADDR_MAX_LOCAL_GROUP, 0xe00000ffu32 as i32);
        hc!("IPPORT_RESERVED", IPPORT_RESERVED, ws::IPPORT_RESERVED);
        hc!("IPPORT_USERRESERVED", IPPORT_USERRESERVED, 5000);
        // ── SOL_* / SO_* (socket level) ──
        hc!("SOL_SOCKET", SOL_SOCKET, ws::SOL_SOCKET);
        hc!("SO_REUSEADDR", SO_REUSEADDR, ws::SO_REUSEADDR);
        hc!(
            "SO_EXCLUSIVEADDRUSE",
            SO_EXCLUSIVEADDRUSE,
            ws::SO_EXCLUSIVEADDRUSE
        );
        hc!("SO_KEEPALIVE", SO_KEEPALIVE, ws::SO_KEEPALIVE);
        hc!("SO_BROADCAST", SO_BROADCAST, ws::SO_BROADCAST);
        hc!("SO_DEBUG", SO_DEBUG, ws::SO_DEBUG);
        hc!("SO_DONTROUTE", SO_DONTROUTE, ws::SO_DONTROUTE);
        hc!("SO_LINGER", SO_LINGER, ws::SO_LINGER);
        hc!("SO_OOBINLINE", SO_OOBINLINE, ws::SO_OOBINLINE);
        hc!("SO_RCVBUF", SO_RCVBUF, ws::SO_RCVBUF);
        hc!("SO_SNDBUF", SO_SNDBUF, ws::SO_SNDBUF);
        hc!("SO_RCVTIMEO", SO_RCVTIMEO, ws::SO_RCVTIMEO);
        hc!("SO_SNDTIMEO", SO_SNDTIMEO, ws::SO_SNDTIMEO);
        hc!("SO_SNDLOWAT", SO_SNDLOWAT, 0x1003);
        hc!("SO_RCVLOWAT", SO_RCVLOWAT, 0x1004);
        hc!("SO_ERROR", SO_ERROR, ws::SO_ERROR);
        hc!("SO_TYPE", SO_TYPE, ws::SO_TYPE);
        hc!("SO_ACCEPTCONN", SO_ACCEPTCONN, ws::SO_ACCEPTCONN);
        hc!("SO_USELOOPBACK", SO_USELOOPBACK, ws::SO_USELOOPBACK);
        hc!("SO_ORIGINAL_DST", SO_ORIGINAL_DST, 12303);
        hc!("BTPROTO_RFCOMM", BTHPROTO_RFCOMM, bt::BTHPROTO_RFCOMM);
        hc!("SOL_RFCOMM", SOL_RFCOMM, 3);
        hc!("SO_BTH_ENCRYPT", SO_BTH_ENCRYPT, 2);
        hc!(signed "SO_BTH_MTU", SO_BTH_MTU, 0x80000007u32 as i32);
        hc!(signed "SO_BTH_MTU_MAX", SO_BTH_MTU_MAX, 0x80000008u32 as i32);
        hc!(signed "SO_BTH_MTU_MIN", SO_BTH_MTU_MIN, 0x8000000au32 as i32);
        // ── TCP-level ──
        hc!("TCP_NODELAY", TCP_NODELAY, ws::TCP_NODELAY);
        hc!("TCP_MAXSEG", TCP_MAXSEG, ws::TCP_MAXSEG);
        hc!("TCP_KEEPIDLE", TCP_KEEPIDLE, 3);
        hc!("TCP_FASTOPEN", TCP_FASTOPEN, 15);
        hc!("TCP_KEEPCNT", TCP_KEEPCNT, 16);
        hc!("TCP_KEEPINTVL", TCP_KEEPINTVL, 17);
        // SIO_TCP_SET_ACK_FREQUENCY, the name this option carries on Windows.
        // It is an ioctl code, published as a signed 32-bit word.
        hc!(signed "TCP_QUICKACK", SIO_TCP_SET_ACK_FREQUENCY, ws::SIO_TCP_SET_ACK_FREQUENCY as i32);
        // ── IP-level ──
        hc!("IP_TTL", IP_TTL, ws::IP_TTL);
        hc!("IP_TOS", IP_TOS, ws::IP_TOS);
        hc!("IP_OPTIONS", IP_OPTIONS, ws::IP_OPTIONS);
        hc!("IP_MULTICAST_TTL", IP_MULTICAST_TTL, ws::IP_MULTICAST_TTL);
        hc!(
            "IP_MULTICAST_LOOP",
            IP_MULTICAST_LOOP,
            ws::IP_MULTICAST_LOOP
        );
        hc!("IP_MULTICAST_IF", IP_MULTICAST_IF, ws::IP_MULTICAST_IF);
        hc!(
            "IP_ADD_MEMBERSHIP",
            IP_ADD_MEMBERSHIP,
            ws::IP_ADD_MEMBERSHIP
        );
        hc!(
            "IP_DROP_MEMBERSHIP",
            IP_DROP_MEMBERSHIP,
            ws::IP_DROP_MEMBERSHIP
        );
        hc!("IP_HDRINCL", IP_HDRINCL, ws::IP_HDRINCL);
        hc!("IP_RECVDSTADDR", IP_RECVDSTADDR, ws::IP_RECVDSTADDR);
        hc!("IP_ADD_SOURCE_MEMBERSHIP", IP_ADD_SOURCE_MEMBERSHIP, 15);
        hc!("IP_DROP_SOURCE_MEMBERSHIP", IP_DROP_SOURCE_MEMBERSHIP, 16);
        hc!("IP_BLOCK_SOURCE", IP_BLOCK_SOURCE, 17);
        hc!("IP_UNBLOCK_SOURCE", IP_UNBLOCK_SOURCE, 18);
        hc!("IP_PKTINFO", IP_PKTINFO, 19);
        hc!("IP_RECVTTL", IP_RECVTTL, 21);
        hc!("IP_RECVTOS", IP_RECVTOS, 40);
        hc!("IP_RECVERR", IP_RECVERR, 75);
        // ── IPv6 ──
        hc!("IPV6_V6ONLY", IPV6_V6ONLY, ws::IPV6_V6ONLY);
        hc!("IPV6_CHECKSUM", IPV6_CHECKSUM, ws::IPV6_CHECKSUM);
        hc!("IPV6_DONTFRAG", IPV6_DONTFRAG, ws::IPV6_DONTFRAG);
        hc!("IPV6_HOPLIMIT", IPV6_HOPLIMIT, ws::IPV6_HOPLIMIT);
        hc!("IPV6_HOPOPTS", IPV6_HOPOPTS, ws::IPV6_HOPOPTS);
        hc!("IPV6_JOIN_GROUP", IPV6_JOIN_GROUP, ws::IPV6_JOIN_GROUP);
        hc!("IPV6_LEAVE_GROUP", IPV6_LEAVE_GROUP, ws::IPV6_LEAVE_GROUP);
        hc!(
            "IPV6_MULTICAST_HOPS",
            IPV6_MULTICAST_HOPS,
            ws::IPV6_MULTICAST_HOPS
        );
        hc!(
            "IPV6_MULTICAST_IF",
            IPV6_MULTICAST_IF,
            ws::IPV6_MULTICAST_IF
        );
        hc!(
            "IPV6_MULTICAST_LOOP",
            IPV6_MULTICAST_LOOP,
            ws::IPV6_MULTICAST_LOOP
        );
        hc!("IPV6_PKTINFO", IPV6_PKTINFO, ws::IPV6_PKTINFO);
        hc!("IPV6_RECVRTHDR", IPV6_RECVRTHDR, ws::IPV6_RECVRTHDR);
        hc!("IPV6_RECVTCLASS", IPV6_RECVTCLASS, ws::IPV6_RECVTCLASS);
        hc!("IPV6_RECVERR", IPV6_RECVERR, 75);
        hc!("IPV6_RTHDR", IPV6_RTHDR, ws::IPV6_RTHDR);
        hc!("IPV6_TCLASS", IPV6_TCLASS, ws::IPV6_TCLASS);
        hc!(
            "IPV6_UNICAST_HOPS",
            IPV6_UNICAST_HOPS,
            ws::IPV6_UNICAST_HOPS
        );
        // ── shutdown how ──
        hc!("SHUT_RD", SD_RECEIVE, ws::SD_RECEIVE);
        hc!("SHUT_WR", SD_SEND, ws::SD_SEND);
        hc!("SHUT_RDWR", SD_BOTH, ws::SD_BOTH);
        // ── Message flags ──
        hc!("MSG_OOB", MSG_OOB, ws::MSG_OOB);
        hc!("MSG_PEEK", MSG_PEEK, ws::MSG_PEEK);
        hc!("MSG_DONTROUTE", MSG_DONTROUTE, ws::MSG_DONTROUTE);
        hc!("MSG_WAITALL", MSG_WAITALL, ws::MSG_WAITALL);
        hc!("MSG_CTRUNC", MSG_CTRUNC, ws::MSG_CTRUNC);
        hc!("MSG_TRUNC", MSG_TRUNC, ws::MSG_TRUNC);
        hc!("MSG_BCAST", MSG_BCAST, ws::MSG_BCAST);
        hc!("MSG_MCAST", MSG_MCAST, ws::MSG_MCAST);
        hc!("MSG_ERRQUEUE", MSG_ERRQUEUE, 0x1000);
        // ── Address-info flags ──
        hc!("AI_PASSIVE", AI_PASSIVE, ws::AI_PASSIVE);
        hc!("AI_CANONNAME", AI_CANONNAME, ws::AI_CANONNAME);
        hc!("AI_NUMERICHOST", AI_NUMERICHOST, ws::AI_NUMERICHOST);
        hc!("AI_NUMERICSERV", AI_NUMERICSERV, ws::AI_NUMERICSERV);
        hc!("AI_ADDRCONFIG", AI_ADDRCONFIG, ws::AI_ADDRCONFIG);
        hc!("AI_V4MAPPED", AI_V4MAPPED, ws::AI_V4MAPPED);
        hc!("AI_ALL", AI_ALL, ws::AI_ALL);
        // ── Name-info flags ──
        hc!("NI_NUMERICHOST", NI_NUMERICHOST, ws::NI_NUMERICHOST);
        hc!("NI_NUMERICSERV", NI_NUMERICSERV, ws::NI_NUMERICSERV);
        hc!("NI_NOFQDN", NI_NOFQDN, ws::NI_NOFQDN);
        hc!("NI_NAMEREQD", NI_NAMEREQD, ws::NI_NAMEREQD);
        hc!("NI_DGRAM", NI_DGRAM, ws::NI_DGRAM);
        hc!("NI_MAXHOST", NI_MAXHOST, ws::NI_MAXHOST);
        hc!("NI_MAXSERV", NI_MAXSERV, ws::NI_MAXSERV);
        // ── EAI_* — WSA error codes under their <ws2tcpip.h> aliases ──
        hc!("EAI_AGAIN", EAI_AGAIN, ws::WSATRY_AGAIN);
        hc!("EAI_BADFLAGS", EAI_BADFLAGS, ws::WSAEINVAL);
        hc!("EAI_FAIL", EAI_FAIL, ws::WSANO_RECOVERY);
        hc!("EAI_FAMILY", EAI_FAMILY, ws::WSAEAFNOSUPPORT);
        hc!("EAI_MEMORY", EAI_MEMORY, ws::WSA_NOT_ENOUGH_MEMORY);
        // EAI_NODATA is spelled EAI_NONAME in the header, so both name
        // WSAHOST_NOT_FOUND.
        hc!("EAI_NODATA", EAI_NODATA, ws::WSAHOST_NOT_FOUND);
        hc!("EAI_NONAME", EAI_NONAME, ws::WSAHOST_NOT_FOUND);
        hc!("EAI_SERVICE", EAI_SERVICE, ws::WSATYPE_NOT_FOUND);
        hc!("EAI_SOCKTYPE", EAI_SOCKTYPE, ws::WSAESOCKTNOSUPPORT);
        // ── WSAIoctl codes `socket.ioctl` names ──
        hc!("SIO_RCVALL", SIO_RCVALL, ws::SIO_RCVALL);
        hc!(
            "SIO_KEEPALIVE_VALS",
            SIO_KEEPALIVE_VALS,
            ws::SIO_KEEPALIVE_VALS
        );
        hc!(
            "SIO_LOOPBACK_FAST_PATH",
            SIO_LOOPBACK_FAST_PATH,
            ws::SIO_LOOPBACK_FAST_PATH
        );
        hc!("RCVALL_OFF", RCVALL_OFF, ws::RCVALL_OFF);
        hc!("RCVALL_ON", RCVALL_ON, ws::RCVALL_ON);
        hc!(
            "RCVALL_SOCKETLEVELONLY",
            RCVALL_SOCKETLEVELONLY,
            ws::RCVALL_SOCKETLEVELONLY
        );
        hc!("RCVALL_MAX", RCVALL_MAX, 3);
        // Hyper-V socket ABI. GUIDs and Bluetooth addresses are strings.
        hc!("HV_PROTOCOL_RAW", HV_PROTOCOL_RAW, 1);
        hc!("HVSOCKET_CONNECT_TIMEOUT", HVSOCKET_CONNECT_TIMEOUT, 1);
        hc!("HVSOCKET_CONNECTED_SUSPEND", HVSOCKET_CONNECTED_SUSPEND, 4);
        hc!(
            "HVSOCKET_CONNECT_TIMEOUT_MAX",
            HVSOCKET_CONNECT_TIMEOUT_MAX,
            300_000
        );
        hc!(
            "HVSOCKET_ADDRESS_FLAG_PASSTHRU",
            HVSOCKET_ADDRESS_FLAG_PASSTHRU,
            1
        );
        #[cfg(feature = "host_env")]
        let address_names = [
            ("BDADDR_ANY", rustpython_host_env::socket::BDADDR_ANY),
            ("BDADDR_LOCAL", rustpython_host_env::socket::BDADDR_LOCAL),
            ("HV_GUID_ZERO", rustpython_host_env::socket::HV_GUID_ZERO),
            (
                "HV_GUID_WILDCARD",
                rustpython_host_env::socket::HV_GUID_WILDCARD,
            ),
            (
                "HV_GUID_BROADCAST",
                rustpython_host_env::socket::HV_GUID_BROADCAST,
            ),
            (
                "HV_GUID_CHILDREN",
                rustpython_host_env::socket::HV_GUID_CHILDREN,
            ),
            (
                "HV_GUID_LOOPBACK",
                rustpython_host_env::socket::HV_GUID_LOOPBACK,
            ),
            (
                "HV_GUID_PARENT",
                rustpython_host_env::socket::HV_GUID_PARENT,
            ),
        ];
        #[cfg(not(feature = "host_env"))]
        let address_names = [
            ("BDADDR_ANY", "00:00:00:00:00:00"),
            ("BDADDR_LOCAL", "00:00:00:FF:FF:FF"),
            ("HV_GUID_ZERO", "00000000-0000-0000-0000-000000000000"),
            ("HV_GUID_WILDCARD", "00000000-0000-0000-0000-000000000000"),
            ("HV_GUID_BROADCAST", "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"),
            ("HV_GUID_CHILDREN", "90DB8B89-0D35-4F79-8CE9-49EA0AC8B7CD"),
            ("HV_GUID_LOOPBACK", "E0E16197-DD56-4A10-9195-5EE7A155A838"),
            ("HV_GUID_PARENT", "A42E7CDA-D03F-480C-9CC2-A4DE20ABB878"),
        ];
        for (name, value) in address_names {
            pyre_interpreter::__pyre_put_new!(ns_slot, name, pyre_object::w_str_new(value));
        }
        // `<winsock2.h>` SOMAXCONN is 0x7fffffff. The Winsock 1.1 value 5 is
        // not the one `listen` grew into.
        hc!("SOMAXCONN", SOMAXCONN, 0x7fff_ffffi64);
    }

    // ── htons / htonl / ntohs / ntohl ──
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "htons",
        pyre_interpreter::make_builtin_function_with_arity(
            "htons",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "htons() missing argument",
                    ));
                }
                let x = c_uint_converter(args[0], 0xffff, "uint16_t")? as u16;
                #[cfg(any(unix, windows))]
                let ordered = majit_rlib::rsocket::htons(x);
                #[cfg(not(any(unix, windows)))]
                let ordered = i64::from(x.to_be());
                Ok(pyre_object::w_int_new(ordered))
            },
            1,
        )
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "ntohs",
        pyre_interpreter::make_builtin_function_with_arity(
            "ntohs",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "ntohs() missing argument",
                    ));
                }
                let x = c_uint_converter(args[0], 0xffff, "uint16_t")? as u16;
                #[cfg(any(unix, windows))]
                let ordered = majit_rlib::rsocket::ntohs(x);
                #[cfg(not(any(unix, windows)))]
                let ordered = i64::from(u16::from_be(x));
                Ok(pyre_object::w_int_new(ordered))
            },
            1,
        )
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "htonl",
        pyre_interpreter::make_builtin_function_with_arity(
            "htonl",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "htonl() missing argument",
                    ));
                }
                let x = c_uint_converter(args[0], 0xffff_ffff, "uint32_t")? as u32;
                #[cfg(any(unix, windows))]
                let ordered = majit_rlib::rsocket::htonl(x);
                #[cfg(not(any(unix, windows)))]
                let ordered = i64::from(x.to_be());
                Ok(pyre_object::w_int_new(ordered))
            },
            1,
        )
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "ntohl",
        pyre_interpreter::make_builtin_function_with_arity(
            "ntohl",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "ntohl() missing argument",
                    ));
                }
                let x = c_uint_converter(args[0], 0xffff_ffff, "uint32_t")? as u32;
                #[cfg(any(unix, windows))]
                let ordered = majit_rlib::rsocket::ntohl(x);
                #[cfg(not(any(unix, windows)))]
                let ordered = i64::from(u32::from_be(x));
                Ok(pyre_object::w_int_new(ordered))
            },
            1,
        )
    );

    // ── inet_aton / inet_ntoa ──
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "inet_aton",
        pyre_interpreter::make_builtin_function_with_arity(
            "inet_aton",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "inet_aton() missing argument",
                    ));
                }
                let s = unsafe {
                    if !pyre_object::is_str(args[0]) {
                        return Err(pyre_interpreter::PyError::type_error(
                            "inet_aton: arg must be a string",
                        ));
                    }
                    pyre_interpreter::baseobjspace::str_utf8_w(args[0])?.to_string()
                };
                let c = std::ffi::CString::new(s.as_bytes()).map_err(|_| {
                    pyre_interpreter::PyError::value_error("embedded null in argument")
                })?;
                #[cfg(any(unix, windows))]
                let bytes = majit_rlib::rsocket::inet_aton(&c)
                    .map_err(|error| pyre_interpreter::PyError::os_error(error.message))?;
                #[cfg(not(any(unix, windows)))]
                let bytes = rffi::inet_aton(&c).ok_or_else(|| {
                    pyre_interpreter::PyError::os_error(
                        "illegal IP address string passed to inet_aton",
                    )
                })?;
                Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
            },
            1,
        )
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "inet_ntoa",
        pyre_interpreter::make_builtin_function_with_arity(
            "inet_ntoa",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "inet_ntoa() missing argument",
                    ));
                }
                let data = unsafe {
                    if !pyre_object::bytesobject::is_bytes_like(args[0]) {
                        return Err(pyre_interpreter::PyError::type_error(
                            "inet_ntoa: argument must be bytes-like",
                        ));
                    }
                    pyre_object::bytesobject::bytes_like_data(args[0])
                };
                #[cfg(any(unix, windows))]
                {
                    let text = majit_rlib::rsocket::inet_ntoa(data)
                        .map_err(|error| pyre_interpreter::PyError::os_error(error.message))?;
                    return Ok(pyre_object::w_str_new_managed(&text));
                }
                #[cfg(not(any(unix, windows)))]
                {
                    if data.len() != 4 {
                        return Err(pyre_interpreter::PyError::os_error(
                            "packed IP wrong length for inet_ntoa",
                        ));
                    }
                    let Some(text) = rffi::inet_ntoa([data[0], data[1], data[2], data[3]]) else {
                        return Err(pyre_interpreter::PyError::os_error("inet_ntoa failed"));
                    };
                    Ok(pyre_object::w_str_new_managed(&text))
                }
            },
            1,
        )
    );

    // inet_pton(af, ip) → bytes
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "inet_pton",
        pyre_interpreter::make_builtin_function_with_arity(
            "inet_pton",
            |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "inet_pton() requires 2 arguments",
                    ));
                }
                let w_af = args[0];
                let mut w_ip = args[1];
                let af = pyre_object::with_roots!(w_ip => pyre_interpreter::baseobjspace::c_int_w(w_af))?;
                let ip = unsafe {
                    if !pyre_object::is_str(w_ip) {
                        return Err(pyre_interpreter::PyError::type_error(
                            "inet_pton: address must be a string",
                        ));
                    }
                    pyre_interpreter::baseobjspace::str_utf8_w(w_ip)?.to_string()
                };
                let c_ip = std::ffi::CString::new(ip.as_bytes()).map_err(|_| {
                    pyre_interpreter::PyError::value_error("embedded null character")
                })?;
                #[cfg(any(unix, windows))]
                let packed =
                    majit_rlib::rsocket::inet_pton(af, &c_ip).map_err(|error| match error {
                        majit_rlib::rsocket::PtonError::Family(code) => {
                            pyre_interpreter::PyError::os_error_syscall(code, pyre_object::PY_NULL)
                        }
                        majit_rlib::rsocket::PtonError::Address => {
                            pyre_interpreter::PyError::os_error(
                                "illegal IP address string passed to inet_pton",
                            )
                        }
                    })?;
                #[cfg(not(any(unix, windows)))]
                let packed = rffi::pton(af, &c_ip).map_err(|error| match error {
                    rffi::PtonError::Family(code) => {
                        pyre_interpreter::PyError::os_error_syscall(code, pyre_object::PY_NULL)
                    }
                    rffi::PtonError::Address => pyre_interpreter::PyError::os_error(
                        "illegal IP address string passed to inet_pton",
                    ),
                })?;
                Ok(pyre_object::bytesobject::w_bytes_from_bytes(&packed))
            },
            2,
        )
    );

    // inet_ntop(af, packed) → str
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "inet_ntop",
        pyre_interpreter::make_builtin_function_with_arity(
            "inet_ntop",
            |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "inet_ntop() requires 2 arguments",
                    ));
                }
                let w_af = args[0];
                let mut w_packed = args[1];
                let af = pyre_object::with_roots!(w_packed => pyre_interpreter::baseobjspace::c_int_w(w_af))?;
                let data = unsafe {
                    if !pyre_object::bytesobject::is_bytes_like(w_packed) {
                        return Err(pyre_interpreter::PyError::type_error(
                            "inet_ntop: argument must be bytes-like",
                        ));
                    }
                    pyre_object::bytesobject::bytes_like_data(w_packed)
                };
                let expected = match af {
                    x if x == rffi::AF_INET => 4,
                    x if x == rffi::AF_INET6 => 16,
                    _ => {
                        return Err(pyre_interpreter::PyError::value_error(format!(
                            "unknown address family {af}"
                        )));
                    }
                };
                if data.len() != expected {
                    return Err(pyre_interpreter::PyError::value_error(
                        "invalid length of packed IP address string",
                    ));
                }
                #[cfg(any(unix, windows))]
                let text = majit_rlib::rsocket::inet_ntop(af, data)
                    .map_err(|_| pyre_interpreter::PyError::os_error("inet_ntop failed"))?;
                #[cfg(not(any(unix, windows)))]
                let text = rffi::ntop(af, data)
                    .ok_or_else(|| pyre_interpreter::PyError::os_error("inet_ntop failed"))?;
                Ok(pyre_object::w_str_new_managed(&text))
            },
            2,
        )
    );

    // gethostname() → str, over the name the host layer reports.  Where there
    // is none, `interp_socket_wasm` registers the one its `uname` answers.
    #[cfg(any(unix, windows))]
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "gethostname",
        pyre_interpreter::make_builtin_function_with_arity(
            "gethostname",
            |_| {
                #[cfg(unix)]
                let name = {
                    use std::os::unix::ffi::OsStringExt;
                    let bytes = majit_rlib::rsocket::gethostname().map_err(|error| {
                        pyre_interpreter::PyError::os_error_with_errno(error.errno, "gethostname")
                    })?;
                    std::ffi::OsString::from_vec(bytes)
                };
                #[cfg(windows)]
                let name = rffi::hostname().map_err(|e| {
                    pyre_interpreter::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        "gethostname",
                    )
                })?;
                // `interp_func.py:24` is
                // `space.fsdecode(space.newbytes(res))` -- the hostname is
                // opaque kernel bytes (`sethostname(2)` takes a plain
                // `const char*`), so a byte with no UTF-8 spelling has to
                // survive as its surrogate escape.
                Ok(pyre_interpreter::gateway::fsdecode_os_str(&name))
            },
            0,
        )
    );
    // `sethostname` alone stays POSIX-only: WinSock has no counterpart and
    // `moduledef.py` does not export it where rsocket cannot provide it.
    #[cfg(unix)]
    {
        // sethostname(name) → None
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "sethostname",
            pyre_interpreter::make_builtin_function_with_arity(
                "sethostname",
                |args| {
                    if args.is_empty() {
                        return Err(pyre_interpreter::PyError::type_error(
                            "sethostname() requires 1 argument",
                        ));
                    }
                    // `interp_func.py:405-411` tests the two accepted types in
                    // turn — a bytes name is taken as it stands, a str one is
                    // fsencoded — so a byte with no UTF-8 spelling reaches the
                    // syscall as itself, and anything else is a TypeError
                    // naming those two types.  `fsencode_w` would also accept a
                    // `__fspath__` object, which this entry point does not.
                    let mut w_name = args[0];
                    let name = unsafe {
                        if pyre_object::bytesobject::is_bytes(w_name) {
                            pyre_object::bytesobject::w_bytes_data(w_name).to_vec()
                        } else if pyre_object::is_str(w_name) {
                            let w_str = w_name;
                            pyre_object::with_roots!(w_name => pyre_interpreter::gateway::fsencode(w_str))?
                        } else {
                            return Err(pyre_interpreter::PyError::type_error(
                                "sethostname() argument 1 must be str or bytes",
                            ));
                        }
                    };
                    // `interp_func.py:412` audits the argument as it was
                    // passed, after the conversion and before the syscall.
                    pyre_interpreter::module::sys::vm::audit("socket.sethostname", &[w_name])?;
                    majit_rlib::rsocket::sethostname(&name).map_err(|error| {
                        let e = std::io::Error::from_raw_os_error(error.errno);
                        pyre_interpreter::PyError::os_error_with_errno(
                            error.errno,
                            format!("sethostname: {e}"),
                        )
                    })?;
                    Ok(pyre_object::w_none())
                },
                1,
            )
        );
    }

    // The legacy resolvers. Both platforms reach them through
    // `rsocket_rffi`'s accessors, because the records they answer with are
    // not the same type on each — see the netdb section there.
    #[cfg(any(unix, windows))]
    {
        // gethostbyname(name) → ip_string.  `interp_func.py` —
        // host argument runs through encode_idna (→ idna_converter)
        // before the rsocket call.
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "gethostbyname",
            pyre_interpreter::make_builtin_function_with_arity(
                "gethostbyname",
                |args| {
                    if args.is_empty() {
                        return Err(pyre_interpreter::PyError::type_error(
                            "gethostbyname() missing argument",
                        ));
                    }
                    let c = socket_idna_host_arg("gethostbyname", args[0])?;
                    // `rsocket.gethostbyname` is `makeipaddr(name, INETAddress())`
                    // and `socket_gethostbyname` is `setipaddr(name, AF_INET)`:
                    // both resolve, neither calls the `gethostbyname` of the
                    // same name, so a name that does not resolve reports
                    // `gaierror` rather than a message of this module's own.
                    let storage = resolve_ip_host(&c, rffi::AF_INET)?;
                    let sin = unsafe { &*(&storage as *const _ as *const rffi::sockaddr_in) };
                    let packed = rffi::sockaddr_in_get_addr(sin);
                    let text =
                        majit_rlib::rsocket::inet_ntoa(&packed.to_ne_bytes()).map_err(|_| {
                            socket_converted_error(
                                "error",
                                None,
                                "gethostbyname: address is not representable",
                            )
                        })?;
                    Ok(pyre_object::w_str_new_managed(&text))
                },
                1,
            )
        );

        // gethostbyname_ex(name) → (name, aliases, addresses)
        // `interp_func.py` — same lookup as gethostbyname but
        // returns the full hostent triple.
        // `__pyre_put_new!` stores only forwarded `ns_slot`
        // (`ShadowStackFrameworkGCTransformer.pop_roots`).
        let mut ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
        pyre_interpreter::__pyre_store!(
            ns,
            "gethostbyname_ex",
            pyre_interpreter::make_builtin_function_with_arity(
                "gethostbyname_ex",
                |args| {
                    if args.is_empty() {
                        return Err(pyre_interpreter::PyError::type_error(
                            "gethostbyname_ex() missing argument",
                        ));
                    }
                    let c = socket_idna_host_arg("gethostbyname_ex", args[0])?;
                    // `rsocket.gethostbyname_ex` resolves the name first and
                    // only then reads the `hostent`, so an unresolvable name
                    // reports `gaierror` and a name the resolver knows but the
                    // host database does not reports `herror`.
                    resolve_ip_host(&c, rffi::AF_INET)?;
                    let _netdb = rffi::netdb_lock();
                    let he = unsafe { rffi::host_by_name(c.as_ptr()) };
                    if he.is_null() {
                        return Err(host_lookup_error());
                    }
                    unpack_hostent(he)
                },
                1,
            )
        );

        // gethostbyaddr(addr) → (name, aliases, addresses)
        // `interp_func.py` — reverse lookup; `addr` is an
        // IPv4/IPv6 string we resolve through inet_pton, then feed
        // to gethostbyaddr.
        pyre_interpreter::__pyre_store!(
            ns,
            "gethostbyaddr",
            pyre_interpreter::make_builtin_function_with_arity(
                "gethostbyaddr",
                |args| {
                    if args.is_empty() {
                        return Err(pyre_interpreter::PyError::type_error(
                            "gethostbyaddr() missing argument",
                        ));
                    }
                    let c = socket_idna_host_arg("gethostbyaddr", args[0])?;
                    // `rsocket.gethostbyaddr` is `makeipaddr(ip)` — the
                    // argument is resolved for whichever family answers, not
                    // required to be numeric — and only the reverse lookup that
                    // follows reports through `h_errno`.
                    let storage = resolve_ip_host(&c, rffi::AF_UNSPEC)?;
                    let family = storage.ss_family as libc::c_int;
                    // `gethostbyaddr` takes the bare address bytes, not the
                    // whole `sockaddr`, and the accessors are what read them on
                    // both platforms — WinSock wraps them in anonymous unions.
                    let mut raw = [0u8; 16];
                    let addr_len = if family == rffi::AF_INET6 {
                        let sin6 = unsafe { &*(&storage as *const _ as *const rffi::sockaddr_in6) };
                        raw = rffi::sockaddr_in6_get_addr(sin6);
                        16 as rffi::SockLen
                    } else {
                        let sin = unsafe { &*(&storage as *const _ as *const rffi::sockaddr_in) };
                        raw[..4].copy_from_slice(&rffi::sockaddr_in_get_addr(sin).to_ne_bytes());
                        4 as rffi::SockLen
                    };
                    let _netdb = rffi::netdb_lock();
                    let he = unsafe {
                        rffi::host_by_addr(raw.as_ptr() as *const libc::c_void, addr_len, family)
                    };
                    if he.is_null() {
                        return Err(host_lookup_error());
                    }
                    unpack_hostent(he)
                },
                1,
            )
        );

        // getservbyname(name[, proto]) → port
        pyre_interpreter::__pyre_store!(
            ns,
            "getservbyname",
            pyre_interpreter::make_builtin_function("getservbyname", |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getservbyname() missing argument",
                    ));
                }
                let w_name = args[0];
                let mut w_proto = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                let name = unsafe {
                    if !pyre_object::is_str(w_name) {
                        return Err(pyre_interpreter::PyError::type_error(
                            "getservbyname: name must be a string",
                        ));
                    }
                    pyre_object::with_roots!(w_proto => pyre_interpreter::baseobjspace::str_utf8_w(w_name))?
                        .to_string()
                };
                let c_name = std::ffi::CString::new(name.as_bytes())
                    .map_err(|_| pyre_interpreter::PyError::value_error("embedded null"))?;
                let proto_c: Option<std::ffi::CString> = if args.len() >= 2
                    && unsafe { pyre_object::is_str(w_proto) }
                {
                    let p = pyre_interpreter::baseobjspace::str_utf8_w(w_proto)?.to_string();
                    Some(
                        std::ffi::CString::new(p.as_bytes())
                            .map_err(|_| pyre_interpreter::PyError::value_error("embedded null"))?,
                    )
                } else {
                    None
                };
                #[cfg(unix)]
                {
                    let port = majit_rlib::rsocket::getservbyname(&c_name, proto_c.as_deref())
                        .map_err(|error| socket_converted_error("error", None, error.message))?;
                    return Ok(pyre_object::w_int_new(port));
                }
                #[cfg(windows)]
                {
                    let p = unsafe {
                        rffi::serv_by_name(
                            c_name.as_ptr(),
                            proto_c
                                .as_ref()
                                .map(|c| c.as_ptr())
                                .unwrap_or(std::ptr::null()),
                        )
                    };
                    if p.is_null() {
                        return Err(socket_converted_error(
                            "error",
                            None,
                            "service/proto not found",
                        ));
                    }
                    let port = unsafe { u16::from_be(rffi::servent_port(p)) };
                    Ok(pyre_object::w_int_new(port as i64))
                }
            })
        );

        // getservbyport(port[, proto]) → name
        pyre_interpreter::__pyre_store!(
            ns,
            "getservbyport",
            pyre_interpreter::make_builtin_function("getservbyport", |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getservbyport() missing argument",
                    ));
                }
                // `getservbyport` rejects a port outside the range before it
                // narrows: taking the low sixteen bits of 70000 would look up
                // 4464 and answer for it.  The clinic spells the parameter
                // `int port`, so an argument that is not one owes a TypeError
                // rather than reading its first word as the port.
                let w_port = args[0];
                let mut w_proto = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                let port = pyre_object::with_roots!(w_proto =>
                    pyre_interpreter::baseobjspace::gateway_int_w(w_port)
                )?;
                if !(0..=0xffff).contains(&port) {
                    return Err(pyre_interpreter::PyError::overflow_error(
                        "getservbyport: port must be 0-65535.",
                    ));
                }
                let port = port as u16;
                let proto_c: Option<std::ffi::CString> = if args.len() >= 2
                    && unsafe { pyre_object::is_str(w_proto) }
                {
                    let p = pyre_interpreter::baseobjspace::str_utf8_w(w_proto)?.to_string();
                    Some(
                        std::ffi::CString::new(p.as_bytes())
                            .map_err(|_| pyre_interpreter::PyError::value_error("embedded null"))?,
                    )
                } else {
                    None
                };
                #[cfg(unix)]
                {
                    let name =
                        majit_rlib::rsocket::getservbyport(i32::from(port), proto_c.as_deref())
                            .map_err(|error| {
                                socket_converted_error("error", None, error.message)
                            })?;
                    return Ok(pyre_object::w_str_new_managed(&name));
                }
                #[cfg(windows)]
                {
                    let p = unsafe {
                        rffi::serv_by_port(
                            port.to_be() as libc::c_int,
                            proto_c
                                .as_ref()
                                .map(|c| c.as_ptr())
                                .unwrap_or(std::ptr::null()),
                        )
                    };
                    if p.is_null() {
                        return Err(socket_converted_error(
                            "error",
                            None,
                            "port/proto not found",
                        ));
                    }
                    let name = unsafe {
                        std::ffi::CStr::from_ptr(rffi::servent_name(p))
                            .to_string_lossy()
                            .into_owned()
                    };
                    Ok(pyre_object::w_str_new_managed(&name))
                }
            })
        );
    }

    // `moduledef.py:12-16`:
    //   error    = get_error(space, "error")
    //   herror   = get_error(space, "herror")
    //   gaierror = get_error(space, "gaierror")
    //   timeout  = space.w_TimeoutError
    // `socketmodule.c` names them `socket.herror` / `socket.gaierror`
    // instead, and `type.__module__` reads the qualified prefix back, so
    // `socket.gaierror.__module__` is `"socket"` rather than `"_socket"`.
    // `new_exception_class` leaves the base's `__new__` in place, so both
    // inherit the OSError family constructor and its errno/strerror parse.
    let mut w_os_error = pyre_interpreter::builtins::lookup_exc_class("OSError")
        .expect("OSError must be installed before _socket init");
    let err_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_os_error);
    let mut ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    pyre_interpreter::__pyre_store!(
        ns,
        "error",
        pyre_object::gc_roots::shadow_stack_get(err_slot)
    );
    w_os_error = pyre_object::gc_roots::shadow_stack_get(err_slot);
    let mut w_herror = pyre_object::with_roots!(ns, w_os_error => pyre_interpreter::builtins::new_exception_class(
        "socket.herror",
        pyre_interpreter::builtins::exc_os_error_new,
        w_os_error,
    ));
    pyre_interpreter::__pyre_store!(ns, "herror", w_herror);
    // The herror store can forward `w_os_error` in `err_slot` without
    // rewriting this local. Reload before the next constructor
    // (`ShadowStackFrameworkGCTransformer.pop_roots`).
    w_os_error = pyre_object::gc_roots::shadow_stack_get(err_slot);
    let mut w_gaierror = pyre_object::with_roots!(ns, w_os_error => pyre_interpreter::builtins::new_exception_class(
        "socket.gaierror",
        pyre_interpreter::builtins::exc_os_error_new,
        w_os_error,
    ));
    pyre_interpreter::__pyre_store!(ns, "gaierror", w_gaierror);
    let mut w_timeout_error = pyre_interpreter::builtins::lookup_exc_class("TimeoutError")
        .expect("TimeoutError must be installed before _socket init");
    pyre_interpreter::__pyre_store!(ns, "timeout", w_timeout_error);

    // Default timeout (None) — modulus has a getter/setter; we just stash
    // a None so attribute lookups succeed.
    pyre_interpreter::__pyre_store!(ns, "_default_timeout", pyre_object::w_none());

    // `_rsocket_rffi.py constants['has_ipv6'] = True` — exposed by
    // PyPy's moduledef.py constants loop as a module-level boolean.  It
    // reports the runtime's support for the family, not the header's number
    // for it, so a target with no socket layer answers false while still
    // carrying `AF_INET6`.
    pyre_interpreter::__pyre_store!(
        ns,
        "has_ipv6",
        pyre_object::boolobject::w_bool_from(cfg!(any(unix, windows)))
    );

    // ── module-level getdefaulttimeout / setdefaulttimeout ──
    // `interp_func.py` — None means "blocking", float means
    // "timeout in seconds".  Stored as a process-wide cell.
    pyre_interpreter::__pyre_store!(
        ns,
        "getdefaulttimeout",
        pyre_interpreter::make_builtin_function_with_arity(
            "getdefaulttimeout",
            |_| Ok(get_default_socket_timeout()),
            0,
        )
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "setdefaulttimeout",
        pyre_interpreter::make_builtin_function_with_arity(
            "setdefaulttimeout",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "setdefaulttimeout() missing argument",
                    ));
                }
                let v = args[0];
                if unsafe { pyre_object::is_none(v) } {
                    set_default_socket_timeout(None);
                    return Ok(pyre_object::w_none());
                }
                let secs = unsafe {
                    if pyre_object::is_int(v) {
                        pyre_object::w_int_get_value(v) as f64
                    } else if pyre_object::is_float(v) {
                        pyre_object::floatobject::w_float_get_value(v)
                    } else {
                        return Err(pyre_interpreter::PyError::type_error(
                            "setdefaulttimeout: value must be a float or None",
                        ));
                    }
                };
                if secs < 0.0 || !secs.is_finite() {
                    return Err(pyre_interpreter::PyError::value_error(
                        "Timeout value out of range",
                    ));
                }
                set_default_socket_timeout(Some(secs));
                Ok(pyre_object::w_none())
            },
            1,
        )
    );

    // ── module-level close(fd) ──
    // `interp_socket.py:close(fd)` — the bare host close, used for
    // cleanup when callers obtain a descriptor via .detach().
    #[cfg(any(unix, windows))]
    pyre_interpreter::__pyre_store!(
        ns,
        "close",
        pyre_interpreter::make_builtin_function_with_arity(
            "close",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error("close() missing fd"));
                }
                if !unsafe { pyre_object::is_int(args[0]) } {
                    return Err(pyre_interpreter::PyError::type_error(
                        "close: fd must be an integer",
                    ));
                }
                let fd = rffi::socket_from_i64(unsafe { pyre_object::w_int_get_value(args[0]) });
                #[cfg(unix)]
                majit_rlib::rsocket::close(fd).map_err(rsocket_os_error)?;
                #[cfg(windows)]
                if unsafe { rffi::close(fd) } != 0 {
                    return Err(socket_last_error());
                }
                Ok(pyre_object::w_none())
            },
            1,
        )
    );

    // ── getprotobyname(name) ──
    // `interp_func.py` — returns the IPPROTO_* number for a
    // protocol name.  libc getprotobyname returns NULL on lookup
    // failure; we surface that as OSError to match `converted_error`.
    #[cfg(any(unix, windows))]
    pyre_interpreter::__pyre_store!(
        ns,
        "getprotobyname",
        pyre_interpreter::make_builtin_function_with_arity(
            "getprotobyname",
            |args| {
                if args.is_empty() || !unsafe { pyre_object::is_str(args[0]) } {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getprotobyname: name must be a string",
                    ));
                }
                let name = pyre_interpreter::baseobjspace::str_utf8_w(args[0])?.to_string();
                let c_name = std::ffi::CString::new(name.as_bytes())
                    .map_err(|_| pyre_interpreter::PyError::value_error("embedded null in name"))?;
                #[cfg(unix)]
                let proto = majit_rlib::rsocket::getprotobyname(&c_name)
                    .map_err(|error| socket_converted_error("error", None, error.message))?;
                #[cfg(windows)]
                let proto = {
                    let Some(proto) = rffi::protocol_by_name(&c_name) else {
                        return Err(socket_converted_error("error", None, "protocol not found"));
                    };
                    i64::from(proto)
                };
                Ok(pyre_object::w_int_new(proto))
            },
            1,
        )
    );

    // ── if_nameindex / if_nametoindex / if_indextoname ──
    // `interp_socket.py:if_nameindex|if_nametoindex|if_indextoname`
    // — direct wrappers around libc's network-interface accessors.
    #[cfg(unix)]
    {
        pyre_interpreter::__pyre_store!(
            ns,
            "if_nameindex",
            pyre_interpreter::make_builtin_function_with_arity(
                "if_nameindex",
                |_| {
                    let interfaces = majit_rlib::rsocket::if_nameindex().map_err(|error| {
                        socket_io_err(std::io::Error::from_raw_os_error(error.errno))
                    })?;
                    let mut result_w = pyre_object::gc_roots::RootedItems::new();
                    for (index, name) in interfaces {
                        let pair = {
                            let mut fields = pyre_object::gc_roots::RootedItems::new();
                            fields.push(pyre_object::w_int_new(index as i64));
                            // Interface names are OS bytes: preserve fsencode/fsdecode
                            // round trips, including non-UTF-8 names.
                            fields.push(pyre_interpreter::gateway::fsdecode_filename_bytes(&name));
                            pyre_object::w_tuple_new(fields.take())
                        };
                        result_w.push(pair);
                    }
                    Ok(pyre_object::w_list_new(result_w.take()))
                },
                0,
            )
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "if_nametoindex",
            pyre_interpreter::make_builtin_function_with_arity(
                "if_nametoindex",
                |args| {
                    if args.is_empty() {
                        return Err(pyre_interpreter::PyError::type_error(
                            "if_nametoindex() requires 1 argument",
                        ));
                    }
                    // An interface name is an OS string.
                    // `interp_socket.py` declares `name='text'` and
                    // compares against `rsocket.if_nameindex()`'s own names;
                    // `socketmodule.c socket_if_nametoindex` instead reads it
                    // with `PyUnicode_FSConverter`, the filesystem encoding
                    // `fsencode_w` applies, and accepts the same
                    // str / bytes / `__fspath__` argument.  Take the 3.14
                    // spelling, which is also what makes this round-trip with
                    // the `if_nameindex` / `if_indextoname` decode above.
                    let name = pyre_interpreter::gateway::fsencode_bytes_w(args[0])?;
                    let c_name = std::ffi::CString::new(name).map_err(|_| {
                        pyre_interpreter::PyError::value_error("embedded null in name")
                    })?;
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = c_name;
                        return Err(pyre_interpreter::PyError::not_implemented(
                            "socket.if_nametoindex requires host_env",
                        ));
                    }
                    #[cfg(feature = "host_env")]
                    {
                        let idx = rustpython_host_env::socket::if_nametoindex_checked(&c_name)
                            .map_err(socket_io_err)?;
                        Ok(pyre_object::w_int_new(idx as i64))
                    }
                },
                1,
            )
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "if_indextoname",
            pyre_interpreter::make_builtin_function_with_arity(
                "if_indextoname",
                |args| {
                    // PyPy `if_indextoname` receives an unsigned interface
                    // index through its gateway.  3.14 exposes the
                    // negative leaf as ValueError (`testInvalidInterfaceIndexToName`),
                    // while PyPy spells it OverflowError; the public 3.14
                    // result governs this conversion, with the same unsigned
                    // storage owner as PyPy.
                    let w_index = pyre_interpreter::baseobjspace::space_index(args[0])?;
                    let index = pyre_interpreter::builtins::space_index_w(w_index)?;
                    if index < 0 {
                        return Err(pyre_interpreter::PyError::value_error(
                            "Cannot convert negative int",
                        ));
                    }
                    let idx = libc::c_uint::try_from(index).map_err(|_| {
                        pyre_interpreter::PyError::overflow_error(
                            "Python int too large for C unsigned int",
                        )
                    })?;
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = idx;
                        return Err(pyre_interpreter::PyError::not_implemented(
                            "socket.if_indextoname requires host_env",
                        ));
                    }
                    #[cfg(feature = "host_env")]
                    {
                        let name = rustpython_host_env::socket::if_indextoname_checked(idx)
                            .map_err(socket_io_err)?
                            .into_bytes();
                        Ok(pyre_interpreter::gateway::fsdecode_filename_bytes(&name))
                    }
                },
                1,
            )
        );
    }

    // Windows has the same public trio but no POSIX `if_nameindex` array.
    // `rustpython_host_env::socket` follows the same IP Helper API route:
    // `GetIfTable2Ex` for enumeration and the WinSock conversion calls for
    // the two scalar directions.
    #[cfg(all(windows, feature = "host_env"))]
    {
        pyre_interpreter::__pyre_store!(
            ns,
            "if_nameindex",
            pyre_interpreter::make_builtin_function_with_arity(
                "if_nameindex",
                |_| windows_if_nameindex(),
                0,
            )
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "if_nametoindex",
            pyre_interpreter::make_builtin_function_with_arity(
                "if_nametoindex",
                |args| {
                    let name = pyre_interpreter::gateway::fsencode_bytes_w(args[0])?;
                    let name = std::ffi::CString::new(name).map_err(|_| {
                        pyre_interpreter::PyError::value_error("embedded null in name")
                    })?;
                    let index = rustpython_host_env::socket::if_nametoindex_checked(&name)
                        .map_err(interface_io_error)?;
                    Ok(pyre_object::w_int_new(index as i64))
                },
                1,
            )
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "if_indextoname",
            pyre_interpreter::make_builtin_function_with_arity(
                "if_indextoname",
                |args| {
                    // `socket_if_indextoname`'s NET_IFINDEX converter reads
                    // `__index__` first, rejects a negative value, and only
                    // then the ones no interface index can hold.  An argument
                    // wider than a machine word has to reach those same two
                    // answers rather than a conversion error of its own, so
                    // take its sign from the object when the word conversion
                    // is the thing that fails.
                    let w_index = pyre_interpreter::baseobjspace::space_index(args[0])?;
                    let word = pyre_interpreter::builtins::space_index_w(w_index).ok();
                    let negative = match word {
                        Some(index) => index < 0,
                        // Only a long fails to fit a machine word.
                        None => unsafe {
                            pyre_object::longobject::jit_bigint_sign_i64(
                                pyre_object::longobject::w_long_get_value(w_index),
                            ) < 0
                        },
                    };
                    if negative {
                        return Err(pyre_interpreter::PyError::value_error(
                            "Cannot convert negative int",
                        ));
                    }
                    let index = word
                        .and_then(|index| u32::try_from(index).ok())
                        .ok_or_else(|| {
                            pyre_interpreter::PyError::overflow_error(
                                "Python int too large for C NET_IFINDEX",
                            )
                        })?;
                    let name = rustpython_host_env::socket::if_indextoname_checked(index)
                        .map_err(interface_io_error)?;
                    Ok(pyre_object::w_str_new_managed(&name))
                },
                1,
            )
        );
    }

    // ── CMSG_SPACE / CMSG_LEN ──
    // `interp_func.py` — POSIX macros, exposed only when the
    // host libc has them.  rust's `libc` crate provides both on every
    // unix target we ship, so we register them under the same cfg.
    #[cfg(unix)]
    {
        pyre_interpreter::__pyre_store!(
            ns,
            "CMSG_SPACE",
            pyre_interpreter::make_builtin_function_with_arity(
                "CMSG_SPACE",
                |args| {
                    let raw = pyre_interpreter::builtins::space_index_w(
                        pyre_interpreter::baseobjspace::space_index(args[0])?,
                    )?;
                    #[cfg(all(feature = "host_env", not(target_os = "redox")))]
                    let n = usize::try_from(raw)
                        .ok()
                        .and_then(rustpython_host_env::socket::checked_cmsg_space)
                        .ok_or_else(|| {
                            pyre_interpreter::PyError::overflow_error(
                                "CMSG_SPACE() argument out of range",
                            )
                        })?;
                    #[cfg(any(not(feature = "host_env"), target_os = "redox"))]
                    let n = {
                        let max_payload =
                            i64::from(libc::c_int::MAX) - i64::from(unsafe { libc::CMSG_SPACE(1) });
                        if raw < 0 || raw > max_payload {
                            return Err(pyre_interpreter::PyError::overflow_error(
                                "CMSG_SPACE() argument out of range",
                            ));
                        }
                        let n = unsafe { libc::CMSG_SPACE(raw as libc::c_uint) };
                        if n == 0 || u64::from(n) > libc::c_int::MAX as u64 {
                            return Err(pyre_interpreter::PyError::overflow_error(
                                "CMSG_SPACE() argument out of range",
                            ));
                        }
                        n as usize
                    };
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            )
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "CMSG_LEN",
            pyre_interpreter::make_builtin_function_with_arity(
                "CMSG_LEN",
                |args| {
                    let raw = pyre_interpreter::builtins::space_index_w(
                        pyre_interpreter::baseobjspace::space_index(args[0])?,
                    )?;
                    #[cfg(all(feature = "host_env", not(target_os = "redox")))]
                    let n = usize::try_from(raw)
                        .ok()
                        .and_then(rustpython_host_env::socket::checked_cmsg_len)
                        .ok_or_else(|| {
                            pyre_interpreter::PyError::overflow_error(
                                "CMSG_LEN() argument out of range",
                            )
                        })?;
                    #[cfg(any(not(feature = "host_env"), target_os = "redox"))]
                    let n = {
                        if raw < 0 || raw > i64::from(libc::c_int::MAX) {
                            return Err(pyre_interpreter::PyError::overflow_error(
                                "CMSG_LEN() argument out of range",
                            ));
                        }
                        let n = unsafe { libc::CMSG_LEN(raw as libc::c_uint) };
                        if n == 0 || u64::from(n) > libc::c_int::MAX as u64 {
                            return Err(pyre_interpreter::PyError::overflow_error(
                                "CMSG_LEN() argument out of range",
                            ));
                        }
                        n as usize
                    };
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            )
        );
    }

    // ── getaddrinfo / getnameinfo ──
    // `interp_func.py` (getaddrinfo) and `:137-156`
    // (getnameinfo) — directly wrap the host's getaddrinfo / getnameinfo
    // and walk the addrinfo linked list.
    #[cfg(any(unix, windows))]
    init_socket_getaddrinfo(ns);

    // ── socket class (slice S2) ──
    #[cfg(any(unix, windows))]
    {
        let socket_tp = socket_type();
        // Expose the type itself as `socket` AND `SocketType` so the
        // stdlib's `class socket(_socket.socket):` pattern works.
        let socket_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(socket_tp);
        ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
        pyre_interpreter::__pyre_store!(
            ns,
            "socket",
            pyre_object::gc_roots::shadow_stack_get(socket_slot)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "SocketType",
            pyre_object::gc_roots::shadow_stack_get(socket_slot)
        );
    }
    // The same two names, and the numbers, where there is no host layer to
    // build the rest of the module out of.
    #[cfg(not(any(unix, windows)))]
    super::interp_socket_wasm::register_names(ns);

    // `socket.py` defines `socketpair` only when `_socket` carries one and
    // falls back to `_fallback_socketpair`'s own AF_INET pair otherwise, and
    // `fromfd` is built out of `dup` at app level everywhere else.  Both stay
    // with the POSIX calls they are made of; `dup` itself is registered for
    // Windows too, further down, out of the WinSock calls that stand in for
    // it.
    #[cfg(unix)]
    {
        // socketpair(family=AF_UNIX, type=SOCK_STREAM, proto=0)
        pyre_interpreter::__pyre_store!(
            ns,
            "socketpair",
            pyre_interpreter::make_builtin_function("socketpair", |args| {
                for (idx, label) in [(0, "family"), (1, "type"), (2, "proto")] {
                    if args.len() > idx && !unsafe { pyre_object::is_int(args[idx]) } {
                        return Err(pyre_interpreter::PyError::type_error(format!(
                            "socketpair: {label} must be an integer"
                        )));
                    }
                }
                let family = if args.is_empty() {
                    libc::AF_UNIX
                } else {
                    unsafe { pyre_object::w_int_get_value(args[0]) as libc::c_int }
                };
                let ty = if args.len() < 2 {
                    rffi::SOCK_STREAM
                } else {
                    unsafe { pyre_object::w_int_get_value(args[1]) as libc::c_int }
                };
                let proto = if args.len() < 3 {
                    0
                } else {
                    unsafe { pyre_object::w_int_get_value(args[2]) as libc::c_int }
                };
                let (fd0, fd1) =
                    majit_rlib::rsocket::socketpair(family, ty, proto).map_err(|error| {
                        socket_io_err(std::io::Error::from_raw_os_error(error.errno))
                    })?;
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(socket_from_fd(fd0, family, ty, proto)?);
                fields.push(socket_from_fd(fd1, family, ty, proto)?);
                Ok(pyre_object::w_tuple_new(fields.take()))
            })
        );

        // dup(fd) → new fd. The duplicated descriptor has FD_CLOEXEC set.
        pyre_interpreter::__pyre_store!(
            ns,
            "dup",
            pyre_interpreter::make_builtin_function_with_arity(
                "dup",
                |args| {
                    if args.is_empty() {
                        return Err(pyre_interpreter::PyError::type_error(
                            "dup() missing argument",
                        ));
                    }
                    if !unsafe { pyre_object::is_int(args[0]) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "dup: fd must be an integer",
                        ));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                    let n = majit_rlib::rsocket::dup(fd).map_err(|error| {
                        socket_io_err(std::io::Error::from_raw_os_error(error.errno))
                    })?;
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            )
        );

        // fromfd(fd, family, type, proto=0) — `interp_func.py:75
        // fromfd_w`: dup() the supplied fd (so the caller still owns the
        // original) and wrap it in a fresh `_socket.socket`.  CPython
        // requires the dup so close() on the returned socket leaves the
        // input descriptor intact.
        pyre_interpreter::__pyre_store!(
            ns,
            "fromfd",
            pyre_interpreter::make_builtin_function("fromfd", |args| {
                if args.len() < 3 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "fromfd() requires fd, family and type",
                    ));
                }
                for (idx, label) in [(0, "fd"), (1, "family"), (2, "type")] {
                    if !unsafe { pyre_object::is_int(args[idx]) } {
                        return Err(pyre_interpreter::PyError::type_error(format!(
                            "fromfd: {label} must be an integer"
                        )));
                    }
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let family = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
                let ty = (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int;
                let proto = if args.len() >= 4 {
                    if !unsafe { pyre_object::is_int(args[3]) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "fromfd: proto must be an integer",
                        ));
                    }
                    (unsafe { pyre_object::w_int_get_value(args[3]) }) as libc::c_int
                } else {
                    0
                };
                let new_fd = majit_rlib::rsocket::dup(fd).map_err(|error| {
                    socket_io_err(std::io::Error::from_raw_os_error(error.errno))
                })?;
                socket_from_fd(new_fd, family, ty, proto)
            })
        );
    }

    // `socket.py`'s `socket.dup` and `fromfd` both go through
    // `_socket.dup`, which on Windows cannot duplicate a descriptor in place:
    // a socket is not a C runtime file descriptor there.  `socket_dup` hands
    // the socket to the process it is already in with `WSADuplicateSocketW`
    // and re-opens it from the protocol info that writes, which is the pair of
    // calls `share_socket` / `socket_from_share_data` make.  The new socket is
    // non-inheritable, as PEP 446 asks and as `socket_from_share_data` leaves
    // it.
    #[cfg(all(windows, feature = "host_env"))]
    pyre_interpreter::__pyre_store!(
        ns,
        "dup",
        pyre_interpreter::make_builtin_function_with_arity(
            "dup",
            |args| {
                let fd = pyre_interpreter::builtins::space_index_w(args[0])?;
                let share = rustpython_host_env::socket::share_socket(fd as _, std::process::id())
                    .map_err(socket_io_err)?;
                let shared = rustpython_host_env::socket::socket_from_share_data(&share)
                    .map_err(socket_io_err)?;
                Ok(pyre_object::w_int_new(shared.raw as i64))
            },
            1,
        )
    );
    Ok(())
}

// ── hostent → (name, aliases, addrs) ──
// `interp_func.py common_wrapgethost` — packs a resolver hostent
// into the 3-tuple shape used by gethostbyname_ex / gethostbyaddr.
#[cfg(any(unix, windows))]
fn unpack_hostent(
    he: *mut rffi::Hostent,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    unsafe {
        let host_name = rffi::hostent_name(he);
        let name = if host_name.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(host_name)
                .to_string_lossy()
                .into_owned()
        };
        // Copy all resolver-owned storage while the process-global netdb lock
        // is held, before allocating any Python objects.
        let aliases = rffi::hostent_aliases(he);
        let mut alias_strings = Vec::new();
        if !aliases.is_null() {
            let mut index = 0;
            loop {
                let alias = rffi::pointer_at(aliases, index);
                if alias.is_null() {
                    break;
                }
                alias_strings.push(
                    std::ffi::CStr::from_ptr(alias)
                        .to_string_lossy()
                        .into_owned(),
                );
                index += 1;
            }
        }
        let addr_list = rffi::hostent_addr_list(he);
        let addr_type = rffi::hostent_addr_type(he);
        let addr_length = rffi::hostent_length(he);
        let mut addr_strings = Vec::new();
        if !addr_list.is_null() {
            let mut index = 0;
            loop {
                let addr_ptr = rffi::pointer_at(addr_list, index);
                if addr_ptr.is_null() {
                    break;
                }
                let addr_str = if addr_type == rffi::AF_INET && addr_length == 4 {
                    let packed = std::ptr::read_unaligned(addr_ptr as *const u32).to_ne_bytes();
                    majit_rlib::rsocket::inet_ntoa(&packed).unwrap_or_default()
                } else if addr_type == rffi::AF_INET6 && addr_length == 16 {
                    let mut packed_addr = [0u8; 16];
                    std::ptr::copy_nonoverlapping(
                        addr_ptr as *const u8,
                        packed_addr.as_mut_ptr(),
                        packed_addr.len(),
                    );
                    majit_rlib::rsocket::inet_ntop(rffi::AF_INET6, &packed_addr).unwrap_or_default()
                } else {
                    String::new()
                };
                addr_strings.push(addr_str);
                index += 1;
            }
        }
        // `w_list_new` roots the items it is handed, not the header it returns,
        // and that header is a movable nursery object (`rlist.py:116 LIST =
        // GcStruct`). Each list therefore stays pinned across the allocation
        // that follows it and is re-read from its slot afterwards.
        let _headers = pyre_object::gc_roots::push_roots();
        let aliases = {
            let mut items = pyre_object::gc_roots::RootedItems::new();
            for alias in &alias_strings {
                items.push(pyre_object::w_str_new_managed(alias));
            }
            pyre_object::w_list_new(items.take())
        };
        let aliases_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(aliases);
        let addrs = {
            let mut items = pyre_object::gc_roots::RootedItems::new();
            for addr in &addr_strings {
                items.push(pyre_object::w_str_new_managed(addr));
            }
            pyre_object::w_list_new(items.take())
        };
        let addrs_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(addrs);
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(pyre_object::w_str_new_managed(&name));
        fields.push(pyre_object::gc_roots::shadow_stack_get(aliases_slot));
        fields.push(pyre_object::gc_roots::shadow_stack_get(addrs_slot));
        Ok(pyre_object::w_tuple_new(fields.take()))
    }
}

// `getdefaulttimeout` / `setdefaulttimeout`. `-1.0` blocks.
// The cell is `Defaults.timeout` on unix and Windows. Wasm keeps a local
// copy because `majit_rlib::rsocket` is not built there.

#[cfg(not(any(unix, windows)))]
static DEFAULT_SOCKET_TIMEOUT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new((-1.0f64).to_bits());

fn default_socket_timeout() -> f64 {
    #[cfg(any(unix, windows))]
    {
        majit_rlib::rsocket::getdefaulttimeout()
    }
    #[cfg(not(any(unix, windows)))]
    {
        f64::from_bits(DEFAULT_SOCKET_TIMEOUT.load(std::sync::atomic::Ordering::Relaxed))
    }
}

fn get_default_socket_timeout() -> pyre_object::PyObjectRef {
    let timeout = default_socket_timeout();
    if timeout < 0.0 {
        pyre_object::w_none()
    } else {
        pyre_object::floatobject::w_float_new(timeout)
    }
}

fn set_default_socket_timeout(v: Option<f64>) {
    let timeout = v.unwrap_or(-1.0);
    #[cfg(any(unix, windows))]
    majit_rlib::rsocket::setdefaulttimeout(timeout);
    #[cfg(not(any(unix, windows)))]
    DEFAULT_SOCKET_TIMEOUT.store(timeout.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

// ── getaddrinfo / getnameinfo wiring ──
//
// `interp_func.py` converts the arguments. `rsocket.getaddrinfo` walks
// the resolver list and `rsocket.getnameinfo` fills the host and service
// buffers. The 5-tuple and the flowinfo/scope fill stay here.
// Windows still calls the WinSock wrappers: `_rsocket_rffi` has no
// `getaddrinfo` there yet.

#[cfg(any(unix, windows))]
struct AddrinfoAnswer {
    family: i64,
    socktype: i64,
    protocol: i64,
    canonname: String,
    storage: rffi::sockaddr_storage,
    addrlen: usize,
}

#[cfg(any(unix, windows))]
fn addrinfo_answer(
    family: i64,
    socktype: i64,
    protocol: i64,
    canonname: String,
    addr: &[u8],
) -> AddrinfoAnswer {
    let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
    let addrlen = addr
        .len()
        .min(core::mem::size_of::<rffi::sockaddr_storage>());
    if addrlen > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                addr.as_ptr(),
                (&mut storage as *mut rffi::sockaddr_storage).cast::<u8>(),
                addrlen,
            );
        }
    }
    AddrinfoAnswer {
        family,
        socktype,
        protocol,
        canonname,
        storage,
        addrlen,
    }
}

#[cfg(unix)]
fn addrinfo_answers(
    host: Option<&std::ffi::CStr>,
    port: Option<&std::ffi::CStr>,
    family: libc::c_int,
    socktype: libc::c_int,
    proto: libc::c_int,
    flags: libc::c_int,
) -> Result<Vec<AddrinfoAnswer>, pyre_interpreter::PyError> {
    let list = majit_rlib::rsocket::getaddrinfo(host, port, family, socktype, proto, flags)
        .map_err(|error| set_gaierror(error.errno))?;
    Ok(list
        .into_iter()
        .map(|info| {
            addrinfo_answer(
                info.family,
                info.socktype,
                info.protocol,
                info.canonname,
                &info.addr,
            )
        })
        .collect())
}

#[cfg(windows)]
fn addrinfo_answers(
    host: Option<&std::ffi::CStr>,
    port: Option<&std::ffi::CStr>,
    family: libc::c_int,
    socktype: libc::c_int,
    proto: libc::c_int,
    flags: libc::c_int,
) -> Result<Vec<AddrinfoAnswer>, pyre_interpreter::PyError> {
    let mut hints: rffi::addrinfo = unsafe { std::mem::zeroed() };
    hints.ai_family = family;
    hints.ai_socktype = socktype;
    hints.ai_protocol = proto;
    hints.ai_flags = flags;
    let mut res: *mut rffi::addrinfo = std::ptr::null_mut();
    let host_ptr = host.map(std::ffi::CStr::as_ptr).unwrap_or(std::ptr::null());
    let port_ptr = port.map(std::ffi::CStr::as_ptr).unwrap_or(std::ptr::null());
    let rc = unsafe { rffi::getaddrinfo(host_ptr, port_ptr, &hints, &mut res) };
    if rc != 0 {
        return Err(set_gaierror(rc));
    }
    let mut out = Vec::new();
    let mut cur = res;
    unsafe {
        while !cur.is_null() {
            let ai = &*cur;
            let canon = if ai.ai_canonname.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(ai.ai_canonname.cast())
                    .to_string_lossy()
                    .into_owned()
            };
            let copy_len =
                (ai.ai_addrlen as usize).min(core::mem::size_of::<rffi::sockaddr_storage>());
            let addr: &[u8] = if ai.ai_addr.is_null() || copy_len == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(ai.ai_addr.cast::<u8>(), copy_len)
            };
            out.push(addrinfo_answer(
                ai.ai_family as i64,
                ai.ai_socktype as i64,
                ai.ai_protocol as i64,
                canon,
                addr,
            ));
            cur = ai.ai_next;
        }
        rffi::freeaddrinfo(res);
    }
    Ok(out)
}

#[cfg(unix)]
fn nameinfo_from_sockaddr(
    resolved: &rffi::sockaddr_storage,
    resolved_len: usize,
    flags: libc::c_int,
) -> Result<(String, String), pyre_interpreter::PyError> {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (resolved as *const rffi::sockaddr_storage).cast::<u8>(),
            resolved_len,
        )
    };
    majit_rlib::rsocket::getnameinfo(bytes, flags).map_err(|error| set_gaierror(error.errno))
}

#[cfg(windows)]
fn nameinfo_from_sockaddr(
    resolved: &rffi::sockaddr_storage,
    resolved_len: usize,
    flags: libc::c_int,
) -> Result<(String, String), pyre_interpreter::PyError> {
    let mut host_buf = [0 as libc::c_char; rffi::NI_MAXHOST as usize];
    let mut serv_buf = [0 as libc::c_char; 32];
    let nrc = unsafe {
        rffi::getnameinfo(
            (resolved as *const rffi::sockaddr_storage).cast(),
            resolved_len as rffi::SockLen,
            host_buf.as_mut_ptr(),
            host_buf.len() as rffi::SockLen,
            serv_buf.as_mut_ptr(),
            serv_buf.len() as rffi::SockLen,
            flags,
        )
    };
    if nrc != 0 {
        return Err(set_gaierror(nrc));
    }
    let host_s = unsafe {
        std::ffi::CStr::from_ptr(host_buf.as_ptr())
            .to_string_lossy()
            .into_owned()
    };
    let serv_s = unsafe {
        std::ffi::CStr::from_ptr(serv_buf.as_ptr())
            .to_string_lossy()
            .into_owned()
    };
    Ok((host_s, serv_s))
}

#[cfg(any(unix, windows))]
fn init_socket_getaddrinfo(mut ns: pyre_object::PyObjectRef) {
    pyre_interpreter::__pyre_store!(
        ns,
        "getaddrinfo",
        pyre_interpreter::make_builtin_function("getaddrinfo", |args| {
            if args.len() < 2 {
                return Err(pyre_interpreter::PyError::type_error(
                    "getaddrinfo() missing host or port",
                ));
            }
            // `socket_encode_idna` runs the codec, which is Python and can
            // collect, and the native argument slice is only current at
            // entry: pin the arguments and read each one back from its slot.
            let _roots = pyre_object::gc_roots::push_roots();
            let args_base = pyre_object::gc_roots::pin_roots(args);
            // host: None | bytes | str
            let host_obj = pyre_object::gc_roots::shadow_stack_get(args_base);
            let host: Option<std::ffi::CString> = unsafe {
                if pyre_object::is_none(host_obj) {
                    None
                } else if pyre_object::bytesobject::is_bytes(host_obj) {
                    Some(socket_cstring_at_nul(
                        pyre_object::bytesobject::bytes_like_data(host_obj).to_vec(),
                    ))
                } else if pyre_object::is_str(host_obj) {
                    Some(socket_cstring_at_nul(socket_encode_idna(
                        "getaddrinfo",
                        host_obj,
                    )?))
                } else {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getaddrinfo() argument 1 must be string or None",
                    ));
                }
            };
            // port: None | int | bytes | str
            let port_obj = pyre_object::gc_roots::shadow_stack_get(args_base + 1);
            let port: Option<std::ffi::CString> = unsafe {
                if pyre_object::is_none(port_obj) {
                    None
                } else if pyre_interpreter::baseobjspace::isinstance_int_w(port_obj) {
                    // `interp_func.getaddrinfo` keeps PyPy's
                    // `isinstance_w(..., w_int)` branch and passes a decimal
                    // service string to `rsocket.getaddrinfo`.  PyPy 3.11's
                    // `space.int_w` narrows that branch to a machine word, but
                    // 3.14 `test_getaddrinfo_int_port_overflow`
                    // requires an arbitrary-size int to reach libc as text:
                    // [3.14-spec] decimal bigint service ↔ PyPy machine-int
                    // service — observable exception is gaierror/success, never
                    // an interpreter OverflowError or TypeError.
                    let exact = if pyre_object::is_bool(port_obj)
                        || pyre_object::is_int(port_obj)
                        || pyre_object::is_long(port_obj)
                    {
                        port_obj
                    } else {
                        pyre_interpreter::baseobjspace::space_int(port_obj)?
                    };
                    let service = pyre_interpreter::builtins::int_to_decimal_string(exact)?;
                    Some(std::ffi::CString::new(service).unwrap())
                } else if pyre_object::bytesobject::is_bytes(port_obj) {
                    Some(socket_cstring_at_nul(
                        pyre_object::bytesobject::bytes_like_data(port_obj).to_vec(),
                    ))
                } else if pyre_object::is_str(port_obj) {
                    // The service is spelled utf-8 rather than idna:
                    // `getaddrinfo` encodes it with
                    // `space.encode_unicode_object(w_port, 'utf-8', 'strict')`.
                    let s = pyre_interpreter::baseobjspace::str_utf8_w(port_obj)?.to_string();
                    Some(socket_cstring_at_nul(s.into_bytes()))
                } else {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getaddrinfo() argument 2 must be integer or string",
                    ));
                }
            };

            let nargs = args.len();
            let int_arg = |idx: usize,
                           default: libc::c_int|
             -> Result<libc::c_int, pyre_interpreter::PyError> {
                if nargs > idx {
                    let v = pyre_object::gc_roots::shadow_stack_get(args_base + idx);
                    if !unsafe { pyre_object::is_int(v) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "getaddrinfo: family/type/proto/flags must be integers",
                        ));
                    }
                    Ok(unsafe { pyre_object::w_int_get_value(v) } as libc::c_int)
                } else {
                    Ok(default)
                }
            };
            let family = int_arg(2, rffi::AF_UNSPEC)?;
            let socktype = int_arg(3, 0)?;
            let proto = int_arg(4, 0)?;
            let flags = int_arg(5, 0)?;

            let answers = addrinfo_answers(
                host.as_deref(),
                port.as_deref(),
                family,
                socktype,
                proto,
                flags,
            )?;
            // Every field and entry is a nursery object and the next lap
            // allocates again. Nested `RootedItems` closes before the outer
            // `push`, so each completed entry stays live on `result_w`.
            let mut result_w = pyre_object::gc_roots::RootedItems::new();
            for info in answers {
                let entry = {
                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                    fields.push(pyre_object::w_int_new(info.family));
                    fields.push(pyre_object::w_int_new(info.socktype));
                    fields.push(pyre_object::w_int_new(info.protocol));
                    fields.push(pyre_object::w_str_new_managed(&info.canonname));
                    fields.push(unpack_inet_addr(
                        &info.storage,
                        info.addrlen as rffi::SockLen,
                    ));
                    pyre_object::w_tuple_new(fields.take())
                };
                result_w.push(entry);
            }
            Ok(pyre_object::w_list_new(result_w.take()))
        })
    );

    pyre_interpreter::__pyre_store!(
        ns,
        "getnameinfo",
        pyre_interpreter::make_builtin_function_with_arity(
            "getnameinfo",
            |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getnameinfo() requires (sockaddr, flags)",
                    ));
                }
                if !unsafe { pyre_object::is_tuple(args[0]) } {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getnameinfo() argument 1 must be a tuple",
                    ));
                }
                if !unsafe { pyre_object::is_int(args[1]) } {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getnameinfo: flags must be an integer",
                    ));
                }
                let flags = unsafe { pyre_object::w_int_get_value(args[1]) } as libc::c_int;
                // Resolve sockaddr via getaddrinfo(AF_UNSPEC, SOCK_DGRAM,
                // AI_NUMERICHOST) so we get a real sockaddr_storage,
                // matching `interp_func.py`.
                //
                // The tuple is parsed with `"si|II"` and a `;`-suffixed custom
                // message, so every shape it rejects — a length outside two to
                // four, a non-string host, a non-integer port — is the one
                // TypeError below rather than a per-item message.
                let sockaddr_len = unsafe { pyre_object::w_tuple_len(args[0]) };
                let item = |i: i64| unsafe { pyre_object::w_tuple_getitem(args[0], i) };
                let illegal_sockaddr = || {
                    pyre_interpreter::PyError::type_error(
                        "getnameinfo(): illegal sockaddr argument",
                    )
                };
                if !(2..=4).contains(&sockaddr_len) {
                    return Err(illegal_sockaddr());
                }
                let host_obj = item(0).ok_or_else(illegal_sockaddr)?;
                let mut port_obj = item(1).ok_or_else(illegal_sockaddr)?;
                if !unsafe { pyre_object::is_str(host_obj) }
                    || !unsafe { pyre_object::is_int(port_obj) }
                {
                    return Err(illegal_sockaddr());
                }
                // `flowinfo` is the third item, and its twenty bits are the
                // only thing checked about it here — the address itself comes
                // from the lookup below.
                let flowinfo = if let Some(flowinfo_obj) = item(2) {
                    if !unsafe { pyre_object::is_int(flowinfo_obj) } {
                        return Err(illegal_sockaddr());
                    }
                    let flowinfo = unsafe { pyre_object::w_int_get_value(flowinfo_obj) };
                    if !(0..=0xfffff).contains(&flowinfo) {
                        return Err(pyre_interpreter::PyError::overflow_error(
                            "getnameinfo(): flowinfo must be 0-1048575.",
                        ));
                    }
                    flowinfo as u32
                } else {
                    0
                };
                let scope_id = if let Some(scope_obj) = item(3) {
                    if !unsafe { pyre_object::is_int(scope_obj) } {
                        return Err(illegal_sockaddr());
                    }
                    let scope_id = unsafe { pyre_object::w_int_get_value(scope_obj) };
                    u32::try_from(scope_id).map_err(|_| {
                        pyre_interpreter::PyError::overflow_error(
                            "getnameinfo(): scope id must be an unsigned 32-bit integer",
                        )
                    })?
                } else {
                    0
                };
                let host = pyre_object::with_roots!(port_obj => pyre_interpreter::baseobjspace::str_utf8_w(host_obj))?.to_string();
                let port_v = unsafe { pyre_object::w_int_get_value(port_obj) };

                let c_host = std::ffi::CString::new(host.as_bytes())
                    .map_err(|_| pyre_interpreter::PyError::value_error("embedded null in host"))?;
                let c_port = std::ffi::CString::new(format!("{port_v}")).unwrap();

                let mut answers = addrinfo_answers(
                    Some(&c_host),
                    Some(&c_port),
                    rffi::AF_UNSPEC,
                    rffi::SOCK_DGRAM,
                    0,
                    rffi::AI_NUMERICHOST,
                )?;
                let ai = match answers.pop() {
                    Some(ai) if answers.is_empty() => ai,
                    _ => {
                        return Err(socket_converted_error(
                            "error",
                            None,
                            "sockaddr resolved to multiple addresses",
                        ));
                    }
                };
                // The extra items only exist for IPv6, so an IPv4 answer with
                // any of them is refused — after the lookup, because until it
                // returns the family is not known.
                if ai.family == i64::from(rffi::AF_INET) && sockaddr_len != 2 {
                    return Err(socket_converted_error(
                        "error",
                        None,
                        "IPv4 sockaddr must be 2 tuple",
                    ));
                }
                let mut resolved = ai.storage;
                let resolved_len = ai.addrlen;
                if ai.family == i64::from(rffi::AF_INET6) {
                    // `INET6Address.__init__` stores both optional tuple
                    // fields in the sockaddr handed to getnameinfo. A
                    // resolver lookup of the bare host cannot recover the
                    // scope id by itself.
                    let sin6 =
                        unsafe { &mut *(&mut resolved as *mut _ as *mut rffi::sockaddr_in6) };
                    sin6.sin6_flowinfo = flowinfo.to_be();
                    rffi::sockaddr_in6_set_scope_id(sin6, scope_id);
                }
                let (host_s, serv_s) = nameinfo_from_sockaddr(&resolved, resolved_len, flags)?;
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(pyre_object::w_str_new_managed(&host_s));
                fields.push(pyre_object::w_str_new_managed(&serv_s));
                Ok(pyre_object::w_tuple_new(fields.take()))
            },
            2,
        )
    );
}

// ── _socket socket() class implementation ─────────────────────────────
//
// Instance state lives in the instance dict under reserved keys
// `_fd` (int) / `_family` (int) / `_type` (int) / `_proto` (int) /
// `_timeout` (float or None).  Methods read/write via baseobjspace.

#[cfg(any(unix, windows))]
fn socket_type() -> pyre_object::PyObjectRef {
    // Process-global immortal type object (see `make_builtin_type`).
    static SOCKET_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
        pyre_object::gc_roots::RootedOnceRef::new();
    SOCKET_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type("socket", init_socket_type);
        // PyPy's `W_Socket.typedef` is a builtin immutable type. 3.14
        // exposes the same immutable public type while constructing it from a
        // module heap spec, so retain HEAPTYPE and set IMMUTABLETYPE as well.
        pyre_interpreter::typedef::mark_cpython_heap_type(tp, true);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        unsafe { pyre_object::w_type_set_hasuserdel(tp, true) };
        tp
    })
}

#[cfg(any(unix, windows))]
fn socket_io_err(e: std::io::Error) -> pyre_interpreter::PyError {
    let code = e.raw_os_error().unwrap_or(0);
    // `socketmodule.c set_error` raises a WinSock failure through
    // `PyErr_SetExcFromWindowsErr`: the code is a Win32 one, so it belongs in
    // `.winerror` with `.errno` derived from it, not read as an errno itself.
    #[cfg(windows)]
    {
        pyre_interpreter::PyError::os_error_win32_syscall2(
            code,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        )
    }
    // `rsocket.py` carries the C `strerror` text; build the OSError in
    // its `(errno, strerror)` form so `e.errno` and `str(e)` match.
    #[cfg(not(windows))]
    {
        let strerror = unsafe {
            let p = libc::strerror(code);
            if p.is_null() {
                format!("Unknown error {code}")
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        pyre_interpreter::PyError::os_error_errno_strerror(code, strerror)
    }
}

/// The exception for a WinSock call that has just failed, read from wherever
/// the host records it.
#[cfg(windows)]
fn socket_last_error() -> pyre_interpreter::PyError {
    socket_io_err(rffi::last_error())
}

#[cfg(unix)]
fn rsocket_os_error(error: majit_rlib::rsocket::CSocketError) -> pyre_interpreter::PyError {
    socket_io_err(std::io::Error::from_raw_os_error(error.errno))
}

/// Release the interpreter around a WinSock call and return the code
/// `last_error_code` reported. WinSock does not write the C runtime's `errno`.
/// `recvmsg` and `sendmsg` stay on `call_external_function`.
#[cfg(windows)]
fn socket_call<R>(f: impl FnOnce() -> R) -> (R, i32) {
    let _blocked = pyre_interpreter::module::thread::before_external_block();
    let result = f();
    (result, rffi::last_error_code())
}

/// `socket_io_err_for_operation` for a caller holding the code itself.
#[cfg(any(unix, windows))]
pub(crate) fn socket_error_for_operation(
    obj: pyre_object::PyObjectRef,
    errno: i32,
) -> pyre_interpreter::PyError {
    socket_io_err_for_operation(obj, std::io::Error::from_raw_os_error(errno))
}

#[cfg(any(unix, windows))]
fn socket_io_err_for_operation(
    _obj: pyre_object::PyObjectRef,
    e: std::io::Error,
) -> pyre_interpreter::PyError {
    // `RSocket.wait_for_data` itself owns SocketTimeout.  Once poll reported
    // readiness, a raced EAGAIN/EWOULDBLOCK is the ordinary socket error the
    // syscall returned; it must not be relabelled as an expired wait.
    socket_io_err(e)
}

/// The socket's timeout when it is a positive number of seconds — the state in
/// which a call has to be bounded rather than left to block.  `None` covers
/// both of the other two: blocking forever, and non-blocking.
#[cfg(any(unix, windows))]
fn socket_positive_timeout(obj: pyre_object::PyObjectRef) -> Option<f64> {
    let dict = pyre_interpreter::baseobjspace::getdict_native(obj);
    (!dict.is_null())
        .then(|| unsafe { pyre_object::w_dict_getitem_str(dict, "_timeout") })
        .flatten()
        .filter(|timeout| unsafe { pyre_object::is_float(*timeout) })
        .map(|timeout| unsafe { pyre_object::floatobject::w_float_get_value(timeout) })
        .filter(|timeout| *timeout > 0.0)
}

/// RPython `RSocket._select`: a positive Python timeout is enforced with poll
/// before entering every libc socket operation.  Zero skips the wait and the
/// descriptor itself answers would-block; a negative timeout is blocking.
///
/// Answers whether the readiness was waited for, which is the distinction
/// `_ssl_select` reports as `SOCKET_OPERATION_OK` against its two
/// nothing-to-wait-for states: a caller that has already been told the
/// operation would block needs it to tell a raced wakeup from a socket whose
/// mode owes it a `WANT_READ`.
///
/// The deadline loop is `rsocket.wait_for_data`. The Python `_timeout` read
/// stays here, and so does the mapping onto `TimeoutError` / `OSError`.
#[cfg(any(unix, windows))]
pub(crate) fn socket_wait_for_data(
    obj: pyre_object::PyObjectRef,
    fd: rffi::Socket,
    for_writing: bool,
) -> Result<bool, pyre_interpreter::PyError> {
    let Some(timeout) = socket_positive_timeout(obj) else {
        return Ok(false);
    };
    // A handled signal restarts the wait, so the deadline is computed once:
    // reusing the full duration on every EINTR would let a steady signal
    // stream extend a finite timeout without bound.  poll carries its timeout
    // in milliseconds as a c_int, which bounds how long a single wait can ask
    // for however long the caller requested.
    let capped = timeout.min(i32::MAX as f64 / 1000.0);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(capped);
    map_wait(majit_rlib::rsocket::wait_for_data(
        fd,
        for_writing,
        deadline,
        poll_socket,
        pyre_interpreter::module::signal::interp_signal::checksignals_now,
    ))
}

#[cfg(any(unix, windows))]
fn socket_wait_readable(
    obj: pyre_object::PyObjectRef,
    fd: rffi::Socket,
) -> Result<(), pyre_interpreter::PyError> {
    socket_wait_for_data(obj, fd, false).map(|_| ())
}

#[cfg(any(unix, windows))]
fn socket_wait_writable(
    obj: pyre_object::PyObjectRef,
    fd: rffi::Socket,
) -> Result<(), pyre_interpreter::PyError> {
    socket_wait_for_data(obj, fd, true).map(|_| ())
}

/// Writable half of `RSocket.wait_for_data`, bounded by one absolute
/// deadline.  `RSocket.sendall` retries EINTR, but the timeout belongs to the
/// operation rather than to each retry; otherwise a steady stream of handled
/// signals can keep a finite send alive forever.
#[cfg(any(unix, windows))]
fn socket_wait_writable_until(
    fd: rffi::Socket,
    deadline: std::time::Instant,
) -> Result<(), pyre_interpreter::PyError> {
    map_wait(majit_rlib::rsocket::wait_for_data(
        fd,
        true,
        deadline,
        poll_socket,
        pyre_interpreter::module::signal::interp_signal::checksignals_now,
    ))
    .map(|_| ())
}

#[cfg(any(unix, windows))]
fn poll_socket(
    fd: rffi::Socket,
    for_writing: bool,
    timeout_ms: i32,
) -> majit_rlib::rsocket::PollWait {
    let (ready, errno) = if for_writing {
        rffi::poll_writable(fd, timeout_ms)
    } else {
        rffi::poll_readable(fd, timeout_ms)
    };
    if ready > 0 {
        majit_rlib::rsocket::PollWait::Ready
    } else if ready == 0 {
        majit_rlib::rsocket::PollWait::TimedOut
    } else if rffi::error_is_interrupted(errno) {
        majit_rlib::rsocket::PollWait::Interrupted
    } else {
        majit_rlib::rsocket::PollWait::Failed(errno)
    }
}

#[cfg(any(unix, windows))]
fn map_wait<T>(
    result: Result<T, majit_rlib::rsocket::WaitError<pyre_interpreter::PyError>>,
) -> Result<T, pyre_interpreter::PyError> {
    match result {
        Ok(value) => Ok(value),
        Err(majit_rlib::rsocket::WaitError::Timeout) => {
            Err(socket_converted_error("timeout", None, "timed out"))
        }
        Err(majit_rlib::rsocket::WaitError::Os(error)) => Err(socket_io_err(
            std::io::Error::from_raw_os_error(error.errno),
        )),
        Err(majit_rlib::rsocket::WaitError::Check(error)) => Err(error),
    }
}

/// `RSocket._connect` for a socket carrying a positive timeout.  Positive
/// timeout descriptors are already non-blocking; connect waits for writable
/// and then reads `SO_ERROR` on both POSIX and WinSock.
///
/// Returns the code the attempt reported, so both `connect` and `connect_ex`
/// can shape it the way each one answers.
#[cfg(any(unix, windows))]
enum SocketConnectFailure {
    Errno(i32),
    Timeout(i32),
    Exception(pyre_interpreter::PyError),
}

#[cfg(any(unix, windows))]
fn socket_connect_wait(
    fd: rffi::Socket,
    storage: &rffi::sockaddr_storage,
    slen: rffi::SockLen,
    timeout: f64,
) -> Result<(), SocketConnectFailure> {
    #[cfg(unix)]
    let (started, errno) = match connect_sockaddr(fd, storage, slen) {
        Ok(()) => (0, 0),
        Err(code) => (-1, code),
    };
    #[cfg(windows)]
    let (started, errno) = socket_call(|| unsafe {
        rffi::connect(fd, storage as *const _ as *const rffi::sockaddr, slen)
    });
    if started == 0 {
        return Ok(());
    }
    if !rffi::error_is_connect_in_progress(errno) {
        return Err(SocketConnectFailure::Errno(errno));
    }
    let capped = timeout.min(i32::MAX as f64 / 1000.0);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(capped);
    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            #[cfg(unix)]
            return Err(SocketConnectFailure::Timeout(libc::EWOULDBLOCK));
            #[cfg(windows)]
            return Err(SocketConnectFailure::Timeout(rffi::WSAEWOULDBLOCK));
        }
        let timeout_ms = remaining.as_millis().max(1).min(i32::MAX as u128) as i32;
        let (ready, poll_errno) = rffi::poll_writable(fd, timeout_ms);
        if ready > 0 {
            break;
        }
        if ready == 0 {
            #[cfg(unix)]
            return Err(SocketConnectFailure::Timeout(libc::EWOULDBLOCK));
            #[cfg(windows)]
            return Err(SocketConnectFailure::Timeout(rffi::WSAEWOULDBLOCK));
        }
        if !rffi::error_is_interrupted(poll_errno) {
            return Err(SocketConnectFailure::Errno(poll_errno));
        }
        if let Err(error) = pyre_interpreter::module::signal::interp_signal::checksignals_now() {
            return Err(SocketConnectFailure::Exception(error));
        }
    }
    // `SO_ERROR` carries the outcome of a connect that finished after its call
    // returned; zero means it succeeded.
    match socket_getsockopt_int(fd, rffi::SOL_SOCKET, rffi::SO_ERROR) {
        Ok(0) => Ok(()),
        Ok(errno) => Err(SocketConnectFailure::Errno(errno)),
        // Reading the option failed; the last-error slot still holds why.
        Err(_) => Err(SocketConnectFailure::Errno(rffi::last_error_code())),
    }
}

#[cfg(any(unix, windows))]
fn socket_get_attr_i64(obj: pyre_object::PyObjectRef, key: &str) -> i64 {
    let d = pyre_interpreter::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return -1;
    }
    if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(d, key) }
        && unsafe { pyre_object::is_int(v) }
    {
        return unsafe { pyre_object::w_int_get_value(v) };
    }
    -1
}

#[cfg(any(unix, windows))]
fn socket_set_attr(obj: pyre_object::PyObjectRef, key: &str, mut v: pyre_object::PyObjectRef) {
    let d = pyre_object::with_roots!(v => pyre_interpreter::baseobjspace::getdict_native(obj));
    if d.is_null() {
        return;
    }
    unsafe {
        pyre_object::w_dict_setitem_str(d, key, v);
    }
}

/// `rsocket.py:RSocket.settimeout` — apply a timeout value to a live fd.
///
/// `timeout < 0` (the "None" sentinel) clears `O_NONBLOCK` so the socket
/// blocks indefinitely.  `timeout == 0` flips `O_NONBLOCK` on for
/// non-blocking mode.  `timeout > 0` clears `O_NONBLOCK` and writes the
/// duration to `SO_RCVTIMEO` + `SO_SNDTIMEO` so the kernel returns
/// `EAGAIN`/`EWOULDBLOCK` after the elapsed time.
///
/// Until this helper landed, `settimeout` only stashed the value in the
/// instance dict and `recv`/`send` blocked indefinitely regardless.
#[cfg(any(unix, windows))]
fn socket_apply_timeout(fd: rffi::Socket, timeout: f64) -> Result<(), pyre_interpreter::PyError> {
    rffi::apply_timeout(fd, timeout).map_err(socket_io_err)
}

#[cfg(any(unix, windows))]
pub(crate) fn socket_fd(
    obj: pyre_object::PyObjectRef,
) -> Result<rffi::Socket, pyre_interpreter::PyError> {
    let fd = rffi::socket_from_i64(socket_get_attr_i64(obj, "_fd"));
    if rffi::is_invalid(fd) {
        // `close` leaves `self.fd = INVALID_SOCKET` (`rsocket.py RSocket.close`)
        // and the operations run the syscall on it anyway, so what a closed
        // socket reports is whatever the host answers for that descriptor.
        // This check stands in for that call and owes the same error — callers
        // read `.errno` to tell a closed socket from a failed one.
        //
        // WinSock answers `WSAENOTSOCK` rather than the kernel's `EBADF`, and
        // `set_error` carries a WinSock code in `.winerror` the way every other
        // socket failure here does.
        #[cfg(windows)]
        return Err(pyre_interpreter::PyError::os_error_win32_syscall2(
            rffi::WSAENOTSOCK,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ));
        #[cfg(not(windows))]
        return Err(pyre_interpreter::PyError::os_error_syscall(
            libc::EBADF,
            pyre_object::PY_NULL,
        ));
    }
    Ok(fd)
}

/// `send` with nothing around it: no interpreter release, and the failure code
/// handed back rather than turned into an exception.  `_ssl`'s record pump
/// already released the interpreter for the whole exchange, so the unix body
/// calls `libc::send` and reads the live `errno`.  Socket methods go through
/// `rsocket.send`, the releasing `llexternal`.
#[cfg(any(unix, windows))]
pub(crate) fn socket_send_raw(
    fd: rffi::Socket,
    buf: &[u8],
    flags: libc::c_int,
) -> Result<isize, i32> {
    #[cfg(unix)]
    let sent = unsafe { libc::send(fd, buf.as_ptr() as *const libc::c_void, buf.len(), flags) };
    #[cfg(windows)]
    let sent = unsafe { rffi::send(fd, buf.as_ptr() as *const libc::c_void, buf.len(), flags) };
    if sent >= 0 {
        Ok(sent)
    } else {
        #[cfg(unix)]
        {
            Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(0))
        }
        #[cfg(windows)]
        {
            Err(rffi::last_error_code())
        }
    }
}

/// The `recv` counterpart of `socket_send_raw`, filling a caller's buffer.
#[cfg(any(unix, windows))]
pub(crate) fn socket_recv_raw(
    fd: rffi::Socket,
    buf: &mut [u8],
    flags: libc::c_int,
) -> Result<usize, i32> {
    #[cfg(unix)]
    let read = unsafe { libc::recv(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len(), flags) };
    #[cfg(windows)]
    let read = unsafe { rffi::recv(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len(), flags) };
    if read >= 0 {
        Ok(read as usize)
    } else {
        #[cfg(unix)]
        {
            Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(0))
        }
        #[cfg(windows)]
        {
            Err(rffi::last_error_code())
        }
    }
}

/// The interpreter release and interrupted-call retry `sock_send` runs around
/// the syscall, lifted out of the exported-buffer scope so the release covers
/// the error paths too.  `_ssl` writes a record through this wherever it
/// cannot hold the release across the whole exchange.
#[cfg(any(unix, windows))]
pub(crate) fn socket_send_bytes(
    obj: pyre_object::PyObjectRef,
    fd: rffi::Socket,
    buf: &[u8],
    flags: libc::c_int,
) -> Result<isize, pyre_interpreter::PyError> {
    loop {
        let outcome = {
            #[cfg(unix)]
            {
                match majit_rlib::rsocket::send(fd, buf, flags) {
                    Ok(sent) => Ok(sent),
                    Err(error) => Err(error.errno),
                }
            }
            #[cfg(windows)]
            {
                socket_call(|| socket_send_raw(fd, buf, flags)).0
            }
        };
        match outcome {
            Ok(sent) => return Ok(sent),
            Err(errno) if !rffi::error_is_interrupted(errno) => {
                return Err(socket_error_for_operation(obj, errno));
            }
            // EINTR: deliver a pending signal, then retry
            // (`converted_error` eintr_retry).
            Err(_) => pyre_interpreter::module::signal::interp_signal::checksignals_now()?,
        }
    }
}

/// The `sock_recv` counterpart of `socket_send_bytes`, filling a caller's
/// buffer rather than a fresh `bytes`.
#[cfg(any(unix, windows))]
pub(crate) fn socket_recv_bytes(
    obj: pyre_object::PyObjectRef,
    fd: rffi::Socket,
    buf: &mut [u8],
    flags: libc::c_int,
) -> Result<usize, pyre_interpreter::PyError> {
    loop {
        let outcome = {
            #[cfg(unix)]
            {
                match majit_rlib::rsocket::recv(fd, buf, flags) {
                    Ok(read) => Ok(read),
                    Err(error) => Err(error.errno),
                }
            }
            #[cfg(windows)]
            {
                socket_call(|| socket_recv_raw(fd, buf, flags)).0
            }
        };
        match outcome {
            Ok(read) => return Ok(read),
            Err(errno) if !rffi::error_is_interrupted(errno) => {
                return Err(socket_error_for_operation(obj, errno));
            }
            // EINTR: deliver a pending signal, then retry
            // (`converted_error` eintr_retry).
            Err(_) => pyre_interpreter::module::signal::interp_signal::checksignals_now()?,
        }
    }
}

#[cfg(any(unix, windows))]
fn socket_from_fd(
    fd: rffi::Socket,
    family: libc::c_int,
    ty: libc::c_int,
    proto: libc::c_int,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    socket_from_fd_with_class(fd, family, ty, proto, socket_type())
}

#[cfg(any(unix, windows))]
fn socket_from_fd_with_class(
    fd: rffi::Socket,
    family: libc::c_int,
    ty: libc::c_int,
    proto: libc::c_int,
    cls: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    // PyPy's typeobject.py allocate_instance preserves the requested
    // W_TypeObject when W_Socket.descr_init initialises a user subclass.
    // The stdlib relies on this for `class socket(_socket.socket)`: losing
    // `cls` here also loses its Python-level makefile() and lifecycle state.
    let mut obj = pyre_object::w_instance_new(cls);
    pyre_object::with_roots!(obj => socket_init_state(obj, fd, family, ty, proto))?;
    Ok(obj)
}

#[cfg(any(unix, windows))]
fn socket_init_state(
    mut obj: pyre_object::PyObjectRef,
    fd: rffi::Socket,
    family: libc::c_int,
    ty: libc::c_int,
    proto: libc::c_int,
) -> Result<(), pyre_interpreter::PyError> {
    pyre_object::with_roots!(obj => socket_set_attr(obj, "_fd", pyre_object::w_int_new(rffi::socket_to_i64(fd))));
    pyre_object::with_roots!(obj => socket_set_attr(obj, "_family", pyre_object::w_int_new(family as i64)));
    // `rsocket.py:RSocket.__init__` clears both creation flags out of the type
    // it keeps, so that `sock.type == SOCK_STREAM` answers the same on a host
    // that accepts them inside `socket()`'s type argument as on one that does
    // not.  Subtracting the bit stands in for `&= ~flag`, which the static
    // analyzer rejects on a signed `c_int`.
    let mut stored_type = ty;
    if stored_type & rffi::SOCK_CLOEXEC != 0 {
        stored_type -= rffi::SOCK_CLOEXEC;
    }
    if stored_type & rffi::SOCK_NONBLOCK != 0 {
        stored_type -= rffi::SOCK_NONBLOCK;
    }
    pyre_object::with_roots!(obj => socket_set_attr(obj, "_type", pyre_object::w_int_new(stored_type as i64)));
    pyre_object::with_roots!(obj => socket_set_attr(obj, "_proto", pyre_object::w_int_new(proto as i64)));
    // `rsocket.py:RSocket.__init__` calls `settimeout(defaults.timeout)` for
    // every new or wrapped descriptor, and `make_socket` copies that same
    // process-wide value.  Applying it here (not merely reporting it from
    // getdefaulttimeout) is what makes a newly constructed socket inherit the
    // default in both blocking mode and its public `gettimeout()` result.
    // A `SOCK_NONBLOCK` creation flag is the one exception: it names the mode
    // the descriptor was opened in, so the timeout it stands for is recorded
    // without a `settimeout` of its own.
    if ty & rffi::SOCK_NONBLOCK != 0 {
        pyre_object::with_roots!(obj => socket_set_attr(obj, "_timeout", pyre_object::floatobject::w_float_new(0.0)));
    } else {
        let timeout = default_socket_timeout();
        let stored_timeout = if timeout < 0.0 {
            pyre_object::w_none()
        } else {
            pyre_object::floatobject::w_float_new(timeout)
        };
        pyre_object::with_roots!(obj => socket_set_attr(obj, "_timeout", stored_timeout));
        socket_apply_timeout(fd, timeout)?;
    }
    // `sock_new` starts this at 0 on every socket object, and only
    // `setsockopt` moves it: `SIO_TCP_SET_ACK_FREQUENCY` has no counterpart to
    // read the live setting back with, so what was written is the answer.
    #[cfg(windows)]
    socket_set_attr(obj, "_quickack", pyre_object::w_int_new(0));
    // `interp_socket.py usecount = 1` — start the refcount at 1 so
    // `_drop` followed by no `_reuse` closes the underlying fd exactly
    // once.
    socket_set_attr(obj, "_usecount", pyre_object::w_int_new(1));
    Ok(())
}

/// `rsocket.py get_socket_family` — `sa_family` from `getsockname`.
/// `CSocketError.errno` is the socket OSError.
#[cfg(any(unix, windows))]
fn socket_detect_family(fd: rffi::Socket) -> Result<libc::c_int, pyre_interpreter::PyError> {
    majit_rlib::rsocket::get_socket_family(fd)
        .map(|family| family as libc::c_int)
        .map_err(|error| socket_io_err(std::io::Error::from_raw_os_error(error.errno)))
}

/// `rsocket.py getsockopt_int` — one `int` socket option.
/// `CSocketError.errno` is the socket OSError.
#[cfg(any(unix, windows))]
fn socket_getsockopt_int(
    fd: rffi::Socket,
    level: libc::c_int,
    option: libc::c_int,
) -> Result<libc::c_int, pyre_interpreter::PyError> {
    majit_rlib::rsocket::getsockopt_int(fd, level, option)
        .map(|value| value as libc::c_int)
        .map_err(|error| socket_io_err(std::io::Error::from_raw_os_error(error.errno)))
}

/// `interp_socket.py get_so_protocol` — the protocol of an existing
/// fd via `SO_PROTOCOL`, or `-1` on platforms without it (`HAS_SO_PROTOCOL`).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn socket_get_so_protocol(fd: rffi::Socket) -> Result<libc::c_int, pyre_interpreter::PyError> {
    socket_getsockopt_int(fd, rffi::SOL_SOCKET, libc::SO_PROTOCOL)
}
#[cfg(all(
    any(unix, windows),
    not(any(target_os = "linux", target_os = "android"))
))]
fn socket_get_so_protocol(_fd: rffi::Socket) -> Result<libc::c_int, pyre_interpreter::PyError> {
    Ok(-1)
}

// ── address pack/unpack helpers ──
//
// Python passes IPv4 addresses as (host, port) tuples and IPv6 as
// (host, port, flowinfo, scopeid).  These helpers convert to/from
// `sockaddr_storage`.

/// `offsetof(_c.sockaddr_un, 'c_sun_path')` (`rsocket.py:403 minlen`) — where
/// the name starts inside the address, and therefore the part of an address
/// length that is not name.
#[cfg(unix)]
const SUN_PATH_OFFSET: usize = core::mem::offset_of!(libc::sockaddr_un, sun_path);

/// `NDIS_IF_MAX_STRING_SIZE` (`ifdef.h`) — the room
/// `ConvertInterfaceLuidToNameW` is given for a name.
#[cfg(all(windows, feature = "host_env"))]
const NDIS_IF_MAX_STRING_SIZE: usize = 255;

/// The Windows arm of `socket_if_nameindex`: `GetIfTable2Ex` for the table,
/// then `ConvertInterfaceLuidToNameW` per row, appended to a list built empty.
///
/// `rustpython_host_env::socket::if_nameindex` walks the same table but hands
/// each name back through `String::from_utf16_lossy`, which spends an unpaired
/// surrogate on U+FFFD; `Py_BuildValue("Iu", ...)` keeps what the call wrote,
/// so the wide buffer is read as WTF-8 here instead.
#[cfg(all(windows, feature = "host_env"))]
fn windows_if_nameindex() -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    use windows_sys::Win32::Foundation::NO_ERROR;
    use windows_sys::Win32::NetworkManagement::IpHelper as ip;

    // Both calls answer with a status of their own instead of setting the last
    // error, and that status is the one reported.
    let win32 = |status| interface_io_error(std::io::Error::from_raw_os_error(status as i32));

    let mut table: *mut ip::MIB_IF_TABLE2 = core::ptr::null_mut();
    let status = unsafe { ip::GetIfTable2Ex(ip::MibIfTableRaw, &mut table) };
    if status != NO_ERROR {
        return Err(win32(status));
    }
    // The list is built empty and appended to, so no freshly boxed row waits
    // in a plain vector while the next allocation runs.
    let _roots = pyre_object::gc_roots::push_roots();
    let list_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_list_new_empty());
    let outcome = (|| {
        for index in 0..unsafe { (*table).NumEntries } as usize {
            let row = unsafe { &*(*table).Table.as_ptr().add(index) };
            let mut name = [0u16; NDIS_IF_MAX_STRING_SIZE + 1];
            let status = unsafe {
                ip::ConvertInterfaceLuidToNameW(&row.InterfaceLuid, name.as_mut_ptr(), name.len())
            };
            if status != NO_ERROR {
                return Err(win32(status));
            }
            let end = name
                .iter()
                .position(|&unit| unit == 0)
                .unwrap_or(name.len());
            let entry = {
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(pyre_object::w_int_new(i64::from(row.InterfaceIndex)));
                fields.push(pyre_object::w_str_from_wtf8_managed(
                    rustpython_wtf8::Wtf8Buf::from_wide(&name[..end]),
                ));
                pyre_object::w_tuple_new(fields.take())
            };
            unsafe {
                pyre_object::listobject::w_list_append(
                    pyre_object::gc_roots::shadow_stack_get(list_slot),
                    entry,
                )
            };
        }
        Ok(())
    })();
    // `FreeMibTable` runs whichever way the walk ended, as it does there.
    unsafe { ip::FreeMibTable(table.cast()) };
    outcome?;
    Ok(pyre_object::gc_roots::shadow_stack_get(list_slot))
}

/// `PyLong_AsUnsignedLongMask`: `PyNumber_Index` first, so `__index__`
/// decides the value and `__int__` is never asked for one, and then the low
/// bits of what it answered.  This is `ioctl`'s `I` code, the `k` code behind
/// its `PyIndex_Check`, and the `unsigned_long(bitwise=True)` converter
/// `share` reads its process id through — one width on a host where
/// `unsigned int`, `unsigned long` and `DWORD` are all 32 bits.  Carrying no
/// `PyIndex_Check` of its own, it names an object answering neither the way
/// `PyNumber_Index` does rather than by giving the argument's position.
#[cfg(windows)]
fn masked_ulong_w(obj: pyre_object::PyObjectRef) -> Result<u32, pyre_interpreter::PyError> {
    Ok(
        pyre_interpreter::baseobjspace::truncatedint_w(
            pyre_interpreter::baseobjspace::space_index(obj)?,
        )? as u32,
    )
}

/// `k` — [`masked_ulong_w`] behind a `PyIndex_Check` that reports the
/// argument it was reading.  `converttuple` nests the position, so an item of
/// the `(kkk)` group names itself as `argument 2, item 0`.
#[cfg(windows)]
fn ioctl_command_w(
    obj: pyre_object::PyObjectRef,
    argument: &str,
) -> Result<u32, pyre_interpreter::PyError> {
    if !unsafe { pyre_object::pyobject::is_int_or_long(obj) }
        && unsafe { pyre_interpreter::baseobjspace::lookup(obj, "__index__") }.is_none()
    {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "ioctl() {argument} must be int, not {}",
            pyre_interpreter::type_methods::clinic_arg_type_name(obj)
        )));
    }
    masked_ulong_w(obj)
}

/// `(kkk)` — the three fields `tcp_keepalive` carries.  A parenthesised group
/// reads its argument through `PySequence_Check` and `PySequence_GetItem`, so
/// a list or a range serves as well as a tuple, and it reports the two
/// failures differently: a wrong length names the length, anything that is
/// not a sequence names the type.
#[cfg(windows)]
fn ioctl_keepalive_w(obj: pyre_object::PyObjectRef) -> Result<[u32; 3], pyre_interpreter::PyError> {
    // `PySequence_Check` guards the group, with `str` turned away beside the
    // objects that answer no subscript at all: a three-character string is a
    // sequence of length three and would otherwise be read as one.
    let is_sequence = !unsafe { pyre_object::is_dict(obj) }
        && !unsafe { pyre_object::is_str(obj) }
        && unsafe { pyre_interpreter::baseobjspace::lookup(obj, "__getitem__") }.is_some();
    if !is_sequence {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "ioctl() argument 2 must be 3-item tuple, not {}",
            pyre_interpreter::type_methods::clinic_arg_type_name(obj)
        )));
    }
    let items = pyre_interpreter::baseobjspace::unpackiterable(obj, -1)?;
    if items.len() != 3 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "ioctl() argument 2 must be tuple of length 3, not {}",
            items.len()
        )));
    }
    // Reading one item can run `__index__`, so all three are published before
    // the first conversion rather than held as a plain vector across it.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    for &item in &items {
        let _ = pyre_object::gc_roots::pin_root(item);
    }
    let mut values = [0u32; 3];
    for (index, value) in values.iter_mut().enumerate() {
        *value = ioctl_command_w(
            pyre_object::gc_roots::shadow_stack_get(base + index),
            &format!("argument 2, item {index}"),
        )?;
    }
    Ok(values)
}

/// One `%X` conversion of the scan `setbdaddr` runs: blanks, an optional sign,
/// an optional `0x` prefix, then hex digits accumulated into the `unsigned
/// int` the C code declares.  Answers the value and what is left to read.
#[cfg(windows)]
fn scan_hex_field(text: &str) -> Option<(u32, &str)> {
    let text = text.trim_start_matches([' ', '\t', '\n', '\r', '\u{b}', '\u{c}']);
    let (negative, text) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    // A `0x` counts as a prefix only when a hex digit follows: otherwise the
    // conversion stops after the `0` and leaves the `x` to be read.
    let digits = match text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        Some(rest) if rest.starts_with(|c: char| c.is_ascii_hexdigit()) => rest,
        _ => text,
    };
    let mut value: u32 = 0;
    let mut end = 0;
    for digit in digits.bytes().take_while(u8::is_ascii_hexdigit) {
        let digit = char::from(digit).to_digit(16).expect("hex digit");
        value = value.wrapping_mul(16).wrapping_add(digit);
        end += 1;
    }
    if end == 0 {
        return None;
    }
    Some((
        if negative {
            value.wrapping_neg()
        } else {
            value
        },
        &digits[end..],
    ))
}

/// `setbdaddr` — six `%X` fields with a colon between each pair, every one of
/// them below 256, and a trailing `%c` that makes anything after the sixth a
/// seventh conversion and so a failure.  The leading field is the most
/// significant octet, and `BTH_ADDR` carries the six as one number.
#[cfg(windows)]
fn parse_bdaddr(name: &str) -> Option<u64> {
    let mut rest = name;
    let mut octets = [0u32; 6];
    for (i, octet) in octets.iter_mut().enumerate() {
        if i > 0 {
            rest = rest.strip_prefix(':')?;
        }
        (*octet, rest) = scan_hex_field(rest)?;
    }
    if !rest.is_empty() || octets.iter().fold(0, |acc, octet| acc | octet) >= 256 {
        return None;
    }
    Some(
        octets
            .iter()
            .fold(0u64, |acc, &octet| (acc << 8) | u64::from(octet)),
    )
}

/// `makebdaddr` — `XX:XX:XX:XX:XX:XX`, most significant octet first.
#[cfg(windows)]
fn bdaddr_string(bdaddr: u64) -> String {
    let octet = |i: u32| (bdaddr >> (8 * i)) & 0xFF;
    format!(
        "{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
        octet(5),
        octet(4),
        octet(3),
        octet(2),
        octet(1),
        octet(0)
    )
}

/// The `AF_BLUETOOTH` case of `getsockaddrarg` (`socketmodule.c:2104-2205`).
/// Only RFCOMM reaches here: it is the one Bluetooth protocol Windows carries,
/// and `SOCKADDR_BTH` is the one address form that goes with it.
#[cfg(windows)]
fn pack_bluetooth_addr(
    caller: &str,
    proto: libc::c_int,
    addr: pyre_object::PyObjectRef,
    storage: &mut rffi::sockaddr_storage,
) -> Result<rffi::SockLen, pyre_interpreter::PyError> {
    use windows_sys::Win32::Devices::Bluetooth as bt;

    if proto != BTHPROTO_RFCOMM {
        return Err(pyre_interpreter::PyError::os_error(format!(
            "{caller}(): unknown Bluetooth protocol"
        )));
    }
    // `PyArg_ParseTuple(args, "sk")` fails and is answered with this one
    // message, so every shape that is not a `str` beside an integer channel —
    // a wrong length, `bytes`, an embedded NUL, a lone surrogate — reads alike.
    let wrong_format = || pyre_interpreter::PyError::os_error(format!("{caller}(): wrong format"));
    if !unsafe { pyre_object::is_tuple(addr) } || unsafe { pyre_object::w_tuple_len(addr) } != 2 {
        return Err(wrong_format());
    }
    let w_name = unsafe { pyre_object::w_tuple_getitem(addr, 0) }.expect("length checked above");
    let w_channel = unsafe { pyre_object::w_tuple_getitem(addr, 1) }.expect("length checked above");
    if !unsafe { pyre_object::is_str(w_name) } || !unsafe { pyre_object::is_int(w_channel) } {
        return Err(wrong_format());
    }
    let name = pyre_interpreter::baseobjspace::str_utf8_w(w_name).map_err(|_| wrong_format())?;
    if name.contains('\0') {
        return Err(wrong_format());
    }
    let bd_addr = parse_bdaddr(name)
        .ok_or_else(|| pyre_interpreter::PyError::os_error("bad bluetooth address"))?;
    let bth = bt::SOCKADDR_BTH {
        addressFamily: AF_BTH as u16,
        btAddr: bd_addr,
        serviceClassId: Default::default(),
        // `k` keeps the low `ULONG` of whatever it is handed rather than
        // reporting an overflow.
        port: (unsafe { pyre_object::w_int_get_value(w_channel) }) as u32,
    };
    // `SOCKADDR_BTH` is packed, so its `BTH_ADDR` sits two bytes into the
    // storage and no aligned write reaches it.
    unsafe { core::ptr::write_unaligned(storage as *mut _ as *mut bt::SOCKADDR_BTH, bth) };
    Ok(core::mem::size_of::<bt::SOCKADDR_BTH>() as rffi::SockLen)
}

/// The RFCOMM case of `makesockaddr` (`socketmodule.c:1546-1560`) — the address
/// as a string beside the channel, the shape `bind` and `connect` take back.
#[cfg(windows)]
fn unpack_bluetooth_addr(storage: &rffi::sockaddr_storage) -> pyre_object::PyObjectRef {
    use windows_sys::Win32::Devices::Bluetooth as bt;

    let bth: bt::SOCKADDR_BTH =
        unsafe { core::ptr::read_unaligned(storage as *const _ as *const bt::SOCKADDR_BTH) };
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::w_str_new_managed(&bdaddr_string(bth.btAddr)));
    fields.push(pyre_object::w_int_new(i64::from(bth.port)));
    pyre_object::w_tuple_new(fields.take())
}

/// `HV_PROTOCOL_RAW` — the only protocol an `AF_HYPERV` socket is opened with.
/// `hvsocket.h` defines it; `windows-sys` does not carry that header.
#[cfg(all(windows, feature = "host_env"))]
const HV_PROTOCOL_RAW: libc::c_int = rustpython_host_env::socket::HV_PROTOCOL_RAW as libc::c_int;
#[cfg(all(windows, not(feature = "host_env")))]
const HV_PROTOCOL_RAW: libc::c_int = 1;

#[cfg(all(windows, feature = "host_env"))]
const AF_HYPERV: libc::c_int = rustpython_host_env::socket::AF_HYPERV as libc::c_int;
#[cfg(all(windows, not(feature = "host_env")))]
const AF_HYPERV: libc::c_int = windows_sys::Win32::Networking::WinSock::AF_HYPERV as libc::c_int;

#[cfg(all(windows, feature = "host_env"))]
const AF_BTH: libc::c_int = rustpython_host_env::socket::AF_BLUETOOTH as libc::c_int;
#[cfg(all(windows, not(feature = "host_env")))]
const AF_BTH: libc::c_int = windows_sys::Win32::Devices::Bluetooth::AF_BTH as libc::c_int;

#[cfg(all(windows, feature = "host_env"))]
const BTHPROTO_RFCOMM: libc::c_int = rustpython_host_env::socket::BTHPROTO_RFCOMM as libc::c_int;
#[cfg(all(windows, not(feature = "host_env")))]
const BTHPROTO_RFCOMM: libc::c_int =
    windows_sys::Win32::Devices::Bluetooth::BTHPROTO_RFCOMM as libc::c_int;

#[cfg(all(windows, feature = "host_env"))]
const SIO_TCP_SET_ACK_FREQUENCY: libc::c_int =
    rustpython_host_env::socket::SIO_TCP_SET_ACK_FREQUENCY as libc::c_int;
#[cfg(all(windows, not(feature = "host_env")))]
const SIO_TCP_SET_ACK_FREQUENCY: libc::c_int =
    windows_sys::Win32::Networking::WinSock::SIO_TCP_SET_ACK_FREQUENCY as libc::c_int;

/// `SOCKADDR_HV` (`hvsocket.h`): the family, a reserved word, and the two
/// GUIDs a Hyper-V endpoint is named by.  36 bytes, so it fits a
/// `sockaddr_storage` like every other form here.
#[cfg(windows)]
#[repr(C)]
#[derive(Clone, Copy)]
struct SockaddrHv {
    family: u16,
    reserved: u16,
    vm_id: windows_sys::core::GUID,
    service_id: windows_sys::core::GUID,
}

/// The single message `PyArg_ParseTuple(args, "UU;...")` produces for every
/// shape that is a tuple but not two `str`s — a wrong length included.
#[cfg(windows)]
fn hyperv_address_shape_error() -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::type_error(
        "AF_HYPERV address must be a str tuple (vm_id, service_id)",
    )
}

/// `UuidFromStringW`, which takes the bare 8-4-4-4-12 spelling and nothing
/// else: a braced or truncated GUID is rejected here rather than reinterpreted.
#[cfg(windows)]
fn parse_hyperv_guid(
    caller: &str,
    field: &str,
    w_text: pyre_object::PyObjectRef,
) -> Result<windows_sys::core::GUID, pyre_interpreter::PyError> {
    let units: Vec<u16> = unsafe { pyre_object::w_str_get_wtf8(w_text) }
        .encode_wide()
        .collect();
    let wide = widestring::WideCString::from_vec(units).map_err(|_| {
        pyre_interpreter::PyError::value_error(format!(
            "{caller}(): AF_HYPERV address {field} is not a valid UUID string"
        ))
    })?;
    rustpython_host_env::socket::uuid_from_string_w(&wide).map_err(|_| {
        pyre_interpreter::PyError::value_error(format!(
            "{caller}(): AF_HYPERV address {field} is not a valid UUID string"
        ))
    })
}

/// The spelling `UuidToStringW` gives a GUID: lower case, unbraced.
#[cfg(windows)]
fn hyperv_guid_string(guid: &windows_sys::core::GUID) -> String {
    rustpython_host_env::socket::uuid_to_string_w(guid).unwrap_or_default()
}

/// The `AF_HYPERV` case of `getsockaddrarg` (`socketmodule.c:2643-2712`).
#[cfg(windows)]
fn pack_hyperv_addr(
    caller: &str,
    proto: libc::c_int,
    addr: pyre_object::PyObjectRef,
    storage: &mut rffi::sockaddr_storage,
) -> Result<rffi::SockLen, pyre_interpreter::PyError> {
    if proto != HV_PROTOCOL_RAW {
        return Err(pyre_interpreter::PyError::os_error(format!(
            "{caller}(): unsupported AF_HYPERV protocol: {proto}"
        )));
    }
    if !unsafe { pyre_object::is_tuple(addr) } {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{caller}(): AF_HYPERV address must be tuple, not {}",
            pyre_interpreter::type_methods::arg_type_name(addr)
        )));
    }
    if unsafe { pyre_object::w_tuple_len(addr) } != 2 {
        return Err(hyperv_address_shape_error());
    }
    let w_vm_id = unsafe { pyre_object::w_tuple_getitem(addr, 0) }.expect("length checked above");
    let w_service_id =
        unsafe { pyre_object::w_tuple_getitem(addr, 1) }.expect("length checked above");
    if !unsafe { pyre_object::is_str(w_vm_id) } || !unsafe { pyre_object::is_str(w_service_id) } {
        return Err(hyperv_address_shape_error());
    }
    // Both GUIDs are parsed before either is stored: no step here runs Python,
    // so the two lookups above stay valid across the second parse.
    let vm_id = parse_hyperv_guid(caller, "vm_id", w_vm_id)?;
    let service_id = parse_hyperv_guid(caller, "service_id", w_service_id)?;
    let hv = unsafe { &mut *(storage as *mut _ as *mut SockaddrHv) };
    *hv = SockaddrHv {
        family: AF_HYPERV as u16,
        reserved: 0,
        vm_id,
        service_id,
    };
    Ok(core::mem::size_of::<SockaddrHv>() as rffi::SockLen)
}

/// The `AF_HYPERV` case of `makesockaddr` (`socketmodule.c:1740-1767`) — the
/// two GUIDs as strings, the same shape `bind` and `connect` accept.
#[cfg(windows)]
fn unpack_hyperv_addr(storage: &rffi::sockaddr_storage) -> pyre_object::PyObjectRef {
    let hv = unsafe { &*(storage as *const _ as *const SockaddrHv) };
    let vm_id = hyperv_guid_string(&hv.vm_id);
    let service_id = hyperv_guid_string(&hv.service_id);
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::w_str_new_managed(&vm_id));
    fields.push(pyre_object::w_str_new_managed(&service_id));
    pyre_object::w_tuple_new(fields.take())
}

/// `_PyLong_UInt16_Converter` / `_PyLong_UInt32_Converter`, the argument
/// clinic converters `htons` / `ntohs` / `htonl` / `ntohl` declare.
///
/// `Py_ASNATIVEBYTES_ALLOW_INDEX` takes anything with `__index__`,
/// `Py_ASNATIVEBYTES_REJECT_NEGATIVE` answers a negative value with
/// `ValueError("Cannot convert negative int")`, and a value wider than the C
/// type is an `OverflowError` naming that type.  Narrowing instead would make
/// `htons(70000)` answer for port 4464.
fn c_uint_converter(
    obj: pyre_object::PyObjectRef,
    max: u64,
    ctype: &str,
) -> Result<u64, pyre_interpreter::PyError> {
    let obj = pyre_interpreter::baseobjspace::space_index(obj)?;
    let negative = unsafe {
        if pyre_object::is_long(obj) {
            pyre_object::w_long_get_value(obj).get_sign() < 0
        } else {
            pyre_object::w_int_get_value(obj) < 0
        }
    };
    if negative {
        return Err(pyre_interpreter::PyError::value_error(
            "Cannot convert negative int",
        ));
    }
    match pyre_interpreter::baseobjspace::uint_w(obj) {
        Ok(value) if value <= max => Ok(value),
        _ => Err(pyre_interpreter::PyError::overflow_error(format!(
            "Python int too large for C {ctype}"
        ))),
    }
}

/// The failure `getaddrinfo` / `getnameinfo` report through their return code
/// rather than through `errno`.
///
/// `set_gaierror` spells the message `gai_strerror(error)` only where
/// `HAVE_GAI_STRERROR` is defined, and the Windows build does not define it —
/// `socket` publishes no `gai_strerror` there either — so the message is the
/// fixed `"getaddrinfo failed"`.  `rsocket.GAIError.get_msg` answers
/// `_rsocket_rffi.gai_strerror_str`, which on Windows is
/// `rwin32.FormatError(errno)`; that spelling is the system message table's,
/// so it comes back in the host's UI language.  MEASURED 2026-08-27 against
/// CPython 3.14.2 on a ko-KR host: `socket.getaddrinfo` on an unresolvable
/// name answers `gaierror(11001, 'getaddrinfo failed')` there and
/// `gaierror(11001, '<localized>')` under PyPy 7.3.22.
#[cfg(any(unix, windows))]
fn set_gaierror(error: libc::c_int) -> pyre_interpreter::PyError {
    #[cfg(unix)]
    let message = rffi::gai_strerror(error);
    #[cfg(windows)]
    let message = "getaddrinfo failed".to_string();
    socket_converted_error("gaierror", Some(error), &message)
}

/// The failure `gethostbyname` / `gethostbyaddr` report, which they leave in
/// `h_errno` rather than `errno`.
///
/// `set_herror` raises `herror(h_errno, hstrerror(h_errno))`.  `HSocketError`
/// carries only the host name, because rsocket says `h_errno` is not reachable
/// from RPython — a caller that classifies on `.errno` sees nothing there, so
/// the code the resolver set is read here.
#[cfg(any(unix, windows))]
fn host_lookup_error() -> pyre_interpreter::PyError {
    let (code, message) = rffi::host_error();
    socket_converted_error("herror", Some(code), &message)
}

/// The length of the `sockaddr` a family stores, or `None` for a family this
/// module has no address form for.
#[cfg(any(unix, windows))]
fn sockaddr_len_of(family: libc::c_int) -> Option<usize> {
    if family == rffi::AF_INET {
        Some(core::mem::size_of::<rffi::sockaddr_in>())
    } else if family == rffi::AF_INET6 {
        Some(core::mem::size_of::<rffi::sockaddr_in6>())
    } else {
        None
    }
}

/// Resolve `host` for `family` and hand back the first answer's `sockaddr`.
/// `AF_UNSPEC` takes whichever of the two families the resolver answers with.
///
/// `rsocket.py makeipaddr` / `setipaddr`: two names never reach the resolver.
/// The empty one is the wildcard — `getaddrinfo(NULL, "0", AI_PASSIVE)` with a
/// dummy socktype, and more than one answer means it did not name a single
/// address.  `<broadcast>` and `255.255.255.255` are IPv4's all-ones address,
/// which `inet_addr` cannot express because it reports failure with the same
/// bit pattern.  Everything else ends in
/// `getaddrinfo(name, None, family=family, address_to_fill=result)`, and
/// `rsocket.py getaddrinfo` answers a non-zero return with
/// `raise GAIError(error)` — so a name that does not resolve reports the
/// `gaierror` `set_gaierror` builds, which is what callers classify on.
/// gethostbyname is not used here: its process-global result buffer is not
/// re-entrant and was corrupted by socketserver worker threads.
#[cfg(any(unix, windows))]
fn resolve_ip_host(
    c_host: &std::ffi::CStr,
    family: libc::c_int,
) -> Result<rffi::sockaddr_storage, pyre_interpreter::PyError> {
    let host = c_host.to_bytes();
    if host == b"<broadcast>" || host == b"255.255.255.255" {
        if family != rffi::AF_INET && family != rffi::AF_UNSPEC {
            return Err(socket_converted_error(
                "error",
                None,
                "address family mismatched",
            ));
        }
        let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
        let sin = unsafe { &mut *(&mut storage as *mut _ as *mut rffi::sockaddr_in) };
        sin.sin_family = rffi::AF_INET as rffi::SaFamily;
        rffi::sockaddr_in_set_addr(sin, rffi::INADDR_BROADCAST);
        return Ok(storage);
    }
    let wildcard = host.is_empty();
    let socktype = if wildcard { rffi::SOCK_DGRAM } else { 0 };
    let flags = if wildcard { rffi::AI_PASSIVE } else { 0 };
    let name = if wildcard { None } else { Some(c_host) };
    let service = if wildcard { Some(c"0") } else { None };
    let list = addrinfo_answers(name, service, family, socktype, 0, flags)?;
    let ambiguous_wildcard = wildcard && list.len() > 1;
    let mut resolved = None;
    for info in &list {
        let wanted = family == rffi::AF_UNSPEC || info.family == i64::from(family);
        if wanted
            && let Some(want) = sockaddr_len_of(info.family as libc::c_int)
            && info.addrlen >= want
        {
            let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&info.storage as *const rffi::sockaddr_storage).cast::<u8>(),
                    (&mut storage as *mut rffi::sockaddr_storage).cast::<u8>(),
                    want,
                );
            }
            resolved = Some(storage);
            break;
        }
    }
    if ambiguous_wildcard {
        return Err(socket_converted_error(
            "error",
            None,
            "wildcard resolved to multiple address",
        ));
    }
    // The hints pin `ai_family`, so a success with no answer of that family
    // is a resolver that ignored them.  `getaddrinfo` returned no code to
    // carry, so this is the plain module error, the way "sockaddr resolved to
    // multiple addresses" is.
    resolved.ok_or_else(|| {
        socket_converted_error(
            "error",
            None,
            "getaddrinfo returned no address of the requested family",
        )
    })
}

#[cfg(any(unix, windows))]
fn pack_inet_addr(
    caller: &str,
    family: libc::c_int,
    proto: libc::c_int,
    addr: pyre_object::PyObjectRef,
) -> Result<(rffi::sockaddr_storage, rffi::SockLen), pyre_interpreter::PyError> {
    // Only the AF_HYPERV form reads these two: `getsockaddrarg` names the
    // calling method in its messages and rejects any protocol but
    // `HV_PROTOCOL_RAW`.
    #[cfg(not(windows))]
    let _ = proto;
    let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
    // AF_UNIX is special: rsocket.py:RSocket.bind/connect accept a bare
    // bytes/str path (or a 1-tuple wrapping the path).  Pull the path
    // out before touching tuple[1], which only the AF_INET/AF_INET6
    // forms guarantee.
    #[cfg(unix)]
    if family == libc::AF_UNIX {
        let path_obj = if unsafe { pyre_object::is_tuple(addr) } {
            unsafe { pyre_object::w_tuple_getitem(addr, 0) }
                .ok_or_else(|| pyre_interpreter::PyError::value_error("address: missing path"))?
        } else {
            addr
        };
        // `interp_socket.py w_address = space.fsencode(w_address)`,
        // with the upstream note that it deliberately avoids `fsencode_w`
        // because Linux allows embedded NULs in an abstract-namespace path.
        // The raw `fsencode` preserves that carve-out.
        let path_bytes_vec: Vec<u8> = unsafe {
            if pyre_object::is_str(path_obj) {
                pyre_interpreter::gateway::fsencode(path_obj)?
            } else if pyre_object::bytesobject::is_bytes_like(path_obj) {
                pyre_object::bytesobject::bytes_like_data(path_obj).to_vec()
            } else {
                return Err(pyre_interpreter::PyError::type_error(
                    "AF_UNIX address must be a string or bytes path",
                ));
            }
        };
        let sun = unsafe { &mut *(&mut storage as *mut _ as *mut libc::sockaddr_un) };
        sun.sun_family = libc::AF_UNIX as rffi::SaFamily;
        // The Linux abstract namespace has two members: an address that
        // starts with a null byte, and the empty address, which asks the
        // kernel to auto-bind an abstract one.  Both are named by their
        // length alone, so neither carries a terminator.
        let abstract_name =
            cfg!(target_os = "linux") && matches!(path_bytes_vec.first(), None | Some(&0));
        // `rsocket.py UNIXAddress.__init__`: an abstract name may fill
        // `sun_path` exactly, a regular one has to leave room for its
        // terminator.
        let capacity = sun.sun_path.len() - usize::from(!abstract_name);
        if path_bytes_vec.len() > capacity {
            return Err(pyre_interpreter::PyError::os_error("AF_UNIX path too long"));
        }
        for (i, &b) in path_bytes_vec.iter().enumerate() {
            sun.sun_path[i] = b as libc::c_char;
        }
        // `rsocket.py self.setdata(sun, baseofs + len(path))`: the
        // terminator is counted only for a regular name that has one.
        let addrlen = SUN_PATH_OFFSET + path_bytes_vec.len() + usize::from(!abstract_name);
        return Ok((storage, addrlen as rffi::SockLen));
    }

    #[cfg(windows)]
    if family == AF_HYPERV {
        let addrlen = pack_hyperv_addr(caller, proto, addr, &mut storage)?;
        return Ok((storage, addrlen));
    }

    #[cfg(windows)]
    if family == AF_BTH {
        let addrlen = pack_bluetooth_addr(caller, proto, addr, &mut storage)?;
        return Ok((storage, addrlen));
    }

    if !unsafe { pyre_object::is_tuple(addr) } {
        // [3.14-spec] PyPy's tuple unpacking exposes its generic iterable
        // error here; 3.14 names the operation and rejected type
        // (`testSendtoErrors`).  Packing still follows PyPy's owner.
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{caller}(): AF_INET address must be tuple, not {}",
            pyre_interpreter::type_methods::arg_type_name(addr)
        )));
    }
    // `getsockaddrarg` parses the AF_INET form with `"O&i"` — exactly two
    // items — and the AF_INET6 form with `"O&i|II"` — two to four.
    let len = unsafe { pyre_object::w_tuple_len(addr) };
    if family == rffi::AF_INET && len != 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "AF_INET address must be a pair (host, port)",
        ));
    }
    if family == rffi::AF_INET6 && !(2..=4).contains(&len) {
        return Err(pyre_interpreter::PyError::type_error(
            "AF_INET6 address must be a tuple (host, port[, flowinfo[, scopeid]])",
        ));
    }
    let host_obj = unsafe { pyre_object::w_tuple_getitem(addr, 0) }
        .ok_or_else(|| pyre_interpreter::PyError::value_error("address: missing host"))?;
    // `idna_converter` runs the codec for a host that is not a plain ASCII
    // `str`, which is Python and can collect, so the tuple is pinned across
    // the conversion and every later read goes through the slot.
    let _roots = pyre_object::gc_roots::push_roots();
    let addr_slot = pyre_object::gc_roots::pin_roots(&[addr, host_obj]);
    let host = socket_idna_converter(pyre_object::gc_roots::shadow_stack_get(addr_slot + 1))?;
    let addr = pyre_object::gc_roots::shadow_stack_get(addr_slot);
    let port_obj = unsafe { pyre_object::w_tuple_getitem(addr, 1) }
        .ok_or_else(|| pyre_interpreter::PyError::value_error("address: missing port"))?;
    if !unsafe { pyre_object::is_int(port_obj) } {
        return Err(pyre_interpreter::PyError::type_error(
            "address port must be an integer",
        ));
    }
    let port_raw = unsafe { pyre_object::w_int_get_value(port_obj) };
    if !(0..=0xFFFF).contains(&port_raw) {
        // `getsockaddrarg` spells the message with the method it was called
        // for: `bind(): port must be 0-65535.`
        return Err(pyre_interpreter::PyError::overflow_error(format!(
            "{caller}(): port must be 0-65535."
        )));
    }
    let port = (port_raw as u16).to_be();

    // The resolver below releases the GIL, so read what is left of the tuple
    // out of it first; `addr` is a bare reference and a collection that runs
    // meanwhile can leave it behind.
    let tuple_unsigned =
        |index: i64, field: &str, max: i64| -> Result<Option<u32>, pyre_interpreter::PyError> {
            let Some(value) = (unsafe { pyre_object::w_tuple_getitem(addr, index) }) else {
                return Ok(None);
            };
            let value = pyre_interpreter::builtins::space_index_w(
                pyre_interpreter::baseobjspace::space_index(value)?,
            )?;
            if value < 0 || value > max {
                return Err(pyre_interpreter::PyError::overflow_error(format!(
                    "{caller}(): {field} must be 0-{max}."
                )));
            }
            Ok(Some(value as u32))
        };
    let flowinfo = if len >= 3 {
        tuple_unsigned(2, "flowinfo", 0xfffff)?
    } else {
        None
    };
    let scope_id = if len >= 4 {
        tuple_unsigned(3, "scope id", i64::from(u32::MAX))?
    } else {
        None
    };

    // `socket_idna_converter` already refused every null.
    let c_host = std::ffi::CString::new(host).unwrap();
    if family == rffi::AF_INET {
        let sin = unsafe { &mut *(&mut storage as *mut _ as *mut rffi::sockaddr_in) };
        sin.sin_family = rffi::AF_INET as rffi::SaFamily;
        sin.sin_port = port;
        // inet_pton handles both "0.0.0.0" and dotted-quad.
        let parsed = if c_host.as_bytes().is_empty() {
            // RSocket.makeipaddr('', result) uses the wildcard address for
            // bind(), as required by socketserver and socket_helper.bind_port.
            rffi::sockaddr_in_set_addr(sin, rffi::INADDR_ANY);
            true
        } else {
            match majit_rlib::rsocket::inet_pton(rffi::AF_INET, &c_host) {
                Ok(packed) => {
                    let mut bytes = [0u8; 4];
                    bytes.copy_from_slice(&packed);
                    rffi::sockaddr_in_set_addr(sin, u32::from_ne_bytes(bytes));
                    true
                }
                Err(_) => false,
            }
        };
        if !parsed {
            let found = resolve_ip_host(&c_host, rffi::AF_INET)?;
            let found = unsafe { &*(&found as *const _ as *const rffi::sockaddr_in) };
            rffi::sockaddr_in_set_addr(sin, rffi::sockaddr_in_get_addr(found));
        }
        Ok((
            storage,
            core::mem::size_of::<rffi::sockaddr_in>() as rffi::SockLen,
        ))
    } else if family == rffi::AF_INET6 {
        let sin6 = unsafe { &mut *(&mut storage as *mut _ as *mut rffi::sockaddr_in6) };
        sin6.sin6_family = rffi::AF_INET6 as rffi::SaFamily;
        sin6.sin6_port = port;
        let mut buf = [0u8; 16];
        let parsed = if c_host.as_bytes().is_empty() {
            true
        } else {
            match majit_rlib::rsocket::inet_pton(rffi::AF_INET6, &c_host) {
                Ok(packed) => {
                    buf.copy_from_slice(&packed);
                    true
                }
                Err(_) => false,
            }
        };
        if !parsed {
            let found = resolve_ip_host(&c_host, rffi::AF_INET6)?;
            let found = unsafe { &*(&found as *const _ as *const rffi::sockaddr_in6) };
            buf = rffi::sockaddr_in6_get_addr(found);
            // `makeipaddr` fills the whole INET6Address from the answer, so
            // the scope id the resolver picked stands unless items 2 and 3 of
            // the tuple below name their own.
            rffi::sockaddr_in6_set_scope_id(sin6, rffi::sockaddr_in6_get_scope_id(found));
        }
        rffi::sockaddr_in6_set_addr(sin6, buf);
        if let Some(v) = flowinfo {
            // `rsocket.py:INET6Address.__init__` stores flowinfo in network
            // byte order just like the port.
            sin6.sin6_flowinfo = v.to_be();
        }
        if let Some(v) = scope_id {
            rffi::sockaddr_in6_set_scope_id(sin6, v);
        }
        Ok((
            storage,
            core::mem::size_of::<rffi::sockaddr_in6>() as rffi::SockLen,
        ))
    } else {
        Err(pyre_interpreter::PyError::os_error(format!(
            "unsupported address family: {family}"
        )))
    }
}

#[cfg(unix)]
fn storage_from_rsocket(bytes: &[u8], slen: i32) -> (rffi::sockaddr_storage, rffi::SockLen) {
    let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
    let n = bytes.len().min(std::mem::size_of_val(&storage));
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), &mut storage as *mut _ as *mut u8, n);
    }
    (storage, slen as rffi::SockLen)
}

/// `accept`, retrying `EINTR`. `FD_CLOEXEC` is set on the new descriptor.
#[cfg(unix)]
fn accept_socket(
    obj: pyre_object::PyObjectRef,
    fd: rffi::Socket,
) -> Result<(rffi::Socket, rffi::sockaddr_storage, rffi::SockLen), pyre_interpreter::PyError> {
    loop {
        match majit_rlib::rsocket::accept(fd) {
            Ok((cfd, bytes, addrlen)) => {
                let (storage, slen) = storage_from_rsocket(&bytes, addrlen);
                return Ok((cfd, storage, slen));
            }
            Err(error) if rffi::error_is_interrupted(error.errno) => {
                pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
            }
            Err(error) => {
                return Err(socket_io_err_for_operation(
                    obj,
                    std::io::Error::from_raw_os_error(error.errno),
                ));
            }
        }
    }
}

/// `connect`. The failure code is the saved `errno`.
#[cfg(unix)]
fn connect_sockaddr(
    fd: rffi::Socket,
    storage: &rffi::sockaddr_storage,
    slen: rffi::SockLen,
) -> Result<(), i32> {
    let bytes =
        unsafe { std::slice::from_raw_parts(storage as *const _ as *const u8, slen as usize) };
    majit_rlib::rsocket::connect(fd, bytes).map_err(|error| error.errno)
}

/// `recvfrom`, retrying `EINTR`. The address is a `sockaddr_storage`.
#[cfg(unix)]
fn recvfrom_socket(
    obj: pyre_object::PyObjectRef,
    fd: rffi::Socket,
    buf: &mut [u8],
    flags: libc::c_int,
) -> Result<(usize, rffi::sockaddr_storage, rffi::SockLen), pyre_interpreter::PyError> {
    loop {
        match majit_rlib::rsocket::recvfrom(fd, buf, flags) {
            Ok((read, bytes, addrlen)) => {
                let (storage, slen) = storage_from_rsocket(&bytes, addrlen);
                return Ok((read, storage, slen));
            }
            Err(error) if rffi::error_is_interrupted(error.errno) => {
                pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
            }
            Err(error) => {
                return Err(socket_io_err_for_operation(
                    obj,
                    std::io::Error::from_raw_os_error(error.errno),
                ));
            }
        }
    }
}

#[cfg(any(unix, windows))]
fn unpack_inet_addr(
    storage: &rffi::sockaddr_storage,
    addrlen: rffi::SockLen,
) -> pyre_object::PyObjectRef {
    let family = storage.ss_family as libc::c_int;
    if family == rffi::AF_INET {
        let sin = unsafe { &*(storage as *const _ as *const rffi::sockaddr_in) };
        let packed = rffi::sockaddr_in_get_addr(sin).to_ne_bytes();
        let host = majit_rlib::rsocket::inet_ntop(rffi::AF_INET, &packed).unwrap_or_default();
        let port = u16::from_be(sin.sin_port) as i64;
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(pyre_object::w_str_new_managed(&host));
        fields.push(pyre_object::w_int_new(port));
        pyre_object::w_tuple_new(fields.take())
    } else if family == rffi::AF_INET6 {
        let sin6 = unsafe { &*(storage as *const _ as *const rffi::sockaddr_in6) };
        let packed = rffi::sockaddr_in6_get_addr(sin6);
        let host = majit_rlib::rsocket::inet_ntop(rffi::AF_INET6, &packed).unwrap_or_default();
        let port = u16::from_be(sin6.sin6_port) as i64;
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(pyre_object::w_str_new_managed(&host));
        fields.push(pyre_object::w_int_new(port));
        fields.push(pyre_object::w_int_new(
            u32::from_be(sin6.sin6_flowinfo) as i64
        ));
        fields.push(pyre_object::w_int_new(
            rffi::sockaddr_in6_get_scope_id(sin6) as i64,
        ));
        pyre_object::w_tuple_new(fields.take())
    } else {
        #[cfg(unix)]
        if family == libc::AF_UNIX {
            return unpack_unix_addr(storage, addrlen);
        }
        #[cfg(windows)]
        if family == AF_HYPERV {
            return unpack_hyperv_addr(storage);
        }
        #[cfg(windows)]
        if family == AF_BTH {
            return unpack_bluetooth_addr(storage);
        }
        pyre_object::w_tuple_new(vec![])
    }
}

/// `interp_socket.py:40-47` — the `sockaddr_un` half of `unpack_inet_addr`.
#[cfg(unix)]
fn unpack_unix_addr(
    storage: &rffi::sockaddr_storage,
    addrlen: rffi::SockLen,
) -> pyre_object::PyObjectRef {
    let sun = unsafe { &*(storage as *const _ as *const libc::sockaddr_un) };
    let maxlength = (addrlen as usize)
        .saturating_sub(SUN_PATH_OFFSET)
        .min(sun.sun_path.len());
    let abstract_name = cfg!(target_os = "linux") && maxlength > 0 && sun.sun_path[0] == 0;
    let end = if abstract_name {
        maxlength
    } else {
        sun.sun_path[..maxlength]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(maxlength)
    };
    let bytes: Vec<u8> = sun.sun_path[..end].iter().map(|&b| b as u8).collect();
    if abstract_name {
        // `interp_socket.py space.newbytes(path)`: abstract names are bytes.
        pyre_object::bytesobject::w_bytes_from_bytes(&bytes)
    } else {
        // `interp_socket.py space.newfilename(path)`: read-back uses the filesystem
        // decoding so a byte with no UTF-8 spelling survives the round trip.
        pyre_interpreter::gateway::fsdecode_filename_bytes(&bytes)
    }
}

/// interp2app wrapper for `socket.bind`. Loop-free, so a trace looks
/// through to `socketbind`.
#[cfg(any(unix, windows))]
pub fn __majit_wrap_socket_bind(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.len() < 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "bind() missing address",
        ));
    }
    let mut obj = args[0];
    let mut w_addr = args[1];
    let fd = pyre_object::with_roots!(obj, w_addr => socket_fd(obj))?;
    let family =
        pyre_object::with_roots!(obj, w_addr => socket_get_attr_i64(obj, "_family")) as libc::c_int;
    let proto =
        pyre_object::with_roots!(w_addr => socket_get_attr_i64(obj, "_proto")) as libc::c_int;
    let (storage, slen) = pack_inet_addr("bind", family, proto, w_addr)?;
    #[cfg(unix)]
    {
        let bytes =
            unsafe { std::slice::from_raw_parts(&storage as *const _ as *const u8, slen as usize) };
        majit_rlib::rsocket::bind(fd, bytes).map_err(rsocket_os_error)?;
        return Ok(pyre_object::w_none());
    }
    #[cfg(windows)]
    {
        let r = unsafe { rffi::bind(fd, &storage as *const _ as *const rffi::sockaddr, slen) };
        if r != 0 {
            return Err(socket_last_error());
        }
        Ok(pyre_object::w_none())
    }
}

#[cfg(any(unix, windows))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_socket_bind,
    __majit_wrap_socket_bind
);

/// interp2app wrapper for `socket.listen`. Loop-free, so a trace looks
/// through to `socketlisten`.
#[cfg(any(unix, windows))]
pub fn __majit_wrap_socket_listen(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let mut w_backlog = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let fd = pyre_object::with_roots!(w_backlog => socket_fd(obj))?;
    let backlog = if args.len() >= 2 {
        pyre_interpreter::baseobjspace::c_int_w(w_backlog)?
    } else {
        128
    };
    #[cfg(unix)]
    {
        majit_rlib::rsocket::listen(fd, backlog).map_err(rsocket_os_error)?;
        return Ok(pyre_object::w_none());
    }
    #[cfg(windows)]
    {
        let r = unsafe { rffi::listen(fd, backlog) };
        if r != 0 {
            return Err(socket_last_error());
        }
        Ok(pyre_object::w_none())
    }
}

#[cfg(any(unix, windows))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_socket_listen,
    __majit_wrap_socket_listen
);

/// interp2app wrapper for `socket.shutdown`. Loop-free, so a trace looks
/// through to `socketshutdown`.
#[cfg(any(unix, windows))]
pub fn __majit_wrap_socket_shutdown(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.len() < 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "shutdown() missing how",
        ));
    }
    let w_self = args[0];
    let mut w_how = args[1];
    let fd = pyre_object::with_roots!(w_how => socket_fd(w_self))?;
    let how = pyre_interpreter::baseobjspace::c_int_w(w_how)?;
    #[cfg(unix)]
    {
        majit_rlib::rsocket::shutdown(fd, how).map_err(rsocket_os_error)?;
        return Ok(pyre_object::w_none());
    }
    #[cfg(windows)]
    {
        let r = unsafe { rffi::shutdown(fd, how) };
        if r != 0 {
            return Err(socket_last_error());
        }
        Ok(pyre_object::w_none())
    }
}

#[cfg(any(unix, windows))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_socket_shutdown,
    __majit_wrap_socket_shutdown
);

/// interp2app wrapper for `socket.getsockname`. Loop-free, so a trace
/// looks through to `socketgetsockname`.
#[cfg(any(unix, windows))]
pub fn __majit_wrap_socket_getsockname(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let fd = socket_fd(args.first().copied().unwrap_or(pyre_object::PY_NULL))?;
    #[cfg(unix)]
    {
        let (bytes, slen) = majit_rlib::rsocket::getsockname(fd)
            .map_err(|error| socket_io_err(std::io::Error::from_raw_os_error(error.errno)))?;
        let (storage, slen) = storage_from_rsocket(&bytes, slen);
        return Ok(unpack_inet_addr(&storage, slen));
    }
    #[cfg(windows)]
    {
        let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
        let mut slen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
        let r = unsafe {
            rffi::getsockname(fd, &mut storage as *mut _ as *mut rffi::sockaddr, &mut slen)
        };
        if r != 0 {
            return Err(socket_last_error());
        }
        Ok(unpack_inet_addr(&storage, slen))
    }
}

#[cfg(any(unix, windows))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_socket_getsockname,
    __majit_wrap_socket_getsockname
);

/// interp2app wrapper for `socket.getpeername`. Loop-free, so a trace
/// looks through to `socketgetpeername`.
#[cfg(any(unix, windows))]
pub fn __majit_wrap_socket_getpeername(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let fd = socket_fd(args.first().copied().unwrap_or(pyre_object::PY_NULL))?;
    #[cfg(unix)]
    {
        let (bytes, slen) = majit_rlib::rsocket::getpeername(fd)
            .map_err(|error| socket_io_err(std::io::Error::from_raw_os_error(error.errno)))?;
        let (storage, slen) = storage_from_rsocket(&bytes, slen);
        return Ok(unpack_inet_addr(&storage, slen));
    }
    #[cfg(windows)]
    {
        let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
        let mut slen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
        let r = unsafe {
            rffi::getpeername(fd, &mut storage as *mut _ as *mut rffi::sockaddr, &mut slen)
        };
        if r != 0 {
            return Err(socket_last_error());
        }
        Ok(unpack_inet_addr(&storage, slen))
    }
}

#[cfg(any(unix, windows))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_socket_getpeername,
    __majit_wrap_socket_getpeername
);

/// interp2app wrapper for `socket.setsockopt`. Loop-free, so a trace
/// looks through to `socketsetsockopt`.
#[cfg(any(unix, windows))]
pub fn __majit_wrap_socket_setsockopt(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.len() < 4 {
        return Err(pyre_interpreter::PyError::type_error(
            "setsockopt() requires self + level + name + value",
        ));
    }
    let mut w_self = args[0];
    let mut w_level = args[1];
    let mut w_name = args[2];
    let mut val = args[3];
    let fd = pyre_object::with_roots!(w_self, w_level, w_name, val => socket_fd(w_self))?;
    let level = pyre_object::with_roots!(w_self, w_name, val =>
        pyre_interpreter::baseobjspace::c_int_w(w_level)
    )?;
    let name = pyre_object::with_roots!(w_self, val =>
        pyre_interpreter::baseobjspace::c_int_w(w_name)
    )?;
    // `sock_setsockopt` sends an int for this option through
    // `WSAIoctl` and keeps the value, because the option number is an
    // ioctl code that `setsockopt` itself rejects.  A bytes value is
    // not covered there either and still goes the ordinary way.
    #[cfg(windows)]
    if name == SIO_TCP_SET_ACK_FREQUENCY && unsafe { pyre_object::is_int(val) } {
        let flag = (unsafe { pyre_object::w_int_get_value(val) }) as libc::c_int;
        if unsafe { rffi::set_ack_frequency(fd, flag) } != 0 {
            return Err(socket_last_error());
        }
        socket_set_attr(w_self, "_quickack", pyre_object::w_int_new(flag as i64));
        return Ok(pyre_object::w_none());
    }
    #[cfg(unix)]
    {
        unsafe {
            if pyre_object::is_int(val) {
                let v = pyre_object::w_int_get_value(val) as libc::c_int;
                majit_rlib::rsocket::setsockopt(fd, level, name, &v.to_ne_bytes())
                    .map_err(rsocket_os_error)?;
            } else if pyre_object::bytesobject::is_bytes_like(val) {
                // Copied before the call: `setsockopt` releases the
                // interpreter, and the bytes object can move.
                let data = pyre_object::bytesobject::bytes_like_data(val).to_vec();
                majit_rlib::rsocket::setsockopt(fd, level, name, &data)
                    .map_err(rsocket_os_error)?;
            } else {
                return Err(pyre_interpreter::PyError::type_error(
                    "setsockopt: value must be int or bytes-like",
                ));
            }
        }
        return Ok(pyre_object::w_none());
    }
    #[cfg(windows)]
    {
        let r = unsafe {
            if pyre_object::is_int(val) {
                let v = pyre_object::w_int_get_value(val) as libc::c_int;
                rffi::setsockopt(
                    fd,
                    level,
                    name,
                    &v as *const _ as *const libc::c_void,
                    core::mem::size_of::<libc::c_int>() as rffi::SockLen,
                )
            } else if pyre_object::bytesobject::is_bytes_like(val) {
                // Copied before the call: `setsockopt` releases the
                // interpreter, and the bytes object can move.
                let data = pyre_object::bytesobject::bytes_like_data(val).to_vec();
                rffi::setsockopt(
                    fd,
                    level,
                    name,
                    data.as_ptr() as *const libc::c_void,
                    data.len() as rffi::SockLen,
                )
            } else {
                return Err(pyre_interpreter::PyError::type_error(
                    "setsockopt: value must be int or bytes-like",
                ));
            }
        };
        if r != 0 {
            return Err(socket_last_error());
        }
        Ok(pyre_object::w_none())
    }
}

#[cfg(any(unix, windows))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_socket_setsockopt,
    __majit_wrap_socket_setsockopt
);

/// interp2app wrapper for `socket.getsockopt`. Loop-free, so a trace
/// looks through to `socketgetsockopt`.
#[cfg(any(unix, windows))]
pub fn __majit_wrap_socket_getsockopt(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.len() < 3 {
        return Err(pyre_interpreter::PyError::type_error(
            "getsockopt() requires self + level + name [+ buflen]",
        ));
    }
    let mut w_self = args[0];
    let mut w_level = args[1];
    let mut w_name = args[2];
    let mut w_buflen = args.get(3).copied().unwrap_or(pyre_object::PY_NULL);
    let fd = pyre_object::with_roots!(w_self, w_level, w_name, w_buflen => socket_fd(w_self))?;
    let level = pyre_object::with_roots!(w_self, w_name, w_buflen =>
        pyre_interpreter::baseobjspace::c_int_w(w_level)
    )?;
    let name = pyre_object::with_roots!(w_self, w_buflen =>
        pyre_interpreter::baseobjspace::c_int_w(w_name)
    )?;
    // `interp_socket.getsockopt_w` — `buflen == 0` (including when
    // omitted) reads an int option; otherwise the length must be in
    // `1..=1024` and a bytes buffer is returned.
    let buflen = if args.len() >= 4 {
        pyre_object::with_roots!(w_self => pyre_interpreter::baseobjspace::c_int_w(w_buflen))?
            as i64
    } else {
        0
    };
    if buflen == 0 {
        // `sock_getsockopt` answers the value `setsockopt` last wrote
        // rather than asking WinSock, which has no call that reads an
        // ioctl's current setting.
        #[cfg(windows)]
        if name == SIO_TCP_SET_ACK_FREQUENCY {
            return Ok(pyre_object::w_int_new(socket_get_attr_i64(
                w_self,
                "_quickack",
            )));
        }
        #[cfg(unix)]
        {
            let v = socket_getsockopt_int(fd, level, name)?;
            return Ok(pyre_object::w_int_new(v as i64));
        }
        #[cfg(windows)]
        {
            let mut v: libc::c_int = 0;
            let mut sz = core::mem::size_of::<libc::c_int>() as rffi::SockLen;
            let r = unsafe {
                rffi::getsockopt(
                    fd,
                    level,
                    name,
                    &mut v as *mut _ as *mut libc::c_void,
                    &mut sz,
                )
            };
            if r != 0 {
                return Err(socket_last_error());
            }
            Ok(pyre_object::w_int_new(v as i64))
        }
    } else {
        if !(0..=1024).contains(&buflen) {
            return Err(pyre_interpreter::PyError::os_error(
                "getsockopt buflen out of range",
            ));
        }
        let buflen = buflen as usize;
        #[cfg(unix)]
        {
            let buf = majit_rlib::rsocket::getsockopt(fd, level, name, buflen)
                .map_err(rsocket_os_error)?;
            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&buf));
        }
        #[cfg(windows)]
        {
            let mut buf = vec![0u8; buflen];
            let mut sz = buflen as rffi::SockLen;
            let r = unsafe {
                rffi::getsockopt(
                    fd,
                    level,
                    name,
                    buf.as_mut_ptr() as *mut libc::c_void,
                    &mut sz,
                )
            };
            if r != 0 {
                return Err(socket_last_error());
            }
            buf.truncate(sz as usize);
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&buf))
        }
    }
}

#[cfg(any(unix, windows))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_socket_getsockopt,
    __majit_wrap_socket_getsockopt
);

#[cfg(any(unix, windows))]
fn init_socket_type(ns: pyre_object::PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // `interp_socket.py:W_Socket.__init__` first creates an empty wrapper;
    // `descr_init` below installs the RSocket state.  Keeping allocation and
    // initialisation separate is required when socket.py's Python subclass
    // explicitly calls `_socket.socket.__init__(self, ...)`.
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "__new__",
            pyre_interpreter::typedef::make_new_descr(|args| {
                let cls = args
                    .first()
                    .copied()
                    .filter(|w_cls| pyre_object::is_type(*w_cls))
                    .unwrap_or_else(socket_type);
                Ok(pyre_object::w_instance_new(cls))
            })
        )
    };
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "__del__",
            pyre_interpreter::make_builtin_function_with_arity(
                "__del__",
                |args| {
                    let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                    let fd = rffi::socket_from_i64(
                        pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_fd")),
                    );
                    if !rffi::is_invalid(fd) {
                        if let Ok(repr) = pyre_object::with_roots!(obj => { pyre_interpreter::display::py_repr_wtf8(obj) })
                        {
                            let _ = pyre_object::with_roots!(obj => pyre_interpreter::warn::warn_category_source(
                                &format!("unclosed {}", repr.to_string_lossy()),
                                "ResourceWarning",
                                1,
                                obj,
                            ));
                        }
                        #[cfg(unix)]
                        {
                            let _ = majit_rlib::rsocket::close(fd);
                        }
                        #[cfg(windows)]
                        {
                            let _ = { rffi::close(fd) };
                        }
                        socket_set_attr(obj, "_fd", pyre_object::w_int_new(-1));
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            )
        )
    };
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "__init__",
            pyre_interpreter::make_builtin_function("__init__", |args| {
                // `interp_socket.py descr_init(family=-1, type=-1, proto=-1,
                // w_fileno=None)`.
                let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let after_self = if args.is_empty() { args } else { &args[1..] };
                let (pos, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(after_self);
                // `descr_init`'s `@unwrap_spec` signature rejects unknown
                // keywords and a parameter supplied both by position and name.
                pyre_interpreter::builtins::kwarg_reject_unknown(
                    kwargs,
                    &["family", "type", "proto", "fileno"],
                    "socket",
                )?;
                pyre_interpreter::builtins::kwarg_reject_duplicate(
                    kwargs,
                    "socket",
                    "family",
                    !pos.is_empty(),
                )?;
                pyre_interpreter::builtins::kwarg_reject_duplicate(
                    kwargs,
                    "socket",
                    "type",
                    pos.len() >= 2,
                )?;
                pyre_interpreter::builtins::kwarg_reject_duplicate(
                    kwargs,
                    "socket",
                    "proto",
                    pos.len() >= 3,
                )?;
                pyre_interpreter::builtins::kwarg_reject_duplicate(
                    kwargs,
                    "socket",
                    "fileno",
                    pos.len() >= 4,
                )?;
                // `interp_socket.py descr_init(family=-1, type=-1, proto=-1,
                // w_fileno=None)` — each parameter comes from its positional
                // slot, then its keyword; family/type/proto keep the sentinel
                // -1 (resolved below from the module defaults or the fd).
                let family_obj = pos
                    .first()
                    .copied()
                    .or_else(|| pyre_interpreter::builtins::kwarg_get(kwargs, "family"));
                let type_obj = pos
                    .get(1)
                    .copied()
                    .or_else(|| pyre_interpreter::builtins::kwarg_get(kwargs, "type"));
                let proto_obj = pos
                    .get(2)
                    .copied()
                    .or_else(|| pyre_interpreter::builtins::kwarg_get(kwargs, "proto"));
                let fileno_obj = pos
                    .get(3)
                    .copied()
                    .or_else(|| pyre_interpreter::builtins::kwarg_get(kwargs, "fileno"));
                // `@unwrap_spec(family=int, type=int, proto=int)` — a present
                // argument goes through the gateway int converter (`__index__` /
                // `__int__`, OverflowError if it does not fit), defaulting to the
                // -1 sentinel when omitted.
                let int_arg =
                |obj: Option<pyre_object::PyObjectRef>| -> Result<libc::c_int, pyre_interpreter::PyError> {
                match obj {
                    Some(o) => Ok(pyre_interpreter::baseobjspace::int_w(o)? as libc::c_int),
                    None => Ok(-1),
                }
            };
                // Each `int_w` can collect. Read every Option first, then keep
                // the socket and the four words in one pin for the rest of init.
                let has_family = family_obj.is_some();
                let family_word = family_obj.unwrap_or(pyre_object::PY_NULL);
                let has_type = type_obj.is_some();
                let type_word = type_obj.unwrap_or(pyre_object::PY_NULL);
                let has_proto = proto_obj.is_some();
                let proto_word = proto_obj.unwrap_or(pyre_object::PY_NULL);
                let has_fileno_obj = fileno_obj.is_some();
                let fileno_word = fileno_obj.unwrap_or(pyre_object::PY_NULL);
                let has_fileno = has_fileno_obj && !pyre_object::is_none(fileno_word);
                let init_roots = pyre_object::gc_roots::push_roots();
                let init_base =
                    init_roots.pin_roots(&[obj, family_word, type_word, proto_word, fileno_word]);
                let mut family = int_arg(has_family.then(|| init_roots.get(init_base + 1)))?;
                let mut ty = int_arg(has_type.then(|| init_roots.get(init_base + 2)))?;
                let mut proto = int_arg(has_proto.then(|| init_roots.get(init_base + 3)))?;
                obj = init_roots.get(init_base);
                if !has_fileno {
                    // `interp_socket.py W_Socket.descr_init` — without a fileno the
                    // sentinels resolve to AF_INET / SOCK_STREAM / 0.
                    if family == -1 {
                        family = rffi::AF_INET;
                    }
                    if ty == -1 {
                        ty = rffi::SOCK_STREAM;
                    }
                    if proto == -1 {
                        proto = 0;
                    }
                    // `socket` and the cloexec `fcntl` release the interpreter.
                    // `obj` is reloaded before `socket_init_state` writes it.
                    // The new descriptor is not inherited across exec.
                    #[cfg(unix)]
                    let fd = {
                        pyre_object::with_roots!(obj => {
                            majit_rlib::rsocket::socket(family, ty, proto).map_err(rsocket_os_error)
                        })?
                    };
                    #[cfg(windows)]
                    let fd = {
                        let fd = pyre_object::with_roots!(obj => rffi::socket(family, ty, proto));
                        if rffi::is_invalid(fd) {
                            return Err(socket_last_error());
                        }
                        pyre_object::with_roots!(obj => rffi::set_cloexec(fd));
                        fd
                    };
                    socket_init_state(obj, fd, family, ty, proto)?;
                    return Ok(pyre_object::w_none());
                }
                // A socket handed over by `share` arrives as the bytes of the
                // `WSAPROTOCOL_INFOW` that wrote it, and re-opening it is what
                // `WSASocketW` under `FROM_PROTOCOL_INFO` does.  The three
                // arguments the caller gave are not read at all: the structure
                // names the family, type and protocol itself, so `fromshare`
                // reaches this with `socket(0, 0, 0, info)`.
                #[cfg(all(windows, feature = "host_env"))]
                if pyre_object::is_bytes(init_roots.get(init_base + 4)) {
                    // Copied out before the interpreter is released: the borrow
                    // would not survive a collection running in another thread.
                    let data =
                        { pyre_object::bytesobject::w_bytes_data(init_roots.get(init_base + 4)) }
                            .to_vec();
                    let size = rustpython_host_env::socket::protocol_info_size();
                    if data.len() != size {
                        return Err(pyre_interpreter::PyError::value_error(format!(
                            "socket descriptor string has wrong size, should be {size} bytes."
                        )));
                    }
                    let shared = {
                        let _blocked = pyre_interpreter::module::thread::before_external_block();
                        rustpython_host_env::socket::socket_from_share_data(&data)
                    }
                    .map_err(socket_io_err)?;
                    // `before_external_block` lets another thread collect.
                    socket_init_state(
                        init_roots.get(init_base),
                        shared.raw,
                        shared.family,
                        shared.socket_type,
                        shared.protocol,
                    )?;
                    return Ok(pyre_object::w_none());
                }
                // `interp_socket.py W_Socket.descr_init` — wrap an existing fd.  A float
                // fileno is a TypeError, a negative fd a ValueError, and any
                // -1 family/type/proto is derived from the descriptor itself.
                if pyre_object::is_float(init_roots.get(init_base + 4)) {
                    return Err(pyre_interpreter::PyError::type_error(
                        "integer argument expected, got float",
                    ));
                }
                // `interp_socket.py` — `space.int_w(w_fileno)` accepts ints,
                // longs, and objects with `__int__` / `__index__`.
                let fd = pyre_object::with_roots!(obj => {
                    pyre_interpreter::baseobjspace::int_w(init_roots.get(init_base + 4))
                })?;
                if fd < 0 {
                    return Err(pyre_interpreter::PyError::value_error(
                        "negative file descriptor",
                    ));
                }
                let fd = rffi::socket_from_i64(fd);
                // [3.14-spec] PyPy `W_Socket.descr_init` only probes SO_TYPE when
                // `type == -1`, and consequently accepts a regular-file fd when
                // the caller supplies family/type.  The public 3.14
                // `test_socket_fileno_requires_socket_fd` requires ENOTSOCK for
                // both forms.  No JIT/immutability hint covers `descr_init`, so
                // validate the descriptor once while keeping PyPy's field-owner
                // and subsequent inference order unchanged.
                let detected_type = pyre_object::with_roots!(obj => socket_getsockopt_int(fd, rffi::SOL_SOCKET, rffi::SO_TYPE))?;
                if family == -1 {
                    family = pyre_object::with_roots!(obj => socket_detect_family(fd))?;
                }
                if ty == -1 {
                    ty = detected_type;
                }
                if proto == -1 {
                    proto = pyre_object::with_roots!(obj => socket_get_so_protocol(fd))?;
                }
                socket_init_state(obj, fd, family, ty, proto)?;
                Ok(pyre_object::w_none())
            })
        )
    };

    // `interp_socket.py:1157-1160` — `family`/`type`/`proto`/`timeout`
    // are GetSetProperty data descriptors (plain attribute access, not
    // callables).  The getter receives `(descriptor, instance)`, so the
    // socket object is `args[1]`.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "family",
        pyre_interpreter::typedef::make_getset_descriptor_named(
            pyre_interpreter::make_builtin_function_with_arity(
                "family",
                |args| {
                    Ok(pyre_object::w_int_new(socket_get_attr_i64(
                        args[1], "_family",
                    )))
                },
                2,
            ),
            "family",
        )
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "type",
        pyre_interpreter::typedef::make_getset_descriptor_named(
            pyre_interpreter::make_builtin_function_with_arity(
                "type",
                |args| {
                    Ok(pyre_object::w_int_new(socket_get_attr_i64(
                        args[1], "_type",
                    )))
                },
                2,
            ),
            "type",
        )
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "proto",
        pyre_interpreter::typedef::make_getset_descriptor_named(
            pyre_interpreter::make_builtin_function_with_arity(
                "proto",
                |args| {
                    Ok(pyre_object::w_int_new(socket_get_attr_i64(
                        args[1], "_proto",
                    )))
                },
                2,
            ),
            "proto",
        )
    );
    // `interp_socket.py gettimeout_w` — `timeout` is the stored
    // `_timeout` object (float, or `None` when disabled).
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "timeout",
            pyre_interpreter::typedef::make_getset_descriptor_named(
                pyre_interpreter::make_builtin_function_with_arity(
                    "timeout",
                    |args| {
                        let d = pyre_interpreter::baseobjspace::getdict_native(args[1]);
                        if d.is_null() {
                            return Ok(pyre_object::w_none());
                        }
                        Ok({ pyre_object::w_dict_getitem_str(d, "_timeout") }
                            .unwrap_or(pyre_object::w_none()))
                    },
                    2,
                ),
                "timeout",
            )
        )
    };

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "fileno",
        pyre_interpreter::make_builtin_function_with_arity(
            "fileno",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_int_new(socket_get_attr_i64(obj, "_fd")))
            },
            1,
        )
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "close",
        pyre_interpreter::make_builtin_function_with_arity(
            "close",
            |args| {
                let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = rffi::socket_from_i64(
                    pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_fd")),
                );
                if !rffi::is_invalid(fd) {
                    // `rsocket.py:RSocket.close` drops ownership before the
                    // syscall and still reports its failure.  This prevents a
                    // later double close while preserving EBADF/ENOTSOCK.
                    socket_set_attr(obj, "_fd", pyre_object::w_int_new(-1));
                    #[cfg(unix)]
                    majit_rlib::rsocket::close(fd).map_err(rsocket_os_error)?;
                    #[cfg(windows)]
                    if unsafe { rffi::close(fd) } != 0 {
                        return Err(socket_last_error());
                    }
                }
                Ok(pyre_object::w_none())
            },
            1,
        )
    );

    // detach() → returns the fd and forgets it.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "detach",
        pyre_interpreter::make_builtin_function_with_arity(
            "detach",
            |args| {
                let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_fd"));
                socket_set_attr(obj, "_fd", pyre_object::w_int_new(-1));
                Ok(pyre_object::w_int_new(fd))
            },
            1,
        )
    );

    // `interp_socket.py _reuse_w / _drop_w` — refcount methods
    // the app-level `socket._socketobject` wrapper uses to share one
    // underlying fd across `socket.makefile()` file-like aliases.
    // `_reuse` increments the usecount; `_drop` decrements and closes
    // when it reaches zero.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "_reuse",
        pyre_interpreter::make_builtin_function_with_arity(
            "_reuse",
            |args| {
                let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let n = pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_usecount"));
                let n = if n < 0 { 1 } else { n };
                socket_set_attr(obj, "_usecount", pyre_object::w_int_new(n + 1));
                Ok(pyre_object::w_none())
            },
            1,
        )
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "_drop",
        pyre_interpreter::make_builtin_function_with_arity(
            "_drop",
            |args| {
                let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let n = pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_usecount"));
                let n = if n < 0 { 1 } else { n };
                let next = n - 1;
                pyre_object::with_roots!(obj => socket_set_attr(obj, "_usecount", pyre_object::w_int_new(next)));
                if next <= 0 {
                    let fd = rffi::socket_from_i64(
                        pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_fd")),
                    );
                    if !rffi::is_invalid(fd) {
                        #[cfg(unix)]
                        {
                            let _ = majit_rlib::rsocket::close(fd);
                        }
                        #[cfg(windows)]
                        {
                            let _ = unsafe { rffi::close(fd) };
                        }
                        socket_set_attr(obj, "_fd", pyre_object::w_int_new(-1));
                    }
                }
                Ok(pyre_object::w_none())
            },
            1,
        )
    );

    // bind(addr) — addr is (host, port) for AF_INET / (host, port, flowinfo,
    // scopeid) for AF_INET6 / path string for AF_UNIX.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "bind",
        pyre_interpreter::make_builtin_function_with_arity("bind", __majit_wrap_socket_bind, 2,)
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "listen",
        pyre_interpreter::make_builtin_function("listen", __majit_wrap_socket_listen)
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "accept",
        pyre_interpreter::make_builtin_function_with_arity(
            "accept",
            |args| {
                let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = pyre_object::with_roots!(obj => socket_fd(obj))?;
                pyre_object::with_roots!(obj => socket_wait_readable(obj, fd))?;
                let family = pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_family"))
                    as libc::c_int;
                let ty = pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_type"))
                    as libc::c_int;
                let proto = pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_proto"))
                    as libc::c_int;
                #[cfg(unix)]
                let (cfd, storage, slen) = accept_socket(obj, fd)?;
                #[cfg(windows)]
                let (cfd, storage, slen) = {
                    let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
                    let mut slen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
                    let cfd = loop {
                        let (r, errno) = socket_call(|| unsafe {
                            rffi::accept(
                                fd,
                                &mut storage as *mut _ as *mut rffi::sockaddr,
                                &mut slen,
                            )
                        });
                        if !rffi::is_invalid(r) {
                            break r;
                        }
                        if !rffi::error_is_interrupted(errno) {
                            return Err(socket_io_err_for_operation(
                                obj,
                                std::io::Error::from_raw_os_error(errno),
                            ));
                        }
                        // EINTR: deliver a pending signal, then retry
                        // (`converted_error` eintr_retry).
                        pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
                    };
                    // The new descriptor is not inherited across exec.
                    rffi::set_cloexec(cfd);
                    (cfd, storage, slen)
                };
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(socket_from_fd(cfd, family, ty, proto)?);
                fields.push(unpack_inet_addr(&storage, slen));
                Ok(pyre_object::w_tuple_new(fields.take()))
            },
            1,
        )
    );

    // `interp_socket.py socketmethodnames _accept` — primitive
    // returning `(fd, addr)`.  CPython's app-level `socket.py def
    // accept` wraps this to construct the new socket object;
    // pyre's `accept` above bundles both steps for callers that
    // bypass the stdlib wrapper.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "_accept",
        pyre_interpreter::make_builtin_function_with_arity(
            "_accept",
            |args| {
                let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = pyre_object::with_roots!(obj => socket_fd(obj))?;
                pyre_object::with_roots!(obj => socket_wait_readable(obj, fd))?;
                #[cfg(unix)]
                let (cfd, storage, slen) = accept_socket(obj, fd)?;
                #[cfg(windows)]
                let (cfd, storage, slen) = {
                    let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
                    let mut slen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
                    let cfd = loop {
                        let (r, errno) = socket_call(|| unsafe {
                            rffi::accept(
                                fd,
                                &mut storage as *mut _ as *mut rffi::sockaddr,
                                &mut slen,
                            )
                        });
                        if !rffi::is_invalid(r) {
                            break r;
                        }
                        if !rffi::error_is_interrupted(errno) {
                            return Err(socket_io_err_for_operation(
                                obj,
                                std::io::Error::from_raw_os_error(errno),
                            ));
                        }
                        // EINTR: deliver a pending signal, then retry
                        // (`converted_error` eintr_retry).
                        pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
                    };
                    // The new descriptor is not inherited across exec.
                    rffi::set_cloexec(cfd);
                    (cfd, storage, slen)
                };
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(pyre_object::w_int_new(rffi::socket_to_i64(cfd)));
                fields.push(unpack_inet_addr(&storage, slen));
                Ok(pyre_object::w_tuple_new(fields.take()))
            },
            1,
        )
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "connect",
        pyre_interpreter::make_builtin_function_with_arity(
            "connect",
            |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "connect() missing address",
                    ));
                }
                // socket_fd and the attribute reads collect before the address
                // is packed. Pin the argument slice once and read it back.
                let args_roots = pyre_object::gc_roots::push_roots();
                let args_base = args_roots.pin_roots(args);
                let mut obj = args_roots.get(args_base);
                let fd = pyre_object::with_roots!(obj => socket_fd(obj))?;
                let family = pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_family"))
                    as libc::c_int;
                let proto = pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_proto"))
                    as libc::c_int;
                // `pack_inet_addr` runs Python for a host that is not a plain
                // ASCII `str`, so the socket is read back from its slot
                // rather than from the native argument slice, which is
                // only current at entry.
                let _roots = pyre_object::gc_roots::push_roots();
                let obj_slot = pyre_object::gc_roots::pin_roots(&[obj]);
                let (storage, slen) =
                    pack_inet_addr("connect", family, proto, args_roots.get(args_base + 1))?;
                let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
                if let Some(timeout) = socket_positive_timeout(obj) {
                    return match socket_connect_wait(fd, &storage, slen, timeout) {
                        Ok(()) => Ok(pyre_object::w_none()),
                        Err(SocketConnectFailure::Errno(errno)) => {
                            Err(socket_io_err_for_operation(
                                obj,
                                std::io::Error::from_raw_os_error(errno),
                            ))
                        }
                        Err(SocketConnectFailure::Timeout(_)) => {
                            Err(socket_converted_error("timeout", None, "timed out"))
                        }
                        Err(SocketConnectFailure::Exception(error)) => Err(error),
                    };
                }
                loop {
                    #[cfg(unix)]
                    let (r, errno) = match connect_sockaddr(fd, &storage, slen) {
                        Ok(()) => (0, 0),
                        Err(code) => (-1, code),
                    };
                    #[cfg(windows)]
                    let (r, errno) = socket_call(|| unsafe {
                        rffi::connect(fd, &storage as *const _ as *const rffi::sockaddr, slen)
                    });
                    if r == 0 {
                        break;
                    }
                    if !rffi::error_is_interrupted(errno) {
                        return Err(socket_io_err(std::io::Error::from_raw_os_error(errno)));
                    }
                    // EINTR: deliver a pending signal, then retry
                    // (`converted_error` eintr_retry).
                    pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
                }
                Ok(pyre_object::w_none())
            },
            2,
        )
    );

    // connect_ex(address) → errno (no exception on error)
    // `interp_socket.py W_Socket.connect_ex_w` — `try: connect; except` equivalent
    // that returns the errno integer instead of raising OSError.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "connect_ex",
        pyre_interpreter::make_builtin_function_with_arity(
            "connect_ex",
            |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "connect_ex() missing address",
                    ));
                }
                // socket_fd and the attribute reads collect before the address
                // is packed. Pin the argument slice once and read it back.
                let args_roots = pyre_object::gc_roots::push_roots();
                let args_base = args_roots.pin_roots(args);
                let mut obj = args_roots.get(args_base);
                let fd = pyre_object::with_roots!(obj => socket_fd(obj))?;
                let family = pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_family"))
                    as libc::c_int;
                let proto = pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_proto"))
                    as libc::c_int;
                // `pack_inet_addr` runs Python for a host that is not a plain
                // ASCII `str`, so the socket is read back from its slot
                // rather than from the native argument slice, which is
                // only current at entry.
                let _roots = pyre_object::gc_roots::push_roots();
                let obj_slot = pyre_object::gc_roots::pin_roots(&[obj]);
                let (storage, slen) =
                    pack_inet_addr("connect_ex", family, proto, args_roots.get(args_base + 1))?;
                let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
                if let Some(timeout) = socket_positive_timeout(obj) {
                    return match socket_connect_wait(fd, &storage, slen, timeout) {
                        Ok(()) => Ok(pyre_object::w_int_new(0)),
                        Err(SocketConnectFailure::Errno(errno))
                        | Err(SocketConnectFailure::Timeout(errno)) => {
                            Ok(pyre_object::w_int_new(errno as i64))
                        }
                        Err(SocketConnectFailure::Exception(error)) => Err(error),
                    };
                }
                // `interp_socket.py W_Socket.connect_ex_w` — retry while the call is
                // interrupted (EINTR), otherwise return the errno.
                let err = loop {
                    #[cfg(unix)]
                    let (r, e) = match connect_sockaddr(fd, &storage, slen) {
                        Ok(()) => (0, 0),
                        Err(code) => (-1, code),
                    };
                    #[cfg(windows)]
                    let (r, e) = socket_call(|| unsafe {
                        rffi::connect(fd, &storage as *const _ as *const rffi::sockaddr, slen)
                    });
                    if r == 0 {
                        break 0;
                    }
                    if !rffi::error_is_interrupted(e) {
                        break e;
                    }
                    // `interp_socket.py W_Socket.connect_ex_w` — deliver a pending signal, then
                    // retry the connect.
                    pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
                };
                Ok(pyre_object::w_int_new(err as i64))
            },
            2,
        )
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "send",
        pyre_interpreter::make_builtin_function("send", |args| {
            if args.len() < 2 {
                return Err(pyre_interpreter::PyError::type_error(
                    "send() missing buffer",
                ));
            }
            let nargs = args.len();
            let args_roots = pyre_object::gc_roots::push_roots();
            let args_base = args_roots.pin_roots(args);
            let mut obj = args_roots.get(args_base);
            let fd = pyre_object::with_roots!(obj => socket_fd(obj))?;
            // The export owns a bracket of its own, above this one; this one
            // keeps `obj` rooted until the export is released.
            let obj_roots = pyre_object::gc_roots::push_roots();
            let obj_base = obj_roots.pin_roots(&[obj]);
            let acquired =
                pyre_interpreter::baseobjspace::simple_buffer_bytes(args_roots.get(args_base + 1));
            obj = obj_roots.get(obj_base);
            let buffer = acquired?.ok_or_else(|| {
                pyre_interpreter::PyError::type_error("send: buffer must be bytes-like")
            })?;
            let flags = if nargs >= 3 {
                pyre_object::with_roots!(obj => {
                    pyre_interpreter::baseobjspace::c_int_w(args_roots.get(args_base + 2))
                })?
            } else {
                0
            };
            // `PyBuffer_Release` after the call — `SimpleBufferBytes` has no
            // `Drop`, so an export that is never released leaves a `bytearray`
            // argument permanently exported and unable to be resized. The
            // result is held rather than propagated so the release covers the
            // error paths too.
            let result = pyre_object::with_roots!(obj => socket_wait_writable(obj, fd))
                .and_then(|()| socket_send_bytes(obj, fd, buffer.as_bytes(), flags));
            match result {
                Ok(n) => {
                    buffer.release();
                    Ok(pyre_object::w_int_new(n as i64))
                }
                Err(error) => {
                    let mut error = error;
                    let error_slot = error.pin(&obj_roots);
                    buffer.release();
                    error.reload(&obj_roots, error_slot);
                    Err(error)
                }
            }
        })
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "sendall",
        pyre_interpreter::make_builtin_function("sendall", |args| {
            if args.len() < 2 {
                return Err(pyre_interpreter::PyError::type_error(
                    "sendall() missing buffer",
                ));
            }
            let nargs = args.len();
            let args_roots = pyre_object::gc_roots::push_roots();
            let args_base = args_roots.pin_roots(args);
            let mut obj = args_roots.get(args_base);
            let fd = pyre_object::with_roots!(obj => socket_fd(obj))?;
            // The export owns a bracket of its own, above this one; this one
            // keeps `obj` rooted until the export is released.
            let obj_roots = pyre_object::gc_roots::push_roots();
            let obj_base = obj_roots.pin_roots(&[obj]);
            let acquired =
                pyre_interpreter::baseobjspace::simple_buffer_bytes(args_roots.get(args_base + 1));
            obj = obj_roots.get(obj_base);
            let buffer = acquired?.ok_or_else(|| {
                pyre_interpreter::PyError::type_error("sendall: buffer must be bytes-like")
            })?;
            let flags = if nargs >= 3 {
                pyre_object::with_roots!(obj => {
                    pyre_interpreter::baseobjspace::c_int_w(args_roots.get(args_base + 2))
                })?
            } else {
                0
            };
            // The public 3.14 sendall timeout is one operation-wide deadline;
            // PyPy's `RSocket.wait_for_data(True)` supplies the writable wait.
            // Compute it once so handled EINTR cannot restart the full period.
            let deadline =
                pyre_object::with_roots!(obj => socket_positive_timeout(obj)).map(|timeout| {
                    std::time::Instant::now()
                        + std::time::Duration::from_secs_f64(timeout.min(i32::MAX as f64 / 1000.0))
                });
            // See `send` above: the export is released once the borrow of
            // `buffer` ends, on the error path as well.
            let result = (|| -> Result<(), pyre_interpreter::PyError> {
                let buf = buffer.as_bytes();
                let mut off = 0usize;
                while off < buf.len() {
                    if let Some(deadline) = deadline {
                        socket_wait_writable_until(fd, deadline)?;
                    }
                    #[cfg(unix)]
                    let (n, errno) = match majit_rlib::rsocket::send(fd, &buf[off..], flags) {
                        Ok(sent) => (sent, 0),
                        Err(error) => (-1, error.errno),
                    };
                    #[cfg(windows)]
                    let (n, errno) = socket_call(|| unsafe {
                        rffi::send(
                            fd,
                            buf[off..].as_ptr() as *const libc::c_void,
                            buf.len() - off,
                            flags,
                        )
                    });
                    // `rsocket.py:RSocket.sendall` invokes its
                    // `signal_checker` after every attempt, not only after an
                    // EINTR.  SA_RESTART can defer the syscall return until a
                    // socket timeout, but the handler still owns the result.
                    pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
                    if n < 0 {
                        if rffi::error_is_interrupted(errno) {
                            continue;
                        }
                        if deadline.is_some() && rffi::error_is_would_block(errno) {
                            // Readiness can be consumed between poll and send;
                            // keep sendall's operation-wide deadline rather
                            // than exposing the transient race.
                            continue;
                        }
                        return Err(socket_io_err_for_operation(
                            obj,
                            std::io::Error::from_raw_os_error(errno),
                        ));
                    }
                    off += n as usize;
                }
                Ok(())
            })();
            match result {
                Ok(()) => {
                    buffer.release();
                    Ok(pyre_object::w_none())
                }
                Err(error) => {
                    let mut error = error;
                    let error_slot = error.pin(&obj_roots);
                    buffer.release();
                    error.reload(&obj_roots, error_slot);
                    Err(error)
                }
            }
        })
    );

    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "recv",
            pyre_interpreter::make_builtin_function("recv", |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error("recv() missing size"));
                }
                if !{ pyre_object::is_int(args[1]) } {
                    return Err(pyre_interpreter::PyError::type_error(
                        "recv: size must be an integer",
                    ));
                }
                let raw = { pyre_object::w_int_get_value(args[1]) };
                if raw < 0 {
                    return Err(pyre_interpreter::PyError::value_error(
                        "negative buffersize in recv",
                    ));
                }
                let nargs = args.len();
                let args_roots = pyre_object::gc_roots::push_roots();
                let args_base = args_roots.pin_roots(args);
                let mut obj = args_roots.get(args_base);
                let fd = pyre_object::with_roots!(obj => socket_fd(obj))?;
                let n = raw as usize;
                let flags = if nargs >= 3 {
                    let flags_obj = args_roots.get(args_base + 2);
                    if !{ pyre_object::is_int(flags_obj) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "recv: flags must be an integer",
                        ));
                    }
                    ({ pyre_object::w_int_get_value(flags_obj) }) as libc::c_int
                } else {
                    0
                };
                let mut buf = pyre_interpreter::builtins::try_vec_zeroed(n)?;
                pyre_object::with_roots!(obj => socket_wait_readable(obj, fd))?;
                let got = socket_recv_bytes(obj, fd, &mut buf, flags)?;
                buf.truncate(got);
                Ok(pyre_object::bytesobject::w_bytes_from_bytes(&buf))
            })
        )
    };

    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "sendto",
            pyre_interpreter::make_builtin_function("sendto", |args| {
                // sendto(buffer, [flags,] address)
                let nargs = args.len();
                if !(3..=4).contains(&nargs) {
                    return Err(pyre_interpreter::PyError::type_error(format!(
                        "sendto() takes 2 or 3 arguments ({} given)",
                        nargs.saturating_sub(1)
                    )));
                }
                let args_roots = pyre_object::gc_roots::push_roots();
                let args_base = args_roots.pin_roots(args);
                let mut obj = args_roots.get(args_base);
                let fd = pyre_object::with_roots!(obj => socket_fd(obj))?;
                // The export owns a bracket of its own, above this one; this one
                // keeps `obj` rooted until the export is released.
                let obj_roots = pyre_object::gc_roots::push_roots();
                let obj_base = obj_roots.pin_roots(&[obj]);
                let acquired = pyre_interpreter::baseobjspace::simple_buffer_bytes(
                    args_roots.get(args_base + 1),
                );
                obj = obj_roots.get(obj_base);
                let buffer = acquired?.ok_or_else(|| {
                    pyre_interpreter::PyError::type_error(format!(
                        "a bytes-like object is required, not '{}'",
                        pyre_interpreter::type_methods::arg_type_name(
                            args_roots.get(args_base + 1)
                        )
                    ))
                })?;
                // 3-arg form: (buf, flags, addr).  4-arg form: (self, buf, flags, addr).
                // We always take self-as-args[0], so 3 args = (self, buf, addr) [no flags]
                // and 4 args = (self, buf, flags, addr).
                let (flags, addr_index) = if nargs == 3 {
                    (0, 2usize)
                } else {
                    let flags = libc::c_int::try_from(pyre_object::with_roots!(obj => {
                        pyre_interpreter::baseobjspace::int_w(args_roots.get(args_base + 2))
                    })?)
                    .map_err(|_| {
                        pyre_interpreter::PyError::overflow_error(
                            "Python int too large to convert to C int",
                        )
                    })?;
                    (flags, 3usize)
                };
                // See `send` above: the export is released once the borrow of
                // `buffer` ends, on the error path as well. `pack_inet_addr` runs
                // Python, so it is inside the released scope too.
                let result = (|| -> Result<isize, pyre_interpreter::PyError> {
                    let buf = buffer.as_bytes();
                    // `getdict_native` can collect; `sendmsg` reads `_family`
                    // / `_proto` from the live pin, and `pack_inet_addr` runs
                    // Python, so every later use reloads the same slot.
                    let family =
                        socket_get_attr_i64(obj_roots.get(obj_base), "_family") as libc::c_int;
                    let proto =
                        socket_get_attr_i64(obj_roots.get(obj_base), "_proto") as libc::c_int;
                    let (storage, slen) = pack_inet_addr(
                        "sendto",
                        family,
                        proto,
                        args_roots.get(args_base + addr_index),
                    )?;
                    socket_wait_writable(obj_roots.get(obj_base), fd)?;
                    loop {
                        #[cfg(unix)]
                        let (r, errno) = {
                            let bytes = std::slice::from_raw_parts(
                                &storage as *const _ as *const u8,
                                slen as usize,
                            );
                            match majit_rlib::rsocket::sendto(fd, buf, flags, bytes) {
                                Ok(sent) => (sent, 0),
                                Err(error) => (-1, error.errno),
                            }
                        };
                        #[cfg(windows)]
                        let (r, errno) = socket_call(|| {
                            rffi::sendto(
                                fd,
                                buf.as_ptr() as *const libc::c_void,
                                buf.len(),
                                flags,
                                &storage as *const _ as *const rffi::sockaddr,
                                slen,
                            )
                        });
                        if r >= 0 {
                            return Ok(r);
                        }
                        if !rffi::error_is_interrupted(errno) {
                            return Err(socket_io_err_for_operation(
                                obj_roots.get(obj_base),
                                std::io::Error::from_raw_os_error(errno),
                            ));
                        }
                        // EINTR: deliver a pending signal, then retry
                        // (`converted_error` eintr_retry).
                        pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
                    }
                })();
                match result {
                    Ok(n) => {
                        buffer.release();
                        Ok(pyre_object::w_int_new(n as i64))
                    }
                    Err(error) => {
                        let mut error = error;
                        let error_slot = error.pin(&obj_roots);
                        buffer.release();
                        error.reload(&obj_roots, error_slot);
                        Err(error)
                    }
                }
            })
        )
    };

    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "recvfrom",
            pyre_interpreter::make_builtin_function("recvfrom", |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "recvfrom() missing size",
                    ));
                }
                if !{ pyre_object::is_int(args[1]) } {
                    return Err(pyre_interpreter::PyError::type_error(
                        "recvfrom: size must be an integer",
                    ));
                }
                let raw = { pyre_object::w_int_get_value(args[1]) };
                if raw < 0 {
                    return Err(pyre_interpreter::PyError::value_error(
                        "negative buffersize in recvfrom",
                    ));
                }
                let nargs = args.len();
                let args_roots = pyre_object::gc_roots::push_roots();
                let args_base = args_roots.pin_roots(args);
                let mut obj = args_roots.get(args_base);
                let fd = pyre_object::with_roots!(obj => socket_fd(obj))?;
                let n = raw as usize;
                let flags = if nargs >= 3 {
                    let flags_obj = args_roots.get(args_base + 2);
                    if !{ pyre_object::is_int(flags_obj) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "recvfrom: flags must be an integer",
                        ));
                    }
                    ({ pyre_object::w_int_get_value(flags_obj) }) as libc::c_int
                } else {
                    0
                };
                let mut buf = pyre_interpreter::builtins::try_vec_zeroed(n)?;
                pyre_object::with_roots!(obj => socket_wait_readable(obj, fd))?;
                #[cfg(unix)]
                let (got, storage, slen) = recvfrom_socket(obj, fd, &mut buf, flags)?;
                #[cfg(windows)]
                let (got, storage, slen) = {
                    let mut storage: rffi::sockaddr_storage = { std::mem::zeroed() };
                    let mut slen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
                    let got = loop {
                        let (r, errno) = socket_call(|| {
                            rffi::recvfrom(
                                fd,
                                buf.as_mut_ptr() as *mut libc::c_void,
                                n,
                                flags,
                                &mut storage as *mut _ as *mut rffi::sockaddr,
                                &mut slen,
                            )
                        });
                        if r >= 0 {
                            break r;
                        }
                        if !rffi::error_is_interrupted(errno) {
                            return Err(socket_io_err_for_operation(
                                obj,
                                std::io::Error::from_raw_os_error(errno),
                            ));
                        }
                        // EINTR: deliver a pending signal, then retry
                        // (`converted_error` eintr_retry).
                        pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
                    };
                    (got as usize, storage, slen)
                };
                buf.truncate(got);
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(pyre_object::bytesobject::w_bytes_from_bytes(&buf));
                fields.push(unpack_inet_addr(&storage, slen));
                Ok(pyre_object::w_tuple_new(fields.take()))
            })
        )
    };

    // recv_into(buffer, [nbytes, flags]) → nbytes_read
    // `interp_socket.py W_Socket.recv_into_w` — writes directly into a writable
    // bytes-like buffer.  nbytes==0 uses the full buffer length.
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "recv_into",
            pyre_interpreter::make_builtin_function("recv_into", |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "recv_into() missing buffer",
                    ));
                }
                let nargs = args.len();
                let _roots = pyre_object::gc_roots::push_roots();
                // `acquire` pins the buffer into this scope. `socket_fd` and
                // `socket_wait_readable` can run Python, so the socket is
                // reloaded from the same pin; the argument slice is not read
                // again after it.
                let args_base = _roots.pin_roots(args);
                let mut buffer = { SocketWritableBuffer::acquire(_roots.get(args_base + 1)) }?;
                let slot = { buffer.as_mut_slice() };
                let buf_len = slot.len();
                let nbytes = if nargs >= 3 {
                    let nbytes_obj = _roots.get(args_base + 2);
                    if !{ pyre_object::is_int(nbytes_obj) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "recv_into: nbytes must be an integer",
                        ));
                    }
                    let raw = { pyre_object::w_int_get_value(nbytes_obj) };
                    if raw < 0 {
                        return Err(pyre_interpreter::PyError::value_error(
                            "negative buffersize in recv_into",
                        ));
                    }
                    let n = raw as usize;
                    if n == 0 { buf_len } else { n }
                } else {
                    buf_len
                };
                if buf_len < nbytes {
                    return Err(pyre_interpreter::PyError::value_error(
                        "buffer too small for requested bytes",
                    ));
                }
                let flags = if nargs >= 4 {
                    let flags_obj = _roots.get(args_base + 3);
                    if !{ pyre_object::is_int(flags_obj) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "recv_into: flags must be an integer",
                        ));
                    }
                    pyre_object::w_int_get_value(flags_obj) as libc::c_int
                } else {
                    0
                };
                let fd = socket_fd(_roots.get(args_base))?;
                socket_wait_readable(_roots.get(args_base), fd)?;
                let got = socket_recv_bytes(_roots.get(args_base), fd, &mut slot[..nbytes], flags)?;
                Ok(pyre_object::w_int_new(got as i64))
            })
        )
    };

    // recvfrom_into(buffer, [nbytes, flags]) → (nbytes, address)
    // `interp_socket.py W_Socket.recvfrom_into_w` — recvfrom variant that fills a
    // caller-provided buffer rather than allocating a new bytes.
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "recvfrom_into",
            pyre_interpreter::make_builtin_function("recvfrom_into", |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "recvfrom_into() missing buffer",
                    ));
                }
                let nargs = args.len();
                let _roots = pyre_object::gc_roots::push_roots();
                // `acquire` pins the buffer into this scope. `socket_fd` and
                // `socket_wait_readable` can run Python, so the socket is
                // reloaded from the same pin; the argument slice is not read
                // again after it.
                let args_base = _roots.pin_roots(args);
                let mut buffer = { SocketWritableBuffer::acquire(_roots.get(args_base + 1)) }?;
                let slot = { buffer.as_mut_slice() };
                let buf_len = slot.len();
                let nbytes = if nargs >= 3 {
                    let nbytes_obj = _roots.get(args_base + 2);
                    if !{ pyre_object::is_int(nbytes_obj) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "recvfrom_into: nbytes must be an integer",
                        ));
                    }
                    let raw = { pyre_object::w_int_get_value(nbytes_obj) };
                    if raw < 0 {
                        return Err(pyre_interpreter::PyError::value_error(
                            "negative buffersize in recvfrom_into",
                        ));
                    }
                    let n = raw as usize;
                    if n == 0 { buf_len } else { n }
                } else {
                    buf_len
                };
                if nbytes > buf_len {
                    return Err(pyre_interpreter::PyError::value_error(
                        "nbytes is greater than the length of the buffer",
                    ));
                }
                let flags = if nargs >= 4 {
                    let flags_obj = _roots.get(args_base + 3);
                    if !{ pyre_object::is_int(flags_obj) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "recvfrom_into: flags must be an integer",
                        ));
                    }
                    pyre_object::w_int_get_value(flags_obj) as libc::c_int
                } else {
                    0
                };
                let fd = socket_fd(_roots.get(args_base))?;
                socket_wait_readable(_roots.get(args_base), fd)?;
                #[cfg(unix)]
                let (got, storage, slen) =
                    recvfrom_socket(_roots.get(args_base), fd, &mut slot[..nbytes], flags)?;
                #[cfg(windows)]
                let (got, storage, slen) = {
                    let mut storage: rffi::sockaddr_storage = { std::mem::zeroed() };
                    let mut slen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
                    let got = loop {
                        let (r, errno) = socket_call(|| {
                            rffi::recvfrom(
                                fd,
                                slot.as_mut_ptr() as *mut libc::c_void,
                                nbytes,
                                flags,
                                &mut storage as *mut _ as *mut rffi::sockaddr,
                                &mut slen,
                            )
                        });
                        if r >= 0 {
                            break r;
                        }
                        if !rffi::error_is_interrupted(errno) {
                            return Err(socket_io_err_for_operation(
                                _roots.get(args_base),
                                std::io::Error::from_raw_os_error(errno),
                            ));
                        }
                        // EINTR: deliver a pending signal, then retry
                        // (`converted_error` eintr_retry).
                        pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
                    };
                    (got as usize, storage, slen)
                };
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(pyre_object::w_int_new(got as i64));
                fields.push(unpack_inet_addr(&storage, slen));
                Ok(pyre_object::w_tuple_new(fields.take()))
            })
        )
    };

    // recvmsg(bufsize, [ancbufsize, flags]) → (data, ancdata, msg_flags, address)
    // `interp_socket.py:525-569` — receives normal + ancillary data
    // via libc::recvmsg.  ancdata is a list of (cmsg_level, cmsg_type,
    // cmsg_data:bytes) triples walked through CMSG_FIRSTHDR /
    // CMSG_NXTHDR / CMSG_DATA.
    // The scatter/gather calls and their ancillary data are POSIX-only;
    // `socket.py` test for each before reaching for it.
    #[cfg(unix)]
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "recvmsg",
            pyre_interpreter::make_builtin_function("recvmsg", |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "recvmsg() missing buffer size",
                    ));
                }
                if !{ pyre_object::is_int(args[1]) } {
                    return Err(pyre_interpreter::PyError::type_error(
                        "recvmsg: bufsize must be an integer",
                    ));
                }
                let bufsize_raw = { pyre_object::w_int_get_value(args[1]) };
                if bufsize_raw < 0 {
                    return Err(pyre_interpreter::PyError::value_error(
                        "negative buffer size in recvmsg()",
                    ));
                }
                let bufsize = bufsize_raw as usize;
                let ancbufsize = if args.len() >= 3 {
                    if !{ pyre_object::is_int(args[2]) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "recvmsg: ancbufsize must be an integer",
                        ));
                    }
                    let raw = { pyre_object::w_int_get_value(args[2]) };
                    if raw < 0 {
                        return Err(pyre_interpreter::PyError::value_error(
                            "invalid ancillary data buffer length",
                        ));
                    }
                    raw as usize
                } else {
                    0
                };
                let flags = if args.len() >= 4 {
                    if !{ pyre_object::is_int(args[3]) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "recvmsg: flags must be an integer",
                        ));
                    }
                    pyre_object::w_int_get_value(args[3]) as libc::c_int
                } else {
                    0
                };
                let mut obj = args[0];
                let fd = pyre_object::with_roots!(obj => socket_fd(obj))?;

                let mut data = pyre_object::with_roots!(obj => pyre_interpreter::builtins::try_vec_zeroed(bufsize))?;
                let mut control = pyre_object::with_roots!(obj => pyre_interpreter::builtins::try_vec_zeroed(ancbufsize))?;
                let mut storage: rffi::sockaddr_storage = { std::mem::zeroed() };
                pyre_object::with_roots!(obj => socket_wait_readable(obj, fd))?;
                let (got, msg_flags, msg_namelen, controllen) = loop {
                    let mut iov = libc::iovec {
                        iov_base: data.as_mut_ptr() as *mut libc::c_void,
                        iov_len: bufsize,
                    };
                    let mut msg: libc::msghdr = { std::mem::zeroed() };
                    msg.msg_name = &mut storage as *mut _ as *mut libc::c_void;
                    msg.msg_namelen =
                        core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
                    msg.msg_iov = &mut iov;
                    msg.msg_iovlen = 1;
                    if ancbufsize > 0 {
                        msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
                        msg.msg_controllen = ancbufsize as _;
                    }
                    let (r, errno) =
                        pyre_interpreter::module::thread::call_external_function(|| {
                            libc::recvmsg(fd, &mut msg, flags)
                        });
                    if r >= 0 {
                        break (r, msg.msg_flags, msg.msg_namelen, msg.msg_controllen);
                    }
                    if !rffi::error_is_interrupted(errno) {
                        return Err(socket_io_err_for_operation(
                            obj,
                            std::io::Error::from_raw_os_error(errno),
                        ));
                    }
                    // EINTR: deliver a pending signal, then retry
                    // (`converted_error` eintr_retry).
                    pyre_object::with_roots!(obj => pyre_interpreter::module::signal::interp_signal::checksignals_now())?;
                };
                data.truncate(got as usize);

                // Walk ancillary data.  Re-run msghdr with the final
                // controllen so CMSG_* macros see the trimmed buffer.  Every
                // field, cmsg tuple, the anc list, and the result tuple is a
                // nursery object. Build the anc list first so its bracket
                // closes before the result fields are pinned.
                let _roots = pyre_object::gc_roots::push_roots();
                let anc = {
                    let mut result_w = pyre_object::gc_roots::RootedItems::new();
                    if ancbufsize > 0 && controllen > 0 {
                        let mut dummy_iov = libc::iovec {
                            iov_base: std::ptr::null_mut(),
                            iov_len: 0,
                        };
                        let mut msg: libc::msghdr = { std::mem::zeroed() };
                        msg.msg_iov = &mut dummy_iov;
                        msg.msg_iovlen = 1;
                        msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
                        msg.msg_controllen = controllen;
                        {
                            let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
                            while !cmsg.is_null() {
                                let header = &*cmsg;
                                let hdr_size = libc::CMSG_LEN(0) as usize;
                                let total = header.cmsg_len as usize;
                                if total < hdr_size {
                                    break;
                                }
                                let payload_ptr = libc::CMSG_DATA(cmsg);
                                let control_start = control.as_ptr() as usize;
                                let data_start = payload_ptr as usize;
                                let control_end = control_start.saturating_add(controllen as usize);
                                if data_start < control_start || data_start > control_end {
                                    break;
                                }
                                // A truncated cmsg may retain its original cmsg_len.
                                // `recvmsg.py` returns only bytes the kernel actually
                                // left inside the final msg_controllen boundary.
                                let available = control_end - data_start;
                                let payload_len = (total - hdr_size).min(available);
                                let payload =
                                    std::slice::from_raw_parts(payload_ptr, payload_len).to_vec();
                                let pair = {
                                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                                    fields.push(pyre_object::w_int_new(header.cmsg_level as i64));
                                    fields.push(pyre_object::w_int_new(header.cmsg_type as i64));
                                    fields.push(pyre_object::bytesobject::w_bytes_from_bytes(
                                        &payload,
                                    ));
                                    pyre_object::w_tuple_new(fields.take())
                                };
                                result_w.push(pair);
                                cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
                            }
                        }
                    }
                    pyre_object::w_list_new(result_w.take())
                };
                let anc_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(anc);
                let data_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(
                    pyre_object::bytesobject::w_bytes_from_bytes(&data),
                );
                let flags_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(msg_flags as i64));
                let addr_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(unpack_inet_addr(&storage, msg_namelen));
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::gc_roots::shadow_stack_get(data_slot),
                    pyre_object::gc_roots::shadow_stack_get(anc_slot),
                    pyre_object::gc_roots::shadow_stack_get(flags_slot),
                    pyre_object::gc_roots::shadow_stack_get(addr_slot),
                ]))
            })
        )
    };

    // recvmsg_into(buffers, [ancbufsize, [flags]]) ->
    //   (nbytes, ancdata, msg_flags, address)
    // `interp_socket.py recvmsg_into_w` — scatter-receive into
    // a list/tuple of writable buffers; each `writebuf_w` slice
    // contributes one iovec entry.
    // The scatter/gather calls and their ancillary data are POSIX-only;
    // `socket.py` test for each before reaching for it.
    #[cfg(unix)]
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "recvmsg_into",
            pyre_interpreter::make_builtin_function("recvmsg_into", |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "recvmsg_into() missing buffers",
                    ));
                }
                let nargs = args.len();
                // One scope for the whole set: `SocketWritableBuffer` records slot
                // indices into it, and the vector's elements are dropped front to
                // back. The argument slice is pinned with that scope: unpacking
                // the buffers collects before the socket and the later integers
                // are read.
                let _roots = pyre_object::gc_roots::push_roots();
                let args_base = _roots.pin_roots(args);
                // `W_Socket.recvmsg_into_w` snapshots an arbitrary iterable with
                // `space.unpackiterable` before acquiring any writable views.  In
                // particular a generator is part of the public accepted surface.
                let buffer_items =
                    pyre_interpreter::baseobjspace::unpackiterable(_roots.get(args_base + 1), -1)?;
                let items_base = pyre_object::gc_roots::pin_roots(&buffer_items);
                let mut buffers: Vec<SocketWritableBuffer> = Vec::with_capacity(buffer_items.len());
                for i in 0..buffer_items.len() {
                    let item = pyre_object::gc_roots::shadow_stack_get(items_base + i);
                    buffers.push({ SocketWritableBuffer::acquire(item) }?);
                }
                let ancbufsize = if nargs >= 3 {
                    let anc_obj = _roots.get(args_base + 2);
                    if !{ pyre_object::is_int(anc_obj) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "recvmsg_into: ancbufsize must be an integer",
                        ));
                    }
                    let raw = { pyre_object::w_int_get_value(anc_obj) };
                    if raw < 0 {
                        return Err(pyre_interpreter::PyError::value_error(
                            "invalid ancillary data buffer length",
                        ));
                    }
                    raw as usize
                } else {
                    0
                };
                let flags = if nargs >= 4 {
                    let flags_obj = _roots.get(args_base + 3);
                    if !{ pyre_object::is_int(flags_obj) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "recvmsg_into: flags must be an integer",
                        ));
                    }
                    pyre_object::w_int_get_value(flags_obj) as libc::c_int
                } else {
                    0
                };
                let fd = socket_fd(_roots.get(args_base))?;

                let mut iovs: Vec<libc::iovec> = buffers
                    .iter_mut()
                    .map(|buffer| {
                        let slice = { buffer.as_mut_slice() };
                        libc::iovec {
                            iov_base: slice.as_mut_ptr() as *mut libc::c_void,
                            iov_len: slice.len(),
                        }
                    })
                    .collect();
                let mut control = pyre_interpreter::builtins::try_vec_zeroed(ancbufsize)?;
                let mut storage: rffi::sockaddr_storage = { std::mem::zeroed() };
                socket_wait_readable(_roots.get(args_base), fd)?;
                let (got, msg_flags, msg_namelen, controllen) = loop {
                    let mut msg: libc::msghdr = { std::mem::zeroed() };
                    msg.msg_name = &mut storage as *mut _ as *mut libc::c_void;
                    msg.msg_namelen =
                        core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
                    msg.msg_iov = iovs.as_mut_ptr();
                    msg.msg_iovlen = iovs.len() as _;
                    if ancbufsize > 0 {
                        msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
                        msg.msg_controllen = ancbufsize as _;
                    }
                    let (r, errno) =
                        pyre_interpreter::module::thread::call_external_function(|| {
                            libc::recvmsg(fd, &mut msg, flags)
                        });
                    if r >= 0 {
                        break (r, msg.msg_flags, msg.msg_namelen, msg.msg_controllen);
                    }
                    if !rffi::error_is_interrupted(errno) {
                        return Err(socket_io_err_for_operation(
                            _roots.get(args_base),
                            std::io::Error::from_raw_os_error(errno),
                        ));
                    }
                    // EINTR: deliver a pending signal, then retry
                    // (`converted_error` eintr_retry).
                    pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
                };

                // Every field, cmsg tuple, the anc list, and the result tuple
                // sit on the same bracket already opened for the writable
                // buffers. Build the anc list first so its bracket closes
                // before the result fields are pinned.
                let anc = {
                    let mut result_w = pyre_object::gc_roots::RootedItems::new();
                    if ancbufsize > 0 && controllen > 0 {
                        let mut dummy_iov = libc::iovec {
                            iov_base: std::ptr::null_mut(),
                            iov_len: 0,
                        };
                        let mut msg: libc::msghdr = { std::mem::zeroed() };
                        msg.msg_iov = &mut dummy_iov;
                        msg.msg_iovlen = 1;
                        msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
                        msg.msg_controllen = controllen;
                        {
                            let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
                            while !cmsg.is_null() {
                                let header = &*cmsg;
                                let hdr_size = libc::CMSG_LEN(0) as usize;
                                let total = header.cmsg_len as usize;
                                if total < hdr_size {
                                    break;
                                }
                                let payload_ptr = libc::CMSG_DATA(cmsg);
                                let control_start = control.as_ptr() as usize;
                                let data_start = payload_ptr as usize;
                                let control_end = control_start.saturating_add(controllen as usize);
                                if data_start < control_start || data_start > control_end {
                                    break;
                                }
                                let available = control_end - data_start;
                                let payload_len = (total - hdr_size).min(available);
                                let payload =
                                    std::slice::from_raw_parts(payload_ptr, payload_len).to_vec();
                                let pair = {
                                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                                    fields.push(pyre_object::w_int_new(header.cmsg_level as i64));
                                    fields.push(pyre_object::w_int_new(header.cmsg_type as i64));
                                    fields.push(pyre_object::bytesobject::w_bytes_from_bytes(
                                        &payload,
                                    ));
                                    pyre_object::w_tuple_new(fields.take())
                                };
                                result_w.push(pair);
                                cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
                            }
                        }
                    }
                    pyre_object::w_list_new(result_w.take())
                };
                let anc_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(anc);
                let got_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(got as i64));
                let flags_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(msg_flags as i64));
                let addr_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(unpack_inet_addr(&storage, msg_namelen));
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::gc_roots::shadow_stack_get(got_slot),
                    pyre_object::gc_roots::shadow_stack_get(anc_slot),
                    pyre_object::gc_roots::shadow_stack_get(flags_slot),
                    pyre_object::gc_roots::shadow_stack_get(addr_slot),
                ]))
            })
        )
    };

    // sendmsg(data_iter[, ancillary[, flags[, address]]]) → bytes_sent
    // `interp_socket.py:711-773` — gather-write of multiple bytes-like
    // buffers plus optional ancillary control messages.  Each cmsg is
    // a (cmsg_level, cmsg_type, cmsg_data) 3-tuple; we lay them out
    // into a single control buffer via CMSG_SPACE / CMSG_NXTHDR.
    // The scatter/gather calls and their ancillary data are POSIX-only;
    // `socket.py` test for each before reaching for it.
    #[cfg(unix)]
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "sendmsg",
            pyre_interpreter::make_builtin_function("sendmsg", |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "sendmsg() missing data",
                    ));
                }
                let nargs = args.len();
                // The argument slice and the socket stay live across unpacking,
                // buffer export, and address packing. Pin the slice once; the
                // socket is reloaded from its own slot after each collecting call.
                let args_roots = pyre_object::gc_roots::push_roots();
                let args_base = args_roots.pin_roots(args);
                let mut obj = args_roots.get(args_base);
                let fd = pyre_object::with_roots!(obj => socket_fd(obj))?;

                // PyPy `interp_socket.py`: `space.unpackiterable(w_data)`,
                // then acquire/release one `BUF_SIMPLE` view per item and retain
                // the copied strings.  In particular, asyncio passes an
                // `itertools.islice` of memoryviews here, not a list or tuple.
                let _roots = pyre_object::gc_roots::push_roots();
                let obj_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(obj);
                let data_items = pyre_interpreter::baseobjspace::unpackiterable(
                    args_roots.get(args_base + 1),
                    -1,
                )?;
                let items_base = pyre_object::gc_roots::pin_roots(&data_items);
                let mut data_buffers: Vec<Vec<u8>> = Vec::with_capacity(data_items.len());
                // `simple_buffer_bytes` looks up `__buffer__` and builds a
                // memoryview, so every iteration can collect; read each item back
                // from its slot instead of consuming the unrooted vector.
                for i in 0..data_items.len() {
                    let item = pyre_object::gc_roots::shadow_stack_get(items_base + i);
                    let Some(buffer) = pyre_interpreter::baseobjspace::simple_buffer_bytes(item)?
                    else {
                        return Err(pyre_interpreter::PyError::type_error(
                            "sendmsg: data items must be bytes-like",
                        ));
                    };
                    data_buffers.push(buffer.as_bytes().to_vec());
                    buffer.release();
                }
                let mut iovs: Vec<libc::iovec> = data_buffers
                    .iter()
                    .map(|s| libc::iovec {
                        iov_base: s.as_ptr() as *mut libc::c_void,
                        iov_len: s.len(),
                    })
                    .collect();

                // Build ancillary control buffer from args[2] (optional).
                let mut cmsgs: Vec<(libc::c_int, libc::c_int, Vec<u8>)> = Vec::new();
                if nargs >= 3 && !pyre_object::is_none(args_roots.get(args_base + 2)) {
                    // PyPy `W_Socket.sendmsg_w` snapshots both iterable levels
                    // with `space.unpackiterable` before converting their fields.
                    // That order is semantic: `__index__` may mutate the original
                    // ancillary list, but it cannot rewrite the captured message.
                    let ancillary_items = pyre_interpreter::baseobjspace::unpackiterable(
                        args_roots.get(args_base + 2),
                        -1,
                    )?;
                    let ancillary_base = pyre_object::gc_roots::pin_roots(&ancillary_items);
                    for i in 0..ancillary_items.len() {
                        let item = pyre_object::gc_roots::shadow_stack_get(ancillary_base + i);
                        let fields = pyre_interpreter::baseobjspace::unpackiterable(item, -1)?;
                        if fields.len() != 3 {
                            return Err(pyre_interpreter::PyError::type_error(
                                "sendmsg: ancillary items must be 3-tuples",
                            ));
                        }
                        let fields_base = pyre_object::gc_roots::pin_roots(&fields);
                        let level_o = pyre_object::gc_roots::shadow_stack_get(fields_base);
                        let level =
                            libc::c_int::try_from(pyre_interpreter::baseobjspace::int_w(level_o)?)
                                .map_err(|_| {
                                    pyre_interpreter::PyError::overflow_error(
                                        "Python int too large to convert to C int",
                                    )
                                })?;
                        let type_o = pyre_object::gc_roots::shadow_stack_get(fields_base + 1);
                        let ty =
                            libc::c_int::try_from(pyre_interpreter::baseobjspace::int_w(type_o)?)
                                .map_err(|_| {
                                pyre_interpreter::PyError::overflow_error(
                                    "Python int too large to convert to C int",
                                )
                            })?;
                        let data_o = pyre_object::gc_roots::shadow_stack_get(fields_base + 2);
                        // interp_socket.py W_Socket.sendmsg_w acquires a BUF_SIMPLE view,
                        // just like the ordinary data items above.  This includes
                        // array.array payloads used by multiprocessing's
                        // SCM_RIGHTS fd transfer, not only bytes/bytearray.
                        let Some(buffer) =
                            pyre_interpreter::baseobjspace::simple_buffer_bytes(data_o)?
                        else {
                            return Err(pyre_interpreter::PyError::type_error(
                                "sendmsg: ancillary data must be bytes-like",
                            ));
                        };
                        let data = buffer.as_bytes().to_vec();
                        buffer.release();
                        cmsgs.push((level, ty, data));
                    }
                }
                let flags = if nargs >= 4 {
                    let flags_obj = args_roots.get(args_base + 3);
                    if !{ pyre_object::is_int(flags_obj) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "sendmsg: flags must be an integer",
                        ));
                    }
                    pyre_object::w_int_get_value(flags_obj) as libc::c_int
                } else {
                    0
                };
                // `pack_inet_addr` runs Python for a host that is not a plain
                // ASCII `str`, so the socket and the address are read back from
                // their slots rather than from the locals copied at entry.
                let (addr_storage, addr_len) = if nargs >= 5
                    && !pyre_object::is_none(args_roots.get(args_base + 4))
                {
                    let family = socket_get_attr_i64(
                        pyre_object::gc_roots::shadow_stack_get(obj_slot),
                        "_family",
                    ) as libc::c_int;
                    let proto = socket_get_attr_i64(
                        pyre_object::gc_roots::shadow_stack_get(obj_slot),
                        "_proto",
                    ) as libc::c_int;
                    let (s, l) =
                        pack_inet_addr("sendmsg", family, proto, args_roots.get(args_base + 4))?;
                    (Some(s), l)
                } else {
                    (None, 0)
                };

                // Lay out cmsgs into a single control buffer.
                #[cfg(all(feature = "host_env", not(target_os = "redox")))]
                let mut control = {
                    let refs: Vec<_> = cmsgs
                        .iter()
                        .map(|(level, ty, data)| (*level, *ty, data.as_slice()))
                        .collect();
                    rustpython_host_env::socket::pack_ancillary_messages(&refs).map_err(
                        |error| {
                            use rustpython_host_env::socket::AncillaryPackError;
                            match error {
                                AncillaryPackError::ItemTooLarge => {
                                    pyre_interpreter::PyError::os_error(
                                        "ancillary data item too large",
                                    )
                                }
                                AncillaryPackError::TooMuchData => {
                                    pyre_interpreter::PyError::os_error("too much ancillary data")
                                }
                                AncillaryPackError::UnexpectedNullHeader => {
                                    pyre_interpreter::PyError::runtime_error(
                                        "unexpected NULL result from CMSG_FIRSTHDR/CMSG_NXTHDR",
                                    )
                                }
                            }
                        },
                    )?
                };
                #[cfg(any(not(feature = "host_env"), target_os = "redox"))]
                let mut control = {
                    let total_control: usize = cmsgs
                        .iter()
                        .map(|(_, _, d)| unsafe {
                            libc::CMSG_SPACE(d.len() as libc::c_uint) as usize
                        })
                        .sum();
                    let mut control = vec![0u8; total_control];
                    if total_control > 0 {
                        let mut packing_msg: libc::msghdr = unsafe { std::mem::zeroed() };
                        packing_msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
                        packing_msg.msg_controllen = total_control as _;
                        unsafe {
                            let mut cur = libc::CMSG_FIRSTHDR(&packing_msg);
                            for (level, ty, data) in &cmsgs {
                                if cur.is_null() {
                                    break;
                                }
                                let cmsg_len = libc::CMSG_LEN(data.len() as libc::c_uint);
                                (*cur).cmsg_level = *level;
                                (*cur).cmsg_type = *ty;
                                (*cur).cmsg_len = cmsg_len as _;
                                std::ptr::copy_nonoverlapping(
                                    data.as_ptr(),
                                    libc::CMSG_DATA(cur),
                                    data.len(),
                                );
                                cur = libc::CMSG_NXTHDR(&packing_msg, cur);
                            }
                        }
                    }
                    control
                };
                let total_control = control.len();
                let mut msg: libc::msghdr = { std::mem::zeroed() };
                msg.msg_iov = iovs.as_mut_ptr();
                msg.msg_iovlen = iovs.len() as _;
                if let Some(ref s) = addr_storage {
                    msg.msg_name = s as *const _ as *mut libc::c_void;
                    msg.msg_namelen = addr_len;
                }
                if total_control > 0 {
                    msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
                    msg.msg_controllen = total_control as _;
                }

                socket_wait_writable(pyre_object::gc_roots::shadow_stack_get(obj_slot), fd)?;
                let sent = loop {
                    let (r, errno) =
                        pyre_interpreter::module::thread::call_external_function(|| {
                            libc::sendmsg(fd, &msg, flags)
                        });
                    if r >= 0 {
                        break r;
                    }
                    if !rffi::error_is_interrupted(errno) {
                        return Err(socket_io_err_for_operation(
                            pyre_object::gc_roots::shadow_stack_get(obj_slot),
                            std::io::Error::from_raw_os_error(errno),
                        ));
                    }
                    // EINTR: deliver a pending signal, then retry
                    // (`converted_error` eintr_retry).
                    pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
                };
                Ok(pyre_object::w_int_new(sent as i64))
            })
        )
    };

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "shutdown",
        pyre_interpreter::make_builtin_function_with_arity(
            "shutdown",
            __majit_wrap_socket_shutdown,
            2,
        )
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "getsockname",
        pyre_interpreter::make_builtin_function_with_arity(
            "getsockname",
            __majit_wrap_socket_getsockname,
            1,
        )
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "getpeername",
        pyre_interpreter::make_builtin_function_with_arity(
            "getpeername",
            __majit_wrap_socket_getpeername,
            1,
        )
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "setsockopt",
        pyre_interpreter::make_builtin_function("setsockopt", __majit_wrap_socket_setsockopt,)
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "getsockopt",
        pyre_interpreter::make_builtin_function("getsockopt", __majit_wrap_socket_getsockopt,)
    );

    // `sock_ioctl` is published where `SIO_RCVALL` is defined, which is
    // Windows.  It is `WSAIoctl` under a socket method, and the three commands
    // it names are the ones whose input is a value rather than a buffer.
    #[cfg(windows)]
    {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "ioctl",
            pyre_interpreter::make_builtin_function("ioctl", |args| {
                use windows_sys::Win32::Networking::WinSock as ws;

                if args.len() != 3 {
                    return Err(pyre_interpreter::PyError::type_error(format!(
                        "ioctl() takes exactly 2 arguments ({} given)",
                        args.len().saturating_sub(1)
                    )));
                }
                let fd = socket_fd(args[0])?;
                let cmd = ioctl_command_w(args[1], "argument 1")?;
                let returned = match cmd {
                    ws::SIO_RCVALL | ws::SIO_LOOPBACK_FAST_PATH => {
                        let value = masked_ulong_w(args[2])?;
                        #[cfg(feature = "host_env")]
                        {
                            rustpython_host_env::socket::ioctl_u32(fd as _, cmd, value).ok()
                        }
                        #[cfg(not(feature = "host_env"))]
                        {
                            let _ = value;
                            None
                        }
                    }
                    ws::SIO_KEEPALIVE_VALS => {
                        let [onoff, keepalivetime, keepaliveinterval] = ioctl_keepalive_w(args[2])?;
                        #[cfg(feature = "host_env")]
                        {
                            rustpython_host_env::socket::ioctl_keepalive(
                                fd as _,
                                rustpython_host_env::socket::TcpKeepalive {
                                    onoff,
                                    keepalivetime,
                                    keepaliveinterval,
                                },
                            )
                            .ok()
                        }
                        #[cfg(not(feature = "host_env"))]
                        {
                            let _ = (onoff, keepalivetime, keepaliveinterval);
                            None
                        }
                    }
                    _ => {
                        return Err(pyre_interpreter::PyError::value_error(format!(
                            "invalid ioctl command {cmd}"
                        )));
                    }
                };
                match returned {
                    Some(returned) => Ok(pyre_object::w_int_new(i64::from(returned))),
                    None => Err(socket_last_error()),
                }
            })
        )
    };

    // `sock_share` sits beside `ioctl`, published on the same Windows-only
    // footing: `WSADuplicateSocketW` writes a `WSAPROTOCOL_INFOW` describing
    // the socket for the process named, and the bytes of that structure are
    // the whole answer.  `socket.py` grows `fromshare` as soon as the method
    // exists, and hands the blob straight back to the constructor.
    #[cfg(all(windows, feature = "host_env"))]
    {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "share",
            pyre_interpreter::make_builtin_function("share", |args| {
                // `METH_O`, so a wrong count is reported by the call machinery
                // rather than by a converter naming the method.
                if args.len() != 2 {
                    return Err(pyre_interpreter::PyError::type_error(format!(
                        "function takes exactly 1 argument ({} given)",
                        args.len().saturating_sub(1)
                    )));
                }
                let fd = socket_fd(args[0])?;
                let process_id = masked_ulong_w(args[1])?;
                let info = {
                    let _blocked = pyre_interpreter::module::thread::before_external_block();
                    rustpython_host_env::socket::share_socket(fd, process_id)
                }
                .map_err(socket_io_err)?;
                Ok(pyre_object::w_bytes_from_bytes(&info))
            })
        )
    };

    // `interp_socket.py setblocking_w` per PyPy docstring: True
    // is equivalent to `settimeout(None)`, False to `settimeout(0.0)`.
    // Routing through `socket_apply_timeout` keeps the SO_*TIMEO state
    // consistent with the timeout attribute and prevents a stale
    // SO_RCVTIMEO from surviving a `setblocking(True)` call.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "setblocking",
        pyre_interpreter::make_builtin_function_with_arity(
            "setblocking",
            |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "setblocking() missing argument",
                    ));
                }
                let mut obj = args[0];
                let w_flag = args[1];
                let blocking = pyre_object::with_roots!(obj => pyre_interpreter::baseobjspace::is_true(w_flag))?;
                let fd = pyre_object::with_roots!(obj => socket_fd(obj))?;
                let timeout = if blocking { -1.0 } else { 0.0 };
                pyre_object::with_roots!(obj => socket_apply_timeout(fd, timeout))?;
                let w_timeout = pyre_object::with_roots!(obj => if blocking {
                    pyre_object::w_none()
                } else {
                    pyre_object::floatobject::w_float_new(0.0)
                });
                socket_set_attr(obj, "_timeout", w_timeout);
                Ok(pyre_object::w_none())
            },
            2,
        )
    );

    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "getblocking",
            pyre_interpreter::make_builtin_function_with_arity(
                "getblocking",
                |args| {
                    let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                    pyre_object::with_roots!(obj => socket_fd(obj))?;
                    // `sock_getblocking` answers from the stored timeout rather
                    // than from the descriptor: `settimeout` is its only writer,
                    // and WinSock's `FIONBIO` cannot be read back.
                    let d = pyre_interpreter::baseobjspace::getdict_native(obj);
                    let blocking =
                        d.is_null() || { pyre_object::w_dict_getitem_str(d, "_timeout") }
                            .filter(|t| pyre_object::is_float(*t))
                            .is_none_or(|t| pyre_object::floatobject::w_float_get_value(t) != 0.0);
                    Ok(pyre_object::w_bool_from(blocking))
                },
                1,
            )
        )
    };

    // `interp_socket.py settimeout_w` then `rsocket.py:RSocket.
    // settimeout`, which ends in `_setblocking(self.timeout < 0.0)`: None →
    // blocking; 0.0 and any positive duration alike → O_NONBLOCK on, the
    // duration being `wait_for_data`'s to enforce rather than the
    // descriptor's; <0 → ValueError "Timeout value out of range".
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "settimeout",
            pyre_interpreter::make_builtin_function_with_arity(
                "settimeout",
                |args| {
                    if args.len() < 2 {
                        return Err(pyre_interpreter::PyError::type_error(
                            "settimeout() missing argument",
                        ));
                    }
                    let mut obj = args[0];
                    let mut w_t = args[1];
                    let timeout: f64 = if pyre_object::is_none(w_t) {
                        -1.0
                    } else {
                        let v = {
                            if pyre_object::is_float(w_t) {
                                pyre_object::floatobject::w_float_get_value(w_t)
                            } else if pyre_object::is_int(w_t) {
                                pyre_object::w_int_get_value(w_t) as f64
                            } else {
                                return Err(pyre_interpreter::PyError::type_error(
                                    "settimeout: timeout must be a float or None",
                                ));
                            }
                        };
                        if v < 0.0 {
                            return Err(pyre_interpreter::PyError::value_error(
                                "Timeout value out of range",
                            ));
                        }
                        v
                    };
                    let fd = pyre_object::with_roots!(obj, w_t => socket_fd(obj))?;
                    socket_apply_timeout(fd, timeout)?;
                    // `gettimeout()` reports a float, and the timeout readers below
                    // recognise only that type; storing the caller's `int` verbatim
                    // would make `settimeout(1)` silently untimed.
                    let stored = if pyre_object::is_none(w_t) {
                        w_t
                    } else {
                        pyre_object::floatobject::w_float_new(timeout)
                    };
                    socket_set_attr(obj, "_timeout", stored);
                    Ok(pyre_object::w_none())
                },
                2,
            )
        )
    };

    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "gettimeout",
            pyre_interpreter::make_builtin_function_with_arity(
                "gettimeout",
                |args| {
                    let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                    let d = pyre_interpreter::baseobjspace::getdict_native(obj);
                    if d.is_null() {
                        return Ok(pyre_object::w_none());
                    }
                    Ok({ pyre_object::w_dict_getitem_str(d, "_timeout") }
                        .unwrap_or(pyre_object::w_none()))
                },
                1,
            )
        )
    };

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
            if let Some(&(mut obj)) = args.first() {
                let fd = rffi::socket_from_i64(
                    pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_fd")),
                );
                if !rffi::is_invalid(fd) {
                    #[cfg(unix)]
                    {
                        let _ = majit_rlib::rsocket::close(fd);
                    }
                    #[cfg(windows)]
                    {
                        let _ = unsafe { rffi::close(fd) };
                    }
                    socket_set_attr(obj, "_fd", pyre_object::w_int_new(-1));
                }
            }
            Ok(pyre_object::w_bool_from(false))
        })
    );

    // __repr__ — `interp_socket.py descr_repr`.  Format
    // matches CPython: `<socket object, fd=N, family=F, type=T, proto=P>`.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__repr__",
        pyre_interpreter::make_builtin_function_with_arity(
            "__repr__",
            |args| {
                let mut obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_fd"));
                let family = pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_family"));
                let ty = pyre_object::with_roots!(obj => socket_get_attr_i64(obj, "_type"));
                let proto = socket_get_attr_i64(obj, "_proto");
                Ok(pyre_object::w_str_new_managed(&format!(
                    "<socket object, fd={fd}, family={family}, type={ty}, proto={proto}>"
                )))
            },
            1,
        )
    );

    // set_inheritable / get_inheritable — `interp_socket.py` wraps whether
    // an exec'd child keeps the descriptor.
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "set_inheritable",
            pyre_interpreter::make_builtin_function_with_arity(
                "set_inheritable",
                |args| {
                    if args.len() < 2 {
                        return Err(pyre_interpreter::PyError::type_error(
                            "set_inheritable() missing argument",
                        ));
                    }
                    let w_self = args[0];
                    let mut w_inheritable = args[1];
                    let fd = pyre_object::with_roots!(w_inheritable => socket_fd(w_self))?;
                    let want_inheritable = {
                        if pyre_object::is_bool(w_inheritable) {
                            pyre_object::boolobject::w_bool_get_value(w_inheritable)
                        } else if pyre_object::is_int(w_inheritable) {
                            pyre_object::w_int_get_value(w_inheritable) != 0
                        } else {
                            return Err(pyre_interpreter::PyError::type_error(
                                "set_inheritable: value must be bool",
                            ));
                        }
                    };
                    rffi::set_inheritable(fd, want_inheritable).map_err(socket_io_err)?;
                    Ok(pyre_object::w_none())
                },
                2,
            )
        )
    };
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "get_inheritable",
        pyre_interpreter::make_builtin_function_with_arity(
            "get_inheritable",
            |args| {
                let fd = socket_fd(args.first().copied().unwrap_or(pyre_object::PY_NULL))?;
                let inheritable = rffi::get_inheritable(fd).map_err(socket_io_err)?;
                Ok(pyre_object::w_bool_from(inheritable))
            },
            1,
        )
    );
}
