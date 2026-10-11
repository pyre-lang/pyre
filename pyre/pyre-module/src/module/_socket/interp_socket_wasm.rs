//! `_socket` on a target with no socket layer — PyPy: pypy/module/_socket/
//!
//! The module is published here the way a build whose C library carries the
//! headers but not the calls publishes it: the `socket` type, the numbers from
//! `<sys/socket.h>` and `<netdb.h>`, and none of the entry points that would
//! need a descriptor.  `socket.py` subclasses `_socket.socket` in its module
//! body and reaches everything else through `hasattr`, so that is the shape
//! that lets `import socket` — and `asyncio`, `socketserver`, `ftplib` behind
//! it — resolve, while an actual connection attempt reports that this platform
//! has none rather than pretending to open one.
//!
//! The numbers are musl's, which is wasi-libc's, matching the `sys.platform`
//! the guest reports.  `has_ipv6` stays false: carrying the `AF_INET6` number
//! is what a header does, and answering that the runtime supports the family
//! is a separate claim.

use pyre_object::PyObjectRef;

/// The `socket` type: real enough to subclass, empty of anything that would
/// need a descriptor behind it.
fn socket_type() -> PyObjectRef {
    static SOCKET_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
        pyre_object::gc_roots::RootedOnceRef::new();
    SOCKET_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type("socket", init_socket_type);
        pyre_interpreter::typedef::mark_cpython_heap_type(tp, false);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp
    })
}

fn init_socket_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // Allocation is separate from initialisation here as it is everywhere
    // else: `socket.py`'s subclass calls `_socket.socket.__init__` itself, so
    // `__new__` must hand back an instance without having opened anything.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__new__",
        pyre_interpreter::typedef::make_new_descr(|args| {
            let cls = args
                .first()
                .copied()
                .filter(|w_cls| unsafe { pyre_object::is_type(*w_cls) })
                .unwrap_or_else(socket_type);
            Ok(pyre_object::w_instance_new(cls))
        })
    );
    // The one call this type would have to make, and the one it cannot: there
    // is no descriptor for a socket to be, so construction reports that rather
    // than handing back an object whose every method would have to.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__init__",
        pyre_interpreter::make_builtin_function("__init__", |_args| {
            Err(pyre_interpreter::PyError::os_error_syscall(
                pyre_interpreter::builtins::wasm_errno::ENOTSUP,
                pyre_object::w_none(),
            ))
        })
    );
}

/// The two families the address converters carry a parser for, as the constant
/// table below spells them.
pub(super) const AF_INET: i32 = 2;
pub(super) const AF_INET6: i32 = 10;

/// The host this guest runs as, which is the `nodename` `uname` reports: wasi's
/// `gethostname` is written over `uname` and hands back that field.
const NODE_NAME: &str = "(none)";

/// `inet_aton` — the lenient parser.
pub(super) fn inet_aton(text: &std::ffi::CStr) -> Option<[u8; 4]> {
    super::inet::aton(text.to_bytes())
}

/// `inet_ntoa` — the dotted-quad spelling of four address bytes.
pub(super) fn inet_ntoa(packed: [u8; 4]) -> Option<String> {
    Some(super::inet::ntoa(packed))
}

/// `inet_pton`'s two failures, which it reports through different channels: a
/// negative return leaves `EAFNOSUPPORT` behind, a zero says the string was
/// read and rejected.
pub(crate) enum PtonError {
    /// A family with no parser behind it.
    Family(i32),
    /// An address this family's parser refused.
    Address,
}

/// `inet_pton` — the strict parser.
pub(super) fn pton(family: i32, text: &std::ffi::CStr) -> Result<Vec<u8>, PtonError> {
    let bytes = text.to_bytes();
    let packed = match family {
        AF_INET => super::inet::pton_v4(bytes).map(|a| a.to_vec()),
        AF_INET6 => super::inet::pton_v6(bytes).map(|a| a.to_vec()),
        _ => {
            return Err(PtonError::Family(
                pyre_interpreter::builtins::wasm_errno::EAFNOSUPPORT,
            ));
        }
    };
    packed.ok_or(PtonError::Address)
}

/// `inet_ntop` — the canonical spelling of a packed address.
pub(super) fn ntop(family: i32, packed: &[u8]) -> Option<String> {
    match family {
        AF_INET => Some(super::inet::ntoa([
            packed[0], packed[1], packed[2], packed[3],
        ])),
        AF_INET6 => Some(super::inet::ntop_v6(packed)),
        _ => None,
    }
}

/// The `<sys/socket.h>` / `<netinet/in.h>` / `<netdb.h>` numbers, as musl
/// spells them.
///
/// `AF_UNIX` is left out because wasi-libc carries no `<sys/un.h>`, and the
/// name is read as a capability rather than as a number: `socketserver`
/// defines its four Unix server classes behind `hasattr(socket, "AF_UNIX")`
/// and `socket.socketpair` defaults to the family.
const CONSTANTS: &[(&str, i64)] = &[
    // ── Address families ──
    ("AF_UNSPEC", 0),
    ("AF_INET", 2),
    ("AF_INET6", 10),
    // ── Socket types ──
    ("SOCK_STREAM", 1),
    ("SOCK_DGRAM", 2),
    ("SOCK_RAW", 3),
    ("SOCK_RDM", 4),
    ("SOCK_SEQPACKET", 5),
    // ── Protocols ──
    ("IPPROTO_IP", 0),
    ("IPPROTO_ICMP", 1),
    ("IPPROTO_TCP", 6),
    ("IPPROTO_UDP", 17),
    ("IPPROTO_IPV6", 41),
    ("IPPROTO_RAW", 255),
    // ── Option levels and names ──
    ("SOL_SOCKET", 1),
    ("SO_DEBUG", 1),
    ("SO_REUSEADDR", 2),
    ("SO_TYPE", 3),
    ("SO_ERROR", 4),
    ("SO_DONTROUTE", 5),
    ("SO_BROADCAST", 6),
    ("SO_SNDBUF", 7),
    ("SO_RCVBUF", 8),
    ("SO_KEEPALIVE", 9),
    ("SO_OOBINLINE", 10),
    ("SO_LINGER", 13),
    ("SO_REUSEPORT", 15),
    ("SO_RCVLOWAT", 18),
    ("SO_SNDLOWAT", 19),
    ("SO_RCVTIMEO", 20),
    ("SO_SNDTIMEO", 21),
    ("SO_ACCEPTCONN", 30),
    ("TCP_NODELAY", 1),
    ("TCP_MAXSEG", 2),
    ("TCP_KEEPIDLE", 4),
    ("TCP_KEEPINTVL", 5),
    ("TCP_KEEPCNT", 6),
    ("IPV6_V6ONLY", 26),
    // ── shutdown(2) ──
    ("SHUT_RD", 0),
    ("SHUT_WR", 1),
    ("SHUT_RDWR", 2),
    // ── send / recv flags ──
    ("MSG_OOB", 0x0001),
    ("MSG_PEEK", 0x0002),
    ("MSG_DONTROUTE", 0x0004),
    ("MSG_CTRUNC", 0x0008),
    ("MSG_TRUNC", 0x0020),
    ("MSG_DONTWAIT", 0x0040),
    ("MSG_EOR", 0x0080),
    ("MSG_WAITALL", 0x0100),
    ("MSG_NOSIGNAL", 0x4000),
    // ── getaddrinfo / getnameinfo ──
    ("AI_PASSIVE", 0x0001),
    ("AI_CANONNAME", 0x0002),
    ("AI_NUMERICHOST", 0x0004),
    ("AI_V4MAPPED", 0x0008),
    ("AI_ALL", 0x0010),
    ("AI_ADDRCONFIG", 0x0020),
    ("AI_NUMERICSERV", 0x0400),
    ("NI_NUMERICHOST", 0x0001),
    ("NI_NUMERICSERV", 0x0002),
    ("NI_NOFQDN", 0x0004),
    ("NI_NAMEREQD", 0x0008),
    ("NI_DGRAM", 0x0010),
    ("NI_MAXHOST", 255),
    ("NI_MAXSERV", 32),
    ("EAI_BADFLAGS", -1),
    ("EAI_NONAME", -2),
    ("EAI_AGAIN", -3),
    ("EAI_FAIL", -4),
    ("EAI_NODATA", -5),
    ("EAI_FAMILY", -6),
    ("EAI_SOCKTYPE", -7),
    ("EAI_SERVICE", -8),
    ("EAI_ADDRFAMILY", -9),
    ("EAI_MEMORY", -10),
    ("EAI_SYSTEM", -11),
    ("EAI_OVERFLOW", -12),
    ("INADDR_ANY", 0x0000_0000),
    ("INADDR_BROADCAST", 0xffff_ffff_u32 as i64),
    ("INADDR_LOOPBACK", 0x7f00_0001),
    ("INADDR_NONE", 0xffff_ffff_u32 as i64),
    ("SOMAXCONN", 128),
];

/// The names `register_module` adds where there is no host socket layer to
/// build the rest out of.
pub(super) fn register_names(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    for (name, value) in CONSTANTS {
        pyre_interpreter::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(*value));
    }
    let socket_tp = socket_type();
    let socket_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(socket_tp);
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "socket",
        pyre_object::gc_roots::shadow_stack_get(socket_slot)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "SocketType",
        pyre_object::gc_roots::shadow_stack_get(socket_slot)
    );
    // `gethostname` needs no socket layer -- wasi answers it out of `uname` --
    // and `platform._node` calls it, so `platform.uname()` depends on it.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "gethostname",
        pyre_interpreter::make_builtin_function_with_arity(
            "gethostname",
            |_args| Ok(pyre_object::w_str_new_managed(NODE_NAME)),
            0,
        )
    );
}
