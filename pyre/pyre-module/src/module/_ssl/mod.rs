//! Rustls-backed `_ssl` module.
//!
//! The public policy and object wrappers remain the unmodified CPython
//! `ssl.py`.  This module supplies its low-level primitives while the actual
//! TLS engine lives in `pyre-native`, outside the translated interpreter.

use pyre_native::ssl::{
    CERT_NONE, CERT_OPTIONAL, CERT_REQUIRED, PROTOCOL_TLS, PROTOCOL_TLS_CLIENT,
    PROTOCOL_TLS_SERVER, PROTOCOL_TLSV1, PROTOCOL_TLSV1_1, PROTOCOL_TLSV1_2, PROTOCOL_TLSV1_3,
};
use pyre_object::*;

/// `ssl.SSLContext` is an app-level subclass of this native type. That
/// instance is `typedef.py` `_getusercls` (`MapdictStorageMixin`), allocated
/// by `objspace.py` `allocate_instance`.
// CPython 3.14 Modules/_ssl.c:_ssl_exec uses PyType_FromModuleAndSpec;
// the SSLContext spec is immutable.
#[pyre_interpreter::pyre_class("_ssl._SSLContext", cpython_heaptype, user_layout)]
#[derive(Default)]
pub struct W_SSLContext {
    pub backend: *mut pyre_native::ssl::Context,
    pub sni_callback: PyObjectRef,
    pub msg_callback: PyObjectRef,
    pub keylog_filename: PyObjectRef,
    pub host_flags: i32,
    pub post_handshake_auth: bool,
    pub num_tickets: i32,
}

/// `MemoryBIO` is not an acceptable base in this module. The payload is the
/// native BIO alone.
// `_ssl_exec` creates the immutable MemoryBIO module heap type.
#[pyre_interpreter::pyre_class("_ssl.MemoryBIO", cpython_heaptype)]
#[derive(Default)]
pub struct W_MemoryBIO {
    pub backend: *mut pyre_native::ssl::MemoryBio,
}

// `_ssl_exec` creates the immutable SSLSession module heap type.
#[pyre_interpreter::pyre_class("_ssl.SSLSession", cpython_heaptype)]
#[derive(Default)]
pub struct W_SSLSession {
    pub backend: *mut pyre_native::ssl::NativeSession,
}

/// A TLS record's header: the content type, the two protocol version bytes,
/// and the two that hold the body's length.
const TLS_RECORD_HEADER_LEN: usize = 5;

/// Where the connection stands inside the TLS record it is reading.
///
/// `ssl3_read_n` takes the 5-byte header and then exactly the body its length
/// names, so OpenSSL never lifts more off the transport than the record it is
/// parsing.  That is what leaves the plaintext following the peer's
/// close_notify in the kernel for the socket `unwrap()` hands back, and it
/// holds for every read rather than only the shutdown exchange: a `recv` that
/// took the close_notify and the bytes behind it would drop those bytes into
/// the TLS stream, where nothing can return them.
///
/// A transport that returns short -- a non-blocking socket, a timed one, an
/// interrupted read -- unwinds back to the caller, which asks again, so this
/// belongs to the connection and not to one call.
#[derive(Default)]
pub struct RecordCursor {
    header: [u8; TLS_RECORD_HEADER_LEN],
    header_read: usize,
    body_left: usize,
}

impl RecordCursor {
    /// The bytes still owed before the next record boundary.
    fn want(&self) -> usize {
        if self.body_left > 0 {
            self.body_left
        } else {
            TLS_RECORD_HEADER_LEN - self.header_read
        }
    }

    /// Account for `data`, which must be no longer than [`want`].
    ///
    /// [`want`]: RecordCursor::want
    fn consume(&mut self, data: &[u8]) {
        if self.body_left > 0 {
            self.body_left -= data.len();
            return;
        }
        self.header[self.header_read..self.header_read + data.len()].copy_from_slice(data);
        self.header_read += data.len();
        if self.header_read == TLS_RECORD_HEADER_LEN {
            // The record's length is the header's last two bytes, big-endian.
            self.body_left = u16::from_be_bytes([self.header[3], self.header[4]]) as usize;
            self.header_read = 0;
        }
    }
}

/// One TLS state machine plus its Python-owned transport endpoints.  Rustls
/// remains opaque in `pyre-native`; these references preserve the same
/// per-object ownership shape as PyPy's `W_SSLObject`.
// `_ssl_exec` creates the immutable SSLSocket module heap type.
#[pyre_interpreter::pyre_class("_ssl._SSLSocket", cpython_heaptype)]
#[derive(Default)]
pub struct W_SSLSocket {
    pub backend: *mut pyre_native::ssl::TlsConnection,
    pub context: PyObjectRef,
    pub socket: PyObjectRef,
    pub incoming: PyObjectRef,
    pub outgoing: PyObjectRef,
    pub owner: PyObjectRef,
    pub server_hostname: PyObjectRef,
    pub server_side: bool,
    pub shutdown_started: bool,
    pub record: RecordCursor,
    pub requested_session: *mut pyre_native::ssl::NativeSession,
}

// `_ssl_exec` creates the immutable Certificate module heap type.
#[pyre_interpreter::pyre_class("_ssl.Certificate", cpython_heaptype)]
#[derive(Default)]
pub struct W_Certificate {
    pub der: PyObjectRef,
    pub hash: i64,
}

/// The `library` and `reason` an OpenSSL-shaped `[library: reason] text`
/// message carries.
fn split_library_reason(message: &str) -> Option<(&str, &str)> {
    let close = message.find(']')?;
    if !message.starts_with('[') {
        return None;
    }
    message[1..close].split_once(": ")
}

fn set_library_reason(exception: PyObjectRef, message: &str) {
    if let Some((library, reason)) = split_library_reason(message) {
        let _roots = pyre_object::gc_roots::push_roots();
        let exc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(exception);
        let library_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(library));
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            pyre_object::gc_roots::shadow_stack_get(exc_slot),
            "library",
            pyre_object::gc_roots::shadow_stack_get(library_slot),
        );
        let reason_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(reason));
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            pyre_object::gc_roots::shadow_stack_get(exc_slot),
            "reason",
            pyre_object::gc_roots::shadow_stack_get(reason_slot),
        );
    }
}

fn ssl_error(message: impl Into<String>) -> pyre_interpreter::PyError {
    let message = message.into();
    let Some(cls) = pyre_interpreter::builtins::lookup_exc_class("ssl.SSLError") else {
        return pyre_interpreter::PyError::os_error(message);
    };
    // `OperationError(w_type, (errno, strerror))`; normalize instantiates
    // `SSLError(0, message)`. `library` / `reason` are stamped onto that
    // instance the way `set_error` writes extra fields before the carrier.
    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let errno_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_int_new(0));
    let msg_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(&message));
    let w_value = w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(errno_slot),
        pyre_object::gc_roots::shadow_stack_get(msg_slot),
    ]);
    let mut err = pyre_interpreter::PyError::from_type_and_value(
        pyre_object::gc_roots::shadow_stack_get(cls_slot),
        w_value,
    );
    let err_slot = err.pin(&_roots);
    if let Ok(exc) = err.normalize_exception(pyre_object::PY_NULL) {
        err.reload(&_roots, err_slot);
        set_library_reason(exc, &message);
    }
    err.reload(&_roots, err_slot);
    err
}

/// CPython's `SSLError` is an `OSError` subclass, but its string form is the
/// TLS message rather than the generic two-item exception tuple.  Keep this
/// policy in the `_ssl` boundary instead of teaching the process-wide
/// exception formatter about an extension-module class.
fn ssl_error_str(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let w_args = unsafe { pyre_object::interp_exceptions::w_exception_get_args(args[0]) };
    let len = unsafe { pyre_object::w_tuple_len(w_args) };
    if len >= 2 {
        let message = unsafe {
            pyre_object::w_tuple_getitem(w_args, 1)
                .expect("SSLError argument tuple has a second element")
        };
        pyre_interpreter::builtins::builtin_str(&[message])
    } else {
        Ok(pyre_object::w_str_from_wtf8_managed(unsafe {
            pyre_interpreter::display::base_exception_str_wtf8(args[0])?
        }))
    }
}

fn native_result<T>(result: Result<T, (i32, String)>) -> Result<T, pyre_interpreter::PyError> {
    result.map_err(|(errno, message)| {
        if errno != 0 {
            pyre_interpreter::PyError::os_error_with_errno(errno, message)
        } else {
            ssl_error(message)
        }
    })
}

fn tls_error(code: i32, message: String) -> pyre_interpreter::PyError {
    // `PyErr_NoMemory()`: a destination the allocator refused is a
    // `MemoryError` carrying no message, not an `SSLError` with an errno.
    if code == pyre_native::ssl::TLS_ERROR_NO_MEMORY {
        return pyre_interpreter::PyError::memory_error("");
    }
    let verify_code = (code >= pyre_native::ssl::TLS_ERROR_CERT_VERIFY_BASE)
        .then_some(code - pyre_native::ssl::TLS_ERROR_CERT_VERIFY_BASE);
    let class_name = match code {
        pyre_native::ssl::TLS_ERROR_WANT_READ => "ssl.SSLWantReadError",
        pyre_native::ssl::TLS_ERROR_WANT_WRITE => "ssl.SSLWantWriteError",
        pyre_native::ssl::TLS_ERROR_ZERO_RETURN => "ssl.SSLZeroReturnError",
        pyre_native::ssl::TLS_ERROR_EOF => "ssl.SSLEOFError",
        _ if verify_code.is_some() => "ssl.SSLCertVerificationError",
        _ => "ssl.SSLError",
    };
    // WANT_READ/WANT_WRITE travel this path on every non-blocking handshake
    // step. `library` / `reason` / `verify_*` are instance attributes, so the
    // instance is built first (the `set_error` shape) and then carried as
    // `OperationError(w_type, inst)`.
    let public_errno = if verify_code.is_some() {
        pyre_native::ssl::TLS_ERROR_SSL
    } else {
        code
    };
    let Some(class) = pyre_interpreter::builtins::lookup_exc_class(class_name) else {
        return pyre_interpreter::PyError::os_error(message);
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let class_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(class);
    let message_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(&message));
    let Ok(exception) = pyre_interpreter::builtins::exc_os_error_new(&[
        pyre_object::gc_roots::shadow_stack_get(class_slot),
        w_int_new(public_errno as i64),
        pyre_object::gc_roots::shadow_stack_get(message_slot),
    ]) else {
        return pyre_interpreter::PyError::os_error(message);
    };
    let exc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exception);
    set_library_reason(pyre_object::gc_roots::shadow_stack_get(exc_slot), &message);
    if let Some(verify_code) = verify_code {
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            pyre_object::gc_roots::shadow_stack_get(exc_slot),
            "verify_code",
            w_int_new(verify_code as i64),
        );
        let verify_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(
            pyre_native::ssl::certificate_verify_message(verify_code),
        ));
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            pyre_object::gc_roots::shadow_stack_get(exc_slot),
            "verify_message",
            pyre_object::gc_roots::shadow_stack_get(verify_slot),
        );
        let library_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_str_new_managed("SSL"));
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            pyre_object::gc_roots::shadow_stack_get(exc_slot),
            "library",
            pyre_object::gc_roots::shadow_stack_get(library_slot),
        );
        let reason_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_str_new_managed("CERTIFICATE_VERIFY_FAILED"));
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            pyre_object::gc_roots::shadow_stack_get(exc_slot),
            "reason",
            pyre_object::gc_roots::shadow_stack_get(reason_slot),
        );
    }
    pyre_interpreter::PyError::from_type_and_value(
        pyre_object::gc_roots::shadow_stack_get(class_slot),
        pyre_object::gc_roots::shadow_stack_get(exc_slot),
    )
}

fn tls_result<T>(result: pyre_native::ssl::TlsResult<T>) -> Result<T, pyre_interpreter::PyError> {
    result.map_err(|(code, message)| tls_error(code, message))
}

fn fs_path(obj: PyObjectRef) -> Result<std::path::PathBuf, pyre_interpreter::PyError> {
    if obj.is_null() || unsafe { is_none(obj) || is_bool(obj) } {
        return Err(pyre_interpreter::PyError::type_error(
            "path should be string, bytes, os.PathLike or integer, not NoneType",
        ));
    }
    // `PyUnicode_FSConverter`: a path is named by a `str`, by the filesystem
    // `bytes` that `str` encodes to, or by an `os.PathLike`, and all three
    // reach the host as those bytes. `fspath_buf` takes a `str` its caller has
    // already established — reading a `bytes` through it decodes the payload
    // as text and takes a length word out of the middle of the path.
    // Those bytes stay a path all the way to the host call. Decoding them to
    // text is lossy on a name the filesystem accepts but UTF-8 does not: two
    // distinct files collapse onto one `U+FFFD` spelling, and re-encoding that
    // spelling names neither of them.
    let bytes = pyre_interpreter::gateway::fsencode_bytes_w(obj)?;
    Ok(pyre_interpreter::gateway::os_string_from_fs_bytes(&bytes).into())
}

fn password_bytes(mut obj: PyObjectRef) -> Result<Option<Vec<u8>>, pyre_interpreter::PyError> {
    if obj.is_null() || unsafe { is_none(obj) } {
        return Ok(None);
    }
    let callable = !unsafe { is_str(obj) }
        && !unsafe { bytesobject::is_bytes_like(obj) }
        && pyre_object::with_roots!(obj => pyre_interpreter::baseobjspace::getattr_str(obj, "__call__")).is_ok();
    let value = if callable {
        pyre_interpreter::call::call_function_impl_result(obj, &[])?
    } else {
        obj
    };
    let bytes = if unsafe { is_str(value) } {
        pyre_interpreter::baseobjspace::str_utf8_w(value)?
            .as_bytes()
            .to_vec()
    } else if let Some(buffer) = pyre_interpreter::baseobjspace::simple_buffer_bytes(value)? {
        let bytes = buffer.as_bytes().to_vec();
        buffer.release();
        bytes
    } else {
        return Err(pyre_interpreter::PyError::type_error(if callable {
            "password callback must return a string"
        } else {
            "password should be a string or callable"
        }));
    };
    if bytes.len() > 1024 {
        return Err(pyre_interpreter::PyError::value_error(
            "password cannot be longer than 1024 bytes",
        ));
    }
    Ok(Some(bytes))
}

fn dict_from_pairs(items: &[(&str, PyObjectRef)]) -> PyObjectRef {
    let dict = w_dict_new();
    for (key, value) in items {
        unsafe { w_dict_setitem_str(dict, key, *value) };
    }
    dict
}

fn dict_put(dict_slot: usize, key: &str, value: PyObjectRef) {
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(value);
    unsafe {
        w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            key,
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        );
    }
}

fn decoded_name(cert: *const pyre_native::ssl::DecodedCertificate, subject: bool) -> PyObjectRef {
    let rdn_count = unsafe { pyre_native::ssl::certificate_name_rdn_count(cert, subject) };
    // Every tuple here is freshly allocated and the next allocation can collect,
    // so each level is pinned as it is produced; an inner bracket closes before
    // its tuple joins the level above, whose slots would otherwise sit inside it.
    let mut rdns = pyre_object::gc_roots::RootedItems::new();
    for rdn in 0..rdn_count {
        let attribute_count =
            unsafe { pyre_native::ssl::certificate_name_attribute_count(cert, subject, rdn) };
        let rdn_tuple = {
            let mut attributes = pyre_object::gc_roots::RootedItems::new();
            for attribute in 0..attribute_count {
                let key = unsafe {
                    pyre_native::ssl::certificate_name_attribute_key(cert, subject, rdn, attribute)
                };
                let value = unsafe {
                    pyre_native::ssl::certificate_name_attribute_value(
                        cert, subject, rdn, attribute,
                    )
                };
                let attribute_tuple = {
                    let mut pair = pyre_object::gc_roots::RootedItems::new();
                    pair.push(w_str_new_managed(&key));
                    pair.push(w_str_new_managed(&value));
                    w_tuple_new(pair.take())
                };
                attributes.push(attribute_tuple);
            }
            w_tuple_new(attributes.take())
        };
        rdns.push(rdn_tuple);
    }
    w_tuple_new(rdns.take())
}

fn decoded_directory_name(
    cert: *const pyre_native::ssl::DecodedCertificate,
    san: usize,
) -> PyObjectRef {
    let rdn_count = unsafe { pyre_native::ssl::certificate_san_directory_rdn_count(cert, san) };
    // Same nesting as `decoded_name`: each level is pinned as it is produced and
    // an inner bracket closes before its tuple joins the level above.
    let mut rdns = pyre_object::gc_roots::RootedItems::new();
    for rdn in 0..rdn_count {
        let count =
            unsafe { pyre_native::ssl::certificate_san_directory_attribute_count(cert, san, rdn) };
        let rdn_tuple = {
            let mut attributes = pyre_object::gc_roots::RootedItems::new();
            for attribute in 0..count {
                let key = unsafe {
                    pyre_native::ssl::certificate_san_directory_attribute_key(
                        cert, san, rdn, attribute,
                    )
                };
                let value = unsafe {
                    pyre_native::ssl::certificate_san_directory_attribute_value(
                        cert, san, rdn, attribute,
                    )
                };
                let attribute_tuple = {
                    let mut pair = pyre_object::gc_roots::RootedItems::new();
                    pair.push(w_str_new_managed(&key));
                    pair.push(w_str_new_managed(&value));
                    w_tuple_new(pair.take())
                };
                attributes.push(attribute_tuple);
            }
            w_tuple_new(attributes.take())
        };
        rdns.push(rdn_tuple);
    }
    w_tuple_new(rdns.take())
}

fn decoded_urls(
    cert: *const pyre_native::ssl::DecodedCertificate,
    kind: i32,
) -> Option<PyObjectRef> {
    let count = unsafe { pyre_native::ssl::certificate_url_count(cert, kind) };
    if count == 0 {
        return None;
    }
    let mut urls = pyre_object::gc_roots::RootedItems::new();
    for index in 0..count {
        urls.push(w_str_new_managed(&unsafe {
            pyre_native::ssl::certificate_url(cert, kind, index)
        }));
    }
    Some(w_tuple_new(urls.take()))
}

fn decoded_certificate_dict(cert: *mut pyre_native::ssl::DecodedCertificate) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_dict_new());
    dict_put(dict_slot, "issuer", decoded_name(cert, false));
    dict_put(
        dict_slot,
        "notAfter",
        w_str_new_managed(&unsafe { pyre_native::ssl::certificate_not_after(cert) }),
    );
    dict_put(
        dict_slot,
        "notBefore",
        w_str_new_managed(&unsafe { pyre_native::ssl::certificate_not_before(cert) }),
    );
    dict_put(
        dict_slot,
        "serialNumber",
        w_str_new_managed(&unsafe { pyre_native::ssl::certificate_serial_number(cert) }),
    );
    dict_put(dict_slot, "subject", decoded_name(cert, true));
    dict_put(
        dict_slot,
        "version",
        w_int_new(unsafe { pyre_native::ssl::certificate_version(cert) } as i64),
    );
    for (name, kind) in [("OCSP", 0), ("caIssuers", 1), ("crlDistributionPoints", 2)] {
        if let Some(urls) = decoded_urls(cert, kind) {
            dict_put(dict_slot, name, urls);
        }
    }
    let san_count = unsafe { pyre_native::ssl::certificate_san_count(cert) };
    if san_count != 0 {
        // Each name tuple is freshly allocated and the next entry allocates
        // again, so both the pair and the tuples are pinned as they arrive.
        let mut names = pyre_object::gc_roots::RootedItems::new();
        for index in 0..san_count {
            let kind = unsafe { pyre_native::ssl::certificate_san_kind(cert, index) };
            let entry = {
                let mut pair = pyre_object::gc_roots::RootedItems::new();
                pair.push(w_str_new_managed(kind));
                let value = if kind == "DirName" {
                    decoded_directory_name(cert, index)
                } else {
                    w_str_new_managed(&unsafe {
                        pyre_native::ssl::certificate_san_value(cert, index)
                    })
                };
                pair.push(value);
                w_tuple_new(pair.take())
            };
            names.push(entry);
        }
        dict_put(dict_slot, "subjectAltName", w_tuple_new(names.take()));
    }
    unsafe { pyre_native::ssl::certificate_free(cert) };
    pyre_object::gc_roots::shadow_stack_get(dict_slot)
}

fn parse_server_hostname(
    value: Option<PyObjectRef>,
) -> Result<(Option<String>, PyObjectRef), pyre_interpreter::PyError> {
    let Some(value) = value.filter(|value| !unsafe { is_none(*value) }) else {
        return Ok((None, w_none()));
    };
    if !unsafe { is_str(value) } {
        return Err(pyre_interpreter::PyError::type_error(
            "server_hostname must be a string",
        ));
    }
    let hostname = pyre_interpreter::baseobjspace::str_utf8_w(value)?.to_string();
    if hostname.as_bytes().contains(&0) {
        return Err(pyre_interpreter::PyError::type_error(
            "argument must be encoded string without null bytes",
        ));
    }
    if hostname.is_empty() || hostname.starts_with('.') {
        return Err(pyre_interpreter::PyError::value_error(
            "server_hostname cannot be an empty string or start with a leading dot",
        ));
    }
    if !hostname.is_ascii() {
        return Err(pyre_interpreter::PyError::value_error(
            "server_hostname must contain ASCII only",
        ));
    }
    Ok((Some(hostname.clone()), w_str_new_managed(&hostname)))
}

fn allocate_ssl_socket(
    backend: *mut pyre_native::ssl::TlsConnection,
    context: PyObjectRef,
    socket: PyObjectRef,
    incoming: PyObjectRef,
    outgoing: PyObjectRef,
    owner: PyObjectRef,
    server_hostname: PyObjectRef,
    server_side: bool,
    requested_session: *mut pyre_native::ssl::NativeSession,
) -> PyObjectRef {
    // Offsets into the pinned-root array below.  The weakref boxing further
    // down rewrites two of these slots by position, so a reorder of the array
    // has to move these constants with it.
    const CONTEXT_SLOT: usize = 0;
    const SOCKET_SLOT: usize = 1;
    const INCOMING_SLOT: usize = 2;
    const OUTGOING_SLOT: usize = 3;
    const OWNER_SLOT: usize = 4;
    const SERVER_HOSTNAME_SLOT: usize = 5;

    let _ = ssl_socket_methods::type_object();
    let _roots = pyre_object::gc_roots::push_roots();
    // Bound rather than spelled twice: `first` is derived from this array's
    // own length, so pinning one more value cannot leave the block start
    // pointing one slot past where the pins actually begin.
    let pinned = [context, socket, incoming, outgoing, owner, server_hostname];
    for value in pinned {
        let _ = pyre_object::gc_roots::pin_root(value);
    }
    let first = pyre_object::gc_roots::shadow_stack_len() - pinned.len();
    for slot in [SOCKET_SLOT, OWNER_SLOT] {
        let rooted = { pyre_object::gc_roots::shadow_stack_get(first + slot) };
        if !unsafe { is_none(rooted) } {
            let weak = pyre_object::weakref::w_gc_weakref_box_new_or_strong(rooted);
            pyre_object::gc_roots::shadow_stack_set(first + slot, weak);
        }
    }
    W_SSLSocket::allocate_stable(W_SSLSocket {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        backend,
        context: { pyre_object::gc_roots::shadow_stack_get(first + CONTEXT_SLOT) },
        socket: { pyre_object::gc_roots::shadow_stack_get(first + SOCKET_SLOT) },
        incoming: { pyre_object::gc_roots::shadow_stack_get(first + INCOMING_SLOT) },
        outgoing: { pyre_object::gc_roots::shadow_stack_get(first + OUTGOING_SLOT) },
        owner: { pyre_object::gc_roots::shadow_stack_get(first + OWNER_SLOT) },
        server_hostname: { pyre_object::gc_roots::shadow_stack_get(first + SERVER_HOSTNAME_SLOT) },
        server_side,
        shutdown_started: false,
        record: RecordCursor::default(),
        requested_session,
    })
}

fn allocate_ssl_session(backend: *mut pyre_native::ssl::NativeSession) -> PyObjectRef {
    let _ = ssl_session_methods::type_object();
    W_SSLSession::allocate_stable(W_SSLSession {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        backend,
    })
}

fn allocate_certificate(der: Vec<u8>) -> PyObjectRef {
    let _ = certificate_methods::type_object();
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_bytes_from_bytes(&der));
    let der_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    W_Certificate::allocate_stable(W_Certificate {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        der: { pyre_object::gc_roots::shadow_stack_get(der_slot) },
        hash: -1,
    })
}

fn clone_requested_session(
    value: Option<PyObjectRef>,
    context: *mut pyre_native::ssl::Context,
) -> Result<*mut pyre_native::ssl::NativeSession, pyre_interpreter::PyError> {
    let Some(value) = value.filter(|value| !unsafe { is_none(*value) }) else {
        return Ok(std::ptr::null_mut());
    };
    let session = W_SSLSession::from_obj(value)
        .ok_or_else(|| pyre_interpreter::PyError::type_error("Value is not a SSLSession."))?;
    if unsafe { pyre_native::ssl::session_context_identity(session.backend) }
        != unsafe { pyre_native::ssl::context_identity(context) }
    {
        return Err(pyre_interpreter::PyError::value_error(
            "Session refers to a different SSLContext.",
        ));
    }
    Ok(unsafe { pyre_native::ssl::session_clone(session.backend) })
}

mod context_methods {
    use super::*;

    fn pin_and_allocate_context(
        cls: PyObjectRef,
        backend: *mut pyre_native::ssl::Context,
    ) -> PyObjectRef {
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(cls);
        let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        W_SSLContext::allocate_instance(
            W_SSLContext {
                ob: PyObject {
                    ob_type: std::ptr::null(),
                    w_class: std::ptr::null_mut(),
                },
                backend,
                sni_callback: w_none(),
                msg_callback: w_none(),
                keylog_filename: w_none(),
                host_flags: 0,
                post_handshake_auth: false,
                num_tickets: 2,
            },
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        )
    }

    #[pyre_interpreter::pyre_methods(doc = "An SSLContext holds TLS configuration and state.")]
    impl W_SSLContext {
        #[staticmethod]
        fn __new__(
            mut cls: PyObjectRef,
            args: &[PyObjectRef],
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
            if pyre_interpreter::builtins::has_real_kwargs(kwargs) {
                return Err(pyre_interpreter::PyError::type_error(
                    "_SSLContext.__new__() takes no keyword arguments",
                ));
            }
            let protocol_obj = positional.get(1).copied().ok_or_else(|| {
                pyre_interpreter::PyError::type_error(
                    "_SSLContext.__new__() missing required argument 'protocol'",
                )
            })?;
            if positional.len() != 2 {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "_SSLContext.__new__() takes 2 arguments ({} given)",
                    positional.len()
                )));
            }
            pyre_interpreter::typedef::check_user_subclass(type_object(), cls)?;
            let protocol = pyre_object::with_roots!(cls => pyre_interpreter::baseobjspace::int_w(protocol_obj))?;
            let protocol = i32::try_from(protocol).map_err(|_| {
                pyre_interpreter::PyError::value_error("invalid or unsupported protocol version")
            })?;
            let backend = pyre_native::ssl::context_new(protocol)
                .map_err(|message| pyre_interpreter::PyError::value_error(message))?;
            if matches!(
                protocol,
                PROTOCOL_TLS | PROTOCOL_TLSV1 | PROTOCOL_TLSV1_1 | PROTOCOL_TLSV1_2
            ) {
                pyre_object::with_roots!(cls => pyre_interpreter::warn::warn_deprecation(&format!(
                    "ssl.PROTOCOL_{} is deprecated",
                    match protocol {
                        PROTOCOL_TLS => "TLS",
                        PROTOCOL_TLSV1 => "TLSv1",
                        PROTOCOL_TLSV1_1 => "TLSv1_1",
                        PROTOCOL_TLSV1_2 => "TLSv1_2",
                        _ => unreachable!(),
                    }
                )))?;
            }
            Ok(pin_and_allocate_context(cls, backend))
        }

        #[getter]
        fn protocol(&self) -> i64 {
            unsafe { pyre_native::ssl::context_protocol(self.backend) as i64 }
        }

        #[getter]
        fn check_hostname(&self) -> bool {
            unsafe { pyre_native::ssl::context_check_hostname(self.backend) }
        }

        #[setter]
        fn set_check_hostname(&mut self, value: bool) {
            unsafe {
                pyre_native::ssl::context_set_check_hostname(self.backend, value);
                if value && pyre_native::ssl::context_verify_mode(self.backend) == CERT_NONE {
                    pyre_native::ssl::context_set_verify_mode(self.backend, CERT_REQUIRED);
                }
            }
        }

        #[getter]
        fn verify_mode(&self) -> i64 {
            unsafe { pyre_native::ssl::context_verify_mode(self.backend) as i64 }
        }

        #[setter]
        fn set_verify_mode(&mut self, value: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            let mode = pyre_interpreter::baseobjspace::int_w(value)?;
            if !(CERT_NONE as i64..=CERT_REQUIRED as i64).contains(&mode) {
                return Err(pyre_interpreter::PyError::value_error(
                    "invalid verify mode",
                ));
            }
            if mode == CERT_NONE as i64 && self.check_hostname() {
                return Err(pyre_interpreter::PyError::value_error(
                    "Cannot set verify_mode to CERT_NONE when check_hostname is enabled",
                ));
            }
            unsafe { pyre_native::ssl::context_set_verify_mode(self.backend, mode as i32) };
            Ok(())
        }

        #[getter]
        fn verify_flags(&self) -> i64 {
            unsafe { pyre_native::ssl::context_verify_flags(self.backend) as i64 }
        }

        #[setter]
        fn set_verify_flags(
            &mut self,
            value: PyObjectRef,
        ) -> Result<(), pyre_interpreter::PyError> {
            let value = pyre_interpreter::baseobjspace::int_w(value)?;
            let value = i32::try_from(value).map_err(|_| {
                pyre_interpreter::PyError::overflow_error("verify_flags out of range")
            })?;
            unsafe { pyre_native::ssl::context_set_verify_flags(self.backend, value) };
            Ok(())
        }

        #[getter]
        fn options(&self) -> i64 {
            unsafe { pyre_native::ssl::context_options(self.backend) as i64 }
        }

        #[setter]
        fn set_options(&mut self, value: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            let value = pyre_interpreter::baseobjspace::uint_w(value)?;
            let old = unsafe { pyre_native::ssl::context_options(self.backend) };
            const DEPRECATED: u64 =
                0x0200_0000 | 0x0400_0000 | 0x1000_0000 | 0x0800_0000 | 0x2000_0000;
            if (!old & value) & DEPRECATED != 0 {
                pyre_interpreter::warn::warn_deprecation(
                    "ssl.OP_NO_SSL*/ssl.OP_NO_TLS* options are deprecated",
                )?;
            }
            unsafe { pyre_native::ssl::context_set_options(self.backend, value) };
            Ok(())
        }

        #[getter]
        fn minimum_version(&self) -> i64 {
            unsafe { pyre_native::ssl::context_minimum_version(self.backend) as i64 }
        }

        #[setter]
        fn set_minimum_version(
            &mut self,
            value: PyObjectRef,
        ) -> Result<(), pyre_interpreter::PyError> {
            let value = validate_tls_version(pyre_interpreter::baseobjspace::int_w(value)?)?;
            let value = if value == -1 { 0x304 } else { value };
            unsafe { pyre_native::ssl::context_set_minimum_version(self.backend, value) };
            Ok(())
        }

        #[getter]
        fn maximum_version(&self) -> i64 {
            unsafe { pyre_native::ssl::context_maximum_version(self.backend) as i64 }
        }

        #[setter]
        fn set_maximum_version(
            &mut self,
            value: PyObjectRef,
        ) -> Result<(), pyre_interpreter::PyError> {
            let value = validate_tls_version(pyre_interpreter::baseobjspace::int_w(value)?)?;
            let value = if value == -2 { 0x303 } else { value };
            unsafe { pyre_native::ssl::context_set_maximum_version(self.backend, value) };
            Ok(())
        }

        #[getter]
        fn _host_flags(&self) -> i64 {
            self.host_flags as i64
        }

        #[setter]
        fn set_host_flags(&mut self, value: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            let value = pyre_interpreter::baseobjspace::int_w(value)?;
            self.host_flags = i32::try_from(value).map_err(|_| {
                pyre_interpreter::PyError::overflow_error("_host_flags out of range")
            })?;
            Ok(())
        }

        #[getter]
        fn post_handshake_auth(&self) -> bool {
            self.post_handshake_auth
        }

        #[setter]
        fn set_post_handshake_auth(&mut self, value: bool) {
            self.post_handshake_auth = value;
        }

        #[getter]
        fn keylog_filename(&self) -> PyObjectRef {
            self.keylog_filename
        }

        #[setter]
        fn set_keylog_filename(
            &mut self,
            mut value: PyObjectRef,
        ) -> Result<(), pyre_interpreter::PyError> {
            // PyPy `_SSLContext.keylog_filename` accepts exactly str or None,
            // closes the old BIO first, opens the replacement in append mode,
            // and retains the original Python str object for the getter.
            if unsafe { is_none(value) } {
                native_result(unsafe {
                    pyre_native::ssl::context_set_keylog_filename(self.backend, None)
                })?;
                pyre_object::gc_hook::try_gc_write_barrier(
                    self as *mut W_SSLContext as pyre_object::gc_hook::GCREF,
                );
                self.keylog_filename = w_none();
                return Ok(());
            }
            if !unsafe { pyre_interpreter::baseobjspace::isinstance_str_w(value) } {
                return Err(pyre_interpreter::PyError::type_error("str path expected"));
            }
            let path = pyre_object::with_roots!(value => fs_path(value))?;
            native_result(unsafe {
                pyre_native::ssl::context_set_keylog_filename(self.backend, Some(&path))
            })?;
            pyre_object::gc_hook::try_gc_write_barrier(
                self as *mut W_SSLContext as pyre_object::gc_hook::GCREF,
            );
            self.keylog_filename = value;
            Ok(())
        }

        #[getter]
        fn num_tickets(&self) -> i64 {
            self.num_tickets as i64
        }

        #[setter]
        fn set_num_tickets(&mut self, value: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            let value = pyre_interpreter::baseobjspace::int_w(value)?;
            if value < 0 {
                return Err(pyre_interpreter::PyError::value_error(
                    "num_tickets must be a non-negative integer",
                ));
            }
            if self.protocol() != PROTOCOL_TLS_SERVER as i64 {
                return Err(pyre_interpreter::PyError::value_error(
                    "num_tickets can only be set on server-side contexts",
                ));
            }
            self.num_tickets = i32::try_from(value).map_err(|_| {
                pyre_interpreter::PyError::overflow_error("num_tickets out of range")
            })?;
            unsafe {
                pyre_native::ssl::context_set_num_tickets(self.backend, self.num_tickets as usize)
            };
            Ok(())
        }

        #[getter]
        fn sni_callback(&self) -> PyObjectRef {
            self.sni_callback
        }

        #[setter]
        fn set_sni_callback(
            &mut self,
            mut value: PyObjectRef,
        ) -> Result<(), pyre_interpreter::PyError> {
            if !unsafe { is_none(value) }
                && pyre_object::with_roots!(value => pyre_interpreter::baseobjspace::getattr_str(value, "__call__")).is_err()
            {
                return Err(pyre_interpreter::PyError::type_error(
                    "not a callable object",
                ));
            }
            pyre_object::gc_hook::try_gc_write_barrier(
                self as *mut W_SSLContext as pyre_object::gc_hook::GCREF,
            );
            self.sni_callback = value;
            Ok(())
        }

        #[getter]
        fn _msg_callback(&self) -> PyObjectRef {
            self.msg_callback
        }

        #[setter]
        fn set_msg_callback(
            &mut self,
            mut value: PyObjectRef,
        ) -> Result<(), pyre_interpreter::PyError> {
            if !unsafe { is_none(value) }
                && pyre_object::with_roots!(value => pyre_interpreter::baseobjspace::getattr_str(value, "__call__")).is_err()
            {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "{} is not callable.",
                    pyre_interpreter::type_methods::arg_type_name(value)
                )));
            }
            pyre_object::gc_hook::try_gc_write_barrier(
                self as *mut W_SSLContext as pyre_object::gc_hook::GCREF,
            );
            self.msg_callback = value;
            Ok(())
        }

        fn load_cert_chain(
            &mut self,
            args: &[PyObjectRef],
        ) -> Result<(), pyre_interpreter::PyError> {
            const KEYWORDS: &[&str] = &["certfile", "keyfile", "password"];
            let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
            let user = positional.get(1..).unwrap_or(&[]);
            if user.len() > 3 {
                return Err(pyre_interpreter::PyError::type_error(
                    "load_cert_chain() takes at most 3 arguments",
                ));
            }
            let cert = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                0,
                "certfile",
                "load_cert_chain",
                1,
            )?
            .ok_or_else(|| {
                pyre_interpreter::PyError::type_error(
                    "load_cert_chain() missing required argument 'certfile'",
                )
            })?;
            let key = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                1,
                "keyfile",
                "load_cert_chain",
                2,
            )?;
            let password = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                2,
                "password",
                "load_cert_chain",
                3,
            )?;
            pyre_interpreter::builtins::kwarg_reject_unknown(kwargs, KEYWORDS, "load_cert_chain")?;
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[
                key.unwrap_or(pyre_object::PY_NULL),
                password.unwrap_or(pyre_object::PY_NULL),
            ]);
            let cert_path = fs_path(cert);
            let w = roots.get(base);
            let key = if w.is_null() { None } else { Some(w) };
            let w = roots.get(base + 1);
            let mut password = if w.is_null() { None } else { Some(w) };
            drop(roots);
            let cert_path = cert_path?;
            let key_path = match key {
                Some(value) if !unsafe { is_none(value) } => {
                    let roots = pyre_object::gc_roots::push_roots();
                    let base = roots.pin_roots(&[password.unwrap_or(pyre_object::PY_NULL)]);
                    let key_path = fs_path(value);
                    let w = roots.get(base);
                    password = if w.is_null() { None } else { Some(w) };
                    drop(roots);
                    key_path?
                }
                _ => cert_path.clone(),
            };
            // OpenSSL asks its callback only after parsing discovers an
            // encrypted key. Preserve that observable ordering: an irrelevant
            // callback must not run for an unencrypted key file.  The backend
            // reports which case it hit, so the container decides this rather
            // than a scan of the file for header text.
            let loaded = native_result(unsafe {
                pyre_native::ssl::context_load_cert_chain(self.backend, &cert_path, &key_path, None)
            })?;
            if loaded {
                return Ok(());
            }
            let Some(password) = password_bytes(password.unwrap_or_else(w_none))? else {
                return Err(ssl_error("[SSL] PEM routines: bad password read"));
            };
            native_result(unsafe {
                pyre_native::ssl::context_load_cert_chain(
                    self.backend,
                    &cert_path,
                    &key_path,
                    Some(&password),
                )
            })
            .map(|_| ())
        }

        fn load_verify_locations(
            &mut self,
            args: &[PyObjectRef],
        ) -> Result<(), pyre_interpreter::PyError> {
            const KEYWORDS: &[&str] = &["cafile", "capath", "cadata"];
            let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
            let user = positional.get(1..).unwrap_or(&[]);
            if user.len() > 3 {
                return Err(pyre_interpreter::PyError::type_error(
                    "load_verify_locations() takes at most 3 arguments",
                ));
            }
            let cafile = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                0,
                "cafile",
                "load_verify_locations",
                1,
            )?;
            let capath = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                1,
                "capath",
                "load_verify_locations",
                2,
            )?;
            let cadata = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                2,
                "cadata",
                "load_verify_locations",
                3,
            )?;
            pyre_interpreter::builtins::kwarg_reject_unknown(
                kwargs,
                KEYWORDS,
                "load_verify_locations",
            )?;
            let cafile = cafile.filter(|value| !unsafe { is_none(*value) });
            let mut capath = capath.filter(|value| !unsafe { is_none(*value) });
            let mut cadata = cadata.filter(|value| !unsafe { is_none(*value) });
            if cafile.is_none() && capath.is_none() && cadata.is_none() {
                return Err(pyre_interpreter::PyError::type_error(
                    "cafile, capath and cadata cannot be all omitted",
                ));
            }
            if let Some(cafile) = cafile {
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[
                    capath.unwrap_or(pyre_object::PY_NULL),
                    cadata.unwrap_or(pyre_object::PY_NULL),
                ]);
                let path = fs_path(cafile);
                let w = roots.get(base);
                capath = if w.is_null() { None } else { Some(w) };
                let w = roots.get(base + 1);
                cadata = if w.is_null() { None } else { Some(w) };
                drop(roots);
                let path = path?;
                native_result(unsafe {
                    pyre_native::ssl::context_load_verify_file(self.backend, &path)
                })?;
            }
            if let Some(capath) = capath {
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[cadata.unwrap_or(pyre_object::PY_NULL)]);
                let path = fs_path(capath);
                let w = roots.get(base);
                cadata = if w.is_null() { None } else { Some(w) };
                drop(roots);
                let path = path?;
                if !path.is_dir() {
                    return Err(pyre_interpreter::PyError::os_error_with_errno(
                        libc::ENOENT,
                        "CA directory does not exist",
                    ));
                }
                unsafe { pyre_native::ssl::context_add_verify_dir(self.backend, &path) };
            }
            if let Some(cadata) = cadata {
                if unsafe { is_str(cadata) } {
                    let data = pyre_interpreter::baseobjspace::str_utf8_w(cadata)?;
                    unsafe {
                        pyre_native::ssl::context_load_verify_data(
                            self.backend,
                            data.as_bytes(),
                            true,
                        )
                    }
                    .map_err(|_| {
                        ssl_error("no start line: cadata does not contain a certificate")
                    })?;
                } else {
                    let buffer = pyre_interpreter::baseobjspace::simple_buffer_bytes(cadata)?
                        .ok_or_else(|| {
                            pyre_interpreter::PyError::type_error(
                                "cadata should be an ASCII string or a bytes-like object",
                            )
                        })?;
                    let loaded = unsafe {
                        pyre_native::ssl::context_load_verify_data(
                            self.backend,
                            buffer.as_bytes(),
                            false,
                        )
                    };
                    let mut failure = match loaded {
                        Ok(_) => None,
                        Err(_) => Some(ssl_error(
                            "not enough data: cadata does not contain a certificate",
                        )),
                    };
                    if let Some(err) = failure.take() {
                        let _roots = pyre_object::gc_roots::push_roots();
                        let mut err = err;
                        let slot = err.pin(&_roots);
                        buffer.release();
                        err.reload(&_roots, slot);
                        return Err(err);
                    }
                    buffer.release();
                }
            }
            Ok(())
        }

        /// `SSL_CTX_set_default_verify_paths` loads two independent sources: a
        /// bundle file and a hashed directory.  `SSL_CERT_FILE` overrides only
        /// the first and `SSL_CERT_DIR` only the second, so neither variable
        /// may suppress the other's source.
        fn set_default_verify_paths(&mut self) -> Result<(), pyre_interpreter::PyError> {
            // `SSL_CERT_FILE` and `SSL_CERT_DIR` name files, so their bytes
            // reach the host as a path for the same reason `fs_path`'s do.
            let env_path = |name: &[u8]| {
                pyre_interpreter::host_seam::getenv(name)
                    .ok()
                    .flatten()
                    .map(|value| {
                        std::path::PathBuf::from(
                            pyre_interpreter::gateway::os_string_from_fs_bytes(&value),
                        )
                    })
            };
            let cert_file = env_path(b"SSL_CERT_FILE").filter(|path| path.is_file());
            match cert_file {
                Some(path) => native_result(unsafe {
                    pyre_native::ssl::context_load_verify_file(self.backend, &path)
                })
                .map(|_| ())?,
                None => native_result(unsafe {
                    pyre_native::ssl::context_load_native_roots(self.backend)
                })
                .map(|_| ())?,
            }
            let (_, default_dir) = pyre_native::ssl::default_verify_paths();
            let cert_dir = env_path(b"SSL_CERT_DIR").unwrap_or_else(|| default_dir.into());
            if cert_dir.is_dir() {
                // OpenSSL defers hashed directory loading until chain lookup,
                // so it does not contribute to cert_store_stats here.
                unsafe { pyre_native::ssl::context_add_verify_dir(self.backend, &cert_dir) };
            }
            Ok(())
        }

        fn cert_store_stats(&self) -> PyObjectRef {
            let (x509, ca) = unsafe { pyre_native::ssl::context_cert_store_stats(self.backend) };
            dict_from_pairs(&[
                ("x509", w_int_new(x509 as i64)),
                ("crl", w_int_new(0)),
                ("x509_ca", w_int_new(ca as i64)),
            ])
        }

        fn get_ca_certs(
            &self,
            #[default(false)] binary_form: bool,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let certs = unsafe { pyre_native::ssl::context_ca_certs(self.backend) };
            // Each entry is freshly allocated and decoding the next one allocates
            // again, so they are pinned as they arrive.
            let mut result = pyre_object::gc_roots::RootedItems::new();
            for cert in certs {
                result.push(if binary_form {
                    w_bytes_from_bytes(&cert)
                } else {
                    let decoded = native_result(pyre_native::ssl::certificate_decode_der(&cert))?;
                    decoded_certificate_dict(decoded)
                });
            }
            Ok(w_list_new(result.take()))
        }

        fn set_ciphers(
            &mut self,
            cipherlist: PyObjectRef,
        ) -> Result<(), pyre_interpreter::PyError> {
            if !unsafe { is_str(cipherlist) } {
                return Err(pyre_interpreter::PyError::type_error(
                    "cipherlist must be a string",
                ));
            }
            let cipherlist = pyre_interpreter::baseobjspace::str_utf8_w(cipherlist)?;
            unsafe { pyre_native::ssl::context_set_cipher_list(self.backend, cipherlist.as_ref()) }
                .map_err(ssl_error)
        }

        /// The suites this context can actually negotiate, so a `set_ciphers()`
        /// filter is visible to configuration auditing rather than only to
        /// connection creation.
        fn get_ciphers(&self) -> PyObjectRef {
            // Each dict is freshly allocated and building the next one allocates
            // again, so they are pinned as they arrive.
            let mut ciphers = pyre_object::gc_roots::RootedItems::new();
            for index in 0..pyre_native::ssl::cipher_count() {
                if !unsafe { pyre_native::ssl::context_cipher_enabled(self.backend, index) } {
                    continue;
                }
                let name = pyre_native::ssl::cipher_name(index);
                let bits = pyre_native::ssl::cipher_bits(index);
                let description = pyre_native::ssl::cipher_description(index);
                let cipher = {
                    let _roots = pyre_object::gc_roots::push_roots();
                    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(w_dict_new());
                    dict_put(
                        dict_slot,
                        "id",
                        w_int_new(pyre_native::ssl::cipher_id(index)),
                    );
                    dict_put(dict_slot, "name", w_str_new_managed(name));
                    dict_put(
                        dict_slot,
                        "protocol",
                        w_str_new_managed(pyre_native::ssl::cipher_protocol(index)),
                    );
                    dict_put(dict_slot, "description", w_str_new_managed(&description));
                    dict_put(dict_slot, "strength_bits", w_int_new(bits as i64));
                    dict_put(dict_slot, "alg_bits", w_int_new(bits as i64));
                    dict_put(
                        dict_slot,
                        "aead",
                        w_bool_from(pyre_native::ssl::cipher_aead(index)),
                    );
                    dict_put(
                        dict_slot,
                        "symmetric",
                        w_str_new_managed(pyre_native::ssl::cipher_symmetric(index)),
                    );
                    dict_put(
                        dict_slot,
                        "digest",
                        w_str_new_managed(pyre_native::ssl::cipher_digest(index)),
                    );
                    dict_put(
                        dict_slot,
                        "kea",
                        w_str_new_managed(pyre_native::ssl::cipher_kea(index)),
                    );
                    dict_put(
                        dict_slot,
                        "auth",
                        w_str_new_managed(pyre_native::ssl::cipher_auth(index)),
                    );
                    pyre_object::gc_roots::shadow_stack_get(dict_slot)
                };
                ciphers.push(cipher);
            }
            w_list_new(ciphers.take())
        }

        fn session_stats(&self) -> PyObjectRef {
            let (accept, hits) = unsafe { pyre_native::ssl::context_session_stats(self.backend) };
            dict_from_pairs(&[
                ("number", w_int_new(0)),
                ("connect", w_int_new(0)),
                ("connect_good", w_int_new(0)),
                ("connect_renegotiate", w_int_new(0)),
                ("accept", w_int_new(accept as i64)),
                ("accept_good", w_int_new(accept as i64)),
                ("accept_renegotiate", w_int_new(0)),
                ("hits", w_int_new(hits as i64)),
                ("misses", w_int_new(0)),
                ("timeouts", w_int_new(0)),
                ("cache_full", w_int_new(0)),
            ])
        }

        /// Validate the file the way `SSL_CTX_set_tmp_dh` would, then discard
        /// the group.  rustls negotiates ECDHE only, so no finite-field DH
        /// parameter can ever take effect; a server that set one still gets a
        /// forward-secret key exchange, and `supports_kx_alias` reports no DHE
        /// so callers can detect the gap.  Raising here instead would reject
        /// `load_dh_params` outright for every caller.
        fn load_dh_params(&self, path: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            if path.is_null() || unsafe { is_none(path) } {
                return Err(pyre_interpreter::PyError::type_error(
                    "load_dh_params() missing required path argument",
                ));
            }
            let path = fs_path(path)?;
            let data = std::fs::read(&path).map_err(|error| {
                pyre_interpreter::PyError::os_error_with_errno(
                    error.raw_os_error().unwrap_or(libc::EIO),
                    error.to_string(),
                )
            })?;
            if !data
                .windows(b"BEGIN DH PARAMETERS".len())
                .any(|window| window == b"BEGIN DH PARAMETERS")
            {
                return Err(ssl_error("[PEM: NO_START_LINE] no start line"));
            }
            Ok(())
        }

        fn set_ecdh_curve(&self, curve: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            let curve = if unsafe { is_str(curve) } {
                pyre_interpreter::baseobjspace::str_utf8_w(curve)?.to_string()
            } else if unsafe { bytesobject::is_bytes_like(curve) } {
                String::from_utf8(unsafe { bytesobject::bytes_like_data(curve) }.to_vec())
                    .map_err(|_| pyre_interpreter::PyError::value_error("invalid curve name"))?
            } else {
                return Err(pyre_interpreter::PyError::type_error(
                    "curve_name must be a string",
                ));
            };
            unsafe { pyre_native::ssl::context_set_ecdh_curve(self.backend, &curve) }
                .map_err(pyre_interpreter::PyError::value_error)
        }

        #[getter]
        fn security_level(&self) -> i64 {
            2
        }

        fn _wrap_bio(
            &self,
            args: &[PyObjectRef],
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            const KEYWORDS: &[&str] = &[
                "incoming",
                "outgoing",
                "server_side",
                "server_hostname",
                "owner",
                "session",
            ];
            let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
            let user = positional.get(1..).unwrap_or(&[]);
            if user.len() > KEYWORDS.len() {
                return Err(pyre_interpreter::PyError::type_error(
                    "_wrap_bio() takes at most 6 arguments",
                ));
            }
            let mut incoming = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                0,
                "incoming",
                "_wrap_bio",
                1,
            )?
            .ok_or_else(|| {
                pyre_interpreter::PyError::type_error(
                    "_wrap_bio() missing required argument 'incoming'",
                )
            })?;
            let mut outgoing = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                1,
                "outgoing",
                "_wrap_bio",
                2,
            )?
            .ok_or_else(|| {
                pyre_interpreter::PyError::type_error(
                    "_wrap_bio() missing required argument 'outgoing'",
                )
            })?;
            let server_side = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                2,
                "server_side",
                "_wrap_bio",
                3,
            )?
            .map(pyre_interpreter::baseobjspace::is_true)
            .transpose()?
            .unwrap_or(false);
            let hostname = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                3,
                "server_hostname",
                "_wrap_bio",
                4,
            )?;
            let mut owner = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                4,
                "owner",
                "_wrap_bio",
                5,
            )?
            .unwrap_or_else(w_none);
            let session = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                5,
                "session",
                "_wrap_bio",
                6,
            )?;
            pyre_interpreter::builtins::kwarg_reject_unknown(kwargs, KEYWORDS, "_wrap_bio")?;
            if W_MemoryBIO::from_obj(incoming).is_none()
                || W_MemoryBIO::from_obj(outgoing).is_none()
            {
                return Err(pyre_interpreter::PyError::type_error(
                    "incoming and outgoing must be MemoryBIO objects",
                ));
            }
            if let Some(session) = session.filter(|value| !unsafe { is_none(*value) })
                && W_SSLSession::from_obj(session).is_none()
            {
                return Err(pyre_interpreter::PyError::type_error(
                    "Value is not a SSLSession.",
                ));
            }
            let has_session = session.is_some();
            let session_roots = pyre_object::gc_roots::push_roots();
            let session_slot = session_roots.pin_roots(&[session.unwrap_or(pyre_object::PY_NULL)]);
            let (hostname, hostname_obj) = pyre_object::with_roots!(incoming, outgoing, owner => parse_server_hostname(hostname))?;
            let session = has_session.then(|| session_roots.get(session_slot));
            if !server_side
                && unsafe { pyre_native::ssl::context_check_hostname(self.backend) }
                && hostname.is_none()
            {
                return Err(pyre_interpreter::PyError::value_error(
                    "check_hostname requires server_hostname",
                ));
            }
            let protocol = unsafe { pyre_native::ssl::context_protocol(self.backend) };
            if server_side && protocol == PROTOCOL_TLS_CLIENT {
                return Err(ssl_error(
                    "Cannot create a server socket with a PROTOCOL_TLS_CLIENT context",
                ));
            }
            if !server_side && protocol == PROTOCOL_TLS_SERVER {
                return Err(ssl_error(
                    "Cannot create a client socket with a PROTOCOL_TLS_SERVER context",
                ));
            }
            let requested_session = clone_requested_session(session, self.backend)?;
            Ok(allocate_ssl_socket(
                std::ptr::null_mut(),
                self as *const W_SSLContext as PyObjectRef,
                w_none(),
                incoming,
                outgoing,
                owner,
                hostname_obj,
                server_side,
                requested_session,
            ))
        }

        fn _wrap_socket(
            &self,
            args: &[PyObjectRef],
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            const KEYWORDS: &[&str] =
                &["sock", "server_side", "server_hostname", "owner", "session"];
            let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
            let user = positional.get(1..).unwrap_or(&[]);
            if user.len() > KEYWORDS.len() {
                return Err(pyre_interpreter::PyError::type_error(
                    "_wrap_socket() takes at most 5 arguments",
                ));
            }
            let mut socket = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                0,
                "sock",
                "_wrap_socket",
                1,
            )?
            .ok_or_else(|| {
                pyre_interpreter::PyError::type_error(
                    "_wrap_socket() missing required argument 'sock'",
                )
            })?;
            let server_side = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                1,
                "server_side",
                "_wrap_socket",
                2,
            )?
            .map(pyre_interpreter::baseobjspace::is_true)
            .transpose()?
            .unwrap_or(false);
            let hostname = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                2,
                "server_hostname",
                "_wrap_socket",
                3,
            )?;
            let mut owner = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                3,
                "owner",
                "_wrap_socket",
                4,
            )?
            .unwrap_or_else(w_none);
            let session = pyre_interpreter::builtins::bind_pos_or_kw(
                user,
                kwargs,
                4,
                "session",
                "_wrap_socket",
                5,
            )?;
            pyre_interpreter::builtins::kwarg_reject_unknown(kwargs, KEYWORDS, "_wrap_socket")?;
            if let Some(session) = session.filter(|value| !unsafe { is_none(*value) })
                && W_SSLSession::from_obj(session).is_none()
            {
                return Err(pyre_interpreter::PyError::type_error(
                    "Value is not a SSLSession.",
                ));
            }
            let has_session = session.is_some();
            let session_roots = pyre_object::gc_roots::push_roots();
            let session_slot = session_roots.pin_roots(&[session.unwrap_or(pyre_object::PY_NULL)]);
            let (hostname, hostname_obj) =
                pyre_object::with_roots!(owner, socket => parse_server_hostname(hostname))?;
            let session = has_session.then(|| session_roots.get(session_slot));
            if !server_side
                && unsafe { pyre_native::ssl::context_check_hostname(self.backend) }
                && hostname.is_none()
            {
                return Err(pyre_interpreter::PyError::value_error(
                    "check_hostname requires server_hostname",
                ));
            }
            let protocol = unsafe { pyre_native::ssl::context_protocol(self.backend) };
            if server_side && protocol == PROTOCOL_TLS_CLIENT {
                return Err(ssl_error(
                    "Cannot create a server socket with a PROTOCOL_TLS_CLIENT context",
                ));
            }
            if !server_side && protocol == PROTOCOL_TLS_SERVER {
                return Err(ssl_error(
                    "Cannot create a client socket with a PROTOCOL_TLS_SERVER context",
                ));
            }
            let requested_session = clone_requested_session(session, self.backend)?;
            Ok(allocate_ssl_socket(
                std::ptr::null_mut(),
                self as *const W_SSLContext as PyObjectRef,
                socket,
                w_none(),
                w_none(),
                owner,
                hostname_obj,
                server_side,
                requested_session,
            ))
        }

        fn _set_alpn_protocols(
            &mut self,
            data: PyObjectRef,
        ) -> Result<(), pyre_interpreter::PyError> {
            let buffer =
                pyre_interpreter::baseobjspace::simple_buffer_bytes(data)?.ok_or_else(|| {
                    pyre_interpreter::PyError::type_error("a bytes-like object is required")
                })?;
            // `PyBuffer_Release` before propagating: a parse failure must not
            // leave a `bytearray` argument permanently exported.
            let parsed = pyre_native::ssl::parse_length_prefixed_alpn(buffer.as_bytes());
            buffer.release();
            let protocols = parsed.map_err(ssl_error)?;
            unsafe { pyre_native::ssl::context_set_alpn(self.backend, protocols) };
            Ok(())
        }

        fn _set_npn_protocols(
            &mut self,
            _data: PyObjectRef,
        ) -> Result<(), pyre_interpreter::PyError> {
            Err(ssl_error("NPN is not supported by rustls"))
        }
    }

    fn validate_tls_version(value: i64) -> Result<i32, pyre_interpreter::PyError> {
        let value = i32::try_from(value).map_err(|_| {
            pyre_interpreter::PyError::value_error(format!("invalid protocol version: {value}"))
        })?;
        if value != -2 && value != -1 && !(0x300..=0x304).contains(&value) {
            return Err(pyre_interpreter::PyError::value_error(format!(
                "invalid protocol version: {value}"
            )));
        }
        Ok(value)
    }
} // context_methods

mod memory_bio_methods {
    use super::*;

    #[pyre_interpreter::pyre_methods(doc = "Memory BIO for TLS protocol data.")]
    impl W_MemoryBIO {
        #[staticmethod]
        fn __new__(
            cls: PyObjectRef,
            args: &[PyObjectRef],
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
            if positional.len() != 1 || pyre_interpreter::builtins::has_real_kwargs(kwargs) {
                return Err(pyre_interpreter::PyError::type_error(
                    "MemoryBIO() takes no arguments",
                ));
            }
            pyre_interpreter::typedef::check_user_subclass(type_object(), cls)?;
            let _roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(cls);
            let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let obj = W_MemoryBIO::allocate_stable(W_MemoryBIO {
                ob: PyObject {
                    ob_type: std::ptr::null(),
                    w_class: std::ptr::null_mut(),
                },
                backend: pyre_native::ssl::memory_bio_new(),
            });
            Ok(pyre_interpreter::typedef::tag_subclass_instance(
                obj,
                pyre_object::gc_roots::shadow_stack_get(cls_slot),
            ))
        }

        fn read(
            &mut self,
            #[default(-1)] size: i64,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let size = if size < 0 {
                unsafe { pyre_native::ssl::memory_bio_pending(self.backend) }
            } else {
                usize::try_from(size).map_err(|_| {
                    pyre_interpreter::PyError::overflow_error("read size out of range")
                })?
            };
            let bytes = unsafe { pyre_native::ssl::memory_bio_read(self.backend, size) };
            Ok(w_bytes_from_bytes(&bytes))
        }

        fn write(&mut self, data: PyObjectRef) -> Result<usize, pyre_interpreter::PyError> {
            let buffer =
                pyre_interpreter::baseobjspace::simple_buffer_bytes(data)?.ok_or_else(|| {
                    pyre_interpreter::PyError::type_error("a bytes-like object is required")
                })?;
            let result =
                unsafe { pyre_native::ssl::memory_bio_write(self.backend, buffer.as_bytes()) }
                    .map_err(ssl_error);
            buffer.release();
            result
        }

        fn write_eof(&mut self) {
            unsafe { pyre_native::ssl::memory_bio_write_eof(self.backend) };
        }

        #[getter]
        fn pending(&self) -> usize {
            unsafe { pyre_native::ssl::memory_bio_pending(self.backend) }
        }

        #[getter]
        fn eof(&self) -> bool {
            unsafe { pyre_native::ssl::memory_bio_eof(self.backend) }
        }
    }
} // memory_bio_methods

mod ssl_socket_methods {
    use super::*;

    fn is_blocking_error(error: &pyre_interpreter::PyError) -> bool {
        let Some(class) = pyre_interpreter::builtins::lookup_exc_class("BlockingIOError") else {
            return false;
        };
        !error.exc_object.is_null()
            && unsafe { pyre_interpreter::baseobjspace::isinstance_w(error.exc_object, class) }
    }

    fn ensure_connection(socket: &mut W_SSLSocket) -> Result<(), pyre_interpreter::PyError> {
        if !socket.backend.is_null() {
            return Ok(());
        }
        let context =
            W_SSLContext::from_obj(socket.context).expect("SSL socket owns a live SSLContext");
        let hostname = if unsafe { is_none(socket.server_hostname) } {
            None
        } else {
            Some(pyre_interpreter::baseobjspace::str_utf8_w(socket.server_hostname)?.to_string())
        };
        socket.backend = native_result(unsafe {
            pyre_native::ssl::connection_new(
                context.backend,
                socket.server_side,
                hostname.as_deref(),
                socket.requested_session,
            )
        })?;
        Ok(())
    }

    fn transport_socket(socket: &W_SSLSocket) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let value =
            unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(socket.socket) };
        if value.is_null() {
            Err(ssl_error("underlying socket has been collected"))
        } else {
            Ok(value)
        }
    }

    fn receive_tls(
        socket: &mut W_SSLSocket,
        data: &[u8],
    ) -> Result<usize, pyre_interpreter::PyError> {
        // `PySSL_BEGIN_ALLOW_THREADS` around `SSL_read`: parsing the records
        // and verifying the peer's certificate is the expensive half of a
        // handshake, and it reads a buffer this module owns rather than
        // anything the interpreter does.  Holding the interpreter across it
        // is what let a server finish rejecting a client certificate — alert
        // and socket close included — inside the client thread's own `send`.
        let outcome = {
            let _blocked = pyre_interpreter::module::thread::before_external_block();
            unsafe { pyre_native::ssl::connection_receive_tls(socket.backend, data) }
        };
        let read = match outcome {
            Ok(read) => read,
            Err(error) => {
                // rustls may have queued the fatal alert describing this
                // protocol error. Send it before unwinding the Python call so
                // the peer cannot remain blocked in its handshake.
                let _ = flush_transport(socket);
                return tls_result(Err(error));
            }
        };
        settle_received_tls(socket)?;
        Ok(read)
    }

    /// What is left to do once rustls has accepted the peer's records: deliver
    /// the protocol events it observed, and keep the trust anchor a lazy CA
    /// directory contributed.  Both need the interpreter, so a released run
    /// leaves them to its caller.
    fn settle_received_tls(socket: &mut W_SSLSocket) -> Result<(), pyre_interpreter::PyError> {
        run_message_callbacks(socket)?;
        if let Some(root) =
            unsafe { pyre_native::ssl::connection_take_verified_root(socket.backend) }
        {
            let context =
                W_SSLContext::from_obj(socket.context).expect("SSL socket owns its context");
            native_result(unsafe {
                pyre_native::ssl::context_add_verified_root(context.backend, root)
            })?;
        }
        Ok(())
    }

    fn run_message_callbacks(socket: &mut W_SSLSocket) -> Result<(), pyre_interpreter::PyError> {
        let events = unsafe { pyre_native::ssl::connection_take_message_events(socket.backend) };
        if events.is_empty() {
            return Ok(());
        }
        let context = W_SSLContext::from_obj(socket.context).expect("SSL socket owns its context");
        if unsafe { is_none(context.msg_callback) } {
            return Ok(());
        }
        let owner = unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(socket.owner) };
        if owner.is_null() {
            return Ok(());
        }
        let _owner_roots = pyre_object::gc_roots::push_roots();
        let owner_live = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(owner);
        for event in events {
            let _event_roots = pyre_object::gc_roots::push_roots();
            let owner_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::gc_roots::shadow_stack_get(
                owner_live,
            ));
            let data_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_bytes_from_bytes(&event.data));
            let direction_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(if event.write {
                "write"
            } else {
                "read"
            }));
            // Ints are non-moving, but minting them can collect and move the
            // direction string, so the call reloads it from the slot.
            let version = w_int_new(event.version as i64);
            let content_type = w_int_new(event.content_type as i64);
            let message_type = w_int_new(event.message_type as i64);
            pyre_interpreter::call::call_function_impl_result(
                context.msg_callback,
                &[
                    pyre_object::gc_roots::shadow_stack_get(owner_slot),
                    pyre_object::gc_roots::shadow_stack_get(direction_slot),
                    version,
                    content_type,
                    message_type,
                    pyre_object::gc_roots::shadow_stack_get(data_slot),
                ],
            )?;
        }
        Ok(())
    }

    fn flush_transport(socket: &mut W_SSLSocket) -> Result<(), pyre_interpreter::PyError> {
        ensure_connection(socket)?;
        if !unsafe { is_none(socket.outgoing) } {
            let data =
                tls_result(unsafe { pyre_native::ssl::connection_take_tls(socket.backend) })?;
            run_message_callbacks(socket)?;
            if !data.is_empty() {
                let outgoing = W_MemoryBIO::from_obj(socket.outgoing)
                    .expect("_wrap_bio validated its outgoing MemoryBIO");
                unsafe { pyre_native::ssl::memory_bio_write(outgoing.backend, &data) }
                    .map_err(ssl_error)?;
            }
            return Ok(());
        }

        // `SSL_set_fd`: the records reach the descriptor from here rather
        // than through the socket's Python `send`, so the whole queue costs
        // one released region instead of one per record.
        if let Some((transport, fd)) = pump_transport(socket)? {
            // The released run lets another thread collect, so the transport
            // is rooted and read back after it.
            let _roots = pyre_object::gc_roots::push_roots();
            let transport_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(transport);
            let transport = || pyre_object::gc_roots::shadow_stack_get(transport_slot);
            let backend = socket.backend;
            loop {
                match pump(backend, fd, &mut [], PumpGoal::Flush, &mut socket.record) {
                    PumpExit::Done(_) => return Ok(()),
                    PumpExit::Interrupted => {
                        pyre_interpreter::module::signal::interp_signal::checksignals_now()?
                    }
                    PumpExit::Failed { write, code }
                        if wait_for_transport(transport(), fd, write, code)? => {}
                    exit => return Err(pump_error(transport(), backend, exit)),
                }
            }
        }

        loop {
            let data =
                tls_result(unsafe { pyre_native::ssl::connection_peek_tls(socket.backend) })?;
            run_message_callbacks(socket)?;
            if data.is_empty() {
                return Ok(());
            }
            // The callbacks above ran Python, so the transport is read back
            // after them; the wait releases the GIL, so it is rooted across it.
            let _roots = pyre_object::gc_roots::push_roots();
            let transport_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(transport_socket(socket)?);
            let transport = || pyre_object::gc_roots::shadow_stack_get(transport_slot);
            let fd = crate::module::_socket::interp_socket::socket_fd(transport())?;
            crate::module::_socket::interp_socket::socket_wait_for_data(transport(), fd, true)?;
            let sent = match crate::module::_socket::interp_socket::socket_send_bytes(
                transport(),
                fd,
                &data,
                0,
            ) {
                Ok(sent) => sent,
                Err(error) if is_blocking_error(&error) => {
                    return Err(tls_error(
                        pyre_native::ssl::TLS_ERROR_WANT_WRITE,
                        "The operation did not complete (write)".to_string(),
                    ));
                }
                Err(error) => return Err(error),
            };
            if sent <= 0 {
                return Err(tls_error(
                    pyre_native::ssl::TLS_ERROR_WANT_WRITE,
                    "The operation did not complete (write)".to_string(),
                ));
            }
            unsafe { pyre_native::ssl::connection_consume_tls(socket.backend, sent as usize) };
        }
    }

    /// What a released run is driving toward.
    enum PumpGoal {
        /// Complete the handshake.
        Handshake,
        /// Produce up to this many bytes of plaintext.
        Read(usize),
        /// Put every record rustls has queued on the wire.
        Flush,
        /// Exchange close_notify records until the peer has answered ours.
        Shutdown,
    }

    /// Why a released run gave the interpreter back.
    enum PumpExit {
        /// The goal is met.  `Read` carries the plaintext it produced; the
        /// other two carry nothing.
        Done(Vec<u8>),
        /// A server has parsed ClientHello and is waiting for
        /// `configure_accepted_server`, which runs Python.
        NeedsConfig,
        /// The transport reported end of file.
        Eof,
        /// A transport call could not proceed. `write` says which half it was,
        /// and `code` is what the socket API reported, or 0 for a send that
        /// accepted nothing.
        Failed { write: bool, code: i32 },
        /// A transport call was interrupted; the caller delivers the signal.
        Interrupted,
        /// rustls rejected the exchange.
        Rejected((i32, String)),
    }

    /// The `PySSL_BEGIN_ALLOW_THREADS` bracket that `SSL_do_handshake`,
    /// `SSL_read` and `SSL_write` each put around their whole exchange: the
    /// peer's records are read, answered and parsed without the interpreter,
    /// so it is released across the exchange rather than taken back between
    /// every record.  A thread that is not asking for the interpreter cannot
    /// land in the middle of another one's `send`, which is what lets a peer's
    /// fatal alert reach a client that is already parked in `recv`.
    ///
    /// Nothing here may touch a Python object.  The states that need one - a
    /// server context to choose, a signal to deliver, an error to raise - end
    /// the run and are answered by the caller.
    ///
    /// `record` is the connection's cursor into the record being read, so it
    /// outlives the call: a transport that returns short unwinds to the
    /// caller, which re-enters here and has to resume where it stopped.
    fn pump(
        backend: *mut pyre_native::ssl::TlsConnection,
        fd: pyre_interpreter::rsocket_rffi::Socket,
        buf: &mut [u8],
        goal: PumpGoal,
        record: &mut RecordCursor,
    ) -> PumpExit {
        use crate::module::_socket::interp_socket::{socket_recv_raw, socket_send_raw};
        use pyre_interpreter::rsocket_rffi::error_is_interrupted;

        let _blocked = pyre_interpreter::module::thread::before_external_block();
        // The shutdown exchange stops at a record boundary.  What follows the
        // peer's close_notify is the plaintext `unwrap()` hands back with the
        // socket, so a read that crossed the boundary would lift those bytes
        // out of the kernel and drop them into the TLS stream, where nothing
        // can return them.  OpenSSL's record layer reads a header and then the
        // body its length names, which is why a socket-backed `SSL_shutdown`
        // leaves the caller's bytes alone; track the same two-step here.  A
        // short read is fed on as it arrives -- the receiver buffers a partial
        // record -- and only the byte count is carried across.  `record` is the
        // connection's, because a transport retry re-enters this function.
        loop {
            loop {
                let data = match unsafe { pyre_native::ssl::connection_peek_tls(backend) } {
                    Ok(data) => data,
                    Err(error) => return PumpExit::Rejected(error),
                };
                if data.is_empty() {
                    break;
                }
                match socket_send_raw(fd, &data, 0) {
                    Ok(sent) if sent > 0 => unsafe {
                        pyre_native::ssl::connection_consume_tls(backend, sent as usize)
                    },
                    Ok(_) => {
                        return PumpExit::Failed {
                            write: true,
                            code: 0,
                        };
                    }
                    Err(code) if error_is_interrupted(code) => return PumpExit::Interrupted,
                    Err(code) => return PumpExit::Failed { write: true, code },
                }
            }
            match goal {
                PumpGoal::Flush => return PumpExit::Done(Vec::new()),
                PumpGoal::Handshake => {
                    if !unsafe { pyre_native::ssl::connection_is_handshaking(backend) } {
                        return PumpExit::Done(Vec::new());
                    }
                    if unsafe { pyre_native::ssl::connection_waiting_for_server_config(backend) } {
                        return PumpExit::NeedsConfig;
                    }
                }
                PumpGoal::Read(size) => {
                    match unsafe { pyre_native::ssl::connection_read_plain(backend, size) } {
                        Ok(data) => return PumpExit::Done(data),
                        Err((code, _)) if code == pyre_native::ssl::TLS_ERROR_WANT_READ => {}
                        Err(error) => return PumpExit::Rejected(error),
                    }
                }
                PumpGoal::Shutdown => {
                    if unsafe { pyre_native::ssl::connection_peer_closed(backend) } {
                        return PumpExit::Done(Vec::new());
                    }
                }
            }
            // Every goal, not only `Shutdown`: a read that stops at the
            // boundary is what keeps the cursor describing the transport, and
            // it is also what leaves a later close_notify's trailing plaintext
            // where `unwrap()` can still hand it over.
            let want = record.want().min(buf.len());
            match socket_recv_raw(fd, &mut buf[..want], 0) {
                Ok(0) => return PumpExit::Eof,
                Ok(read) => {
                    record.consume(&buf[..read]);
                    if let Err(error) =
                        unsafe { pyre_native::ssl::connection_receive_tls(backend, &buf[..read]) }
                    {
                        return PumpExit::Rejected(error);
                    }
                }
                Err(code) if error_is_interrupted(code) => return PumpExit::Interrupted,
                Err(code) => return PumpExit::Failed { write: false, code },
            }
        }
    }

    /// `_ssl_select`: a transport call that could not proceed waits for the
    /// half it needs, and the exchange resumes where it stopped.  Only a
    /// socket carrying a positive timeout has anything to wait for — a
    /// blocking one never reports would-block, and a non-blocking one owes its
    /// caller the `WANT_READ`/`WANT_WRITE` it is asking for — so this is also
    /// where an exchange that outlasts its socket's timeout is cut off, with
    /// the `TimeoutError` `wait_for_data` raises.
    fn wait_for_transport(
        transport: PyObjectRef,
        fd: pyre_interpreter::rsocket_rffi::Socket,
        write: bool,
        code: i32,
    ) -> Result<bool, pyre_interpreter::PyError> {
        if !pyre_interpreter::rsocket_rffi::error_is_would_block(code) {
            return Ok(false);
        }
        crate::module::_socket::interp_socket::socket_wait_for_data(transport, fd, write)
    }

    /// The exception a released run's transport failure raises, once the
    /// interpreter is back.  Every caller answers the exits that are not
    /// failures - and `Rejected`, whose alert has to go out first - before
    /// reaching here.
    fn pump_error(
        transport: PyObjectRef,
        #[cfg_attr(not(windows), allow(unused_variables))]
        backend: *const pyre_native::ssl::TlsConnection,
        exit: PumpExit,
    ) -> pyre_interpreter::PyError {
        match exit {
            PumpExit::Eof => tls_error(
                pyre_native::ssl::TLS_ERROR_EOF,
                "EOF occurred in violation of protocol".to_string(),
            ),
            PumpExit::Failed { write, code } => {
                #[cfg(all(windows, feature = "host_env"))]
                const WSAECONNABORTED: i32 = rustpython_host_env::socket::WSAECONNABORTED;
                #[cfg(all(windows, not(feature = "host_env")))]
                const WSAECONNABORTED: i32 =
                    windows_sys::Win32::Networking::WinSock::WSAECONNABORTED;
                #[cfg(all(windows, feature = "host_env"))]
                const WSAECONNRESET: i32 = rustpython_host_env::socket::WSAECONNRESET;
                #[cfg(all(windows, not(feature = "host_env")))]
                const WSAECONNRESET: i32 = windows_sys::Win32::Networking::WinSock::WSAECONNRESET;
                #[cfg(windows)]
                if !write
                    && !unsafe { pyre_native::ssl::connection_is_handshaking(backend) }
                    && matches!(code, WSAECONNABORTED | WSAECONNRESET)
                {
                    // [3.14-spec] `ThreadedTests.test_wrong_cert_tls13`
                    // accepts a protocol EOF when the server's fatal alert
                    // races its close.  PyPy `_cffi_ssl.error.pyssl_error`
                    // normally preserves a WinSock error, but rustls rejects
                    // the client certificate early enough that `closesocket`
                    // can replace the queued alert with WSAECONNABORTED or
                    // WSAECONNRESET when the client's first application
                    // record is already unread.  Once the handshake has
                    // completed, that abrupt transport loss is the same
                    // missing-close-notify protocol EOF.  A reset during the
                    // handshake is not rewritten: CPython
                    // `TestPreHandshakeClose` relies on that transport error
                    // to distinguish pre-authentication data, and raw socket
                    // operations likewise continue to expose 10053/10054.
                    return tls_error(
                        pyre_native::ssl::TLS_ERROR_EOF,
                        "EOF occurred in violation of protocol".to_string(),
                    );
                }
                if code != 0 {
                    let error = crate::module::_socket::interp_socket::socket_error_for_operation(
                        transport, code,
                    );
                    if !is_blocking_error(&error) {
                        return error;
                    }
                }
                let (want, message) = if write {
                    (
                        pyre_native::ssl::TLS_ERROR_WANT_WRITE,
                        "The operation did not complete (write)",
                    )
                } else {
                    (
                        pyre_native::ssl::TLS_ERROR_WANT_READ,
                        "The operation did not complete (read)",
                    )
                };
                tls_error(want, message.to_string())
            }
            PumpExit::Rejected((code, message)) => tls_error(code, message),
            PumpExit::Done(_) | PumpExit::NeedsConfig | PumpExit::Interrupted => {
                unreachable!("the caller answers every exit that is not a transport failure")
            }
        }
    }

    /// The transport a released run drives, or `None` for a connection that
    /// has to reach Python between records: a `MemoryBIO` pair has no
    /// descriptor behind it, and a registered `_msg_callback` is called for
    /// every record as it is seen.
    fn pump_transport(
        socket: &W_SSLSocket,
    ) -> Result<
        Option<(PyObjectRef, pyre_interpreter::rsocket_rffi::Socket)>,
        pyre_interpreter::PyError,
    > {
        if !unsafe { is_none(socket.incoming) } || !unsafe { is_none(socket.outgoing) } {
            return Ok(None);
        }
        let context = W_SSLContext::from_obj(socket.context).expect("SSL socket owns its context");
        if !unsafe { is_none(context.msg_callback) } {
            return Ok(None);
        }
        let fd = crate::module::_socket::interp_socket::socket_fd(transport_socket(socket)?)?;
        // Reading the descriptor can materialise the socket's dict, so the
        // transport is read back after it.
        Ok(Some((transport_socket(socket)?, fd)))
    }

    /// The callback-aware counterpart of [`pump`], which returns to the
    /// interpreter after every record instead of holding it across the
    /// exchange.  It stops at the same boundary, for the same reason.
    fn receive_transport(socket: &mut W_SSLSocket) -> Result<(), pyre_interpreter::PyError> {
        ensure_connection(socket)?;
        if !unsafe { is_none(socket.incoming) } {
            let incoming = W_MemoryBIO::from_obj(socket.incoming)
                .expect("_wrap_bio validated its incoming MemoryBIO");
            let pending = unsafe { pyre_native::ssl::memory_bio_pending(incoming.backend) };
            if pending == 0 {
                if unsafe { pyre_native::ssl::memory_bio_eof(incoming.backend) } {
                    let _ = receive_tls(socket, &[]);
                    return Err(tls_error(
                        pyre_native::ssl::TLS_ERROR_EOF,
                        "EOF occurred in violation of protocol".to_string(),
                    ));
                }
                return Err(tls_error(
                    pyre_native::ssl::TLS_ERROR_WANT_READ,
                    "The operation did not complete (read)".to_string(),
                ));
            }
            let data = unsafe { pyre_native::ssl::memory_bio_read(incoming.backend, pending) };
            receive_tls(socket, &data)?;
            return Ok(());
        }

        // The wait releases the GIL, so the transport is rooted across it.
        let _roots = pyre_object::gc_roots::push_roots();
        let transport_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(transport_socket(socket)?);
        let transport = || pyre_object::gc_roots::shadow_stack_get(transport_slot);
        let fd = crate::module::_socket::interp_socket::socket_fd(transport())?;
        crate::module::_socket::interp_socket::socket_wait_for_data(transport(), fd, false)?;
        let mut buf = vec![0u8; 32 * 1024];
        let want = socket.record.want().min(buf.len());
        let read = match crate::module::_socket::interp_socket::socket_recv_bytes(
            transport(),
            fd,
            &mut buf[..want],
            0,
        ) {
            Ok(read) => read,
            Err(error) if is_blocking_error(&error) => {
                return Err(tls_error(
                    pyre_native::ssl::TLS_ERROR_WANT_READ,
                    "The operation did not complete (read)".to_string(),
                ));
            }
            Err(error) => return Err(error),
        };
        if read == 0 {
            return Err(tls_error(
                pyre_native::ssl::TLS_ERROR_EOF,
                "EOF occurred in violation of protocol".to_string(),
            ));
        }
        socket.record.consume(&buf[..read]);
        receive_tls(socket, &buf[..read]).map(|_| ())
    }

    fn reject_sni(
        socket: &mut W_SSLSocket,
        alert: u8,
        reason: &str,
    ) -> Result<(), pyre_interpreter::PyError> {
        tls_result(unsafe { pyre_native::ssl::connection_reject_server(socket.backend, alert) })?;
        // The peer must receive the fatal alert before the server-side
        // operation reports its callback failure.
        flush_transport(socket)?;
        Err(tls_error(
            pyre_native::ssl::TLS_ERROR_SSL,
            format!("[SSL: {reason}] SNI callback failed"),
        ))
    }

    /// Finish a server configuration after rustls has parsed ClientHello.
    ///
    /// This must operate through a raw pointer: the Python SNI callback may
    /// re-enter `_SSLSocket.context` and mutate this exact object. Keeping a
    /// Rust `&mut W_SSLSocket` live across that call would violate its unique
    /// alias and lets optimized builds reuse the pre-callback context. PyPy's
    /// `servername_callback` likewise reloads the SSL object's context after
    /// the app-level callback returns.
    #[inline(never)]
    unsafe fn configure_accepted_server(
        socket: *mut W_SSLSocket,
    ) -> Result<(), pyre_interpreter::PyError> {
        let backend = unsafe { (*socket).backend };
        if !unsafe { (*socket).server_side }
            || !unsafe { pyre_native::ssl::connection_waiting_for_server_config(backend) }
        {
            return Ok(());
        }

        let initial_context = unsafe { (*socket).context };
        let context = W_SSLContext::from_obj(initial_context)
            .expect("SSL socket owns a live initial context");
        let callback = context.sni_callback;
        if !unsafe { is_none(callback) } {
            let owner =
                unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref((*socket).owner) };
            if owner.is_null() {
                return Err(tls_error(
                    pyre_native::ssl::TLS_ERROR_SSL,
                    "[SSL: PARSE_TLSEXT] SNI callback owner is no longer available".to_string(),
                ));
            }
            let _sni_roots = pyre_object::gc_roots::push_roots();
            let owner_slot = pyre_object::gc_roots::pin_roots(&[owner, initial_context, callback]);
            let context_slot = owner_slot + 1;
            let callback_slot = owner_slot + 2;
            let name_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(unsafe {
                pyre_native::ssl::connection_server_name(backend)
                    .map(|name| w_str_new_managed(&name))
                    .unwrap_or_else(w_none)
            });
            let result = match pyre_interpreter::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(callback_slot),
                &[
                    pyre_object::gc_roots::shadow_stack_get(owner_slot),
                    pyre_object::gc_roots::shadow_stack_get(name_slot),
                    pyre_object::gc_roots::shadow_stack_get(context_slot),
                ],
            ) {
                Ok(result) => result,
                Err(mut error) => {
                    error.write_unraisable(
                        w_none(),
                        rustpython_wtf8::Wtf8::new("in SNI callback"),
                        pyre_object::gc_roots::shadow_stack_get(callback_slot),
                    );
                    return reject_sni(unsafe { &mut *socket }, 40, "CALLBACK_FAILED");
                }
            };
            if !unsafe { is_none(result) } {
                let alert = match pyre_interpreter::baseobjspace::int_w(result) {
                    Ok(alert) if (0..=u8::MAX as i64).contains(&alert) => alert as u8,
                    _ => {
                        let mut error = pyre_interpreter::PyError::type_error(
                            "SNI callback must return None or a TLS alert integer",
                        );
                        error.write_unraisable(
                            w_none(),
                            rustpython_wtf8::Wtf8::new("in SNI callback"),
                            pyre_object::gc_roots::shadow_stack_get(callback_slot),
                        );
                        return reject_sni(unsafe { &mut *socket }, 80, "CALLBACK_FAILED");
                    }
                };
                return reject_sni(unsafe { &mut *socket }, alert, "CALLBACK_FAILED");
            }
        }

        // The callback is allowed to assign ssl_sock.context. Reload through
        // the raw object pointer after re-entry; `initial_context` is only the
        // third callback argument, never the selected configuration anchor.
        let selected = W_SSLContext::from_obj(unsafe { (*socket).context })
            .expect("SNI callback preserves an SSLContext on the socket");
        match unsafe { pyre_native::ssl::connection_accept_server(backend, selected.backend) } {
            Ok(()) => Ok(()),
            Err(error) => {
                // Accepted::into_connection returns the alert bytes alongside
                // the error. Deliver those bytes before reporting failure.
                let _ = flush_transport(unsafe { &mut *socket });
                tls_result(Err(error))
            }
        }
    }

    #[pyre_interpreter::pyre_methods]
    impl W_SSLSocket {
        #[staticmethod]
        fn __new__(_cls: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            Err(pyre_interpreter::PyError::type_error(
                "cannot create '_ssl._SSLSocket' instances",
            ))
        }

        #[getter]
        fn owner(&self) -> PyObjectRef {
            let owner =
                unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(self.owner) };
            if owner.is_null() { w_none() } else { owner }
        }

        #[getter]
        fn server_side(&self) -> bool {
            self.server_side
        }

        #[getter]
        fn server_hostname(&self) -> PyObjectRef {
            self.server_hostname
        }

        #[getter]
        fn context(&self) -> PyObjectRef {
            self.context
        }

        #[setter]
        fn set_context(&mut self, value: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            if W_SSLContext::from_obj(value).is_none() {
                return Err(pyre_interpreter::PyError::type_error(
                    "context must be an SSLContext",
                ));
            }
            pyre_object::gc_hook::try_gc_write_barrier(
                self as *mut W_SSLSocket as pyre_object::gc_hook::GCREF,
            );
            self.context = value;
            Ok(())
        }

        #[getter]
        fn session_reused(&self) -> PyObjectRef {
            if self.backend.is_null() {
                return w_none();
            }
            unsafe { pyre_native::ssl::connection_session_reused(self.backend) }
                .map(w_bool_from)
                .unwrap_or_else(w_none)
        }

        #[getter]
        fn session(&self) -> PyObjectRef {
            if self.server_side || self.backend.is_null() {
                return w_none();
            }
            let backend = unsafe { pyre_native::ssl::connection_session(self.backend) };
            if backend.is_null() {
                w_none()
            } else {
                allocate_ssl_session(backend)
            }
        }

        #[setter]
        fn set_session(&mut self, value: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            if !unsafe { is_none(value) } && W_SSLSession::from_obj(value).is_none() {
                return Err(pyre_interpreter::PyError::type_error(
                    "Value is not a SSLSession.",
                ));
            }
            if !self.backend.is_null()
                && !unsafe { pyre_native::ssl::connection_is_handshaking(self.backend) }
            {
                return Err(pyre_interpreter::PyError::value_error(
                    "Cannot set session after handshake.",
                ));
            }
            let replacement = clone_requested_session(Some(value), {
                let context =
                    W_SSLContext::from_obj(self.context).expect("SSL socket owns its context");
                context.backend
            })?;
            unsafe { pyre_native::ssl::session_free(self.requested_session) };
            self.requested_session = replacement;
            Ok(())
        }

        fn do_handshake(&mut self) -> Result<(), pyre_interpreter::PyError> {
            ensure_connection(self)?;
            if self.server_side {
                let context =
                    W_SSLContext::from_obj(self.context).expect("SSL socket owns its context");
                let owner =
                    unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(self.owner) };
                if !unsafe { is_none(context.sni_callback) } && owner.is_null() {
                    return Err(tls_error(
                        pyre_native::ssl::TLS_ERROR_SSL,
                        "[SSL: PARSE_TLSEXT] SNI callback owner is no longer available".to_string(),
                    ));
                }
            }
            let mut pump_buffer = Vec::new();
            loop {
                unsafe { configure_accepted_server(self as *mut W_SSLSocket) }?;
                let handshaking =
                    unsafe { pyre_native::ssl::connection_is_handshaking(self.backend) };
                if let Some((transport, fd)) = if handshaking {
                    pump_transport(self)?
                } else {
                    None
                } {
                    // The released run lets another thread collect, so the
                    // transport is rooted and read back after it.
                    let _roots = pyre_object::gc_roots::push_roots();
                    let transport_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(transport);
                    let transport = || pyre_object::gc_roots::shadow_stack_get(transport_slot);
                    if pump_buffer.is_empty() {
                        pump_buffer.resize(32 * 1024, 0u8);
                    }
                    let backend = self.backend;
                    match pump(
                        backend,
                        fd,
                        &mut pump_buffer,
                        PumpGoal::Handshake,
                        &mut self.record,
                    ) {
                        PumpExit::Done(_) => {
                            settle_received_tls(self)?;
                            return Ok(());
                        }
                        PumpExit::NeedsConfig => {
                            settle_received_tls(self)?;
                            continue;
                        }
                        PumpExit::Interrupted => {
                            pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
                            continue;
                        }
                        PumpExit::Failed { write, code }
                            if wait_for_transport(transport(), fd, write, code)? =>
                        {
                            continue;
                        }
                        PumpExit::Rejected(error) => {
                            // rustls may have queued the fatal alert describing
                            // this protocol error. Send it before unwinding so
                            // the peer cannot remain blocked in its handshake.
                            let _ = flush_transport(self);
                            return tls_result(Err(error));
                        }
                        exit => return Err(pump_error(transport(), self.backend, exit)),
                    }
                }
                flush_transport(self)?;
                if !unsafe { pyre_native::ssl::connection_is_handshaking(self.backend) } {
                    return Ok(());
                }
                receive_transport(self)?;
                unsafe { configure_accepted_server(self as *mut W_SSLSocket) }?;
                if !unsafe { is_none(self.incoming) } {
                    flush_transport(self)?;
                    if unsafe { pyre_native::ssl::connection_is_handshaking(self.backend) } {
                        return Err(tls_error(
                            pyre_native::ssl::TLS_ERROR_WANT_READ,
                            "The operation did not complete (read)".to_string(),
                        ));
                    }
                    return Ok(());
                }
            }
        }

        fn write(&mut self, mut data: PyObjectRef) -> Result<usize, pyre_interpreter::PyError> {
            if self.shutdown_started {
                return Err(ssl_error("TLS/SSL connection has been closed"));
            }
            pyre_object::with_roots!(data => ensure_connection(self))?;
            let buffer =
                pyre_interpreter::baseobjspace::simple_buffer_bytes(data)?.ok_or_else(|| {
                    pyre_interpreter::PyError::type_error("a bytes-like object is required")
                })?;
            let written_or_err = tls_result(unsafe {
                pyre_native::ssl::connection_write_plain(self.backend, buffer.as_bytes())
            });
            let mut failure = None;
            let written = match written_or_err {
                Ok(n) => Some(n),
                Err(err) => {
                    failure = Some(err);
                    None
                }
            };
            if let Some(err) = failure.take() {
                let _roots = pyre_object::gc_roots::push_roots();
                let mut err = err;
                let slot = err.pin(&_roots);
                buffer.release();
                err.reload(&_roots, slot);
                return Err(err);
            }
            buffer.release();
            let written = written.unwrap();
            flush_transport(self)?;
            Ok(written)
        }

        fn read(&mut self, args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            // `args` is the gateway's native copy; a collection in
            // `ensure_connection` or `int_w` does not forward it.
            let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
            let mut w_kwargs = kwargs.unwrap_or(PY_NULL);
            let mut w_size = positional.get(1).copied().unwrap_or(PY_NULL);
            let mut w_buffer = positional.get(2).copied().unwrap_or(PY_NULL);
            pyre_object::with_roots!(w_kwargs, w_size, w_buffer => ensure_connection(self))?;
            let kwargs = (!w_kwargs.is_null()).then_some(w_kwargs);
            if pyre_interpreter::builtins::has_real_kwargs(kwargs) || positional.len() > 3 {
                return Err(pyre_interpreter::PyError::type_error(
                    "read() takes at most 2 arguments",
                ));
            }
            let requested = if w_size.is_null() {
                1024
            } else {
                pyre_object::with_roots!(w_buffer => pyre_interpreter::baseobjspace::int_w(w_size))?
            };
            let output_buffer = (!w_buffer.is_null()).then_some(w_buffer);
            if requested < 0 && output_buffer.is_none() {
                return Err(pyre_interpreter::PyError::value_error(
                    "size should not be negative",
                ));
            }
            if requested == 0 && output_buffer.is_none() {
                return Ok(w_bytes_from_bytes(&[]));
            }
            let mut writable = if let Some(buffer) = output_buffer {
                Some(
                    unsafe { pyre_interpreter::builtins::WritableBuffer::acquire(buffer) }
                        .map_err(|_| {
                            pyre_interpreter::PyError::type_error(
                                "read() argument 2 must be a writable buffer",
                            )
                        })?,
                )
            } else {
                None
            };
            let size = if let Some(buffer) = writable.as_mut() {
                let capacity = unsafe { buffer.as_mut_slice() }.len();
                // `_read_buf`: a length that is not positive, or larger than
                // the buffer, is the buffer's own length.
                if requested <= 0 || requested as usize > capacity {
                    capacity
                } else {
                    requested as usize
                }
            } else {
                requested as usize
            };
            // `_read_buf` answers a zero-length buffer with 0 before reaching
            // `SSL_read`, so an empty request never observes the connection --
            // a plaintext read of no bytes would otherwise report the peer's
            // EOF and, after a local shutdown, raise `SSLZeroReturnError`.
            if size == 0 {
                return Ok(w_int_new(0));
            }
            let data = if let Some((transport, fd)) = pump_transport(self)? {
                // `SSL_read` under one `PySSL_BEGIN_ALLOW_THREADS`: the
                // plaintext drain, the transport read and the record parsing
                // all run in the released region, so a caller waiting for the
                // peer is parked in `recv` rather than holding the
                // interpreter between them.  The released run lets another
                // thread collect, so the transport is rooted and read back
                // after it.
                let _roots = pyre_object::gc_roots::push_roots();
                let transport_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(transport);
                let transport = || pyre_object::gc_roots::shadow_stack_get(transport_slot);
                let mut buf = vec![0u8; 32 * 1024];
                let backend = self.backend;
                loop {
                    match pump(
                        backend,
                        fd,
                        &mut buf,
                        PumpGoal::Read(size),
                        &mut self.record,
                    ) {
                        PumpExit::Done(data) => {
                            settle_received_tls(self)?;
                            break data;
                        }
                        PumpExit::NeedsConfig => {
                            // The events the acceptor observed belong to the
                            // context that was current while it read them, so
                            // they are delivered before the callback below is
                            // given the chance to choose another one.
                            settle_received_tls(self)?;
                            unsafe { configure_accepted_server(self as *mut W_SSLSocket) }?
                        }
                        PumpExit::Interrupted => {
                            pyre_interpreter::module::signal::interp_signal::checksignals_now()?
                        }
                        PumpExit::Failed { write, code }
                            if wait_for_transport(transport(), fd, write, code)? => {}
                        PumpExit::Rejected(error) => {
                            // rustls may have queued the fatal alert describing
                            // this protocol error. Send it before unwinding so
                            // the peer cannot remain blocked.
                            let _ = flush_transport(self);
                            return tls_result(Err(error));
                        }
                        exit => return Err(pump_error(transport(), self.backend, exit)),
                    }
                }
            } else {
                loop {
                    match unsafe { pyre_native::ssl::connection_read_plain(self.backend, size) } {
                        Ok(data) => break data,
                        Err((code, _)) if code == pyre_native::ssl::TLS_ERROR_WANT_READ => {
                            receive_transport(self)?;
                            flush_transport(self)?;
                        }
                        Err((code, message)) => return Err(tls_error(code, message)),
                    }
                }
            };
            // PyPy `_cffi_ssl._stdssl._SSLSocket._read` distinguishes the
            // two clean-close states through `SSL_get_shutdown`: a peer-only
            // `SSL_RECEIVED_SHUTDOWN` is stream EOF and returns zero bytes,
            // while a read after our own `shutdown()` has also sent
            // close_notify sees `SSL_SENT_SHUTDOWN | SSL_RECEIVED_SHUTDOWN`
            // and falls through to `pyssl_error`, which raises
            // `SSLZeroReturnError`.  rustls reports both as `Ok(0)`, so retain
            // the same distinction with the object-owned shutdown flag.
            if data.is_empty() && self.shutdown_started {
                return Err(tls_error(
                    pyre_native::ssl::TLS_ERROR_ZERO_RETURN,
                    "TLS/SSL connection has been closed (EOF)".to_string(),
                ));
            }
            if let Some(buffer) = writable.as_mut() {
                let target = unsafe { buffer.as_mut_slice() };
                target[..data.len()].copy_from_slice(&data);
                Ok(w_int_new(data.len() as i64))
            } else {
                Ok(w_bytes_from_bytes(&data))
            }
        }

        fn pending(&mut self) -> usize {
            if self.backend.is_null() {
                0
            } else {
                unsafe { pyre_native::ssl::connection_pending_plaintext(self.backend) }
            }
        }

        fn selected_alpn_protocol(&self) -> PyObjectRef {
            if self.backend.is_null() {
                return w_none();
            }
            unsafe { pyre_native::ssl::connection_alpn(self.backend) }
                .map(|value| w_str_new_managed(&String::from_utf8_lossy(&value)))
                .unwrap_or_else(w_none)
        }

        fn version(&self) -> PyObjectRef {
            if self.backend.is_null() {
                return w_none();
            }
            unsafe { pyre_native::ssl::connection_version(self.backend) }
                .map(w_str_new_managed)
                .unwrap_or_else(w_none)
        }

        fn compression(&self) -> PyObjectRef {
            w_none()
        }

        fn shared_ciphers(&self) -> PyObjectRef {
            if !self.server_side || self.backend.is_null() {
                return w_none();
            }
            let Some((name, bits)) = (unsafe { pyre_native::ssl::connection_cipher(self.backend) })
            else {
                return w_none();
            };
            let version =
                unsafe { pyre_native::ssl::connection_version(self.backend) }.unwrap_or("unknown");
            let mut fields = pyre_object::gc_roots::RootedItems::new();
            fields.push(w_str_new_managed(&name));
            fields.push(w_str_new_managed(version));
            fields.push(w_int_new(bits as i64));
            w_list_new(vec![w_tuple_new(fields.take())])
        }

        fn cipher(&self) -> PyObjectRef {
            if self.backend.is_null() {
                return w_none();
            }
            let Some(version) = (unsafe { pyre_native::ssl::connection_version(self.backend) })
            else {
                return w_none();
            };
            let Some((name, bits)) = (unsafe { pyre_native::ssl::connection_cipher(self.backend) })
            else {
                return w_none();
            };
            let mut fields = pyre_object::gc_roots::RootedItems::new();
            fields.push(w_str_new_managed(&name));
            fields.push(w_str_new_managed(version));
            fields.push(w_int_new(bits as i64));
            w_tuple_new(fields.take())
        }

        fn getpeercert(
            &self,
            #[default(false)] binary_form: bool,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            if self.backend.is_null()
                || unsafe { pyre_native::ssl::connection_is_handshaking(self.backend) }
            {
                return Err(pyre_interpreter::PyError::value_error(
                    "handshake not done yet",
                ));
            }
            let Some(der) =
                (unsafe { pyre_native::ssl::connection_peer_certificate(self.backend) })
            else {
                return Ok(w_none());
            };
            if binary_form {
                return Ok(w_bytes_from_bytes(&der));
            }
            let context =
                W_SSLContext::from_obj(self.context).expect("SSL socket owns its context");
            if unsafe { pyre_native::ssl::context_verify_mode(context.backend) } == CERT_NONE {
                return Ok(w_dict_new());
            }
            let decoded = native_result(pyre_native::ssl::certificate_decode_der(&der))?;
            Ok(decoded_certificate_dict(decoded))
        }

        /// PyPy `_cffi_ssl/_stdssl/__init__.py:get_unverified_chain`: expose
        /// one `_ssl.Certificate` per certificate sent by the peer.  Python
        /// 3.14's public `ssl` layer converts these objects to DER bytes for
        /// `SSLSocket`/`SSLObject`, which is the API pip's truststore consumes.
        fn get_unverified_chain(&self) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            if self.backend.is_null()
                || unsafe { pyre_native::ssl::connection_is_handshaking(self.backend) }
            {
                return Err(pyre_interpreter::PyError::value_error(
                    "handshake not done yet",
                ));
            }
            let chain = unsafe { pyre_native::ssl::connection_peer_certificates(self.backend) };
            if chain.is_empty() {
                return Ok(w_none());
            }
            // Each certificate object is freshly allocated and the next one
            // allocates again, so they are pinned as they arrive.
            let mut items = pyre_object::gc_roots::RootedItems::new();
            for der in chain {
                items.push(allocate_certificate(der));
            }
            Ok(w_list_new(items.take()))
        }

        /// PyPy `_cffi_ssl/_stdssl/__init__.py:get_verified_chain`: return the
        /// path selected by certificate verification, including its trust
        /// anchor even when the peer did not transmit that anchor.
        fn get_verified_chain(&mut self) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            if self.backend.is_null()
                || unsafe { pyre_native::ssl::connection_is_handshaking(self.backend) }
            {
                return Err(pyre_interpreter::PyError::value_error(
                    "handshake not done yet",
                ));
            }
            let chain = unsafe { pyre_native::ssl::connection_verified_certificates(self.backend) };
            if chain.is_empty() {
                return Ok(w_none());
            }
            // Each certificate object is freshly allocated and the next one
            // allocates again, so they are pinned as they arrive.
            let mut items = pyre_object::gc_roots::RootedItems::new();
            for der in chain {
                items.push(allocate_certificate(der));
            }
            Ok(w_list_new(items.take()))
        }

        fn get_channel_binding(
            &mut self,
            #[default("tls-unique")] kind: &str,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            if kind != "tls-unique" {
                return Err(pyre_interpreter::PyError::value_error(
                    "'channel_binding_type' must be 'tls-unique'",
                ));
            }
            // CPython permits querying a lazily-created SSLObject before its
            // first handshake step.  There is no TLS channel to bind yet.
            if self.backend.is_null()
                || unsafe { pyre_native::ssl::connection_is_handshaking(self.backend) }
            {
                return Ok(w_none());
            }
            let Some(binding) = (unsafe { pyre_native::ssl::connection_tls_unique(self.backend) })
            else {
                return Ok(w_none());
            };
            Ok(w_bytes_from_bytes(&binding))
        }

        fn shutdown(&mut self) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            ensure_connection(self)?;
            if !self.shutdown_started {
                unsafe { pyre_native::ssl::connection_send_close_notify(self.backend) };
                self.shutdown_started = true;
            }
            flush_transport(self)?;
            // A `MemoryBIO` pair has no descriptor to wait on, so
            // `_ssl_select` answers `SOCKET_IS_NONBLOCKING` for it and
            // `shutdown` leaves the loop with the error `SSL_shutdown` last
            // reported.  Until the peer's close-notify has been fed in, that
            // is `WANT_READ`.
            if !unsafe { is_none(self.incoming) } {
                // PyPy `_cffi_ssl._stdssl._SSLSocket.shutdown` calls
                // `SSL_shutdown` before `_ssl_select`: once OpenSSL has
                // latched `SSL_RECEIVED_SHUTDOWN`, it succeeds without
                // consulting an empty MemoryBIO.  Check rustls' equivalent
                // sticky state before asking the transport for another
                // record, or a second `unwrap()` incorrectly reports
                // `SSLWantReadError` after `read()` saw close_notify.
                if unsafe { pyre_native::ssl::connection_peer_closed(self.backend) } {
                    return Ok(w_none());
                }
                receive_transport(self)?;
                if !unsafe { pyre_native::ssl::connection_peer_closed(self.backend) } {
                    return Err(tls_error(
                        pyre_native::ssl::TLS_ERROR_WANT_READ,
                        "The operation did not complete (read)".to_string(),
                    ));
                }
            }
            // PyPy `_cffi_ssl._stdssl._SSLSocket.shutdown` loops
            // `SSL_shutdown` and `_ssl_select` until the peer's close_notify
            // arrives.  A socket-backed connection must do the same: merely
            // flushing our close_notify lets the caller close the TCP socket
            // before the peer can send its answer.  Drive this through the
            // same released transport loop as SSL_read/SSL_do_handshake.
            if let Some((transport, fd)) = pump_transport(self)? {
                // The released run lets another thread collect, so the
                // transport is rooted and read back after it.
                let _roots = pyre_object::gc_roots::push_roots();
                let transport_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(transport);
                let transport = || pyre_object::gc_roots::shadow_stack_get(transport_slot);
                let mut buf = vec![0u8; 32 * 1024];
                loop {
                    match pump(
                        self.backend,
                        fd,
                        &mut buf,
                        PumpGoal::Shutdown,
                        &mut self.record,
                    ) {
                        PumpExit::Done(_) => {
                            settle_received_tls(self)?;
                            break;
                        }
                        PumpExit::Interrupted => {
                            pyre_interpreter::module::signal::interp_signal::checksignals_now()?
                        }
                        PumpExit::Failed { write, code }
                            if wait_for_transport(transport(), fd, write, code)? => {}
                        PumpExit::Rejected(error) => {
                            let _ = flush_transport(self);
                            return tls_result(Err(error));
                        }
                        exit => return Err(pump_error(transport(), self.backend, exit)),
                    }
                }
            } else if unsafe { is_none(self.incoming) } {
                // A message callback requires returning to the interpreter
                // after every record.  Keep PyPy's shutdown loop in that
                // case too, but use the callback-aware transport helpers
                // instead of the released pump -- which reads to the same
                // record boundary.
                while !unsafe { pyre_native::ssl::connection_peer_closed(self.backend) } {
                    receive_transport(self)?;
                    flush_transport(self)?;
                }
            }
            if unsafe { is_none(self.socket) } {
                Ok(w_none())
            } else {
                transport_socket(self)
            }
        }
    }
} // ssl_socket_methods

mod ssl_session_methods {
    use super::*;

    #[pyre_interpreter::pyre_methods]
    impl W_SSLSession {
        #[staticmethod]
        fn __new__(_cls: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            Err(pyre_interpreter::PyError::type_error(
                "cannot create '_ssl.SSLSession' instances",
            ))
        }

        #[getter]
        fn id(&self) -> PyObjectRef {
            w_bytes_from_bytes(&unsafe { pyre_native::ssl::session_id(self.backend) })
        }

        #[getter]
        fn time(&self) -> i64 {
            unsafe { pyre_native::ssl::session_creation_time(self.backend) as i64 }
        }

        #[getter]
        fn timeout(&self) -> i64 {
            unsafe { pyre_native::ssl::session_timeout(self.backend) as i64 }
        }

        #[getter]
        fn ticket_lifetime_hint(&self) -> i64 {
            unsafe { pyre_native::ssl::session_timeout(self.backend) as i64 }
        }

        #[getter]
        fn has_ticket(&self) -> bool {
            true
        }

        fn __eq__(&self, other: PyObjectRef) -> PyObjectRef {
            let Some(other) = W_SSLSession::from_obj(other) else {
                return pyre_object::w_not_implemented();
            };
            w_bool_from(
                unsafe { pyre_native::ssl::session_id(self.backend) }
                    == unsafe { pyre_native::ssl::session_id(other.backend) },
            )
        }

        fn __repr__(&self) -> PyObjectRef {
            w_str_new_managed("<_ssl.SSLSession>")
        }
    }
} // ssl_session_methods

mod certificate_methods {
    use super::*;

    #[pyre_interpreter::pyre_methods]
    impl W_Certificate {
        #[staticmethod]
        fn __new__(_cls: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            Err(pyre_interpreter::PyError::type_error(
                "cannot create '_ssl.Certificate' instances",
            ))
        }

        fn public_bytes(
            &self,
            #[default(1)] encoding: i64,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            if encoding == 2 {
                return Ok(self.der);
            }
            if encoding == 1 || encoding == 0x101 {
                let encoded = rustpython_common::binascii::b2a_base64(
                    unsafe { pyre_object::bytesobject::w_bytes_data(self.der) },
                    false,
                );
                let mut pem = String::with_capacity(encoded.len() + 64);
                pem.push_str("-----BEGIN CERTIFICATE-----\n");
                for line in encoded.chunks(64) {
                    pem.push_str(std::str::from_utf8(line).expect("base64 is ASCII"));
                    pem.push('\n');
                }
                pem.push_str("-----END CERTIFICATE-----\n");
                // Peer/path certificates carry no OpenSSL auxiliary trust
                // data, so PEM_write_bio_X509_AUX has the same output as PEM.
                return Ok(w_str_new_managed(&pem));
            }
            Err(pyre_interpreter::PyError::value_error("Unsupported format"))
        }

        fn get_info(&self) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let der = unsafe { pyre_object::bytesobject::w_bytes_data(self.der) };
            let decoded = native_result(pyre_native::ssl::certificate_decode_der(der))?;
            Ok(decoded_certificate_dict(decoded))
        }

        fn __repr__(&self) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let der = unsafe { pyre_object::bytesobject::w_bytes_data(self.der) };
            let subject = native_result(pyre_native::ssl::certificate_subject_rfc2253(der))?;
            Ok(w_str_new_managed(&format!("<Certificate '{subject}'>")))
        }

        fn __hash__(&mut self) -> Result<i64, pyre_interpreter::PyError> {
            if self.hash == -1 {
                let der = unsafe { pyre_object::bytesobject::w_bytes_data(self.der) };
                self.hash = native_result(pyre_native::ssl::certificate_subject_hash(der))?;
                if self.hash == -1 {
                    self.hash = -2;
                }
            }
            Ok(self.hash)
        }

        fn __eq__(&self, other: PyObjectRef) -> PyObjectRef {
            let exact = unsafe {
                pyre_object::is_exact_type(
                    other,
                    &*<W_Certificate as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
                )
            };
            if !exact {
                return pyre_object::w_not_implemented();
            }
            let other = W_Certificate::from_obj(other).expect("exact Certificate instance");
            w_bool_from(
                unsafe { pyre_object::bytesobject::w_bytes_data(self.der) }
                    == unsafe { pyre_object::bytesobject::w_bytes_data(other.der) },
            )
        }
    }
} // certificate_methods

/// Sweep-time cleanup for opaque native TLS state.
///
/// # Safety
/// `obj` must be a GC-dead instance of the named class.
pub unsafe fn w_ssl_context_dealloc(obj: PyObjectRef) {
    if let Some(context) = W_SSLContext::from_obj(obj) {
        unsafe { pyre_native::ssl::context_free(context.backend) };
        context.backend = std::ptr::null_mut();
    }
}

/// # Safety
/// `obj` must be a GC-dead `W_MemoryBIO`.
pub unsafe fn w_memory_bio_dealloc(obj: PyObjectRef) {
    if let Some(bio) = W_MemoryBIO::from_obj(obj) {
        unsafe { pyre_native::ssl::memory_bio_free(bio.backend) };
        bio.backend = std::ptr::null_mut();
    }
}

/// # Safety
/// `obj` must be a GC-dead `W_SSLSession`.
pub unsafe fn w_ssl_session_dealloc(obj: PyObjectRef) {
    if let Some(session) = W_SSLSession::from_obj(obj) {
        unsafe { pyre_native::ssl::session_free(session.backend) };
        session.backend = std::ptr::null_mut();
    }
}

/// # Safety
/// `obj` must be a GC-dead `W_SSLSocket`.
pub unsafe fn w_ssl_socket_dealloc(obj: PyObjectRef) {
    if let Some(socket) = W_SSLSocket::from_obj(obj) {
        unsafe { pyre_native::ssl::connection_free(socket.backend) };
        unsafe { pyre_native::ssl::session_free(socket.requested_session) };
        socket.backend = std::ptr::null_mut();
        socket.requested_session = std::ptr::null_mut();
    }
}

fn rand_status(_args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(w_bool_from(true))
}

fn rand_add(_args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(w_none())
}

fn rand_bytes(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let size = pyre_interpreter::baseobjspace::int_w(args[0])?;
    if size < 0 {
        return Err(pyre_interpreter::PyError::value_error(
            "num must be positive",
        ));
    }
    let size = usize::try_from(size)
        .map_err(|_| pyre_interpreter::PyError::overflow_error("RAND_bytes size out of range"))?;
    let bytes = pyre_interpreter::importing::host::os::urandom(size)
        .map_err(|err| ssl_error(format!("RAND_bytes failed: {err}")))?;
    Ok(w_bytes_from_bytes(&bytes))
}

fn oid_tuple(entry: pyre_native::ssl::OidInfo) -> PyObjectRef {
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(w_int_new(i64::from(entry.nid)));
    fields.push(w_str_new_managed(entry.short_name));
    fields.push(w_str_new_managed(entry.long_name));
    fields.push(entry.oid.map(w_str_new_managed).unwrap_or_else(w_none));
    w_tuple_new(fields.take())
}

fn txt2obj(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    if positional.is_empty() || positional.len() > 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "txt2obj() takes 1 or 2 arguments",
        ));
    }
    let npos = positional.len();
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.publish(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
    let pos_base = roots.publish(positional);
    roots.normalize(base, 1 + npos);
    let value = pyre_interpreter::baseobjspace::str_utf8_w(roots.get(pos_base));
    let w = roots.get(base);
    let kwargs = if w.is_null() { None } else { Some(w) };
    let mut pos_buf = vec![pyre_object::PY_NULL; npos];
    pyre_object::gc_roots::shadow_stack_copy_range(pos_base, &mut pos_buf);
    drop(roots);
    let value = value?;
    let name_arg =
        pyre_interpreter::builtins::bind_pos_or_kw(&pos_buf, kwargs, 1, "name", "txt2obj", 2)?;
    pyre_interpreter::builtins::kwarg_reject_unknown(kwargs, &["name"], "txt2obj")?;
    let allow_names = name_arg
        .map(pyre_interpreter::baseobjspace::is_true)
        .transpose()?
        .unwrap_or(false);
    let name: &str = value.as_ref();
    let entry = if name
        .bytes()
        .next()
        .is_some_and(|byte| byte.is_ascii_digit())
    {
        pyre_native::ssl::oid_by_oid_string(name)
    } else if allow_names {
        pyre_native::ssl::oid_by_name(name)
    } else {
        None
    };
    entry
        .map(oid_tuple)
        .ok_or_else(|| pyre_interpreter::PyError::value_error(format!("unknown object '{name}'")))
}

fn nid2obj(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let nid = pyre_interpreter::baseobjspace::int_w(args[0])?;
    let Ok(nid32) = i32::try_from(nid) else {
        return Err(pyre_interpreter::PyError::value_error(format!(
            "unknown NID {nid}"
        )));
    };
    pyre_native::ssl::oid_by_nid(nid32)
        .map(oid_tuple)
        .ok_or_else(|| pyre_interpreter::PyError::value_error(format!("unknown NID {nid}")))
}

fn get_default_verify_paths(
    _args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let (cert_file, cert_dir) = pyre_native::ssl::default_verify_paths();
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(w_str_new_managed("SSL_CERT_FILE"));
    fields.push(w_str_new_managed(&cert_file));
    fields.push(w_str_new_managed("SSL_CERT_DIR"));
    fields.push(w_str_new_managed(&cert_dir));
    Ok(w_tuple_new(fields.take()))
}

/// The Windows system-certificate-store readers behind `ssl.enum_certificates`
/// and `ssl.enum_crls` (`_ssl_enum_certificates_impl`, `_ssl_enum_crls_impl`).
/// Verification does not depend on them — `set_default_verify_paths` loads the
/// Windows roots through the platform certificate provider — but callers that
/// mine the store themselves (`wincertstore`, trust-list builders) read it
/// through these.
#[cfg(all(windows, feature = "host_env"))]
mod cert_store {
    use pyre_object::{PyObjectRef, gc_roots::RootedItems};
    use rustpython_host_env::cert_store::{self as host, CertificateUses, EncodingType};

    fn win32_error(error: std::io::Error) -> pyre_interpreter::PyError {
        pyre_interpreter::PyError::os_error_win32_syscall2(
            error.raw_os_error().unwrap_or(0),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        )
    }

    fn encoding_type(encoding: EncodingType) -> PyObjectRef {
        match encoding {
            EncodingType::X509Asn => pyre_object::w_str_new_managed("x509_asn"),
            EncodingType::Pkcs7Asn => pyre_object::w_str_new_managed("pkcs_7_asn"),
            EncodingType::Other(value) => pyre_object::w_int_new(value as i64),
        }
    }

    pub fn enum_certificates(
        args: &[PyObjectRef],
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let name = pyre_interpreter::baseobjspace::str_utf8_w(args[0])?;
        let certs = host::enum_certificates(name);
        if !certs.had_open_store {
            return Err(pyre_interpreter::PyError::os_error(format!(
                "failed to open certificate store {name:?}"
            )));
        }
        let mut items = RootedItems::new();
        for entry in certs.entries {
            let tuple = {
                let mut fields = RootedItems::new();
                fields.push(pyre_object::w_bytes_from_bytes(&entry.der));
                fields.push(encoding_type(entry.encoding));
                // A store can contain hundreds of objects; root both the
                // completed tuples and every intermediate allocation.
                let trust = match entry.valid_uses.map_err(win32_error)? {
                    CertificateUses::All => pyre_object::w_bool_from(true),
                    CertificateUses::Oids(oids) => {
                        let mut strings = RootedItems::new();
                        for oid in oids {
                            strings.push(pyre_object::w_str_new_managed(&oid));
                        }
                        pyre_object::w_frozenset_from_items(&strings.take())
                    }
                };
                fields.push(trust);
                pyre_object::w_tuple_new(fields.take())
            };
            items.push(tuple);
        }
        Ok(pyre_object::w_list_new(items.take()))
    }

    pub fn enum_crls(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let name = pyre_interpreter::baseobjspace::str_utf8_w(args[0])?;
        let entries = host::enum_crls(name).map_err(|_| {
            pyre_interpreter::PyError::os_error(format!(
                "failed to open certificate store {name:?}"
            ))
        })?;
        let mut items = RootedItems::new();
        for entry in entries {
            let tuple = {
                let mut fields = RootedItems::new();
                fields.push(pyre_object::w_bytes_from_bytes(&entry.der));
                fields.push(encoding_type(entry.encoding));
                pyre_object::w_tuple_new(fields.take())
            };
            items.push(tuple);
        }
        Ok(pyre_object::w_list_new(items.take()))
    }
}

fn test_decode_cert(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let path = fs_path(args[0])?;
    let cert = native_result(pyre_native::ssl::certificate_decode_file(&path))?;
    Ok(decoded_certificate_dict(cert))
}

pyre_interpreter::py_module! {
    "_ssl",
    interpleveldefs: {
        "_SSLContext" => context_methods::type_object(),
        "MemoryBIO" => memory_bio_methods::type_object(),
        "SSLSession" => ssl_session_methods::type_object(),
        "_SSLSocket" => ssl_socket_methods::type_object(),
        "Certificate" => certificate_methods::type_object(),
        // `SSLError` alone is spelled `make_exc_type`: `sslmodule_init_types`
        // builds it from a spec that names `sizeof(PySSLErrorObject)` as its
        // own `basicsize`, which leaves it without a managed weakref.  The
        // subclasses below come from `PyErr_NewExceptionWithDoc` and do have
        // one.
        "SSLError" => pyre_interpreter::builtins::make_exc_type(
            "ssl.SSLError",
            pyre_interpreter::builtins::exc_os_error_new,
            pyre_interpreter::builtins::lookup_exc_class("OSError").expect("OSError installed"),
        ),
        "SSLZeroReturnError" => pyre_interpreter::builtins::new_exception_class(
            "ssl.SSLZeroReturnError",
            pyre_interpreter::builtins::exc_os_error_new,
            pyre_interpreter::builtins::lookup_exc_class("ssl.SSLError").expect("SSLError installed"),
        ),
        "SSLWantReadError" => pyre_interpreter::builtins::new_exception_class(
            "ssl.SSLWantReadError",
            pyre_interpreter::builtins::exc_os_error_new,
            pyre_interpreter::builtins::lookup_exc_class("ssl.SSLError").expect("SSLError installed"),
        ),
        "SSLWantWriteError" => pyre_interpreter::builtins::new_exception_class(
            "ssl.SSLWantWriteError",
            pyre_interpreter::builtins::exc_os_error_new,
            pyre_interpreter::builtins::lookup_exc_class("ssl.SSLError").expect("SSLError installed"),
        ),
        "SSLSyscallError" => pyre_interpreter::builtins::new_exception_class(
            "ssl.SSLSyscallError",
            pyre_interpreter::builtins::exc_os_error_new,
            pyre_interpreter::builtins::lookup_exc_class("ssl.SSLError").expect("SSLError installed"),
        ),
        "SSLEOFError" => pyre_interpreter::builtins::new_exception_class(
            "ssl.SSLEOFError",
            pyre_interpreter::builtins::exc_os_error_new,
            pyre_interpreter::builtins::lookup_exc_class("ssl.SSLError").expect("SSLError installed"),
        ),
        // OpenSSL-shaped compatibility fields are required by `ssl.py` and
        // consumers such as urllib3, while the suffix names the real backend.
        "OPENSSL_VERSION" => w_str_new("OpenSSL 3.0.0-compatible (AWS-LC/rustls 0.23)"),
        "OPENSSL_VERSION_NUMBER" => w_int_new(0x3000_0000),
        "OPENSSL_VERSION_INFO" => w_tuple_new(vec![w_int_new(3), w_int_new(0), w_int_new(0), w_int_new(0), w_int_new(15)]),
        "_OPENSSL_API_VERSION" => w_tuple_new(vec![w_int_new(3), w_int_new(0), w_int_new(0), w_int_new(0), w_int_new(15)]),
        "_DEFAULT_CIPHERS" => w_str_new(&pyre_native::ssl::default_cipher_string()),
        "HAS_SNI" => w_bool_from(true),
        "HAS_ECDH" => w_bool_from(true),
        "HAS_NPN" => w_bool_from(false),
        "HAS_ALPN" => w_bool_from(true),
        "HAS_SSLv2" => w_bool_from(false),
        "HAS_SSLv3" => w_bool_from(false),
        "HAS_TLSv1" => w_bool_from(false),
        "HAS_TLSv1_1" => w_bool_from(false),
        "HAS_TLSv1_2" => w_bool_from(true),
        "HAS_TLSv1_3" => w_bool_from(true),
        "HAS_PSK" => w_bool_from(false),
        "HAS_PHA" => w_bool_from(false)
    },
    int_constants: {
        "PROTOCOL_SSLv23" => PROTOCOL_TLS,
        "PROTOCOL_TLS" => PROTOCOL_TLS,
        "PROTOCOL_TLS_CLIENT" => PROTOCOL_TLS_CLIENT,
        "PROTOCOL_TLS_SERVER" => PROTOCOL_TLS_SERVER,
        "PROTOCOL_TLSv1" => PROTOCOL_TLSV1,
        "PROTOCOL_TLSv1_1" => PROTOCOL_TLSV1_1,
        "PROTOCOL_TLSv1_2" => PROTOCOL_TLSV1_2,
        "PROTOCOL_TLSv1_3" => PROTOCOL_TLSV1_3,
        "PROTO_MINIMUM_SUPPORTED" => -2,
        "PROTO_MAXIMUM_SUPPORTED" => -1,
        "PROTO_SSLv3" => 0x300,
        "PROTO_TLSv1" => 0x301,
        "PROTO_TLSv1_1" => 0x302,
        "PROTO_TLSv1_2" => pyre_native::ssl::PROTO_TLSV1_2,
        "PROTO_TLSv1_3" => pyre_native::ssl::PROTO_TLSV1_3,
        "CERT_NONE" => CERT_NONE,
        "CERT_OPTIONAL" => CERT_OPTIONAL,
        "CERT_REQUIRED" => CERT_REQUIRED,
        "VERIFY_DEFAULT" => pyre_native::ssl::VerifyFlags::DEFAULT.bits(),
        "VERIFY_CRL_CHECK_LEAF" => pyre_native::ssl::VerifyFlags::CRL_CHECK_LEAF.bits(),
        "VERIFY_CRL_CHECK_CHAIN" => pyre_native::ssl::VerifyFlags::CRL_CHECK_CHAIN.bits(),
        "VERIFY_X509_STRICT" => pyre_native::ssl::VerifyFlags::X509_STRICT.bits(),
        "VERIFY_ALLOW_PROXY_CERTS" => pyre_native::ssl::VerifyFlags::ALLOW_PROXY_CERTS.bits(),
        "VERIFY_X509_TRUSTED_FIRST" => pyre_native::ssl::VerifyFlags::X509_TRUSTED_FIRST.bits(),
        "VERIFY_X509_PARTIAL_CHAIN" => pyre_native::ssl::VerifyFlags::X509_PARTIAL_CHAIN.bits(),
        "OP_ALL" => pyre_native::ssl::SslOp::ALL.bits() as i32,
        "OP_NO_SSLv2" => 0,
        "OP_NO_SSLv3" => pyre_native::ssl::SslOp::NO_SSLV3.bits() as i32,
        "OP_NO_TLSv1" => pyre_native::ssl::SslOp::NO_TLSV1.bits() as i32,
        "OP_NO_TLSv1_1" => pyre_native::ssl::SslOp::NO_TLSV1_1.bits() as i32,
        "OP_NO_TLSv1_2" => pyre_native::ssl::SslOp::NO_TLSV1_2.bits() as i32,
        "OP_NO_TLSv1_3" => pyre_native::ssl::SslOp::NO_TLSV1_3.bits() as i32,
        "OP_NO_COMPRESSION" => pyre_native::ssl::SslOp::NO_COMPRESSION.bits() as i32,
        "OP_CIPHER_SERVER_PREFERENCE" => {
            pyre_native::ssl::SslOp::CIPHER_SERVER_PREFERENCE.bits() as i32
        },
        "OP_SINGLE_DH_USE" => 0,
        "OP_SINGLE_ECDH_USE" => 0,
        "OP_NO_TICKET" => pyre_native::ssl::SslOp::NO_TICKET.bits() as i32,
        "OP_LEGACY_SERVER_CONNECT" => {
            pyre_native::ssl::SslOp::LEGACY_SERVER_CONNECT.bits() as i32
        },
        "OP_NO_RENEGOTIATION" => pyre_native::ssl::SslOp::NO_RENEGOTIATION.bits() as i32,
        "OP_IGNORE_UNEXPECTED_EOF" => {
            pyre_native::ssl::SslOp::IGNORE_UNEXPECTED_EOF.bits() as i32
        },
        "OP_ENABLE_MIDDLEBOX_COMPAT" => {
            pyre_native::ssl::SslOp::ENABLE_MIDDLEBOX_COMPAT.bits() as i32
        },
        "SSL_ERROR_NONE" => 0,
        "SSL_ERROR_SSL" => 1,
        "SSL_ERROR_WANT_READ" => 2,
        "SSL_ERROR_WANT_WRITE" => 3,
        "SSL_ERROR_WANT_X509_LOOKUP" => 4,
        "SSL_ERROR_SYSCALL" => 5,
        "SSL_ERROR_ZERO_RETURN" => 6,
        "SSL_ERROR_WANT_CONNECT" => 7,
        "SSL_ERROR_EOF" => 8,
        "SSL_ERROR_INVALID_ERROR_CODE" => 10,
        "HOSTFLAG_NEVER_CHECK_SUBJECT" => 0x20,
        "SSL3_RT_CHANGE_CIPHER_SPEC" => pyre_native::ssl::SSL3_RT_CHANGE_CIPHER_SPEC,
        "SSL3_RT_ALERT" => pyre_native::ssl::SSL3_RT_ALERT,
        "SSL3_RT_HANDSHAKE" => pyre_native::ssl::SSL3_RT_HANDSHAKE,
        "SSL3_RT_APPLICATION_DATA" => pyre_native::ssl::SSL3_RT_APPLICATION_DATA,
        "SSL3_RT_HEADER" => pyre_native::ssl::SSL3_RT_HEADER,
        "SSL3_MT_CHANGE_CIPHER_SPEC" => pyre_native::ssl::SSL3_MT_CHANGE_CIPHER_SPEC,
        "ENCODING_PEM" => 1,
        "ENCODING_DER" => 2,
        "ENCODING_PEM_AUX" => 0x101
    },
    functions: {
        "RAND_status" / 0 = rand_status,
        "RAND_add" / * = rand_add,
        "RAND_bytes" / 1 = rand_bytes,
        "txt2obj" / * = txt2obj,
        "nid2obj" / 1 = nid2obj,
        "get_default_verify_paths" / 0 = get_default_verify_paths,
        "_test_decode_cert" / 1 = test_decode_cert
    },
    extra_init: |ns| {
        // `py_module!` pins the module dict in `ns_slot`, which this body
        // cannot name. Publish that live word into a slot the constructor
        // below can reload (`ShadowStackFrameworkGCTransformer.pop_roots`).
        let ns_here = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        // [3.14-spec] CPython 3.14 rejects each of these immutable module
        // heap types as a base.  PyPy's shared public pair, `MemoryBIO` and
        // `SSLSession`, remain app-level classes in `_cffi_ssl._stdssl`;
        // preserve those owner/storage choices and suppress only the
        // caller-visible per-type BASETYPE capability.
        for name in ["MemoryBIO", "SSLSession", "Certificate", "_SSLSocket"] {
            let ty = pyre_interpreter::module_ns_get(
                pyre_object::gc_roots::shadow_stack_get(ns_here),
                name,
            )
            .expect("_ssl type installed");
            unsafe { pyre_object::w_type_suppress_cpython_basetype(ty) };
        }
        // The Windows-only store readers. They live here rather than in
        // `functions:` because that table has no per-entry platform guard, and
        // `ssl.py:257-258` imports both names behind `sys.platform == "win32"`.
        #[cfg(all(windows, feature = "host_env"))]
        for (name, func) in [
            (
                "enum_certificates",
                pyre_interpreter::py_module_fn!("enum_certificates", 1, cert_store::enum_certificates),
            ),
            (
                "enum_crls",
                pyre_interpreter::py_module_fn!("enum_crls", 1, cert_store::enum_crls),
            ),
        ] {
            pyre_interpreter::__pyre_put_new!(ns_here, name, pyre_interpreter::gateway::with_module("_ssl", func));
        }
        let ssl_error = pyre_interpreter::builtins::lookup_exc_class("ssl.SSLError")
            .expect("SSLError installed");
        let ssl_error_dict =
            unsafe { pyre_object::w_type_get_dict_ptr(ssl_error) as PyObjectRef };
        // Publish the class and its dict together. Sequential `pin_root`
        // would query after the first write and leave the other word
        // invisible (`RootScope::pin_roots`).
        let ssl_error_base = pyre_object::gc_roots::pin_roots(&[ssl_error, ssl_error_dict]);
        pyre_interpreter::__pyre_put_new!(
            ssl_error_base + 1,
            "__str__",
            pyre_interpreter::make_builtin_function_with_arity("__str__", ssl_error_str, 1)
        );
        let mut value_error = pyre_interpreter::builtins::lookup_exc_class("ValueError")
            .expect("ValueError installed");
        // The constructor below reloads both bases
        // (`ShadowStackFrameworkGCTransformer.pop_roots`).
        let mut ns = pyre_object::gc_roots::shadow_stack_get(ns_here);
        let mut ssl_error = pyre_object::gc_roots::shadow_stack_get(ssl_error_base);
        let mut cert_error = pyre_object::with_roots!(ns, ssl_error, value_error => {
            pyre_interpreter::builtins::make_exc_type_multi(
                "ssl.SSLCertVerificationError",
                pyre_interpreter::builtins::exc_os_error_new,
                &[ssl_error, value_error],
            )
        });
        pyre_interpreter::__pyre_store!(ns, "SSLCertVerificationError", cert_error);
        for (name, value) in [
            ("ALERT_DESCRIPTION_CLOSE_NOTIFY", 0),
            ("ALERT_DESCRIPTION_UNEXPECTED_MESSAGE", 10),
            ("ALERT_DESCRIPTION_BAD_RECORD_MAC", 20),
            ("ALERT_DESCRIPTION_DECRYPTION_FAILED", 21),
            ("ALERT_DESCRIPTION_RECORD_OVERFLOW", 22),
            ("ALERT_DESCRIPTION_DECOMPRESSION_FAILURE", 30),
            ("ALERT_DESCRIPTION_HANDSHAKE_FAILURE", 40),
            ("ALERT_DESCRIPTION_NO_CERTIFICATE", 41),
            ("ALERT_DESCRIPTION_BAD_CERTIFICATE", 42),
            ("ALERT_DESCRIPTION_UNSUPPORTED_CERTIFICATE", 43),
            ("ALERT_DESCRIPTION_CERTIFICATE_REVOKED", 44),
            ("ALERT_DESCRIPTION_CERTIFICATE_EXPIRED", 45),
            ("ALERT_DESCRIPTION_CERTIFICATE_UNKNOWN", 46),
            ("ALERT_DESCRIPTION_ILLEGAL_PARAMETER", 47),
            ("ALERT_DESCRIPTION_UNKNOWN_CA", 48),
            ("ALERT_DESCRIPTION_ACCESS_DENIED", 49),
            ("ALERT_DESCRIPTION_DECODE_ERROR", 50),
            ("ALERT_DESCRIPTION_DECRYPT_ERROR", 51),
            ("ALERT_DESCRIPTION_EXPORT_RESTRICTION", 60),
            ("ALERT_DESCRIPTION_PROTOCOL_VERSION", 70),
            ("ALERT_DESCRIPTION_INSUFFICIENT_SECURITY", 71),
            ("ALERT_DESCRIPTION_INTERNAL_ERROR", 80),
            ("ALERT_DESCRIPTION_INAPPROPRIATE_FALLBACK", 86),
            ("ALERT_DESCRIPTION_USER_CANCELLED", 90),
            ("ALERT_DESCRIPTION_NO_RENEGOTIATION", 100),
            ("ALERT_DESCRIPTION_MISSING_EXTENSION", 109),
            ("ALERT_DESCRIPTION_UNSUPPORTED_EXTENSION", 110),
            ("ALERT_DESCRIPTION_CERTIFICATE_UNOBTAINABLE", 111),
            ("ALERT_DESCRIPTION_UNRECOGNIZED_NAME", 112),
            ("ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE", 113),
            ("ALERT_DESCRIPTION_BAD_CERTIFICATE_HASH_VALUE", 114),
            ("ALERT_DESCRIPTION_UNKNOWN_PSK_IDENTITY", 115),
            ("ALERT_DESCRIPTION_CERTIFICATE_REQUIRED", 116),
            ("ALERT_DESCRIPTION_NO_APPLICATION_PROTOCOL", 120),
        ] {
            pyre_interpreter::__pyre_store!(ns, name, w_int_new(value));
        }
    },
}

/// `_ssl._SSLSocket` owns its context, transport endpoints, public owner,
/// and hostname directly on the typed object.
unsafe fn ssl_socket_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let socket = unsafe { &mut *(obj_addr as *mut W_SSLSocket) };
    f(std::ptr::addr_of_mut!(socket.ob.w_class) as *mut majit_ir::GcRef);
    f(std::ptr::addr_of_mut!(socket.context) as *mut majit_ir::GcRef);
    f(std::ptr::addr_of_mut!(socket.socket) as *mut majit_ir::GcRef);
    f(std::ptr::addr_of_mut!(socket.incoming) as *mut majit_ir::GcRef);
    f(std::ptr::addr_of_mut!(socket.outgoing) as *mut majit_ir::GcRef);
    f(std::ptr::addr_of_mut!(socket.owner) as *mut majit_ir::GcRef);
    f(std::ptr::addr_of_mut!(socket.server_hostname) as *mut majit_ir::GcRef);
}

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // rustls objects sit behind opaque native pointers. `_SSLContext` traces
    // its callback and path references inline; a user subclass is
    // `typedef.py` `_getusercls` and adds mapdict storage. `MemoryBIO` is
    // not a base. The sweep destructors release the opaque rustls allocations.
    let pyre_class = ModuleGcLayout::PyreClass {
        memory_pressure_offset: None,
    };
    types.push(ModuleGcType {
        descriptor: <W_SSLContext as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: pyre_class,
        destructor: Some(gc_destructor!(w_ssl_context_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: &W_SSLCONTEXT_USER_PYRE_CLASS_DESCRIPTOR,
        layout: pyre_class,
        destructor: Some(gc_destructor!(w_ssl_context_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: <W_MemoryBIO as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: pyre_class,
        destructor: Some(gc_destructor!(w_memory_bio_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: <W_SSLSession as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::Object,
        destructor: Some(gc_destructor!(w_ssl_session_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: <W_SSLSocket as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::CustomTrace(ssl_socket_custom_trace),
        destructor: Some(gc_destructor!(w_ssl_socket_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: <W_Certificate as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
}
