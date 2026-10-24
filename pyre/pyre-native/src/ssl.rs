//! rustls backend for Python's `_ssl` module.
//!
//! This crate is deliberately outside the Charon/LLBC extraction.  The
//! interpreter owns each backend value through an opaque pointer and reaches
//! it only through these non-generic, non-inlined functions, exactly as it
//! reaches the native hash and zlib engines.  No Python object lives here.

use parking_lot::Mutex;
use std::io::Cursor;
#[cfg(not(feature = "host_env"))]
use std::io::Write;
use std::sync::Arc;
#[cfg(not(feature = "host_env"))]
use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rustls::client::ClientSessionStore;
use rustls::pki_types::{
    CertificateDer, CertificateRevocationListDer, PrivateKeyDer, PrivatePkcs8KeyDer,
};
use rustls::sign::CertifiedKey;
use sha1::{Digest, Sha1};
#[cfg(not(feature = "host_env"))]
use x509_parser::prelude::FromDer;

use rustpython_host_env::ssl as host_ssl;
#[cfg(feature = "host_env")]
use rustpython_host_env::ssl::{cipher, keylog, msg, providers::CryptoExt};

#[cfg(not(feature = "host_env"))]
static INSTALL_PROVIDER: Once = Once::new();

/// Install the process-wide rustls provider before constructing TLS state.
///
/// Rustls requires one provider for a process.  AWS-LC is selected by the
/// workspace feature because it is RustPython's tested rustls provider and its
/// name is already admitted by CPython's `test_ssl` backend-version check.
#[inline(never)]
pub fn ensure_provider() {
    #[cfg(feature = "host_env")]
    {
        use rustls::crypto::aws_lc_rs;
        let ext = CryptoExt {
            all_cipher_suites: Some(aws_lc_rs::ALL_CIPHER_SUITES),
            default_cipher_suites: Some(aws_lc_rs::DEFAULT_CIPHER_SUITES),
            all_kx_groups: Some(aws_lc_rs::ALL_KX_GROUPS),
            any_supported_key: Some(aws_lc_rs::sign::any_supported_type),
            ticketer: aws_lc_rs::Ticketer::new,
        };
        let _ = CryptoExt::set_provider(aws_lc_rs::default_provider(), ext);
    }
    #[cfg(not(feature = "host_env"))]
    INSTALL_PROVIDER.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    });
}

/// In-memory encrypted transport used by `SSLObject`.
///
/// host_env owns the buffer/`start` compaction; the pointer wrappers below
/// keep the interpreter off that type graph.
pub type MemoryBio = host_ssl::MemoryBio;

#[inline(never)]
pub fn memory_bio_new() -> *mut MemoryBio {
    Box::into_raw(Box::new(MemoryBio::new()))
}

/// # Safety
/// `bio` must be null or a pointer returned by [`memory_bio_new`] that has not
/// already been freed.
#[inline(never)]
pub unsafe fn memory_bio_free(bio: *mut MemoryBio) {
    if !bio.is_null() {
        unsafe { drop(Box::from_raw(bio)) };
    }
}

/// # Safety
/// `bio` must point to a live [`MemoryBio`].
#[inline(never)]
pub unsafe fn memory_bio_read(bio: *mut MemoryBio, size: usize) -> Vec<u8> {
    unsafe { (*bio).read(size) }
}

/// # Safety
/// `bio` must point to a live [`MemoryBio`].
#[inline(never)]
pub unsafe fn memory_bio_write(bio: *mut MemoryBio, data: &[u8]) -> Result<usize, &'static str> {
    unsafe { (*bio).write(data) }.map_err(|_| "cannot write() after write_eof()")
}

/// # Safety
/// `bio` must point to a live [`MemoryBio`].
#[inline(never)]
pub unsafe fn memory_bio_write_eof(bio: *mut MemoryBio) {
    unsafe { (*bio).write_eof() };
}

/// # Safety
/// `bio` must point to a live [`MemoryBio`].
#[inline(never)]
pub unsafe fn memory_bio_pending(bio: *const MemoryBio) -> usize {
    unsafe { (*bio).pending() }
}

/// # Safety
/// `bio` must point to a live [`MemoryBio`].
#[inline(never)]
pub unsafe fn memory_bio_eof(bio: *const MemoryBio) -> bool {
    unsafe { (*bio).eof() }
}

/// Mutable Python-visible SSL context settings plus rustls trust material.
/// Connection configs are built from this state when `_wrap_socket` or
/// `_wrap_bio` is called; rustls configs themselves are intentionally not
/// mutated after publication.
pub struct Context {
    /// Process-wide monotonic identity.  A freed context's address can be
    /// reused by the allocator, so an `SSLSession` must not be matched to its
    /// originating context by pointer value.
    identity: usize,
    protocol: i32,
    check_hostname: bool,
    verify_mode: i32,
    verify_flags: i32,
    options: u64,
    minimum_version: i32,
    maximum_version: i32,
    alpn_protocols: Vec<Vec<u8>>,
    roots: rustls::RootCertStore,
    root_der: Vec<Vec<u8>>,
    // OpenSSL's hashed CA directories are lookup sources, not eagerly loaded
    // stores.  Keep the directories themselves and materialize candidates in
    // each immutable rustls ClientConfig; the certificate actually selected
    // for a successful chain is published to `root_der` after the handshake.
    capaths: Vec<std::path::PathBuf>,
    // Materializing a hashed directory costs one read plus one PEM parse per
    // entry, and a system store holds hundreds.  Cache the result against the
    // directory list and their modification times so a connection pays only
    // one stat per directory once the contents are stable.
    capath_cache: Mutex<Option<(Vec<CapathStamp>, Arc<Vec<Vec<u8>>>)>>,
    crls: Vec<CertificateRevocationListDer<'static>>,
    cipher_suites: Option<Vec<rustls::SupportedCipherSuite>>,
    /// TLS 1.3 suites from `SSLContext.set_ciphersuites`.  Independent of
    /// `cipher_suites` so `SSL_CTX_set_cipher_list` cannot drop TLS 1.3 and
    /// `SSL_CTX_set_ciphersuites` cannot drop TLS 1.2.
    tls13_cipher_suites: Option<Vec<rustls::SupportedCipherSuite>>,
    kx_groups: Option<Vec<&'static dyn rustls::crypto::SupportedKxGroup>>,
    /// Schemes from `SSLContext.set_server_sigalgs` (`SSL_CTX_set1_sigalgs_list`).
    server_sigalgs: Option<Vec<rustls::SignatureScheme>>,
    certified_keys: Vec<Arc<CertifiedKey>>,
    num_tickets: Option<usize>,
    server_session_store: Arc<dyn rustls::server::StoresServerSessions>,
    server_ticketer: Arc<dyn rustls::server::ProducesTickets>,
    /// PyPy `_SSLContext.keylog_bio`: one context-owned destination shared by
    /// every connection created from the context.  The file itself is behind
    /// a lock because rustls can report secrets from concurrent handshakes.
    key_log_file: Arc<ContextKeyLog>,
    accept_count: AtomicUsize,
    session_hits: AtomicUsize,
}

/// One `capath` directory together with the modification time observed when
/// its certificates were parsed.
type CapathStamp = (std::path::PathBuf, Option<SystemTime>);

static NEXT_CONTEXT_IDENTITY: AtomicUsize = AtomicUsize::new(1);

#[allow(non_upper_case_globals)]
pub use host_ssl::{
    ALERT_DESCRIPTION_ACCESS_DENIED, ALERT_DESCRIPTION_BAD_CERTIFICATE,
    ALERT_DESCRIPTION_BAD_CERTIFICATE_HASH_VALUE,
    ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE, ALERT_DESCRIPTION_BAD_RECORD_MAC,
    ALERT_DESCRIPTION_CERTIFICATE_EXPIRED, ALERT_DESCRIPTION_CERTIFICATE_REQUIRED,
    ALERT_DESCRIPTION_CERTIFICATE_REVOKED, ALERT_DESCRIPTION_CERTIFICATE_UNKNOWN,
    ALERT_DESCRIPTION_CERTIFICATE_UNOBTAINABLE, ALERT_DESCRIPTION_CLOSE_NOTIFY,
    ALERT_DESCRIPTION_DECODE_ERROR, ALERT_DESCRIPTION_DECOMPRESSION_FAILURE,
    ALERT_DESCRIPTION_DECRYPT_ERROR, ALERT_DESCRIPTION_DECRYPTION_FAILED,
    ALERT_DESCRIPTION_EXPORT_RESTRICTION, ALERT_DESCRIPTION_HANDSHAKE_FAILURE,
    ALERT_DESCRIPTION_ILLEGAL_PARAMETER, ALERT_DESCRIPTION_INAPPROPRIATE_FALLBACK,
    ALERT_DESCRIPTION_INSUFFICIENT_SECURITY, ALERT_DESCRIPTION_INTERNAL_ERROR,
    ALERT_DESCRIPTION_MISSING_EXTENSION, ALERT_DESCRIPTION_NO_APPLICATION_PROTOCOL,
    ALERT_DESCRIPTION_NO_CERTIFICATE, ALERT_DESCRIPTION_NO_RENEGOTIATION,
    ALERT_DESCRIPTION_PROTOCOL_VERSION, ALERT_DESCRIPTION_RECORD_OVERFLOW,
    ALERT_DESCRIPTION_UNEXPECTED_MESSAGE, ALERT_DESCRIPTION_UNKNOWN_CA,
    ALERT_DESCRIPTION_UNKNOWN_PSK_IDENTITY, ALERT_DESCRIPTION_UNRECOGNIZED_NAME,
    ALERT_DESCRIPTION_UNSUPPORTED_CERTIFICATE, ALERT_DESCRIPTION_UNSUPPORTED_EXTENSION,
    ALERT_DESCRIPTION_USER_CANCELLED, CERT_NONE, CERT_OPTIONAL, CERT_REQUIRED, ENCODING_DER,
    ENCODING_PEM, ENCODING_PEM_AUX, HOSTFLAG_NEVER_CHECK_SUBJECT, OP_ALL,
    OP_CIPHER_SERVER_PREFERENCE, OP_ENABLE_MIDDLEBOX_COMPAT, OP_IGNORE_UNEXPECTED_EOF,
    OP_LEGACY_SERVER_CONNECT, OP_NO_COMPRESSION, OP_NO_RENEGOTIATION, OP_NO_SSLv2, OP_NO_SSLv3,
    OP_NO_TICKET, OP_NO_TLSV1, OP_NO_TLSV1_1, OP_NO_TLSV1_2, OP_NO_TLSV1_3, OP_SINGLE_DH_USE,
    OP_SINGLE_ECDH_USE, PROTO_MAXIMUM_SUPPORTED, PROTO_MINIMUM_SUPPORTED, PROTO_SSL3, PROTO_TLSV1,
    PROTO_TLSV1_1, PROTO_TLSV1_2, PROTO_TLSV1_3, PROTOCOL_TLS, PROTOCOL_TLS_CLIENT,
    PROTOCOL_TLS_SERVER, PROTOCOL_TLSV1, PROTOCOL_TLSV1_1, PROTOCOL_TLSV1_2, PROTOCOL_TLSV1_3,
    SSL_ERROR_EOF, SSL_ERROR_INVALID_ERROR_CODE, SSL_ERROR_NONE, SSL_ERROR_SSL, SSL_ERROR_SYSCALL,
    SSL_ERROR_WANT_CONNECT, SSL_ERROR_WANT_READ, SSL_ERROR_WANT_WRITE, SSL_ERROR_WANT_X509_LOOKUP,
    SSL_ERROR_ZERO_RETURN, SSL3_MT_CHANGE_CIPHER_SPEC, SSL3_RT_ALERT, SSL3_RT_APPLICATION_DATA,
    SSL3_RT_CHANGE_CIPHER_SPEC, SSL3_RT_HANDSHAKE, SSL3_RT_HEADER, TLS_ERROR_CERT_VERIFY_BASE,
    TLS_ERROR_EOF, TLS_ERROR_NO_MEMORY, TLS_ERROR_SSL, TLS_ERROR_WANT_READ, TLS_ERROR_WANT_WRITE,
    TLS_ERROR_ZERO_RETURN,
};

bitflags::bitflags! {
    /// `SSL_OP_*` bits the `_ssl` module publishes.  Both `SslOp::DEFAULT` and
    /// [`enabled_versions`] decode this space, so the two must not name the same
    /// bit differently.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct SslOp: u64 {
        const ALL = host_ssl::OP_ALL as u64;
        const NO_SSLV3 = host_ssl::OP_NO_SSLv3 as u64;
        const NO_TLSV1 = host_ssl::OP_NO_TLSV1 as u64;
        const NO_TLSV1_1 = host_ssl::OP_NO_TLSV1_1 as u64;
        const NO_TLSV1_2 = host_ssl::OP_NO_TLSV1_2 as u64;
        const NO_TLSV1_3 = host_ssl::OP_NO_TLSV1_3 as u64;
        const NO_COMPRESSION = host_ssl::OP_NO_COMPRESSION as u64;
        const CIPHER_SERVER_PREFERENCE = host_ssl::OP_CIPHER_SERVER_PREFERENCE as u64;
        const ENABLE_MIDDLEBOX_COMPAT = host_ssl::OP_ENABLE_MIDDLEBOX_COMPAT as u64;
        const NO_TICKET = host_ssl::OP_NO_TICKET as u64;
        const LEGACY_SERVER_CONNECT = host_ssl::OP_LEGACY_SERVER_CONNECT as u64;
        const NO_RENEGOTIATION = host_ssl::OP_NO_RENEGOTIATION as u64;
        const IGNORE_UNEXPECTED_EOF = host_ssl::OP_IGNORE_UNEXPECTED_EOF as u64;
        const DEFAULT = Self::ALL.bits()
            | Self::NO_SSLV3.bits()
            | Self::NO_COMPRESSION.bits()
            | Self::CIPHER_SERVER_PREFERENCE.bits()
            | Self::ENABLE_MIDDLEBOX_COMPAT.bits();
    }
}

bitflags::bitflags! {
    /// `X509_V_FLAG_*` bits carried by `SSLContext.verify_flags`.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct VerifyFlags: i32 {
        const DEFAULT = host_ssl::VERIFY_DEFAULT;
        const CRL_CHECK_LEAF = host_ssl::VERIFY_CRL_CHECK_LEAF;
        const CRL_CHECK_CHAIN = host_ssl::VERIFY_CRL_CHECK_CHAIN;
        const X509_STRICT = host_ssl::VERIFY_X509_STRICT;
        const ALLOW_PROXY_CERTS = host_ssl::VERIFY_ALLOW_PROXY_CERTS;
        const X509_TRUSTED_FIRST = host_ssl::VERIFY_X509_TRUSTED_FIRST;
        const X509_PARTIAL_CHAIN = host_ssl::VERIFY_X509_PARTIAL_CHAIN;
    }
}

/// Revocation scope requested by `verify_flags`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CrlScope {
    Disabled,
    EndEntityOnly,
    FullChain,
}

fn crl_scope(verify_flags: i32) -> CrlScope {
    if verify_flags & VerifyFlags::CRL_CHECK_LEAF.bits() == 0 {
        CrlScope::Disabled
    } else if verify_flags & VerifyFlags::CRL_CHECK_CHAIN.bits()
        == VerifyFlags::CRL_CHECK_CHAIN.bits()
    {
        CrlScope::FullChain
    } else {
        CrlScope::EndEntityOnly
    }
}

impl Context {
    fn new(protocol: i32) -> Result<Self, &'static str> {
        ensure_provider();
        if !matches!(
            protocol,
            PROTOCOL_TLS
                | PROTOCOL_TLS_CLIENT
                | PROTOCOL_TLS_SERVER
                | PROTOCOL_TLSV1_2
                | PROTOCOL_TLSV1_3
        ) {
            return Err("invalid or unsupported protocol version");
        }
        let client = protocol == PROTOCOL_TLS_CLIENT;
        let server_ticketer = rustls::crypto::aws_lc_rs::Ticketer::new()
            .map_err(|_| "failed to initialize TLS session ticket encryption")?;
        Ok(Self {
            identity: NEXT_CONTEXT_IDENTITY.fetch_add(1, Ordering::Relaxed),
            protocol,
            check_hostname: client,
            verify_mode: if client { CERT_REQUIRED } else { CERT_NONE },
            verify_flags: 32768,
            options: SslOp::DEFAULT.bits(),
            minimum_version: match protocol {
                PROTOCOL_TLSV1_2 => 0x303,
                PROTOCOL_TLSV1_3 => 0x304,
                _ => -2,
            },
            maximum_version: match protocol {
                PROTOCOL_TLSV1_2 => 0x303,
                PROTOCOL_TLSV1_3 => 0x304,
                _ => -1,
            },
            alpn_protocols: Vec::new(),
            roots: rustls::RootCertStore::empty(),
            root_der: Vec::new(),
            capaths: Vec::new(),
            capath_cache: Mutex::new(None),
            crls: Vec::new(),
            cipher_suites: None,
            tls13_cipher_suites: None,
            kx_groups: None,
            server_sigalgs: None,
            certified_keys: Vec::new(),
            num_tickets: None,
            server_session_store: rustls::server::ServerSessionMemoryCache::new(256),
            server_ticketer,
            key_log_file: Arc::new(ContextKeyLog::default()),
            accept_count: AtomicUsize::new(0),
            session_hits: AtomicUsize::new(0),
        })
    }
}

/// Stable identity for `SSLSession` ownership checks.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_identity(context: *const Context) -> usize {
    unsafe { (*context).identity }
}

/// Bound the TLS 1.3 session tickets a server issues.  `None` keeps rustls'
/// default; `Some(0)` disables resumption tickets entirely.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_set_num_tickets(context: *mut Context, tickets: usize) {
    unsafe { (*context).num_tickets = Some(tickets) };
}

/// PyPy `_SSLContext.keylog_filename` replaces the context's BIO before
/// installing the keylog callback.  Rustls takes a `KeyLog` object on each
/// immutable connection config, so the context owns a shared file sink and
/// existing hooks observe setter changes through it.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_set_keylog_filename(
    context: *mut Context,
    path: Option<&std::path::Path>,
) -> NativeResult<()> {
    unsafe { &*context }
        .key_log_file
        .set_path(path)
        .map_err(io_error)
}

#[inline(never)]
pub unsafe fn context_session_stats(context: *const Context) -> (usize, usize) {
    let context = unsafe { &*context };
    (
        context.accept_count.load(Ordering::Relaxed),
        context.session_hits.load(Ordering::Relaxed),
    )
}

#[inline(never)]
pub fn context_new(protocol: i32) -> Result<*mut Context, &'static str> {
    Context::new(protocol).map(|context| Box::into_raw(Box::new(context)))
}

/// # Safety
/// `context` must be null or a pointer returned by [`context_new`] that has not
/// already been freed.
#[inline(never)]
pub unsafe fn context_free(context: *mut Context) {
    if !context.is_null() {
        unsafe { drop(Box::from_raw(context)) };
    }
}

macro_rules! context_scalar {
    ($get:ident, $set:ident, $field:ident, $ty:ty) => {
        #[inline(never)]
        pub unsafe fn $get(context: *const Context) -> $ty {
            unsafe { (*context).$field }
        }

        #[inline(never)]
        pub unsafe fn $set(context: *mut Context, value: $ty) {
            unsafe { (*context).$field = value };
        }
    };
}

context_scalar!(context_protocol, context_set_protocol, protocol, i32);
context_scalar!(
    context_check_hostname,
    context_set_check_hostname,
    check_hostname,
    bool
);
context_scalar!(
    context_verify_mode,
    context_set_verify_mode,
    verify_mode,
    i32
);
context_scalar!(
    context_verify_flags,
    context_set_verify_flags,
    verify_flags,
    i32
);
context_scalar!(context_options, context_set_options, options, u64);
context_scalar!(
    context_minimum_version,
    context_set_minimum_version,
    minimum_version,
    i32
);
context_scalar!(
    context_maximum_version,
    context_set_maximum_version,
    maximum_version,
    i32
);

/// Parse `[len][proto]...` the way `ssl.py` hands it to `_set_alpn_protocols`.
#[inline(never)]
pub fn parse_length_prefixed_alpn(data: &[u8]) -> Result<Vec<Vec<u8>>, &'static str> {
    host_ssl::parse_length_prefixed_alpn(data).map_err(|_| "invalid ALPN protocol list")
}

/// Store the wire-format ALPN list after the interpreter has validated it.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_set_alpn(context: *mut Context, protocols: Vec<Vec<u8>>) {
    unsafe { (*context).alpn_protocols = protocols };
}

/// Add DER trust anchors, returning `(accepted, rejected)`.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_add_roots(
    context: *mut Context,
    certs: Vec<CertificateDer<'static>>,
) -> (usize, usize) {
    let context = unsafe { &mut *context };
    let mut accepted = 0;
    let mut rejected = 0;
    for cert in certs {
        match context.add_root(cert.as_ref().to_vec()) {
            Ok(true) => accepted += 1,
            Ok(false) => {}
            Err(_) => rejected += 1,
        }
    }
    (accepted, rejected)
}

pub type NativeResult<T> = Result<T, (i32, String)>;

fn io_error(error: std::io::Error) -> (i32, String) {
    // PyPy `_SSLContext.keylog_filename` raises the errno returned by
    // `BIO_new_file`, and CPython `_Py_GetErrorHandler` likewise exposes a C
    // errno to the OSError subclass machinery.  Rust's `raw_os_error()` is a
    // Win32 status on Windows, so normalise the kinds used by SSL file I/O to
    // their POSIX errno before crossing the native/interpreter boundary.
    #[cfg(windows)]
    let errno = match error.kind() {
        std::io::ErrorKind::NotFound => libc::ENOENT,
        std::io::ErrorKind::AlreadyExists => libc::EEXIST,
        std::io::ErrorKind::PermissionDenied => libc::EACCES,
        std::io::ErrorKind::Interrupted => libc::EINTR,
        std::io::ErrorKind::InvalidInput => libc::EINVAL,
        _ => error.raw_os_error().unwrap_or(libc::EIO),
    };
    #[cfg(not(windows))]
    let errno = error.raw_os_error().unwrap_or(-1);

    (errno, error.to_string())
}

fn pem_error(message: impl std::fmt::Display) -> (i32, String) {
    (0, format!("[SSL] PEM routines: {message}"))
}

impl Context {
    /// Add one trust anchor while retaining its full DER exactly once.  The
    /// full certificate is needed for `get_ca_certs`; rustls intentionally
    /// stores only the parsed trust-anchor projection.
    fn add_root(&mut self, der: Vec<u8>) -> NativeResult<bool> {
        if self.root_der.iter().any(|known| known == &der) {
            return Ok(false);
        }
        self.roots
            .add(CertificateDer::from(der.clone()))
            .map_err(pem_error)?;
        self.root_der.push(der);
        Ok(true)
    }
}

fn read_pem_certificates(data: &[u8]) -> NativeResult<Vec<CertificateDer<'static>>> {
    #[cfg(feature = "host_env")]
    {
        let certs = host_ssl::cert::read_certificates(data).map_err(pem_error)?;
        if certs.is_empty() {
            return Err(pem_error("no start line"));
        }
        Ok(certs.into_iter().map(CertificateDer::from).collect())
    }
    #[cfg(not(feature = "host_env"))]
    {
        let mut cursor = Cursor::new(data);
        let certs = rustls_pemfile::certs(&mut cursor)
            .collect::<Result<Vec<_>, _>>()
            .map_err(pem_error)?;
        if certs.is_empty() {
            return Err(pem_error("no start line"));
        }
        Ok(certs)
    }
}

/// An encrypted private key, in whichever container the file uses.
enum EncryptedKey {
    /// PKCS#8 `EncryptedPrivateKeyInfo`, either DER or its PEM armouring.
    Pkcs8(Vec<u8>),
    /// RFC 1421 `Proc-Type: 4,ENCRYPTED`.  The password callback still has to
    /// run so the eventual failure names the container rather than reporting a
    /// missing key.
    Legacy,
}

/// True for an RFC 1421 header line marking the following key as encrypted.
/// The space after the colon is conventional, not required.
fn has_legacy_encryption_header(data: &[u8]) -> bool {
    String::from_utf8_lossy(data).lines().any(|line| {
        line.strip_prefix("Proc-Type:")
            .is_some_and(|value| value.split_whitespace().collect::<String>() == "4,ENCRYPTED")
    })
}

/// Recognize an encrypted private key from the parsed container rather than
/// from a scan for header text, so DER `EncryptedPrivateKeyInfo` files are
/// classified the same way as their PEM form.
fn encrypted_private_key(data: &[u8]) -> NativeResult<Option<EncryptedKey>> {
    use der::SecretDocument;
    use pkcs8::EncryptedPrivateKeyInfoRef;

    let pem = String::from_utf8_lossy(data);
    if let Some(start) = pem.find("-----BEGIN ENCRYPTED PRIVATE KEY-----") {
        let tail = &pem[start..];
        let end_marker = "-----END ENCRYPTED PRIVATE KEY-----";
        let end = tail
            .find(end_marker)
            .ok_or_else(|| pem_error("unterminated encrypted private key"))?
            + end_marker.len();
        let (_, document) = SecretDocument::from_pem(&tail[..end])
            .map_err(|error| pem_error(format!("bad encrypted private key: {error}")))?;
        return Ok(Some(EncryptedKey::Pkcs8(document.as_bytes().to_vec())));
    }
    if has_legacy_encryption_header(data) {
        return Ok(Some(EncryptedKey::Legacy));
    }
    if EncryptedPrivateKeyInfoRef::try_from(data).is_ok() {
        return Ok(Some(EncryptedKey::Pkcs8(data.to_vec())));
    }
    Ok(None)
}

/// Either the parsed key, or the fact that one cannot be produced without a
/// password.
enum PrivateKey {
    Key(PrivateKeyDer<'static>),
    PasswordRequired,
}

fn read_private_key(data: &[u8], password: Option<&[u8]>) -> NativeResult<PrivateKey> {
    if let Some(encrypted) = encrypted_private_key(data)? {
        let Some(password) = password else {
            return Ok(PrivateKey::PasswordRequired);
        };
        let EncryptedKey::Pkcs8(der) = encrypted else {
            return Err(pem_error("unsupported encrypted private key format"));
        };
        let encrypted = pkcs8::EncryptedPrivateKeyInfoRef::try_from(der.as_slice())
            .map_err(|error| pem_error(format!("bad encrypted private key: {error}")))?;
        let decrypted = encrypted
            .decrypt(password)
            .map_err(|error| pem_error(format!("bad decrypt: {error}")))?;
        return Ok(PrivateKey::Key(PrivateKeyDer::Pkcs8(
            PrivatePkcs8KeyDer::from(decrypted.as_bytes().to_vec()),
        )));
    }

    rustls_pemfile::private_key(&mut Cursor::new(data))
        .map_err(pem_error)?
        .ok_or_else(|| pem_error("no private key found"))
        .map(PrivateKey::Key)
}

/// Load and validate the context's certificate/private-key pair.
///
/// The replacement is committed only after parsing, provider key loading, and
/// public/private key matching all succeed, so concurrent connection creation
/// never observes a half-updated pair.
///
/// Returns `false` when the key turned out to be encrypted and no password was
/// supplied; the caller obtains one and calls again.  Discovering that from the
/// parse keeps an irrelevant password callback from running for an unencrypted
/// key without pre-scanning the file for header text.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_load_cert_chain(
    context: *mut Context,
    cert_path: &std::path::Path,
    key_path: &std::path::Path,
    password: Option<&[u8]>,
) -> NativeResult<bool> {
    ensure_provider();
    let cert_data = std::fs::read(cert_path).map_err(io_error)?;
    let certs = read_pem_certificates(&cert_data)?;
    let key_data = std::fs::read(key_path).map_err(io_error)?;
    let key = match read_private_key(&key_data, password)? {
        PrivateKey::Key(key) => key,
        PrivateKey::PasswordRequired => return Ok(false),
    };
    let signing_key = rustls::crypto::aws_lc_rs::sign::any_supported_type(&key)
        .map_err(|error| pem_error(format!("unsupported private key: {error}")))?;
    let certified = CertifiedKey::new(certs, signing_key);
    certified.keys_match().map_err(|_| {
        (
            0,
            "[SSL: KEY_VALUES_MISMATCH] key values mismatch".to_string(),
        )
    })?;
    let context = unsafe { &mut *context };
    let algorithm = certified.key.algorithm();
    let certified = Arc::new(certified);
    if let Some(existing) = context
        .certified_keys
        .iter_mut()
        .find(|known| known.key.algorithm() == algorithm)
    {
        *existing = certified;
    } else {
        context.certified_keys.push(certified);
    }
    Ok(true)
}

fn parse_concatenated_der(mut data: &[u8]) -> NativeResult<Vec<Vec<u8>>> {
    let mut certs = Vec::new();
    while !data.is_empty() {
        let before = data.len();
        let (remaining, _) = x509_parser::parse_x509_certificate(data)
            .map_err(|error| pem_error(format!("not enough data: {error}")))?;
        let consumed = before - remaining.len();
        if consumed == 0 {
            return Err(pem_error("not enough data"));
        }
        certs.push(data[..consumed].to_vec());
        data = remaining;
    }
    Ok(certs)
}

/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_load_verify_file(
    context: *mut Context,
    path: &std::path::Path,
) -> NativeResult<usize> {
    let data = std::fs::read(path).map_err(io_error)?;
    let items = rustls_pemfile::read_all(&mut Cursor::new(data))
        .collect::<Result<Vec<_>, _>>()
        .map_err(pem_error)?;
    let mut certificates = Vec::new();
    let mut crls = Vec::new();
    for item in items {
        match item {
            rustls_pemfile::Item::X509Certificate(cert) => {
                certificates.push(cert.as_ref().to_vec())
            }
            rustls_pemfile::Item::Crl(crl) => crls.push(crl),
            _ => {}
        }
    }
    if certificates.is_empty() && crls.is_empty() {
        return Err(pem_error("no start line"));
    }
    let added = unsafe { context_add_verify_der(context, certificates.into_iter())? };
    let context = unsafe { &mut *context };
    for crl in crls {
        if !context
            .crls
            .iter()
            .any(|known| known.as_ref() == crl.as_ref())
        {
            context.crls.push(crl);
        }
    }
    Ok(added)
}

/// Register an OpenSSL-style hashed certificate directory for lazy lookup.
/// The directory has already been validated by the interpreter boundary.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_add_verify_dir(context: *mut Context, path: &std::path::Path) {
    let context = unsafe { &mut *context };
    let path = path.to_path_buf();
    if !context.capaths.iter().any(|known| known == &path) {
        context.capaths.push(path);
        *context.capath_cache.lock() = None;
    }
}

/// Publish the trust anchor selected from a lazy `capath` lookup.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_add_verified_root(context: *mut Context, der: Vec<u8>) -> NativeResult<bool> {
    unsafe { (&mut *context).add_root(der) }
}

/// Load PEM text or one/more concatenated DER certificates.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_load_verify_data(
    context: *mut Context,
    data: &[u8],
    pem: bool,
) -> NativeResult<usize> {
    let certs: Vec<Vec<u8>> = if pem {
        read_pem_certificates(data)?
            .into_iter()
            .map(|cert| cert.as_ref().to_vec())
            .collect()
    } else {
        parse_concatenated_der(data)?
    };
    unsafe { context_add_verify_der(context, certs.into_iter()) }
}

unsafe fn context_add_verify_der(
    context: *mut Context,
    certs: impl Iterator<Item = Vec<u8>>,
) -> NativeResult<usize> {
    let context = unsafe { &mut *context };
    let mut added = 0;
    for der in certs {
        added += usize::from(context.add_root(der)?);
    }
    Ok(added)
}

/// The trust file and directory this platform actually carries, for
/// `_ssl.get_default_verify_paths()`.  Reporting the historical
/// `/etc/ssl/cert.pem` pair everywhere names a store most Linux distributions
/// do not have.  The environment variables that shadow these values are
/// applied by `ssl.py`, so the compiled-in defaults are returned unshadowed.
#[cfg(feature = "host_env")]
#[inline(never)]
pub fn default_verify_paths() -> (String, String) {
    rustpython_host_env::native_certs::default_verify_paths()
}

/// The host-less build has no filesystem seam through which to discover a
/// platform store, so retain OpenSSL's conventional compiled-in pair.
#[cfg(not(feature = "host_env"))]
#[inline(never)]
pub fn default_verify_paths() -> (String, String) {
    (
        "/etc/ssl/cert.pem".to_string(),
        "/etc/ssl/certs".to_string(),
    )
}

/// Load trust anchors from the platform provider (Keychain on macOS, native
/// certificate stores on Windows, discovered bundle/directories on Unix).
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_load_native_roots(context: *mut Context) -> NativeResult<usize> {
    #[cfg(feature = "host_env")]
    let result = rustpython_host_env::native_certs::load();
    #[cfg(not(feature = "host_env"))]
    let result = rustls_native_certs::load_native_certs();

    #[cfg(feature = "host_env")]
    let certs = result.certs.into_iter();
    #[cfg(not(feature = "host_env"))]
    let certs = result.certs.into_iter().map(|cert| cert.as_ref().to_vec());
    let added = unsafe { context_add_verify_der(context, certs)? };
    if added == 0 && !result.errors.is_empty() {
        return Err(pem_error(&result.errors[0]));
    }
    Ok(added)
}

fn certificate_is_ca(der: &[u8]) -> bool {
    #[cfg(feature = "host_env")]
    {
        host_ssl::is_ca_certificate(der)
    }
    #[cfg(not(feature = "host_env"))]
    {
        x509_parser::certificate::X509Certificate::from_der(der)
            .ok()
            .map(|(_, cert)| {
                cert.basic_constraints()
                    .ok()
                    .flatten()
                    .is_some_and(|constraints| constraints.value.ca)
                    // OpenSSL's X509_check_ca retains its legacy rule for a
                    // self-issued X.509v1 trust anchor without BasicConstraints.
                    || (cert.version().0 == 0 && cert.subject() == cert.issuer())
            })
            .unwrap_or(false)
    }
}

/// Return `(all_x509, ca_x509)`.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_cert_store_stats(context: *const Context) -> (usize, usize) {
    let context = unsafe { &*context };
    (
        context.root_der.len(),
        context
            .root_der
            .iter()
            .filter(|der| certificate_is_ca(der))
            .count(),
    )
}

/// Full DER for CA certificates only.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_ca_certs(context: *const Context) -> Vec<Vec<u8>> {
    unsafe { &*context }
        .root_der
        .iter()
        .filter(|der| certificate_is_ca(der))
        .cloned()
        .collect()
}

#[cfg(feature = "host_env")]
type DistinguishedName = host_ssl::cert::DistinguishedName;
#[cfg(not(feature = "host_env"))]
type DistinguishedName = Vec<Vec<(String, String)>>;

#[cfg(feature = "host_env")]
use host_ssl::cert::SubjectAlternativeName;
#[cfg(not(feature = "host_env"))]
struct SubjectAlternativeName {
    kind: &'static str,
    value: String,
    directory_name: DistinguishedName,
}

/// Owned projection of the X.509 fields exposed by `_test_decode_cert()`
/// and by `SSLContext.get_ca_certs()`. Keeping this projection native
/// prevents x509-parser's borrowed type graph from entering the translated
/// interpreter.
#[cfg(feature = "host_env")]
pub type DecodedCertificate = host_ssl::DecodedCertificate;

/// Owned projection of the X.509 fields exposed by `_test_decode_cert()`
/// and by `SSLContext.get_ca_certs()`. Keeping this projection native
/// prevents x509-parser's borrowed type graph from entering the translated
/// interpreter.
#[cfg(not(feature = "host_env"))]
pub struct DecodedCertificate {
    issuer: DistinguishedName,
    subject: DistinguishedName,
    not_after: String,
    not_before: String,
    serial_number: String,
    version: i32,
    ocsp: Vec<String>,
    ca_issuers: Vec<String>,
    crl_distribution_points: Vec<String>,
    subject_alt_names: Vec<SubjectAlternativeName>,
}

#[cfg(not(feature = "host_env"))]
fn oid_attribute_name(oid: &str) -> String {
    match oid {
        "2.5.4.3" => "commonName".to_string(),
        "2.5.4.6" => "countryName".to_string(),
        "2.5.4.7" => "localityName".to_string(),
        "2.5.4.8" => "stateOrProvinceName".to_string(),
        "2.5.4.10" => "organizationName".to_string(),
        "2.5.4.11" => "organizationalUnitName".to_string(),
        "1.2.840.113549.1.9.1" => "emailAddress".to_string(),
        _ => oid.to_string(),
    }
}

#[cfg(not(feature = "host_env"))]
fn decode_name(name: &x509_parser::x509::X509Name<'_>) -> DistinguishedName {
    name.iter()
        .map(|rdn| {
            rdn.iter()
                .map(|attribute| {
                    let oid = attribute.attr_type().to_id_string();
                    let value = attribute
                        .attr_value()
                        .as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|_| {
                            String::from_utf8_lossy(attribute.attr_value().data).into_owned()
                        });
                    (oid_attribute_name(&oid), value)
                })
                .collect()
        })
        .collect()
}

#[cfg(not(feature = "host_env"))]
fn format_certificate_time(value: &x509_parser::time::ASN1Time) -> String {
    let date = value.to_datetime();
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    format!(
        "{} {:>2} {:02}:{:02}:{:02} {:04} GMT",
        MONTHS[date.month() as usize - 1],
        date.day(),
        date.hour(),
        date.minute(),
        date.second(),
        date.year(),
    )
}

#[cfg(not(feature = "host_env"))]
fn format_ip_address(ip: &[u8]) -> String {
    match ip.len() {
        4 => format!("{}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3]),
        16 => ip
            .chunks_exact(2)
            .map(|part| format!("{:X}", u16::from_be_bytes([part[0], part[1]])))
            .collect::<Vec<_>>()
            .join(":"),
        _ => "<invalid>".to_string(),
    }
}

#[cfg(feature = "host_env")]
fn decode_certificate(der: &[u8]) -> NativeResult<DecodedCertificate> {
    host_ssl::decode_certificate(der).map_err(pem_error)
}

#[cfg(not(feature = "host_env"))]
fn decode_certificate(der: &[u8]) -> NativeResult<DecodedCertificate> {
    use x509_parser::extensions::{DistributionPointName, GeneralName, ParsedExtension};
    use x509_parser::oid_registry::{
        OID_PKIX_AUTHORITY_INFO_ACCESS, OID_X509_EXT_CRL_DISTRIBUTION_POINTS,
    };

    let (_, cert) = x509_parser::parse_x509_certificate(der)
        .map_err(|error| pem_error(format!("failed to parse certificate: {error}")))?;
    let mut decoded = DecodedCertificate {
        issuer: decode_name(cert.issuer()),
        subject: decode_name(cert.subject()),
        not_after: format_certificate_time(&cert.validity().not_after),
        not_before: format_certificate_time(&cert.validity().not_before),
        serial_number: {
            let mut serial = cert.serial.to_str_radix(16).to_uppercase();
            if serial.len() % 2 != 0 {
                serial.insert(0, '0');
            }
            serial
        },
        version: cert.version().0 as i32 + 1,
        ocsp: Vec::new(),
        ca_issuers: Vec::new(),
        crl_distribution_points: Vec::new(),
        subject_alt_names: Vec::new(),
    };

    if let Ok(extensions) = cert.tbs_certificate.extensions_map() {
        if let Some(extension) = extensions.get(&OID_PKIX_AUTHORITY_INFO_ACCESS)
            && let ParsedExtension::AuthorityInfoAccess(access) = extension.parsed_extension()
        {
            for description in &access.accessdescs {
                if let GeneralName::URI(uri) = &description.access_location {
                    match description.access_method.to_id_string().as_str() {
                        "1.3.6.1.5.5.7.48.1" => decoded.ocsp.push((*uri).to_string()),
                        "1.3.6.1.5.5.7.48.2" => decoded.ca_issuers.push((*uri).to_string()),
                        _ => {}
                    }
                }
            }
        }
        if let Some(extension) = extensions.get(&OID_X509_EXT_CRL_DISTRIBUTION_POINTS)
            && let ParsedExtension::CRLDistributionPoints(points) = extension.parsed_extension()
        {
            for point in &points.points {
                if let Some(DistributionPointName::FullName(names)) = &point.distribution_point {
                    for name in names {
                        if let GeneralName::URI(uri) = name {
                            decoded.crl_distribution_points.push((*uri).to_string());
                        }
                    }
                }
            }
        }
    }

    if let Ok(Some(extension)) = cert.subject_alternative_name() {
        for name in &extension.value.general_names {
            let entry = match name {
                GeneralName::DNSName(value) => SubjectAlternativeName {
                    kind: "DNS",
                    value: (*value).to_string(),
                    directory_name: Vec::new(),
                },
                GeneralName::IPAddress(value) => SubjectAlternativeName {
                    kind: "IP Address",
                    value: format_ip_address(value),
                    directory_name: Vec::new(),
                },
                GeneralName::RFC822Name(value) => SubjectAlternativeName {
                    kind: "email",
                    value: (*value).to_string(),
                    directory_name: Vec::new(),
                },
                GeneralName::URI(value) => SubjectAlternativeName {
                    kind: "URI",
                    value: (*value).to_string(),
                    directory_name: Vec::new(),
                },
                GeneralName::OtherName(_, _) => SubjectAlternativeName {
                    kind: "othername",
                    value: "<unsupported>".to_string(),
                    directory_name: Vec::new(),
                },
                GeneralName::DirectoryName(value) => SubjectAlternativeName {
                    kind: "DirName",
                    value: String::new(),
                    directory_name: decode_name(value),
                },
                GeneralName::RegisteredID(value) => SubjectAlternativeName {
                    kind: "Registered ID",
                    value: value.to_id_string(),
                    directory_name: Vec::new(),
                },
                _ => continue,
            };
            decoded.subject_alt_names.push(entry);
        }
    }
    Ok(decoded)
}

#[inline(never)]
pub fn certificate_decode_der(der: &[u8]) -> NativeResult<*mut DecodedCertificate> {
    decode_certificate(der).map(|cert| Box::into_raw(Box::new(cert)))
}

#[inline(never)]
pub fn certificate_decode_file(path: &std::path::Path) -> NativeResult<*mut DecodedCertificate> {
    let data = std::fs::read(path).map_err(io_error)?;
    let certs = read_pem_certificates(&data)?;
    certificate_decode_der(certs[0].as_ref())
}

fn push_rfc2253_value(output: &mut String, value: &[u8]) {
    for (index, byte) in value.iter().copied().enumerate() {
        let must_hex_escape = !(0x20..=0x7e).contains(&byte) || byte == 0;
        if must_hex_escape {
            use std::fmt::Write;
            let _ = write!(output, "\\{byte:02X}");
            continue;
        }
        let leading = index == 0 && (byte == b' ' || byte == b'#');
        let trailing = index + 1 == value.len() && byte == b' ';
        if leading || trailing || matches!(byte, b',' | b'+' | b'"' | b'\\' | b'<' | b'>' | b';') {
            output.push('\\');
        }
        output.push(byte as char);
    }
}

/// RFC 2253 subject rendering used by PyPy's `Certificate.__repr__`.
#[inline(never)]
pub fn certificate_subject_rfc2253(der: &[u8]) -> NativeResult<String> {
    let (_, certificate) = x509_parser::parse_x509_certificate(der)
        .map_err(|error| pem_error(format!("unable to decode certificate: {error}")))?;
    let mut output = String::new();
    let rdns: Vec<_> = certificate.subject().iter_rdn().collect();
    for (rdn_index, rdn) in rdns.into_iter().rev().enumerate() {
        if rdn_index != 0 {
            output.push(',');
        }
        for (attribute_index, attribute) in rdn.iter().enumerate() {
            if attribute_index != 0 {
                output.push('+');
            }
            let name = x509_parser::objects::oid2abbrev(
                attribute.attr_type(),
                x509_parser::objects::oid_registry(),
            )
            .map(str::to_owned)
            .unwrap_or_else(|_| attribute.attr_type().to_id_string());
            output.push_str(&name);
            output.push('=');
            push_rfc2253_value(&mut output, attribute.as_slice());
        }
    }
    Ok(output)
}

/// OpenSSL's `X509_subject_name_hash`: the low 32 bits of SHA-1 over the
/// canonical subject Name encoding, interpreted in little-endian order.
fn der_length(output: &mut Vec<u8>, length: usize) {
    if length < 128 {
        output.push(length as u8);
        return;
    }
    let bytes = length.to_be_bytes();
    let first = bytes
        .iter()
        .position(|byte| *byte != 0)
        .unwrap_or(bytes.len() - 1);
    output.push(0x80 | (bytes.len() - first) as u8);
    output.extend_from_slice(&bytes[first..]);
}

fn der_value(tag: u8, contents: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(contents.len() + 6);
    output.push(tag);
    der_length(&mut output, contents.len());
    output.extend_from_slice(contents);
    output
}

fn canonical_string_value(
    attribute: &x509_parser::x509::AttributeTypeAndValue<'_>,
) -> (u8, Vec<u8>) {
    let tag = attribute.attr_value().tag().0;
    let input = attribute.as_slice();
    let decoded = match tag {
        // UTF8String plus the single-byte ASN.1 character string types in
        // OpenSSL's ASN1_MASK_CANON.
        12 | 18 | 19 | 20 | 22 | 26 => {
            if tag == 20 {
                input
                    .iter()
                    .flat_map(|byte| char::from(*byte).to_string().into_bytes())
                    .collect()
            } else {
                input.to_vec()
            }
        }
        // UniversalString and BMPString.
        28 => input
            .as_chunks::<4>()
            .0
            .iter()
            .filter_map(|bytes| {
                char::from_u32(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            })
            .flat_map(|value| value.to_string().into_bytes())
            .collect(),
        30 => char::decode_utf16(
            input
                .as_chunks::<2>()
                .0
                .iter()
                .map(|bytes| u16::from_be_bytes([bytes[0], bytes[1]])),
        )
        .map(|value| value.unwrap_or(char::REPLACEMENT_CHARACTER))
        .collect::<String>()
        .into_bytes(),
        // Types outside ASN1_MASK_CANON are copied without conversion.
        _ => return (tag as u8, input.to_vec()),
    };
    let is_space = |byte: &u8| matches!(*byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r');
    let mut output = Vec::with_capacity(decoded.len());
    for word in decoded.split(is_space).filter(|part| !part.is_empty()) {
        if !output.is_empty() {
            output.push(b' ');
        }
        output.extend(word.iter().map(u8::to_ascii_lowercase));
    }
    (12, output)
}

fn canonical_name_der(name: &x509_parser::x509::X509Name<'_>) -> Vec<u8> {
    let mut output = Vec::new();
    for rdn in name.iter_rdn() {
        let mut entries = Vec::new();
        for attribute in rdn.iter() {
            let mut entry = der_value(6, attribute.attr_type().as_bytes());
            let (tag, value) = canonical_string_value(attribute);
            entry.extend_from_slice(&der_value(tag, &value));
            entries.push(der_value(0x30, &entry));
        }
        // ASN.1 SET OF canonical ordering is lexicographic over each complete
        // DER element. OpenSSL's X509_NAME_ENTRIES encoder applies this sort.
        entries.sort();
        let contents: Vec<u8> = entries.into_iter().flatten().collect();
        output.extend_from_slice(&der_value(0x31, &contents));
    }
    output
}

#[inline(never)]
pub fn certificate_subject_hash(der: &[u8]) -> NativeResult<i64> {
    let (_, certificate) = x509_parser::parse_x509_certificate(der)
        .map_err(|error| pem_error(format!("unable to decode certificate: {error}")))?;
    let digest = Sha1::digest(canonical_name_der(certificate.subject()));
    Ok(u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) as i64)
}

/// # Safety
/// `cert` must be null or a live pointer returned by a certificate decoder.
#[inline(never)]
pub unsafe fn certificate_free(cert: *mut DecodedCertificate) {
    if !cert.is_null() {
        unsafe { drop(Box::from_raw(cert)) };
    }
}

macro_rules! certificate_string {
    ($name:ident, $field:ident) => {
        #[inline(never)]
        pub unsafe fn $name(cert: *const DecodedCertificate) -> String {
            unsafe { (*cert).$field.clone() }
        }
    };
}

certificate_string!(certificate_not_after, not_after);
certificate_string!(certificate_not_before, not_before);
certificate_string!(certificate_serial_number, serial_number);

#[inline(never)]
pub unsafe fn certificate_version(cert: *const DecodedCertificate) -> i32 {
    unsafe { (*cert).version }
}

fn decoded_name(cert: &DecodedCertificate, subject: bool) -> &DistinguishedName {
    if subject { &cert.subject } else { &cert.issuer }
}

#[inline(never)]
pub unsafe fn certificate_name_rdn_count(cert: *const DecodedCertificate, subject: bool) -> usize {
    decoded_name(unsafe { &*cert }, subject).len()
}

#[inline(never)]
pub unsafe fn certificate_name_attribute_count(
    cert: *const DecodedCertificate,
    subject: bool,
    rdn: usize,
) -> usize {
    decoded_name(unsafe { &*cert }, subject)[rdn].len()
}

#[inline(never)]
pub unsafe fn certificate_name_attribute_key(
    cert: *const DecodedCertificate,
    subject: bool,
    rdn: usize,
    attribute: usize,
) -> String {
    decoded_name(unsafe { &*cert }, subject)[rdn][attribute]
        .0
        .clone()
}

#[inline(never)]
pub unsafe fn certificate_name_attribute_value(
    cert: *const DecodedCertificate,
    subject: bool,
    rdn: usize,
    attribute: usize,
) -> String {
    decoded_name(unsafe { &*cert }, subject)[rdn][attribute]
        .1
        .clone()
}

fn certificate_urls(cert: &DecodedCertificate, kind: i32) -> &Vec<String> {
    match kind {
        0 => &cert.ocsp,
        1 => &cert.ca_issuers,
        _ => &cert.crl_distribution_points,
    }
}

#[inline(never)]
pub unsafe fn certificate_url_count(cert: *const DecodedCertificate, kind: i32) -> usize {
    certificate_urls(unsafe { &*cert }, kind).len()
}

#[inline(never)]
pub unsafe fn certificate_url(cert: *const DecodedCertificate, kind: i32, index: usize) -> String {
    certificate_urls(unsafe { &*cert }, kind)[index].clone()
}

#[inline(never)]
pub unsafe fn certificate_san_count(cert: *const DecodedCertificate) -> usize {
    unsafe { (*cert).subject_alt_names.len() }
}

fn decoded_san(cert: &DecodedCertificate, index: usize) -> &SubjectAlternativeName {
    &cert.subject_alt_names[index]
}

#[inline(never)]
pub unsafe fn certificate_san_kind(cert: *const DecodedCertificate, index: usize) -> &'static str {
    decoded_san(unsafe { &*cert }, index).kind
}

#[inline(never)]
pub unsafe fn certificate_san_value(cert: *const DecodedCertificate, index: usize) -> String {
    decoded_san(unsafe { &*cert }, index).value.clone()
}

#[inline(never)]
pub unsafe fn certificate_san_directory_rdn_count(
    cert: *const DecodedCertificate,
    index: usize,
) -> usize {
    decoded_san(unsafe { &*cert }, index).directory_name.len()
}

#[inline(never)]
pub unsafe fn certificate_san_directory_attribute_count(
    cert: *const DecodedCertificate,
    index: usize,
    rdn: usize,
) -> usize {
    decoded_san(unsafe { &*cert }, index).directory_name[rdn].len()
}

#[inline(never)]
pub unsafe fn certificate_san_directory_attribute_key(
    cert: *const DecodedCertificate,
    index: usize,
    rdn: usize,
    attribute: usize,
) -> String {
    decoded_san(unsafe { &*cert }, index).directory_name[rdn][attribute]
        .0
        .clone()
}

#[inline(never)]
pub unsafe fn certificate_san_directory_attribute_value(
    cert: *const DecodedCertificate,
    index: usize,
    rdn: usize,
    attribute: usize,
) -> String {
    decoded_san(unsafe { &*cert }, index).directory_name[rdn][attribute]
        .1
        .clone()
}

/// One OpenSSL object, as `_ssl.txt2obj` / `nid2obj` report it.
#[derive(Clone, Copy, Debug)]
pub struct OidInfo {
    pub nid: i32,
    pub short_name: &'static str,
    pub long_name: &'static str,
    pub oid: Option<&'static str>,
}

fn oid_info(entry: &'static host_ssl::oid::OidEntry) -> OidInfo {
    OidInfo {
        nid: entry.nid,
        short_name: entry.short_name,
        long_name: entry.long_name,
        oid: entry.oid_string(),
    }
}

#[inline(never)]
pub fn oid_by_nid(nid: i32) -> Option<OidInfo> {
    host_ssl::oid::find_by_nid(nid).map(oid_info)
}

#[inline(never)]
pub fn oid_by_oid_string(oid: &str) -> Option<OidInfo> {
    host_ssl::oid::find_by_oid_string(oid).map(oid_info)
}

#[inline(never)]
pub fn oid_by_name(name: &str) -> Option<OidInfo> {
    host_ssl::oid::find_by_name(name).map(oid_info)
}

#[derive(Clone, Copy)]
struct CipherInfo {
    name: &'static str,
    protocol: &'static str,
    bits: i32,
    aead: bool,
    symmetric: &'static str,
    digest: &'static str,
    kea: &'static str,
    auth: &'static str,
}

const CIPHERS: &[CipherInfo] = &[
    CipherInfo {
        name: "TLS_AES_128_GCM_SHA256",
        protocol: "TLSv1.3",
        bits: 128,
        aead: true,
        symmetric: "aes-128-gcm",
        digest: "sha256",
        kea: "kx-any",
        auth: "auth-any",
    },
    CipherInfo {
        name: "TLS_AES_256_GCM_SHA384",
        protocol: "TLSv1.3",
        bits: 256,
        aead: true,
        symmetric: "aes-256-gcm",
        digest: "sha384",
        kea: "kx-any",
        auth: "auth-any",
    },
    CipherInfo {
        name: "TLS_CHACHA20_POLY1305_SHA256",
        protocol: "TLSv1.3",
        bits: 256,
        aead: true,
        symmetric: "chacha20-poly1305",
        digest: "sha256",
        kea: "kx-any",
        auth: "auth-any",
    },
    CipherInfo {
        name: "ECDHE-ECDSA-AES128-GCM-SHA256",
        protocol: "TLSv1.2",
        bits: 128,
        aead: true,
        symmetric: "aes-128-gcm",
        digest: "sha256",
        kea: "kx-ecdhe",
        auth: "auth-ecdsa",
    },
    CipherInfo {
        name: "ECDHE-ECDSA-AES256-GCM-SHA384",
        protocol: "TLSv1.2",
        bits: 256,
        aead: true,
        symmetric: "aes-256-gcm",
        digest: "sha384",
        kea: "kx-ecdhe",
        auth: "auth-ecdsa",
    },
    CipherInfo {
        name: "ECDHE-RSA-AES128-GCM-SHA256",
        protocol: "TLSv1.2",
        bits: 128,
        aead: true,
        symmetric: "aes-128-gcm",
        digest: "sha256",
        kea: "kx-ecdhe",
        auth: "auth-rsa",
    },
    CipherInfo {
        name: "ECDHE-RSA-AES256-GCM-SHA384",
        protocol: "TLSv1.2",
        bits: 256,
        aead: true,
        symmetric: "aes-256-gcm",
        digest: "sha384",
        kea: "kx-ecdhe",
        auth: "auth-rsa",
    },
];

#[inline(never)]
pub fn cipher_count() -> usize {
    CIPHERS.len()
}
#[inline(never)]
pub fn cipher_name(index: usize) -> &'static str {
    CIPHERS[index].name
}
#[inline(never)]
pub fn cipher_id(index: usize) -> i64 {
    #[cfg(feature = "host_env")]
    {
        if let Some(suite) = suite_by_openssl_name(CIPHERS[index].name) {
            return i64::from(cipher::describe(&suite).id);
        }
    }
    index as i64
}
#[inline(never)]
pub fn cipher_description(index: usize) -> String {
    #[cfg(feature = "host_env")]
    {
        if let Some(suite) = suite_by_openssl_name(CIPHERS[index].name) {
            return cipher::describe(&suite).description;
        }
    }
    CIPHERS[index].name.to_string()
}
#[inline(never)]
pub fn cipher_protocol(index: usize) -> &'static str {
    CIPHERS[index].protocol
}
#[inline(never)]
pub fn cipher_bits(index: usize) -> i32 {
    CIPHERS[index].bits
}
#[inline(never)]
pub fn cipher_aead(index: usize) -> bool {
    CIPHERS[index].aead
}
#[inline(never)]
pub fn cipher_symmetric(index: usize) -> &'static str {
    CIPHERS[index].symmetric
}
#[inline(never)]
pub fn cipher_digest(index: usize) -> &'static str {
    CIPHERS[index].digest
}
#[inline(never)]
pub fn cipher_kea(index: usize) -> &'static str {
    CIPHERS[index].kea
}
#[inline(never)]
pub fn cipher_auth(index: usize) -> &'static str {
    CIPHERS[index].auth
}

#[inline(never)]
pub fn validate_cipher_string(pattern: &str) -> Result<(), &'static str> {
    parse_cipher_string(pattern).map(|_| ())
}

#[inline(never)]
pub fn default_cipher_string() -> String {
    #[cfg(feature = "host_env")]
    {
        ensure_provider();
        cipher::default_cipher_string()
    }
    #[cfg(not(feature = "host_env"))]
    {
        "rustls default cipher suites".to_string()
    }
}

fn suite_by_openssl_name(name: &str) -> Option<rustls::SupportedCipherSuite> {
    ensure_provider();
    #[cfg(feature = "host_env")]
    {
        CryptoExt::get_ext()
            .all_ciphers_or_default()
            .iter()
            .copied()
            .find(|suite| cipher::describe(suite).name == name)
    }
    #[cfg(not(feature = "host_env"))]
    {
        rustls::crypto::aws_lc_rs::ALL_CIPHER_SUITES
            .iter()
            .copied()
            .find(|suite| openssl_cipher_name(*suite) == Some(name))
    }
}

/// OpenSSL's name for each cipher suite rustls can negotiate.
///
/// `rustls::CipherSuite` renders through `Debug`, which carries no stability
/// guarantee, and its TLS 1.2 spelling differs from OpenSSL's anyway.  Both
/// `set_ciphers()` selection and `SSLSocket.cipher()` read this table so a
/// rustls upgrade cannot silently change either.
///
/// A suite absent from this table is neither selectable nor reportable, so a
/// provider gaining one has to be named here as well as in [`CIPHERS`].
fn openssl_cipher_name(suite: rustls::SupportedCipherSuite) -> Option<&'static str> {
    #[cfg(feature = "host_env")]
    {
        Some(cipher::describe(&suite).name)
    }
    #[cfg(not(feature = "host_env"))]
    {
        use rustls::CipherSuite;
        Some(match suite.suite() {
            CipherSuite::TLS13_AES_128_GCM_SHA256 => "TLS_AES_128_GCM_SHA256",
            CipherSuite::TLS13_AES_256_GCM_SHA384 => "TLS_AES_256_GCM_SHA384",
            CipherSuite::TLS13_CHACHA20_POLY1305_SHA256 => "TLS_CHACHA20_POLY1305_SHA256",
            CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256 => "ECDHE-ECDSA-AES128-GCM-SHA256",
            CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384 => "ECDHE-ECDSA-AES256-GCM-SHA384",
            CipherSuite::TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256 => {
                "ECDHE-ECDSA-CHACHA20-POLY1305"
            }
            CipherSuite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 => "ECDHE-RSA-AES128-GCM-SHA256",
            CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384 => "ECDHE-RSA-AES256-GCM-SHA384",
            CipherSuite::TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256 => {
                "ECDHE-RSA-CHACHA20-POLY1305"
            }
            _ => return None,
        })
    }
}

#[cfg(not(feature = "host_env"))]
fn cipher_pattern_matches(suite: rustls::SupportedCipherSuite, pattern: &str) -> bool {
    if suite.tls13().is_some() {
        return false;
    }
    let Some(name) = openssl_cipher_name(suite) else {
        return false;
    };
    match pattern {
        "ALL" | "DEFAULT" | "HIGH" => true,
        "AES128" => name.contains("AES128"),
        "AES256" => name.contains("AES256"),
        "AESGCM" => name.contains("AES") && name.contains("GCM"),
        "CHACHA20" => name.contains("CHACHA20"),
        "ECDHE" | "KECDHE" => name.contains("ECDHE"),
        "ECDSA" | "AECDSA" => name.contains("ECDSA"),
        "RSA" | "ARSA" | "KRSA" => name.contains("RSA"),
        "NULL" | "ANULL" | "ENULL" => false,
        _ => {
            let compact_name = name.replace('_', "").replace('-', "");
            let compact_pattern = pattern.replace('_', "").replace('-', "");
            compact_name.contains(&compact_pattern)
        }
    }
}

/// Whether the cipher named by [`cipher_name`] survives this context's
/// `set_ciphers()` / `set_ciphersuites()` filters.  TLS 1.3 suites ignore
/// `SSL_CTX_set_cipher_list` and follow `SSL_CTX_set_ciphersuites`.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_cipher_enabled(context: *const Context, index: usize) -> bool {
    let context = unsafe { &*context };
    let entry = CIPHERS[index];
    if entry.protocol == "TLSv1.3" {
        return match &context.tls13_cipher_suites {
            Some(selected) => selected
                .iter()
                .any(|suite| openssl_cipher_name(*suite) == Some(entry.name)),
            None => true,
        };
    }
    let Some(selected) = &context.cipher_suites else {
        return true;
    };
    selected
        .iter()
        .any(|suite| openssl_cipher_name(*suite) == Some(entry.name))
}

fn parse_cipher_string(pattern: &str) -> Result<Vec<rustls::SupportedCipherSuite>, &'static str> {
    ensure_provider();
    #[cfg(feature = "host_env")]
    {
        let (selected, _) = cipher::CipherList::parse_to_rustls(pattern)
            .ok()
            .filter(|(suites, _)| suites.iter().any(|suite| suite.tls13().is_none()))
            .ok_or("No cipher can be selected")?;
        Ok(cipher::restore_default_tls13(
            selected,
            CryptoExt::get_ext().default_ciphers_or_provider(),
        ))
    }
    #[cfg(not(feature = "host_env"))]
    {
        let provider =
            rustls::crypto::CryptoProvider::get_default().expect("the _ssl provider is installed");
        let mut selected = Vec::new();
        let mut exclusions = Vec::new();
        for raw in pattern.split(':') {
            let token = raw.trim().to_ascii_uppercase();
            if token.is_empty() || token.starts_with('@') || token.starts_with('+') {
                continue;
            }
            if let Some(excluded) = token.strip_prefix('!') {
                exclusions.push(excluded.to_string());
                continue;
            }
            let parts: Vec<&str> = token.split('+').collect();
            for suite in &provider.cipher_suites {
                if parts
                    .iter()
                    .all(|part| cipher_pattern_matches(*suite, part))
                    && !selected
                        .iter()
                        .any(|known: &rustls::SupportedCipherSuite| known.suite() == suite.suite())
                {
                    selected.push(*suite);
                }
            }
        }
        selected.retain(|suite| {
            !exclusions
                .iter()
                .any(|pattern| cipher_pattern_matches(*suite, pattern))
        });
        if selected.is_empty() {
            Err("No cipher can be selected")
        } else {
            Ok(selected)
        }
    }
}

/// Apply OpenSSL's TLS <= 1.2 cipher-list selection. TLS 1.3 suites remain
/// provider defaults, matching `SSL_CTX_set_cipher_list` semantics.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_set_cipher_list(
    context: *mut Context,
    pattern: &str,
) -> Result<(), &'static str> {
    unsafe { (*context).cipher_suites = Some(parse_cipher_string(pattern)?) };
    Ok(())
}

/// Restrict key exchange to the curve selected by SSLContext.set_ecdh_curve.
/// OpenSSL stores this choice directly on SSL_CTX; keep the equivalent state
/// on our native Context and apply it to each immutable rustls provider built
/// from that context.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_set_ecdh_curve(
    context: *mut Context,
    curve: &str,
) -> Result<(), &'static str> {
    #[cfg(feature = "host_env")]
    let _ = cipher::kx_group_by_openssl_name(curve).ok_or("unknown elliptic curve")?;
    let group = kx_group_from_openssl_list_name(curve).ok_or("unknown elliptic curve name")?;
    unsafe { (*context).kx_groups = Some(vec![group]) };
    Ok(())
}

fn parse_ciphersuites(pattern: &str) -> Result<Vec<rustls::SupportedCipherSuite>, &'static str> {
    ensure_provider();
    let mut selected = Vec::new();
    let mut saw_token = false;
    for raw in pattern.split(':') {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        saw_token = true;
        let Some(suite) = suite_by_openssl_name(name).filter(|suite| suite.tls13().is_some())
        else {
            continue;
        };
        if !selected
            .iter()
            .any(|known: &rustls::SupportedCipherSuite| known.suite() == suite.suite())
        {
            selected.push(suite);
        }
    }
    if selected.is_empty() && (saw_token || !pattern.is_empty()) {
        return Err("No cipher suite can be selected.");
    }
    Ok(selected)
}

/// `_ssl__SSLContext_set_ciphersuites_impl`: TLS 1.3 suites only.  Unknown
/// names are skipped when at least one name is valid; an empty string clears
/// the TLS 1.3 list.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_set_ciphersuites(
    context: *mut Context,
    pattern: &str,
) -> Result<(), &'static str> {
    unsafe { (*context).tls13_cipher_suites = Some(parse_ciphersuites(pattern)?) };
    Ok(())
}

fn kx_group_from_openssl_list_name(
    name: &str,
) -> Option<&'static dyn rustls::crypto::SupportedKxGroup> {
    let mapped = match name {
        "P-256" => "prime256v1",
        "P-384" => "secp384r1",
        "P-521" => "secp521r1",
        other => other,
    };
    #[cfg(feature = "host_env")]
    {
        cipher::kx_group_by_openssl_name(mapped).or_else(|| cipher::kx_group_by_openssl_name(name))
    }
    #[cfg(not(feature = "host_env"))]
    {
        Some(match mapped {
            "prime256v1" | "secp256r1" => rustls::crypto::aws_lc_rs::kx_group::SECP256R1,
            "secp384r1" | "prime384v1" => rustls::crypto::aws_lc_rs::kx_group::SECP384R1,
            "X25519" | "x25519" => rustls::crypto::aws_lc_rs::kx_group::X25519,
            _ => return None,
        })
    }
}

fn parse_groups(
    list: &str,
) -> Result<Option<Vec<&'static dyn rustls::crypto::SupportedKxGroup>>, &'static str> {
    if list.is_empty() {
        return Ok(None);
    }
    let mut selected = Vec::new();
    let mut saw_token = false;
    for raw in list.split(':') {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        saw_token = true;
        let group = kx_group_from_openssl_list_name(name).ok_or("unrecognized group")?;
        if !selected
            .iter()
            .any(|known: &&'static dyn rustls::crypto::SupportedKxGroup| {
                known.name() == group.name()
            })
        {
            selected.push(group);
        }
    }
    if !saw_token {
        return Err("unrecognized group");
    }
    Ok(Some(selected))
}

/// `_ssl__SSLContext_set_groups_impl`: colon-separated OpenSSL group names.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_set_groups(context: *mut Context, list: &str) -> Result<(), &'static str> {
    unsafe { (*context).kx_groups = parse_groups(list)? };
    Ok(())
}

fn signature_scheme_from_openssl_name(name: &str) -> Option<rustls::SignatureScheme> {
    use rustls::SignatureScheme::*;
    Some(match name {
        "rsa_pss_rsae_sha256" => RSA_PSS_SHA256,
        "rsa_pss_rsae_sha384" => RSA_PSS_SHA384,
        "rsa_pss_rsae_sha512" => RSA_PSS_SHA512,
        "rsa_pkcs1_sha256" => RSA_PKCS1_SHA256,
        "rsa_pkcs1_sha384" => RSA_PKCS1_SHA384,
        "rsa_pkcs1_sha512" => RSA_PKCS1_SHA512,
        "ecdsa_secp256r1_sha256" => ECDSA_NISTP256_SHA256,
        "ecdsa_secp384r1_sha384" => ECDSA_NISTP384_SHA384,
        "ecdsa_secp521r1_sha512" => ECDSA_NISTP521_SHA512,
        "ed25519" => ED25519,
        "ed448" => ED448,
        "rsa_pkcs1_sha1" => RSA_PKCS1_SHA1,
        "ecdsa_sha1" => ECDSA_SHA1_Legacy,
        _ => return None,
    })
}

fn parse_sigalgs(list: &str) -> Result<Vec<rustls::SignatureScheme>, &'static str> {
    let mut selected = Vec::new();
    let mut saw_token = false;
    for raw in list.split(':') {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        saw_token = true;
        let scheme =
            signature_scheme_from_openssl_name(name).ok_or("unrecognized signature algorithm")?;
        if !selected.contains(&scheme) {
            selected.push(scheme);
        }
    }
    if !saw_token {
        return Err("unrecognized signature algorithm");
    }
    Ok(selected)
}

/// `_ssl__SSLContext_set_server_sigalgs_impl`: `SSL_CTX_set1_sigalgs_list`.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_set_server_sigalgs(
    context: *mut Context,
    list: &str,
) -> Result<(), &'static str> {
    unsafe { (*context).server_sigalgs = Some(parse_sigalgs(list)?) };
    Ok(())
}

#[derive(Debug)]
struct NoCertificateVerification;

impl rustls::client::danger::ServerCertVerifier for NoCertificateVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        let provider =
            rustls::crypto::CryptoProvider::get_default().expect("the _ssl provider is installed");
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            signature,
            &provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        let provider =
            rustls::crypto::CryptoProvider::get_default().expect("the _ssl provider is installed");
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            signature,
            &provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::CryptoProvider::get_default()
            .expect("the _ssl provider is installed")
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Keep WebPKI's chain, time, purpose, and signature validation while making
/// Python's `SSLContext.check_hostname` an independent policy switch.
///
/// rustls 0.23's WebPkiServerVerifier validates the complete chain before its
/// final server-name check. We deliberately supply an unrelated valid DNS
/// name and suppress only the two typed name-mismatch outcomes. Unlike
/// RustPython's certificate-name extraction workaround, this neither parses
/// SAN/CN a second time nor invents wildcard normalization rules.
#[derive(Debug)]
struct ChainOnlyServerVerifier {
    inner: Arc<dyn rustls::client::danger::ServerCertVerifier>,
}

#[derive(Debug)]
struct PolicyServerVerifier {
    inner: Arc<dyn rustls::client::danger::ServerCertVerifier>,
    require_authority_key_identifier: bool,
    require_crl: bool,
    has_crl: bool,
}

/// Preserve OpenSSL's explicit end-entity trust semantics on top of WebPKI.
///
/// RustPython's `PartialChainVerifier` performs the same fallback: a
/// self-issued leaf explicitly loaded into the trust store is trusted without
/// `VERIFY_X509_PARTIAL_CHAIN`; a non-self-issued leaf additionally requires
/// that flag.  The exact DER match is the trust decision, but certificate
/// time, purpose, and server-name policy still apply.
#[derive(Debug)]
struct ExplicitEndEntityVerifier {
    inner: Arc<dyn rustls::client::danger::ServerCertVerifier>,
    trusted_der: Vec<Vec<u8>>,
    verify_flags: i32,
}

impl rustls::client::danger::ServerCertVerifier for ExplicitEndEntityVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &rustls::pki_types::ServerName<'_>,
        ocsp_response: &[u8],
        now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        let original_error = match self.inner.verify_server_cert(
            end_entity,
            intermediates,
            server_name,
            ocsp_response,
            now,
        ) {
            Ok(verified) => return Ok(verified),
            Err(error) => error,
        };

        let exact_match = self
            .trusted_der
            .iter()
            .any(|trusted| trusted.as_slice() == end_entity.as_ref());
        if !exact_match {
            return Err(original_error);
        }

        let (_, certificate) =
            x509_parser::parse_x509_certificate(end_entity.as_ref()).map_err(|_| {
                rustls::Error::InvalidCertificate(rustls::CertificateError::BadEncoding)
            })?;
        let self_issued = certificate.subject() == certificate.issuer();
        if !self_issued && self.verify_flags & VerifyFlags::X509_PARTIAL_CHAIN.bits() == 0 {
            return Err(original_error);
        }

        let now = i64::try_from(now.as_secs()).map_err(|_| {
            rustls::Error::InvalidCertificate(
                rustls::CertificateError::ApplicationVerificationFailure,
            )
        })?;
        if now < certificate.validity().not_before.timestamp() {
            return Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::NotValidYet,
            ));
        }
        if now > certificate.validity().not_after.timestamp() {
            return Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::Expired,
            ));
        }
        if certificate
            .extended_key_usage()
            .map_err(|_| rustls::Error::InvalidCertificate(rustls::CertificateError::BadEncoding))?
            .is_some_and(|usage| !usage.value.any && !usage.value.server_auth)
        {
            return Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::InvalidPurpose,
            ));
        }

        let certificate = rustls::server::ParsedCertificate::try_from(end_entity)?;
        rustls::client::verify_server_name(&certificate, server_name)?;
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls12_signature(message, cert, signature)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls13_signature(message, cert, signature)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.inner.supported_verify_schemes()
    }
}

impl rustls::client::danger::ServerCertVerifier for PolicyServerVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &rustls::pki_types::ServerName<'_>,
        ocsp_response: &[u8],
        now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        if self.require_crl && !self.has_crl {
            return Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::UnknownRevocationStatus,
            ));
        }
        if self.require_authority_key_identifier {
            let (_, certificate) = x509_parser::parse_x509_certificate(end_entity.as_ref())
                .map_err(|_| {
                    rustls::Error::InvalidCertificate(rustls::CertificateError::BadEncoding)
                })?;
            let has_aki = certificate
                .extensions()
                .iter()
                .any(|extension| extension.oid.to_id_string() == "2.5.29.35");
            if certificate.subject() != certificate.issuer() && !has_aki {
                return Err(rustls::Error::InvalidCertificate(
                    rustls::CertificateError::ApplicationVerificationFailure,
                ));
            }
            for intermediate in intermediates {
                let (_, certificate) = x509_parser::parse_x509_certificate(intermediate.as_ref())
                    .map_err(|_| {
                    rustls::Error::InvalidCertificate(rustls::CertificateError::BadEncoding)
                })?;
                let has_aki = certificate
                    .extensions()
                    .iter()
                    .any(|extension| extension.oid.to_id_string() == "2.5.29.35");
                if !has_aki {
                    return Err(rustls::Error::InvalidCertificate(
                        rustls::CertificateError::ApplicationVerificationFailure,
                    ));
                }
            }
        }
        self.inner
            .verify_server_cert(end_entity, intermediates, server_name, ocsp_response, now)
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls12_signature(message, cert, signature)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls13_signature(message, cert, signature)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.inner.supported_verify_schemes()
    }
}

impl rustls::client::danger::ServerCertVerifier for ChainOnlyServerVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        ocsp_response: &[u8],
        now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        let unrelated_name = rustls::pki_types::ServerName::try_from("pyre.invalid")
            .expect("the fixed chain-only verifier name is valid");
        match self.inner.verify_server_cert(
            end_entity,
            intermediates,
            &unrelated_name,
            ocsp_response,
            now,
        ) {
            Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::NotValidForName
                | rustls::CertificateError::NotValidForNameContext { .. },
            )) => Ok(rustls::client::danger::ServerCertVerified::assertion()),
            result => result,
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls12_signature(message, cert, signature)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls13_signature(message, cert, signature)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.inner.supported_verify_schemes()
    }
}

fn is_capath_hash_name(name: &std::ffi::OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return false;
    };
    let Some((hash, suffix)) = name.split_once('.') else {
        return false;
    };
    hash.len() == 8
        && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
        && !suffix.is_empty()
        && suffix.bytes().all(|byte| byte.is_ascii_digit())
}

fn capath_stamps(context: &Context) -> Vec<CapathStamp> {
    context
        .capaths
        .iter()
        .map(|directory| {
            let modified = std::fs::metadata(directory)
                .and_then(|metadata| metadata.modified())
                .ok();
            (directory.clone(), modified)
        })
        .collect()
}

fn read_capath_certificates(context: &Context) -> Vec<Vec<u8>> {
    let mut paths = Vec::new();
    for directory in &context.capaths {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            if is_capath_hash_name(&entry.file_name()) {
                paths.push(entry.path());
            }
        }
    }
    paths.sort();

    let mut result: Vec<Vec<u8>> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for path in paths {
        let Ok(data) = std::fs::read(path) else {
            continue;
        };
        let Ok(certificates) = read_pem_certificates(&data) else {
            continue;
        };
        for certificate in certificates {
            let der = certificate.as_ref().to_vec();
            if seen.insert(der.clone()) {
                result.push(der);
            }
        }
    }
    result
}

/// Materialize every certificate reachable through the context's hashed CA
/// directories, reusing the previous scan while the directories are unchanged.
fn capath_certificates(context: &Context) -> Arc<Vec<Vec<u8>>> {
    if context.capaths.is_empty() {
        return Arc::new(Vec::new());
    }
    let stamps = capath_stamps(context);
    let mut cache = context.capath_cache.lock();
    if let Some((cached_stamps, certificates)) = cache.as_ref()
        && cached_stamps == &stamps
    {
        return certificates.clone();
    }
    let certificates = Arc::new(read_capath_certificates(context));
    *cache = Some((stamps, certificates.clone()));
    certificates
}

fn provider_for_context(context: &Context) -> Arc<rustls::crypto::CryptoProvider> {
    let mut provider = rustls::crypto::CryptoProvider::get_default()
        .expect("the _ssl provider is installed")
        .as_ref()
        .clone();
    let tls12: Vec<_> = match &context.cipher_suites {
        Some(selected) => selected
            .iter()
            .copied()
            .filter(|suite| suite.tls13().is_none())
            .collect(),
        None => provider
            .cipher_suites
            .iter()
            .copied()
            .filter(|suite| suite.tls13().is_none())
            .collect(),
    };
    let tls13: Vec<_> = match &context.tls13_cipher_suites {
        Some(selected) => selected.clone(),
        None => provider
            .cipher_suites
            .iter()
            .copied()
            .filter(|suite| suite.tls13().is_some())
            .collect(),
    };
    provider.cipher_suites = tls13.into_iter().chain(tls12).collect();
    if let Some(groups) = &context.kx_groups {
        provider.kx_groups = groups.clone();
    }
    Arc::new(provider)
}

#[derive(Debug)]
struct RestrictedSigningKey {
    inner: Arc<dyn rustls::sign::SigningKey>,
    schemes: Vec<rustls::SignatureScheme>,
}

impl rustls::sign::SigningKey for RestrictedSigningKey {
    fn choose_scheme(
        &self,
        offered: &[rustls::SignatureScheme],
    ) -> Option<Box<dyn rustls::sign::Signer>> {
        let filtered: Vec<_> = offered
            .iter()
            .copied()
            .filter(|scheme| self.schemes.contains(scheme))
            .collect();
        self.inner.choose_scheme(&filtered)
    }

    fn public_key(&self) -> Option<rustls::pki_types::SubjectPublicKeyInfoDer<'_>> {
        self.inner.public_key()
    }

    fn algorithm(&self) -> rustls::SignatureAlgorithm {
        self.inner.algorithm()
    }
}

#[derive(Debug)]
struct RestrictedVerifySchemes {
    inner: Arc<dyn rustls::client::danger::ServerCertVerifier>,
    schemes: Vec<rustls::SignatureScheme>,
}

impl RestrictedVerifySchemes {
    fn scheme_allowed(&self, signature: &rustls::DigitallySignedStruct) -> bool {
        self.schemes.contains(&signature.scheme)
    }
}

impl rustls::client::danger::ServerCertVerifier for RestrictedVerifySchemes {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &rustls::pki_types::ServerName<'_>,
        ocsp_response: &[u8],
        now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        self.inner
            .verify_server_cert(end_entity, intermediates, server_name, ocsp_response, now)
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        if !self.scheme_allowed(signature) {
            return Err(rustls::PeerMisbehaved::SignedHandshakeWithUnadvertisedSigScheme.into());
        }
        self.inner.verify_tls12_signature(message, cert, signature)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        if !self.scheme_allowed(signature) {
            return Err(rustls::PeerMisbehaved::SignedHandshakeWithUnadvertisedSigScheme.into());
        }
        self.inner.verify_tls13_signature(message, cert, signature)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.inner
            .supported_verify_schemes()
            .into_iter()
            .filter(|scheme| self.schemes.contains(scheme))
            .collect()
    }

    fn requires_raw_public_keys(&self) -> bool {
        self.inner.requires_raw_public_keys()
    }

    fn root_hint_subjects(&self) -> Option<&[rustls::DistinguishedName]> {
        self.inner.root_hint_subjects()
    }
}

fn restrict_server_verifier(
    verifier: Arc<dyn rustls::client::danger::ServerCertVerifier>,
    schemes: Option<&[rustls::SignatureScheme]>,
) -> Arc<dyn rustls::client::danger::ServerCertVerifier> {
    match schemes {
        Some(schemes) => Arc::new(RestrictedVerifySchemes {
            inner: verifier,
            schemes: schemes.to_vec(),
        }),
        None => verifier,
    }
}

fn restrict_certified_keys(
    keys: Vec<Arc<CertifiedKey>>,
    schemes: Option<&[rustls::SignatureScheme]>,
) -> Vec<Arc<CertifiedKey>> {
    let Some(schemes) = schemes else {
        return keys;
    };
    keys.into_iter()
        .map(|key| {
            let mut wrapped = CertifiedKey::new(
                key.cert.clone(),
                Arc::new(RestrictedSigningKey {
                    inner: key.key.clone(),
                    schemes: schemes.to_vec(),
                }),
            );
            wrapped.ocsp = key.ocsp.clone();
            Arc::new(wrapped)
        })
        .collect()
}

#[derive(Debug)]
struct MultiCertResolver {
    keys: Vec<Arc<CertifiedKey>>,
}

impl MultiCertResolver {
    fn choose(&self, schemes: &[rustls::SignatureScheme]) -> Option<Arc<CertifiedKey>> {
        self.keys
            .iter()
            .find(|key| key.key.choose_scheme(schemes).is_some())
            .cloned()
    }
}

impl rustls::server::ResolvesServerCert for MultiCertResolver {
    fn resolve(&self, client_hello: rustls::server::ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        self.choose(client_hello.signature_schemes())
    }
}

impl rustls::client::ResolvesClientCert for MultiCertResolver {
    fn resolve(
        &self,
        _root_hint_subjects: &[&[u8]],
        signature_schemes: &[rustls::SignatureScheme],
    ) -> Option<Arc<CertifiedKey>> {
        self.choose(signature_schemes)
    }

    fn has_certs(&self) -> bool {
        !self.keys.is_empty()
    }
}

/// Per-connection rustls store. CPython only resumes a client connection when
/// an SSLSession is supplied explicitly, so unlike rustls' default config this
/// cache is not shared implicitly by every connection from one context.
#[derive(Debug)]
struct CapturingClientSessionStore {
    inner: rustls::client::ClientSessionMemoryCache,
    latest_tls12: Mutex<
        Option<(
            rustls::pki_types::ServerName<'static>,
            rustls::client::Tls12ClientSessionValue,
        )>,
    >,
    public_id: Mutex<Option<Vec<u8>>>,
    creation_time: Mutex<Option<u64>>,
}

impl CapturingClientSessionStore {
    fn new() -> Self {
        Self {
            inner: rustls::client::ClientSessionMemoryCache::new(8),
            latest_tls12: Mutex::new(None),
            public_id: Mutex::new(None),
            creation_time: Mutex::new(None),
        }
    }

    fn seed(&self, session: &NativeSession) {
        self.inner
            .set_tls12_session(session.server_name.clone(), session.value.clone());
        *self.latest_tls12.lock() = Some((session.server_name.clone(), session.value.clone()));
        *self.public_id.lock() = Some(session.id.clone());
        *self.creation_time.lock() = Some(session.creation_time);
    }

    fn ensure_metadata(&self) {
        let mut id = self.public_id.lock();
        if id.is_none() {
            let mut bytes = vec![0u8; 32];
            let provider = rustls::crypto::CryptoProvider::get_default()
                .expect("the _ssl provider is installed");
            if provider.secure_random.fill(&mut bytes).is_ok() {
                *id = Some(bytes);
            }
        }
        drop(id);
        let mut created = self.creation_time.lock();
        if created.is_none() {
            *created = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            );
        }
    }

    fn snapshot(
        &self,
        context_identity: usize,
        config: Arc<rustls::ClientConfig>,
        verified_chain_builder: Arc<VerifiedChainBuilder>,
    ) -> Option<NativeSession> {
        let (server_name, value) = self.latest_tls12.lock().clone()?;
        self.ensure_metadata();
        Some(NativeSession {
            context_identity,
            config,
            verified_chain_builder,
            server_name,
            value,
            id: self.public_id.lock().clone()?,
            creation_time: self.creation_time.lock().unwrap_or(0),
            timeout: 43_200,
        })
    }
}

impl rustls::client::ClientSessionStore for CapturingClientSessionStore {
    fn set_kx_hint(
        &self,
        server_name: rustls::pki_types::ServerName<'static>,
        group: rustls::NamedGroup,
    ) {
        self.inner.set_kx_hint(server_name, group);
    }

    fn kx_hint(
        &self,
        server_name: &rustls::pki_types::ServerName<'_>,
    ) -> Option<rustls::NamedGroup> {
        self.inner.kx_hint(server_name)
    }

    fn set_tls12_session(
        &self,
        server_name: rustls::pki_types::ServerName<'static>,
        value: rustls::client::Tls12ClientSessionValue,
    ) {
        self.inner
            .set_tls12_session(server_name.clone(), value.clone());
        *self.latest_tls12.lock() = Some((server_name, value));
        self.ensure_metadata();
    }

    fn tls12_session(
        &self,
        server_name: &rustls::pki_types::ServerName<'_>,
    ) -> Option<rustls::client::Tls12ClientSessionValue> {
        // Keep the one TLS 1.2 value directly on this per-connection store,
        // matching ClientSessionStore's documented cardinality.  The inner
        // cache remains responsible for KX hints and TLS 1.3 ticket queues.

        self.latest_tls12
            .lock()
            .as_ref()
            .filter(|(known_name, _)| known_name == server_name)
            .map(|(_, value)| value.clone())
    }

    fn remove_tls12_session(&self, server_name: &rustls::pki_types::ServerName<'static>) {
        self.inner.remove_tls12_session(server_name);
        let mut latest = self.latest_tls12.lock();
        if latest
            .as_ref()
            .is_some_and(|(known_name, _)| known_name == server_name)
        {
            *latest = None;
        }
    }

    fn insert_tls13_ticket(
        &self,
        server_name: rustls::pki_types::ServerName<'static>,
        value: rustls::client::Tls13ClientSessionValue,
    ) {
        self.inner.insert_tls13_ticket(server_name, value);
    }

    fn take_tls13_ticket(
        &self,
        server_name: &rustls::pki_types::ServerName<'static>,
    ) -> Option<rustls::client::Tls13ClientSessionValue> {
        self.inner.take_tls13_ticket(server_name)
    }
}

pub struct NativeSession {
    context_identity: usize,
    config: Arc<rustls::ClientConfig>,
    verified_chain_builder: Arc<VerifiedChainBuilder>,
    server_name: rustls::pki_types::ServerName<'static>,
    value: rustls::client::Tls12ClientSessionValue,
    id: Vec<u8>,
    creation_time: u64,
    timeout: u64,
}

#[inline(never)]
pub unsafe fn session_clone(session: *const NativeSession) -> *mut NativeSession {
    if session.is_null() {
        return std::ptr::null_mut();
    }
    let session = unsafe { &*session };
    Box::into_raw(Box::new(NativeSession {
        context_identity: session.context_identity,
        config: session.config.clone(),
        verified_chain_builder: session.verified_chain_builder.clone(),
        server_name: session.server_name.clone(),
        value: session.value.clone(),
        id: session.id.clone(),
        creation_time: session.creation_time,
        timeout: session.timeout,
    }))
}

#[inline(never)]
pub unsafe fn session_free(session: *mut NativeSession) {
    if !session.is_null() {
        unsafe { drop(Box::from_raw(session)) };
    }
}

#[inline(never)]
pub unsafe fn session_context_identity(session: *const NativeSession) -> usize {
    unsafe { (&*session).context_identity }
}

#[inline(never)]
pub unsafe fn session_id(session: *const NativeSession) -> Vec<u8> {
    unsafe { (&*session).id.clone() }
}

#[inline(never)]
pub unsafe fn session_creation_time(session: *const NativeSession) -> u64 {
    unsafe { (&*session).creation_time }
}

#[inline(never)]
pub unsafe fn session_timeout(session: *const NativeSession) -> u64 {
    unsafe { (&*session).timeout }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CertificatePurpose {
    Unverified,
    ServerAuth,
    ClientAuth,
}

/// Immutable verification inputs owned by one TLS connection.
///
/// PyPy obtains the chain from the live `SSL` object with
/// `SSL_get0_verified_chain`. rustls does not retain WebPKI's selected path, so
/// after a successful full handshake we run the same WebPKI path builder over
/// the same roots, CRLs, algorithms, time, and peer certificates. This keeps
/// the selected trust anchor on the connection instead of reconstructing it
/// from issuer-name equality.
#[derive(Debug)]
struct VerifiedChainBuilder {
    purpose: CertificatePurpose,
    roots: Arc<rustls::RootCertStore>,
    root_der: Vec<Vec<u8>>,
    deferred_roots: Vec<Vec<u8>>,
    crls: Vec<CertificateRevocationListDer<'static>>,
    crl_scope: CrlScope,
    supported: rustls::crypto::WebPkiSupportedAlgorithms,
    verify_flags: i32,
}

impl VerifiedChainBuilder {
    fn unverified(supported: rustls::crypto::WebPkiSupportedAlgorithms) -> Self {
        Self {
            purpose: CertificatePurpose::Unverified,
            roots: Arc::new(rustls::RootCertStore::empty()),
            root_der: Vec::new(),
            deferred_roots: Vec::new(),
            crls: Vec::new(),
            crl_scope: CrlScope::Disabled,
            supported,
            verify_flags: 0,
        }
    }

    fn build(
        &self,
        peer_chain: &[CertificateDer<'_>],
        now: rustls::pki_types::UnixTime,
    ) -> Option<Vec<Vec<u8>>> {
        let (end_entity, intermediates) = peer_chain.split_first()?;
        if self.purpose == CertificatePurpose::Unverified {
            return Some(
                peer_chain
                    .iter()
                    .map(|certificate| certificate.as_ref().to_vec())
                    .collect(),
            );
        }

        let certificate = webpki::EndEntityCert::try_from(end_entity).ok()?;
        let parsed_crls = self
            .crls
            .iter()
            .map(|crl| {
                webpki::BorrowedCertRevocationList::from_der(crl.as_ref())
                    .map(webpki::CertRevocationList::from)
            })
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        let crl_refs: Vec<&webpki::CertRevocationList<'_>> = parsed_crls.iter().collect();
        let revocation = if crl_refs.is_empty() {
            None
        } else {
            let builder = webpki::RevocationOptionsBuilder::new(&crl_refs).ok()?;
            let builder = if self.crl_scope == CrlScope::EndEntityOnly {
                builder.with_depth(webpki::RevocationCheckDepth::EndEntity)
            } else {
                builder
            };
            Some(builder.build())
        };
        let usage = match self.purpose {
            CertificatePurpose::ServerAuth => webpki::KeyUsage::server_auth(),
            CertificatePurpose::ClientAuth => webpki::KeyUsage::client_auth(),
            CertificatePurpose::Unverified => unreachable!(),
        };
        let path = certificate.verify_for_usage(
            self.supported.all,
            &self.roots.roots,
            intermediates,
            now,
            usage,
            revocation,
            None,
        );
        let Ok(path) = path else {
            // `ExplicitEndEntityVerifier` is the WebPKI-independent fallback
            // corresponding to OpenSSL's explicitly trusted leaf / partial
            // chain behavior. Its verified path contains that leaf only.
            if self.purpose == CertificatePurpose::ServerAuth
                && self
                    .root_der
                    .iter()
                    .any(|trusted| trusted.as_slice() == end_entity.as_ref())
                && x509_parser::parse_x509_certificate(end_entity.as_ref())
                    .ok()
                    .is_some_and(|(_, certificate)| {
                        certificate.subject() == certificate.issuer()
                            || self.verify_flags & VerifyFlags::X509_PARTIAL_CHAIN.bits() != 0
                    })
            {
                return Some(vec![end_entity.as_ref().to_vec()]);
            }
            return None;
        };

        let anchor_index = self
            .roots
            .roots
            .iter()
            .position(|anchor| std::ptr::eq(anchor, path.anchor()))?;
        let anchor_der = self.root_der.get(anchor_index)?;
        let mut chain = Vec::with_capacity(path.intermediate_certificates().count() + 2);
        chain.push(end_entity.as_ref().to_vec());
        chain.extend(
            path.intermediate_certificates()
                .map(|certificate| certificate.der().as_ref().to_vec()),
        );
        if chain
            .last()
            .is_none_or(|certificate| certificate != anchor_der)
        {
            chain.push(anchor_der.clone());
        }
        Some(chain)
    }
}

/// Clone the eager roots and append only usable hashed-directory anchors,
/// retaining a DER vector in exactly the same order as `RootCertStore.roots`.
fn verification_roots(
    context: &Context,
) -> (Arc<rustls::RootCertStore>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut roots = context.roots.clone();
    let mut root_der = context.root_der.clone();
    let mut deferred_roots = Vec::new();
    for der in capath_certificates(context).iter() {
        if root_der.iter().any(|known| known == der) {
            continue;
        }
        if roots.add(CertificateDer::from(der.clone())).is_ok() {
            root_der.push(der.clone());
            deferred_roots.push(der.clone());
        }
    }
    (Arc::new(roots), root_der, deferred_roots)
}

fn verified_chain_builder(
    context: &Context,
    purpose: CertificatePurpose,
    roots: Arc<rustls::RootCertStore>,
    root_der: Vec<Vec<u8>>,
    deferred_roots: Vec<Vec<u8>>,
    supported: rustls::crypto::WebPkiSupportedAlgorithms,
) -> Arc<VerifiedChainBuilder> {
    Arc::new(VerifiedChainBuilder {
        purpose,
        roots,
        root_der,
        deferred_roots,
        crls: context.crls.clone(),
        crl_scope: crl_scope(context.verify_flags),
        supported,
        verify_flags: context.verify_flags,
    })
}

/// OpenSSL builds the chain sent for a configured leaf from the context's
/// certificate store at handshake time. Complete each immutable rustls
/// `CertifiedKey` through WebPKI as its `ClientConfig`/`ServerConfig` is built,
/// preserving the originally loaded chain when no valid completion exists.
fn completed_certified_keys(
    context: &Context,
    purpose: CertificatePurpose,
    roots: Arc<rustls::RootCertStore>,
    root_der: Vec<Vec<u8>>,
    supported: rustls::crypto::WebPkiSupportedAlgorithms,
) -> Vec<Arc<CertifiedKey>> {
    let builder = VerifiedChainBuilder {
        purpose,
        roots,
        root_der,
        deferred_roots: Vec::new(),
        crls: Vec::new(),
        crl_scope: CrlScope::Disabled,
        supported,
        verify_flags: context.verify_flags,
    };
    context
        .certified_keys
        .iter()
        .map(|key| {
            let Some(chain) = builder.build(&key.cert, rustls::pki_types::UnixTime::now()) else {
                return key.clone();
            };
            let mut completed = CertifiedKey::new(
                chain.into_iter().map(CertificateDer::from).collect(),
                key.key.clone(),
            );
            completed.ocsp = key.ocsp.clone();
            Arc::new(completed)
        })
        .collect()
}

fn client_config(
    context: &Context,
) -> NativeResult<(rustls::ClientConfig, Arc<VerifiedChainBuilder>)> {
    let provider = provider_for_context(context);
    let supported = provider.signature_verification_algorithms;
    let builder = rustls::ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(enabled_versions(context)?)
        .map_err(|error| (0, format!("[SSL] invalid TLS configuration: {error}")))?;
    let (roots, root_der, deferred_roots) = verification_roots(context);
    let chain_builder = if context.verify_mode == CERT_NONE {
        Arc::new(VerifiedChainBuilder::unverified(supported))
    } else {
        verified_chain_builder(
            context,
            CertificatePurpose::ServerAuth,
            roots.clone(),
            root_der.clone(),
            deferred_roots,
            supported,
        )
    };
    let certified_keys = completed_certified_keys(
        context,
        CertificatePurpose::ClientAuth,
        roots.clone(),
        root_der.clone(),
        supported,
    );
    let wants_client_cert = if context.verify_mode == CERT_NONE {
        let verifier: Arc<dyn rustls::client::danger::ServerCertVerifier> =
            Arc::new(NoCertificateVerification);
        builder
            .dangerous()
            .with_custom_certificate_verifier(restrict_server_verifier(
                verifier,
                context.server_sigalgs.as_deref(),
            ))
    } else if !roots.is_empty() {
        let mut verifier_builder = rustls::client::WebPkiServerVerifier::builder(roots);
        if !context.crls.is_empty() {
            verifier_builder = verifier_builder.with_crls(context.crls.clone());
            if crl_scope(context.verify_flags) == CrlScope::EndEntityOnly {
                verifier_builder = verifier_builder.only_check_end_entity_revocation();
            }
        }
        let verifier = verifier_builder.build().map_err(|error| {
            (
                0,
                format!("[SSL] cannot build certificate verifier: {error}"),
            )
        })?;
        let verifier: Arc<dyn rustls::client::danger::ServerCertVerifier> =
            Arc::new(ExplicitEndEntityVerifier {
                inner: verifier,
                trusted_der: root_der,
                verify_flags: context.verify_flags,
            });
        let mut verifier: Arc<dyn rustls::client::danger::ServerCertVerifier> =
            Arc::new(PolicyServerVerifier {
                inner: verifier,
                require_authority_key_identifier: context.verify_flags
                    & VerifyFlags::X509_STRICT.bits()
                    != 0,
                require_crl: crl_scope(context.verify_flags) != CrlScope::Disabled,
                has_crl: !context.crls.is_empty(),
            });
        if !context.check_hostname {
            verifier = Arc::new(ChainOnlyServerVerifier { inner: verifier });
        }
        builder
            .dangerous()
            .with_custom_certificate_verifier(restrict_server_verifier(
                verifier,
                context.server_sigalgs.as_deref(),
            ))
    } else {
        builder.with_root_certificates((*roots).clone())
    };
    let mut config = if !certified_keys.is_empty() {
        wants_client_cert.with_client_cert_resolver(Arc::new(MultiCertResolver {
            keys: certified_keys,
        }))
    } else {
        wants_client_cert.with_no_client_auth()
    };
    config.alpn_protocols = context.alpn_protocols.clone();
    Ok((config, chain_builder))
}

fn server_config(
    context: &Context,
) -> NativeResult<(rustls::ServerConfig, Option<Arc<VerifiedChainBuilder>>)> {
    if context.certified_keys.is_empty() {
        return Err((
            0,
            "[SSL] server-side connection requires a certificate and private key".to_string(),
        ));
    }
    let provider = provider_for_context(context);
    let supported = provider.signature_verification_algorithms;
    let builder = rustls::ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(enabled_versions(context)?)
        .map_err(|error| (0, format!("[SSL] invalid TLS configuration: {error}")))?;
    let (roots, root_der, deferred_roots) = verification_roots(context);
    let certified_keys = restrict_certified_keys(
        completed_certified_keys(
            context,
            CertificatePurpose::ServerAuth,
            roots.clone(),
            root_der.clone(),
            supported,
        ),
        context.server_sigalgs.as_deref(),
    );
    let (wants_server_cert, chain_builder) = if context.verify_mode == CERT_NONE {
        (builder.with_no_client_auth(), None)
    } else {
        // A hashed directory is as much a trust source for client certificates
        // as an eagerly loaded file, so it must reach the verifier.  Silently
        // dropping client authentication when no root is configured would
        // accept unauthenticated clients under CERT_REQUIRED.
        if roots.is_empty() {
            return Err((
                0,
                "[SSL] client authentication requires at least one trusted CA certificate"
                    .to_string(),
            ));
        }
        let chain_builder = verified_chain_builder(
            context,
            CertificatePurpose::ClientAuth,
            roots.clone(),
            root_der,
            deferred_roots,
            supported,
        );
        let mut verifier = rustls::server::WebPkiClientVerifier::builder(roots);
        if !context.crls.is_empty() {
            verifier = verifier.with_crls(context.crls.clone());
            if crl_scope(context.verify_flags) == CrlScope::EndEntityOnly {
                verifier = verifier.only_check_end_entity_revocation();
            }
        }
        let verifier = if context.verify_mode == CERT_OPTIONAL {
            verifier.allow_unauthenticated()
        } else {
            verifier
        }
        .build()
        .map_err(|error| (0, format!("[SSL] cannot build client verifier: {error}")))?;
        (
            builder.with_client_cert_verifier(verifier),
            Some(chain_builder),
        )
    };
    let mut config = wants_server_cert.with_cert_resolver(Arc::new(MultiCertResolver {
        keys: certified_keys,
    }));
    config.alpn_protocols = context.alpn_protocols.clone();
    config.session_storage = context.server_session_store.clone();
    config.ticketer = context.server_ticketer.clone();
    if let Some(tickets) = context.num_tickets {
        config.send_tls13_tickets = tickets;
    }
    Ok((config, chain_builder))
}

fn enabled_versions(
    context: &Context,
) -> NativeResult<&'static [&'static rustls::SupportedProtocolVersion]> {
    let minimum = context.minimum_version;
    let maximum = context.maximum_version;
    let tls12 = (minimum < 0 || minimum <= 0x303)
        && !(0..0x303).contains(&maximum)
        && context.options & SslOp::NO_TLSV1_2.bits() == 0;
    let tls13 = (minimum < 0 || minimum <= 0x304)
        && !(0..0x304).contains(&maximum)
        && context.options & SslOp::NO_TLSV1_3.bits() == 0;
    if !tls12 && !tls13 {
        return Err((0, "[SSL] no protocols available".to_string()));
    }
    #[cfg(feature = "host_env")]
    {
        let options = (context.options & (SslOp::NO_TLSV1_2 | SslOp::NO_TLSV1_3).bits()) as i32;
        Ok(host_ssl::rustls_versions(minimum, maximum, options))
    }
    #[cfg(not(feature = "host_env"))]
    {
        static TLS12_ONLY: &[&rustls::SupportedProtocolVersion] = &[&rustls::version::TLS12];
        static TLS13_ONLY: &[&rustls::SupportedProtocolVersion] = &[&rustls::version::TLS13];
        match (tls12, tls13) {
            (true, true) => Ok(rustls::DEFAULT_VERSIONS),
            (true, false) => Ok(TLS12_ONLY),
            (false, true) => Ok(TLS13_ONLY),
            (false, false) => unreachable!("empty protocol set rejected above"),
        }
    }
}

pub type TlsResult<T> = Result<T, (i32, String)>;

/// One protocol-level event for CPython's private `_msg_callback` hook.
/// Record framing stays in the TLS engine; the interpreter only turns these
/// inert values into Python callback arguments.
pub struct TlsMessageEvent {
    pub write: bool,
    pub version: u16,
    pub content_type: u16,
    pub message_type: u16,
    pub data: Vec<u8>,
}

#[cfg(not(feature = "host_env"))]
#[derive(Default)]
struct TlsRecordObserver {
    records: Vec<u8>,
    handshakes: Vec<u8>,
    encrypted: bool,
}

#[cfg(not(feature = "host_env"))]
impl TlsRecordObserver {
    fn observe(
        &mut self,
        bytes: &[u8],
        write: bool,
        events: &mut Vec<TlsMessageEvent>,
        handshake_transcript: &mut Vec<u8>,
    ) {
        self.records.extend_from_slice(bytes);
        loop {
            if self.records.len() < 5 {
                return;
            }
            let content_type = self.records[0];
            let version = u16::from_be_bytes([self.records[1], self.records[2]]);
            let payload_len = u16::from_be_bytes([self.records[3], self.records[4]]) as usize;
            if self.records.len() < 5 + payload_len {
                return;
            }
            let record: Vec<u8> = self.records.drain(..5 + payload_len).collect();
            events.push(TlsMessageEvent {
                write,
                version,
                content_type: 0x100,
                message_type: content_type as u16,
                data: record[..5].to_vec(),
            });
            let payload = &record[5..];
            match content_type {
                20 => {
                    events.push(TlsMessageEvent {
                        write,
                        version,
                        content_type: 20,
                        message_type: 0x101,
                        data: payload.to_vec(),
                    });
                    // In TLS 1.2, protocol messages following CCS are
                    // encrypted. TLS 1.3 uses application-data records for
                    // encrypted handshake traffic and never enters here.
                    self.encrypted = true;
                    self.handshakes.clear();
                }
                21 if payload.len() >= 2 => events.push(TlsMessageEvent {
                    write,
                    version,
                    content_type: 21,
                    message_type: payload[1] as u16,
                    data: payload.to_vec(),
                }),
                22 if !self.encrypted => {
                    self.handshakes.extend_from_slice(payload);
                    loop {
                        if self.handshakes.len() < 4 {
                            break;
                        }
                        let message_len = ((self.handshakes[1] as usize) << 16)
                            | ((self.handshakes[2] as usize) << 8)
                            | self.handshakes[3] as usize;
                        if self.handshakes.len() < 4 + message_len {
                            break;
                        }
                        let message: Vec<u8> = self.handshakes.drain(..4 + message_len).collect();
                        handshake_transcript.extend_from_slice(&message);
                        events.push(TlsMessageEvent {
                            write,
                            version,
                            content_type: 22,
                            message_type: message[0] as u16,
                            data: message,
                        });
                    }
                }
                _ => {}
            }
        }
    }
}

/// PyPy `_SSLContext.keylog_bio`: an append-only destination owned by one
/// context.  OpenSSL serializes its BIO callback with
/// `keylog_callback_lock`; a per-context mutex supplies the same guarantee
/// without widening unrelated contexts onto one process-global lock.
#[derive(Debug, Default)]
struct ContextKeyLog {
    #[cfg(feature = "host_env")]
    file: keylog::KeyLog,
    #[cfg(not(feature = "host_env"))]
    file: Mutex<Option<std::fs::File>>,
}

impl ContextKeyLog {
    #[cfg(feature = "host_env")]
    fn set_path(&self, path: Option<&std::path::Path>) -> std::io::Result<()> {
        self.file.0.set_path(path, keylog::HEADER)
    }

    #[cfg(not(feature = "host_env"))]
    fn set_path(&self, path: Option<&std::path::Path>) -> std::io::Result<()> {
        let mut slot = self.file.lock();
        // `_SSLContext.keylog_filename` removes the old callback and closes
        // its BIO before it handles either None or the replacement path.
        *slot = None;
        let Some(path) = path else {
            return Ok(());
        };
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let metadata = file.metadata()?;
        // `BIO_tell(...) == 0` writes the header only for an empty seekable
        // file; pipes and other non-files are deliberately excluded.
        if metadata.file_type().is_file() && metadata.len() == 0 {
            file.write_all(b"# TLS secrets log file, generated by OpenSSL / Python\n")?;
            file.flush()?;
        }
        *slot = Some(file);
        Ok(())
    }

    #[cfg(feature = "host_env")]
    fn enabled(&self) -> bool {
        self.file.0.enabled()
    }

    #[cfg(not(feature = "host_env"))]
    fn enabled(&self) -> bool {
        self.file.lock().is_some()
    }

    fn log(&self, label: &str, client_random: &[u8], secret: &[u8]) {
        // Like PyPy's OpenSSL callback, the asynchronous logger cannot
        // propagate write errors into the handshake; setter errors are eager.
        #[cfg(feature = "host_env")]
        rustls::KeyLog::log(&self.file, label, client_random, secret);
        #[cfg(not(feature = "host_env"))]
        {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let mut line =
                Vec::with_capacity(label.len() + 2 * (client_random.len() + secret.len()) + 3);
            line.extend_from_slice(label.as_bytes());
            line.push(b' ');
            for byte in client_random {
                line.push(HEX[(byte >> 4) as usize]);
                line.push(HEX[(byte & 0x0f) as usize]);
            }
            line.push(b' ');
            for byte in secret {
                line.push(HEX[(byte >> 4) as usize]);
                line.push(HEX[(byte & 0x0f) as usize]);
            }
            line.push(b'\n');
            let mut slot = self.file.lock();
            if let Some(file) = slot.as_mut() {
                // Rustls' callback has no error return, just like the OpenSSL
                // callback boundary.  A setter/open error is reported eagerly;
                // later asynchronous write errors cannot change the handshake.
                let _ = file.write_all(&line);
                let _ = file.flush();
            }
        }
    }
}

/// Per-connection capture of the TLS 1.2 master secret. rustls deliberately
/// exposes this only through its `KeyLog` hook; retaining it on the connection
/// lets us implement RFC 5929 `tls-unique` while the context-owned sink above
/// implements PyPy's public keylog callback.
#[derive(Debug)]
struct CapturingKeyLog {
    tls12_master_secret: Mutex<Option<Vec<u8>>>,
    sink: Mutex<Arc<ContextKeyLog>>,
}

impl CapturingKeyLog {
    fn new(sink: Arc<ContextKeyLog>) -> Self {
        Self {
            tls12_master_secret: Mutex::new(None),
            sink: Mutex::new(sink),
        }
    }

    fn set_sink(&self, sink: Arc<ContextKeyLog>) {
        *self.sink.lock() = sink;
    }
}

impl rustls::KeyLog for CapturingKeyLog {
    fn log(&self, label: &str, client_random: &[u8], secret: &[u8]) {
        if label == "CLIENT_RANDOM" {
            *self.tls12_master_secret.lock() = Some(secret.to_vec());
        }
        let sink = self.sink.lock().clone();
        sink.log(label, client_random, secret);
    }

    fn will_log(&self, label: &str) -> bool {
        label == "CLIENT_RANDOM" || self.sink.lock().enabled()
    }
}

#[cfg(not(feature = "host_env"))]
macro_rules! hmac_digest {
    ($name:ident, $digest:ty, $block_size:expr) => {
        fn $name(key: &[u8], data: &[u8]) -> Vec<u8> {
            let mut key_block = vec![0u8; $block_size];
            if key.len() > $block_size {
                let digest = <$digest>::digest(key);
                key_block[..digest.len()].copy_from_slice(&digest);
            } else {
                key_block[..key.len()].copy_from_slice(key);
            }
            let mut inner_pad = key_block.clone();
            let mut outer_pad = key_block;
            for byte in &mut inner_pad {
                *byte ^= 0x36;
            }
            for byte in &mut outer_pad {
                *byte ^= 0x5c;
            }
            let mut inner = <$digest>::new();
            inner.update(&inner_pad);
            inner.update(data);
            let inner = inner.finalize();
            let mut outer = <$digest>::new();
            outer.update(&outer_pad);
            outer.update(inner);
            outer.finalize().to_vec()
        }
    };
}

#[cfg(not(feature = "host_env"))]
hmac_digest!(hmac_sha256, sha2::Sha256, 64);
#[cfg(not(feature = "host_env"))]
hmac_digest!(hmac_sha384, sha2::Sha384, 128);

#[cfg(not(feature = "host_env"))]
fn tls12_p_hash(
    secret: &[u8],
    seed: &[u8],
    hmac: fn(&[u8], &[u8]) -> Vec<u8>,
    length: usize,
) -> Vec<u8> {
    let mut a = hmac(secret, seed);
    let mut output = Vec::with_capacity(length);
    while output.len() < length {
        let mut input = Vec::with_capacity(a.len() + seed.len());
        input.extend_from_slice(&a);
        input.extend_from_slice(seed);
        output.extend_from_slice(&hmac(secret, &input));
        a = hmac(secret, &a);
    }
    output.truncate(length);
    output
}

#[cfg(not(feature = "host_env"))]
fn tls12_finished(secret: &[u8], transcript: &[u8], label: &[u8], sha384: bool) -> Vec<u8> {
    let transcript_hash = if sha384 {
        sha2::Sha384::digest(transcript).to_vec()
    } else {
        sha2::Sha256::digest(transcript).to_vec()
    };
    let mut seed = Vec::with_capacity(label.len() + transcript_hash.len());
    seed.extend_from_slice(label);
    seed.extend_from_slice(&transcript_hash);
    tls12_p_hash(
        secret,
        &seed,
        if sha384 { hmac_sha384 } else { hmac_sha256 },
        12,
    )
}

fn rustls_error(error: impl std::fmt::Display) -> (i32, String) {
    (TLS_ERROR_SSL, format!("[SSL] {error}"))
}

/// Map a rustls verification failure onto its OpenSSL X509 verification code.
/// The accompanying text comes from [`certificate_verify_message`], the single
/// code-to-message table, so `SSLCertVerificationError.verify_message` and the
/// exception text cannot drift apart.
#[allow(deprecated)] // rustls can still return the compatibility variant.
fn certificate_error_details(error: &rustls::CertificateError) -> (i32, &'static str) {
    #[cfg(feature = "host_env")]
    use host_ssl::x509::{
        X509_V_ERR_CERT_HAS_EXPIRED, X509_V_ERR_CERT_NOT_YET_VALID, X509_V_ERR_CERT_REVOKED,
        X509_V_ERR_HOSTNAME_MISMATCH, X509_V_ERR_INVALID_PURPOSE, X509_V_ERR_UNABLE_TO_GET_CRL,
        X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY, X509_V_ERR_UNSPECIFIED,
    };
    use rustls::CertificateError;
    #[cfg(not(feature = "host_env"))]
    const X509_V_ERR_UNSPECIFIED: i32 = 1;
    #[cfg(not(feature = "host_env"))]
    const X509_V_ERR_UNABLE_TO_GET_CRL: i32 = 3;
    #[cfg(not(feature = "host_env"))]
    const X509_V_ERR_CERT_NOT_YET_VALID: i32 = 9;
    #[cfg(not(feature = "host_env"))]
    const X509_V_ERR_CERT_HAS_EXPIRED: i32 = 10;
    #[cfg(not(feature = "host_env"))]
    const X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY: i32 = 20;
    #[cfg(not(feature = "host_env"))]
    const X509_V_ERR_CERT_REVOKED: i32 = 23;
    #[cfg(not(feature = "host_env"))]
    const X509_V_ERR_INVALID_PURPOSE: i32 = 26;
    #[cfg(not(feature = "host_env"))]
    const X509_V_ERR_HOSTNAME_MISMATCH: i32 = 62;
    let code = match error {
        CertificateError::Expired | CertificateError::ExpiredContext { .. } => {
            X509_V_ERR_CERT_HAS_EXPIRED
        }
        CertificateError::NotValidYet | CertificateError::NotValidYetContext { .. } => {
            X509_V_ERR_CERT_NOT_YET_VALID
        }
        CertificateError::Revoked => X509_V_ERR_CERT_REVOKED,
        CertificateError::UnknownIssuer => X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY,
        CertificateError::BadSignature => 7,
        CertificateError::NotValidForName | CertificateError::NotValidForNameContext { .. } => {
            X509_V_ERR_HOSTNAME_MISMATCH
        }
        CertificateError::InvalidPurpose | CertificateError::InvalidPurposeContext { .. } => {
            X509_V_ERR_INVALID_PURPOSE
        }
        CertificateError::BadEncoding => 5,
        CertificateError::UnhandledCriticalExtension => 34,
        CertificateError::UnknownRevocationStatus => X509_V_ERR_UNABLE_TO_GET_CRL,
        CertificateError::ExpiredRevocationList
        | CertificateError::ExpiredRevocationListContext { .. } => 12,
        CertificateError::UnsupportedSignatureAlgorithm
        | CertificateError::UnsupportedSignatureAlgorithmContext { .. }
        | CertificateError::UnsupportedSignatureAlgorithmForPublicKeyContext { .. } => 7,
        CertificateError::InvalidOcspResponse => 50,
        _ => X509_V_ERR_UNSPECIFIED,
    };
    (code, certificate_verify_message(code))
}

fn rustls_protocol_error(error: rustls::Error) -> (i32, String) {
    if let rustls::Error::InvalidCertificate(certificate_error) = &error {
        let (verify_code, verify_message) = certificate_error_details(certificate_error);
        return (
            TLS_ERROR_CERT_VERIFY_BASE + verify_code,
            format!("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: {verify_message}"),
        );
    }
    if let rustls::Error::AlertReceived(alert) = &error {
        let reason = match *alert {
            rustls::AlertDescription::AccessDenied => "TLSV1_ALERT_ACCESS_DENIED",
            rustls::AlertDescription::HandshakeFailure => "SSLV3_ALERT_HANDSHAKE_FAILURE",
            rustls::AlertDescription::InternalError => "TLSV1_ALERT_INTERNAL_ERROR",
            rustls::AlertDescription::ProtocolVersion => "TLSV1_ALERT_PROTOCOL_VERSION",
            rustls::AlertDescription::UnknownCA => "TLSV1_ALERT_UNKNOWN_CA",
            _ => "TLSV1_ALERT_UNKNOWN",
        };
        return (
            TLS_ERROR_SSL,
            format!("[SSL: {reason}] received fatal TLS alert"),
        );
    }
    if matches!(
        error,
        rustls::Error::PeerIncompatible(rustls::PeerIncompatible::NoCipherSuitesInCommon)
    ) {
        return (
            TLS_ERROR_SSL,
            "[SSL: NO_SHARED_CIPHER] no shared cipher".to_string(),
        );
    }
    rustls_error(error)
}

#[inline(never)]
pub fn certificate_verify_message(code: i32) -> &'static str {
    match code {
        10 => "certificate has expired",
        9 => "certificate is not yet valid",
        23 => "certificate revoked",
        20 => "unable to get local issuer certificate",
        7 => "certificate signature failure",
        62 => "hostname mismatch",
        26 => "unsuitable certificate purpose",
        5 => "unable to decode certificate",
        34 => "unhandled critical extension",
        3 => "unable to get certificate CRL",
        12 => "CRL has expired",
        50 => "application verification failure",
        _ => "certificate verify failed",
    }
}

pub struct TlsConnection {
    inner: Option<rustls::Connection>,
    acceptor: Option<rustls::server::Acceptor>,
    accepted: Option<rustls::server::Accepted>,
    pending_tls: Vec<u8>,
    pending_tls_start: usize,
    verified_chain_builder: Option<Arc<VerifiedChainBuilder>>,
    verified_chain: Option<Vec<Vec<u8>>>,
    verified_chain_computed: bool,
    verified_root_taken: bool,
    #[cfg(feature = "host_env")]
    msg_state: msg::MsgState,
    #[cfg(not(feature = "host_env"))]
    incoming_observer: TlsRecordObserver,
    #[cfg(not(feature = "host_env"))]
    outgoing_observer: TlsRecordObserver,
    message_events: Vec<TlsMessageEvent>,
    #[cfg(not(feature = "host_env"))]
    tls12_handshake_transcript: Vec<u8>,
    key_log: Arc<CapturingKeyLog>,
    tls_unique: Option<Vec<u8>>,
    pending_received_tls: Vec<u8>,
    pending_received_tls_start: usize,
    client_config: Option<Arc<rustls::ClientConfig>>,
    client_session_store: Option<Arc<CapturingClientSessionStore>>,
    context_identity: usize,
    server_context: *const Context,
    server_hit_counted: bool,
    /// `SSL_RECEIVED_SHUTDOWN`: a fatal alert from the peer ends the
    /// session, and every later operation answers from that state rather
    /// than from the wire.  rustls latches the same fact inside
    /// `process_new_packets`, but only a call that consumed fresh
    /// ciphertext reaches it, so the latch is kept here too.
    peer_sent_fatal_alert: bool,
    /// OpenSSL's `SSL_RECEIVED_SHUTDOWN` is sticky for the lifetime of the
    /// connection.  rustls reports `peer_has_closed()` on the `IoState`
    /// returned while the close_notify record is processed; a later
    /// `process_new_packets()` call need not report that historical event.
    /// Keep the state on the connection, where OpenSSL/PyPy keep it.
    peer_sent_close_notify: bool,
}

impl TlsConnection {
    fn observe_records(&mut self, write: bool, bytes: &[u8]) {
        #[cfg(feature = "host_env")]
        {
            for event in self.msg_state.observe(write, bytes) {
                self.message_events.push(TlsMessageEvent {
                    write,
                    version: event.version as u16,
                    content_type: event.content_type as u16,
                    message_type: event.msg_type as u16,
                    data: event.data,
                });
            }
        }
        #[cfg(not(feature = "host_env"))]
        {
            let observer = if write {
                &mut self.outgoing_observer
            } else {
                &mut self.incoming_observer
            };
            observer.observe(
                bytes,
                write,
                &mut self.message_events,
                &mut self.tls12_handshake_transcript,
            );
        }
    }

    fn active_mut(&mut self) -> TlsResult<&mut rustls::Connection> {
        self.inner.as_mut().ok_or_else(|| {
            (
                TLS_ERROR_WANT_READ,
                "TLS server handshake is waiting for ClientHello configuration".to_string(),
            )
        })
    }

    fn fill_pending_tls(&mut self) -> TlsResult<()> {
        if self.pending_tls_start == self.pending_tls.len() {
            self.pending_tls.clear();
            self.pending_tls_start = 0;
        }
        if self.inner.is_none() {
            return Ok(());
        }
        loop {
            self.write_pending_tls()?;
            // `wants_read` is also false while an outgoing flight is still
            // queued and application data cannot flow yet, which is the state
            // a client is in between emitting its ChangeCipherSpec and its
            // Finished.  A peer flight that arrives in a single read therefore
            // leaves a tail in `pending_received_tls` that no later call comes
            // back for: the transport goes straight to a blocking receive
            // while the peer waits for the rest of our handshake, and both
            // sides stop.  Emptying the queued flight above lifts the refusal,
            // so re-offer the tail here and write out whatever it produces.
            let remaining = self.pending_received_tls.len() - self.pending_received_tls_start;
            if remaining == 0 {
                break;
            }
            self.process_received_tls()?;
            // Every extra pass must consume ciphertext, so the loop ends.
            if self.pending_received_tls.len() - self.pending_received_tls_start >= remaining {
                break;
            }
        }
        self.capture_tls_unique();
        Ok(())
    }

    /// Move every record rustls has queued for the peer into `pending_tls`.
    fn write_pending_tls(&mut self) -> TlsResult<()> {
        loop {
            let before = self.pending_tls.len();
            {
                let Some(inner) = self.inner.as_mut() else {
                    return Ok(());
                };
                if !inner.wants_write() {
                    break;
                }
                inner
                    .write_tls(&mut self.pending_tls)
                    .map_err(rustls_error)?;
                if self.pending_tls.len() == before {
                    break;
                }
            }
            let chunk = self.pending_tls[before..].to_vec();
            self.observe_records(true, &chunk);
        }
        Ok(())
    }

    fn compact_received_tls(&mut self) {
        if self.pending_received_tls_start == self.pending_received_tls.len() {
            self.pending_received_tls.clear();
            self.pending_received_tls_start = 0;
        } else if self.pending_received_tls_start >= 16 * 1024 {
            self.pending_received_tls
                .drain(..self.pending_received_tls_start);
            self.pending_received_tls_start = 0;
        }
    }

    /// Feed as much queued ciphertext as rustls can currently accept.  Its
    /// plaintext queue intentionally applies backpressure at 16 KiB, so an
    /// unread tail remains owned by this connection until Python drains data.
    fn process_received_tls(&mut self) -> TlsResult<()> {
        loop {
            if self.pending_received_tls_start == self.pending_received_tls.len() {
                self.compact_received_tls();
                return Ok(());
            }
            let Some(inner) = self.inner.as_mut() else {
                return Ok(());
            };
            // `wants_read` covers two refusals: plaintext backpressure, and an
            // outgoing flight still queued before application data may flow.
            // The corresponding `read_tls` failure carries only an unstable
            // message string.  Either way the untouched tail stays queued, and
            // `fill_pending_tls` re-offers it once the refusal lifts.
            if !inner.wants_read() {
                return Ok(());
            }
            let mut cursor =
                Cursor::new(&self.pending_received_tls[self.pending_received_tls_start..]);
            let read = match inner.read_tls(&mut cursor) {
                Ok(0) => return Ok(()),
                Ok(read) => read,
                Err(error) => return Err(rustls_error(error)),
            };
            self.pending_received_tls_start += read;
            match inner.process_new_packets() {
                Ok(state) => self.peer_sent_close_notify |= state.peer_has_closed(),
                Err(error) => {
                    self.peer_sent_fatal_alert |= matches!(error, rustls::Error::AlertReceived(_));
                    return Err(rustls_protocol_error(error));
                }
            }
            self.note_server_resumption();
            self.compact_received_tls();
        }
    }

    fn note_server_resumption(&mut self) {
        if self.server_hit_counted || self.server_context.is_null() {
            return;
        }
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        if inner.is_handshaking() {
            return;
        }
        if inner.handshake_kind() == Some(rustls::HandshakeKind::Resumed) {
            unsafe { &*self.server_context }
                .session_hits
                .fetch_add(1, Ordering::Relaxed);
        }
        self.server_hit_counted = true;
    }

    fn capture_verified_chain(&mut self) {
        if self.verified_chain_computed {
            return;
        }
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        if inner.is_handshaking() {
            return;
        }
        self.verified_chain_computed = true;
        // PyPy/OpenSSL's SSL_get0_verified_chain returns NULL for a resumed
        // TLS 1.2 connection even though SSL_get_peer_cert_chain retains the
        // peer certificate list from the session.
        if inner.handshake_kind() == Some(rustls::HandshakeKind::Resumed) {
            return;
        }
        let (Some(builder), Some(peer_chain)) = (
            self.verified_chain_builder.as_ref(),
            inner.peer_certificates(),
        ) else {
            return;
        };
        self.verified_chain = builder.build(peer_chain, rustls::pki_types::UnixTime::now());
    }

    fn capture_tls_unique(&mut self) {
        if self.tls_unique.is_some() {
            return;
        }
        #[cfg(not(feature = "host_env"))]
        if !(self.incoming_observer.encrypted || self.outgoing_observer.encrypted) {
            return;
        }
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        // rustls can report `handshake_kind() == Full` as soon as the
        // ServerHello flight is processed, while `is_handshaking()` is still
        // true and ClientKeyExchange has not been written. Capturing then
        // freezes a client transcript that is missing CKE, so the two peers
        // derive different RFC 5929 bindings.
        if inner.is_handshaking() {
            return;
        }
        if inner.protocol_version() != Some(rustls::ProtocolVersion::TLSv1_2) {
            return;
        }
        let Some(handshake_kind) = inner.handshake_kind() else {
            return;
        };
        let Some(suite) = inner.negotiated_cipher_suite() else {
            return;
        };
        let secret = self.key_log.tls12_master_secret.lock().clone();
        let Some(secret) = secret else {
            return;
        };
        #[cfg(feature = "host_env")]
        {
            let server_side = matches!(inner, rustls::Connection::Server(_));
            self.tls_unique = msg::tls12_unique(
                suite,
                &secret,
                self.msg_state.transcript(),
                server_side,
                handshake_kind == rustls::HandshakeKind::Resumed,
            );
        }
        #[cfg(not(feature = "host_env"))]
        {
            let sha384 = matches!(
                suite.suite(),
                rustls::CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
                    | rustls::CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
            );
            let label = if handshake_kind == rustls::HandshakeKind::Resumed {
                b"server finished".as_slice()
            } else {
                b"client finished".as_slice()
            };
            self.tls_unique = Some(tls12_finished(
                &secret,
                &self.tls12_handshake_transcript,
                label,
                sha384,
            ));
            self.tls12_handshake_transcript.clear();
        }
        // Do not retain key material or the handshake transcript once the
        // RFC 5929 binding has been derived.
        self.key_log.tls12_master_secret.lock().take();
    }

    fn process_acceptor_tls(&mut self) -> TlsResult<()> {
        loop {
            if self.pending_received_tls_start == self.pending_received_tls.len() {
                self.compact_received_tls();
                return Ok(());
            }
            let acceptor = self.acceptor.as_mut().ok_or_else(|| {
                (
                    TLS_ERROR_SSL,
                    "[SSL] server acceptor is no longer available".to_string(),
                )
            })?;
            let mut cursor =
                Cursor::new(&self.pending_received_tls[self.pending_received_tls_start..]);
            let read = acceptor.read_tls(&mut cursor).map_err(rustls_error)?;
            if read == 0 {
                return Ok(());
            }
            self.pending_received_tls_start += read;
            match acceptor.accept() {
                Ok(Some(accepted)) => {
                    self.accepted = Some(accepted);
                    self.acceptor = None;
                    self.compact_received_tls();
                    return Ok(());
                }
                Ok(None) => self.compact_received_tls(),
                Err((error, mut alert)) => {
                    let before = self.pending_tls.len();
                    let _ = alert.write_all(&mut self.pending_tls);
                    let chunk = self.pending_tls[before..].to_vec();
                    self.observe_records(true, &chunk);
                    self.acceptor = None;
                    return Err(rustls_protocol_error(error));
                }
            }
        }
    }
}

/// Create one rustls state machine. It has no knowledge of Python sockets or
/// BIO objects; the interpreter owns transport policy and explicitly moves TLS
/// records through the primitive functions below.
#[inline(never)]
pub unsafe fn connection_new(
    context: *const Context,
    server_side: bool,
    server_hostname: Option<&str>,
    session: *const NativeSession,
) -> NativeResult<*mut TlsConnection> {
    ensure_provider();
    let context = unsafe { &*context };
    if server_side && context.protocol == PROTOCOL_TLS_CLIENT {
        return Err((
            0,
            "Cannot create a server socket with a PROTOCOL_TLS_CLIENT context".to_string(),
        ));
    }
    if !server_side && context.protocol == PROTOCOL_TLS_SERVER {
        return Err((
            0,
            "Cannot create a client socket with a PROTOCOL_TLS_SERVER context".to_string(),
        ));
    }
    let mut retained_client_config = None;
    let mut retained_client_store = None;
    let key_log = Arc::new(CapturingKeyLog::new(context.key_log_file.clone()));
    let (connection, acceptor, verified_chain_builder) = if server_side {
        (None, Some(rustls::server::Acceptor::default()), None)
    } else {
        let name = match server_hostname {
            Some(hostname) => rustls::pki_types::ServerName::try_from(hostname.to_string())
                .map_err(|error| (0, format!("invalid server hostname: {error}")))?,
            None => rustls::pki_types::ServerName::IpAddress(
                std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST).into(),
            ),
        };
        let store = Arc::new(CapturingClientSessionStore::new());
        let (mut config, verified_chain_builder) = if session.is_null() {
            let (config, builder) = client_config(context)?;
            (config, builder)
        } else {
            let session = unsafe { &*session };
            if session.context_identity != context.identity {
                return Err((0, "Session refers to a different SSLContext.".to_string()));
            }
            if session.server_name == name {
                store.seed(session);
            }
            (
                (*session.config).clone(),
                session.verified_chain_builder.clone(),
            )
        };
        config.resumption = rustls::client::Resumption::store(store.clone());
        config.key_log = key_log.clone();
        let config = Arc::new(config);
        retained_client_config = Some(config.clone());
        retained_client_store = Some(store);
        (
            Some(rustls::Connection::Client(
                rustls::ClientConnection::new(config, name).map_err(rustls_error)?,
            )),
            None,
            Some(verified_chain_builder),
        )
    };
    Ok(Box::into_raw(Box::new(TlsConnection {
        inner: connection,
        acceptor,
        accepted: None,
        pending_tls: Vec::new(),
        pending_tls_start: 0,
        verified_chain_builder,
        verified_chain: None,
        verified_chain_computed: false,
        verified_root_taken: false,
        #[cfg(feature = "host_env")]
        msg_state: msg::MsgState::default(),
        #[cfg(not(feature = "host_env"))]
        incoming_observer: TlsRecordObserver::default(),
        #[cfg(not(feature = "host_env"))]
        outgoing_observer: TlsRecordObserver::default(),
        message_events: Vec::new(),
        #[cfg(not(feature = "host_env"))]
        tls12_handshake_transcript: Vec::new(),
        key_log,
        tls_unique: None,
        pending_received_tls: Vec::new(),
        pending_received_tls_start: 0,
        client_config: retained_client_config,
        client_session_store: retained_client_store,
        context_identity: context.identity,
        server_context: std::ptr::null(),
        server_hit_counted: false,
        peer_sent_fatal_alert: false,
        peer_sent_close_notify: false,
    })))
}

/// # Safety
/// `connection` must be null or a live pointer returned by `connection_new`.
#[inline(never)]
pub unsafe fn connection_free(connection: *mut TlsConnection) {
    if !connection.is_null() {
        unsafe { drop(Box::from_raw(connection)) };
    }
}

/// Feed encrypted TLS records into rustls and process every complete packet.
/// Returns the number of bytes accepted from `data`.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_receive_tls(
    connection: *mut TlsConnection,
    data: &[u8],
) -> TlsResult<usize> {
    let connection = unsafe { &mut *connection };
    connection.observe_records(false, data);
    connection.pending_received_tls.extend_from_slice(data);
    if connection.inner.is_none() {
        if connection.accepted.is_some() {
            return Ok(data.len());
        }
        connection.process_acceptor_tls()?;
        return Ok(data.len());
    }

    connection.process_received_tls()?;
    connection.capture_verified_chain();
    connection.capture_tls_unique();
    Ok(data.len())
}

/// Whether a server-side connection has parsed ClientHello and is waiting for
/// the interpreter to run the Python SNI callback and choose a context.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_waiting_for_server_config(connection: *const TlsConnection) -> bool {
    unsafe { (&*connection).accepted.is_some() }
}

/// The SNI DNS name from the accepted ClientHello.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_server_name(connection: *const TlsConnection) -> Option<String> {
    unsafe { (&*connection).accepted.as_ref() }
        .and_then(|accepted| accepted.client_hello().server_name().map(ToOwned::to_owned))
}

/// Resume an accepted server ClientHello with the context selected by the
/// Python SNI callback.
///
/// # Safety
/// Both pointers must refer to live native values.
#[inline(never)]
pub unsafe fn connection_accept_server(
    connection: *mut TlsConnection,
    context: *const Context,
) -> TlsResult<()> {
    let connection = unsafe { &mut *connection };
    let accepted = connection.accepted.take().ok_or_else(|| {
        (
            TLS_ERROR_SSL,
            "[SSL] no accepted ClientHello is waiting for configuration".to_string(),
        )
    })?;
    connection
        .key_log
        .set_sink(unsafe { &*context }.key_log_file.clone());
    let (mut config, verified_chain_builder) = server_config(unsafe { &*context })?;
    config.key_log = connection.key_log.clone();
    if !config.alpn_protocols.is_empty()
        && let Some(offered) = accepted.client_hello().alpn()
        && !offered
            .into_iter()
            .any(|protocol| config.alpn_protocols.iter().any(|known| known == protocol))
    {
        // OpenSSL completes a handshake without ALPN when both peers offered
        // ALPN but their lists are disjoint. rustls defaults to a fatal
        // no_application_protocol alert, so clear only this connection's
        // immutable config in the disjoint case.
        config.alpn_protocols.clear();
    }
    let config = Arc::new(config);
    match accepted.into_connection(config) {
        Ok(server) => {
            connection.inner = Some(rustls::Connection::Server(server));
            connection.verified_chain_builder = verified_chain_builder;
            connection.verified_chain = None;
            connection.verified_chain_computed = false;
            connection.verified_root_taken = false;
            connection.tls_unique = None;
            connection.server_context = context;
            unsafe { &*context }
                .accept_count
                .fetch_add(1, Ordering::Relaxed);
            connection.process_received_tls()?;
            connection.capture_verified_chain();
            connection.capture_tls_unique();
            Ok(())
        }
        Err((error, mut alert)) => {
            let _ = alert.write_all(&mut connection.pending_tls);
            Err(rustls_protocol_error(error))
        }
    }
}

/// Reject an accepted ClientHello with a caller-selected fatal TLS alert.
/// Alerts at this stage are plaintext TLS records by protocol definition.
///
/// # Safety
/// `connection` must point to a live connection waiting for server config.
#[inline(never)]
pub unsafe fn connection_reject_server(
    connection: *mut TlsConnection,
    alert_description: u8,
) -> TlsResult<()> {
    let connection = unsafe { &mut *connection };
    if connection.accepted.take().is_none() {
        return Err((
            TLS_ERROR_SSL,
            "[SSL] no accepted ClientHello is waiting for rejection".to_string(),
        ));
    }
    let alert = [21, 3, 3, 0, 2, 2, alert_description];
    connection.pending_tls.extend_from_slice(&alert);
    connection.observe_records(true, &alert);
    Ok(())
}

/// Drain protocol events observed since the previous call.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_take_message_events(
    connection: *mut TlsConnection,
) -> Vec<TlsMessageEvent> {
    std::mem::take(unsafe { &mut (&mut *connection).message_events })
}

/// Return RFC 5929 `tls-unique` for a completed TLS 1.2 handshake.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_tls_unique(connection: *mut TlsConnection) -> Option<Vec<u8>> {
    let connection = unsafe { &mut *connection };
    connection.capture_tls_unique();
    connection.tls_unique.clone()
}

/// Return, once, the trust anchor selected from an OpenSSL-style lazy CA
/// directory after a successful client handshake.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_take_verified_root(connection: *mut TlsConnection) -> Option<Vec<u8>> {
    let connection = unsafe { &mut *connection };
    connection.capture_verified_chain();
    if connection.verified_root_taken {
        return None;
    }
    let root = connection.verified_chain.as_ref()?.last()?;
    let builder = connection.verified_chain_builder.as_ref()?;
    let root = builder
        .deferred_roots
        .iter()
        .find(|candidate| *candidate == root)?
        .clone();
    connection.verified_root_taken = true;
    Some(root)
}

/// Drain all currently generated encrypted TLS records.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_take_tls(connection: *mut TlsConnection) -> TlsResult<Vec<u8>> {
    let connection = unsafe { &mut *connection };
    connection.fill_pending_tls()?;
    let output = connection.pending_tls[connection.pending_tls_start..].to_vec();
    connection.pending_tls.clear();
    connection.pending_tls_start = 0;
    Ok(output)
}

/// Copy, but do not consume, generated TLS records. Socket transports use
/// this together with `connection_consume_tls` so partial non-blocking sends
/// cannot lose bytes already drained from rustls.
#[inline(never)]
pub unsafe fn connection_peek_tls(connection: *mut TlsConnection) -> TlsResult<Vec<u8>> {
    let connection = unsafe { &mut *connection };
    connection.fill_pending_tls()?;
    Ok(connection.pending_tls[connection.pending_tls_start..].to_vec())
}

#[inline(never)]
pub unsafe fn connection_consume_tls(connection: *mut TlsConnection, count: usize) {
    let connection = unsafe { &mut *connection };
    connection.pending_tls_start =
        (connection.pending_tls_start + count).min(connection.pending_tls.len());
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_is_handshaking(connection: *const TlsConnection) -> bool {
    unsafe { (&*connection).inner.as_ref() }.is_none_or(|inner| inner.is_handshaking())
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_wants_read(connection: *const TlsConnection) -> bool {
    let connection = unsafe { &*connection };
    connection
        .inner
        .as_ref()
        .map_or(connection.acceptor.is_some(), |inner| inner.wants_read())
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_write_plain(
    connection: *mut TlsConnection,
    data: &[u8],
) -> TlsResult<usize> {
    use std::io::Write;
    let connection = unsafe { &mut *connection };
    if connection.peer_sent_fatal_alert {
        return Err((
            TLS_ERROR_ZERO_RETURN,
            "TLS/SSL connection has been closed (EOF)".to_string(),
        ));
    }
    connection
        .active_mut()?
        .writer()
        .write(data)
        .map_err(rustls_error)
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_read_plain(
    connection: *mut TlsConnection,
    size: usize,
) -> TlsResult<Vec<u8>> {
    use std::io::Read;
    let connection = unsafe { &mut *connection };
    // `_ssl__SSLSocket_read_impl` allocates the destination with
    // `PyBytes_FromStringAndSize(NULL, len)` before it reaches `SSL_read`, so
    // a length no allocator can satisfy raises rather than aborting.  The
    // `resize` below cannot grow past what this reserved, so it cannot abort
    // either.  It also comes before the queued records are fed in: those
    // advance the read cursor and latch the peer's close/alert state, and a
    // refused reservation must not consume them.
    let mut output: Vec<u8> = Vec::new();
    if output.try_reserve_exact(size).is_err() {
        return Err((TLS_ERROR_NO_MEMORY, String::new()));
    }
    output.resize(size, 0);
    connection.process_received_tls()?;
    match connection.active_mut()?.reader().read(&mut output) {
        // A clean close_notify is EOF at the Python stream layer. CPython's
        // SSL_read wrapper returns b"" here; SSLZeroReturnError is reserved
        // for lower-level error reporting paths, not ordinary recv().
        Ok(0) => {
            // `Read::read` answers `Ok(0)` for an empty buffer whatever the
            // connection holds, so a zero-length request is not that
            // observation and must not record a close the peer never sent.
            if size != 0 {
                connection.peer_sent_close_notify = true;
            }
            Ok(Vec::new())
        }
        Ok(read) => {
            output.truncate(read);
            Ok(output)
        }
        // A session the peer ended with a fatal alert has no more plaintext
        // coming, and `SSL_read` answers `SSL_ERROR_ZERO_RETURN` from
        // `SSL_RECEIVED_SHUTDOWN` rather than asking the transport again.
        // rustls' reader reports `WouldBlock` for both, so without the latch
        // the caller goes back to the socket and reports whatever it finds
        // there instead of the alert.
        Err(error)
            if error.kind() == std::io::ErrorKind::WouldBlock
                && connection.peer_sent_fatal_alert =>
        {
            Ok(Vec::new())
        }
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Err((
            TLS_ERROR_WANT_READ,
            "The operation did not complete (read)".to_string(),
        )),
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => Err((
            TLS_ERROR_EOF,
            "EOF occurred in violation of protocol".to_string(),
        )),
        Err(error) => Err(rustls_error(error)),
    }
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_send_close_notify(connection: *mut TlsConnection) {
    if let Some(inner) = unsafe { (&mut *connection).inner.as_mut() } {
        inner.send_close_notify();
    }
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_pending_plaintext(connection: *mut TlsConnection) -> usize {
    unsafe { (&mut *connection).inner.as_mut() }
        .and_then(|inner| inner.process_new_packets().ok())
        .map(|state| state.plaintext_bytes_to_read())
        .unwrap_or(0)
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_peer_closed(connection: *mut TlsConnection) -> bool {
    let connection = unsafe { &mut *connection };
    if connection.peer_sent_close_notify {
        return true;
    }
    let closed = connection
        .inner
        .as_mut()
        .and_then(|inner| inner.process_new_packets().ok())
        .is_some_and(|state| state.peer_has_closed());
    connection.peer_sent_close_notify |= closed;
    connection.peer_sent_close_notify
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_alpn(connection: *const TlsConnection) -> Option<Vec<u8>> {
    unsafe { (&*connection).inner.as_ref() }
        .and_then(|inner| inner.alpn_protocol())
        .map(ToOwned::to_owned)
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_version(connection: *const TlsConnection) -> Option<&'static str> {
    match unsafe { (&*connection).inner.as_ref() }.and_then(|inner| inner.protocol_version()) {
        Some(rustls::ProtocolVersion::TLSv1_2) => Some("TLSv1.2"),
        Some(rustls::ProtocolVersion::TLSv1_3) => Some("TLSv1.3"),
        _ => None,
    }
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_peer_certificate(connection: *const TlsConnection) -> Option<Vec<u8>> {
    unsafe { (&*connection).inner.as_ref() }
        .and_then(|inner| inner.peer_certificates())
        .and_then(|certs| certs.first())
        .map(|cert| cert.as_ref().to_vec())
}

/// Return the certificate list exactly as transmitted by the peer.  This is
/// OpenSSL's `SSL_get_peer_cert_chain` / PyPy's `get_unverified_chain`, not a
/// chain reconstructed from issuer names.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_peer_certificates(connection: *const TlsConnection) -> Vec<Vec<u8>> {
    unsafe { (&*connection).inner.as_ref() }
        .and_then(|inner| inner.peer_certificates())
        .map(|certs| certs.iter().map(|cert| cert.as_ref().to_vec()).collect())
        .unwrap_or_default()
}

/// Return the path selected by certificate verification for this connection.
/// With verification disabled this is the peer's transmitted chain, matching
/// PyPy's `SSL_get0_verified_chain` behavior. A resumed TLS 1.2 session has no
/// new verified path and therefore returns an empty vector.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_verified_certificates(connection: *mut TlsConnection) -> Vec<Vec<u8>> {
    let connection = unsafe { &mut *connection };
    connection.capture_verified_chain();
    connection.verified_chain.clone().unwrap_or_default()
}

#[inline(never)]
pub unsafe fn connection_session_reused(connection: *const TlsConnection) -> Option<bool> {
    let inner = unsafe { (&*connection).inner.as_ref() }?;
    if inner.is_handshaking() {
        return None;
    }
    Some(inner.handshake_kind() == Some(rustls::HandshakeKind::Resumed))
}

/// Snapshot the actual opaque rustls TLS 1.2 resumption value captured for
/// this connection. The returned session keeps its ClientConfig alive so the
/// verifier/credential identity checks performed by rustls remain valid.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_session(connection: *const TlsConnection) -> *mut NativeSession {
    let connection = unsafe { &*connection };
    let Some(inner) = connection.inner.as_ref() else {
        return std::ptr::null_mut();
    };
    if inner.is_handshaking() || inner.protocol_version() != Some(rustls::ProtocolVersion::TLSv1_2)
    {
        return std::ptr::null_mut();
    }
    let (Some(store), Some(config), Some(verified_chain_builder)) = (
        connection.client_session_store.as_ref(),
        connection.client_config.as_ref(),
        connection.verified_chain_builder.as_ref(),
    ) else {
        return std::ptr::null_mut();
    };
    store
        .snapshot(
            connection.context_identity,
            config.clone(),
            verified_chain_builder.clone(),
        )
        .map(|session| Box::into_raw(Box::new(session)))
        .unwrap_or(std::ptr::null_mut())
}

/// Negotiated OpenSSL-style cipher name and effective key size.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_cipher(connection: *const TlsConnection) -> Option<(String, i32)> {
    let suite = unsafe { (&*connection).inner.as_ref() }?.negotiated_cipher_suite()?;
    #[cfg(feature = "host_env")]
    {
        let desc = cipher::describe(&suite);
        Some((desc.name.to_string(), i32::from(desc.bits)))
    }
    #[cfg(not(feature = "host_env"))]
    {
        let name = openssl_cipher_name(suite)?;
        let bits = if name.contains("AES128") { 128 } else { 256 };
        Some((name.to_string(), bits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keylog_appends_header_once_and_disables_on_failed_replacement() {
        let directory = std::env::temp_dir().join(format!(
            "pyre-keylog-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("keys.log");
        let sink = ContextKeyLog::default();
        assert!(!sink.enabled());
        sink.set_path(Some(&path)).unwrap();
        assert!(sink.enabled());
        sink.log("CLIENT_RANDOM", &[0xab, 0x00], &[0x01, 0xff]);
        sink.set_path(Some(&path)).unwrap();
        sink.log("CLIENT_RANDOM", &[0x02], &[0x03]);
        let expected = b"# TLS secrets log file, generated by OpenSSL / Python\n\
                         CLIENT_RANDOM ab00 01ff\nCLIENT_RANDOM 02 03\n";
        assert_eq!(std::fs::read(&path).unwrap(), expected);
        assert!(
            sink.set_path(Some(&directory.join("missing/keys.log")))
                .is_err()
        );
        assert!(!sink.enabled());
        sink.log("CLIENT_RANDOM", &[0x04], &[0x05]);
        assert_eq!(std::fs::read(&path).unwrap(), expected);
        sink.set_path(Some(&path)).unwrap();
        sink.set_path(None).unwrap();
        assert!(!sink.enabled());
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&directory).unwrap();
    }

    const SERVER_CERTIFICATE: &[u8] =
        include_bytes!("../../../lib-python/3/test/certdata/keycert3.pem");

    fn server_der() -> Vec<u8> {
        read_pem_certificates(SERVER_CERTIFICATE).unwrap()[0]
            .as_ref()
            .to_vec()
    }

    #[test]
    fn certificate_subject_matches_openssl_rfc2253() {
        assert_eq!(
            certificate_subject_rfc2253(&server_der()).unwrap(),
            "CN=localhost,O=Python Software Foundation,L=Castle Anthrax,C=XY"
        );
    }

    #[test]
    fn certificate_subject_hash_matches_openssl_canonical_name() {
        assert_eq!(
            certificate_subject_hash(&server_der()).unwrap(),
            0x2aff_206c
        );
    }

    #[cfg(feature = "host_env")]
    #[test]
    fn tls_unique_matches_across_client_and_server() {
        let cert_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../lib-python/3/test/certdata/keycert3.pem");
        let client_ctx = context_new(PROTOCOL_TLS_CLIENT).unwrap();
        let server_ctx = context_new(PROTOCOL_TLS_SERVER).unwrap();
        unsafe {
            context_set_verify_mode(client_ctx, CERT_NONE);
            context_set_check_hostname(client_ctx, false);
            context_set_maximum_version(client_ctx, 0x303);
            context_set_maximum_version(server_ctx, 0x303);
            context_load_cert_chain(server_ctx, &cert_path, &cert_path, None).unwrap();
            let client =
                connection_new(client_ctx, false, Some("localhost"), std::ptr::null()).unwrap();
            let server = connection_new(server_ctx, true, None, std::ptr::null()).unwrap();
            for _ in 0..16 {
                let c2s = connection_take_tls(client).unwrap();
                if !c2s.is_empty() {
                    connection_receive_tls(server, &c2s).unwrap();
                }
                if connection_waiting_for_server_config(server) {
                    connection_accept_server(server, server_ctx).unwrap();
                }
                let s2c = connection_take_tls(server).unwrap();
                if !s2c.is_empty() {
                    connection_receive_tls(client, &s2c).unwrap();
                }
                if !connection_is_handshaking(client) && !connection_is_handshaking(server) {
                    break;
                }
                assert!(!c2s.is_empty() || !s2c.is_empty(), "handshake stalled");
            }
            let client_unique = connection_tls_unique(client);
            let server_unique = connection_tls_unique(server);
            connection_free(client);
            connection_free(server);
            context_free(client_ctx);
            context_free(server_ctx);
            assert_eq!(client_unique, server_unique);
            assert_eq!(client_unique.as_ref().map(Vec::len), Some(12));
        }
    }

    #[cfg(feature = "host_env")]
    unsafe fn handshake_pair(
        client: *mut TlsConnection,
        server: *mut TlsConnection,
        server_ctx: *mut Context,
    ) -> Result<(), String> {
        unsafe {
            for _ in 0..32 {
                let c2s = connection_take_tls(client).map_err(|error| format!("{error:?}"))?;
                if !c2s.is_empty() {
                    connection_receive_tls(server, &c2s).map_err(|error| format!("{error:?}"))?;
                }
                if connection_waiting_for_server_config(server) {
                    connection_accept_server(server, server_ctx)
                        .map_err(|error| format!("{error:?}"))?;
                }
                let s2c = connection_take_tls(server).map_err(|error| format!("{error:?}"))?;
                if !s2c.is_empty() {
                    connection_receive_tls(client, &s2c).map_err(|error| format!("{error:?}"))?;
                }
                if !connection_is_handshaking(client) && !connection_is_handshaking(server) {
                    return Ok(());
                }
                if c2s.is_empty() && s2c.is_empty() {
                    return Err("handshake stalled".into());
                }
            }
            Err("handshake did not finish".into())
        }
    }

    #[cfg(feature = "host_env")]
    unsafe fn tls13_pair(
        configure: impl Fn(*mut Context, *mut Context),
    ) -> (
        *mut Context,
        *mut Context,
        *mut TlsConnection,
        *mut TlsConnection,
    ) {
        unsafe {
            let cert_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../lib-python/3/test/certdata/keycert3.pem");
            let client_ctx = context_new(PROTOCOL_TLS_CLIENT).unwrap();
            let server_ctx = context_new(PROTOCOL_TLS_SERVER).unwrap();
            context_set_verify_mode(client_ctx, CERT_NONE);
            context_set_check_hostname(client_ctx, false);
            context_load_cert_chain(server_ctx, &cert_path, &cert_path, None).unwrap();
            configure(client_ctx, server_ctx);
            let client =
                connection_new(client_ctx, false, Some("localhost"), std::ptr::null()).unwrap();
            let server = connection_new(server_ctx, true, None, std::ptr::null()).unwrap();
            (client_ctx, server_ctx, client, server)
        }
    }

    #[cfg(feature = "host_env")]
    unsafe fn free_pair(
        client_ctx: *mut Context,
        server_ctx: *mut Context,
        client: *mut TlsConnection,
        server: *mut TlsConnection,
    ) {
        unsafe {
            connection_free(client);
            connection_free(server);
            context_free(client_ctx);
            context_free(server_ctx);
        }
    }

    #[test]
    fn set_ciphersuites_rejects_unknown_only_lists() {
        let ctx = context_new(PROTOCOL_TLS_CLIENT).unwrap();
        unsafe {
            assert_eq!(
                context_set_ciphersuites(ctx, "XXX").unwrap_err(),
                "No cipher suite can be selected."
            );
            assert_eq!(
                context_set_ciphersuites(ctx, ":").unwrap_err(),
                "No cipher suite can be selected."
            );
            assert!(context_set_ciphersuites(ctx, "").is_ok());
            assert!(context_set_ciphersuites(ctx, "TLS_AES_128_GCM_SHA256:XXX").is_ok());
            assert!(context_cipher_enabled(ctx, 0));
            context_set_ciphersuites(ctx, "TLS_AES_128_GCM_SHA256").unwrap();
            assert!(context_cipher_enabled(ctx, 0));
            assert!(!context_cipher_enabled(ctx, 1));
            context_free(ctx);
        }
    }

    #[test]
    fn set_groups_parses_openssl_names() {
        let ctx = context_new(PROTOCOL_TLS_CLIENT).unwrap();
        unsafe {
            assert!(context_set_groups(ctx, "P-256:P-384").is_ok());
            assert!(context_set_groups(ctx, "secp384r1").is_ok());
            assert!(context_set_groups(ctx, "prime256v1").is_ok());
            assert!(context_set_groups(ctx, "").is_ok());
            assert_eq!(
                context_set_groups(ctx, "P-256:foo").unwrap_err(),
                "unrecognized group"
            );
            assert_eq!(
                context_set_groups(ctx, ":").unwrap_err(),
                "unrecognized group"
            );
            context_free(ctx);
        }
    }

    #[test]
    fn set_server_sigalgs_parses_openssl_names() {
        let ctx = context_new(PROTOCOL_TLS_CLIENT).unwrap();
        unsafe {
            assert!(context_set_server_sigalgs(ctx, "rsa_pss_rsae_sha256").is_ok());
            assert!(context_set_server_sigalgs(ctx, "rsa_pss_rsae_sha384").is_ok());
            assert_eq!(
                context_set_server_sigalgs(ctx, "rsa_pss_rsae_sha256:foo").unwrap_err(),
                "unrecognized signature algorithm"
            );
            assert_eq!(
                context_set_server_sigalgs(ctx, "").unwrap_err(),
                "unrecognized signature algorithm"
            );
            context_free(ctx);
        }
    }

    #[cfg(feature = "host_env")]
    #[test]
    fn ciphersuites_constrain_tls13_handshake() {
        unsafe {
            let (client_ctx, server_ctx, client, server) = tls13_pair(|client, server| {
                context_set_ciphersuites(client, "TLS_AES_128_GCM_SHA256").unwrap();
                context_set_ciphersuites(server, "TLS_AES_128_GCM_SHA256").unwrap();
            });
            handshake_pair(client, server, server_ctx).unwrap();
            let (name, _) = connection_cipher(client).expect("negotiated cipher");
            assert_eq!(name, "TLS_AES_128_GCM_SHA256");
            free_pair(client_ctx, server_ctx, client, server);

            let (client_ctx, server_ctx, client, server) = tls13_pair(|client, server| {
                context_set_ciphersuites(client, "TLS_AES_128_GCM_SHA256").unwrap();
                context_set_ciphersuites(server, "TLS_AES_256_GCM_SHA384").unwrap();
            });
            assert!(handshake_pair(client, server, server_ctx).is_err());
            free_pair(client_ctx, server_ctx, client, server);
        }
    }

    #[cfg(feature = "host_env")]
    #[test]
    fn groups_constrain_tls13_handshake() {
        unsafe {
            let (client_ctx, server_ctx, client, server) = tls13_pair(|_client, server| {
                context_set_groups(server, "secp384r1").unwrap();
            });
            handshake_pair(client, server, server_ctx).unwrap();
            free_pair(client_ctx, server_ctx, client, server);

            let (client_ctx, server_ctx, client, server) = tls13_pair(|client, server| {
                context_set_groups(client, "prime256v1").unwrap();
                context_set_groups(server, "secp384r1").unwrap();
            });
            assert!(handshake_pair(client, server, server_ctx).is_err());
            free_pair(client_ctx, server_ctx, client, server);
        }
    }

    #[cfg(feature = "host_env")]
    #[test]
    fn server_sigalgs_constrain_tls13_handshake() {
        unsafe {
            let (client_ctx, server_ctx, client, server) = tls13_pair(|_client, server| {
                context_set_server_sigalgs(server, "rsa_pss_rsae_sha384").unwrap();
            });
            handshake_pair(client, server, server_ctx).unwrap();
            free_pair(client_ctx, server_ctx, client, server);

            let (client_ctx, server_ctx, client, server) = tls13_pair(|client, server| {
                context_set_server_sigalgs(client, "rsa_pss_rsae_sha256").unwrap();
                context_set_server_sigalgs(server, "rsa_pss_rsae_sha384").unwrap();
            });
            assert!(handshake_pair(client, server, server_ctx).is_err());
            free_pair(client_ctx, server_ctx, client, server);
        }
    }

    #[test]
    fn oid_lookup_uses_openssl_object_database() {
        let cn = oid_by_nid(13).unwrap();
        assert_eq!(cn.short_name, "CN");
        assert_eq!(cn.long_name, "commonName");
        assert_eq!(cn.oid, Some("2.5.4.3"));
        assert_eq!(oid_by_name("serverAuth").unwrap().nid, 129);
        assert!(oid_by_name("serverauth").is_none());
        assert_eq!(oid_by_oid_string("2.5.4.3").unwrap().nid, 13);
        assert_eq!(oid_by_nid(130).unwrap().short_name, "clientAuth");
        #[cfg(feature = "host_env")]
        {
            assert_eq!(oid_by_oid_string("2.005.004.003.").unwrap().nid, 13);
            assert_eq!(oid_by_nid(181).unwrap().oid, None);
        }
    }
}
