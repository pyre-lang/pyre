//! Wasm `_ssl` backend. rustls + aws-lc cannot build for
//! `wasm32-unknown-unknown`, so TLS operations fail. MemoryBIO, constants,
//! OID tables, and ALPN parsing come from `rustpython_host_env::ssl`.

use rustpython_host_env::ssl as host_ssl;

const UNAVAILABLE: &str = "TLS is not available on this platform";

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

pub type NativeResult<T> = Result<T, (i32, String)>;
pub type TlsResult<T> = Result<T, (i32, String)>;

bitflags::bitflags! {
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

fn unavailable<T>() -> NativeResult<T> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}

pub fn ensure_provider() {}

pub type MemoryBio = host_ssl::MemoryBio;

pub fn memory_bio_new() -> *mut MemoryBio {
    Box::into_raw(Box::new(MemoryBio::new()))
}

pub unsafe fn memory_bio_free(bio: *mut MemoryBio) {
    if !bio.is_null() {
        unsafe { drop(Box::from_raw(bio)) };
    }
}

pub unsafe fn memory_bio_read(bio: *mut MemoryBio, size: usize) -> Vec<u8> {
    unsafe { (*bio).read(size) }
}

pub unsafe fn memory_bio_write(bio: *mut MemoryBio, data: &[u8]) -> Result<usize, &'static str> {
    unsafe { (*bio).write(data) }.map_err(|_| "cannot write() after write_eof()")
}

pub unsafe fn memory_bio_write_eof(bio: *mut MemoryBio) {
    unsafe { (*bio).write_eof() };
}

pub unsafe fn memory_bio_pending(bio: *const MemoryBio) -> usize {
    unsafe { (*bio).pending() }
}

pub unsafe fn memory_bio_eof(bio: *const MemoryBio) -> bool {
    unsafe { (*bio).eof() }
}

pub struct Context {
    identity: usize,
    protocol: i32,
    check_hostname: bool,
    verify_mode: i32,
    verify_flags: i32,
    options: u64,
    minimum_version: i32,
    maximum_version: i32,
}

static NEXT_CONTEXT_IDENTITY: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(1);

pub fn context_new(protocol: i32) -> Result<*mut Context, &'static str> {
    Ok(Box::into_raw(Box::new(Context {
        identity: NEXT_CONTEXT_IDENTITY.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        protocol,
        check_hostname: protocol == PROTOCOL_TLS_CLIENT,
        verify_mode: if protocol == PROTOCOL_TLS_CLIENT {
            CERT_REQUIRED
        } else {
            CERT_NONE
        },
        verify_flags: VerifyFlags::DEFAULT.bits(),
        options: SslOp::DEFAULT.bits(),
        minimum_version: -2,
        maximum_version: -1,
    })))
}

pub unsafe fn context_free(context: *mut Context) {
    if !context.is_null() {
        unsafe { drop(Box::from_raw(context)) };
    }
}

macro_rules! context_scalar {
    ($get:ident, $set:ident, $field:ident, $ty:ty) => {
        pub unsafe fn $get(context: *const Context) -> $ty {
            unsafe { (*context).$field }
        }
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

pub unsafe fn context_identity(context: *const Context) -> usize {
    unsafe { (*context).identity }
}
pub unsafe fn context_set_num_tickets(_context: *mut Context, _tickets: usize) {}
pub unsafe fn context_set_keylog_filename(
    _context: *mut Context,
    _path: Option<&std::path::Path>,
) -> NativeResult<()> {
    unavailable()
}
pub unsafe fn context_session_stats(_context: *const Context) -> (usize, usize) {
    (0, 0)
}
pub fn parse_length_prefixed_alpn(data: &[u8]) -> Result<Vec<Vec<u8>>, &'static str> {
    host_ssl::parse_length_prefixed_alpn(data).map_err(|_| "invalid ALPN protocol list")
}
pub unsafe fn context_set_alpn(_context: *mut Context, _protocols: Vec<Vec<u8>>) {}
pub unsafe fn context_add_roots(_context: *mut Context, _ders: Vec<Vec<u8>>) {}
pub unsafe fn context_load_cert_chain(
    _context: *mut Context,
    _cert: &std::path::Path,
    _key: &std::path::Path,
    _password: Option<&[u8]>,
) -> NativeResult<bool> {
    unavailable()
}
pub unsafe fn context_load_verify_file(
    _context: *mut Context,
    _path: &std::path::Path,
) -> NativeResult<usize> {
    unavailable()
}
pub unsafe fn context_add_verify_dir(_context: *mut Context, _path: &std::path::Path) {}
pub unsafe fn context_add_verified_root(
    _context: *mut Context,
    _der: Vec<u8>,
) -> NativeResult<bool> {
    unavailable()
}
pub unsafe fn context_load_verify_data(
    _context: *mut Context,
    _data: &[u8],
    _pem: bool,
) -> NativeResult<usize> {
    unavailable()
}
pub fn default_verify_paths() -> (String, String) {
    (String::new(), String::new())
}
pub unsafe fn context_load_native_roots(_context: *mut Context) -> NativeResult<usize> {
    unavailable()
}
pub unsafe fn context_cert_store_stats(_context: *const Context) -> (usize, usize) {
    (0, 0)
}
pub unsafe fn context_ca_certs(_context: *const Context) -> Vec<Vec<u8>> {
    Vec::new()
}
pub unsafe fn context_cipher_enabled(_context: *const Context, _index: usize) -> bool {
    false
}
pub unsafe fn context_set_cipher_list(
    _context: *mut Context,
    _list: &str,
) -> Result<(), &'static str> {
    Err(UNAVAILABLE)
}
pub unsafe fn context_set_ecdh_curve(
    _context: *mut Context,
    _curve: &str,
) -> Result<(), &'static str> {
    Err(UNAVAILABLE)
}
pub unsafe fn context_set_ciphersuites(
    _context: *mut Context,
    _list: &str,
) -> Result<(), &'static str> {
    Err(UNAVAILABLE)
}
pub unsafe fn context_set_groups(_context: *mut Context, _list: &str) -> Result<(), &'static str> {
    Err(UNAVAILABLE)
}
pub unsafe fn context_set_server_sigalgs(
    _context: *mut Context,
    _list: &str,
) -> Result<(), &'static str> {
    Err(UNAVAILABLE)
}
pub unsafe fn context_set_client_sigalgs(
    _context: *mut Context,
    _list: &str,
) -> Result<(), &'static str> {
    Err(UNAVAILABLE)
}

pub struct DecodedCertificate;

pub fn certificate_decode_der(_der: &[u8]) -> NativeResult<*mut DecodedCertificate> {
    unavailable()
}
pub fn certificate_decode_file(_path: &std::path::Path) -> NativeResult<*mut DecodedCertificate> {
    unavailable()
}
pub fn certificate_subject_rfc2253(_der: &[u8]) -> NativeResult<String> {
    unavailable()
}
pub fn certificate_subject_hash(_der: &[u8]) -> NativeResult<i64> {
    unavailable()
}
pub unsafe fn certificate_free(cert: *mut DecodedCertificate) {
    if !cert.is_null() {
        unsafe { drop(Box::from_raw(cert)) };
    }
}
pub unsafe fn certificate_version(_cert: *const DecodedCertificate) -> i32 {
    0
}
pub unsafe fn certificate_name_rdn_count(
    _cert: *const DecodedCertificate,
    _subject: bool,
) -> usize {
    0
}
pub unsafe fn certificate_name_attribute_count(
    _cert: *const DecodedCertificate,
    _subject: bool,
    _rdn: usize,
) -> usize {
    0
}
pub unsafe fn certificate_name_attribute_key(
    _cert: *const DecodedCertificate,
    _subject: bool,
    _rdn: usize,
    _attribute: usize,
) -> String {
    String::new()
}
pub unsafe fn certificate_name_attribute_value(
    _cert: *const DecodedCertificate,
    _subject: bool,
    _rdn: usize,
    _attribute: usize,
) -> String {
    String::new()
}
pub unsafe fn certificate_url_count(_cert: *const DecodedCertificate, _kind: i32) -> usize {
    0
}
pub unsafe fn certificate_url(
    _cert: *const DecodedCertificate,
    _kind: i32,
    _index: usize,
) -> String {
    String::new()
}
pub unsafe fn certificate_san_count(_cert: *const DecodedCertificate) -> usize {
    0
}
pub unsafe fn certificate_san_kind(
    _cert: *const DecodedCertificate,
    _index: usize,
) -> &'static str {
    ""
}
pub unsafe fn certificate_san_value(_cert: *const DecodedCertificate, _index: usize) -> String {
    String::new()
}
pub unsafe fn certificate_san_directory_rdn_count(
    _cert: *const DecodedCertificate,
    _san: usize,
) -> usize {
    0
}
pub unsafe fn certificate_san_directory_attribute_count(
    _cert: *const DecodedCertificate,
    _san: usize,
    _rdn: usize,
) -> usize {
    0
}
pub unsafe fn certificate_san_directory_attribute_key(
    _cert: *const DecodedCertificate,
    _san: usize,
    _rdn: usize,
    _attribute: usize,
) -> String {
    String::new()
}
pub unsafe fn certificate_san_directory_attribute_value(
    _cert: *const DecodedCertificate,
    _san: usize,
    _rdn: usize,
    _attribute: usize,
) -> String {
    String::new()
}
pub unsafe fn certificate_not_after(_cert: *const DecodedCertificate) -> String {
    String::new()
}
pub unsafe fn certificate_not_before(_cert: *const DecodedCertificate) -> String {
    String::new()
}
pub unsafe fn certificate_serial_number(_cert: *const DecodedCertificate) -> String {
    String::new()
}

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

pub fn oid_by_nid(nid: i32) -> Option<OidInfo> {
    host_ssl::oid::find_by_nid(nid).map(oid_info)
}
pub fn oid_by_oid_string(oid: &str) -> Option<OidInfo> {
    host_ssl::oid::find_by_oid_string(oid).map(oid_info)
}
pub fn oid_by_name(name: &str) -> Option<OidInfo> {
    host_ssl::oid::find_by_name(name).map(oid_info)
}

pub fn cipher_count() -> usize {
    0
}
pub fn cipher_name(_index: usize) -> &'static str {
    ""
}
pub fn cipher_id(_index: usize) -> i64 {
    0
}
pub fn cipher_description(_index: usize) -> String {
    String::new()
}
pub fn cipher_protocol(_index: usize) -> &'static str {
    ""
}
pub fn cipher_bits(_index: usize) -> i32 {
    0
}
pub fn cipher_aead(_index: usize) -> bool {
    false
}
pub fn cipher_symmetric(_index: usize) -> &'static str {
    ""
}
pub fn cipher_digest(_index: usize) -> &'static str {
    ""
}
pub fn cipher_kea(_index: usize) -> &'static str {
    ""
}
pub fn cipher_auth(_index: usize) -> &'static str {
    ""
}
pub fn validate_cipher_string(_pattern: &str) -> Result<(), &'static str> {
    Err(UNAVAILABLE)
}
pub fn default_cipher_string() -> String {
    String::new()
}

pub struct NativeSession {
    context_identity: usize,
}

pub unsafe fn session_clone(session: *const NativeSession) -> *mut NativeSession {
    if session.is_null() {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(NativeSession {
        context_identity: unsafe { (*session).context_identity },
    }))
}
pub unsafe fn session_free(session: *mut NativeSession) {
    if !session.is_null() {
        unsafe { drop(Box::from_raw(session)) };
    }
}
pub unsafe fn session_context_identity(session: *const NativeSession) -> usize {
    if session.is_null() {
        0
    } else {
        unsafe { (*session).context_identity }
    }
}
pub unsafe fn session_id(_session: *const NativeSession) -> Vec<u8> {
    Vec::new()
}
pub unsafe fn session_creation_time(_session: *const NativeSession) -> u64 {
    0
}
pub unsafe fn session_timeout(_session: *const NativeSession) -> u64 {
    0
}
pub unsafe fn session_has_ticket(_session: *const NativeSession) -> bool {
    false
}
pub unsafe fn session_ticket_lifetime_hint(_session: *const NativeSession) -> u64 {
    0
}

pub struct TlsMessageEvent {
    pub write: bool,
    pub version: u16,
    pub content_type: u16,
    pub message_type: u16,
    pub data: Vec<u8>,
}

pub fn certificate_verify_message(_code: i32) -> &'static str {
    UNAVAILABLE
}

pub struct TlsConnection;

pub unsafe fn connection_new(
    _context: *const Context,
    _server_side: bool,
    _server_hostname: Option<&str>,
    _session: *const NativeSession,
) -> NativeResult<*mut TlsConnection> {
    unavailable()
}
pub unsafe fn connection_free(connection: *mut TlsConnection) {
    if !connection.is_null() {
        unsafe { drop(Box::from_raw(connection)) };
    }
}
pub unsafe fn connection_receive_tls(
    _connection: *mut TlsConnection,
    _data: &[u8],
) -> TlsResult<usize> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_waiting_for_server_config(_connection: *const TlsConnection) -> bool {
    false
}
pub unsafe fn connection_server_name(_connection: *const TlsConnection) -> Option<String> {
    None
}
pub unsafe fn connection_accept_server(
    _connection: *mut TlsConnection,
    _context: *const Context,
) -> TlsResult<()> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_reject_server(
    _connection: *mut TlsConnection,
    _alert: u8,
) -> TlsResult<()> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_take_message_events(
    _connection: *mut TlsConnection,
) -> Vec<TlsMessageEvent> {
    Vec::new()
}
pub unsafe fn connection_tls_unique(_connection: *mut TlsConnection) -> Option<Vec<u8>> {
    None
}
pub unsafe fn connection_take_verified_root(_connection: *mut TlsConnection) -> Option<Vec<u8>> {
    None
}
pub unsafe fn connection_take_tls(_connection: *mut TlsConnection) -> TlsResult<Vec<u8>> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_peek_tls(_connection: *mut TlsConnection) -> TlsResult<Vec<u8>> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_consume_tls(_connection: *mut TlsConnection, _count: usize) {}
pub unsafe fn connection_is_handshaking(_connection: *const TlsConnection) -> bool {
    false
}
pub unsafe fn connection_wants_read(_connection: *const TlsConnection) -> bool {
    false
}
pub unsafe fn connection_write_plain(
    _connection: *mut TlsConnection,
    _data: &[u8],
) -> TlsResult<usize> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_read_plain(
    _connection: *mut TlsConnection,
    _size: usize,
) -> TlsResult<Vec<u8>> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_send_close_notify(_connection: *mut TlsConnection) {}
pub unsafe fn connection_pending_plaintext(_connection: *mut TlsConnection) -> usize {
    0
}
pub unsafe fn connection_peer_closed(_connection: *mut TlsConnection) -> bool {
    false
}
pub unsafe fn connection_alpn(_connection: *const TlsConnection) -> Option<Vec<u8>> {
    None
}
pub unsafe fn connection_version(_connection: *const TlsConnection) -> Option<&'static str> {
    None
}
pub unsafe fn connection_peer_certificate(_connection: *const TlsConnection) -> Option<Vec<u8>> {
    None
}
pub unsafe fn connection_peer_certificates(_connection: *const TlsConnection) -> Vec<Vec<u8>> {
    Vec::new()
}
pub unsafe fn connection_verified_certificates(_connection: *mut TlsConnection) -> Vec<Vec<u8>> {
    Vec::new()
}
pub unsafe fn connection_session_reused(_connection: *const TlsConnection) -> Option<bool> {
    None
}
pub unsafe fn connection_session(_connection: *const TlsConnection) -> *mut NativeSession {
    std::ptr::null_mut()
}
pub unsafe fn connection_cipher(_connection: *const TlsConnection) -> Option<(String, i32)> {
    None
}
