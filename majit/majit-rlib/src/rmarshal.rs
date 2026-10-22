//! Byte-exact port of the RPython sandbox wire format.
//!
//! Two upstream files define the format, and they differ in one respect that is
//! load-bearing for interop:
//!
//!   - `rpython/rlib/rmarshal.py` is the codec used by the sandboxed *client*
//!     (`rsandbox.py`). On a 64-bit host its `dump_int` always emits
//!     `TYPE_INT64` (`rmarshal.py` -> `dump_longlong`).
//!   - `rpython/translator/sandbox/_marshal.py` is the *controller* codec
//!     (`sandlib.py`). Its `dump_int` emits `TYPE_INT` for ints that fit in 31
//!     bits and `TYPE_INT64` otherwise (`_marshal.py`).
//!
//! Both *decoders* accept either int tag, which is why the wire interoperates.
//! [`IntFlavor`] selects the encoder convention; the loaders accept both.

use crate::rsandbox::{SandboxError, SandboxResult};

// rmarshal.py:71-80
pub const TYPE_NONE: u8 = b'N';
pub const TYPE_FALSE: u8 = b'F';
pub const TYPE_TRUE: u8 = b'T';
pub const TYPE_INT: u8 = b'i';
pub const TYPE_INT64: u8 = b'I';
pub const TYPE_FLOAT: u8 = b'f';
pub const TYPE_STRING: u8 = b's';
pub const TYPE_TUPLE: u8 = b'(';
pub const TYPE_LIST: u8 = b'[';
pub const TYPE_DICT: u8 = b'{';

/// End-of-dict marker byte (`rmarshal.py:436`, the ASCII char `'0'`).
const DICT_END: u8 = b'0';

/// Which `dump_int` convention to use when encoding ints.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntFlavor {
    /// `rmarshal.py` (client): every int is `TYPE_INT64`.
    Rmarshal,
    /// `_marshal.py` (controller): small ints `TYPE_INT`, large `TYPE_INT64`.
    Marshal,
}

/// A generic marshalled value. Strings hold raw bytes (the wire allows embedded
/// NULs and non-UTF-8 — e.g. `os.read` of binary data), so this is `Vec<u8>`,
/// not `String`.
#[derive(Clone, Debug, PartialEq)]
pub enum MarshalValue {
    None,
    Bool(bool),
    Int(i64),
    Str(Vec<u8>),
    Float(f64),
    Tuple(Vec<MarshalValue>),
    List(Vec<MarshalValue>),
    Dict(Vec<(MarshalValue, MarshalValue)>),
}

/// 10-field wire `os.stat_result` (`vfs.py` `StatResult`, format
/// `"iIIiiiIfff"` in `sandlib.py` `RESULTTYPE_STATRESULT`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatResult {
    pub st_mode: u32,
    pub st_ino: u64,
    pub st_dev: u64,
    pub st_nlink: u64,
    pub st_uid: u32,
    pub st_gid: u32,
    pub st_size: u64,
    pub st_atime: i64,
    pub st_mtime: i64,
    pub st_ctime: i64,
}

/// Argument side of `rmarshal.get_marshaller` — the trampoline dumps each
/// argument with the client int convention ([`IntFlavor::Rmarshal`]).
pub trait SandboxArg {
    fn to_marshal(&self) -> MarshalValue;
}

macro_rules! sandbox_arg_int {
    ($($t:ty),+ $(,)?) => {$(
        impl SandboxArg for $t {
            fn to_marshal(&self) -> MarshalValue {
                MarshalValue::Int(*self as i64)
            }
        }
    )+};
}

sandbox_arg_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl SandboxArg for bool {
    fn to_marshal(&self) -> MarshalValue {
        MarshalValue::Bool(*self)
    }
}

impl SandboxArg for f64 {
    fn to_marshal(&self) -> MarshalValue {
        MarshalValue::Float(*self)
    }
}

impl SandboxArg for [u8] {
    fn to_marshal(&self) -> MarshalValue {
        MarshalValue::Str(self.to_vec())
    }
}

impl SandboxArg for Vec<u8> {
    fn to_marshal(&self) -> MarshalValue {
        MarshalValue::Str(self.clone())
    }
}

impl<'a> SandboxArg for &'a [u8] {
    fn to_marshal(&self) -> MarshalValue {
        MarshalValue::Str((*self).to_vec())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Encoders (append to a Vec<u8>, mirroring rmarshal's "buf is a list of chars").
// ─────────────────────────────────────────────────────────────────────────────

/// rmarshal.py `w_long` — the low 32 bits, little-endian.
pub fn w_long(buf: &mut Vec<u8>, x: i64) {
    buf.extend_from_slice(&(x as u32).to_le_bytes());
}

/// rmarshal.py `dump_longlong` body — low dword then high dword.
fn w_long64(buf: &mut Vec<u8>, x: i64) {
    w_long(buf, x);
    w_long(buf, x >> 32);
}

/// rmarshal.py dump_none
pub fn dump_none(buf: &mut Vec<u8>) {
    buf.push(TYPE_NONE);
}

/// rmarshal.py dump_bool
pub fn dump_bool(buf: &mut Vec<u8>, x: bool) {
    buf.push(if x { TYPE_TRUE } else { TYPE_FALSE });
}

/// rmarshal.py dump_int (client) / _marshal.py (controller).
pub fn dump_int(buf: &mut Vec<u8>, x: i64, flavor: IntFlavor) {
    match flavor {
        IntFlavor::Rmarshal => {
            buf.push(TYPE_INT64);
            w_long64(buf, x);
        }
        IntFlavor::Marshal => {
            let y = x >> 31;
            if y != 0 && y != -1 {
                buf.push(TYPE_INT64);
                w_long64(buf, x);
            } else {
                buf.push(TYPE_INT);
                w_long(buf, x);
            }
        }
    }
}

/// rmarshal.py `dump_string_or_none` (the non-None branch).
pub fn dump_string(buf: &mut Vec<u8>, s: &[u8]) {
    buf.push(TYPE_STRING);
    w_long(buf, s.len() as i64);
    buf.extend_from_slice(s);
}

/// rmarshal.py `dump_float` — `formatd(x, 'g', 17)` with a single length
/// byte.
///
/// The wire field is `'f'` + one length byte + an ASCII float that the peer
/// parses back with `float()` (`rmarshal.py` / `_marshal.py:...`). The
/// single length byte caps the text at 255 chars, so `%.17g` is the right
/// choice: it always round-trips and stays compact (it uses an exponent for
/// large/small magnitudes), unlike a plain decimal expansion. The controller
/// (`_marshal.py`) nominally uses `repr(x)`; for an all-Rust client+controller
/// the exact digits are an implementation detail of the format, and `%.17g`
/// round-trips identically.
pub fn dump_float(buf: &mut Vec<u8>, x: f64) {
    let s = c_format_g(x, 17);
    buf.push(TYPE_FLOAT);
    buf.push(s.len() as u8);
    buf.extend_from_slice(&s);
}

/// rmarshal.py `dump_tuple`.
pub fn dump_tuple(buf: &mut Vec<u8>, items: &[MarshalValue], flavor: IntFlavor) {
    buf.push(TYPE_TUPLE);
    w_long(buf, items.len() as i64);
    for item in items {
        dump_value(buf, item, flavor);
    }
}

/// rmarshal.py `dump_list_or_none` (non-None branch).
pub fn dump_list(buf: &mut Vec<u8>, items: &[MarshalValue], flavor: IntFlavor) {
    buf.push(TYPE_LIST);
    w_long(buf, items.len() as i64);
    for item in items {
        dump_value(buf, item, flavor);
    }
}

/// rmarshal.py `dump_dict_or_none` (non-None branch).
pub fn dump_dict(buf: &mut Vec<u8>, items: &[(MarshalValue, MarshalValue)], flavor: IntFlavor) {
    buf.push(TYPE_DICT);
    for (key, value) in items {
        dump_value(buf, key, flavor);
        dump_value(buf, value, flavor);
    }
    buf.push(DICT_END);
}

/// Dispatch encoder for a [`MarshalValue`].
pub fn dump_value(buf: &mut Vec<u8>, value: &MarshalValue, flavor: IntFlavor) {
    match value {
        MarshalValue::None => dump_none(buf),
        MarshalValue::Bool(b) => dump_bool(buf, *b),
        MarshalValue::Int(i) => dump_int(buf, *i, flavor),
        MarshalValue::Str(s) => dump_string(buf, s),
        MarshalValue::Float(f) => dump_float(buf, *f),
        MarshalValue::Tuple(items) => dump_tuple(buf, items, flavor),
        MarshalValue::List(items) => dump_list(buf, items, flavor),
        MarshalValue::Dict(items) => dump_dict(buf, items, flavor),
    }
}

// ── The two hand-packed reply encoders (sandlib.py:43-64) ────────────────────

/// `RESULTTYPE_STATRESULT` (`sandlib.py`), format string `"iIIiiiIfff"`
/// over the 10-field `os.stat_result`. rmarshal's stat loader insists on the
/// exact per-field int widths, so this is hand-packed rather than a plain tuple.
pub fn dump_statresult(buf: &mut Vec<u8>, st: &StatResult) {
    buf.push(TYPE_TUPLE);
    w_long(buf, 10);
    pack_i(buf, st.st_mode as i64); // st_mode
    pack_big(buf, st.st_ino as i64); // st_ino
    pack_big(buf, st.st_dev as i64); // st_dev
    pack_i(buf, st.st_nlink as i64); // st_nlink
    pack_i(buf, st.st_uid as i64); // st_uid
    pack_i(buf, st.st_gid as i64); // st_gid
    pack_big(buf, st.st_size as i64); // st_size
    pack_g(buf, st.st_atime as f64); // st_atime
    pack_g(buf, st.st_mtime as f64); // st_mtime
    pack_g(buf, st.st_ctime as f64); // st_ctime
}

/// `RESULTTYPE_LONGLONG` (`sandlib.py`), `struct.pack("<cq", 'I', msg)`.
pub fn dump_longlong_result(buf: &mut Vec<u8>, v: i64) {
    buf.push(TYPE_INT64);
    buf.extend_from_slice(&v.to_le_bytes());
}

// `struct.pack("<ci", 'i', v)` — tag then 4-byte LE.
fn pack_i(buf: &mut Vec<u8>, v: i64) {
    buf.push(TYPE_INT);
    w_long(buf, v);
}

// `struct.pack("<cq", 'I', v)` — tag then 8-byte LE.
fn pack_big(buf: &mut Vec<u8>, v: i64) {
    buf.push(TYPE_INT64);
    buf.extend_from_slice(&v.to_le_bytes());
}

// `'f'` + one length byte + `"%g" % v`.
fn pack_g(buf: &mut Vec<u8>, v: f64) {
    let s = c_format_g(v, 6);
    buf.push(TYPE_FLOAT);
    buf.push(s.len() as u8);
    buf.extend_from_slice(&s);
}

// ─────────────────────────────────────────────────────────────────────────────
// Loaders (a streaming Loader over a NeedMore source — port of rmarshal.Loader
// + rsandbox.FdLoader).
// ─────────────────────────────────────────────────────────────────────────────

/// Source of more bytes when the loader runs out. `rmarshal.Loader.need_more_data`
/// (`rmarshal.py`) errors; `rsandbox.FdLoader` (`rsandbox.py`) reads the
/// pipe. Returning an empty `Vec` signals EOF.
pub trait NeedMore {
    fn need_more(&mut self) -> SandboxResult<Vec<u8>>;
}

/// A `NeedMore` over a complete in-memory buffer: any request for more data is a
/// protocol error (`rmarshal.py:293`).
pub struct Complete;

impl NeedMore for Complete {
    fn need_more(&mut self) -> SandboxResult<Vec<u8>> {
        Err(SandboxError::Protocol("not enough data".into()))
    }
}

/// A `NeedMore` that pulls more bytes from any reader (the controller reading the
/// child's stdout pipe — the `FdLoader` analog on the controller side). A short
/// read of 0 bytes means EOF and is surfaced as an empty `Vec`.
pub struct ReadNeedMore<R: std::io::Read> {
    reader: R,
    chunk: usize,
}

impl<R: std::io::Read> ReadNeedMore<R> {
    pub fn new(reader: R) -> Self {
        ReadNeedMore {
            reader,
            chunk: 4096,
        }
    }
}

impl<R: std::io::Read> NeedMore for ReadNeedMore<R> {
    fn need_more(&mut self) -> SandboxResult<Vec<u8>> {
        let mut buf = vec![0u8; self.chunk];
        let n = self
            .reader
            .read(&mut buf)
            .map_err(|e| SandboxError::Protocol(e.to_string()))?;
        buf.truncate(n);
        Ok(buf)
    }
}

/// rmarshal.py `Loader`.
pub struct Loader<N: NeedMore> {
    buf: Vec<u8>,
    pos: usize,
    src: N,
}

impl Loader<Complete> {
    /// Build a loader over a fully-available buffer.
    pub fn from_bytes(buf: Vec<u8>) -> Self {
        Loader {
            buf,
            pos: 0,
            src: Complete,
        }
    }
}

impl<N: NeedMore> Loader<N> {
    pub fn new(buf: Vec<u8>, src: N) -> Self {
        Loader { buf, pos: 0, src }
    }

    fn ensure(&mut self, end: usize) -> SandboxResult<()> {
        while end > self.buf.len() {
            let more = self.src.need_more()?;
            if more.is_empty() {
                return Err(SandboxError::Protocol("unexpected EOF".into()));
            }
            self.buf.extend_from_slice(&more);
        }
        Ok(())
    }

    // rmarshal.py readchr
    fn readchr(&mut self) -> SandboxResult<u8> {
        self.ensure(self.pos + 1)?;
        let c = self.buf[self.pos];
        self.pos += 1;
        Ok(c)
    }

    // rmarshal.py peekchr
    fn peekchr(&mut self) -> SandboxResult<u8> {
        self.ensure(self.pos + 1)?;
        Ok(self.buf[self.pos])
    }

    // rmarshal.py `readlong` — 4 signed LE bytes.
    fn readlong(&mut self) -> SandboxResult<i32> {
        self.ensure(self.pos + 4)?;
        let b = &self.buf[self.pos..self.pos + 4];
        let v = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        self.pos += 4;
        Ok(v)
    }

    // rmarshal.py `readstr`.
    fn readstr(&mut self, count: usize) -> SandboxResult<Vec<u8>> {
        self.ensure(self.pos + count)?;
        let s = self.buf[self.pos..self.pos + count].to_vec();
        self.pos += count;
        Ok(s)
    }

    fn read_i64_le(&mut self) -> SandboxResult<i64> {
        // low dword zero-extended, high dword sign-extended (rmarshal.py:176-178).
        let lo = self.readlong()? as u32 as i64;
        let hi = (self.readlong()? as i64) << 32;
        Ok(lo | hi)
    }

    /// rmarshal.py `check_finished`.
    pub fn check_finished(&self) -> SandboxResult<()> {
        if self.pos != self.buf.len() {
            Err(SandboxError::Protocol("not all data consumed".into()))
        } else {
            Ok(())
        }
    }

    /// Drop the bytes already consumed (`0..pos`), so a long-lived request
    /// stream does not let the buffer grow without bound; the next message then
    /// starts at offset 0. `drain(..pos)` (not `clear`) preserves any bytes of a
    /// following pipelined message already read into the buffer.
    pub fn drain_consumed(&mut self) {
        self.buf.drain(..self.pos);
        self.pos = 0;
    }

    /// Returns `true` if the stream is exhausted at a message boundary (no
    /// buffered bytes and the source is at EOF). This is the controller's
    /// `read_message` -> `EOFError` check (`sandlib.py`): a clean EOF
    /// between messages ends the loop, whereas EOF mid-message is an error.
    pub fn at_message_boundary_eof(&mut self) -> SandboxResult<bool> {
        if self.pos < self.buf.len() {
            return Ok(false);
        }
        let more = self.src.need_more()?;
        if more.is_empty() {
            return Ok(true);
        }
        self.buf.extend_from_slice(&more);
        Ok(false)
    }
}

/// rmarshal.py `load_int` — accepts both `TYPE_INT` and `TYPE_INT64`.
pub fn load_int<N: NeedMore>(ld: &mut Loader<N>) -> SandboxResult<i64> {
    match ld.readchr()? {
        TYPE_INT64 => ld.read_i64_le(),
        TYPE_INT => Ok(ld.readlong()? as i64),
        _ => Err(SandboxError::Protocol("expected an int".into())),
    }
}

/// rmarshal.py `load_int_nonneg`.
pub fn load_int_nonneg<N: NeedMore>(ld: &mut Loader<N>) -> SandboxResult<i64> {
    let x = load_int(ld)?;
    if x < 0 {
        return Err(SandboxError::Protocol("expected a non-negative int".into()));
    }
    Ok(x)
}

/// rmarshal.py `load_longlong` — requires `TYPE_INT64`.
pub fn load_longlong<N: NeedMore>(ld: &mut Loader<N>) -> SandboxResult<i64> {
    if ld.readchr()? != TYPE_INT64 {
        return Err(SandboxError::Protocol("expected a longlong".into()));
    }
    ld.read_i64_le()
}

/// rmarshal.py `load_string` — bytes (NULs allowed).
pub fn load_string<N: NeedMore>(ld: &mut Loader<N>) -> SandboxResult<Vec<u8>> {
    if ld.readchr()? != TYPE_STRING {
        return Err(SandboxError::Protocol("expected a string".into()));
    }
    let length = ld.readlong()?;
    if length < 0 {
        return Err(SandboxError::Protocol("negative string length".into()));
    }
    ld.readstr(length as usize)
}

/// rmarshal.py `load_float`.
pub fn load_float<N: NeedMore>(ld: &mut Loader<N>) -> SandboxResult<f64> {
    if ld.readchr()? != TYPE_FLOAT {
        return Err(SandboxError::Protocol("expected a float".into()));
    }
    let length = ld.readchr()? as usize;
    let s = ld.readstr(length)?;
    parse_ascii_float(&s)
}

/// rmarshal.py `load_bool`.
pub fn load_bool<N: NeedMore>(ld: &mut Loader<N>) -> SandboxResult<bool> {
    match ld.readchr()? {
        TYPE_TRUE => Ok(true),
        TYPE_FALSE => Ok(false),
        _ => Err(SandboxError::Protocol("expected a bool".into())),
    }
}

/// Generic value loader (used by the controller to read `(fnname, args)` and by
/// round-trip tests). Dispatches on the tag byte.
pub fn load_value<N: NeedMore>(ld: &mut Loader<N>) -> SandboxResult<MarshalValue> {
    let tag = ld.readchr()?;
    match tag {
        TYPE_NONE => Ok(MarshalValue::None),
        TYPE_TRUE => Ok(MarshalValue::Bool(true)),
        TYPE_FALSE => Ok(MarshalValue::Bool(false)),
        TYPE_INT => Ok(MarshalValue::Int(ld.readlong()? as i64)),
        TYPE_INT64 => Ok(MarshalValue::Int(ld.read_i64_le()?)),
        TYPE_FLOAT => {
            let length = ld.readchr()? as usize;
            let s = ld.readstr(length)?;
            Ok(MarshalValue::Float(parse_ascii_float(&s)?))
        }
        TYPE_STRING => {
            let length = ld.readlong()?;
            if length < 0 {
                return Err(SandboxError::Protocol("negative string length".into()));
            }
            Ok(MarshalValue::Str(ld.readstr(length as usize)?))
        }
        TYPE_TUPLE | TYPE_LIST => {
            let length = ld.readlong()?;
            if length < 0 {
                return Err(SandboxError::Protocol("negative sequence length".into()));
            }
            // Do not pre-size from the wire-declared length: an untrusted child
            // can claim a huge count in a few bytes and force a multi-gigabyte
            // `with_capacity` (OOM) before any item is read. Grow as items
            // actually arrive — a lying length simply hits EOF below.
            let mut items = Vec::new();
            for _ in 0..length {
                items.push(load_value(ld)?);
            }
            if tag == TYPE_TUPLE {
                Ok(MarshalValue::Tuple(items))
            } else {
                Ok(MarshalValue::List(items))
            }
        }
        TYPE_DICT => {
            let mut items = Vec::new();
            while ld.peekchr()? != DICT_END {
                let key = load_value(ld)?;
                let value = load_value(ld)?;
                items.push((key, value));
            }
            ld.readchr()?; // consume DICT_END
            Ok(MarshalValue::Dict(items))
        }
        other => Err(SandboxError::Protocol(format!(
            "bad marshal tag {other:#x}"
        ))),
    }
}

/// Decode a `RESULTTYPE_STATRESULT` reply into a [`StatResult`]. The field order
/// and per-field tag widths mirror [`dump_statresult`].
pub fn load_statresult<N: NeedMore>(ld: &mut Loader<N>) -> SandboxResult<StatResult> {
    if ld.readchr()? != TYPE_TUPLE {
        return Err(SandboxError::Protocol("expected a stat tuple".into()));
    }
    let count = ld.readlong()?;
    if count != 10 {
        return Err(SandboxError::Protocol(
            "stat tuple must have 10 fields".into(),
        ));
    }
    let st_mode = load_int(ld)? as u32;
    let st_ino = load_int(ld)? as u64;
    let st_dev = load_int(ld)? as u64;
    let st_nlink = load_int(ld)? as u64;
    let st_uid = load_int(ld)? as u32;
    let st_gid = load_int(ld)? as u32;
    let st_size = load_int(ld)? as u64;
    let st_atime = load_float(ld)? as i64;
    let st_mtime = load_float(ld)? as i64;
    let st_ctime = load_float(ld)? as i64;
    Ok(StatResult {
        st_mode,
        st_ino,
        st_dev,
        st_nlink,
        st_uid,
        st_gid,
        st_size,
        st_atime,
        st_mtime,
        st_ctime,
    })
}

// ── float formatting helpers ─────────────────────────────────────────────────

/// C `printf("%.*g", precision, x)` — matches RPython `formatd(x, 'g', precision)`.
fn c_format_g(x: f64, precision: i32) -> Vec<u8> {
    unsafe {
        let fmt = c"%.*g".as_ptr();
        let needed = libc::snprintf(std::ptr::null_mut(), 0, fmt, precision as libc::c_int, x);
        if needed < 0 {
            return Vec::new();
        }
        let mut out = vec![0u8; needed as usize + 1];
        libc::snprintf(
            out.as_mut_ptr() as *mut libc::c_char,
            out.len(),
            fmt,
            precision as libc::c_int,
            x,
        );
        out.truncate(needed as usize); // drop the trailing NUL
        out
    }
}

fn parse_ascii_float(s: &[u8]) -> SandboxResult<f64> {
    let text =
        std::str::from_utf8(s).map_err(|_| SandboxError::Protocol("non-ascii float".into()))?;
    match text {
        "inf" => Ok(f64::INFINITY),
        "-inf" => Ok(f64::NEG_INFINITY),
        "nan" => Ok(f64::NAN),
        _ => text
            .parse::<f64>()
            .map_err(|_| SandboxError::Protocol(format!("bad float {text:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The golden byte vectors below were generated by an independent Python
    // oracle replicating the exact `rmarshal.py` / `sandlib.py` byte layout (NOT
    // by hand). They freeze the wire format.

    #[test]
    fn golden_request_open() {
        // ("ll_os.ll_os_open", ("/tmp/foobar", 0, 0o777))
        let mut buf = Vec::new();
        dump_string(&mut buf, b"ll_os.ll_os_open");
        dump_tuple(
            &mut buf,
            &[
                MarshalValue::Str(b"/tmp/foobar".to_vec()),
                MarshalValue::Int(0),
                MarshalValue::Int(0o777),
            ],
            IntFlavor::Rmarshal,
        );
        let expected: &[u8] = &[
            0x73, 0x10, 0x00, 0x00, 0x00, 0x6c, 0x6c, 0x5f, 0x6f, 0x73, 0x2e, 0x6c, 0x6c, 0x5f,
            0x6f, 0x73, 0x5f, 0x6f, 0x70, 0x65, 0x6e, 0x28, 0x03, 0x00, 0x00, 0x00, 0x73, 0x0b,
            0x00, 0x00, 0x00, 0x2f, 0x74, 0x6d, 0x70, 0x2f, 0x66, 0x6f, 0x6f, 0x62, 0x61, 0x72,
            0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x49, 0xff, 0x01, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(buf.as_slice(), expected);
    }

    #[test]
    fn golden_reply_ok_fd77() {
        // error code 0, then fd 77 — controller `_marshal` int flavor.
        let mut buf = Vec::new();
        dump_int(&mut buf, 0, IntFlavor::Marshal);
        dump_int(&mut buf, 77, IntFlavor::Marshal);
        let expected: &[u8] = &[0x69, 0x00, 0x00, 0x00, 0x00, 0x69, 0x4d, 0x00, 0x00, 0x00];
        assert_eq!(buf.as_slice(), expected);
    }

    #[test]
    fn golden_reply_oserror_2() {
        // write_exception: exception code 1 then errno 2 (no leading success 0).
        let mut buf = Vec::new();
        dump_int(&mut buf, 1, IntFlavor::Marshal);
        dump_int(&mut buf, 2, IntFlavor::Marshal);
        let expected: &[u8] = &[0x69, 0x01, 0x00, 0x00, 0x00, 0x69, 0x02, 0x00, 0x00, 0x00];
        assert_eq!(buf.as_slice(), expected);
    }

    #[test]
    fn golden_statresult() {
        // os.stat_result((55,0,0,0,0,0,0x12380000007,0,0,0))
        let st = StatResult {
            st_mode: 55,
            st_ino: 0,
            st_dev: 0,
            st_nlink: 0,
            st_uid: 0,
            st_gid: 0,
            st_size: 0x12380000007,
            st_atime: 0,
            st_mtime: 0,
            st_ctime: 0,
        };
        let mut buf = Vec::new();
        dump_statresult(&mut buf, &st);
        let expected: &[u8] = &[
            0x28, 0x0a, 0x00, 0x00, 0x00, 0x69, 0x37, 0x00, 0x00, 0x00, 0x49, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x69, 0x00, 0x00, 0x00, 0x00, 0x69, 0x00, 0x00, 0x00, 0x00, 0x69, 0x00, 0x00, 0x00,
            0x00, 0x49, 0x07, 0x00, 0x00, 0x80, 0x23, 0x01, 0x00, 0x00, 0x66, 0x01, 0x30, 0x66,
            0x01, 0x30, 0x66, 0x01, 0x30,
        ];
        assert_eq!(buf.as_slice(), expected);
    }

    #[test]
    fn golden_longlong_result() {
        let mut buf = Vec::new();
        dump_longlong_result(&mut buf, 0x12380000007);
        let expected: &[u8] = &[0x49, 0x07, 0x00, 0x00, 0x80, 0x23, 0x01, 0x00, 0x00];
        assert_eq!(buf.as_slice(), expected);
    }

    fn roundtrip(value: MarshalValue, flavor: IntFlavor) {
        let mut buf = Vec::new();
        dump_value(&mut buf, &value, flavor);
        let encoded = buf.clone();
        let mut ld = Loader::from_bytes(buf);
        let back = load_value(&mut ld)
            .unwrap_or_else(|e| panic!("load {value:?} ({flavor:?}) from {encoded:02x?}: {e}"));
        ld.check_finished()
            .unwrap_or_else(|e| panic!("finish {value:?} ({flavor:?}) from {encoded:02x?}: {e}"));
        assert_eq!(back, value);
    }

    #[test]
    fn roundtrip_all_types() {
        for flavor in [IntFlavor::Rmarshal, IntFlavor::Marshal] {
            roundtrip(MarshalValue::None, flavor);
            roundtrip(MarshalValue::Bool(true), flavor);
            roundtrip(MarshalValue::Bool(false), flavor);
            roundtrip(MarshalValue::Int(0), flavor);
            roundtrip(MarshalValue::Int(77), flavor);
            roundtrip(MarshalValue::Int(-1), flavor);
            roundtrip(MarshalValue::Int(0x12380000007), flavor);
            roundtrip(MarshalValue::Int(i64::MIN), flavor);
            roundtrip(MarshalValue::Int(i64::MAX), flavor);
            // embedded NUL + non-ascii bytes
            roundtrip(MarshalValue::Str(b"he\x00llo\xff".to_vec()), flavor);
            roundtrip(MarshalValue::Str(Vec::new()), flavor);
            roundtrip(MarshalValue::Float(0.0), flavor);
            roundtrip(MarshalValue::Float(3.011), flavor);
            roundtrip(MarshalValue::Float(-1.5e300), flavor);
            roundtrip(
                MarshalValue::Tuple(vec![
                    MarshalValue::Str(b"/tmp/foobar".to_vec()),
                    MarshalValue::Int(0),
                    MarshalValue::Int(0o777),
                    MarshalValue::Bool(true),
                ]),
                flavor,
            );
            roundtrip(
                MarshalValue::List(vec![MarshalValue::Int(1), MarshalValue::Int(2)]),
                flavor,
            );
            roundtrip(
                MarshalValue::Dict(vec![(
                    MarshalValue::Str(b"k".to_vec()),
                    MarshalValue::Str(b"v".to_vec()),
                )]),
                flavor,
            );
        }
    }

    #[test]
    fn both_int_tags_decode_equal() {
        // 'i'-encoded and 'I'-encoded 77 must load to the same value.
        let mut small = Vec::new();
        dump_int(&mut small, 77, IntFlavor::Marshal); // 'i'
        let mut big = Vec::new();
        dump_int(&mut big, 77, IntFlavor::Rmarshal); // 'I'
        assert_eq!(small[0], TYPE_INT);
        assert_eq!(big[0], TYPE_INT64);
        let a = load_int(&mut Loader::from_bytes(small)).unwrap();
        let b = load_int(&mut Loader::from_bytes(big)).unwrap();
        assert_eq!(a, 77);
        assert_eq!(b, 77);
    }

    #[test]
    fn statresult_roundtrip() {
        let st = StatResult {
            st_mode: 0o100644,
            st_ino: 42,
            st_dev: 1,
            st_nlink: 1,
            st_uid: 0,
            st_gid: 0,
            st_size: 0x12380000007,
            st_atime: 0,
            st_mtime: 0,
            st_ctime: 0,
        };
        let mut buf = Vec::new();
        dump_statresult(&mut buf, &st);
        let mut ld = Loader::from_bytes(buf);
        let back = load_statresult(&mut ld).unwrap();
        ld.check_finished().unwrap();
        assert_eq!(back, st);
    }

    #[test]
    fn longlong_result_roundtrips_via_load_longlong() {
        let mut buf = Vec::new();
        dump_longlong_result(&mut buf, -5);
        let mut ld = Loader::from_bytes(buf);
        assert_eq!(load_longlong(&mut ld).unwrap(), -5);
        ld.check_finished().unwrap();
    }
}
