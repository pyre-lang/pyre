//! Text streams — PyPy `pypy/module/_io/interp_textio.py`.
//!
//! Keep the stream state on the typed object, matching
//! `W_TextIOWrapper`.  In particular, the buffer and the
//! ZERO/OK/DETACHED state are not instance-dict side data.

use majit_rlib::rbigint::{RBigInt, RBigIntGcRoot};
use pyre_object::*;
use rustpython_wtf8::{Wtf8, Wtf8Buf};

const STATE_ZERO: i64 = 0;
const STATE_OK: i64 = 1;
const STATE_DETACHED: i64 = 2;

/// PyPy `PositionCookie`.  Each field occupies one native unsigned word in
/// the opaque integer returned by `tell()`.
#[derive(Default)]
struct PositionCookie {
    start_pos: u64,
    dec_flags: u64,
    bytes_to_feed: u64,
    chars_to_skip: u64,
    need_eof: bool,
}

impl PositionCookie {
    const BITS: usize = u64::BITS as usize;

    fn unpack(mut value: RBigInt) -> Result<Self, crate::PyError> {
        if value.int_lt(0) {
            return Err(crate::PyError::value_error("negative seek position"));
        }
        // interp_textio.py `PositionCookie.__init__`: extract each
        // native word with rbigint's mask conversion, then shift the source.
        // In particular, do not call `to_u64()` on the whole cookie — valid
        // packed cookies intentionally occupy several machine words.
        let start_pos = value.ulonglongmask();
        value = value
            .rshift(Self::BITS as i64, false)
            .expect("native-word shift");
        let dec_flags = value.uintmask();
        value = value
            .rshift(Self::BITS as i64, false)
            .expect("native-word shift");
        let bytes_to_feed = value.uintmask();
        value = value
            .rshift(Self::BITS as i64, false)
            .expect("native-word shift");
        let chars_to_skip = value.uintmask();
        value = value
            .rshift(Self::BITS as i64, false)
            .expect("native-word shift");
        let need_eof = value.tobool();
        Ok(Self {
            start_pos,
            dec_flags,
            bytes_to_feed,
            chars_to_skip,
            need_eof,
        })
    }

    fn pack(&self) -> RBigInt {
        // Each shifted term collects, so it is computed before `result` is
        // read for the `or_`.
        let mut result = RBigIntGcRoot::new(RBigInt::from(self.start_pos));
        let term = RBigInt::from(self.dec_flags)
            .lshift(Self::BITS as i64)
            .expect("native-word shift");
        result.set(result.or_(&term));
        let term = RBigInt::from(self.bytes_to_feed)
            .lshift((Self::BITS * 2) as i64)
            .expect("native-word shift");
        result.set(result.or_(&term));
        let term = RBigInt::from(self.chars_to_skip)
            .lshift((Self::BITS * 3) as i64)
            .expect("native-word shift");
        result.set(result.or_(&term));
        if self.need_eof {
            let term = RBigInt::one()
                .lshift((Self::BITS * 4) as i64)
                .expect("native-word shift");
            result.set(result.or_(&term));
        }
        result.translated_alias()
    }

    fn to_object(&self) -> PyObjectRef {
        crate::objspace::descroperation::bigint_result(self.pack())
    }
}

/// PyPy `PositionSnapshot`.
struct PositionSnapshot {
    flags: u64,
    input: Vec<u8>,
}

/// PyPy `interp_textio.DecodeBuffer`.
#[derive(Default)]
struct DecodeBuffer {
    text: Option<Wtf8Buf>,
    pos: usize,
    upos: usize,
    ulen: usize,
}

impl DecodeBuffer {
    fn set(&mut self, decoded: PyObjectRef) -> Result<(), crate::PyError> {
        if unsafe { !pyre_object::is_str(decoded) } {
            return Err(crate::PyError::type_error(format!(
                "decoder should return a string result, not '{}'",
                crate::type_methods::arg_type_name(decoded)
            )));
        }
        let text = unsafe { pyre_object::w_str_get_wtf8(decoded) }.to_wtf8_buf();
        self.ulen = text.code_points().count();
        self.text = Some(text);
        self.pos = 0;
        self.upos = 0;
        Ok(())
    }

    fn reset(&mut self) {
        self.text = None;
        self.pos = 0;
        self.upos = 0;
        self.ulen = 0;
    }

    fn has_data(&self) -> bool {
        self.text.is_some() && !self.exhausted()
    }

    fn exhausted(&self) -> bool {
        self.text.as_ref().is_none_or(|text| self.pos >= text.len())
    }

    fn available(&self) -> usize {
        self.ulen.saturating_sub(self.upos)
    }

    fn advance(&mut self) {
        let text = self.text.as_ref().expect("DecodeBuffer text");
        let cp = text[self.pos..]
            .code_points()
            .next()
            .expect("advance requires available data");
        self.pos += cp.len_wtf8();
        self.upos += 1;
    }

    fn get_chars(&mut self, size: Option<usize>) -> Wtf8Buf {
        let Some(text) = self.text.as_ref() else {
            return Wtf8Buf::new();
        };
        let count = size.unwrap_or(self.available()).min(self.available());
        let start = self.pos;
        let end = text[start..]
            .code_point_indices()
            .nth(count)
            .map_or(text.len(), |(offset, _)| start + offset);
        let result = text[start..end].to_wtf8_buf();
        self.pos = end;
        self.upos += count;
        result
    }

    fn find_char(&mut self, marker: u8, limit: Option<usize>) -> bool {
        let mut scanned = 0;
        while limit.is_none_or(|limit| scanned < limit) && !self.exhausted() {
            let found = self
                .text
                .as_ref()
                .expect("DecodeBuffer text")
                .ascii_byte_at(self.pos)
                == marker;
            self.advance();
            scanned += 1;
            if found {
                return true;
            }
        }
        false
    }

    fn find_newline_universal(&mut self, limit: Option<usize>) -> bool {
        let mut scanned = 0;
        while limit.is_none_or(|limit| scanned < limit) && !self.exhausted() {
            let byte = self
                .text
                .as_ref()
                .expect("DecodeBuffer text")
                .ascii_byte_at(self.pos);
            self.advance();
            scanned += 1;
            if byte == b'\n' {
                return true;
            }
            if byte == b'\r' {
                if limit.is_some_and(|limit| scanned >= limit) || self.exhausted() {
                    return true;
                }
                if self
                    .text
                    .as_ref()
                    .expect("DecodeBuffer text")
                    .ascii_byte_at(self.pos)
                    == b'\n'
                {
                    self.advance();
                }
                return true;
            }
        }
        false
    }

    fn find_crlf(&mut self, limit: Option<usize>) -> bool {
        let mut scanned = 0;
        while limit.is_none_or(|limit| scanned < limit) && !self.exhausted() {
            let byte = self
                .text
                .as_ref()
                .expect("DecodeBuffer text")
                .ascii_byte_at(self.pos);
            if byte != b'\r' {
                self.advance();
                scanned += 1;
                continue;
            }
            let saved_pos = self.pos;
            let saved_upos = self.upos;
            self.advance();
            scanned += 1;
            if limit.is_some_and(|limit| scanned >= limit) {
                return false;
            }
            if self.exhausted() {
                self.pos = saved_pos;
                self.upos = saved_upos;
                return false;
            }
            if self
                .text
                .as_ref()
                .expect("DecodeBuffer text")
                .ascii_byte_at(self.pos)
                == b'\n'
            {
                self.advance();
                return true;
            }
        }
        false
    }

    fn consumed_from(&self, start: usize) -> Wtf8Buf {
        self.text
            .as_ref()
            .map_or_else(Wtf8Buf::new, |text| text[start..self.pos].to_wtf8_buf())
    }

    fn starts_with_lf(&self) -> bool {
        self.text
            .as_ref()
            .is_some_and(|text| self.pos < text.len() && text.ascii_byte_at(self.pos) == b'\n')
    }
}

/// The `read1` probe of `W_TextIOWrapper.descr_init`.
///
/// [3.15-spec] `space.findattr(w_buffer, "read1")` swallows every
/// non-async OperationError, so a buffer whose `read1` lookup raises
/// `KeyError` reads as having no `read1` (measured: PyPy 8.0.0).
/// `Modules/_io/textio.c` (v3.15.0) probes with `PyObject_HasAttrWithError`:
/// only an AttributeError means "no read1" and any other error propagates
/// (measured: CPython 3.15.0, `KeyError` and `NameError`).  Neither
/// `findattr` nor `descr_init` carries a JIT hint.
fn buffer_has_read1(w_buffer: PyObjectRef) -> Result<bool, crate::PyError> {
    match crate::baseobjspace::getattr_str(w_buffer, "read1") {
        Ok(_) => Ok(true),
        Err(error) if error.kind == crate::PyErrorKind::AttributeError => Ok(false),
        Err(error) => Err(error),
    }
}

// CPython 3.14 Modules/_io/_iomodule.c:ADD_TYPE creates the immutable
// TextIOWrapper heap spec.
#[crate::pyre_class("_io.TextIOWrapper", cpython_heaptype, user_layout, weakrefable)]
pub struct W_TextIOWrapper {
    state: i64,
    w_buffer: PyObjectRef,
    w_encoding: PyObjectRef,
    w_errors: PyObjectRef,
    w_newline: PyObjectRef,
    w_stdio_name: PyObjectRef,
    w_encoder: PyObjectRef,
    w_decoder: PyObjectRef,
    line_buffering: bool,
    write_through: bool,
    decoded: DecodeBuffer,
    snapshot: Option<PositionSnapshot>,
    pending_bytes: Option<Vec<Vec<u8>>>,
    pending_bytes_count: usize,
    chunk_size: i64,
    b2cratio: f64,
    has_read1: bool,
    readuniversal: bool,
    readtranslate: bool,
    seekable_flag: bool,
    telling: bool,
    encoding_start_of_stream: bool,
    // interp_iobase.py W_IOBase.w_dict — null until getdict.
    pub(crate) w_dict: PyObjectRef,
}

impl Default for W_TextIOWrapper {
    fn default() -> Self {
        Self {
            ob: PyObject::default(),
            state: STATE_ZERO,
            w_buffer: PY_NULL,
            w_encoding: PY_NULL,
            w_errors: PY_NULL,
            w_newline: PY_NULL,
            w_stdio_name: PY_NULL,
            w_encoder: PY_NULL,
            w_decoder: PY_NULL,
            line_buffering: false,
            write_through: false,
            decoded: DecodeBuffer::default(),
            snapshot: None,
            pending_bytes: None,
            pending_bytes_count: 0,
            chunk_size: 8192,
            b2cratio: 0.0,
            has_read1: false,
            readuniversal: false,
            readtranslate: false,
            seekable_flag: false,
            telling: false,
            encoding_start_of_stream: false,
            w_dict: PY_NULL,
            lifeline: PY_NULL,
        }
    }
}

impl W_TextIOWrapper {
    fn self_obj(&self) -> PyObjectRef {
        self as *const Self as PyObjectRef
    }

    /// Re-read the receiver from `slot` after Python code ran.
    ///
    /// Codec lookup, `w_str_new_managed`, and `buffer.readable()` each
    /// allocate; a collection inside one of those moves a nursery wrapper —
    /// leaving the `&mut self` the method was entered with behind the
    /// forwarding pointer, so later stores land in the abandoned body.
    /// RPython's GC transform keeps `self` live across the call
    /// (`interp_textio.py` `W_TextIOWrapper`); reload from the shadow-stack
    /// slot instead.
    fn from_slot(slot: usize) -> &'static mut Self {
        super::from_pinned_slot(slot)
    }

    /// Pin the receiver so [`Self::from_slot`] can recover it, and answer the
    /// slot it landed in.
    fn pin_self(&self) -> usize {
        super::pin_io_self(self.self_obj())
    }

    fn check_init(&self) -> Result<(), crate::PyError> {
        if self.state == STATE_ZERO {
            Err(crate::PyError::value_error(
                "I/O operation on uninitialized object",
            ))
        } else {
            Ok(())
        }
    }

    fn check_attached(&self) -> Result<(), crate::PyError> {
        if self.state == STATE_DETACHED {
            return Err(crate::PyError::value_error(
                "underlying buffer has been detached",
            ));
        }
        self.check_init()
    }

    fn buffer_closed(&self) -> Result<bool, crate::PyError> {
        self.check_attached()?;
        let closed = crate::baseobjspace::getattr_str(self.w_buffer, "closed")?;
        crate::baseobjspace::is_true(closed)
    }

    /// `textio.c CHECK_CLOSED`.
    ///
    /// The sentence ends in a period here and in `iobase.c`, and does not in
    /// `fileio.c` or `stringio.c`; which spelling a program sees says which
    /// layer refused it.
    fn check_closed(&self) -> Result<(), crate::PyError> {
        if self.buffer_closed()? {
            Err(crate::PyError::value_error("I/O operation on closed file."))
        } else {
            Ok(())
        }
    }

    fn call_buffer(&self, name: &str, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // `check_attached` / `buffer_closed` call Python (`closed` on the
        // raw stream) and can collect. Pin the wrapper and the call args
        // before that, then read `w_buffer` back from the live object.
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let args_base = _roots.pin_roots(args);
        Self::from_slot(slot).check_attached()?;
        let buffer_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_buffer);
        let mut call_args = Vec::with_capacity(args.len());
        for i in 0..args.len() {
            call_args.push(_roots.get(args_base + i));
        }
        super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(buffer_slot),
            name,
            &call_args,
        )
    }

    /// PyPy `text0_or_none` / `io_check_errors` first route text through a
    /// real Unicode encoding operation.  Do the same instead of borrowing a
    /// Rust `&str`: a Python string may contain a lone surrogate, which must
    /// raise `UnicodeEncodeError` rather than panic in `w_str_get_value`.
    fn checked_text(
        obj: PyObjectRef,
        default: &str,
        argument: &str,
    ) -> Result<String, crate::PyError> {
        unsafe {
            if pyre_object::is_none(obj) {
                return Ok(default.to_string());
            }
            if !pyre_object::is_str(obj) {
                return Err(crate::PyError::type_error(format!(
                    "{argument} must be a str"
                )));
            }
        }
        let bytes = crate::type_methods::encode_object(obj, "utf-8", "strict")?;
        Ok(String::from_utf8(bytes)
            .expect("the utf-8 codec must only return valid UTF-8 for valid Unicode text"))
    }

    fn checked_text0(
        obj: PyObjectRef,
        default: &str,
        argument: &str,
    ) -> Result<String, crate::PyError> {
        let value = Self::checked_text(obj, default, argument)?;
        if value.contains('\0') {
            return Err(crate::PyError::value_error("embedded null character"));
        }
        Ok(value)
    }

    /// `interp_textio.io_check_errors` tail: `checked_text0` already applied
    /// the text0 + utf-8 strict-encode checks, so the only remaining step is
    /// the dev-mode error-handler name lookup.
    fn io_check_errors(errors: &str) -> Result<(), crate::PyError> {
        if crate::importing::dev_mode_flag() {
            crate::module::_codecs::validate_error_handler(errors)?;
        }
        Ok(())
    }

    /// `encoding="locale"` selects the current locale's encoding, which is
    /// what `_Py_GetLocaleEncodingObject` answers for it and for the
    /// unspecified argument that resolves to it.  A sandbox build has no host
    /// locale to ask and reads utf-8, the answer a `_Py_FORCE_UTF8_LOCALE`
    /// build gives without asking one.
    fn resolve_locale_encoding(encoding: String) -> String {
        if encoding == "locale" {
            crate::module::_locale::interp_locale::locale_encoding()
        } else {
            encoding
        }
    }

    /// PyPy `_io.interp_iobase.unwrap_newline`.
    fn unwrap_newline(newline: PyObjectRef) -> Result<Option<String>, crate::PyError> {
        unsafe {
            if pyre_object::is_none(newline) {
                return Ok(None);
            }
            if !pyre_object::is_str(newline) {
                return Err(crate::PyError::type_error("illegal newline type"));
            }
            // `interp_textio.py unwrap_newline` performs a direct
            // `space.utf8_w` conversion followed by the value branch.  Spell
            // the fallible host conversion as the same direct branch instead
            // of introducing Rust's generic `Result::map_err` closure into
            // the translated graph.
            let value = match pyre_object::w_str_get_wtf8(newline).as_str() {
                Ok(value) => value,
                Err(_) => return Err(crate::PyError::value_error("illegal newline value")),
            };
            if !matches!(value, "" | "\n" | "\r" | "\r\n") {
                // Unquoted on purpose. `interp_textio.py`'s `unwrap_newline`
                // raises with `%R`, but CPython 3.14 prints the value bare
                // (`illegal newline value: x`), and the observable text is
                // CPython's. Do not "restore" the repr quoting.
                return Err(crate::PyError::value_error(format!(
                    "illegal newline value: {value}"
                )));
            }
            Ok(Some(value.to_string()))
        }
    }

    fn configured_newline(&self) -> Option<&str> {
        unsafe {
            if pyre_object::is_none(self.w_newline) {
                None
            } else {
                pyre_object::w_str_get_value_opt(self.w_newline)
            }
        }
    }

    /// PyPy `W_TextIOWrapper._set_newline`.
    fn set_newline(&mut self, newline: Option<&str>) {
        self.readuniversal = newline.is_none_or(str::is_empty);
        self.readtranslate = newline.is_none();
    }

    /// The string `write` substitutes for `'\n'`, or `None` for no
    /// translation.  PyPy `W_TextIOWrapper` `writetranslate`/`writenl`: an
    /// explicit `'\r'` or `'\r\n'` is honored on every platform; `newline=None`
    /// writes the platform line separator (`'\r\n'` on Windows, untranslated
    /// elsewhere); `''` and `'\n'` are written verbatim.
    fn write_newline(&self) -> Option<&'static str> {
        match self.configured_newline() {
            None => {
                #[cfg(windows)]
                {
                    Some("\r\n")
                }
                #[cfg(not(windows))]
                {
                    None
                }
            }
            Some("\r") => Some("\r"),
            Some("\r\n") => Some("\r\n"),
            _ => None,
        }
    }

    fn size_limit(w_size: PyObjectRef) -> Result<Option<usize>, crate::PyError> {
        if w_size.is_null() || unsafe { pyre_object::is_none(w_size) } {
            return Ok(None);
        }
        let size = crate::builtins::space_index_w(w_size)?;
        if size < 0 {
            Ok(None)
        } else {
            Ok(Some(size as usize))
        }
    }

    /// Record `self` in the remembered set (incminimark.py write_barrier_from_array
    /// `write_barrier_from_array`).  An already-old wrapper is NOT scanned by a
    /// minor collection unless the barrier ran since the previous one, so a
    /// reference field storing a fresh (nursery) object has to be published
    /// before the next operation that can collect — not once at the end of a
    /// sequence that allocates and calls back into Python between the stores.
    #[inline]
    fn publish_refs(&mut self) {
        pyre_object::gc_hook::try_gc_write_barrier(
            self as *mut Self as pyre_object::gc_hook::GCREF,
        );
    }

    /// PyPy `W_TextIOWrapper._set_encoder_decoder`.
    fn set_encoder_decoder(&mut self, codec: PyObjectRef) -> Result<(), crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let codec_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(codec);
        {
            let this = Self::from_slot(slot);
            this.w_encoder = PY_NULL;
            this.w_decoder = PY_NULL;
        }

        if pyre_object::gc_roots::shadow_stack_get(codec_slot).is_null() {
            return Ok(());
        }

        let buffer_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_buffer);
        let errors_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_errors);

        let w_readable = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(buffer_slot),
            "readable",
            &[],
        )?;
        if crate::baseobjspace::is_true(w_readable)? {
            let mut decoder = super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(codec_slot),
                "incrementaldecoder",
                &[pyre_object::gc_roots::shadow_stack_get(errors_slot)],
            )?;
            if Self::from_slot(slot).readuniversal {
                let decoder_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(decoder);
                let io = crate::importing::get_builtin_module("_io").ok_or_else(|| {
                    crate::PyError::runtime_error("_io module is not initialized")
                })?;
                let io_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(io);
                let decoder_type = crate::baseobjspace::getattr_str(
                    pyre_object::gc_roots::shadow_stack_get(io_slot),
                    "IncrementalNewlineDecoder",
                )?;
                let type_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(decoder_type);
                decoder = crate::call::call_function_impl_result(
                    pyre_object::gc_roots::shadow_stack_get(type_slot),
                    &[
                        pyre_object::gc_roots::shadow_stack_get(decoder_slot),
                        w_bool_from(Self::from_slot(slot).readtranslate),
                    ],
                )?;
            }
            Self::from_slot(slot).w_decoder = decoder;
            // `writable` below runs arbitrary Python and can collect.
            Self::from_slot(slot).publish_refs();
        }

        let w_writable = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(buffer_slot),
            "writable",
            &[],
        )?;
        if crate::baseobjspace::is_true(w_writable)? {
            Self::from_slot(slot).w_encoder = super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(codec_slot),
                "incrementalencoder",
                &[pyre_object::gc_roots::shadow_stack_get(errors_slot)],
            )?;
            Self::from_slot(slot).publish_refs();
        }
        Ok(())
    }

    /// Interpreter bootstrap creates the three stdio wrappers before importing
    /// the codec registry is safe, so `allocate_stdio` cannot run the ordinary
    /// `__init__` codec lookup.  Complete that same initialization on the
    /// first typed read.  Ordinary TextIOWrapper instances already have a
    /// decoder and take the fast return; detached and write-only streams keep
    /// the normal "not readable" result.
    fn ensure_read_decoder(&mut self) -> Result<(), crate::PyError> {
        if !self.w_decoder.is_null() || self.state != STATE_OK || self.w_buffer.is_null() {
            return Ok(());
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let buffer_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_buffer);
        if !crate::baseobjspace::is_true(super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(buffer_slot),
            "readable",
            &[],
        )?)? {
            return Ok(());
        }
        let encoding_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_encoding);
        let encoding = crate::baseobjspace::str_utf8_w(pyre_object::gc_roots::shadow_stack_get(
            encoding_slot,
        ))?
        .to_string();
        let codec = Self::lookup_text_codec(&encoding)?;
        Self::from_slot(slot).set_encoder_decoder(codec)
    }

    /// PyPy `_read_chunk` decoder-state validation.
    fn decoder_getstate(&self) -> Result<(Vec<u8>, u64), crate::PyError> {
        let state = super::call_method_result(self.w_decoder, "getstate", &[])?;
        if unsafe { !pyre_object::is_tuple(state) || pyre_object::w_tuple_len(state) != 2 } {
            return Err(crate::PyError::type_error("illegal decoder state"));
        }
        let items = unsafe { pyre_object::w_tuple_items_copy_as_vec(state) };
        if unsafe { !pyre_object::bytesobject::is_bytes(items[0]) } {
            return Err(crate::PyError::type_error(format!(
                "illegal decoder state: the first value should be a bytes object not '{}'",
                crate::type_methods::arg_type_name(items[0])
            )));
        }
        let flags = crate::builtins::space_index_w(items[1])?;
        if flags < 0 {
            return Err(crate::PyError::type_error("illegal decoder state"));
        }
        Ok((
            unsafe { pyre_object::bytesobject::w_bytes_data(items[0]) }.to_vec(),
            flags as u64,
        ))
    }

    /// PyPy `_decoder_setstate`.
    fn decoder_setstate(&self, cookie: &PositionCookie) -> Result<(), crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let decoder_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_decoder);
        if cookie.start_pos == 0 && cookie.dec_flags == 0 {
            super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(decoder_slot),
                "reset",
                &[],
            )?;
        } else {
            let mut fields = pyre_object::gc_roots::RootedItems::new();
            fields.push(pyre_object::bytesobject::w_bytes_empty());
            fields.push(w_int_new(cookie.dec_flags as i64));
            let state = w_tuple_new(fields.take());
            super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(decoder_slot),
                "setstate",
                &[state],
            )?;
        }
        Ok(())
    }

    /// PyPy `_encoder_reset`.
    fn encoder_reset(&mut self, start_of_stream: bool) -> Result<(), crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let encoder_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_encoder);
        if start_of_stream {
            super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(encoder_slot),
                "reset",
                &[],
            )?;
            Self::from_slot(slot).encoding_start_of_stream = true;
        } else {
            let state_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_int_new(0));
            super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(encoder_slot),
                "setstate",
                &[pyre_object::gc_roots::shadow_stack_get(state_slot)],
            )?;
            Self::from_slot(slot).encoding_start_of_stream = false;
        }
        Ok(())
    }

    /// PyPy `W_TextIOWrapper._read_chunk`.
    fn read_chunk(&mut self, size_hint: usize) -> Result<bool, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        if Self::from_slot(slot).w_decoder.is_null() {
            return Err(super::unsupported("not readable"));
        }
        let (dec_buffer, dec_flags) = if Self::from_slot(slot).telling {
            let (buffer, flags) = Self::from_slot(slot).decoder_getstate()?;
            (Some(buffer), flags)
        } else {
            (None, 0)
        };
        let this = Self::from_slot(slot);
        let scaled_hint = if size_hint == 0 {
            0
        } else {
            (this.b2cratio.max(1.0) * size_hint as f64) as usize
        };
        let chunk_size = (this.chunk_size as usize).max(scaled_hint);
        let method = if this.has_read1 { "read1" } else { "read" };
        let buffer_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_buffer);
        let decoder_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_decoder);
        let size_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_int_new(chunk_size as i64));
        let input = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(buffer_slot),
            method,
            &[pyre_object::gc_roots::shadow_stack_get(size_slot)],
        )?;
        if unsafe { pyre_object::is_none(input) } {
            return Err(super::interp_bufferedio::make_blocking_error());
        }
        let input_bytes =
            unsafe { crate::builtins::file_write_buffer_bytes(input) }.map_err(|_| {
                crate::PyError::type_error(format!(
                    "underlying {method}() should have returned a bytes-like object, not '{}'",
                    crate::type_methods::arg_type_name(input)
                ))
            })?;
        let nbytes = input_bytes.len();
        let eof = nbytes == 0;
        let bytes_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::bytesobject::w_bytes_from_bytes(
            &input_bytes,
        ));
        let eof_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_bool_from(eof));
        let decoded = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(decoder_slot),
            "decode",
            &[
                pyre_object::gc_roots::shadow_stack_get(bytes_slot),
                pyre_object::gc_roots::shadow_stack_get(eof_slot),
            ],
        )?;
        let this = Self::from_slot(slot);
        this.decoded.set(decoded)?;
        let nchars = this.decoded.ulen;
        if nchars > 0 {
            this.b2cratio = nbytes as f64 / nchars as f64;
            if let Some(mut next_input) = dec_buffer {
                next_input.extend_from_slice(&input_bytes);
                this.snapshot = Some(PositionSnapshot {
                    flags: dec_flags,
                    input: next_input,
                });
            }
            Ok(true)
        } else {
            this.b2cratio = 0.0;
            if let Some(mut next_input) = dec_buffer {
                next_input.extend_from_slice(&input_bytes);
                this.snapshot = Some(PositionSnapshot {
                    flags: dec_flags,
                    input: next_input,
                });
            }
            Ok(!eof)
        }
    }

    /// PyPy `W_TextIOWrapper._ensure_data`.
    fn ensure_data(&mut self, size_hint: usize) -> Result<bool, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        while !Self::from_slot(slot).decoded.has_data() {
            if !Self::from_slot(slot).read_chunk(size_hint)? {
                let this = Self::from_slot(slot);
                this.decoded.reset();
                this.snapshot = None;
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn scan_line_ending(&mut self, limit: Option<usize>) -> bool {
        if self.readtranslate {
            return self.decoded.find_char(b'\n', limit);
        }
        if self.readuniversal {
            return self.decoded.find_newline_universal(limit);
        }
        match self.configured_newline().unwrap_or("\n") {
            "\r\n" => self.decoded.find_crlf(limit),
            "\r" => self.decoded.find_char(b'\r', limit),
            _ => self.decoded.find_char(b'\n', limit),
        }
    }

    fn read_all(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let mut result = Self::from_slot(slot).decoded.get_chars(None);
        let input = Self::from_slot(slot).call_buffer("read", &[])?;
        if unsafe { pyre_object::is_none(input) } {
            return Err(super::interp_bufferedio::make_blocking_error());
        }
        let input_bytes =
            unsafe { crate::builtins::file_write_buffer_bytes(input) }.map_err(|_| {
                crate::PyError::type_error(format!(
                    "underlying read() should have returned a bytes-like object, not '{}'",
                    crate::type_methods::arg_type_name(input)
                ))
            })?;
        let bytes_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::bytesobject::w_bytes_from_bytes(
            &input_bytes,
        ));
        let eof_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_bool_from(true));
        let decoder_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_decoder);
        let decoded = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(decoder_slot),
            "decode",
            &[
                pyre_object::gc_roots::shadow_stack_get(bytes_slot),
                pyre_object::gc_roots::shadow_stack_get(eof_slot),
            ],
        )?;
        if unsafe { !pyre_object::is_str(decoded) } {
            return Err(crate::PyError::type_error(format!(
                "decoder should return a string result, not '{}'",
                crate::type_methods::arg_type_name(decoded)
            )));
        }
        result.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(decoded) });
        {
            let this = Self::from_slot(slot);
            if this.snapshot.is_some() {
                this.decoded.reset();
                this.snapshot = None;
            }
        }
        Ok(pyre_object::unicodeobject::w_str_from_wtf8_managed(result))
    }

    fn read_n(&mut self, size: usize) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let mut remaining = size;
        let mut result = Wtf8Buf::new();
        while remaining > 0 {
            if !Self::from_slot(slot).ensure_data(remaining)? {
                break;
            }
            let chars = Self::from_slot(slot).decoded.get_chars(Some(remaining));
            remaining -= chars.code_points().count();
            result.push_wtf8(&chars);
        }
        Ok(pyre_object::unicodeobject::w_str_from_wtf8_managed(result))
    }

    fn readline_impl(&mut self, limit: Option<usize>) -> Result<PyObjectRef, crate::PyError> {
        if limit == Some(0) {
            return Ok(w_str_new(""));
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let mut result = Wtf8Buf::new();
        let mut pending_cr = false;
        loop {
            if !Self::from_slot(slot).ensure_data(0)? {
                if pending_cr {
                    result.push_char('\r');
                }
                break;
            }

            let this = Self::from_slot(slot);
            if pending_cr {
                result.push_char('\r');
                pending_cr = false;
                if this.decoded.starts_with_lf() {
                    result.push_char('\n');
                    this.decoded.advance();
                    break;
                }
            }

            let used = result.code_points().count();
            let remaining = limit.map(|limit| limit.saturating_sub(used));
            let start = this.decoded.pos;
            let found = this.scan_line_ending(remaining);
            result.push_wtf8(&this.decoded.consumed_from(start));

            if found || limit.is_some_and(|limit| result.code_points().count() >= limit) {
                break;
            }

            if !this.decoded.exhausted() {
                let remnant = this.decoded.get_chars(None);
                if this.configured_newline() == Some("\r\n")
                    && remnant.code_points().count() == 1
                    && remnant.ascii_byte_at(0) == b'\r'
                {
                    pending_cr = true;
                } else {
                    result.push_wtf8(&remnant);
                }
            }
            this.decoded.reset();
        }
        Ok(pyre_object::unicodeobject::w_str_from_wtf8_managed(result))
    }

    /// PyPy `W_TextIOWrapper._fix_encoder_state` — `if self.seekable and
    /// self.w_encoder:`.  The encoder is half of that condition: a stream with
    /// nothing to encode has no start-of-stream state to fix, and asking the
    /// buffer where it is would cost an `lseek` for an answer nobody reads.
    fn reset_encoder_state(&mut self) {
        self.encoding_start_of_stream = false;
        if self.seekable_flag && !self.w_encoder.is_null() {
            self.encoding_start_of_stream = true;
            let _roots = pyre_object::gc_roots::push_roots();
            let slot = self.pin_self();
            let buffer_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_buffer);
            if let Ok(w_position) = super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(buffer_slot),
                "tell",
                &[],
            ) && crate::builtins::space_index_w(w_position).unwrap_or(0) != 0
            {
                Self::from_slot(slot).encoding_start_of_stream = false;
                let encoder_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_encoder);
                let state_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(w_int_new(0));
                let _ = super::call_method_result(
                    pyre_object::gc_roots::shadow_stack_get(encoder_slot),
                    "setstate",
                    &[pyre_object::gc_roots::shadow_stack_get(state_slot)],
                );
            }
        }
    }

    #[allow(dead_code)]
    fn size_args(w_size: PyObjectRef) -> Vec<PyObjectRef> {
        if unsafe { pyre_object::is_none(w_size) } {
            Vec::new()
        } else {
            vec![w_size]
        }
    }

    fn lookup_text_codec(encoding: &str) -> Result<PyObjectRef, crate::PyError> {
        // Normal interpreter startup installs an ExecutionContext before
        // Python-visible I/O can run.  A handful of Rust-level `open()` unit
        // tests deliberately exercise the builtin without booting an
        // interpreter; codec lookup cannot import `encodings` in that host
        // seam because no module globals owner exists yet.
        if crate::call::getexecutioncontext().is_null() {
            return Ok(PY_NULL);
        }
        crate::module::_codecs::lookup_text_codec("open", encoding)
    }

    /// Install the buffer and everything derived from it.  The interpreter
    /// built standard streams run this too, so they reach the same state a
    /// constructed wrapper does instead of only the fields a struct literal
    /// can spell.
    #[allow(clippy::too_many_arguments)]
    fn attach_buffer(
        &mut self,
        buffer: PyObjectRef,
        encoding: &str,
        errors: &str,
        newline: PyObjectRef,
        newline_value: Option<&str>,
        codec: PyObjectRef,
        line_buffering: bool,
        write_through: bool,
    ) -> Result<(), crate::PyError> {
        // Every `w_str_new` here allocates and `set_newline` /
        // `getattr_str` / `set_encoder_decoder` below run Python, so a
        // collection can land between any two of these stores.  Publish after
        // each reference store instead of once at the end, or a store made
        // before a collection is one the collection never traced.  Reload
        // `self` from its slot after each allocating call — a nursery
        // `TextIOWrapper` subclass (`StdIOBuffer`) moves.
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let buffer_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(buffer);
        let newline_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(newline);
        let codec_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(codec);

        {
            let this = Self::from_slot(slot);
            this.w_buffer = pyre_object::gc_roots::shadow_stack_get(buffer_slot);
            this.publish_refs();
        }
        let encoding_obj = w_str_new_managed(encoding);
        {
            let this = Self::from_slot(slot);
            this.w_encoding = encoding_obj;
            this.publish_refs();
        }
        let errors_obj = w_str_new_managed(errors);
        {
            let this = Self::from_slot(slot);
            this.w_errors = errors_obj;
            this.publish_refs();
        }
        {
            let this = Self::from_slot(slot);
            this.w_newline = pyre_object::gc_roots::shadow_stack_get(newline_slot);
            this.publish_refs();
            this.line_buffering = line_buffering;
            this.write_through = write_through;
            this.decoded.reset();
            this.snapshot = None;
            this.pending_bytes = None;
            this.pending_bytes_count = 0;
            this.chunk_size = 8192;
            this.b2cratio = 0.0;
            this.set_newline(newline_value);
        }
        Self::from_slot(slot)
            .set_encoder_decoder(pyre_object::gc_roots::shadow_stack_get(codec_slot))?;
        let seekable = crate::baseobjspace::is_true(super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(buffer_slot),
            "seekable",
            &[],
        )?)?;
        {
            let this = Self::from_slot(slot);
            this.seekable_flag = seekable;
            this.telling = seekable;
        }
        let has_read1 = buffer_has_read1(pyre_object::gc_roots::shadow_stack_get(buffer_slot))?;
        {
            let this = Self::from_slot(slot);
            this.has_read1 = has_read1;
            this.state = STATE_OK;
            this.reset_encoder_state();
            this.publish_refs();
        }
        Ok(())
    }

    /// Allocate the typed payload used by the interpreter-created standard
    /// streams.  Their methods and metadata remain instance overrides until
    /// sys stream construction is ported to a real FileIO-backed pipeline.
    pub(crate) fn allocate_stdio(
        name: &str,
        buffer: PyObjectRef,
        encoding: &str,
        errors: &str,
        line_buffering: bool,
        write_through: bool,
    ) -> PyObjectRef {
        // Establish the Python type and its mapdict layout before allocating
        // a payload whose stdio methods are installed in the instance dict.
        let _ = type_object();
        let obj = Self::allocate_stable(Self {
            w_stdio_name: w_str_new_managed(name),
            ..Self::default()
        });
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let buffer_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(buffer);
        // `app_main.py create_stdio:494`
        // `newline = None if sys.platform == 'win32' else '\n'`.  Universal
        // newline mode must never be on for POSIX stdio — '\r\n' in stdin does
        // not mean '\n' there, unlike a file explicitly opened in text mode —
        // while Windows wants the translation in both directions.
        let newline_value = if cfg!(windows) { None } else { Some("\n") };
        let newline = newline_value.map_or_else(w_none, w_str_new);
        // No codec here: looking one up imports `encodings`, whose own
        // `import sys` would re-enter the `sys` creation this runs inside and
        // hand the import system a second, half built module.
        // [`attach_stdio_codec`] supplies it once the import system is up.
        if Self::from_slot(obj_slot)
            .attach_buffer(
                pyre_object::gc_roots::shadow_stack_get(buffer_slot),
                encoding,
                errors,
                newline,
                newline_value,
                PY_NULL,
                line_buffering,
                write_through,
            )
            .is_err()
        {
            // With a null codec the only fallible step `attach_buffer` reaches
            // is `buffer.seekable()`, and a descriptor the host refused to open
            // arrives as no buffer at all.  `create_stdio`
            // (`app_main.py`) propagates such a failure, but this runs
            // inside `sys` module creation — there is no interpreter to raise
            // into yet, and the streams have to exist for one to start.  Finish
            // the two fields the early return skipped rather than leaving them
            // at their `Default`, so the stream below is initialized whichever
            // path built it.
            let payload = Self::from_slot(obj_slot);
            payload.seekable_flag = false;
            payload.telling = false;
            payload.reset_encoder_state();
        }
        {
            let payload = Self::from_slot(obj_slot);
            payload.state = STATE_OK;
            payload.publish_refs();
        }
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(name));
        crate::baseobjspace::setdictvalue_native(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            "name",
            pyre_object::gc_roots::shadow_stack_get(name_slot),
        );
        pyre_object::gc_roots::shadow_stack_get(obj_slot)
    }

    /// What [`Self::write_newline`] answers for `stream`, for the paths that
    /// reach fd 1/2 without going through `write` and so have to apply the
    /// substitution themselves.
    ///
    /// The outer `None` is "not a wrapper, so it states no mode" — a caller
    /// writing to the descriptor regardless of what `sys.stderr` was replaced
    /// with still has to pick one — and is distinct from the inner `None`,
    /// which is a wrapper asking for no translation.
    pub fn stdio_write_newline(stream: PyObjectRef) -> Option<Option<&'static str>> {
        if stream.is_null()
            || !std::ptr::eq(
                unsafe { pyre_object::ll_type(stream) },
                <Self as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
            )
        {
            return None;
        }
        Some(unsafe { &*(stream as *const Self) }.write_newline())
    }

    /// The error handler the native `print()` path has to render with, for a
    /// `stream` whose codec that path can spell at all.
    ///
    /// `print()` short-circuits the unmodified `sys.stdout` to `print_output`,
    /// which writes utf-8 bytes straight to the descriptor, so the
    /// short-circuit stands in for the stream's own `write` only while the
    /// stream's codec is utf-8: under `PYTHONIOENCODING=cp424`,
    /// `sys.stdout.write(chr(0xa2))` writes `b'J'` where the native path writes
    /// `b'\xc2\xa2'`. Both handlers pyre's own standard streams take -- `strict`
    /// and the `surrogateescape` a C/POSIX locale asks for -- are spelled by
    /// `encode_object`, and they part company only for a string holding a
    /// surrogate. `reconfigure` rewrites both fields, so this follows it.
    ///
    /// `None` for every other stream, including one this did not build: it
    /// states no codec here, and the caller has to go through `write`.
    ///
    /// A closed stream is one of them.  `write` starts at `CHECK_CLOSED` and
    /// the descriptor path has no such step, so short-circuiting a stream the
    /// program has closed would keep writing where `print` owes a
    /// `ValueError` -- the whole stack is closed by then, so there is nothing
    /// left for the fast path to write to either.  Declining costs nothing
    /// when the check itself raises: `write` runs the same check and reports
    /// what it raised.
    pub fn stdio_native_print_errors(stream: PyObjectRef) -> Option<&'static str> {
        if stream.is_null()
            || !std::ptr::eq(
                unsafe { pyre_object::ll_type(stream) },
                <Self as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
            )
        {
            return None;
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(stream);
        if Self::from_slot(slot).check_closed().is_err() {
            return None;
        }
        let payload = Self::from_slot(slot);
        let (Some(encoding), Some(errors)) = (
            unsafe { pyre_object::w_str_get_value_opt(payload.w_encoding) },
            unsafe { pyre_object::w_str_get_value_opt(payload.w_errors) },
        ) else {
            return None;
        };
        // `encodings.normalize_encoding` folds case and separators, so the
        // aliases `utf8`, `utf-8` and `UTF_8` all name the same codec.
        let normalized: String = encoding
            .chars()
            .filter(|c| !matches!(c, '-' | '_'))
            .flat_map(char::to_lowercase)
            .collect();
        if normalized != "utf8" {
            return None;
        }
        match errors {
            "strict" => Some("strict"),
            "surrogateescape" => Some("surrogateescape"),
            _ => None,
        }
    }

    /// Give a stream [`allocate_stdio`] built the encoder and decoder it had to
    /// go without.  Every read path needs the decoder: without it the stream
    /// reports itself unreadable however readable its buffer is, and only the
    /// methods `make_std_stream` installs as instance overrides work.
    ///
    /// The standard streams are constructed while the `sys` module itself is
    /// being created, and the codec lookup imports `encodings` — an import that
    /// cannot run before `importlib._bootstrap` registers itself, which is why
    /// the bootstrap tail is what calls this.
    ///
    /// Skips a stream this did not build — a replaced `sys.stdout`, one that
    /// already has a codec, one carrying no encoding string.  A codec the
    /// registry refuses, or an incremental encoder/decoder that will not
    /// construct, is NOT skipped: `create_stdio` (`app_main.py`) builds
    /// the wrapper with its codec in one step and propagates both, and a stream
    /// left silently without one reports itself unreadable however readable its
    /// buffer is.
    pub fn attach_stdio_codec(stream: PyObjectRef) -> Result<(), crate::PyError> {
        if stream.is_null()
            || !std::ptr::eq(
                unsafe { pyre_object::ll_type(stream) },
                <Self as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
            )
        {
            return Ok(());
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let stream_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(stream);
        let payload = Self::from_slot(stream_slot);
        // The buffer is read below for the probes `descr_init` runs against it,
        // so a wrapper without one has nothing to attach rather than a null to
        // dereference.
        if !payload.w_encoder.is_null()
            || !payload.w_decoder.is_null()
            || payload.w_buffer.is_null()
            || unsafe { pyre_object::is_none(payload.w_buffer) }
        {
            return Ok(());
        }
        let Some(encoding) =
            unsafe { pyre_object::w_str_get_value_opt(payload.w_encoding) }.map(str::to_owned)
        else {
            return Ok(());
        };
        let codec = Self::lookup_text_codec(&encoding)?;
        let _ = pyre_object::gc_roots::pin_root(codec);
        let codec_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        // `app_main.create_stdio` constructs the standard streams through
        // `_io.open`, whose codec lookup publishes the canonical spelling
        // (`UTF-8` -> `utf-8`, `latin1` -> `iso8859-1`).  A directly-created
        // TextIOWrapper still retains the caller's spelling; only this deferred
        // standard-stream path performs the canonicalisation.
        let canonical_name = crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(codec_slot),
            "name",
        )
        .ok()
        .filter(|name| unsafe { pyre_object::is_str(*name) });
        let canonical_slot = canonical_name.map(|name| {
            let _ = pyre_object::gc_roots::pin_root(name);
            pyre_object::gc_roots::shadow_stack_len() - 1
        });
        let payload = Self::from_slot(stream_slot);
        if let Some(slot) = canonical_slot {
            payload.w_encoding = pyre_object::gc_roots::shadow_stack_get(slot);
            payload.publish_refs();
        }
        Self::from_slot(stream_slot)
            .set_encoder_decoder(pyre_object::gc_roots::shadow_stack_get(codec_slot))?;
        // The rest of the `descr_init` tail `allocate_stdio` also had to defer:
        // the seekable/telling pair and the `read1` probe both call into the
        // buffer, which is why they wait for the same moment the codec does.
        // `interp_textio.py` runs them in this order, after the codec.
        let buffer_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(stream_slot).w_buffer);
        if let Ok(w_seekable) = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(buffer_slot),
            "seekable",
            &[],
        ) && let Ok(seekable) = crate::baseobjspace::is_true(w_seekable)
        {
            let payload = Self::from_slot(stream_slot);
            payload.seekable_flag = seekable;
            payload.telling = seekable;
        }
        Self::from_slot(stream_slot).has_read1 =
            buffer_has_read1(pyre_object::gc_roots::shadow_stack_get(buffer_slot))?;
        Ok(())
    }
}

#[crate::pyre_methods(
    base = super::text_iobase_type(),
    weakrefable,
    doc = "TextIOWrapper(buffer, encoding=None, errors=None, newline=None, line_buffering=False, write_through=False)"
)]
impl W_TextIOWrapper {
    #[staticmethod]
    fn __new__(cls: PyObjectRef, _args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // `check_user_subclass` can collect. The slot is the live word;
        // this pin's argument is not read again.
        let _roots = pyre_object::gc_roots::push_roots();
        let cls_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(cls);
        crate::typedef::check_user_subclass(
            type_object(),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        )?;
        let obj = Self::allocate_instance(
            Self::default(),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        );
        Ok(super::tag_io_instance(
            obj,
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        ))
    }

    fn __init__(
        &mut self,
        mut buffer: PyObjectRef,
        #[default(pyre_object::w_none())] mut encoding: PyObjectRef,
        #[default(pyre_object::w_none())] mut errors: PyObjectRef,
        #[default(pyre_object::w_none())] mut newline: PyObjectRef,
        #[default(pyre_object::w_bool_from(false))] mut line_buffering: PyObjectRef,
        #[default(pyre_object::w_bool_from(false))] mut write_through: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        // Every argument is converted before the body runs, in the order the
        // signature gives them: `encoding` and `newline` are accepted only as
        // `str` or `None`, and the two flags are truth-tested.  3.14's
        // constructor uses the `bool` converter for those, unlike
        // `reconfigure`'s `int` one — an object with `__bool__` but no
        // `__index__` is accepted here — and a `__bool__` that raises ends
        // the call before `self->ok = 0`, leaving a stream that was already
        // open still open.
        let unspecified = if crate::importing::utf8_mode_flag() != 0 {
            "utf-8"
        } else {
            "locale"
        };
        // A heap-type subclass (`StdIOBuffer(TextIOWrapper)`) is nursery-born.
        // `checked_text0` / `is_true` / codec lookup each allocate; pin the
        // wrapper and every argument and reload `self` after those calls
        // (`interp_textio.py` `descr_init` — RPython roots `self`).
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let args_base = _roots.pin_roots(&[
            buffer,
            encoding,
            errors,
            newline,
            line_buffering,
            write_through,
        ]);
        buffer = _roots.get(args_base);
        encoding = _roots.get(args_base + 1);
        errors = _roots.get(args_base + 2);
        newline = _roots.get(args_base + 3);
        line_buffering = _roots.get(args_base + 4);
        write_through = _roots.get(args_base + 5);
        let encoding_text = pyre_object::with_roots!(buffer, encoding, errors, line_buffering, newline, write_through => Self::checked_text0(encoding, unspecified, "encoding"))?;
        if !unsafe { pyre_object::is_none(newline) || pyre_object::is_str(newline) } {
            return Err(crate::PyError::type_error("illegal newline type"));
        }
        let line_buffering = pyre_object::with_roots!(buffer, encoding, errors, newline, write_through => crate::baseobjspace::is_true(line_buffering))?;
        let write_through = pyre_object::with_roots!(buffer, encoding, errors, newline => crate::baseobjspace::is_true(write_through))?;

        // PyPy starts every initialization attempt in STATE_ZERO.  A failed
        // reinitialization must leave all I/O operations uninitialized.
        {
            let this = Self::from_slot(slot);
            this.state = STATE_ZERO;
            this.w_buffer = PY_NULL;
            this.publish_refs();
        }

        // An unspecified `encoding` reads the locale's, unless UTF-8 mode has
        // already answered the question.  `_io.text_encoding` is not on this
        // path - a direct `TextIOWrapper(...)` call reaches the constructor
        // itself - so the warning that argument's absence carries is raised
        // here, at this frame.  [3.15-spec] No trailing period, as in
        // `textio.c _io_TextIOWrapper___init___impl`; see `text_encoding`.
        if unsafe { pyre_object::is_none(encoding) }
            && crate::importing::warn_default_encoding_flag()
        {
            pyre_object::with_roots!(buffer, errors, newline => crate::warn::warn_category("'encoding' argument not specified", "EncodingWarning", 1))?;
        }
        let errors = pyre_object::with_roots!(buffer, newline => Self::checked_text0(errors, "strict", "errors"))?;
        pyre_object::with_roots!(buffer, newline => Self::io_check_errors(&errors))?;
        let newline_value = Self::unwrap_newline(newline)?;
        let encoding = Self::resolve_locale_encoding(encoding_text);
        let codec =
            pyre_object::with_roots!(buffer, newline => Self::lookup_text_codec(&encoding))?;

        Self::from_slot(slot).attach_buffer(
            buffer,
            &encoding,
            &errors,
            newline,
            newline_value.as_deref(),
            codec,
            line_buffering,
            write_through,
        )
    }

    fn read(
        &mut self,
        #[default(pyre_object::w_none())] mut w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        pyre_object::with_roots!(w_size => Self::from_slot(slot).check_closed())?;
        pyre_object::with_roots!(w_size => Self::from_slot(slot).ensure_read_decoder())?;
        if Self::from_slot(slot).w_decoder.is_null() {
            return Err(super::unsupported("not readable"));
        }
        pyre_object::with_roots!(w_size => Self::from_slot(slot).write_flush())?;
        match Self::size_limit(w_size)? {
            None => Self::from_slot(slot).read_all(),
            Some(size) => Self::from_slot(slot).read_n(size),
        }
    }

    fn readline(
        &mut self,
        #[default(pyre_object::PY_NULL)] mut w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        pyre_object::with_roots!(w_size => Self::from_slot(slot).check_closed())?;
        pyre_object::with_roots!(w_size => Self::from_slot(slot).ensure_read_decoder())?;
        if Self::from_slot(slot).w_decoder.is_null() {
            return Err(super::unsupported("not readable"));
        }
        pyre_object::with_roots!(w_size => Self::from_slot(slot).write_flush())?;
        // PyPy `W_TextIOWrapper.readline_w` keeps the interp2app default as
        // the untranslated null sentinel and rejects only an explicit None.
        if !w_size.is_null() && unsafe { pyre_object::is_none(w_size) } {
            return Err(crate::PyError::type_error(
                "'NoneType' object cannot be interpreted as an integer",
            ));
        }
        Self::from_slot(slot).readline_impl(Self::size_limit(w_size)?)
    }

    fn readlines(
        &mut self,
        #[default(pyre_object::w_none())] w_hint: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        super::iobase_readlines(&[self.self_obj(), w_hint])
    }

    /// PyPy `_writeflush_loop`; kept separate so the generated JIT can still
    /// trace through the ordinary `write()` fast path.
    fn write_flush_loop(&mut self) -> Result<(), crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        while Self::from_slot(slot).pending_bytes.is_some() {
            Self::from_slot(slot).write_flush()?;
        }
        Ok(())
    }

    /// PyPy `_writeflush` inlinable fast path.
    fn write_flush(&mut self) -> Result<(), crate::PyError> {
        if self.pending_bytes.is_none() {
            return Ok(());
        }
        self.really_flush()
    }

    /// PyPy `_really_flush`: clear the pending list before invoking
    /// `buffer.write`, so a reentrant `TextIOWrapper.write` starts a new list
    /// which the outer `_writeflush_loop` can subsequently drain.
    fn really_flush(&mut self) -> Result<(), crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let this = Self::from_slot(slot);
        let chunks = this
            .pending_bytes
            .take()
            .expect("write_flush only calls really_flush with pending bytes");
        this.pending_bytes_count = 0;
        let total = chunks.iter().map(Vec::len).sum();
        let mut pending = Vec::with_capacity(total);
        for chunk in chunks {
            pending.extend_from_slice(&chunk);
        }
        let bytes = pyre_object::bytesobject::w_bytes_from_bytes(&pending);
        Self::from_slot(slot).call_buffer("write", &[bytes])?;
        Ok(())
    }

    fn write(&mut self, mut text: PyObjectRef) -> Result<i64, crate::PyError> {
        let roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        let text_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(text);
        text = roots.get(text_slot);
        pyre_object::with_roots!(text => Self::from_slot(slot).check_closed())?;
        text = roots.get(text_slot);
        if Self::from_slot(slot).w_encoder.is_null() {
            return Err(super::unsupported("not writable"));
        }
        if unsafe { !pyre_object::is_str(text) } {
            return Err(crate::PyError::type_error(format!(
                "unicode argument expected, got '{}'",
                crate::type_methods::arg_type_name(text)
            )));
        }
        let nchars = unsafe { pyre_object::w_str_len(text) };
        let text_wtf8 = unsafe { pyre_object::w_str_get_wtf8(text) };
        let has_lf = text_wtf8
            .code_points()
            .any(|cp| cp.to_u32() == b'\n' as u32);
        let has_cr = text_wtf8
            .code_points()
            .any(|cp| cp.to_u32() == b'\r' as u32);

        let mut to_encode_slot = text_slot;
        if has_lf && let Some(writenl) = Self::from_slot(slot).write_newline() {
            let nl_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(w_str_new_managed("\n"));
            let wr_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(w_str_new_managed(writenl));
            let replaced = super::call_method_result(
                roots.get(text_slot),
                "replace",
                &[roots.get(nl_slot), roots.get(wr_slot)],
            )?;
            to_encode_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(replaced);
        }

        let this = Self::from_slot(slot);
        let need_flush = this.line_buffering && (has_lf || has_cr);
        let text_need_flush = this.write_through;
        let encoder_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(this.w_encoder);
        let encoded = super::call_method_result(
            roots.get(encoder_slot),
            "encode",
            &[roots.get(to_encode_slot)],
        )?;
        if unsafe { !pyre_object::bytesobject::is_bytes(encoded) } {
            return Err(crate::PyError::type_error(format!(
                "encoder should return a bytes object, not '{}'",
                crate::type_methods::arg_type_name(encoded)
            )));
        }
        Self::from_slot(slot).encoding_start_of_stream = false;
        let bytes = unsafe { pyre_object::bytesobject::w_bytes_data(encoded) }.to_vec();

        {
            let this = Self::from_slot(slot);
            if bytes.len() >= this.chunk_size as usize && this.pending_bytes.is_some() {
                this.write_flush_loop()?;
            }
        }
        {
            let this = Self::from_slot(slot);
            match this.pending_bytes.as_mut() {
                Some(pending) => pending.push(bytes),
                None => this.pending_bytes = Some(vec![bytes]),
            }
            this.pending_bytes_count += this
                .pending_bytes
                .as_ref()
                .and_then(|pending| pending.last())
                .map_or(0, Vec::len);

            if this.pending_bytes_count >= this.chunk_size as usize || need_flush || text_need_flush
            {
                this.write_flush()?;
            }
        }
        if need_flush {
            Self::from_slot(slot).call_buffer("flush", &[])?;
        }
        {
            let this = Self::from_slot(slot);
            if this.snapshot.is_some() {
                this.decoded.reset();
                this.snapshot = None;
            }
        }
        if !Self::from_slot(slot).w_decoder.is_null() {
            let decoder_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(Self::from_slot(slot).w_decoder);
            super::call_method_result(roots.get(decoder_slot), "reset", &[])?;
        }
        Ok(nchars as i64)
    }

    fn writelines(&self, lines: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        super::iobase_writelines(&[self.self_obj(), lines])
    }

    fn flush(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_closed()?;
        let this = Self::from_slot(slot);
        this.telling = this.seekable_flag;
        this.write_flush()?;
        Self::from_slot(slot).call_buffer("flush", &[])
    }

    fn close(&mut self) -> Result<(), crate::PyError> {
        // Keep the original buffer alive across the user-overridable
        // `closed` property.  CPython GH-142594: that property may reenter
        // and detach this wrapper. Pin `self` first: `getattr`/`flush`/
        // `close` each allocate, and a nursery `TextIOWrapper` subclass
        // (`StdIOBuffer`) moves (`interp_textio.py` `close_w`).
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_attached()?;
        let buffer_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_buffer);
        let closed = crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(buffer_slot),
            "closed",
        )?;
        if crate::baseobjspace::is_true(closed)? {
            return Ok(());
        }
        if Self::from_slot(slot).state == STATE_DETACHED {
            super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(buffer_slot),
                "close",
                &[],
            )?;
            return Ok(());
        }

        // PyPy: `try: self.flush() finally: self.buffer.close()`.  The flush
        // is virtual, and a close failure replaces it while retaining the
        // flush exception as `__context__`.
        let flush_error =
            super::call_method_result(Self::from_slot(slot).self_obj(), "flush", &[]).err();
        let (close_result, flush_error) = if let Some(error) = flush_error {
            let _roots = pyre_object::gc_roots::push_roots();
            let mut error = error;
            let error_slot = error.pin(&_roots);
            let close_result = super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(buffer_slot),
                "close",
                &[],
            );
            error.reload(&_roots, error_slot);
            (close_result, Some(error))
        } else {
            (
                super::call_method_result(
                    pyre_object::gc_roots::shadow_stack_get(buffer_slot),
                    "close",
                    &[],
                ),
                None,
            )
        };
        if let Err(close_error) = close_result {
            if let Some(flush_error) = flush_error {
                let _roots = pyre_object::gc_roots::push_roots();
                let mut close_error = close_error;
                let mut flush_error = flush_error;
                let close_slot = close_error.pin(&_roots);
                let flush_slot = flush_error.pin(&_roots);
                let flush_obj = flush_error.to_exc_object();
                flush_error.reload(&_roots, flush_slot);
                let _ = pyre_object::gc_roots::pin_root(flush_obj);
                let flush_obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                close_error.reload(&_roots, close_slot);
                let close_obj = close_error.to_exc_object();
                close_error.reload(&_roots, close_slot);
                let _ = pyre_object::gc_roots::pin_root(close_obj);
                let close_obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                unsafe {
                    pyre_object::interp_exceptions::w_exception_set_context(
                        pyre_object::gc_roots::shadow_stack_get(close_obj_slot),
                        pyre_object::gc_roots::shadow_stack_get(flush_obj_slot),
                    )
                };
                close_error.reload(&_roots, close_slot);
                close_error.set_exc_object(pyre_object::gc_roots::shadow_stack_get(close_obj_slot));
                return Err(close_error);
            }
            return Err(close_error);
        }
        if let Some(error) = flush_error {
            return Err(error);
        }
        // interp_textio.py `close_w`: the wrapper is closed, so `iobase_del`'s
        // own work is done.  On the success path only — a close that raised
        // leaves the object in a state its finalizer still has something to
        // say about.
        super::maybe_unregister_rpython_finalizer_io(Self::from_slot(slot).self_obj());
        Ok(())
    }

    fn _dealloc_warn(&self, w_source: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        self.check_attached()?;
        super::call_method_result(self.w_buffer, "_dealloc_warn", &[w_source])
    }

    fn detach(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_attached()?;
        super::call_method_result(Self::from_slot(slot).self_obj(), "flush", &[])?;
        let this = Self::from_slot(slot);
        let buffer = this.w_buffer;
        this.w_buffer = PY_NULL;
        this.state = STATE_DETACHED;
        pyre_object::gc_hook::try_gc_write_barrier(
            this as *mut Self as pyre_object::gc_hook::GCREF,
        );
        Ok(buffer)
    }

    fn tell(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let _self_roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_closed()?;
        let this = Self::from_slot(slot);
        if !this.seekable_flag {
            return Err(super::unsupported("underlying stream is not seekable"));
        }
        if !this.telling {
            return Err(crate::PyError::os_error(
                "telling position disabled by next() call",
            ));
        }

        Self::from_slot(slot).write_flush()?;
        super::call_method_result(Self::from_slot(slot).self_obj(), "flush", &[])?;
        let w_pos = Self::from_slot(slot).call_buffer("tell", &[])?;
        let this = Self::from_slot(slot);
        if this.w_decoder.is_null() || this.snapshot.is_none() {
            return Ok(w_pos);
        }

        let w_index = crate::baseobjspace::space_index(w_pos)?;
        let raw_pos = unsafe { crate::builtins::obj_to_bigint(w_index) };
        let mut cookie = PositionCookie::unpack(raw_pos)?;
        let this = Self::from_slot(slot);
        let snapshot = this.snapshot.as_ref().expect("checked above");
        let input = snapshot.input.clone();
        cookie.dec_flags = snapshot.flags;
        if input.len() as u64 > cookie.start_pos {
            // interp_textio.py:tell_w, CPython GH-95782: devices may report
            // zero even though the decoder snapshot still contains input.
            return Ok(pyre_object::w_int_new(0));
        }
        cookie.start_pos -= input.len() as u64;
        if this.decoded.pos == 0 {
            return Ok(cookie.to_object());
        }

        let mut chars_to_skip = this.decoded.upos as u64;
        let decoder_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_decoder);
        let saved_state = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(decoder_slot),
            "getstate",
            &[],
        )?;
        let _saved_roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(saved_state);
        let saved_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

        let result = (|| -> Result<PyObjectRef, crate::PyError> {
            // PyPy's b2cratio heuristic searches backward for a decoder state
            // with no buffered bytes.
            let mut skip_bytes =
                ((Self::from_slot(slot).b2cratio * chars_to_skip as f64) as usize).min(input.len());
            let mut skip_back = 1usize;
            while skip_bytes > 0 {
                Self::from_slot(slot).decoder_setstate(&cookie)?;
                let bytes = pyre_object::bytesobject::w_bytes_from_bytes(&input[..skip_bytes]);
                let decoded = super::call_method_result(
                    pyre_object::gc_roots::shadow_stack_get(decoder_slot),
                    "decode",
                    &[bytes],
                )?;
                if unsafe { !pyre_object::is_str(decoded) } {
                    return Err(crate::PyError::type_error(format!(
                        "decoder should return a string result, not '{}'",
                        crate::type_methods::arg_type_name(decoded)
                    )));
                }
                let chars_decoded = unsafe { pyre_object::w_str_len(decoded) } as u64;
                if chars_decoded <= chars_to_skip {
                    let (dec_buffer, flags) = Self::from_slot(slot).decoder_getstate()?;
                    if dec_buffer.is_empty() {
                        cookie.dec_flags = flags;
                        chars_to_skip -= chars_decoded;
                        break;
                    }
                    skip_bytes = skip_bytes.saturating_sub(dec_buffer.len());
                    skip_back = 1;
                } else {
                    skip_bytes = skip_bytes.saturating_sub(skip_back);
                    skip_back = skip_back.saturating_mul(2);
                }
            }
            if skip_bytes == 0 {
                Self::from_slot(slot).decoder_setstate(&cookie)?;
            }
            cookie.start_pos += skip_bytes as u64;
            cookie.chars_to_skip = chars_to_skip;
            if chars_to_skip == 0 {
                return Ok(cookie.to_object());
            }

            Self::from_slot(slot).decoder_setstate(&cookie)?;
            let mut chars_decoded = 0u64;
            let mut i = skip_bytes;
            while i < input.len() {
                let bytes = pyre_object::bytesobject::w_bytes_from_bytes(&input[i..i + 1]);
                let decoded = super::call_method_result(
                    pyre_object::gc_roots::shadow_stack_get(decoder_slot),
                    "decode",
                    &[bytes],
                )?;
                if unsafe { !pyre_object::is_str(decoded) } {
                    return Err(crate::PyError::type_error(format!(
                        "decoder should return a string result, not '{}'",
                        crate::type_methods::arg_type_name(decoded)
                    )));
                }
                chars_decoded += unsafe { pyre_object::w_str_len(decoded) } as u64;
                cookie.bytes_to_feed += 1;

                let (dec_buffer, flags) = Self::from_slot(slot).decoder_getstate()?;
                if dec_buffer.is_empty() && chars_decoded <= chars_to_skip {
                    cookie.start_pos += cookie.bytes_to_feed;
                    chars_to_skip -= chars_decoded;
                    cookie.dec_flags = flags;
                    cookie.bytes_to_feed = 0;
                    chars_decoded = 0;
                }
                if chars_decoded >= chars_to_skip {
                    break;
                }
                i += 1;
            }
            if chars_decoded < chars_to_skip {
                let bytes = pyre_object::bytesobject::w_bytes_empty();
                let decoded = super::call_method_result(
                    pyre_object::gc_roots::shadow_stack_get(decoder_slot),
                    "decode",
                    &[bytes, w_bool_from(true)],
                )?;
                if unsafe { !pyre_object::is_str(decoded) } {
                    return Err(crate::PyError::type_error(format!(
                        "decoder should return a string result, not '{}'",
                        crate::type_methods::arg_type_name(decoded)
                    )));
                }
                chars_decoded += unsafe { pyre_object::w_str_len(decoded) } as u64;
                cookie.need_eof = true;
                if chars_decoded < chars_to_skip {
                    return Err(crate::PyError::os_error(
                        "can't reconstruct logical file position",
                    ));
                }
            }
            cookie.chars_to_skip = chars_to_skip;
            Ok(cookie.to_object())
        })();

        match result {
            Err(error) => {
                let _roots = pyre_object::gc_roots::push_roots();
                let mut error = error;
                let error_slot = error.pin(&_roots);
                let _ = super::call_method_result(
                    pyre_object::gc_roots::shadow_stack_get(decoder_slot),
                    "setstate",
                    &[pyre_object::gc_roots::shadow_stack_get(saved_slot)],
                );
                error.reload(&_roots, error_slot);
                Err(error)
            }
            Ok(value) => {
                let _roots = pyre_object::gc_roots::push_roots();
                let value = pyre_object::gc_roots::pin_root(value);
                match super::call_method_result(
                    pyre_object::gc_roots::shadow_stack_get(decoder_slot),
                    "setstate",
                    &[pyre_object::gc_roots::shadow_stack_get(saved_slot)],
                ) {
                    Err(error) => Err(error),
                    Ok(_) => Ok(value),
                }
            }
        }
    }

    fn seek(
        &mut self,
        mut cookie: PyObjectRef,
        #[default(0i64)] whence: i64,
    ) -> Result<PyObjectRef, crate::PyError> {
        let _self_roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        pyre_object::with_roots!(cookie => Self::from_slot(slot).check_closed())?;
        if !Self::from_slot(slot).seekable_flag {
            return Err(super::unsupported("underlying stream is not seekable"));
        }
        let mut w_position = crate::baseobjspace::space_index(cookie)?;
        let mut position = unsafe { crate::builtins::obj_to_bigint(w_position) };

        if whence == 1 {
            if position.tobool() {
                return Err(super::unsupported("can't do nonzero cur-relative seeks"));
            }
            w_position = super::call_method_result(Self::from_slot(slot).self_obj(), "tell", &[])?;
            let indexed = pyre_object::with_roots!(w_position => crate::baseobjspace::space_index(w_position))?;
            position = unsafe { crate::builtins::obj_to_bigint(indexed) };
        } else if whence == 2 {
            if position.tobool() {
                return Err(super::unsupported("can't do nonzero end-relative seeks"));
            }
            // `flush`, `decoder.reset()` and `encoder_reset` call Python, so
            // `w_position` and the buffer's `seek` result are rooted across
            // them and read back from their slots.
            let _roots = pyre_object::gc_roots::push_roots();
            let pos_slot = pyre_object::gc_roots::pin_roots(&[w_position]);
            super::call_method_result(Self::from_slot(slot).self_obj(), "flush", &[])?;
            {
                let this = Self::from_slot(slot);
                this.decoded.reset();
                this.snapshot = None;
            }
            if !Self::from_slot(slot).w_decoder.is_null() {
                let decoder_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_decoder);
                super::call_method_result(
                    pyre_object::gc_roots::shadow_stack_get(decoder_slot),
                    "reset",
                    &[],
                )?;
            }
            let whence_slot = pyre_object::gc_roots::pin_roots(&[w_int_new(whence)]);
            let result = Self::from_slot(slot).call_buffer(
                "seek",
                &[
                    pyre_object::gc_roots::shadow_stack_get(pos_slot),
                    pyre_object::gc_roots::shadow_stack_get(whence_slot),
                ],
            )?;
            let result_slot = pyre_object::gc_roots::pin_roots(&[result]);
            if !Self::from_slot(slot).w_encoder.is_null() {
                let at_start = crate::builtins::space_index_w(
                    pyre_object::gc_roots::shadow_stack_get(result_slot),
                )? == 0;
                Self::from_slot(slot).encoder_reset(at_start)?;
            }
            return Ok(pyre_object::gc_roots::shadow_stack_get(result_slot));
        } else if whence != 0 {
            return Err(crate::PyError::value_error(format!(
                "invalid whence ({whence}, should be 0, 1 or 2)"
            )));
        }

        let position_cookie =
            pyre_object::with_roots!(w_position => PositionCookie::unpack(position))?;
        pyre_object::with_roots!(w_position => super::call_method_result(Self::from_slot(slot).self_obj(), "flush", &[]))?;
        let start = pyre_object::with_roots!(w_position => crate::objspace::descroperation::bigint_result(RBigInt::from(
            position_cookie.start_pos,
        )));
        pyre_object::with_roots!(w_position => Self::from_slot(slot).call_buffer("seek", &[start]))?;
        {
            let this = Self::from_slot(slot);
            this.decoded.reset();
            this.snapshot = None;
        }
        if !Self::from_slot(slot).w_decoder.is_null() {
            pyre_object::with_roots!(w_position => Self::from_slot(slot).decoder_setstate(&position_cookie))?;
        }

        if position_cookie.chars_to_skip != 0 {
            let _roots = pyre_object::gc_roots::push_roots();
            let pos_slot = _roots.pin_roots(&[w_position]);
            let buffer_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_buffer);
            let size_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ =
                pyre_object::gc_roots::pin_root(w_int_new(position_cookie.bytes_to_feed as i64));
            let chunk = super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(buffer_slot),
                "read",
                &[pyre_object::gc_roots::shadow_stack_get(size_slot)],
            )?;
            if unsafe { !pyre_object::bytesobject::is_bytes(chunk) } {
                return Err(crate::PyError::type_error(format!(
                    "underlying read() should have returned a bytes object, not '{}'",
                    crate::type_methods::arg_type_name(chunk)
                )));
            }
            let input = unsafe { pyre_object::bytesobject::w_bytes_data(chunk) }.to_vec();
            Self::from_slot(slot).snapshot = Some(PositionSnapshot {
                flags: position_cookie.dec_flags,
                input,
            });
            let decoder_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_decoder);
            let decoded = super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(decoder_slot),
                "decode",
                &[chunk, w_bool_from(position_cookie.need_eof)],
            )?;
            if unsafe { !pyre_object::is_str(decoded) } {
                return Err(crate::PyError::type_error(format!(
                    "decoder should return a string result, not '{}'",
                    crate::type_methods::arg_type_name(decoded)
                )));
            }
            if unsafe { pyre_object::w_str_len(decoded) } < position_cookie.chars_to_skip as usize {
                return Err(crate::PyError::os_error(
                    "can't restore logical file position",
                ));
            }
            let this = Self::from_slot(slot);
            this.decoded.set(decoded)?;
            this.decoded
                .get_chars(Some(position_cookie.chars_to_skip as usize));
            w_position = _roots.get(pos_slot);
        } else {
            Self::from_slot(slot).snapshot = Some(PositionSnapshot {
                flags: position_cookie.dec_flags,
                input: Vec::new(),
            });
        }

        if !Self::from_slot(slot).w_encoder.is_null() {
            pyre_object::with_roots!(w_position => Self::from_slot(slot).encoder_reset(position_cookie.start_pos == 0 && position_cookie.dec_flags == 0))?;
        }
        Ok(w_position)
    }

    fn truncate(
        &self,
        #[default(pyre_object::w_none())] mut size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        pyre_object::with_roots!(size => Self::from_slot(slot).check_closed())?;
        pyre_object::with_roots!(size => super::call_method_result(Self::from_slot(slot).self_obj(), "flush", &[]))?;
        if unsafe { pyre_object::is_none(size) } {
            Self::from_slot(slot).call_buffer("truncate", &[])
        } else {
            Self::from_slot(slot).call_buffer("truncate", &[size])
        }
    }

    fn reconfigure(
        &mut self,
        #[default(pyre_object::w_none())] encoding: PyObjectRef,
        #[default(pyre_object::w_none())] mut errors: PyObjectRef,
        #[default(pyre_object::PY_NULL)] mut newline: PyObjectRef,
        #[default(pyre_object::w_none())] mut line_buffering: PyObjectRef,
        #[default(pyre_object::w_none())] mut write_through: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_attached()?;
        if Self::from_slot(slot).decoded.text.is_some()
            && (!unsafe { pyre_object::is_none(encoding) }
                || !unsafe { pyre_object::is_none(errors) }
                || !newline.is_null())
        {
            return Err(super::unsupported(
                "It is not possible to set the encoding or newline of stream after the first read",
            ));
        }

        let new_encoding = if unsafe { pyre_object::is_none(encoding) } {
            None
        } else {
            let value = pyre_object::with_roots!(errors, line_buffering, newline, write_through => Self::checked_text(encoding, "", "encoding"))?;
            Some(Self::resolve_locale_encoding(value))
        };
        let new_errors = if unsafe { pyre_object::is_none(errors) } {
            None
        } else {
            let value = pyre_object::with_roots!(line_buffering, newline, write_through => Self::checked_text0(errors, "", "errors"))?;
            pyre_object::with_roots!(line_buffering, newline, write_through => Self::io_check_errors(&value))?;
            Some(value)
        };
        let new_newline = if newline.is_null() {
            None
        } else {
            Some(Self::unwrap_newline(newline)?)
        };
        // CPython 3.14's clinic converter still uses the integer/index
        // protocol for these two flags (the CPython tests deliberately
        // distinguish it from truth testing).
        let new_line_buffering = if unsafe { pyre_object::is_none(line_buffering) } {
            None
        } else {
            Some(
                pyre_object::with_roots!(newline, write_through => crate::builtins::space_index_w(line_buffering))?
                    != 0,
            )
        };
        let new_write_through = if unsafe { pyre_object::is_none(write_through) } {
            None
        } else {
            Some(
                pyre_object::with_roots!(newline => crate::builtins::space_index_w(write_through))?
                    != 0,
            )
        };
        // The incremental newline decoder captures its translation at build
        // time, so a newline change must rebuild the codec whenever it flips
        // `readuniversal` or `readtranslate`; otherwise a universal-to-fixed
        // reconfigure would keep translating `\r\n` with the stale decoder.
        let newline_forces_reset = new_newline.as_ref().is_some_and(|value| {
            let new_readuniversal = value.as_deref().is_none_or(str::is_empty);
            let new_readtranslate = value.is_none();
            let this = Self::from_slot(slot);
            this.readuniversal != new_readuniversal || this.readtranslate != new_readtranslate
        });
        let reset_codec = new_encoding.is_some() || new_errors.is_some() || newline_forces_reset;
        // PyPy/CPython prepare the replacement codec before mutating the
        // wrapper.  In particular, a failing codec lookup must leave
        // `encoding`, `errors`, and the incremental encoder/decoder intact.
        let new_codec = if reset_codec {
            let encoding = if let Some(encoding) = new_encoding.as_deref() {
                encoding
            } else {
                pyre_object::with_roots!(newline => crate::baseobjspace::str_utf8_w(Self::from_slot(slot).w_encoding))?
            };
            // CPython 3.14's TextIOWrapper reconfigure path reaches codec
            // lookup with this value (test_io:test_reconfigure_errors), where
            // an embedded NUL is an unknown encoding rather than the
            // `text0_w` ValueError exposed by codecs.lookup itself.
            if encoding.contains('\0') {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::LookupError,
                    format!("unknown encoding: {encoding}"),
                ));
            }
            Some(pyre_object::with_roots!(newline => Self::lookup_text_codec(encoding))?)
        } else {
            None
        };

        // CPython 3.14 `_textiowrapper_writeflush`: every reconfiguration
        // first commits pending output, even when every option is omitted.
        let has_new_codec = new_codec.is_some();
        let mut new_codec_obj = new_codec.unwrap_or(pyre_object::PY_NULL);
        pyre_object::with_roots!(new_codec_obj, newline => {
            super::call_method_result(Self::from_slot(slot).self_obj(), "flush", &[])
        })?;
        if let Some(value) = new_newline.as_ref() {
            let this = Self::from_slot(slot);
            this.w_newline = newline;
            this.publish_refs();
            this.set_newline(value.as_deref());
        }
        if let Some(value) = new_encoding.as_ref() {
            let encoding_obj = w_str_new_managed(value);
            {
                let this = Self::from_slot(slot);
                this.w_encoding = encoding_obj;
                this.publish_refs();
            }
            let errors_obj = w_str_new_managed(new_errors.as_deref().unwrap_or("strict"));
            let this = Self::from_slot(slot);
            this.w_errors = errors_obj;
            this.publish_refs();
        } else if let Some(value) = new_errors.as_ref() {
            let errors_obj = w_str_new_managed(value);
            let this = Self::from_slot(slot);
            this.w_errors = errors_obj;
            this.publish_refs();
        }
        if has_new_codec {
            Self::from_slot(slot).set_encoder_decoder(new_codec_obj)?;
        }
        if let Some(value) = new_line_buffering {
            Self::from_slot(slot).line_buffering = value;
        }
        if let Some(value) = new_write_through {
            Self::from_slot(slot).write_through = value;
        }
        Self::from_slot(slot).reset_encoder_state();
        Self::from_slot(slot).b2cratio = 0.0;
        Ok(())
    }

    fn readable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.call_buffer("readable", &[])
    }

    fn writable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.call_buffer("writable", &[])
    }

    fn seekable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.call_buffer("seekable", &[])
    }

    fn isatty(&self) -> Result<PyObjectRef, crate::PyError> {
        self.call_buffer("isatty", &[])
    }

    fn fileno(&self) -> Result<PyObjectRef, crate::PyError> {
        self.call_buffer("fileno", &[])
    }

    #[getter]
    fn buffer(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_attached()?;
        Ok(self.w_buffer)
    }

    #[getter]
    fn closed(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_attached()?;
        if unsafe { pyre_object::is_none(self.w_buffer) } {
            return Ok(w_bool_from(false));
        }
        crate::baseobjspace::getattr_str(self.w_buffer, "closed")
    }

    #[getter]
    fn name(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_attached()?;
        if !self.w_stdio_name.is_null() {
            return Ok(self.w_stdio_name);
        }
        crate::baseobjspace::getattr_str(self.w_buffer, "name")
    }

    #[getter]
    fn encoding(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        Ok(self.w_encoding)
    }

    #[getter]
    fn errors(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        Ok(self.w_errors)
    }

    #[getter]
    fn line_buffering(&self) -> Result<bool, crate::PyError> {
        self.check_init()?;
        Ok(self.line_buffering)
    }

    #[getter]
    fn write_through(&self) -> Result<bool, crate::PyError> {
        self.check_init()?;
        Ok(self.write_through)
    }

    #[getter]
    fn newlines(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_attached()?;
        if self.w_decoder.is_null() {
            return Ok(w_none());
        }
        match crate::baseobjspace::getattr_str(self.w_decoder, "newlines") {
            Ok(value) => Ok(value),
            Err(error) if error.kind == crate::PyErrorKind::AttributeError => Ok(w_none()),
            Err(error) => Err(error),
        }
    }

    #[getter]
    #[allow(non_snake_case)]
    fn _CHUNK_SIZE(&self) -> Result<i64, crate::PyError> {
        self.check_attached()?;
        Ok(self.chunk_size)
    }

    #[setter]
    #[allow(non_snake_case)]
    fn set_CHUNK_SIZE(&mut self, size: PyObjectRef) -> Result<(), crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_attached()?;
        let size = crate::baseobjspace::int_w(size)?;
        if size <= 0 {
            return Err(crate::PyError::value_error(
                "a strictly positive integer is required",
            ));
        }
        Self::from_slot(slot).chunk_size = size;
        Ok(())
    }

    fn __enter__(&self) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_closed()?;
        Ok(Self::from_slot(slot).self_obj())
    }

    fn __exit__(
        &mut self,
        _exc_type: PyObjectRef,
        _exc: PyObjectRef,
        _tb: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        self.close()
    }

    fn __iter__(&self) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_closed()?;
        Ok(Self::from_slot(slot).self_obj())
    }

    fn __next__(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_attached()?;
        Self::from_slot(slot).telling = false;
        // PyPy `W_TextIOWrapper.next_w` reaches a virtual `readline()` with
        // no limit argument; preserve that missing sentinel instead of
        // manufacturing an explicit Python None.
        let line = Self::from_slot(slot).readline(PY_NULL)?;
        if unsafe { pyre_object::w_str_len(line) == 0 } {
            let this = Self::from_slot(slot);
            this.telling = this.seekable_flag;
            Err(crate::PyError::stop_iteration())
        } else {
            Ok(line)
        }
    }

    fn __repr__(&self) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = self.pin_self();
        Self::from_slot(slot).check_init()?;
        let mut self_obj = Self::from_slot(slot).self_obj();
        let Some(_guard) = crate::display::ReprGuard::enter(self_obj) else {
            return Err(crate::PyError::runtime_error(
                "reentrant call inside TextIOWrapper.__repr__",
            ));
        };

        let typename = crate::type_methods::arg_type_name(self_obj);
        let mut fields = rustpython_wtf8::Wtf8Buf::new();
        if Self::from_slot(slot).state != STATE_DETACHED {
            let buffer_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_buffer);
            if let Ok(mut name) = pyre_object::with_roots!(self_obj => crate::baseobjspace::getattr_str(
                pyre_object::gc_roots::shadow_stack_get(buffer_slot),
                "name",
            )) {
                fields.push_str(" name=");
                fields.push_wtf8(&unsafe {
                    pyre_object::with_roots!(self_obj, name => crate::display::py_repr_wtf8(name))?
                });
            }
        }
        if let Ok(mut mode) =
            pyre_object::with_roots!(self_obj => crate::baseobjspace::getattr_str(self_obj, "mode"))
        {
            fields.push_str(" mode=");
            fields.push_wtf8(&unsafe {
                pyre_object::with_roots!(self_obj, mode => crate::display::py_repr_wtf8(mode))?
            });
        }
        fields.push_str(" encoding=");
        let encoding_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(Self::from_slot(slot).w_encoding);
        fields.push_wtf8(&unsafe {
            crate::display::py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(encoding_slot))?
        });
        Ok(crate::display::wtf8_format!(
            format!("<{typename}"),
            fields,
            ">"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_cookie_round_trips_all_native_words() {
        let original = PositionCookie {
            start_pos: 0xfedc_ba98_7654_3210,
            dec_flags: 0x0123_4567_89ab_cdef,
            bytes_to_feed: 0x8877_6655_4433_2211,
            chars_to_skip: 0x1020_3040_5060_7080,
            need_eof: true,
        };
        let packed = original.pack();
        assert!(
            packed.bits() > u64::BITS as u64,
            "a packed cookie must not be narrowed through to_u64"
        );

        let decoded = PositionCookie::unpack(packed).expect("packed cookie");
        assert_eq!(decoded.start_pos, original.start_pos);
        assert_eq!(decoded.dec_flags, original.dec_flags);
        assert_eq!(decoded.bytes_to_feed, original.bytes_to_feed);
        assert_eq!(decoded.chars_to_skip, original.chars_to_skip);
        assert_eq!(decoded.need_eof, original.need_eof);
    }
}
