//! `Cpu` trait impl for pyre's runtime string layout.
//!
//! `llmodel.py:557 gc_ll_descr.str_descr / unicode_descr` parity — the
//! typed `ArrayDescr` that backend init caches and that the speculative
//! protect / length read / per-character read all route through.
//! `model.py:209+` for the read-family. PyPy stores
//! `gc_ll_descr.str_descr` once at backend init; pyre exposes it via
//! the `Cpu` trait so `protect_speculative_string`, `bh_strlen` and
//! `bh_strgetitem` all reach the same descr.
//!
//! `str_descr` is rstr `STR` (`{hash, len, chars}`, `rstr.py`) — the
//! `_utf8` payload `descr_add` concatenates. `unicode_descr` stays the
//! `W_UnicodeObject` wrapper for `UNICODELEN` / `UNICODEGETITEM` until
//! those ops are retargeted the same way.

use std::sync::{Arc, OnceLock};

use majit_ir::{ArrayDescr, Descr, FieldDescr, GcRef, Type};
use majit_metainterp::cpu::{Cpu, DefaultCpu, SpeculativeError};
use pyre_object::lowlevel_string::{
    LOWLEVEL_STR_BASE_SIZE, LOWLEVEL_STRING_LEN_OFFSET, bh_lowlevel_string_len,
    lowlevel_str_gc_type_id,
};
use pyre_object::rutf8::Utf8IndexStorage;
use pyre_object::unicodeobject::utf8_payload_bytes;
use pyre_object::unicodeobject::{
    UNICODE_BYTE_LEN_OFFSET, UNICODE_INDEX_STORAGE_OFFSET, UNICODE_LEN_OFFSET,
    UNICODE_VALUE_OFFSET, W_UNICODE_GC_TYPE_ID, W_UNICODE_OBJECT_SIZE,
};

/// FieldDescr for rstr `STR.length` — `rstr.py` varsize `len` word.
/// `llmodel.py bh_strlen` reads `len(s.chars)`.
#[derive(Debug)]
struct PyreStrLenFieldDescr;

impl Descr for PyreStrLenFieldDescr {}

impl FieldDescr for PyreStrLenFieldDescr {
    fn offset(&self) -> usize {
        LOWLEVEL_STRING_LEN_OFFSET
    }
    fn field_size(&self) -> usize {
        std::mem::size_of::<usize>()
    }
    fn field_type(&self) -> Type {
        Type::Int
    }
    fn is_field_signed(&self) -> bool {
        true
    }
    fn field_name(&self) -> &'static str {
        "rstr.STR.length"
    }
}

/// FieldDescr for `W_UnicodeObject.len` — codepoint count.
/// RPython UNICODE uses codepoint-indexed arrays;
/// `bh_unicodelen` reads codepoint count.
#[derive(Debug)]
struct PyreUnicodeLenFieldDescr;

impl Descr for PyreUnicodeLenFieldDescr {}

impl FieldDescr for PyreUnicodeLenFieldDescr {
    fn offset(&self) -> usize {
        UNICODE_LEN_OFFSET
    }
    fn field_size(&self) -> usize {
        // `W_UnicodeObject.len` is a `usize`: 8 bytes on 64-bit, 4 on wasm32.
        // A hardcoded 8 reads the adjacent field into the high half on a
        // 32-bit target.
        std::mem::size_of::<usize>()
    }
    fn field_type(&self) -> Type {
        Type::Int
    }
    fn is_field_signed(&self) -> bool {
        true
    }
    fn field_name(&self) -> &'static str {
        "W_UnicodeObject.len"
    }
}

/// ArrayDescr for rstr `STR` (`rstr.py GcStruct('rpy_string', …)`).
/// `len_descr` → the varsize `len` word at [`LOWLEVEL_STRING_LEN_OFFSET`].
#[derive(Debug)]
struct PyreStrDescr;

/// ArrayDescr for UNICODE (codepoint string).
/// `len_descr` → `len` (codepoint count) field.
#[derive(Debug)]
struct PyreUnicodeDescr;

const PYRE_STR_LEN_DESCR: PyreStrLenFieldDescr = PyreStrLenFieldDescr;
const PYRE_UNICODE_LEN_DESCR: PyreUnicodeLenFieldDescr = PyreUnicodeLenFieldDescr;
const PYRE_STR_DESCR: PyreStrDescr = PyreStrDescr;
const PYRE_UNICODE_DESCR: PyreUnicodeDescr = PyreUnicodeDescr;

impl Descr for PyreStrDescr {}

impl ArrayDescr for PyreStrDescr {
    fn base_size(&self) -> usize {
        LOWLEVEL_STR_BASE_SIZE
    }
    fn item_size(&self) -> usize {
        1
    }
    fn type_id(&self) -> u32 {
        lowlevel_str_gc_type_id()
    }
    fn cache_key(&self) -> u64 {
        majit_ir::descr::array_identity_hash("rpy_string")
    }
    fn item_type(&self) -> Type {
        Type::Int
    }
    fn is_item_signed(&self) -> bool {
        false
    }
    fn len_descr(&self) -> Option<&dyn FieldDescr> {
        Some(&PYRE_STR_LEN_DESCR)
    }
}

impl Descr for PyreUnicodeDescr {}

impl ArrayDescr for PyreUnicodeDescr {
    fn base_size(&self) -> usize {
        W_UNICODE_OBJECT_SIZE
    }
    fn item_size(&self) -> usize {
        4
    }
    fn type_id(&self) -> u32 {
        W_UNICODE_GC_TYPE_ID as u32
    }
    fn cache_key(&self) -> u64 {
        majit_ir::descr::array_identity_hash("rpy_unicode")
    }
    fn item_type(&self) -> Type {
        Type::Int
    }
    fn is_item_signed(&self) -> bool {
        false
    }
    fn len_descr(&self) -> Option<&dyn FieldDescr> {
        Some(&PYRE_UNICODE_LEN_DESCR)
    }
}

/// `Cpu` impl for pyre's runtime.  Delegates to `DefaultCpu` for the
/// methods `DefaultCpu` overrides (`cls_of_box` / `bh_classof` /
/// `bh_getfield_gc_{i,r,f}`) and exposes pyre-specific descrs for the
/// str / unicode family.  `bh_strgetitem` / `bh_unicodegetitem` follow
/// the `W_UnicodeObject.value: *mut String` indirection that PyPy's STR
/// layout does not need (PyPy stores chars in-line after the header).
pub struct PyreCpu(DefaultCpu);

impl PyreCpu {
    pub fn new() -> Self {
        publish_str_unicode_array_identities();
        Self(DefaultCpu)
    }
}

/// `cpu.arraydescrof(STR)` / `UNICODE`: publish the descrs into
/// `_cache_array` under their `array_identity_hash` so EffectInfo
/// members resolve (`descr.py get_array_descr`).
fn publish_str_unicode_array_identities() {
    use majit_ir::descr::{LLType, gc_cache};
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut gc = gc_cache().lock();
        gc.register_keyed_array(
            LLType::Array(majit_ir::descr::array_identity_hash("rpy_string")),
            Arc::new(PyreStrDescr),
        );
        gc.register_keyed_array(
            LLType::Array(majit_ir::descr::array_identity_hash("rpy_unicode")),
            Arc::new(PyreUnicodeDescr),
        );
    });
}

impl Default for PyreCpu {
    fn default() -> Self {
        Self::new()
    }
}

/// The offset a blackhole item read takes, refused unless it is one: `as
/// usize` wraps a negative operand and truncates one wider than the target's
/// `usize`, and either turns the bounds test that follows into a read of the
/// wrong element.
fn item_index(index: i64) -> Option<usize> {
    usize::try_from(index).ok()
}

impl Cpu for PyreCpu {
    fn bh_classof(&self, gcref: GcRef) -> i64 {
        self.0.bh_classof(gcref)
    }
    fn bh_getfield_gc_i(&self, struct_ptr: usize, fd: &dyn FieldDescr) -> i64 {
        self.0.bh_getfield_gc_i(struct_ptr, fd)
    }
    fn bh_getfield_gc_r(&self, struct_ptr: usize, fd: &dyn FieldDescr) -> GcRef {
        self.0.bh_getfield_gc_r(struct_ptr, fd)
    }
    fn bh_getfield_gc_f(&self, struct_ptr: usize, fd: &dyn FieldDescr) -> f64 {
        self.0.bh_getfield_gc_f(struct_ptr, fd)
    }

    fn str_descr(&self) -> Option<&dyn ArrayDescr> {
        publish_str_unicode_array_identities();
        Some(&PYRE_STR_DESCR)
    }
    fn unicode_descr(&self) -> Option<&dyn ArrayDescr> {
        publish_str_unicode_array_identities();
        Some(&PYRE_UNICODE_DESCR)
    }

    fn protect_speculative_string(&self, gcptr: GcRef) -> Result<(), SpeculativeError> {
        // `llmodel.py protect_speculative_string` → `protect_speculative_array`
        // with `gc_ll_descr.str_descr`.  A GC object must carry the STR tid:
        // a `W_UnicodeObject` wrapper is not a payload, and `bh_strlen`
        // would read its header as `len`.
        //
        // Immortal interned `_utf8` is a raw STR (`unicodeobject.rs`
        // `alloc_utf8_payload`: an immortal wrapper must not grey a young
        // GC box).  Those have no tid; reject only when the pointer is
        // itself a Python `str`.
        if gcptr.is_null() {
            return Err(SpeculativeError);
        }
        if !majit_gc::supports_guard_gc_type() {
            return Ok(());
        }
        let want = lowlevel_str_gc_type_id();
        if let Some(actual) = majit_gc::get_actual_typeid(gcptr) {
            if want == 0 || actual != want {
                return Err(SpeculativeError);
            }
            return Ok(());
        }
        // No GC header.  An exact `str` wrapper stores `ob_type` at
        // offset 0 (`STR_TYPE`).  A raw STR stores `hash` there (0 or
        // a siphash) — do not dereference it as a type.
        let first = unsafe { *(gcptr.0 as *const usize) };
        if first == std::ptr::from_ref(&pyre_object::STR_TYPE).addr() {
            return Err(SpeculativeError);
        }
        Ok(())
    }

    fn bh_strlen(&self, string: GcRef) -> Option<i64> {
        // `llmodel.py bh_strlen`: `len(s.chars)` on rstr `STR`.
        if string.is_null() {
            return None;
        }
        Some(bh_lowlevel_string_len(string.0 as i64) as i64)
    }

    fn bh_strgetitem(&self, string: GcRef, index: i64) -> Option<i64> {
        // `llmodel.py bh_strgetitem`: `ord(s.chars[index])` on rstr `STR`.
        if string.is_null() {
            return None;
        }
        let bytes = unsafe {
            utf8_payload_bytes(string.0 as *const pyre_object::unicodeobject::UnicodeValueStorage)
        };
        let i = item_index(index)?;
        bytes.get(i).map(|&b| b as i64)
    }

    fn bh_unicodegetitem(&self, unicode: GcRef, index: i64) -> Option<i64> {
        // RPython UNICODE is codepoint-indexed; UNICODEGETITEM returns the
        // codepoint value, `to_u32` (including lone surrogates D800-DFFF).
        // Pyre's `W_UnicodeObject` stores WTF-8, where a codepoint index is a
        // byte offset only for an ASCII payload, so this resolves it the two
        // ways `w_str_codepoint_at` does.
        //
        // Neither arm builds the index table: a blackhole runs inside a deopt,
        // so it reads a table that is already there and otherwise walks. The
        // walk is the cost the table exists to remove, and it is the one
        // upstream's array read never pays.
        if unicode.is_null() {
            return None;
        }
        let value_addr = unicode.0 + UNICODE_VALUE_OFFSET;
        let value_ptr = unsafe {
            *(value_addr as *const *const pyre_object::unicodeobject::UnicodeValueStorage)
        };
        if value_ptr.is_null() {
            return None;
        }
        let s = unsafe { pyre_object::unicodeobject::utf8_payload_wtf8(value_ptr) };
        let i = item_index(index)?;
        let len = unsafe { *((unicode.0 + UNICODE_LEN_OFFSET) as *const usize) };
        if i >= len {
            return None;
        }
        let byte_len = unsafe { *((unicode.0 + UNICODE_BYTE_LEN_OFFSET) as *const usize) };
        // `w_str_is_ascii` — one byte per codepoint, so the index is the offset.
        if len == byte_len {
            return Some(s.as_bytes()[i] as i64);
        }
        let storage = unsafe {
            *((unicode.0 + UNICODE_INDEX_STORAGE_OFFSET) as *const *const Utf8IndexStorage)
        };
        if storage.is_null() {
            return s.code_points().nth(i).map(|c| c.to_u32() as i64);
        }
        Some(pyre_object::rutf8::codepoint_at_index(
            s,
            unsafe { &*storage },
            i,
        ))
    }
}

/// Shared `Arc<dyn Cpu>` for pyre.  Initialised once per process and
/// installed on `MetaInterp<PyreMeta>` via `set_cpu` at the
/// `trace_bytecode` entry point.
pub fn shared() -> Arc<dyn Cpu> {
    static CELL: OnceLock<Arc<dyn Cpu>> = OnceLock::new();
    CELL.get_or_init(|| Arc::new(PyreCpu::new()) as Arc<dyn Cpu>)
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_wtf8::CodePoint;

    /// Every arm of `bh_unicodegetitem` must answer what a codepoint walk
    /// answers. The arms are chosen by two facts about the operand — whether
    /// it is ASCII, and whether its index table has been built — so each case
    /// below puts a string in one of those states and compares the whole
    /// index range against the walk.
    fn agrees_with_walk(obj: pyre_object::PyObjectRef) {
        let cpu = PyreCpu::new();
        let gc = GcRef(obj as usize);
        let walk: Vec<i64> = unsafe { pyre_object::w_str_get_wtf8(obj) }
            .code_points()
            .map(|c| c.to_u32() as i64)
            .collect();
        for (i, expected) in walk.iter().enumerate() {
            assert_eq!(
                cpu.bh_unicodegetitem(gc, i as i64),
                Some(*expected),
                "index {i}"
            );
        }
        assert_eq!(
            cpu.bh_unicodegetitem(gc, walk.len() as i64),
            None,
            "one past the end"
        );
        assert_eq!(cpu.bh_unicodegetitem(gc, -1), None, "negative index");
    }

    /// The offset conversion refuses what it cannot represent, rather than
    /// wrapping into range. A wrapped index passes the bounds test that
    /// follows it and reads some other element, which is a wrong answer where
    /// the refusal is a `None` the caller already handles.
    #[test]
    fn an_index_that_does_not_fit_the_targets_usize_is_refused() {
        assert_eq!(item_index(0), Some(0));
        assert_eq!(item_index(7), Some(7));
        assert_eq!(item_index(-1), None);
        assert_eq!(item_index(i64::MIN), None);
        // `as usize` truncates this to 0 where `usize` is 32 bits, which is
        // the wasm32 target.
        const PAST_U32: i64 = 1 << 32;
        #[cfg(target_pointer_width = "32")]
        assert_eq!(item_index(PAST_U32), None);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(item_index(PAST_U32), Some(PAST_U32 as usize));
    }

    #[test]
    fn bh_strlen_reads_the_str_payload() {
        let obj = pyre_object::w_str_new("hello");
        let payload = unsafe { pyre_object::unicodeobject::w_str_storage(obj) };
        let cpu = PyreCpu::new();
        assert_eq!(cpu.bh_strlen(GcRef(payload as usize)), Some(5));
        assert_eq!(
            cpu.bh_strgetitem(GcRef(payload as usize), 1),
            Some(b'e' as i64)
        );
        assert!(
            cpu.protect_speculative_string(GcRef(payload as usize))
                .is_ok()
        );
    }

    #[test]
    fn bh_unicodegetitem_ascii_reads_the_byte() {
        let obj = pyre_object::w_str_new("hello");
        assert!(unsafe { pyre_object::unicodeobject::w_str_is_ascii(obj) });
        agrees_with_walk(obj);
    }

    #[test]
    fn bh_unicodegetitem_wide_without_a_table_walks() {
        let obj = pyre_object::w_str_new("héllo wörld ☃");
        assert!(!unsafe { pyre_object::unicodeobject::w_str_is_ascii(obj) });
        agrees_with_walk(obj);
    }

    #[test]
    fn bh_unicodegetitem_wide_with_a_table_reads_the_table() {
        // Long enough to span more than one 64-codepoint group, so the read
        // exercises `baseindex` selection rather than only the first entry.
        let obj = pyre_object::w_str_new(&"ábç".repeat(60));
        // Force the lazy build the blackhole arm refuses to do itself.
        assert!(unsafe { pyre_object::w_str_codepoint_at(obj, 100) }.is_some());
        agrees_with_walk(obj);
    }

    #[test]
    fn bh_unicodegetitem_yields_a_lone_surrogate() {
        let mut buf = rustpython_wtf8::Wtf8Buf::new();
        buf.push(CodePoint::from_char('a'));
        buf.push(CodePoint::from_u32(0xD800).unwrap());
        buf.push(CodePoint::from_char('b'));
        let obj = pyre_object::unicodeobject::w_str_from_wtf8(buf);
        let cpu = PyreCpu::new();
        assert_eq!(cpu.bh_unicodegetitem(GcRef(obj as usize), 1), Some(0xD800));
        agrees_with_walk(obj);
    }
}
