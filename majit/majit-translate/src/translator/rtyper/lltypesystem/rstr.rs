//! RPython `rpython/rtyper/lltypesystem/rstr.py` — minimal slice for
//! the `Ptr(STR)` / `Ptr(UNICODE)` low-level types that
//! `OBJECT_VTABLE.name` (rclass.py) and
//! `ClassesPBCRepr.rtype_getattr('__name__')` (rpbc.py)
//! consume, plus the `UNICODE` data shape that the eventual
//! `UnicodeRepr` port will dereference.
//!
//! Upstream rstr.py weighs ~3000 LOC and houses every operation that
//! rtypes Python's `str` / `unicode` types (`StringRepr`,
//! `UnicodeRepr`, `LLHelpers`, `ll_strconcat`, ...). Pyre lands only
//! the data shape this commit needs:
//!
//! ```python
//! STR = GcForwardReference()
//! UNICODE = GcForwardReference()
//! ...
//! STR.become(GcStruct('rpy_string',
//!                     ('hash', Signed),
//!                     ('chars', Array(Char, hints={'immutable': True,
//!                                     'extra_item_after_alloc': 1})),
//!                     adtmeths={...},
//!                     hints={'remove_hash': True}))
//! UNICODE.become(GcStruct('rpy_unicode',
//!                         ('hash', Signed),
//!                         ('chars', Array(UniChar, hints={'immutable': True})),
//!                         adtmeths={...},
//!                         hints={'remove_hash': True}))
//! ```
//!
//! The full repr port (`AbstractStringRepr`/`AbstractUnicodeRepr`
//! methods, `LLHelpers.ll_*` helpers, comparison, formatting,
//! adtmeths) is deferred to follow-on work.
//!
//! ## Why a separate module today
//!
//! Both `OBJECT_VTABLE.name` (rclass.py) and
//! `ClassesPBCRepr.rtype_getattr('__name__')` (rpbc.py) want
//! `Ptr(STR)`. Defining `STR` / `UNICODE` here keeps the dependency
//! direction `rclass / rpbc → rstr → lltype` clean and gives a
//! natural home for the eventual full `StringRepr` / `UnicodeRepr`
//! port.

#![allow(non_upper_case_globals)]

use std::sync::LazyLock;

#[cfg(test)]
use crate::annotator::annrpython::RPythonAnnotator;
use crate::flowspace::model::{
    Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, GraphKey,
    GraphRef, Hlvalue, Link, SpaceOperation, Variable,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    _func, _ptr, _ptr_obj, Array, ForwardReference, FuncType, GcKind, LowLevelType, LowLevelValue,
    MallocFlavor, Ptr, PtrTarget, Struct, functionptr, malloc,
};
use crate::translator::rtyper::pairtype::ReprClassId;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, HighLevelOp, RPythonTyper, constant_with_lltype, exception_args,
    helper_pygraph_from_graph, variable_with_lltype, void_field_const,
};

/// RPython `STR = GcForwardReference()` resolved via
/// `STR.become(GcStruct('rpy_string', ('hash', Signed), ('chars',
/// Array(Char, ...)), hints={'remove_hash': True}))`
/// (rstr.py:32 + rstr.py:1226-1237).
///
/// Pyre lands the data shape — the `adtmeths` table (mallocstr,
/// emptystrfun, copy_string_contents, ll_strhash, ll_length, ll_find,
/// ll_rfind) is deferred. `OBJECT_VTABLE.name`'s `Ptr(STR)` only
/// needs the structural shape today; concrete consumers that emit
/// `getfield` / dereferences will lower through generic `_ptr.getattr`
/// against the resolved Struct.
pub static STR: LazyLock<LowLevelType> = LazyLock::new(|| {
    // `Array(Char, hints={'immutable': True, 'extra_item_after_alloc':
    // 1})` (rstr.py:1227-1228). `extra_item_after_alloc` reserves the
    // trailing NUL slot that `sizeof` / `get_array_token` add on top of
    // the requested length; `immutable` marks the char array read-only.
    let chars = Array::with_hints(
        LowLevelType::Char,
        vec![
            (
                "immutable".into(),
                crate::flowspace::model::ConstValue::Bool(true),
            ),
            (
                "extra_item_after_alloc".into(),
                crate::flowspace::model::ConstValue::Int(1),
            ),
        ],
    );
    let body = Struct::gc_with_hints(
        "rpy_string",
        vec![
            ("hash".into(), LowLevelType::Signed),
            ("chars".into(), LowLevelType::Array(Box::new(chars))),
        ],
        vec![(
            "remove_hash".into(),
            crate::flowspace::model::ConstValue::Bool(true),
        )],
    );
    let fwd = ForwardReference::gc();
    fwd.r#become(LowLevelType::Struct(Box::new(body)))
        .expect("STR.become should succeed");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `Ptr(STR)` — surfaces as the lltype of
/// `OBJECT_VTABLE.name` (rclass.py) and the result type of
/// `genop('getfield', [vcls, 'name'], resulttype=Ptr(rstr.STR))` in
/// `ClassesPBCRepr.rtype_getattr('__name__')` (rpbc.py).
pub static STRPTR: LazyLock<LowLevelType> = LazyLock::new(|| {
    let str_t = STR.clone();
    let LowLevelType::ForwardReference(fwd) = str_t else {
        panic!("STR must be a ForwardReference");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::ForwardReference(*fwd),
    }))
});

fn rstr_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!(
        "lltypesystem.rstr.{name} string helper surface deferred"
    ))
}

/// RPython `new_malloc(TP, name)`.
pub fn new_malloc() -> Result<(), TyperError> {
    Err(rstr_deferred("new_malloc"))
}

/// RPython `llstr(s)` (`annlowlevel.py` via
/// `rstr.py mallocstr`): build the prebuilt `Ptr(STR)`
/// container for a host string constant — `mallocstr(len(s))` plus a
/// per-index `chars[i] = s[i]` fill.  `hash` keeps malloc's `Signed`
/// default `0` (upstream computes it lazily through `ll_strhash`).
/// The container is immortal: prebuilt string constants live for the
/// whole translation, exactly like upstream's `CONST_STR_CACHE`
/// entries.
pub(crate) fn llstr(s: &[u8]) -> Result<_ptr, String> {
    let LowLevelType::ForwardReference(fwd) = STR.clone() else {
        return Err("STR must be a ForwardReference".to_string());
    };
    let str_struct = fwd
        .resolved()
        .ok_or_else(|| "STR forward reference unresolved".to_string())?;
    let p = malloc(str_struct, Some(s.len()), MallocFlavor::Gc, true)?;
    let obj = p
        ._obj0_value()
        .map_err(|_| "llstr: malloc returned a delayed pointer".to_string())?
        .ok_or_else(|| "llstr: malloc returned a null pointer".to_string())?;
    let _ptr_obj::Struct(st) = obj else {
        return Err("llstr: malloc(STR) must produce a struct container".to_string());
    };
    let fields = st._fields.lock();
    let Some((_, LowLevelValue::Array(chars))) = fields.iter().find(|(n, _)| n == "chars") else {
        return Err("llstr: rpy_string container lacks a chars array".to_string());
    };
    for (i, b) in s.iter().enumerate() {
        // Py2 `str` chars are bytes; `u8 as char` is the Latin-1
        // embedding, which round-trips every byte value.
        if !chars.setitem(i, LowLevelValue::Char(*b as char)) {
            return Err(format!("llstr: chars[{i}] write out of bounds"));
        }
    }
    drop(fields);
    Ok(p)
}

/// `_hash_string(s)` (objectmodel.py): the modified-FNV hash
/// behind `compute_hash()` — `x = ord(s[0]) << 7`, then
/// `x = intmask((1000003*x) ^ ord(s[i]))` per char, `x ^= length`,
/// empty string → `-1`.  `intmask` is machine-word wraparound, hence
/// `wrapping_mul` on the `Signed` width.  Byte chars hash by their
/// Latin-1 ord, matching the `Char` element embedding in [`llstr`].
pub(crate) fn hash_string_bytes(s: &[u8]) -> i64 {
    let Some(&first) = s.first() else {
        return -1;
    };
    let mut x = (first as i64) << 7;
    for &c in s {
        x = 1000003i64.wrapping_mul(x) ^ c as i64;
    }
    x ^ s.len() as i64
}

/// `_ll_strhash(s)` value computation (rstr.py): malloc
/// zero-fills the `hash` slot, so `0` is reserved as the
/// not-computed-yet marker — a real hash of `0` is replaced with
/// `29872897` (the same fixup the runtime helper graph
/// `build_ll_strhash_internal_helper_graph` applies).
pub(crate) fn ll_strhash_value(s: &[u8]) -> i64 {
    let x = hash_string_bytes(s);
    if x == 0 { 29872897 } else { x }
}

thread_local! {
    /// `CONST_STR_CACHE = WeakValueDictionary()` (`rstr.py` /
    /// `StringRepr.CACHE`, `lltypesystem/rstr.py`): one host string →
    /// one prebuilt container identity, so repeated `convert_const` calls
    /// on the same literal return the same `_ptr` (prebuilt-constant
    /// `is`-identity upstream relies on).  Thread-lifetime strong
    /// cache; upstream's weak semantics only matter for memory, not
    /// identity.  Upstream is module-global because one translation owns
    /// the process; translation sessions here are single-threaded, so
    /// thread_local confines the cache to one session at a time (the
    /// same convention as `classdesc::FORCE_ATTRIBUTES_INTO_CLASSES`) —
    /// cached `_ptr`s carry a mutable `hash` slot written back by
    /// `build_ll_strhash_internal_helper_graph`, which must not be shared
    /// across concurrently-running sessions.  Sequential sessions on one
    /// thread DO share entries, which is sound: the value is determined
    /// by the key bytes alone ([`llstr`] container + [`ll_strhash_value`]
    /// fill), and the only mutation is that idempotent hash fill.
    static CONST_STR_CACHE: std::cell::RefCell<std::collections::HashMap<Vec<u8>, _ptr>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// `self.CACHE.get(value)` / `self.ll.llstr(value)` pair from
/// `BaseLLStringRepr.convert_const` (`lltypesystem/rstr.py`).
pub(crate) fn const_str_cache_llstr(s: &[u8]) -> Result<_ptr, String> {
    CONST_STR_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(p) = cache.get(s) {
            return Ok(p.clone());
        }
        let mut p = llstr(s)?;
        // `p.hash = 0; self.ll.ll_strhash(p)` (rstr.py):
        // precompute the hash before caching the container, so every
        // prebuilt string constant ships with its hash slot filled.
        p.setattr("hash", LowLevelValue::Signed(ll_strhash_value(s)))?;
        cache.insert(s.to_vec(), p.clone());
        Ok(p)
    })
}

/// RPython `conststr(s)`.
pub fn conststr(s: &str) -> Result<_ptr, String> {
    const_str_cache_llstr(s.as_bytes())
}

/// Inverse of [`llstr`] / [`const_str_cache_llstr`]: read back the
/// Latin-1 byte payload and `hash`-slot value of a prebuilt `Ptr(STR)`
/// container.  The codewriter uses this to lower a prebuilt-string
/// `ConstValue::LLPtr(STR)` into a runtime-materialized descriptor: the
/// build-time `_ptr` is process-local, so only its bytes + precomputed
/// hash survive into the serialized jitcode.
///
/// Returns `None` for null, a non-struct target, a missing/empty-typed
/// `chars` field, or a non-`Char` element (a `Ptr(UNICODE)` container,
/// whose `chars` are `UniChar`, is out of scope here — Unicode prebuilt
/// constants are a separate lowering).
pub(crate) fn prebuilt_str_bytes_and_hash(p: &_ptr) -> Option<(Vec<u8>, i64)> {
    if !p.nonzero() {
        return None;
    }
    let _ptr_obj::Struct(st) = p._obj0_value().ok().flatten()? else {
        return None;
    };
    let fields = st._fields.lock();
    // `chars` mirrors the read side of `llstr`'s per-index fill.
    let Some((_, LowLevelValue::Array(chars))) = fields.iter().find(|(n, _)| n == "chars") else {
        return None;
    };
    let mut bytes = Vec::with_capacity(chars.getlength());
    for i in 0..chars.getlength() {
        match chars.getitem(i) {
            // `u8 as char` is the Latin-1 embedding `llstr` writes, so the
            // low byte round-trips every value 0..=255.
            Some(LowLevelValue::Char(c)) => bytes.push(c as u32 as u8),
            _ => return None,
        }
    }
    // `const_str_cache_llstr` precomputes and stores `hash` (rstr.py:199);
    // a bare `llstr` leaves the malloc-zeroed `0` (lazy not-computed).
    let hash = match fields.iter().find(|(n, _)| n == "hash") {
        Some((_, LowLevelValue::Signed(h))) => *h,
        _ => ll_strhash_value(&bytes),
    };
    Some((bytes, hash))
}

/// `nullptr(self.lowleveltype.TO)` for the `value is None` arm of
/// `BaseLLStringRepr.convert_const` (`lltypesystem/rstr.py`).
pub(crate) fn null_str_ptr() -> _ptr {
    let LowLevelType::Ptr(ptr_t) = STRPTR.clone() else {
        panic!("STRPTR must be a Ptr lowleveltype");
    };
    _ptr::new(*ptr_t, Ok(None))
}

/// RPython `UNICODE = GcForwardReference()` resolved via
/// `UNICODE.become(GcStruct('rpy_unicode', ('hash', Signed), ('chars',
/// Array(UniChar, hints={'immutable': True})), hints={'remove_hash':
/// True}))` (rstr.py:33 + rstr.py:1238-1246).
///
/// Mirror of [`STR`] with the `chars` element type swapped from
/// `Char` to `UniChar`. The `extra_item_after_alloc` hint that STR
/// carries (for the trailing NUL slot) is omitted upstream for
/// UNICODE — only `immutable` is set on the array. The Rust port
/// elides both today (see [`STR`] docstring); structural shape is
/// what the eventual `UnicodeRepr` port consumes.
pub static UNICODE: LazyLock<LowLevelType> = LazyLock::new(|| {
    // `Array(UniChar, hints={'immutable': True})` (rstr.py) —
    // marks the unichar array read-only; no trailing-NUL reservation.
    let chars = Array::with_hints(
        LowLevelType::UniChar,
        vec![(
            "immutable".into(),
            crate::flowspace::model::ConstValue::Bool(true),
        )],
    );
    let body = Struct::gc_with_hints(
        "rpy_unicode",
        vec![
            ("hash".into(), LowLevelType::Signed),
            ("chars".into(), LowLevelType::Array(Box::new(chars))),
        ],
        vec![(
            "remove_hash".into(),
            crate::flowspace::model::ConstValue::Bool(true),
        )],
    );
    let fwd = ForwardReference::gc();
    fwd.r#become(LowLevelType::Struct(Box::new(body)))
        .expect("UNICODE.become should succeed");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `Ptr(UNICODE)` — surfaces as the lltype of
/// `UnicodeRepr.lowleveltype` (rstr.py) and is the result type of
/// every `gendirectcall(self.ll.ll_*unicode*, ...)` inside
/// `AbstractUnicodeRepr` methods.
pub static UNICODEPTR: LazyLock<LowLevelType> = LazyLock::new(|| {
    let unicode_t = UNICODE.clone();
    let LowLevelType::ForwardReference(fwd) = unicode_t else {
        panic!("UNICODE must be a ForwardReference");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::ForwardReference(*fwd),
    }))
});

thread_local! {
    /// RPython `CONST_UNICODE_CACHE = WeakValueDictionary()`.
    pub static CONST_UNICODE_CACHE: std::cell::RefCell<std::collections::HashMap<String, _ptr>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// RPython `emptyunicodefun()`.
pub fn emptyunicodefun() -> Result<_ptr, String> {
    alloc_array_unicode("")
}

/// RPython `emptystrfun()`.
pub fn emptystrfun() -> Result<_ptr, String> {
    const_str_cache_llstr(b"")
}

/// RPython `alloc_array_name(name)` (rclass.py):
///
/// ```python
/// def alloc_array_name(name):
///     return rstr.string_repr.convert_const(name)
/// ```
///
/// Delegates to the `convert_const` slice
/// ([`const_str_cache_llstr`]), so class-name STRs share the prebuilt
/// constant cache (same-name → same container identity) and the hash
/// precompute.  The adtmeth table stays in the future full rstr port.
pub(crate) fn alloc_array_name(name: &str) -> Result<_ptr, String> {
    const_str_cache_llstr(name.as_bytes())
}

/// `unicode_repr.convert_const(value)` mirror of [`alloc_array_name`]
/// for the `UNICODE` GcStruct. Materialises an immortal `_ptr` to a
/// `UNICODE` carrying the Unicode codepoint sequence of `value` and
/// initialises `hash` to 0. The full `UnicodeRepr.convert_const` and
/// hash precomputation stay in the future full rstr port.
fn alloc_array_unicode(value: &str) -> Result<_ptr, String> {
    let unicode_body = match UNICODE.clone() {
        LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
            "alloc_array_unicode: UNICODE ForwardReference is not resolved".to_string()
        })?,
        other => other,
    };
    let codepoint_count = value.chars().count();
    let mut ptr = malloc(unicode_body, Some(codepoint_count), MallocFlavor::Gc, true)?;
    ptr.setattr("hash", LowLevelValue::Signed(0))?;
    let Some(obj) = ptr._obj0_value().map_err(|_| {
        "alloc_array_unicode: delayed UNICODE pointer cannot be initialised".to_string()
    })?
    else {
        return Err("alloc_array_unicode: malloc returned null UNICODE pointer".to_string());
    };
    let _ptr_obj::Struct(s) = obj else {
        return Err("alloc_array_unicode: UNICODE pointer does not target a Struct".to_string());
    };
    let chars_value = s
        ._getattr("chars")
        .ok_or_else(|| "alloc_array_unicode: UNICODE struct has no chars field".to_string())?;
    let LowLevelValue::Array(chars) = chars_value else {
        return Err("alloc_array_unicode: UNICODE struct chars field is not an Array".to_string());
    };
    for (i, ch) in value.chars().enumerate() {
        chars.setitem(i, LowLevelValue::UniChar(ch));
    }
    Ok(ptr)
}

/// RPython `_new_copy_contents_fun(...)`.
fn _new_copy_contents_fun() -> Result<(), TyperError> {
    Err(rstr_deferred("_new_copy_contents_fun"))
}

/// RPython `TEMP_UNICODE = GcArray(Ptr(UNICODE))`.
pub static TEMP_UNICODE: LazyLock<LowLevelType> =
    LazyLock::new(|| LowLevelType::Array(Box::new(Array::gc(UNICODEPTR.clone()))));

/// RPython `ll_join = LLHelpers.ll_join`.
pub fn ll_join() -> Result<(), TyperError> {
    Err(rstr_deferred("ll_join"))
}

/// RPython `do_stringformat = LLHelpers.do_stringformat`.
pub fn do_stringformat() -> Result<(), TyperError> {
    Err(rstr_deferred("do_stringformat"))
}

/// RPython `class BaseLLStringRepr(Repr)` (`lltypesystem/rstr.py`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BaseLLStringRepr;

/// RPython `class StringRepr(BaseLLStringRepr, AbstractStringRepr)`.
pub type StringRepr = crate::translator::rtyper::rstr::AbstractStringRepr;

/// RPython `class UnicodeRepr(BaseLLStringRepr, AbstractUnicodeRepr)`.
pub type UnicodeRepr = crate::translator::rtyper::rstr::AbstractUnicodeRepr;

/// RPython `class CharRepr(AbstractCharRepr, StringRepr)`.
pub type CharRepr = crate::translator::rtyper::rstr::AbstractCharRepr;

/// RPython `class UniCharRepr(AbstractUniCharRepr, UnicodeRepr)`.
pub type UniCharRepr = crate::translator::rtyper::rstr::AbstractUniCharRepr;

/// RPython `class LLHelpers(AbstractLLHelpers)` (`lltypesystem/rstr.py`).
pub type LLHelpers = crate::translator::rtyper::rstr::AbstractLLHelpers;

/// RPython `class BaseStringIteratorRepr(AbstractStringIteratorRepr)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BaseStringIteratorRepr;

/// RPython `class StringIteratorRepr(BaseStringIteratorRepr)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StringIteratorRepr;

/// RPython `class UnicodeIteratorRepr(BaseStringIteratorRepr)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UnicodeIteratorRepr;

/// Translator-synthesised `Utf8StringIterator` (`rutf8.py`):
///
/// ```python
/// class Utf8StringIterator(object):
///     def __init__(self, utf8s):
///         self._utf8 = utf8s
///         self._end = len(utf8s)
///         self._pos = 0
///     def next(self):          # returns int code point; raises StopIteration
/// ```
///
/// `lowleveltype = Ptr(GcStruct('utf8iter', ('_utf8', Ptr(STR)),
/// ('_end', Signed), ('_pos', Signed)))`.  Distinct from
/// [`StringIteratorRepr`] (byte `ll_striter` / `ll_strnext`).
#[derive(Debug)]
pub struct Utf8StringIteratorRepr {
    state: ReprState,
    /// `Ptr(GcStruct('utf8iter', ('_utf8', STRPTR), ('_end', Signed),
    /// ('_pos', Signed)))`.
    lltype: LowLevelType,
    /// `r_str.lowleveltype` — the `_utf8` field type.
    str_lltype: LowLevelType,
}

impl Utf8StringIteratorRepr {
    /// Mint the utf8iter struct over a `Ptr(STR)` (or unicode) string.
    pub fn new(str_lltype: LowLevelType) -> Result<Self, TyperError> {
        let signed = LowLevelType::Signed;
        // Field names follow `Utf8StringIterator` (`_utf8`, `_end`, `_pos`).
        // `Struct::gc` rejects a leading `_` (`lltype.py` NameError);
        // `registry_fields` is the same exemption `__pos_0` uses.
        let st = Struct::registry_fields(
            "utf8iter",
            vec![
                ("_utf8".to_string(), str_lltype.clone()),
                ("_end".to_string(), signed.clone()),
                ("_pos".to_string(), signed),
            ],
            GcKind::Gc,
        );
        let lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(st),
        }));
        Ok(Utf8StringIteratorRepr {
            state: ReprState::new(),
            lltype,
            str_lltype,
        })
    }
}

impl Repr for Utf8StringIteratorRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "Utf8StringIteratorRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::Utf8StringIteratorRepr
    }

    /// `IteratorRepr.rtype_iter` (`rmodel.py`) — an iterator is its own
    /// iterator.
    fn rtype_iter(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        Ok(Some(vlist[0].clone()))
    }

    /// `ListIteratorRepr.newiter` shape (`rlist.py`): `gendirectcall` of
    /// the synthesised `ll_utf8iter` helper.
    fn newiter(&self, hop: &HighLevelOp) -> RTypeResult {
        let r_str = {
            let args_r = hop.args_r.borrow();
            args_r.first().and_then(|o| o.clone()).ok_or_else(|| {
                TyperError::message("Utf8StringIteratorRepr.newiter: arg0 repr missing")
            })?
        };
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_str.as_ref())])?;
        hop.exception_cannot_occur()?;
        let str_lltype = self.str_lltype.clone();
        let iter_lltype = self.lltype.clone();
        let str_for_builder = str_lltype.clone();
        let iter_for_builder = iter_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_utf8iter".to_string(),
            vec![str_lltype],
            iter_lltype,
            move |_rtyper, _args, _result| {
                build_ll_utf8iter_helper_graph(
                    "ll_utf8iter",
                    str_for_builder.clone(),
                    iter_for_builder.clone(),
                )
            },
        )?;
        hop.gendirectcall(&helper, vlist)
    }

    /// `ListIteratorRepr.rtype_next` shape (`rlist.py`):
    /// `has_implicit_exception(StopIteration)` / `exception_is_here()`
    /// then `gendirectcall` of `ll_utf8next`.
    fn rtype_next(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_iter = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        hop.has_implicit_exception("StopIteration");
        hop.exception_is_here()?;
        let iter_lltype = self.lltype.clone();
        let str_lltype = self.str_lltype.clone();
        let iter_for_builder = iter_lltype.clone();
        let str_for_builder = str_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_utf8next".to_string(),
            vec![iter_lltype],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_utf8next_helper_graph(
                    "ll_utf8next",
                    iter_for_builder.clone(),
                    str_for_builder.clone(),
                )
            },
        )?;
        hop.gendirectcall(&helper, v_iter)
    }
}

/// Synthesise `ll_utf8iter` — `Utf8StringIterator.__init__` (`rutf8.py`):
///
/// ```python
/// def __init__(self, utf8s):
///     self._utf8 = utf8s
///     self._end = len(utf8s)
///     self._pos = 0
/// ```
///
/// Template: `build_ll_listiter_helper_graph` (`rlist.py`).
pub(crate) fn build_ll_utf8iter_helper_graph(
    name: &str,
    str_lltype: LowLevelType,
    iter_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let s_arg = variable_with_lltype("s", str_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s_arg.clone())]);
    let return_var = variable_with_lltype("result", iter_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let LowLevelType::Ptr(ptr) = &iter_lltype else {
        return Err(TyperError::message(
            "build_ll_utf8iter_helper_graph: iter lltype is not Ptr",
        ));
    };
    let inner_struct = match &ptr.TO {
        PtrTarget::Struct(body) => body.clone(),
        other => {
            return Err(TyperError::message(format!(
                "build_ll_utf8iter_helper_graph: Ptr target must be Struct, got {other:?}"
            )));
        }
    };
    let c1 = Constant::with_concretetype(
        ConstValue::LowLevelType(Box::new(LowLevelType::Struct(Box::new(inner_struct)))),
        LowLevelType::Void,
    );
    let cflags = Constant::with_concretetype(ConstValue::byte_str("flavor=gc"), LowLevelType::Void);
    let v_iter = variable_with_lltype("iter", iter_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc",
        vec![Hlvalue::Constant(c1), Hlvalue::Constant(cflags)],
        Hlvalue::Variable(v_iter.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            void_field_const("_utf8"),
            Hlvalue::Variable(s_arg.clone()),
        ],
        Hlvalue::Variable(variable_with_lltype("v0", LowLevelType::Void)),
    ));
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&str_lltype)?;
    let v_end = emit_chars_length_ops(
        &startblock,
        Hlvalue::Variable(s_arg),
        chars_array_ptr_lltype,
    );
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            void_field_const("_end"),
            v_end,
        ],
        Hlvalue::Variable(variable_with_lltype("v1", LowLevelType::Void)),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            void_field_const("_pos"),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
        ],
        Hlvalue::Variable(variable_with_lltype("v2", LowLevelType::Void)),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_iter)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise `ll_utf8next` — inlined `Utf8StringIterator.next` (`rutf8.py`):
/// `@always_inline` (`objectmodel.always_inline` sets `_always_inline_ = True`).
/// `_pos == _end` raises StopIteration; else 1–4 byte decode with thresholds
/// 0x7F / 0xDF / 0xEF and the same arithmetic as
/// `pyre_object::rutf8::codepoint_at_pos` + `next_codepoint_pos`.
/// Byte loads are getsubstruct+getarrayitem+cast_char_to_int (the
/// `__string_byte_getitem` getitem+ord pair).
///
/// Template: `build_ll_listnext_helper_graph` / `build_ll_rangenext_helper_graph`.
pub(crate) fn build_ll_utf8next_helper_graph(
    name: &str,
    iter_lltype: LowLevelType,
    str_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&str_lltype)?;
    let iter_arg = variable_with_lltype("iter", iter_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(iter_arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let signed = || variable_with_lltype("v", LowLevelType::Signed);
    let boolv = || variable_with_lltype("c", LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let push = |block: &BlockRef, op: &str, args: Vec<Hlvalue>, result: Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            op,
            args,
            Hlvalue::Variable(result),
        ));
    };
    let switch_bool = |block: &BlockRef,
                       cond: Variable,
                       true_args: Vec<Hlvalue>,
                       true_tgt: BlockRef,
                       false_args: Vec<Hlvalue>,
                       false_tgt: BlockRef| {
        block.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
        block.closeblock(vec![
            Link::new(true_args, Some(true_tgt), Some(bool_const(true))).into_ref(),
            Link::new(false_args, Some(false_tgt), Some(bool_const(false))).into_ref(),
        ]);
    };
    let load_ord = |block: &BlockRef, chars: &Variable, idx: &Variable, name: &str| -> Variable {
        let ch = variable_with_lltype(name, LowLevelType::Char);
        push(
            block,
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(idx.clone()),
            ],
            ch.clone(),
        );
        let ordch = signed();
        push(
            block,
            "cast_char_to_int",
            vec![Hlvalue::Variable(ch)],
            ordch.clone(),
        );
        ordch
    };

    // `Utf8StringIterator.next` subtracted biases (`rutf8.py`).
    const BIAS2: i64 = (0xC0 << 6) + 0x80;
    const BIAS3: i64 = (0xE0 << 12) + (0x80 << 6) + 0x80;
    const BIAS4: i64 = (0xF0 << 18) + (0x80 << 12) + (0x80 << 6) + 0x80;

    let v_s = variable_with_lltype("s", str_lltype.clone());
    let v_end = variable_with_lltype("end", LowLevelType::Signed);
    let v_pos = variable_with_lltype("pos", LowLevelType::Signed);
    let v_atend = boolv();
    push(
        &startblock,
        "getfield",
        vec![
            Hlvalue::Variable(iter_arg.clone()),
            void_field_const("_utf8"),
        ],
        v_s.clone(),
    );
    push(
        &startblock,
        "getfield",
        vec![
            Hlvalue::Variable(iter_arg.clone()),
            void_field_const("_end"),
        ],
        v_end.clone(),
    );
    push(
        &startblock,
        "getfield",
        vec![
            Hlvalue::Variable(iter_arg.clone()),
            void_field_const("_pos"),
        ],
        v_pos.clone(),
    );
    push(
        &startblock,
        "int_eq",
        vec![
            Hlvalue::Variable(v_pos.clone()),
            Hlvalue::Variable(v_end.clone()),
        ],
        v_atend.clone(),
    );

    let a_iter = variable_with_lltype("iter", iter_lltype.clone());
    let a_s = variable_with_lltype("s", str_lltype.clone());
    let a_end = variable_with_lltype("end", LowLevelType::Signed);
    let a_pos = variable_with_lltype("pos", LowLevelType::Signed);
    let ascii = Block::shared(vec![
        Hlvalue::Variable(a_iter.clone()),
        Hlvalue::Variable(a_s.clone()),
        Hlvalue::Variable(a_end.clone()),
        Hlvalue::Variable(a_pos.clone()),
    ]);

    let exc_args = exception_args("StopIteration")?;
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_atend.clone()));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(iter_arg),
                Hlvalue::Variable(v_s),
                Hlvalue::Variable(v_end),
                Hlvalue::Variable(v_pos),
            ],
            Some(ascii.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let c_iter = variable_with_lltype("iter", iter_lltype.clone());
    let c_newpos = variable_with_lltype("newpos", LowLevelType::Signed);
    let c_result = variable_with_lltype("cp", LowLevelType::Signed);
    let commit = Block::shared(vec![
        Hlvalue::Variable(c_iter.clone()),
        Hlvalue::Variable(c_newpos.clone()),
        Hlvalue::Variable(c_result.clone()),
    ]);
    push(
        &commit,
        "setfield",
        vec![
            Hlvalue::Variable(c_iter),
            void_field_const("_pos"),
            Hlvalue::Variable(c_newpos),
        ],
        variable_with_lltype("vset", LowLevelType::Void),
    );
    commit.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c_result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let a_chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    push(
        &ascii,
        "getsubstruct",
        vec![Hlvalue::Variable(a_s), void_field_const("chars")],
        a_chars.clone(),
    );
    let a_ordch1 = load_ord(&ascii, &a_chars, &a_pos, "ch1");
    let a_pos1 = signed();
    push(
        &ascii,
        "int_add",
        vec![Hlvalue::Variable(a_pos.clone()), signed_const(1)],
        a_pos1.clone(),
    );
    let a_is1 = boolv();
    push(
        &ascii,
        "int_le",
        vec![Hlvalue::Variable(a_ordch1.clone()), signed_const(0x7F)],
        a_is1.clone(),
    );

    let t_iter = variable_with_lltype("iter", iter_lltype.clone());
    let t_end = variable_with_lltype("end", LowLevelType::Signed);
    let t_pos = variable_with_lltype("pos", LowLevelType::Signed);
    let t_pos1 = variable_with_lltype("pos1", LowLevelType::Signed);
    let t_chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let t_ordch1 = variable_with_lltype("ordch1", LowLevelType::Signed);
    let trunc = Block::shared(vec![
        Hlvalue::Variable(t_iter.clone()),
        Hlvalue::Variable(t_end.clone()),
        Hlvalue::Variable(t_pos.clone()),
        Hlvalue::Variable(t_pos1.clone()),
        Hlvalue::Variable(t_chars.clone()),
        Hlvalue::Variable(t_ordch1.clone()),
    ]);
    switch_bool(
        &ascii,
        a_is1,
        vec![
            Hlvalue::Variable(a_iter.clone()),
            Hlvalue::Variable(a_pos1.clone()),
            Hlvalue::Variable(a_ordch1.clone()),
        ],
        commit.clone(),
        vec![
            Hlvalue::Variable(a_iter),
            Hlvalue::Variable(a_end),
            Hlvalue::Variable(a_pos),
            Hlvalue::Variable(a_pos1),
            Hlvalue::Variable(a_chars),
            Hlvalue::Variable(a_ordch1),
        ],
        trunc.clone(),
    );

    let t_trunc = boolv();
    push(
        &trunc,
        "int_ge",
        vec![Hlvalue::Variable(t_pos1.clone()), Hlvalue::Variable(t_end)],
        t_trunc.clone(),
    );

    let two_iter = variable_with_lltype("iter", iter_lltype.clone());
    let two_pos = variable_with_lltype("pos", LowLevelType::Signed);
    let two_pos1 = variable_with_lltype("pos1", LowLevelType::Signed);
    let two_chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let two_ordch1 = variable_with_lltype("ordch1", LowLevelType::Signed);
    let two = Block::shared(vec![
        Hlvalue::Variable(two_iter.clone()),
        Hlvalue::Variable(two_pos.clone()),
        Hlvalue::Variable(two_pos1.clone()),
        Hlvalue::Variable(two_chars.clone()),
        Hlvalue::Variable(two_ordch1.clone()),
    ]);
    switch_bool(
        &trunc,
        t_trunc,
        vec![
            Hlvalue::Variable(t_iter.clone()),
            Hlvalue::Variable(t_pos1.clone()),
            Hlvalue::Variable(t_ordch1.clone()),
        ],
        commit.clone(),
        vec![
            Hlvalue::Variable(t_iter),
            Hlvalue::Variable(t_pos),
            Hlvalue::Variable(t_pos1),
            Hlvalue::Variable(t_chars),
            Hlvalue::Variable(t_ordch1),
        ],
        two.clone(),
    );

    let two_ordch2 = load_ord(&two, &two_chars, &two_pos1, "ch2");
    let two_pos2 = signed();
    push(
        &two,
        "int_add",
        vec![Hlvalue::Variable(two_pos.clone()), signed_const(2)],
        two_pos2.clone(),
    );
    let two_is2 = boolv();
    push(
        &two,
        "int_le",
        vec![Hlvalue::Variable(two_ordch1.clone()), signed_const(0xDF)],
        two_is2.clone(),
    );

    let d2_iter = variable_with_lltype("iter", iter_lltype.clone());
    let d2_pos2 = variable_with_lltype("pos2", LowLevelType::Signed);
    let d2_ordch1 = variable_with_lltype("ordch1", LowLevelType::Signed);
    let d2_ordch2 = variable_with_lltype("ordch2", LowLevelType::Signed);
    let decode2 = Block::shared(vec![
        Hlvalue::Variable(d2_iter.clone()),
        Hlvalue::Variable(d2_pos2.clone()),
        Hlvalue::Variable(d2_ordch1.clone()),
        Hlvalue::Variable(d2_ordch2.clone()),
    ]);
    let three_iter = variable_with_lltype("iter", iter_lltype.clone());
    let three_pos = variable_with_lltype("pos", LowLevelType::Signed);
    let three_pos2 = variable_with_lltype("pos2", LowLevelType::Signed);
    let three_chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let three_ordch1 = variable_with_lltype("ordch1", LowLevelType::Signed);
    let three_ordch2 = variable_with_lltype("ordch2", LowLevelType::Signed);
    let three = Block::shared(vec![
        Hlvalue::Variable(three_iter.clone()),
        Hlvalue::Variable(three_pos.clone()),
        Hlvalue::Variable(three_pos2.clone()),
        Hlvalue::Variable(three_chars.clone()),
        Hlvalue::Variable(three_ordch1.clone()),
        Hlvalue::Variable(three_ordch2.clone()),
    ]);
    switch_bool(
        &two,
        two_is2,
        vec![
            Hlvalue::Variable(two_iter.clone()),
            Hlvalue::Variable(two_pos2.clone()),
            Hlvalue::Variable(two_ordch1.clone()),
            Hlvalue::Variable(two_ordch2.clone()),
        ],
        decode2.clone(),
        vec![
            Hlvalue::Variable(two_iter),
            Hlvalue::Variable(two_pos),
            Hlvalue::Variable(two_pos2),
            Hlvalue::Variable(two_chars),
            Hlvalue::Variable(two_ordch1),
            Hlvalue::Variable(two_ordch2),
        ],
        three.clone(),
    );

    let d2_s1 = signed();
    push(
        &decode2,
        "int_lshift",
        vec![Hlvalue::Variable(d2_ordch1), signed_const(6)],
        d2_s1.clone(),
    );
    let d2_s2 = signed();
    push(
        &decode2,
        "int_add",
        vec![Hlvalue::Variable(d2_s1), Hlvalue::Variable(d2_ordch2)],
        d2_s2.clone(),
    );
    let d2_cp = signed();
    push(
        &decode2,
        "int_sub",
        vec![Hlvalue::Variable(d2_s2), signed_const(BIAS2)],
        d2_cp.clone(),
    );
    decode2.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(d2_iter),
                Hlvalue::Variable(d2_pos2),
                Hlvalue::Variable(d2_cp),
            ],
            Some(commit.clone()),
            None,
        )
        .into_ref(),
    ]);

    let three_ordch3 = load_ord(&three, &three_chars, &three_pos2, "ch3");
    let three_pos3 = signed();
    push(
        &three,
        "int_add",
        vec![Hlvalue::Variable(three_pos.clone()), signed_const(3)],
        three_pos3.clone(),
    );
    let three_is3 = boolv();
    push(
        &three,
        "int_le",
        vec![Hlvalue::Variable(three_ordch1.clone()), signed_const(0xEF)],
        three_is3.clone(),
    );

    let d3_iter = variable_with_lltype("iter", iter_lltype.clone());
    let d3_pos3 = variable_with_lltype("pos3", LowLevelType::Signed);
    let d3_ordch1 = variable_with_lltype("ordch1", LowLevelType::Signed);
    let d3_ordch2 = variable_with_lltype("ordch2", LowLevelType::Signed);
    let d3_ordch3 = variable_with_lltype("ordch3", LowLevelType::Signed);
    let decode3 = Block::shared(vec![
        Hlvalue::Variable(d3_iter.clone()),
        Hlvalue::Variable(d3_pos3.clone()),
        Hlvalue::Variable(d3_ordch1.clone()),
        Hlvalue::Variable(d3_ordch2.clone()),
        Hlvalue::Variable(d3_ordch3.clone()),
    ]);
    let four_iter = variable_with_lltype("iter", iter_lltype.clone());
    let four_pos3 = variable_with_lltype("pos3", LowLevelType::Signed);
    let four_chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    let four_ordch1 = variable_with_lltype("ordch1", LowLevelType::Signed);
    let four_ordch2 = variable_with_lltype("ordch2", LowLevelType::Signed);
    let four_ordch3 = variable_with_lltype("ordch3", LowLevelType::Signed);
    let four = Block::shared(vec![
        Hlvalue::Variable(four_iter.clone()),
        Hlvalue::Variable(four_pos3.clone()),
        Hlvalue::Variable(four_chars.clone()),
        Hlvalue::Variable(four_ordch1.clone()),
        Hlvalue::Variable(four_ordch2.clone()),
        Hlvalue::Variable(four_ordch3.clone()),
    ]);
    switch_bool(
        &three,
        three_is3,
        vec![
            Hlvalue::Variable(three_iter.clone()),
            Hlvalue::Variable(three_pos3.clone()),
            Hlvalue::Variable(three_ordch1.clone()),
            Hlvalue::Variable(three_ordch2.clone()),
            Hlvalue::Variable(three_ordch3.clone()),
        ],
        decode3.clone(),
        vec![
            Hlvalue::Variable(three_iter),
            Hlvalue::Variable(three_pos3),
            Hlvalue::Variable(three_chars),
            Hlvalue::Variable(three_ordch1),
            Hlvalue::Variable(three_ordch2),
            Hlvalue::Variable(three_ordch3),
        ],
        four.clone(),
    );

    let d3_s1 = signed();
    push(
        &decode3,
        "int_lshift",
        vec![Hlvalue::Variable(d3_ordch1), signed_const(12)],
        d3_s1.clone(),
    );
    let d3_s2 = signed();
    push(
        &decode3,
        "int_lshift",
        vec![Hlvalue::Variable(d3_ordch2), signed_const(6)],
        d3_s2.clone(),
    );
    let d3_s3 = signed();
    push(
        &decode3,
        "int_add",
        vec![Hlvalue::Variable(d3_s1), Hlvalue::Variable(d3_s2)],
        d3_s3.clone(),
    );
    let d3_s4 = signed();
    push(
        &decode3,
        "int_add",
        vec![Hlvalue::Variable(d3_s3), Hlvalue::Variable(d3_ordch3)],
        d3_s4.clone(),
    );
    let d3_cp = signed();
    push(
        &decode3,
        "int_sub",
        vec![Hlvalue::Variable(d3_s4), signed_const(BIAS3)],
        d3_cp.clone(),
    );
    decode3.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(d3_iter),
                Hlvalue::Variable(d3_pos3),
                Hlvalue::Variable(d3_cp),
            ],
            Some(commit.clone()),
            None,
        )
        .into_ref(),
    ]);

    let four_ordch4 = load_ord(&four, &four_chars, &four_pos3, "ch4");
    let four_pos4 = signed();
    push(
        &four,
        "int_add",
        vec![Hlvalue::Variable(four_pos3.clone()), signed_const(1)],
        four_pos4.clone(),
    );
    let f_s1 = signed();
    push(
        &four,
        "int_lshift",
        vec![Hlvalue::Variable(four_ordch1), signed_const(18)],
        f_s1.clone(),
    );
    let f_s2 = signed();
    push(
        &four,
        "int_lshift",
        vec![Hlvalue::Variable(four_ordch2), signed_const(12)],
        f_s2.clone(),
    );
    let f_s3 = signed();
    push(
        &four,
        "int_lshift",
        vec![Hlvalue::Variable(four_ordch3), signed_const(6)],
        f_s3.clone(),
    );
    let f_a1 = signed();
    push(
        &four,
        "int_add",
        vec![Hlvalue::Variable(f_s1), Hlvalue::Variable(f_s2)],
        f_a1.clone(),
    );
    let f_a2 = signed();
    push(
        &four,
        "int_add",
        vec![Hlvalue::Variable(f_a1), Hlvalue::Variable(f_s3)],
        f_a2.clone(),
    );
    let f_a3 = signed();
    push(
        &four,
        "int_add",
        vec![Hlvalue::Variable(f_a2), Hlvalue::Variable(four_ordch4)],
        f_a3.clone(),
    );
    let f_cp = signed();
    push(
        &four,
        "int_sub",
        vec![Hlvalue::Variable(f_a3), signed_const(BIAS4)],
        f_cp.clone(),
    );
    four.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(four_iter),
                Hlvalue::Variable(four_pos4),
                Hlvalue::Variable(f_cp),
            ],
            Some(commit),
            None,
        )
        .into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `Utf8StringIterator.next` is decorated `@always_inline`
    // (`rutf8.py` / `objectmodel.always_inline`).
    func._always_inline_ = crate::flowspace::model::AlwaysInline::True;
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["iter".to_string()],
        func,
    ))
}

/// Synthesise `rutf8.unichr_as_utf8(code, allow_surrogates=True)`
/// (`rpython/rlib/rutf8.py`):
///
/// ```python
/// def unichr_as_utf8(code, allow_surrogates=False):
///     """Encode code (numeric value) as utf8 encoded string
///     """
///     assert code >= 0
///     code = r_uint(code)
///     if code <= r_uint(0x7F):
///         # Encode ASCII
///         return chr(code)
///     if code <= r_uint(0x07FF):
///         return chr((0xc0 | (code >> 6))) + chr((0x80 | (code & 0x3f)))
///     if code <= r_uint(0xFFFF):
///         if not allow_surrogates and 0xD800 <= code <= 0xDfff:
///             raise OutOfRange(code)
///         return (chr((0xe0 | (code >> 12))) +
///                 chr((0x80 | ((code >> 6) & 0x3f))) +
///                 chr((0x80 | (code & 0x3f))))
///     if code <= r_uint(0x10FFFF):
///         return (chr((0xf0 | (code >> 18))) +
///                 chr((0x80 | ((code >> 12) & 0x3f))) +
///                 chr((0x80 | ((code >> 6) & 0x3f))) +
///                 chr((0x80 | (code & 0x3f))))
///     raise OutOfRange(code)
/// ```
///
/// The only rtyper call site is `unichr_as_utf8(code, allow_surrogates=True)`
/// (`rtype_builtin_unichr_as_utf8` / front `__majit_unichr_as_utf8`), so
/// the `0xD800 <= code <= 0xDfff` `OutOfRange` branch is folded as that
/// constant-True specialization. `assert code >= 0` is `int_ge` +
/// `debug_assert` (`removeassert.kill_assertion_link`; same form as
/// `ll_newlist_hint` in `rlist.build_ll_newlist_helper_graph_inner`).
/// `code = r_uint(code)` is `cast_int_to_uint`; every range test is `uint_le`
/// (`rint.rtype_compare_template` with `unsigned_repr` prefix `uint_`).
/// `raise OutOfRange(code)` is `InstanceRepr.new_instance` then
/// `InstanceRepr.setfield` of `code` (`OutOfRange.__init__` /
/// `rclass.py`).
pub(crate) fn build_ll_unichr_as_utf8_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&STRPTR)?;
    let str_struct_lltype = struct_lltype_from_strptr(&STRPTR)?;
    let gc_flavor = gc_flavor_const()?;
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let unsigned_const =
        |n: i64| constant_with_lltype(ConstValue::UInt(n as u64), LowLevelType::Unsigned);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);
    let push = |block: &BlockRef, op: &str, args: Vec<Hlvalue>, result: Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            op,
            args,
            Hlvalue::Variable(result),
        ));
    };
    let switch_bool = |block: &BlockRef,
                       cond: Variable,
                       true_args: Vec<Hlvalue>,
                       true_tgt: BlockRef,
                       false_args: Vec<Hlvalue>,
                       false_tgt: BlockRef| {
        block.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
        block.closeblock(vec![
            Link::new(true_args, Some(true_tgt), Some(bool_const(true))).into_ref(),
            Link::new(false_args, Some(false_tgt), Some(bool_const(false))).into_ref(),
        ]);
    };
    // After `code = r_uint(code)`, shifts/masks/ors are `uint_*`
    // (`rint.rtype_template` with prefix `uint_`). `chr` is
    // `IntegerRepr.rtype_chr`: `hop.inputargs(Signed)` then
    // `cast_int_to_char`, so Unsigned values go through `cast_uint_to_int`.
    let utf8_byte =
        |block: &BlockRef, src: &Variable, shift: i64, mask: i64, prefix: i64, tag: &str| {
            let mut cur = src.clone();
            if shift != 0 {
                let shifted = variable_with_lltype(tag, LowLevelType::Unsigned);
                push(
                    block,
                    "uint_rshift",
                    vec![Hlvalue::Variable(cur), signed_const(shift)],
                    shifted.clone(),
                );
                cur = shifted;
            }
            if mask != !0 {
                let masked = variable_with_lltype(tag, LowLevelType::Unsigned);
                push(
                    block,
                    "uint_and",
                    vec![Hlvalue::Variable(cur), unsigned_const(mask)],
                    masked.clone(),
                );
                cur = masked;
            }
            if prefix != 0 {
                let orred = variable_with_lltype(tag, LowLevelType::Unsigned);
                push(
                    block,
                    "uint_or",
                    vec![unsigned_const(prefix), Hlvalue::Variable(cur)],
                    orred.clone(),
                );
                cur = orred;
            }
            let as_int = variable_with_lltype(tag, LowLevelType::Signed);
            push(
                block,
                "cast_uint_to_int",
                vec![Hlvalue::Variable(cur)],
                as_int.clone(),
            );
            let ch = variable_with_lltype(tag, LowLevelType::Char);
            push(
                block,
                "cast_int_to_char",
                vec![Hlvalue::Variable(as_int)],
                ch.clone(),
            );
            ch
        };
    let store_byte = |block: &BlockRef, chars: &Variable, index: i64, byte: Variable| {
        let void_set = variable_with_lltype("set", LowLevelType::Void);
        push(
            block,
            "setarrayitem",
            vec![
                Hlvalue::Variable(chars.clone()),
                signed_const(index),
                Hlvalue::Variable(byte),
            ],
            void_set,
        );
    };

    let code = variable_with_lltype("code", LowLevelType::Signed);
    let startblock = Block::shared(vec![Hlvalue::Variable(code.clone())]);
    let return_var = variable_with_lltype("result", STRPTR.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    // `unichr_as_utf8` raises `OutOfRange(code)` (`rutf8.OutOfRange`).
    // The raising block is `InstanceRepr.new_instance` plus `setfield`
    // of `code`; the except link is `(OutOfRange vtable, instance)`
    // (`ClassRepr.convert_desc` / `rutf8.py`).

    // `assert code >= 0` (`unichr_as_utf8`). Python `assert` becomes
    // `debug_assert` in `removeassert.kill_assertion_link`; same `int_ge`
    // + `debug_assert` pairing as `ll_newlist_hint`
    // (`rlist.build_ll_newlist_helper_graph_inner`).
    let ge0 = variable_with_lltype("ge0", LowLevelType::Bool);
    push(
        &startblock,
        "int_ge",
        vec![Hlvalue::Variable(code.clone()), signed_const(0)],
        ge0.clone(),
    );
    let assert_void = variable_with_lltype("v", LowLevelType::Void);
    push(
        &startblock,
        "debug_assert",
        vec![
            Hlvalue::Variable(ge0),
            constant_with_lltype(
                ConstValue::byte_str(format!("assertion failed in {name}")),
                LowLevelType::Void,
            ),
        ],
        assert_void,
    );
    // `code = r_uint(code)` (`unichr_as_utf8`); `rint.convert_from_to`
    // Signed→Unsigned is `cast_int_to_uint`.
    let code_u = variable_with_lltype("code", LowLevelType::Unsigned);
    push(
        &startblock,
        "cast_int_to_uint",
        vec![Hlvalue::Variable(code)],
        code_u.clone(),
    );

    let is_ascii = variable_with_lltype("is_ascii", LowLevelType::Bool);
    push(
        &startblock,
        "uint_le",
        vec![Hlvalue::Variable(code_u.clone()), unsigned_const(0x7F)],
        is_ascii.clone(),
    );

    let alloc1_code = variable_with_lltype("code", LowLevelType::Unsigned);
    let alloc1 = Block::shared(vec![Hlvalue::Variable(alloc1_code.clone())]);
    let two_code = variable_with_lltype("code", LowLevelType::Unsigned);
    let two = Block::shared(vec![Hlvalue::Variable(two_code.clone())]);
    switch_bool(
        &startblock,
        is_ascii,
        vec![Hlvalue::Variable(code_u.clone())],
        alloc1.clone(),
        vec![Hlvalue::Variable(code_u)],
        two.clone(),
    );

    let is_two = variable_with_lltype("is_two", LowLevelType::Bool);
    push(
        &two,
        "uint_le",
        vec![Hlvalue::Variable(two_code.clone()), unsigned_const(0x07FF)],
        is_two.clone(),
    );

    let alloc2_code = variable_with_lltype("code", LowLevelType::Unsigned);
    let alloc2 = Block::shared(vec![Hlvalue::Variable(alloc2_code.clone())]);
    let three_code = variable_with_lltype("code", LowLevelType::Unsigned);
    let three = Block::shared(vec![Hlvalue::Variable(three_code.clone())]);
    switch_bool(
        &two,
        is_two,
        vec![Hlvalue::Variable(two_code.clone())],
        alloc2.clone(),
        vec![Hlvalue::Variable(two_code)],
        three.clone(),
    );

    let is_three = variable_with_lltype("is_three", LowLevelType::Bool);
    push(
        &three,
        "uint_le",
        vec![
            Hlvalue::Variable(three_code.clone()),
            unsigned_const(0xFFFF),
        ],
        is_three.clone(),
    );

    let alloc3_code = variable_with_lltype("code", LowLevelType::Unsigned);
    let alloc3 = Block::shared(vec![Hlvalue::Variable(alloc3_code.clone())]);
    let four_code = variable_with_lltype("code", LowLevelType::Unsigned);
    let four = Block::shared(vec![Hlvalue::Variable(four_code.clone())]);
    switch_bool(
        &three,
        is_three,
        vec![Hlvalue::Variable(three_code.clone())],
        alloc3.clone(),
        vec![Hlvalue::Variable(three_code)],
        four.clone(),
    );

    let is_four = variable_with_lltype("is_four", LowLevelType::Bool);
    push(
        &four,
        "uint_le",
        vec![
            Hlvalue::Variable(four_code.clone()),
            unsigned_const(0x10FFFF),
        ],
        is_four.clone(),
    );
    let alloc4_code = variable_with_lltype("code", LowLevelType::Unsigned);
    let alloc4 = Block::shared(vec![Hlvalue::Variable(alloc4_code.clone())]);
    let raise_code = variable_with_lltype("code", LowLevelType::Unsigned);
    let raise_block = Block::shared(vec![Hlvalue::Variable(raise_code.clone())]);
    switch_bool(
        &four,
        is_four,
        vec![Hlvalue::Variable(four_code.clone())],
        alloc4.clone(),
        vec![Hlvalue::Variable(four_code)],
        raise_block.clone(),
    );
    let exc_args = crate::rlib::rutf8::raise_out_of_range_args(rtyper, &raise_block, &raise_code)?;
    raise_block.closeblock(vec![
        Link::new(exc_args, Some(graph.exceptblock.clone()), None).into_ref(),
    ]);

    let malloc_str = |block: &BlockRef, n: i64| -> (Variable, Variable) {
        let result = variable_with_lltype("s", STRPTR.clone());
        push(
            block,
            "malloc_varsize",
            vec![
                lowlevel_type_const(str_struct_lltype.clone()),
                gc_flavor.clone(),
                signed_const(n),
            ],
            result.clone(),
        );
        let chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
        push(
            block,
            "getsubstruct",
            vec![Hlvalue::Variable(result.clone()), chars_field_const()],
            chars.clone(),
        );
        (result, chars)
    };
    let finish = |block: &BlockRef, result: Variable| {
        block.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(result)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
    };

    let (s1, chars1) = malloc_str(&alloc1, 1);
    let b1 = utf8_byte(&alloc1, &alloc1_code, 0, !0, 0, "b0");
    store_byte(&alloc1, &chars1, 0, b1);
    finish(&alloc1, s1);

    let (s2, chars2) = malloc_str(&alloc2, 2);
    let b20 = utf8_byte(&alloc2, &alloc2_code, 6, !0, 0xc0, "b0");
    let b21 = utf8_byte(&alloc2, &alloc2_code, 0, 0x3f, 0x80, "b1");
    store_byte(&alloc2, &chars2, 0, b20);
    store_byte(&alloc2, &chars2, 1, b21);
    finish(&alloc2, s2);

    let (s3, chars3) = malloc_str(&alloc3, 3);
    let b30 = utf8_byte(&alloc3, &alloc3_code, 12, !0, 0xe0, "b0");
    let b31 = utf8_byte(&alloc3, &alloc3_code, 6, 0x3f, 0x80, "b1");
    let b32 = utf8_byte(&alloc3, &alloc3_code, 0, 0x3f, 0x80, "b2");
    store_byte(&alloc3, &chars3, 0, b30);
    store_byte(&alloc3, &chars3, 1, b31);
    store_byte(&alloc3, &chars3, 2, b32);
    finish(&alloc3, s3);

    let (s4, chars4) = malloc_str(&alloc4, 4);
    let b40 = utf8_byte(&alloc4, &alloc4_code, 18, !0, 0xf0, "b0");
    let b41 = utf8_byte(&alloc4, &alloc4_code, 12, 0x3f, 0x80, "b1");
    let b42 = utf8_byte(&alloc4, &alloc4_code, 6, 0x3f, 0x80, "b2");
    let b43 = utf8_byte(&alloc4, &alloc4_code, 0, 0x3f, 0x80, "b3");
    store_byte(&alloc4, &chars4, 0, b40);
    store_byte(&alloc4, &chars4, 1, b41);
    store_byte(&alloc4, &chars4, 2, b42);
    store_byte(&alloc4, &chars4, 3, b43);
    finish(&alloc4, s4);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["code".to_string()],
        func,
    ))
}

pub fn ll_striter() -> Result<(), TyperError> {
    Err(rstr_deferred("ll_striter"))
}

pub fn ll_strnext() -> Result<(), TyperError> {
    Err(rstr_deferred("ll_strnext"))
}

pub fn ll_getnextindex() -> Result<(), TyperError> {
    Err(rstr_deferred("ll_getnextindex"))
}

// LLHelpers — `lltypesystem/rstr.py` `class LLHelpers(AbstractLLHelpers)`.
//
// Each helper is synthesised as a single-block low-level graph the
// rtyper inserts via `gendirectcall`. The helpers correspond to
// upstream `LLHelpers.ll_*` static methods.

/// Build the `getsubstruct('chars')` + `getarraysize` op pair shared
/// by `ll_strlen` and `ll_unilen`. The `chars` field is an inline
/// `Array(Char)` / `Array(UniChar)` inside the `STR`/`UNICODE`
/// GcStruct, so RPython emits `getsubstruct` (returning an interior
/// `Ptr(Array(...))`) rather than `getfield`. Compare
/// `rmodel.rs`'s `is_container_type` branch for the same
/// dispatch on the rtyper side.
fn emit_chars_length_ops(
    startblock: &crate::flowspace::model::BlockRef,
    arg: Hlvalue,
    chars_array_ptr_lltype: LowLevelType,
) -> Hlvalue {
    let v_chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![
            arg,
            constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
        ],
        Hlvalue::Variable(v_chars.clone()),
    ));
    let v_len = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(v_chars)],
        Hlvalue::Variable(v_len.clone()),
    ));
    Hlvalue::Variable(v_len)
}

/// Build the chars Array's `Ptr(Array(...))` lltype for a resolved
/// `STR`/`UNICODE` `ForwardReference`. The result wraps the field's
/// inline `Array(Char|UniChar)` in `Ptr` so the synthesised
/// `getsubstruct` op's result variable carries a valid
/// `SomePtr`-shaped annotation (upstream `lltype.py+`'s
/// `SomePtr.__init__` rejects bare container types — see
/// `llannotation.rs`).
pub(crate) fn chars_array_ptr_lltype_from_strptr(
    ptr_lltype: &LowLevelType,
) -> Result<LowLevelType, TyperError> {
    let LowLevelType::Ptr(ptr) = ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_strlen helper expects Ptr(STR/UNICODE), got {ptr_lltype:?}"
        )));
    };
    let body = match &ptr.TO {
        PtrTarget::ForwardReference(fwd) => fwd
            .resolved()
            .ok_or_else(|| TyperError::message("STR/UNICODE ForwardReference not resolved"))?,
        PtrTarget::Struct(s) => LowLevelType::Struct(Box::new(s.clone())),
        other => {
            return Err(TyperError::message(format!(
                "ll_strlen helper Ptr.TO must target STR/UNICODE struct, got {other:?}"
            )));
        }
    };
    let LowLevelType::Struct(body) = body else {
        return Err(TyperError::message(
            "STR/UNICODE ForwardReference must resolve to Struct",
        ));
    };
    let chars_field = body
        ._flds
        .get("chars")
        .cloned()
        .ok_or_else(|| TyperError::message("STR/UNICODE struct has no chars field"))?;
    let LowLevelType::Array(arr) = chars_field else {
        return Err(TyperError::message(
            "STR/UNICODE chars field must be Array(Char|UniChar)",
        ));
    };
    Ok(LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(*arr),
    })))
}

/// Resolve the `Ptr(STR/UNICODE)` wrapper to the concrete
/// `LowLevelType::Struct(body)` payload that names the GcStruct.
/// `malloc_varsize` SpaceOperation's first arg requires this concrete
/// struct lltype (mirrors upstream `cTEMP = inputconst(Void, TEMPBUF)`
/// at rstr.py:1129 where `TEMPBUF` is the resolved GcStruct itself,
/// not its Ptr wrapper).
pub(crate) fn struct_lltype_from_strptr(
    ptr_lltype: &LowLevelType,
) -> Result<LowLevelType, TyperError> {
    let LowLevelType::Ptr(ptr) = ptr_lltype else {
        return Err(TyperError::message(format!(
            "struct_lltype_from_strptr expects Ptr(STR/UNICODE), got {ptr_lltype:?}"
        )));
    };
    let body = match &ptr.TO {
        PtrTarget::ForwardReference(fwd) => fwd
            .resolved()
            .ok_or_else(|| TyperError::message("STR/UNICODE ForwardReference not resolved"))?,
        PtrTarget::Struct(s) => LowLevelType::Struct(Box::new(s.clone())),
        other => {
            return Err(TyperError::message(format!(
                "struct_lltype_from_strptr Ptr.TO must target STR/UNICODE struct, got {other:?}"
            )));
        }
    };
    if !matches!(body, LowLevelType::Struct(_)) {
        return Err(TyperError::message(
            "STR/UNICODE ForwardReference must resolve to Struct",
        ));
    }
    Ok(body)
}

/// Synthesise the helper graph for `LLHelpers.ll_strlen` /
/// `LLHelpers.ll_length` (`lltypesystem/rstr.py, 417-418`):
///
/// ```python
/// @staticmethod
/// def ll_strlen(s):
///     return len(s.chars)
/// ```
///
/// Single-block graph:
/// `getsubstruct(s, 'chars') -> Ptr(Array); getarraysize(...) -> Signed`.
/// `name` is the helper identity (`"ll_strlen"` / `"ll_unilen"`)
/// and `ptr_lltype` selects between `Ptr(STR)` and `Ptr(UNICODE)`.
pub(crate) fn build_ll_strlen_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let arg = variable_with_lltype("s", ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_len = emit_chars_length_ops(&startblock, Hlvalue::Variable(arg), chars_array_ptr_lltype);
    startblock.closeblock(vec![
        Link::new(vec![v_len], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise `mallocstr` / `mallocunicode` (`lltypesystem/rstr.py`):
///
/// ```python
/// def mallocstr(length):
///     ll_assert(length >= 0, "negative string length")
///     r = malloc(TP, length)
///     if not we_are_translated() or not malloc_zero_filled:
///         r.hash = 0
///     return r
/// ```
///
/// Single block: `malloc_varsize(STR/UNICODE, gc, length)` then
/// `setfield(r, 'hash', 0)`. `ll_assert` is a translation-time no-op, and
/// the zero-fill guard collapses to always initialising the hash field
/// (matching the inline `mallocstr` shape emitted by the int-to-string
/// helpers). `name` selects the helper identity (`"mallocstr"` /
/// `"mallocunicode"`) and `ptr_lltype` selects `Ptr(STR)` vs
/// `Ptr(UNICODE)`.
pub(crate) fn build_ll_mallocstr_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let struct_lltype = struct_lltype_from_strptr(&ptr_lltype)?;
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let hash_field_const =
        || constant_with_lltype(ConstValue::byte_str("hash"), LowLevelType::Void);

    let length_arg = variable_with_lltype("length", LowLevelType::Signed);
    let startblock = Block::shared(vec![Hlvalue::Variable(length_arg.clone())]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let result = variable_with_lltype("r", ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(struct_lltype),
            gc_flavor_const()?,
            Hlvalue::Variable(length_arg),
        ],
        Hlvalue::Variable(result.clone()),
    ));
    let set_hash = variable_with_lltype("set", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(result.clone()),
            hash_field_const(),
            signed_const(0),
        ],
        Hlvalue::Variable(set_hash),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["length".to_string()],
        func,
    ))
}

/// Synthesise `LLHelpers.ll_chr2str` (`lltypesystem/rstr.py`):
///
/// ```python
/// def ll_chr2str(ch):
///     if typeOf(ch) is Char:
///         malloc = mallocstr
///     else:
///         malloc = mallocunicode
///     s = malloc(1)
///     s.chars[0] = ch
///     return s
/// ```
///
/// Pyre specialises the upstream `typeOf(ch)` branch at helper-build
/// time via `(ptr_lltype, elem_lltype)`: `Char -> Ptr(STR)` or
/// `UniChar -> Ptr(UNICODE)`. The graph is a single block:
/// `malloc_varsize(STR|UNICODE, gc, 1)`, `getsubstruct(..., 'chars')`,
/// `setarrayitem(chars, 0, ch)`, return the allocated string.
pub(crate) fn build_ll_chr2str_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    elem_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_chr2str helper expects Ptr(Array(Char|UniChar)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_chr2str helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    if arr.OF != elem_lltype {
        return Err(TyperError::message(format!(
            "ll_chr2str helper element mismatch: chars array stores {:?}, argument is {elem_lltype:?}",
            arr.OF
        )));
    }
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_chr2str helper unsupported char element type {elem_lltype:?}"
        )));
    }

    let struct_lltype = struct_lltype_from_strptr(&ptr_lltype)?;
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let ch = variable_with_lltype("ch", elem_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(ch.clone())]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let newstr = variable_with_lltype("s", ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(struct_lltype),
            gc_flavor_const()?,
            signed_const(1),
        ],
        Hlvalue::Variable(newstr.clone()),
    ));

    let newchars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(newstr.clone()), chars_field_const()],
        Hlvalue::Variable(newchars.clone()),
    ));

    let void_set = variable_with_lltype("set0", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(newchars),
            signed_const(0),
            Hlvalue::Variable(ch),
        ],
        Hlvalue::Variable(void_set),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(newstr)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ch".to_string()],
        func,
    ))
}

/// Synthesise `LLHelpers.ll_str2unicode` (`lltypesystem/rstr.py`):
///
/// ```python
/// def ll_str2unicode(str):
///     lgt = len(str.chars)
///     s = mallocunicode(lgt)
///     for i in range(lgt):
///         if ord(str.chars[i]) > 127:
///             raise UnicodeDecodeError
///         s.chars[i] = cast_primitive(UniChar, str.chars[i])
///     return s
/// ```
///
/// 4-block CFG plus returnblock and exceptblock:
/// - **start**: read source length, allocate UNICODE with
///   `malloc_varsize`, then enter the loop at `i = 0`.
/// - **loop_cond**: `i < length`; False returns the allocated unicode.
/// - **check_char**: load `str.chars[i]`, cast to int, raise
///   `UnicodeDecodeError` when the codepoint is > 127.
/// - **store_char**: cast the ASCII code to UniChar, store it, increment
///   `i`, and jump back to `loop_cond`.
pub(crate) fn build_ll_str2unicode_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let src_chars_ptr_lltype = chars_array_ptr_lltype_from_strptr(&STRPTR)?;
    let dst_chars_ptr_lltype = chars_array_ptr_lltype_from_strptr(&UNICODEPTR)?;
    let unicode_struct_lltype = struct_lltype_from_strptr(&UNICODEPTR)?;

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);
    let exc_args = exception_args("UnicodeDecodeError")?;

    let str_arg = variable_with_lltype("str", STRPTR.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(str_arg.clone())]);
    let return_var = variable_with_lltype("result", UNICODEPTR.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let src_chars_for_cond = variable_with_lltype("chars", src_chars_ptr_lltype.clone());
    let dst_uni_for_cond = variable_with_lltype("s", UNICODEPTR.clone());
    let dst_chars_for_cond = variable_with_lltype("newchars", dst_chars_ptr_lltype.clone());
    let length_for_cond = variable_with_lltype("lgt", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(src_chars_for_cond.clone()),
        Hlvalue::Variable(dst_uni_for_cond.clone()),
        Hlvalue::Variable(dst_chars_for_cond.clone()),
        Hlvalue::Variable(length_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    let src_chars_for_check = variable_with_lltype("chars", src_chars_ptr_lltype.clone());
    let dst_uni_for_check = variable_with_lltype("s", UNICODEPTR.clone());
    let dst_chars_for_check = variable_with_lltype("newchars", dst_chars_ptr_lltype.clone());
    let length_for_check = variable_with_lltype("lgt", LowLevelType::Signed);
    let i_for_check = variable_with_lltype("i", LowLevelType::Signed);
    let block_check_char = Block::shared(vec![
        Hlvalue::Variable(src_chars_for_check.clone()),
        Hlvalue::Variable(dst_uni_for_check.clone()),
        Hlvalue::Variable(dst_chars_for_check.clone()),
        Hlvalue::Variable(length_for_check.clone()),
        Hlvalue::Variable(i_for_check.clone()),
    ]);

    let src_chars_for_store = variable_with_lltype("chars", src_chars_ptr_lltype.clone());
    let dst_uni_for_store = variable_with_lltype("s", UNICODEPTR.clone());
    let dst_chars_for_store = variable_with_lltype("newchars", dst_chars_ptr_lltype.clone());
    let length_for_store = variable_with_lltype("lgt", LowLevelType::Signed);
    let i_for_store = variable_with_lltype("i", LowLevelType::Signed);
    let c_int_for_store = variable_with_lltype("c_int", LowLevelType::Signed);
    let block_store_char = Block::shared(vec![
        Hlvalue::Variable(src_chars_for_store.clone()),
        Hlvalue::Variable(dst_uni_for_store.clone()),
        Hlvalue::Variable(dst_chars_for_store.clone()),
        Hlvalue::Variable(length_for_store.clone()),
        Hlvalue::Variable(i_for_store.clone()),
        Hlvalue::Variable(c_int_for_store.clone()),
    ]);

    // start: source chars/length, mallocunicode(length), dest chars.
    let src_chars = variable_with_lltype("chars", src_chars_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(str_arg), chars_field_const()],
        Hlvalue::Variable(src_chars.clone()),
    ));
    let length = variable_with_lltype("lgt", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(src_chars.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let newuni = variable_with_lltype("s", UNICODEPTR.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(unicode_struct_lltype),
            gc_flavor_const()?,
            Hlvalue::Variable(length.clone()),
        ],
        Hlvalue::Variable(newuni.clone()),
    ));
    let newchars = variable_with_lltype("newchars", dst_chars_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(newuni.clone()), chars_field_const()],
        Hlvalue::Variable(newchars.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_chars),
                Hlvalue::Variable(newuni),
                Hlvalue::Variable(newchars),
                Hlvalue::Variable(length),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // loop_cond: i < lgt.
    let keep_going = variable_with_lltype("keep_going", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(length_for_cond.clone()),
            ],
            Hlvalue::Variable(keep_going.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(keep_going));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_chars_for_cond),
                Hlvalue::Variable(dst_uni_for_cond.clone()),
                Hlvalue::Variable(dst_chars_for_cond),
                Hlvalue::Variable(length_for_cond),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_check_char.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(dst_uni_for_cond)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // check_char: ord(str.chars[i]) > 127 raises UnicodeDecodeError.
    let c = variable_with_lltype("c", LowLevelType::Char);
    block_check_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(src_chars_for_check.clone()),
                Hlvalue::Variable(i_for_check.clone()),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    let c_int = variable_with_lltype("c_int", LowLevelType::Signed);
    block_check_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_char_to_int",
            vec![Hlvalue::Variable(c)],
            Hlvalue::Variable(c_int.clone()),
        ));
    let too_large = variable_with_lltype("too_large", LowLevelType::Bool);
    block_check_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_gt",
            vec![Hlvalue::Variable(c_int.clone()), signed_const(127)],
            Hlvalue::Variable(too_large.clone()),
        ));
    block_check_char.borrow_mut().exitswitch = Some(Hlvalue::Variable(too_large));
    block_check_char.closeblock(vec![
        Link::new(exc_args, Some(graph.exceptblock.clone()), Some(bool_true())).into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(src_chars_for_check),
                Hlvalue::Variable(dst_uni_for_check),
                Hlvalue::Variable(dst_chars_for_check),
                Hlvalue::Variable(length_for_check),
                Hlvalue::Variable(i_for_check),
                Hlvalue::Variable(c_int),
            ],
            Some(block_store_char.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // store_char: cast ASCII code to UniChar, store, increment.
    let uc = variable_with_lltype("uc", LowLevelType::UniChar);
    block_store_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_int_to_unichar",
            vec![Hlvalue::Variable(c_int_for_store)],
            Hlvalue::Variable(uc.clone()),
        ));
    let void_set = variable_with_lltype("set", LowLevelType::Void);
    block_store_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(dst_chars_for_store.clone()),
                Hlvalue::Variable(i_for_store.clone()),
                Hlvalue::Variable(uc),
            ],
            Hlvalue::Variable(void_set),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_store_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_store), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_store_char.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_chars_for_store),
                Hlvalue::Variable(dst_uni_for_store),
                Hlvalue::Variable(dst_chars_for_store),
                Hlvalue::Variable(length_for_store),
                Hlvalue::Variable(i_next),
            ],
            Some(block_loop_cond),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["str".to_string()],
        func,
    ))
}

fn join_chars_item_lltype(items_ptr_lltype: &LowLevelType) -> Result<LowLevelType, TyperError> {
    let LowLevelType::Ptr(ptr) = items_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_join_chars helper expects Ptr(GcArray(Unsigned|Char)), got {items_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_join_chars helper Ptr.TO must be Array, got {:?}",
            ptr.TO
        )));
    };
    match &arr.OF {
        LowLevelType::Unsigned | LowLevelType::Char => Ok(arr.OF.clone()),
        other => Err(TyperError::message(format!(
            "ll_join_chars helper array element must be Unsigned or Char, got {other:?}"
        ))),
    }
}

fn string2list_dst_items(
    list_ptr_lltype: &LowLevelType,
    item_lltype: &LowLevelType,
) -> Result<(bool, LowLevelType), TyperError> {
    let LowLevelType::Ptr(ptr) = list_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_string2list helper expects Ptr(list or GcArray), got {list_ptr_lltype:?}"
        )));
    };
    match &ptr.TO {
        PtrTarget::Array(arr) => {
            if &arr.OF != item_lltype {
                return Err(TyperError::message(format!(
                    "ll_string2list helper fixed list stores {:?}, item_lltype is {item_lltype:?}",
                    arr.OF
                )));
            }
            Ok((false, list_ptr_lltype.clone()))
        }
        PtrTarget::Struct(st) => {
            let items = st._flds.get("items").cloned().ok_or_else(|| {
                TyperError::message("ll_string2list helper resized list has no items field")
            })?;
            Ok((true, items))
        }
        other => Err(TyperError::message(format!(
            "ll_string2list helper Ptr.TO must be Array or Struct, got {other:?}"
        ))),
    }
}

/// Function pointer whose `_name` is the CallPath `lib.rs` marks
/// `jit.isvirtual`. Spine B splits that name on `::` so
/// `rewrite_op_direct_call` emits `ref_isvirtual`.
fn isvirtual_intrinsic_funcptr(ptr_lltype: LowLevelType) -> Constant {
    let ft = FuncType {
        args: vec![ptr_lltype],
        result: LowLevelType::Bool,
    };
    let ptr = functionptr(
        ft.clone(),
        "majit_rlib::jit::isvirtual",
        None,
        Some("majit_rlib::jit::isvirtual".to_string()),
    );
    Constant::with_concretetype(
        ConstValue::LLPtr(Box::new(ptr)),
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Func(ft),
        })),
    )
}

/// Synthesise `LLHelpers.ll_join_chars` (`lltypesystem/rstr.py`):
///
/// ```python
/// @jit.look_inside_iff(lambda length, chars, RES: jit.isconstant(length) and jit.isvirtual(chars))
/// def ll_join_chars(length, chars, RES):
///     result = mallocstr(length)
///     res_chars = result.chars
///     i = 0
///     while i < length:
///         res_chars[i] = cast_primitive(Char, chars[i])
///         i += 1
///     return result
/// ```
///
/// This graph is the original loop. `look_inside_iff` (`rlib/jit.py`)
/// wraps it: `unroll_safe(func)` then `specialize:call_location`.
/// `mallocstr` is inlined as `malloc_varsize(STR)` plus `setfield(hash, 0)`.
/// `cast_primitive(Char, Unsigned)` is `cast_uint_to_int` then
/// `cast_int_to_char`; a `Char` item needs no cast.
pub(crate) fn build_ll_join_chars_helper_graph(
    name: &str,
    items_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let item_lltype = join_chars_item_lltype(&items_ptr_lltype)?;
    let dst_chars_ptr_lltype = chars_array_ptr_lltype_from_strptr(&STRPTR)?;
    let str_struct_lltype = struct_lltype_from_strptr(&STRPTR)?;

    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);
    let hash_field_const =
        || constant_with_lltype(ConstValue::byte_str("hash"), LowLevelType::Void);

    let length_arg = variable_with_lltype("length", LowLevelType::Signed);
    let chars_arg = variable_with_lltype("chars", items_ptr_lltype.clone());
    let res_arg = variable_with_lltype("RES", LowLevelType::Void);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(length_arg.clone()),
        Hlvalue::Variable(chars_arg.clone()),
        Hlvalue::Variable(res_arg),
    ]);
    let return_var = variable_with_lltype("result", STRPTR.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let chars_for_cond = variable_with_lltype("chars", items_ptr_lltype.clone());
    let result_for_cond = variable_with_lltype("result", STRPTR.clone());
    let res_chars_for_cond = variable_with_lltype("res_chars", dst_chars_ptr_lltype.clone());
    let length_for_cond = variable_with_lltype("length", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars_for_cond.clone()),
        Hlvalue::Variable(result_for_cond.clone()),
        Hlvalue::Variable(res_chars_for_cond.clone()),
        Hlvalue::Variable(length_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    let chars_for_body = variable_with_lltype("chars", items_ptr_lltype);
    let result_for_body = variable_with_lltype("result", STRPTR.clone());
    let res_chars_for_body = variable_with_lltype("res_chars", dst_chars_ptr_lltype.clone());
    let length_for_body = variable_with_lltype("length", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let block_body = Block::shared(vec![
        Hlvalue::Variable(chars_for_body.clone()),
        Hlvalue::Variable(result_for_body.clone()),
        Hlvalue::Variable(res_chars_for_body.clone()),
        Hlvalue::Variable(length_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
    ]);

    let result = variable_with_lltype("result", STRPTR.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(str_struct_lltype),
            gc_flavor_const()?,
            Hlvalue::Variable(length_arg.clone()),
        ],
        Hlvalue::Variable(result.clone()),
    ));
    let set_hash = variable_with_lltype("set_hash", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(result.clone()),
            hash_field_const(),
            signed_const(0),
        ],
        Hlvalue::Variable(set_hash),
    ));
    let res_chars = variable_with_lltype("res_chars", dst_chars_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(result.clone()), chars_field_const()],
        Hlvalue::Variable(res_chars.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_arg),
                Hlvalue::Variable(result),
                Hlvalue::Variable(res_chars),
                Hlvalue::Variable(length_arg),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let keep_going = variable_with_lltype("keep_going", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(length_for_cond.clone()),
            ],
            Hlvalue::Variable(keep_going.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(keep_going));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_cond),
                Hlvalue::Variable(result_for_cond.clone()),
                Hlvalue::Variable(res_chars_for_cond),
                Hlvalue::Variable(length_for_cond),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(result_for_cond)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let stored = match item_lltype {
        LowLevelType::Unsigned => {
            let v = variable_with_lltype("v", LowLevelType::Unsigned);
            block_body.borrow_mut().operations.push(SpaceOperation::new(
                "getarrayitem",
                vec![
                    Hlvalue::Variable(chars_for_body.clone()),
                    Hlvalue::Variable(i_for_body.clone()),
                ],
                Hlvalue::Variable(v.clone()),
            ));
            let t = variable_with_lltype("t", LowLevelType::Signed);
            block_body.borrow_mut().operations.push(SpaceOperation::new(
                "cast_uint_to_int",
                vec![Hlvalue::Variable(v)],
                Hlvalue::Variable(t.clone()),
            ));
            let c = variable_with_lltype("c", LowLevelType::Char);
            block_body.borrow_mut().operations.push(SpaceOperation::new(
                "cast_int_to_char",
                vec![Hlvalue::Variable(t)],
                Hlvalue::Variable(c.clone()),
            ));
            c
        }
        LowLevelType::Char => {
            let c = variable_with_lltype("c", LowLevelType::Char);
            block_body.borrow_mut().operations.push(SpaceOperation::new(
                "getarrayitem",
                vec![
                    Hlvalue::Variable(chars_for_body.clone()),
                    Hlvalue::Variable(i_for_body.clone()),
                ],
                Hlvalue::Variable(c.clone()),
            ));
            c
        }
        _ => unreachable!("join_chars_item_lltype rejects other elements"),
    };
    let void_set = variable_with_lltype("set", LowLevelType::Void);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(res_chars_for_body.clone()),
            Hlvalue::Variable(i_for_body.clone()),
            Hlvalue::Variable(stored),
        ],
        Hlvalue::Variable(void_set),
    ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(i_for_body), signed_const(1)],
        Hlvalue::Variable(i_next.clone()),
    ));
    block_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_body),
                Hlvalue::Variable(result_for_body),
                Hlvalue::Variable(res_chars_for_body),
                Hlvalue::Variable(length_for_body),
                Hlvalue::Variable(i_next),
            ],
            Some(block_loop_cond),
            None,
        )
        .into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `look_inside_iff` (`rlib/jit.py`): `unroll_safe(func)` then
    // `func._annspecialcase_ = "specialize:call_location"`.
    func._jit_unroll_safe_ = true;
    func.annspecialcase = Some("specialize:call_location".to_string());
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["length".to_string(), "chars".to_string(), "RES".to_string()],
        func,
    ))
}

/// `look_inside_iff`'s `@dont_look_inside` trampoline: one `direct_call`
/// of the original `ll_join_chars` loop (`rlib/jit.py` `look_inside_iff`).
pub(crate) fn build_ll_join_chars_trampoline_helper_graph(
    name: &str,
    items_ptr_lltype: LowLevelType,
    orig_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let length_arg = variable_with_lltype("length", LowLevelType::Signed);
    let chars_arg = variable_with_lltype("chars", items_ptr_lltype);
    let res_arg = variable_with_lltype("RES", LowLevelType::Void);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(length_arg.clone()),
        Hlvalue::Variable(chars_arg.clone()),
        Hlvalue::Variable(res_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", STRPTR.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let result = variable_with_lltype("result", STRPTR.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(orig_fn),
            Hlvalue::Variable(length_arg),
            Hlvalue::Variable(chars_arg),
            Hlvalue::Variable(res_arg),
        ],
        Hlvalue::Variable(result.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);
    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `@dont_look_inside` (`rlib/jit.py` `look_inside_iff`); the trampoline
    // also receives `specialize:call_location`.
    func._jit_look_inside_ = Some(false);
    func.annspecialcase = Some("specialize:call_location".to_string());
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["length".to_string(), "chars".to_string(), "RES".to_string()],
        func,
    ))
}

/// `rlib/jit.py isvirtual`: the interpreter fallback is
/// `NonConstant(False)` inside the marked intrinsic; the traced path is
/// the oopspec `jit.isvirtual`, rewritten to `ref_isvirtual`. The
/// argument is the chars array pointer (`LIST.ll_items`), matching
/// `LLHelpers.ll_join_chars`.
pub(crate) fn build_ll_join_chars_isvirtual_helper_graph(
    name: &str,
    items_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let value = variable_with_lltype("value", items_ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(value.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let result = variable_with_lltype("result", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(isvirtual_intrinsic_funcptr(items_ptr_lltype)),
            Hlvalue::Variable(value),
        ],
        Hlvalue::Variable(result.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);
    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `rlib/jit.py isvirtual` is `@specialize.call_location()`.
    func.annspecialcase = Some("specialize:call_location".to_string());
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["value".to_string()],
        func,
    ))
}

/// `rlib/jit.py look_inside_iff` wrapper around `ll_join_chars`:
///
/// ```python
/// def f(length, chars, RES):
///     if not we_are_jitted() or (isconstant(length) and isvirtual(chars)):
///         return orig(length, chars, RES)
///     else:
///         return trampoline(length, chars, RES)
/// ```
///
/// `we_are_jitted` is the `SpecTag` exitswitch (true link first, matching
/// `build_ll_alloc_and_set_look_inside_iff_helper_graph`). The predicate
/// short-circuits: `isconstant` false goes to the trampoline, true checks
/// `isvirtual(chars)`.
pub(crate) fn build_ll_join_chars_look_inside_iff_helper_graph(
    name: &str,
    items_ptr_lltype: LowLevelType,
    orig_fn: Constant,
    trampoline_fn: Constant,
    isconstant_fn: Constant,
    isvirtual_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let length_arg = variable_with_lltype("length", LowLevelType::Signed);
    let chars_arg = variable_with_lltype("chars", items_ptr_lltype.clone());
    let res_arg = variable_with_lltype("RES", LowLevelType::Void);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(length_arg.clone()),
        Hlvalue::Variable(chars_arg.clone()),
        Hlvalue::Variable(res_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", STRPTR.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);

    let orig_length = variable_with_lltype("length", LowLevelType::Signed);
    let orig_chars = variable_with_lltype("chars", items_ptr_lltype.clone());
    let orig_res = variable_with_lltype("RES", LowLevelType::Void);
    let block_orig = Block::shared(vec![
        Hlvalue::Variable(orig_length.clone()),
        Hlvalue::Variable(orig_chars.clone()),
        Hlvalue::Variable(orig_res.clone()),
    ]);
    let orig_result = variable_with_lltype("result", STRPTR.clone());
    block_orig.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(orig_fn),
            Hlvalue::Variable(orig_length),
            Hlvalue::Variable(orig_chars),
            Hlvalue::Variable(orig_res),
        ],
        Hlvalue::Variable(orig_result.clone()),
    ));
    block_orig.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(orig_result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let tr_length = variable_with_lltype("length", LowLevelType::Signed);
    let tr_chars = variable_with_lltype("chars", items_ptr_lltype.clone());
    let tr_res = variable_with_lltype("RES", LowLevelType::Void);
    let block_tramp = Block::shared(vec![
        Hlvalue::Variable(tr_length.clone()),
        Hlvalue::Variable(tr_chars.clone()),
        Hlvalue::Variable(tr_res.clone()),
    ]);
    let tr_result = variable_with_lltype("result", STRPTR.clone());
    block_tramp
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(trampoline_fn),
                Hlvalue::Variable(tr_length),
                Hlvalue::Variable(tr_chars),
                Hlvalue::Variable(tr_res),
            ],
            Hlvalue::Variable(tr_result.clone()),
        ));
    block_tramp.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(tr_result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let virt_length = variable_with_lltype("length", LowLevelType::Signed);
    let virt_chars = variable_with_lltype("chars", items_ptr_lltype.clone());
    let virt_res = variable_with_lltype("RES", LowLevelType::Void);
    let block_isvirtual = Block::shared(vec![
        Hlvalue::Variable(virt_length.clone()),
        Hlvalue::Variable(virt_chars.clone()),
        Hlvalue::Variable(virt_res.clone()),
    ]);
    let is_virt = variable_with_lltype("is_virtual", LowLevelType::Bool);
    block_isvirtual
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(isvirtual_fn),
                Hlvalue::Variable(virt_chars.clone()),
            ],
            Hlvalue::Variable(is_virt.clone()),
        ));
    block_isvirtual.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_virt));
    block_isvirtual.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(virt_length.clone()),
                Hlvalue::Variable(virt_chars.clone()),
                Hlvalue::Variable(virt_res.clone()),
            ],
            Some(block_orig.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(virt_length),
                Hlvalue::Variable(virt_chars),
                Hlvalue::Variable(virt_res),
            ],
            Some(block_tramp.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let const_length = variable_with_lltype("length", LowLevelType::Signed);
    let const_chars = variable_with_lltype("chars", items_ptr_lltype);
    let const_res = variable_with_lltype("RES", LowLevelType::Void);
    let block_isconstant = Block::shared(vec![
        Hlvalue::Variable(const_length.clone()),
        Hlvalue::Variable(const_chars.clone()),
        Hlvalue::Variable(const_res.clone()),
    ]);
    let is_const = variable_with_lltype("is_constant", LowLevelType::Bool);
    block_isconstant
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(isconstant_fn),
                Hlvalue::Variable(const_length.clone()),
            ],
            Hlvalue::Variable(is_const.clone()),
        ));
    block_isconstant.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_const));
    block_isconstant.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(const_length.clone()),
                Hlvalue::Variable(const_chars.clone()),
                Hlvalue::Variable(const_res.clone()),
            ],
            Some(block_isvirtual),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(const_length),
                Hlvalue::Variable(const_chars),
                Hlvalue::Variable(const_res),
            ],
            Some(block_tramp),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    startblock.borrow_mut().exitswitch = Some(Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID),
        LowLevelType::Bool,
    )));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(length_arg.clone()),
                Hlvalue::Variable(chars_arg.clone()),
                Hlvalue::Variable(res_arg.clone()),
            ],
            Some(block_isconstant),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(length_arg),
                Hlvalue::Variable(chars_arg),
                Hlvalue::Variable(res_arg),
            ],
            Some(block_orig),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["length".to_string(), "chars".to_string(), "RES".to_string()],
        func,
    ))
}

/// Synthesise `LLHelpers.ll_string2list` (`lltypesystem/rstr.py`):
///
/// ```python
/// @jit.dont_look_inside
/// def ll_string2list(RESLIST, src):
///     length = len(src.chars)
///     lst = RESLIST.ll_newlist(length)
///     dst = lst.ll_items()
///     # copy src.chars[i] into dst[i]
///     return lst
/// ```
///
/// `RESLIST.ll_newlist` is the `newlist_fn` callee, the same way the
/// alloc_and_set builders take `newlist_fn`. `DST.OF` of a `Vec<u8>`
/// list is `Unsigned`, which is not `SRC.chars.OF` (`Char`), so the
/// copy is a per-element loop with `cast_char_to_int` then
/// `cast_int_to_uint`. A `Char` item list needs no cast. No
/// `raw_memcopy`. `_jit_look_inside_ = False` (`dont_look_inside`);
/// there is no trampoline.
pub(crate) fn build_ll_string2list_helper_graph(
    name: &str,
    list_ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    newlist_fn: Constant,
) -> Result<PyGraph, TyperError> {
    if !matches!(item_lltype, LowLevelType::Unsigned | LowLevelType::Char) {
        return Err(TyperError::message(format!(
            "ll_string2list helper item type must be Unsigned or Char, got {item_lltype:?}"
        )));
    }
    let (resized, items_ptr_lltype) = string2list_dst_items(&list_ptr_lltype, &item_lltype)?;
    let src_chars_ptr_lltype = chars_array_ptr_lltype_from_strptr(&STRPTR)?;

    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);
    let items_field_const =
        || constant_with_lltype(ConstValue::byte_str("items"), LowLevelType::Void);

    let reslist_arg = variable_with_lltype("RESLIST", LowLevelType::Void);
    let src_arg = variable_with_lltype("src", STRPTR.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(reslist_arg),
        Hlvalue::Variable(src_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", list_ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let src_chars_for_cond = variable_with_lltype("chars", src_chars_ptr_lltype.clone());
    let lst_for_cond = variable_with_lltype("lst", list_ptr_lltype.clone());
    let dst_for_cond = variable_with_lltype("dst", items_ptr_lltype.clone());
    let length_for_cond = variable_with_lltype("length", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(src_chars_for_cond.clone()),
        Hlvalue::Variable(lst_for_cond.clone()),
        Hlvalue::Variable(dst_for_cond.clone()),
        Hlvalue::Variable(length_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    let src_chars_for_body = variable_with_lltype("chars", src_chars_ptr_lltype.clone());
    let lst_for_body = variable_with_lltype("lst", list_ptr_lltype.clone());
    let dst_for_body = variable_with_lltype("dst", items_ptr_lltype.clone());
    let length_for_body = variable_with_lltype("length", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let block_body = Block::shared(vec![
        Hlvalue::Variable(src_chars_for_body.clone()),
        Hlvalue::Variable(lst_for_body.clone()),
        Hlvalue::Variable(dst_for_body.clone()),
        Hlvalue::Variable(length_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
    ]);

    let src_chars = variable_with_lltype("chars", src_chars_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(src_arg), chars_field_const()],
        Hlvalue::Variable(src_chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(src_chars.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let lst = variable_with_lltype("lst", list_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(newlist_fn),
            Hlvalue::Variable(length.clone()),
        ],
        Hlvalue::Variable(lst.clone()),
    ));
    let dst = if resized {
        let items = variable_with_lltype("dst", items_ptr_lltype);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(lst.clone()), items_field_const()],
            Hlvalue::Variable(items.clone()),
        ));
        items
    } else {
        lst.clone()
    };
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_chars),
                Hlvalue::Variable(lst),
                Hlvalue::Variable(dst),
                Hlvalue::Variable(length),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let keep_going = variable_with_lltype("keep_going", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(length_for_cond.clone()),
            ],
            Hlvalue::Variable(keep_going.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(keep_going));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_chars_for_cond),
                Hlvalue::Variable(lst_for_cond.clone()),
                Hlvalue::Variable(dst_for_cond),
                Hlvalue::Variable(length_for_cond),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(lst_for_cond)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let stored = match item_lltype {
        LowLevelType::Unsigned => {
            let c = variable_with_lltype("c", LowLevelType::Char);
            block_body.borrow_mut().operations.push(SpaceOperation::new(
                "getarrayitem",
                vec![
                    Hlvalue::Variable(src_chars_for_body.clone()),
                    Hlvalue::Variable(i_for_body.clone()),
                ],
                Hlvalue::Variable(c.clone()),
            ));
            let t = variable_with_lltype("t", LowLevelType::Signed);
            block_body.borrow_mut().operations.push(SpaceOperation::new(
                "cast_char_to_int",
                vec![Hlvalue::Variable(c)],
                Hlvalue::Variable(t.clone()),
            ));
            let v = variable_with_lltype("v", LowLevelType::Unsigned);
            block_body.borrow_mut().operations.push(SpaceOperation::new(
                "cast_int_to_uint",
                vec![Hlvalue::Variable(t)],
                Hlvalue::Variable(v.clone()),
            ));
            v
        }
        LowLevelType::Char => {
            let c = variable_with_lltype("c", LowLevelType::Char);
            block_body.borrow_mut().operations.push(SpaceOperation::new(
                "getarrayitem",
                vec![
                    Hlvalue::Variable(src_chars_for_body.clone()),
                    Hlvalue::Variable(i_for_body.clone()),
                ],
                Hlvalue::Variable(c.clone()),
            ));
            c
        }
        _ => unreachable!("item_lltype already validated"),
    };
    let void_set = variable_with_lltype("set", LowLevelType::Void);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(dst_for_body.clone()),
            Hlvalue::Variable(i_for_body.clone()),
            Hlvalue::Variable(stored),
        ],
        Hlvalue::Variable(void_set),
    ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(i_for_body), signed_const(1)],
        Hlvalue::Variable(i_next.clone()),
    ));
    block_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_chars_for_body),
                Hlvalue::Variable(lst_for_body),
                Hlvalue::Variable(dst_for_body),
                Hlvalue::Variable(length_for_body),
                Hlvalue::Variable(i_next),
            ],
            Some(block_loop_cond),
            None,
        )
        .into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `@jit.dont_look_inside` (`rlib/jit.py`): `func._jit_look_inside_ = False`.
    func._jit_look_inside_ = Some(false);
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["RESLIST".to_string(), "src".to_string()],
        func,
    ))
}

/// Synthesise the lltypesystem half of
/// `AbstractUniCharRepr.ll_str(ch) -> str(unicode(ch))`
/// (`rtyper/rstr.py`).
///
/// Upstream goes through the default unicode-to-byte-string encoding
/// path. For a single `UniChar`, that is an ASCII check followed by a
/// one-byte `STR` allocation; non-ASCII codepoints raise
/// `UnicodeEncodeError`.
pub(crate) fn build_ll_unichr2str_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let dst_chars_ptr_lltype = chars_array_ptr_lltype_from_strptr(&STRPTR)?;
    let str_struct_lltype = struct_lltype_from_strptr(&STRPTR)?;

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);
    let exc_args = exception_args("UnicodeEncodeError")?;

    let ch = variable_with_lltype("ch", LowLevelType::UniChar);
    let startblock = Block::shared(vec![Hlvalue::Variable(ch.clone())]);
    let return_var = variable_with_lltype("result", STRPTR.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let code_for_alloc = variable_with_lltype("code", LowLevelType::Signed);
    let block_alloc = Block::shared(vec![Hlvalue::Variable(code_for_alloc.clone())]);

    let code = variable_with_lltype("code", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_unichar_to_int",
        vec![Hlvalue::Variable(ch)],
        Hlvalue::Variable(code.clone()),
    ));
    let too_large = variable_with_lltype("too_large", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_gt",
        vec![Hlvalue::Variable(code.clone()), signed_const(127)],
        Hlvalue::Variable(too_large.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(too_large));
    startblock.closeblock(vec![
        Link::new(exc_args, Some(graph.exceptblock.clone()), Some(bool_true())).into_ref(),
        Link::new(
            vec![Hlvalue::Variable(code)],
            Some(block_alloc.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let byte = variable_with_lltype("byte", LowLevelType::Char);
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_int_to_char",
            vec![Hlvalue::Variable(code_for_alloc)],
            Hlvalue::Variable(byte.clone()),
        ));
    let result = variable_with_lltype("result", STRPTR.clone());
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "malloc_varsize",
            vec![
                lowlevel_type_const(str_struct_lltype),
                gc_flavor_const()?,
                signed_const(1),
            ],
            Hlvalue::Variable(result.clone()),
        ));
    let chars = variable_with_lltype("chars", dst_chars_ptr_lltype);
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(result.clone()), chars_field_const()],
            Hlvalue::Variable(chars.clone()),
        ));
    let void_set = variable_with_lltype("set", LowLevelType::Void);
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(chars),
                signed_const(0),
                Hlvalue::Variable(byte),
            ],
            Hlvalue::Variable(void_set),
        ));
    block_alloc.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ch".to_string()],
        func,
    ))
}

/// Build a standalone `Ptr(GcArray(Char))` lltype — the `CHAR_ARRAY =
/// GcArray(Char)` scratch buffer and the `hex_chars` digit table both
/// have this shape (`lltypesystem/ll_str.py`).
fn char_gcarray_ptr_lltype() -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::new(LowLevelType::Char)),
    }))
}

/// Materialise the immortal `hex_chars = malloc(Array(Char), 16,
/// immortal=True)` table whose entries are `"%x" % i`
/// (`lltypesystem/ll_str.py`).
fn hex_chars_constant() -> Result<Hlvalue, TyperError> {
    let array_lltype = LowLevelType::Array(Box::new(Array::new(LowLevelType::Char)));
    let ptr = malloc(array_lltype, Some(16), MallocFlavor::Gc, true)
        .map_err(|e| TyperError::message(format!("hex_chars malloc: {e}")))?;
    let Ok(Some(_ptr_obj::Array(arr))) = ptr._obj0_value() else {
        return Err(TyperError::message(
            "hex_chars malloc did not produce an Array container".to_string(),
        ));
    };
    for i in 0..16u32 {
        let digit = std::char::from_digit(i, 16).expect("from_digit(0..16, 16) is always Some");
        arr.setitem(i as usize, LowLevelValue::Char(digit));
    }
    Ok(constant_with_lltype(
        ConstValue::LLPtr(Box::new(ptr)),
        char_gcarray_ptr_lltype(),
    ))
}

/// Synthesise `ll_int2hex` (`lltypesystem/ll_str.py`):
///
/// ```python
/// @jit.elidable
/// def ll_int2hex(i, addPrefix):
///     temp = malloc(CHAR_ARRAY, 20)
///     len = 0
///     sign = 0
///     if i < 0:
///         sign = 1
///         i = ll_unsigned(-i)
///     else:
///         i = ll_unsigned(i)
///     if i == 0:
///         len = 1
///         temp[0] = '0'
///     else:
///         while i:
///             temp[len] = hex_chars[i & 0xf]
///             i >>= 4
///             len += 1
///     len += sign
///     if addPrefix:
///         len += 2
///     result = mallocstr(len)
///     result.hash = 0
///     j = 0
///     if sign:
///         result.chars[0] = '-'
///         j = 1
///     if addPrefix:
///         result.chars[j] = '0'
///         result.chars[j+1] = 'x'
///         j += 2
///     while j < len:
///         result.chars[j] = temp[len-j-1]
///         j += 1
///     return result
/// ```
///
/// `i` is reduced to `r_uint` so `i & 0xf` / `i >>= 4` are the unsigned
/// `uint_and` / `uint_rshift`; literal digit/prefix/sign chars are produced
/// with `cast_int_to_char` of their ASCII codes. `signed_input` selects the
/// specialisation: `true` materialises the Signed one (full `if i < 0` sign
/// branch, ll_str.py); `false` the Unsigned one (`r_uint` input, sign
/// branch pruned) used by the `raddress.py` address path.
pub(crate) fn build_ll_int2hex_helper_graph(
    name: &str,
    signed_input: bool,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let str_struct_lltype = struct_lltype_from_strptr(&STRPTR)?;
    let result_chars_ptr_lltype = chars_array_ptr_lltype_from_strptr(&STRPTR)?;
    let temp_ptr_lltype = char_gcarray_ptr_lltype();
    let char_array_lltype = LowLevelType::Array(Box::new(Array::new(LowLevelType::Char)));
    let c_hex_chars = hex_chars_constant()?;

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let unsigned_const =
        |n: i64| constant_with_lltype(ConstValue::UInt(n as u64), LowLevelType::Unsigned);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);
    let hash_field_const =
        || constant_with_lltype(ConstValue::byte_str("hash"), LowLevelType::Void);
    let char_for = |code: i64, block: &crate::flowspace::model::BlockRef| -> Hlvalue {
        let ch = variable_with_lltype("ch", LowLevelType::Char);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "cast_int_to_char",
            vec![signed_const(code)],
            Hlvalue::Variable(ch.clone()),
        ));
        Hlvalue::Variable(ch)
    };

    // ll_str.py `ll_int2hex(i, addPrefix)` is specialised by the type
    // of `i`. `raddress.py` calls `ll_int2hex(r_uint(id), True)`, so the
    // address path needs the `r_uint` (Unsigned) specialisation, in which
    // `if i < 0` is provably false and the whole sign branch is pruned.
    // `signed_input=true` materialises the Signed specialisation (full sign
    // branch, ll_str.py); `false` materialises the Unsigned one.
    let i_lltype = if signed_input {
        LowLevelType::Signed
    } else {
        LowLevelType::Unsigned
    };
    let i_arg = variable_with_lltype("i", i_lltype.clone());
    let add_prefix_arg = variable_with_lltype("addPrefix", LowLevelType::Bool);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(i_arg.clone()),
        Hlvalue::Variable(add_prefix_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", STRPTR.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let mk_block = |spec: Vec<(&str, LowLevelType)>| {
        Block::shared(
            spec.into_iter()
                .map(|(n, lltype)| Hlvalue::Variable(variable_with_lltype(n, lltype)))
                .collect(),
        )
    };
    // The sign branch (b_sign / b_pos) exists only in the Signed
    // specialisation; the Unsigned specialisation links startblock
    // straight to b_zero with sign=0.
    let b_sign = signed_input.then(|| {
        mk_block(vec![
            ("i", LowLevelType::Signed),
            ("addPrefix", LowLevelType::Bool),
            ("temp", temp_ptr_lltype.clone()),
        ])
    });
    let b_pos = signed_input.then(|| {
        mk_block(vec![
            ("i", LowLevelType::Signed),
            ("addPrefix", LowLevelType::Bool),
            ("temp", temp_ptr_lltype.clone()),
        ])
    });
    let b_zero = mk_block(vec![
        ("addPrefix", LowLevelType::Bool),
        ("temp", temp_ptr_lltype.clone()),
        ("i_u", LowLevelType::Unsigned),
        ("sign", LowLevelType::Signed),
    ]);
    let b_iszero = mk_block(vec![
        ("addPrefix", LowLevelType::Bool),
        ("temp", temp_ptr_lltype.clone()),
        ("sign", LowLevelType::Signed),
    ]);
    let b_loop_cond = mk_block(vec![
        ("addPrefix", LowLevelType::Bool),
        ("temp", temp_ptr_lltype.clone()),
        ("i_u", LowLevelType::Unsigned),
        ("sign", LowLevelType::Signed),
        ("len", LowLevelType::Signed),
    ]);
    let b_loop_body = mk_block(vec![
        ("addPrefix", LowLevelType::Bool),
        ("temp", temp_ptr_lltype.clone()),
        ("i_u", LowLevelType::Unsigned),
        ("sign", LowLevelType::Signed),
        ("len", LowLevelType::Signed),
    ]);
    let b_after_digits = mk_block(vec![
        ("addPrefix", LowLevelType::Bool),
        ("temp", temp_ptr_lltype.clone()),
        ("sign", LowLevelType::Signed),
        ("len", LowLevelType::Signed),
    ]);
    let b_addprefix_len = mk_block(vec![
        ("addPrefix", LowLevelType::Bool),
        ("temp", temp_ptr_lltype.clone()),
        ("sign", LowLevelType::Signed),
        ("len", LowLevelType::Signed),
    ]);
    let b_alloc = mk_block(vec![
        ("addPrefix", LowLevelType::Bool),
        ("temp", temp_ptr_lltype.clone()),
        ("sign", LowLevelType::Signed),
        ("len", LowLevelType::Signed),
    ]);
    let b_write_sign = mk_block(vec![
        ("addPrefix", LowLevelType::Bool),
        ("temp", temp_ptr_lltype.clone()),
        ("result", STRPTR.clone()),
        ("result_chars", result_chars_ptr_lltype.clone()),
        ("len", LowLevelType::Signed),
    ]);
    let b_check_prefix = mk_block(vec![
        ("addPrefix", LowLevelType::Bool),
        ("temp", temp_ptr_lltype.clone()),
        ("result", STRPTR.clone()),
        ("result_chars", result_chars_ptr_lltype.clone()),
        ("len", LowLevelType::Signed),
        ("j", LowLevelType::Signed),
    ]);
    let b_write_prefix = mk_block(vec![
        ("temp", temp_ptr_lltype.clone()),
        ("result", STRPTR.clone()),
        ("result_chars", result_chars_ptr_lltype.clone()),
        ("len", LowLevelType::Signed),
        ("j", LowLevelType::Signed),
    ]);
    let b_copy_cond = mk_block(vec![
        ("temp", temp_ptr_lltype.clone()),
        ("result", STRPTR.clone()),
        ("result_chars", result_chars_ptr_lltype.clone()),
        ("len", LowLevelType::Signed),
        ("j", LowLevelType::Signed),
    ]);
    let b_copy_body = mk_block(vec![
        ("temp", temp_ptr_lltype.clone()),
        ("result", STRPTR.clone()),
        ("result_chars", result_chars_ptr_lltype.clone()),
        ("len", LowLevelType::Signed),
        ("j", LowLevelType::Signed),
    ]);

    // Snapshot each block's input Variables up front, so reading them
    // back at op-emission sites does not re-borrow a block that is
    // already mutably borrowed by the enclosing `operations.push`.
    let mut input_snapshots: std::collections::HashMap<
        *const std::cell::RefCell<crate::flowspace::model::Block>,
        Vec<Variable>,
    > = std::collections::HashMap::new();
    let mut snapshot_blocks: Vec<&crate::flowspace::model::BlockRef> = Vec::new();
    if let Some(b) = &b_sign {
        snapshot_blocks.push(b);
    }
    if let Some(b) = &b_pos {
        snapshot_blocks.push(b);
    }
    snapshot_blocks.extend([
        &b_zero,
        &b_iszero,
        &b_loop_cond,
        &b_loop_body,
        &b_after_digits,
        &b_addprefix_len,
        &b_alloc,
        &b_write_sign,
        &b_check_prefix,
        &b_write_prefix,
        &b_copy_cond,
        &b_copy_body,
    ]);
    for block in snapshot_blocks {
        let vars = block
            .borrow()
            .inputargs
            .iter()
            .map(|h| match h {
                Hlvalue::Variable(v) => v.clone(),
                other => panic!("ll_int2hex block input must be Variable, got {other:?}"),
            })
            .collect();
        input_snapshots.insert(std::rc::Rc::as_ptr(block), vars);
    }
    let var_of = |block: &crate::flowspace::model::BlockRef, idx: usize| -> Variable {
        input_snapshots[&std::rc::Rc::as_ptr(block)][idx].clone()
    };

    // start: temp = malloc(CHAR_ARRAY, 20); branch on i < 0.
    let temp = variable_with_lltype("temp", temp_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(char_array_lltype),
            gc_flavor_const()?,
            signed_const(20),
        ],
        Hlvalue::Variable(temp.clone()),
    ));
    if signed_input {
        let b_sign = b_sign.as_ref().expect("b_sign exists in signed build");
        let b_pos = b_pos.as_ref().expect("b_pos exists in signed build");
        let neg = variable_with_lltype("neg", LowLevelType::Bool);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_lt",
            vec![Hlvalue::Variable(i_arg.clone()), signed_const(0)],
            Hlvalue::Variable(neg.clone()),
        ));
        startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(neg));
        startblock.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(i_arg.clone()),
                    Hlvalue::Variable(add_prefix_arg.clone()),
                    Hlvalue::Variable(temp.clone()),
                ],
                Some(b_sign.clone()),
                Some(bool_true()),
            )
            .into_ref(),
            Link::new(
                vec![
                    Hlvalue::Variable(i_arg.clone()),
                    Hlvalue::Variable(add_prefix_arg.clone()),
                    Hlvalue::Variable(temp.clone()),
                ],
                Some(b_pos.clone()),
                Some(bool_false()),
            )
            .into_ref(),
        ]);

        // b_sign: i < 0 → sign = 1; i = r_uint(-i).
        let ni = variable_with_lltype("ni", LowLevelType::Signed);
        b_sign.borrow_mut().operations.push(SpaceOperation::new(
            "int_neg",
            vec![Hlvalue::Variable(var_of(b_sign, 0))],
            Hlvalue::Variable(ni.clone()),
        ));
        let i_u_sign = variable_with_lltype("i_u", LowLevelType::Unsigned);
        b_sign.borrow_mut().operations.push(SpaceOperation::new(
            "cast_primitive",
            vec![Hlvalue::Variable(ni)],
            Hlvalue::Variable(i_u_sign.clone()),
        ));
        b_sign.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(var_of(b_sign, 1)),
                    Hlvalue::Variable(var_of(b_sign, 2)),
                    Hlvalue::Variable(i_u_sign),
                    signed_const(1),
                ],
                Some(b_zero.clone()),
                None,
            )
            .into_ref(),
        ]);

        // b_pos: i >= 0 → sign = 0; i = r_uint(i).
        let i_u_pos = variable_with_lltype("i_u", LowLevelType::Unsigned);
        b_pos.borrow_mut().operations.push(SpaceOperation::new(
            "cast_primitive",
            vec![Hlvalue::Variable(var_of(b_pos, 0))],
            Hlvalue::Variable(i_u_pos.clone()),
        ));
        b_pos.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(var_of(b_pos, 1)),
                    Hlvalue::Variable(var_of(b_pos, 2)),
                    Hlvalue::Variable(i_u_pos),
                    signed_const(0),
                ],
                Some(b_zero.clone()),
                None,
            )
            .into_ref(),
        ]);
    } else {
        // Unsigned specialisation (`r_uint` input): `if i < 0` is provably
        // false, so the sign branch is pruned — go straight to b_zero with
        // the already-unsigned i and sign=0.
        startblock.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(add_prefix_arg.clone()),
                    Hlvalue::Variable(temp.clone()),
                    Hlvalue::Variable(i_arg.clone()),
                    signed_const(0),
                ],
                Some(b_zero.clone()),
                None,
            )
            .into_ref(),
        ]);
    }

    // b_zero: branch on i == 0 (i.e. not uint_is_true(i)).
    let nz = variable_with_lltype("nz", LowLevelType::Bool);
    b_zero.borrow_mut().operations.push(SpaceOperation::new(
        "uint_is_true",
        vec![Hlvalue::Variable(var_of(&b_zero, 2))],
        Hlvalue::Variable(nz.clone()),
    ));
    b_zero.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz));
    b_zero.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_zero, 0)),
                Hlvalue::Variable(var_of(&b_zero, 1)),
                Hlvalue::Variable(var_of(&b_zero, 2)),
                Hlvalue::Variable(var_of(&b_zero, 3)),
                signed_const(0),
            ],
            Some(b_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_zero, 0)),
                Hlvalue::Variable(var_of(&b_zero, 1)),
                Hlvalue::Variable(var_of(&b_zero, 3)),
            ],
            Some(b_iszero.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // b_iszero: len = 1; temp[0] = '0'.
    let zero_ch = char_for(48, &b_iszero);
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    b_iszero.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(var_of(&b_iszero, 1)),
            signed_const(0),
            zero_ch,
        ],
        Hlvalue::Variable(set_void),
    ));
    b_iszero.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_iszero, 0)),
                Hlvalue::Variable(var_of(&b_iszero, 1)),
                Hlvalue::Variable(var_of(&b_iszero, 2)),
                signed_const(1),
            ],
            Some(b_after_digits.clone()),
            None,
        )
        .into_ref(),
    ]);

    // b_loop_cond: while i: ...
    let keep = variable_with_lltype("keep", LowLevelType::Bool);
    b_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_is_true",
            vec![Hlvalue::Variable(var_of(&b_loop_cond, 2))],
            Hlvalue::Variable(keep.clone()),
        ));
    b_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(keep));
    b_loop_cond.closeblock(vec![
        Link::new(
            (0..5)
                .map(|k| Hlvalue::Variable(var_of(&b_loop_cond, k)))
                .collect(),
            Some(b_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_loop_cond, 0)),
                Hlvalue::Variable(var_of(&b_loop_cond, 1)),
                Hlvalue::Variable(var_of(&b_loop_cond, 3)),
                Hlvalue::Variable(var_of(&b_loop_cond, 4)),
            ],
            Some(b_after_digits.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // b_loop_body: temp[len] = hex_chars[i & 0xf]; i >>= 4; len += 1.
    let nib = variable_with_lltype("nib", LowLevelType::Unsigned);
    b_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_and",
            vec![
                Hlvalue::Variable(var_of(&b_loop_body, 2)),
                unsigned_const(15),
            ],
            Hlvalue::Variable(nib.clone()),
        ));
    let digit = variable_with_lltype("digit", LowLevelType::Char);
    b_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![c_hex_chars, Hlvalue::Variable(nib)],
            Hlvalue::Variable(digit.clone()),
        ));
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    b_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(var_of(&b_loop_body, 1)),
                Hlvalue::Variable(var_of(&b_loop_body, 4)),
                Hlvalue::Variable(digit),
            ],
            Hlvalue::Variable(set_void),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Unsigned);
    b_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_rshift",
            vec![Hlvalue::Variable(var_of(&b_loop_body, 2)), signed_const(4)],
            Hlvalue::Variable(i_next.clone()),
        ));
    let len_next = variable_with_lltype("len_next", LowLevelType::Signed);
    b_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(var_of(&b_loop_body, 4)), signed_const(1)],
            Hlvalue::Variable(len_next.clone()),
        ));
    b_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_loop_body, 0)),
                Hlvalue::Variable(var_of(&b_loop_body, 1)),
                Hlvalue::Variable(i_next),
                Hlvalue::Variable(var_of(&b_loop_body, 3)),
                Hlvalue::Variable(len_next),
            ],
            Some(b_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // b_after_digits: len += sign; branch on addPrefix.
    let len_signed = variable_with_lltype("len1", LowLevelType::Signed);
    b_after_digits
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(var_of(&b_after_digits, 3)),
                Hlvalue::Variable(var_of(&b_after_digits, 2)),
            ],
            Hlvalue::Variable(len_signed.clone()),
        ));
    b_after_digits.borrow_mut().exitswitch = Some(Hlvalue::Variable(var_of(&b_after_digits, 0)));
    b_after_digits.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_after_digits, 0)),
                Hlvalue::Variable(var_of(&b_after_digits, 1)),
                Hlvalue::Variable(var_of(&b_after_digits, 2)),
                Hlvalue::Variable(len_signed.clone()),
            ],
            Some(b_addprefix_len.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_after_digits, 0)),
                Hlvalue::Variable(var_of(&b_after_digits, 1)),
                Hlvalue::Variable(var_of(&b_after_digits, 2)),
                Hlvalue::Variable(len_signed),
            ],
            Some(b_alloc.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // b_addprefix_len: len += 2.
    let len_prefixed = variable_with_lltype("len2", LowLevelType::Signed);
    b_addprefix_len
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(var_of(&b_addprefix_len, 3)),
                signed_const(2),
            ],
            Hlvalue::Variable(len_prefixed.clone()),
        ));
    b_addprefix_len.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_addprefix_len, 0)),
                Hlvalue::Variable(var_of(&b_addprefix_len, 1)),
                Hlvalue::Variable(var_of(&b_addprefix_len, 2)),
                Hlvalue::Variable(len_prefixed),
            ],
            Some(b_alloc.clone()),
            None,
        )
        .into_ref(),
    ]);

    // b_alloc: result = mallocstr(len); result.hash = 0; branch on sign.
    let result = variable_with_lltype("result", STRPTR.clone());
    b_alloc.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(str_struct_lltype),
            gc_flavor_const()?,
            Hlvalue::Variable(var_of(&b_alloc, 3)),
        ],
        Hlvalue::Variable(result.clone()),
    ));
    let set_hash = variable_with_lltype("set", LowLevelType::Void);
    b_alloc.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(result.clone()),
            hash_field_const(),
            signed_const(0),
        ],
        Hlvalue::Variable(set_hash),
    ));
    let result_chars = variable_with_lltype("result_chars", result_chars_ptr_lltype.clone());
    b_alloc.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(result.clone()), chars_field_const()],
        Hlvalue::Variable(result_chars.clone()),
    ));
    let sign_b = variable_with_lltype("sign_b", LowLevelType::Bool);
    b_alloc.borrow_mut().operations.push(SpaceOperation::new(
        "int_is_true",
        vec![Hlvalue::Variable(var_of(&b_alloc, 2))],
        Hlvalue::Variable(sign_b.clone()),
    ));
    b_alloc.borrow_mut().exitswitch = Some(Hlvalue::Variable(sign_b));
    b_alloc.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_alloc, 0)),
                Hlvalue::Variable(var_of(&b_alloc, 1)),
                Hlvalue::Variable(result.clone()),
                Hlvalue::Variable(result_chars.clone()),
                Hlvalue::Variable(var_of(&b_alloc, 3)),
            ],
            Some(b_write_sign.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_alloc, 0)),
                Hlvalue::Variable(var_of(&b_alloc, 1)),
                Hlvalue::Variable(result),
                Hlvalue::Variable(result_chars),
                Hlvalue::Variable(var_of(&b_alloc, 3)),
                signed_const(0),
            ],
            Some(b_check_prefix.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // b_write_sign: result.chars[0] = '-'; j = 1.
    let minus = char_for(45, &b_write_sign);
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    b_write_sign
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(var_of(&b_write_sign, 3)),
                signed_const(0),
                minus,
            ],
            Hlvalue::Variable(set_void),
        ));
    b_write_sign.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_write_sign, 0)),
                Hlvalue::Variable(var_of(&b_write_sign, 1)),
                Hlvalue::Variable(var_of(&b_write_sign, 2)),
                Hlvalue::Variable(var_of(&b_write_sign, 3)),
                Hlvalue::Variable(var_of(&b_write_sign, 4)),
                signed_const(1),
            ],
            Some(b_check_prefix.clone()),
            None,
        )
        .into_ref(),
    ]);

    // b_check_prefix: branch on addPrefix.
    b_check_prefix.borrow_mut().exitswitch = Some(Hlvalue::Variable(var_of(&b_check_prefix, 0)));
    b_check_prefix.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_check_prefix, 1)),
                Hlvalue::Variable(var_of(&b_check_prefix, 2)),
                Hlvalue::Variable(var_of(&b_check_prefix, 3)),
                Hlvalue::Variable(var_of(&b_check_prefix, 4)),
                Hlvalue::Variable(var_of(&b_check_prefix, 5)),
            ],
            Some(b_write_prefix.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_check_prefix, 1)),
                Hlvalue::Variable(var_of(&b_check_prefix, 2)),
                Hlvalue::Variable(var_of(&b_check_prefix, 3)),
                Hlvalue::Variable(var_of(&b_check_prefix, 4)),
                Hlvalue::Variable(var_of(&b_check_prefix, 5)),
            ],
            Some(b_copy_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // b_write_prefix: result.chars[j] = '0'; result.chars[j+1] = 'x'; j += 2.
    let prefix_zero = char_for(48, &b_write_prefix);
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    b_write_prefix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(var_of(&b_write_prefix, 2)),
                Hlvalue::Variable(var_of(&b_write_prefix, 4)),
                prefix_zero,
            ],
            Hlvalue::Variable(set_void),
        ));
    let j_plus1 = variable_with_lltype("j1", LowLevelType::Signed);
    b_write_prefix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(var_of(&b_write_prefix, 4)),
                signed_const(1),
            ],
            Hlvalue::Variable(j_plus1.clone()),
        ));
    let prefix_x = char_for(120, &b_write_prefix);
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    b_write_prefix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(var_of(&b_write_prefix, 2)),
                Hlvalue::Variable(j_plus1),
                prefix_x,
            ],
            Hlvalue::Variable(set_void),
        ));
    let j_plus2 = variable_with_lltype("j2", LowLevelType::Signed);
    b_write_prefix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(var_of(&b_write_prefix, 4)),
                signed_const(2),
            ],
            Hlvalue::Variable(j_plus2.clone()),
        ));
    b_write_prefix.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_write_prefix, 0)),
                Hlvalue::Variable(var_of(&b_write_prefix, 1)),
                Hlvalue::Variable(var_of(&b_write_prefix, 2)),
                Hlvalue::Variable(var_of(&b_write_prefix, 3)),
                Hlvalue::Variable(j_plus2),
            ],
            Some(b_copy_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // b_copy_cond: while j < len.
    let copy_go = variable_with_lltype("copy_go", LowLevelType::Bool);
    b_copy_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(var_of(&b_copy_cond, 4)),
                Hlvalue::Variable(var_of(&b_copy_cond, 3)),
            ],
            Hlvalue::Variable(copy_go.clone()),
        ));
    b_copy_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(copy_go));
    b_copy_cond.closeblock(vec![
        Link::new(
            (0..5)
                .map(|k| Hlvalue::Variable(var_of(&b_copy_cond, k)))
                .collect(),
            Some(b_copy_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(var_of(&b_copy_cond, 1))],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // b_copy_body: result.chars[j] = temp[len-j-1]; j += 1.
    let len_minus_j = variable_with_lltype("lmj", LowLevelType::Signed);
    b_copy_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![
                Hlvalue::Variable(var_of(&b_copy_body, 3)),
                Hlvalue::Variable(var_of(&b_copy_body, 4)),
            ],
            Hlvalue::Variable(len_minus_j.clone()),
        ));
    let src_idx = variable_with_lltype("src_idx", LowLevelType::Signed);
    b_copy_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(len_minus_j), signed_const(1)],
            Hlvalue::Variable(src_idx.clone()),
        ));
    let src_ch = variable_with_lltype("src_ch", LowLevelType::Char);
    b_copy_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(var_of(&b_copy_body, 0)),
                Hlvalue::Variable(src_idx),
            ],
            Hlvalue::Variable(src_ch.clone()),
        ));
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    b_copy_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(var_of(&b_copy_body, 2)),
                Hlvalue::Variable(var_of(&b_copy_body, 4)),
                Hlvalue::Variable(src_ch),
            ],
            Hlvalue::Variable(set_void),
        ));
    let j_inc = variable_with_lltype("j_inc", LowLevelType::Signed);
    b_copy_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(var_of(&b_copy_body, 4)), signed_const(1)],
            Hlvalue::Variable(j_inc.clone()),
        ));
    b_copy_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_copy_body, 0)),
                Hlvalue::Variable(var_of(&b_copy_body, 1)),
                Hlvalue::Variable(var_of(&b_copy_body, 2)),
                Hlvalue::Variable(var_of(&b_copy_body, 3)),
                Hlvalue::Variable(j_inc),
            ],
            Some(b_copy_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["i".to_string(), "addPrefix".to_string()],
        func,
    ))
}

/// Synthesise `IntegerRepr.ll_str` (`rint.py`) = `ll_int2dec(val)`
/// (`ll_str.py`) as a low-level helper graph. Where `ll_int2hex`
/// fills a scratch array forward then copies it reversed, `ll_int2dec`
/// counts the digits first, allocates the exact-length result, and writes
/// each decimal digit straight to its final reversed slot
/// `total_len - j - 1`.
///
/// `signed_input=true` is the Signed specialisation (the `if val < 0`
/// branch takes the magnitude); `false` is the Unsigned (`r_uint`) one,
/// where the sign is provably 0 and that branch is pruned.
pub(crate) fn build_ll_int2dec_helper_graph(
    name: &str,
    signed_input: bool,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let str_struct_lltype = struct_lltype_from_strptr(&STRPTR)?;
    let result_chars_ptr_lltype = chars_array_ptr_lltype_from_strptr(&STRPTR)?;

    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let unsigned_const =
        |n: i64| constant_with_lltype(ConstValue::UInt(n as u64), LowLevelType::Unsigned);
    let hash_field_const =
        || constant_with_lltype(ConstValue::byte_str("hash"), LowLevelType::Void);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);
    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let char_for = |code: i64, block: &crate::flowspace::model::BlockRef| -> Hlvalue {
        let ch = variable_with_lltype("ch", LowLevelType::Char);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "cast_int_to_char",
            vec![signed_const(code)],
            Hlvalue::Variable(ch.clone()),
        ));
        Hlvalue::Variable(ch)
    };

    let i_lltype = if signed_input {
        LowLevelType::Signed
    } else {
        LowLevelType::Unsigned
    };
    let i_arg = variable_with_lltype("i", i_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(i_arg.clone())]);
    let return_var = variable_with_lltype("result", STRPTR.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let mk_block = |spec: Vec<(&str, LowLevelType)>| {
        Block::shared(
            spec.into_iter()
                .map(|(n, lltype)| Hlvalue::Variable(variable_with_lltype(n, lltype)))
                .collect(),
        )
    };

    let b_sign = signed_input.then(|| mk_block(vec![("i", LowLevelType::Signed)]));
    let b_pos = signed_input.then(|| mk_block(vec![("i", LowLevelType::Signed)]));
    let b_count_cond = mk_block(vec![
        ("val", LowLevelType::Unsigned),
        ("sign", LowLevelType::Signed),
        ("i_count", LowLevelType::Unsigned),
        ("len", LowLevelType::Signed),
    ]);
    let b_count_body = mk_block(vec![
        ("val", LowLevelType::Unsigned),
        ("sign", LowLevelType::Signed),
        ("i_count", LowLevelType::Unsigned),
        ("len", LowLevelType::Signed),
    ]);
    let b_total = mk_block(vec![
        ("val", LowLevelType::Unsigned),
        ("sign", LowLevelType::Signed),
        ("len", LowLevelType::Signed),
    ]);
    let b_alloc = mk_block(vec![
        ("val", LowLevelType::Unsigned),
        ("sign", LowLevelType::Signed),
        ("len", LowLevelType::Signed),
        ("zero_add", LowLevelType::Signed),
    ]);
    let chars_slot = || ("result_chars", result_chars_ptr_lltype.clone());
    let b_write_minus = mk_block(vec![
        ("result", STRPTR.clone()),
        chars_slot(),
        ("val", LowLevelType::Unsigned),
        ("len", LowLevelType::Signed),
        ("total_len", LowLevelType::Signed),
    ]);
    let b_check_zero = mk_block(vec![
        ("result", STRPTR.clone()),
        chars_slot(),
        ("val", LowLevelType::Unsigned),
        ("len", LowLevelType::Signed),
        ("total_len", LowLevelType::Signed),
    ]);
    let b_write_zero = mk_block(vec![
        ("result", STRPTR.clone()),
        chars_slot(),
        ("val", LowLevelType::Unsigned),
        ("len", LowLevelType::Signed),
        ("total_len", LowLevelType::Signed),
    ]);
    let b_digit_cond = mk_block(vec![
        ("result", STRPTR.clone()),
        chars_slot(),
        ("val", LowLevelType::Unsigned),
        ("len", LowLevelType::Signed),
        ("total_len", LowLevelType::Signed),
        ("j", LowLevelType::Signed),
    ]);
    let b_digit_body = mk_block(vec![
        ("result", STRPTR.clone()),
        chars_slot(),
        ("val", LowLevelType::Unsigned),
        ("len", LowLevelType::Signed),
        ("total_len", LowLevelType::Signed),
        ("j", LowLevelType::Signed),
    ]);

    // Snapshot each block's input Variables up front (see `ll_int2hex`).
    let mut input_snapshots: std::collections::HashMap<
        *const std::cell::RefCell<crate::flowspace::model::Block>,
        Vec<Variable>,
    > = std::collections::HashMap::new();
    let mut snapshot_blocks: Vec<&crate::flowspace::model::BlockRef> = Vec::new();
    if let Some(b) = &b_sign {
        snapshot_blocks.push(b);
    }
    if let Some(b) = &b_pos {
        snapshot_blocks.push(b);
    }
    snapshot_blocks.extend([
        &b_count_cond,
        &b_count_body,
        &b_total,
        &b_alloc,
        &b_write_minus,
        &b_check_zero,
        &b_write_zero,
        &b_digit_cond,
        &b_digit_body,
    ]);
    for block in snapshot_blocks {
        let vars = block
            .borrow()
            .inputargs
            .iter()
            .map(|h| match h {
                Hlvalue::Variable(v) => v.clone(),
                other => panic!("ll_int2dec block input must be Variable, got {other:?}"),
            })
            .collect();
        input_snapshots.insert(std::rc::Rc::as_ptr(block), vars);
    }
    let var_of = |block: &crate::flowspace::model::BlockRef, idx: usize| -> Variable {
        input_snapshots[&std::rc::Rc::as_ptr(block)][idx].clone()
    };

    // start: compute sign + unsigned magnitude, enter the count loop.
    if signed_input {
        let b_sign = b_sign.as_ref().expect("b_sign exists in signed build");
        let b_pos = b_pos.as_ref().expect("b_pos exists in signed build");
        let neg = variable_with_lltype("neg", LowLevelType::Bool);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_lt",
            vec![Hlvalue::Variable(i_arg.clone()), signed_const(0)],
            Hlvalue::Variable(neg.clone()),
        ));
        startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(neg));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(i_arg.clone())],
                Some(b_sign.clone()),
                Some(bool_true()),
            )
            .into_ref(),
            Link::new(
                vec![Hlvalue::Variable(i_arg.clone())],
                Some(b_pos.clone()),
                Some(bool_false()),
            )
            .into_ref(),
        ]);

        // b_sign: val = r_uint(-i); sign = 1.
        let ni = variable_with_lltype("ni", LowLevelType::Signed);
        b_sign.borrow_mut().operations.push(SpaceOperation::new(
            "int_neg",
            vec![Hlvalue::Variable(var_of(b_sign, 0))],
            Hlvalue::Variable(ni.clone()),
        ));
        let vu = variable_with_lltype("val", LowLevelType::Unsigned);
        b_sign.borrow_mut().operations.push(SpaceOperation::new(
            "cast_primitive",
            vec![Hlvalue::Variable(ni)],
            Hlvalue::Variable(vu.clone()),
        ));
        b_sign.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(vu.clone()),
                    signed_const(1),
                    Hlvalue::Variable(vu),
                    signed_const(0),
                ],
                Some(b_count_cond.clone()),
                None,
            )
            .into_ref(),
        ]);

        // b_pos: val = r_uint(i); sign = 0.
        let vup = variable_with_lltype("val", LowLevelType::Unsigned);
        b_pos.borrow_mut().operations.push(SpaceOperation::new(
            "cast_primitive",
            vec![Hlvalue::Variable(var_of(b_pos, 0))],
            Hlvalue::Variable(vup.clone()),
        ));
        b_pos.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(vup.clone()),
                    signed_const(0),
                    Hlvalue::Variable(vup),
                    signed_const(0),
                ],
                Some(b_count_cond.clone()),
                None,
            )
            .into_ref(),
        ]);
    } else {
        // Unsigned specialisation: i is already r_uint, sign = 0.
        startblock.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(i_arg.clone()),
                    signed_const(0),
                    Hlvalue::Variable(i_arg.clone()),
                    signed_const(0),
                ],
                Some(b_count_cond.clone()),
                None,
            )
            .into_ref(),
        ]);
    }

    // b_count_cond: while i_count != 0.
    let keep = variable_with_lltype("keep", LowLevelType::Bool);
    b_count_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_is_true",
            vec![Hlvalue::Variable(var_of(&b_count_cond, 2))],
            Hlvalue::Variable(keep.clone()),
        ));
    b_count_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(keep));
    b_count_cond.closeblock(vec![
        Link::new(
            (0..4)
                .map(|k| Hlvalue::Variable(var_of(&b_count_cond, k)))
                .collect(),
            Some(b_count_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_count_cond, 0)),
                Hlvalue::Variable(var_of(&b_count_cond, 1)),
                Hlvalue::Variable(var_of(&b_count_cond, 3)),
            ],
            Some(b_total.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // b_count_body: i_count //= 10; len += 1.
    let i_next = variable_with_lltype("i_next", LowLevelType::Unsigned);
    b_count_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_floordiv",
            vec![
                Hlvalue::Variable(var_of(&b_count_body, 2)),
                unsigned_const(10),
            ],
            Hlvalue::Variable(i_next.clone()),
        ));
    let len_next = variable_with_lltype("len_next", LowLevelType::Signed);
    b_count_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(var_of(&b_count_body, 3)), signed_const(1)],
            Hlvalue::Variable(len_next.clone()),
        ));
    b_count_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_count_body, 0)),
                Hlvalue::Variable(var_of(&b_count_body, 1)),
                Hlvalue::Variable(i_next),
                Hlvalue::Variable(len_next),
            ],
            Some(b_count_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // b_total: zero_add = int(val == 0); go to alloc.
    let nz = variable_with_lltype("nz", LowLevelType::Bool);
    b_total.borrow_mut().operations.push(SpaceOperation::new(
        "uint_is_true",
        vec![Hlvalue::Variable(var_of(&b_total, 0))],
        Hlvalue::Variable(nz.clone()),
    ));
    b_total.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz));
    b_total.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_total, 0)),
                Hlvalue::Variable(var_of(&b_total, 1)),
                Hlvalue::Variable(var_of(&b_total, 2)),
                signed_const(0),
            ],
            Some(b_alloc.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_total, 0)),
                Hlvalue::Variable(var_of(&b_total, 1)),
                Hlvalue::Variable(var_of(&b_total, 2)),
                signed_const(1),
            ],
            Some(b_alloc.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // b_alloc: total_len = sign + len + zero_add; result = mallocstr(total_len).
    let s1 = variable_with_lltype("s1", LowLevelType::Signed);
    b_alloc.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![
            Hlvalue::Variable(var_of(&b_alloc, 1)),
            Hlvalue::Variable(var_of(&b_alloc, 2)),
        ],
        Hlvalue::Variable(s1.clone()),
    ));
    let total_len = variable_with_lltype("total_len", LowLevelType::Signed);
    b_alloc.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![
            Hlvalue::Variable(s1),
            Hlvalue::Variable(var_of(&b_alloc, 3)),
        ],
        Hlvalue::Variable(total_len.clone()),
    ));
    let result = variable_with_lltype("result", STRPTR.clone());
    b_alloc.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(str_struct_lltype),
            gc_flavor_const()?,
            Hlvalue::Variable(total_len.clone()),
        ],
        Hlvalue::Variable(result.clone()),
    ));
    let set_hash = variable_with_lltype("set", LowLevelType::Void);
    b_alloc.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(result.clone()),
            hash_field_const(),
            signed_const(0),
        ],
        Hlvalue::Variable(set_hash),
    ));
    let result_chars = variable_with_lltype("result_chars", result_chars_ptr_lltype.clone());
    b_alloc.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(result.clone()), chars_field_const()],
        Hlvalue::Variable(result_chars.clone()),
    ));
    let sign_b = variable_with_lltype("sign_b", LowLevelType::Bool);
    b_alloc.borrow_mut().operations.push(SpaceOperation::new(
        "int_is_true",
        vec![Hlvalue::Variable(var_of(&b_alloc, 1))],
        Hlvalue::Variable(sign_b.clone()),
    ));
    b_alloc.borrow_mut().exitswitch = Some(Hlvalue::Variable(sign_b));
    b_alloc.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(result.clone()),
                Hlvalue::Variable(result_chars.clone()),
                Hlvalue::Variable(var_of(&b_alloc, 0)),
                Hlvalue::Variable(var_of(&b_alloc, 2)),
                Hlvalue::Variable(total_len.clone()),
            ],
            Some(b_write_minus.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(result),
                Hlvalue::Variable(result_chars),
                Hlvalue::Variable(var_of(&b_alloc, 0)),
                Hlvalue::Variable(var_of(&b_alloc, 2)),
                Hlvalue::Variable(total_len),
            ],
            Some(b_check_zero.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // b_write_minus: result.chars[0] = '-'; enter digit loop (j = 0).
    let minus = char_for(45, &b_write_minus);
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    b_write_minus
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(var_of(&b_write_minus, 1)),
                signed_const(0),
                minus,
            ],
            Hlvalue::Variable(set_void),
        ));
    b_write_minus.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_write_minus, 0)),
                Hlvalue::Variable(var_of(&b_write_minus, 1)),
                Hlvalue::Variable(var_of(&b_write_minus, 2)),
                Hlvalue::Variable(var_of(&b_write_minus, 3)),
                Hlvalue::Variable(var_of(&b_write_minus, 4)),
                signed_const(0),
            ],
            Some(b_digit_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // b_check_zero: if val == 0 write '0', else straight to digit loop.
    let nz2 = variable_with_lltype("nz2", LowLevelType::Bool);
    b_check_zero
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_is_true",
            vec![Hlvalue::Variable(var_of(&b_check_zero, 2))],
            Hlvalue::Variable(nz2.clone()),
        ));
    b_check_zero.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz2));
    b_check_zero.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_check_zero, 0)),
                Hlvalue::Variable(var_of(&b_check_zero, 1)),
                Hlvalue::Variable(var_of(&b_check_zero, 2)),
                Hlvalue::Variable(var_of(&b_check_zero, 3)),
                Hlvalue::Variable(var_of(&b_check_zero, 4)),
                signed_const(0),
            ],
            Some(b_digit_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_check_zero, 0)),
                Hlvalue::Variable(var_of(&b_check_zero, 1)),
                Hlvalue::Variable(var_of(&b_check_zero, 2)),
                Hlvalue::Variable(var_of(&b_check_zero, 3)),
                Hlvalue::Variable(var_of(&b_check_zero, 4)),
            ],
            Some(b_write_zero.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // b_write_zero: result.chars[0] = '0'; enter digit loop (len == 0).
    let zero_ch = char_for(48, &b_write_zero);
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    b_write_zero
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(var_of(&b_write_zero, 1)),
                signed_const(0),
                zero_ch,
            ],
            Hlvalue::Variable(set_void),
        ));
    b_write_zero.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_write_zero, 0)),
                Hlvalue::Variable(var_of(&b_write_zero, 1)),
                Hlvalue::Variable(var_of(&b_write_zero, 2)),
                Hlvalue::Variable(var_of(&b_write_zero, 3)),
                Hlvalue::Variable(var_of(&b_write_zero, 4)),
                signed_const(0),
            ],
            Some(b_digit_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // b_digit_cond: while j < len.
    let go = variable_with_lltype("go", LowLevelType::Bool);
    b_digit_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(var_of(&b_digit_cond, 5)),
                Hlvalue::Variable(var_of(&b_digit_cond, 3)),
            ],
            Hlvalue::Variable(go.clone()),
        ));
    b_digit_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(go));
    b_digit_cond.closeblock(vec![
        Link::new(
            (0..6)
                .map(|k| Hlvalue::Variable(var_of(&b_digit_cond, k)))
                .collect(),
            Some(b_digit_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(var_of(&b_digit_cond, 0))],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // b_digit_body: chars[total_len-j-1] = chr(val%10 + '0'); val //= 10; j += 1.
    let t1 = variable_with_lltype("t1", LowLevelType::Signed);
    b_digit_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![
                Hlvalue::Variable(var_of(&b_digit_body, 4)),
                Hlvalue::Variable(var_of(&b_digit_body, 5)),
            ],
            Hlvalue::Variable(t1.clone()),
        ));
    let dst = variable_with_lltype("dst", LowLevelType::Signed);
    b_digit_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(t1), signed_const(1)],
            Hlvalue::Variable(dst.clone()),
        ));
    let rem = variable_with_lltype("rem", LowLevelType::Unsigned);
    b_digit_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_mod",
            vec![
                Hlvalue::Variable(var_of(&b_digit_body, 2)),
                unsigned_const(10),
            ],
            Hlvalue::Variable(rem.clone()),
        ));
    let rem_s = variable_with_lltype("rem_s", LowLevelType::Signed);
    b_digit_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_uint_to_int",
            vec![Hlvalue::Variable(rem)],
            Hlvalue::Variable(rem_s.clone()),
        ));
    let code = variable_with_lltype("code", LowLevelType::Signed);
    b_digit_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(rem_s), signed_const(48)],
            Hlvalue::Variable(code.clone()),
        ));
    let ch = variable_with_lltype("ch", LowLevelType::Char);
    b_digit_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_int_to_char",
            vec![Hlvalue::Variable(code)],
            Hlvalue::Variable(ch.clone()),
        ));
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    b_digit_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(var_of(&b_digit_body, 1)),
                Hlvalue::Variable(dst),
                Hlvalue::Variable(ch),
            ],
            Hlvalue::Variable(set_void),
        ));
    let val_next = variable_with_lltype("val_next", LowLevelType::Unsigned);
    b_digit_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_floordiv",
            vec![
                Hlvalue::Variable(var_of(&b_digit_body, 2)),
                unsigned_const(10),
            ],
            Hlvalue::Variable(val_next.clone()),
        ));
    let j_next = variable_with_lltype("j_next", LowLevelType::Signed);
    b_digit_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(var_of(&b_digit_body, 5)), signed_const(1)],
            Hlvalue::Variable(j_next.clone()),
        ));
    b_digit_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(var_of(&b_digit_body, 0)),
                Hlvalue::Variable(var_of(&b_digit_body, 1)),
                Hlvalue::Variable(val_next),
                Hlvalue::Variable(var_of(&b_digit_body, 3)),
                Hlvalue::Variable(var_of(&b_digit_body, 4)),
                Hlvalue::Variable(j_next),
            ],
            Some(b_digit_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["i".to_string()],
        func,
    ))
}

/// Synthesise `LLHelpers.ll_int` (`lltypesystem/rstr.py`):
///
/// ```python
/// def ll_int(s, base):
///     if not 2 <= base <= 36:
///         raise ValueError
///     chars = s.chars
///     strlen = len(chars)
///     i = 0
///     while i < strlen and ord(chars[i]) == ord(' '):
///         i += 1
///     if not i < strlen:
///         raise ValueError
///     sign = 1
///     if ord(chars[i]) == ord('-'):
///         sign = -1
///         i += 1
///     elif ord(chars[i]) == ord('+'):
///         i += 1
///     while i < strlen and ord(chars[i]) == ord(' '):
///         i += 1
///     val = 0
///     oldpos = i
///     while i < strlen:
///         ...
///     if i == oldpos:
///         raise ValueError
///     while i < strlen and ord(chars[i]) == ord(' '):
///         i += 1
///     if not i == strlen:
///         raise ValueError
///     return sign * val
/// ```
///
/// The graph is intentionally verbose: each Python short-circuit and
/// `break` point is a separate block so the control-flow shape remains
/// close to the upstream loop-and-branch structure.
pub(crate) fn build_ll_int_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_int helper expects Ptr(Array(Char|UniChar)), got {chars_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_int helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let char_lltype = arr.OF.clone();
    let cast_op = match char_lltype {
        LowLevelType::Char => "cast_char_to_int",
        LowLevelType::UniChar => "cast_unichar_to_int",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_int helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);
    let exc_args = exception_args("ValueError")?;

    let s_arg = variable_with_lltype("s", ptr_lltype.clone());
    let base_arg = variable_with_lltype("base", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s_arg.clone()),
        Hlvalue::Variable(base_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let mk_block = |spec: Vec<(&str, LowLevelType)>| {
        Block::shared(
            spec.into_iter()
                .map(|(name, lltype)| Hlvalue::Variable(variable_with_lltype(name, lltype)))
                .collect(),
        )
    };

    let block_base_hi = mk_block(vec![
        ("s", ptr_lltype.clone()),
        ("base", LowLevelType::Signed),
    ]);
    let block_init = mk_block(vec![
        ("s", ptr_lltype.clone()),
        ("base", LowLevelType::Signed),
    ]);
    let block_leading_cond = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
    ]);
    let block_leading_char = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
    ]);
    let block_leading_inc = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
    ]);
    let block_sign_minus = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
    ]);
    let block_sign_plus = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_sign_minus_inc = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
    ]);
    let block_sign_plus_inc = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
    ]);
    let block_after_sign_cond = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
    ]);
    let block_after_sign_char = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
    ]);
    let block_after_sign_inc = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
    ]);
    let block_digit_cond = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
    ]);
    let block_digit_load = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
    ]);
    let block_lower_hi = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_lower_digit = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_upper_lo = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_upper_hi = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_upper_digit = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_num_lo = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_num_hi = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_num_digit = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_digit_base = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("digit", LowLevelType::Signed),
    ]);
    let block_digit_accum = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("digit", LowLevelType::Signed),
    ]);
    let block_empty_check = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
    ]);
    let block_trailing_cond = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
    ]);
    let block_trailing_char = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
    ]);
    let block_trailing_inc = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
    ]);
    let block_return_value = mk_block(vec![
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
    ]);

    let args = |block: &crate::flowspace::model::BlockRef| block.borrow().inputargs.clone();
    let var = |block: &crate::flowspace::model::BlockRef, idx: usize| {
        let Hlvalue::Variable(v) = block.borrow().inputargs[idx].clone() else {
            unreachable!("helper graph block inputargs are variables")
        };
        v
    };

    // start: lower base bound.
    let base_ge2 = variable_with_lltype("base_ge2", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![Hlvalue::Variable(base_arg.clone()), signed_const(2)],
        Hlvalue::Variable(base_ge2.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(base_ge2));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s_arg), Hlvalue::Variable(base_arg)],
            Some(block_base_hi.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let s = var(&block_base_hi, 0);
    let base = var(&block_base_hi, 1);
    let base_le36 = variable_with_lltype("base_le36", LowLevelType::Bool);
    block_base_hi
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_le",
            vec![Hlvalue::Variable(base.clone()), signed_const(36)],
            Hlvalue::Variable(base_le36.clone()),
        ));
    block_base_hi.borrow_mut().exitswitch = Some(Hlvalue::Variable(base_le36));
    block_base_hi.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s), Hlvalue::Variable(base)],
            Some(block_init.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let s = var(&block_init, 0);
    let base = var(&block_init, 1);
    let chars = variable_with_lltype("chars", chars_ptr_lltype.clone());
    block_init.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let strlen = variable_with_lltype("strlen", LowLevelType::Signed);
    block_init.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(strlen.clone()),
    ));
    block_init.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                signed_const(0),
            ],
            Some(block_leading_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let chars = var(&block_leading_cond, 0);
    let strlen = var(&block_leading_cond, 1);
    let base = var(&block_leading_cond, 2);
    let i = var(&block_leading_cond, 3);
    let has_char = variable_with_lltype("has_char", LowLevelType::Bool);
    block_leading_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(strlen.clone()),
            ],
            Hlvalue::Variable(has_char.clone()),
        ));
    block_leading_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(has_char));
    block_leading_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i),
            ],
            Some(block_leading_char.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_leading_char, 0);
    let strlen = var(&block_leading_char, 1);
    let base = var(&block_leading_char, 2);
    let i = var(&block_leading_char, 3);
    let ch = variable_with_lltype("ch", char_lltype.clone());
    block_leading_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Hlvalue::Variable(ch.clone()),
        ));
    let c = variable_with_lltype("c", LowLevelType::Signed);
    block_leading_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(ch)],
            Hlvalue::Variable(c.clone()),
        ));
    let is_space = variable_with_lltype("is_space", LowLevelType::Bool);
    block_leading_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(c), signed_const(32)],
            Hlvalue::Variable(is_space.clone()),
        ));
    block_leading_char.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_space));
    block_leading_char.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(strlen.clone()),
                Hlvalue::Variable(base.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Some(block_leading_inc.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i),
            ],
            Some(block_sign_minus.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_leading_inc, 0);
    let strlen = var(&block_leading_inc, 1);
    let base = var(&block_leading_inc, 2);
    let i = var(&block_leading_inc, 3);
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_leading_inc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_leading_inc.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i_next),
            ],
            Some(block_leading_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let chars = var(&block_sign_minus, 0);
    let strlen = var(&block_sign_minus, 1);
    let base = var(&block_sign_minus, 2);
    let i = var(&block_sign_minus, 3);
    let ch = variable_with_lltype("ch", char_lltype.clone());
    block_sign_minus
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Hlvalue::Variable(ch.clone()),
        ));
    let c = variable_with_lltype("c", LowLevelType::Signed);
    block_sign_minus
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(ch)],
            Hlvalue::Variable(c.clone()),
        ));
    let is_minus = variable_with_lltype("is_minus", LowLevelType::Bool);
    block_sign_minus
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(c.clone()), signed_const(45)],
            Hlvalue::Variable(is_minus.clone()),
        ));
    block_sign_minus.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_minus));
    block_sign_minus.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(strlen.clone()),
                Hlvalue::Variable(base.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Some(block_sign_minus_inc.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i),
                Hlvalue::Variable(c),
            ],
            Some(block_sign_plus.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_sign_plus, 0);
    let strlen = var(&block_sign_plus, 1);
    let base = var(&block_sign_plus, 2);
    let i = var(&block_sign_plus, 3);
    let c = var(&block_sign_plus, 4);
    let is_plus = variable_with_lltype("is_plus", LowLevelType::Bool);
    block_sign_plus
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(c), signed_const(43)],
            Hlvalue::Variable(is_plus.clone()),
        ));
    block_sign_plus.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_plus));
    block_sign_plus.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(strlen.clone()),
                Hlvalue::Variable(base.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Some(block_sign_plus_inc.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i),
                signed_const(1),
            ],
            Some(block_after_sign_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    for (block_inc, sign_value) in [
        (block_sign_minus_inc.clone(), -1),
        (block_sign_plus_inc.clone(), 1),
    ] {
        let chars = var(&block_inc, 0);
        let strlen = var(&block_inc, 1);
        let base = var(&block_inc, 2);
        let i = var(&block_inc, 3);
        let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
        block_inc.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
        block_inc.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(chars),
                    Hlvalue::Variable(strlen),
                    Hlvalue::Variable(base),
                    Hlvalue::Variable(i_next),
                    signed_const(sign_value),
                ],
                Some(block_after_sign_cond.clone()),
                None,
            )
            .into_ref(),
        ]);
    }

    let chars = var(&block_after_sign_cond, 0);
    let strlen = var(&block_after_sign_cond, 1);
    let base = var(&block_after_sign_cond, 2);
    let i = var(&block_after_sign_cond, 3);
    let sign = var(&block_after_sign_cond, 4);
    let has_char = variable_with_lltype("has_char", LowLevelType::Bool);
    block_after_sign_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(strlen.clone()),
            ],
            Hlvalue::Variable(has_char.clone()),
        ));
    block_after_sign_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(has_char));
    block_after_sign_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(strlen.clone()),
                Hlvalue::Variable(base.clone()),
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(sign.clone()),
            ],
            Some(block_after_sign_char.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(sign),
                signed_const(0),
                Hlvalue::Variable(i),
            ],
            Some(block_digit_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_after_sign_char, 0);
    let strlen = var(&block_after_sign_char, 1);
    let base = var(&block_after_sign_char, 2);
    let i = var(&block_after_sign_char, 3);
    let sign = var(&block_after_sign_char, 4);
    let ch = variable_with_lltype("ch", char_lltype.clone());
    block_after_sign_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Hlvalue::Variable(ch.clone()),
        ));
    let c = variable_with_lltype("c", LowLevelType::Signed);
    block_after_sign_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(ch)],
            Hlvalue::Variable(c.clone()),
        ));
    let is_space = variable_with_lltype("is_space", LowLevelType::Bool);
    block_after_sign_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(c), signed_const(32)],
            Hlvalue::Variable(is_space.clone()),
        ));
    block_after_sign_char.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_space));
    block_after_sign_char.closeblock(vec![
        Link::new(
            args(&block_after_sign_char),
            Some(block_after_sign_inc.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(sign),
                signed_const(0),
                Hlvalue::Variable(i),
            ],
            Some(block_digit_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_after_sign_inc, 0);
    let strlen = var(&block_after_sign_inc, 1);
    let base = var(&block_after_sign_inc, 2);
    let i = var(&block_after_sign_inc, 3);
    let sign = var(&block_after_sign_inc, 4);
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_after_sign_inc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_after_sign_inc.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i_next),
                Hlvalue::Variable(sign),
            ],
            Some(block_after_sign_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let chars = var(&block_digit_cond, 0);
    let strlen = var(&block_digit_cond, 1);
    let _base = var(&block_digit_cond, 2);
    let i = var(&block_digit_cond, 3);
    let sign = var(&block_digit_cond, 4);
    let val = var(&block_digit_cond, 5);
    let oldpos = var(&block_digit_cond, 6);
    let has_char = variable_with_lltype("has_char", LowLevelType::Bool);
    block_digit_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(strlen.clone()),
            ],
            Hlvalue::Variable(has_char.clone()),
        ));
    block_digit_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(has_char));
    block_digit_cond.closeblock(vec![
        Link::new(
            args(&block_digit_cond),
            Some(block_digit_load.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
                Hlvalue::Variable(oldpos),
            ],
            Some(block_empty_check.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_digit_load, 0);
    let strlen = var(&block_digit_load, 1);
    let base = var(&block_digit_load, 2);
    let i = var(&block_digit_load, 3);
    let sign = var(&block_digit_load, 4);
    let val = var(&block_digit_load, 5);
    let oldpos = var(&block_digit_load, 6);
    let ch = variable_with_lltype("ch", char_lltype.clone());
    block_digit_load
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Hlvalue::Variable(ch.clone()),
        ));
    let c = variable_with_lltype("c", LowLevelType::Signed);
    block_digit_load
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(ch)],
            Hlvalue::Variable(c.clone()),
        ));
    let ge_a = variable_with_lltype("ge_a", LowLevelType::Bool);
    block_digit_load
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ge",
            vec![Hlvalue::Variable(c.clone()), signed_const(97)],
            Hlvalue::Variable(ge_a.clone()),
        ));
    block_digit_load.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge_a));
    block_digit_load.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(strlen.clone()),
                Hlvalue::Variable(base.clone()),
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(sign.clone()),
                Hlvalue::Variable(val.clone()),
                Hlvalue::Variable(oldpos.clone()),
                Hlvalue::Variable(c.clone()),
            ],
            Some(block_lower_hi.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
                Hlvalue::Variable(oldpos),
                Hlvalue::Variable(c),
            ],
            Some(block_upper_lo.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let close_digit_test =
        |block: &crate::flowspace::model::BlockRef,
         op: &str,
         rhs: i64,
         true_target: crate::flowspace::model::BlockRef,
         false_target: crate::flowspace::model::BlockRef| {
            let c = var(block, 7);
            let ok = variable_with_lltype("ok", LowLevelType::Bool);
            block.borrow_mut().operations.push(SpaceOperation::new(
                op,
                vec![Hlvalue::Variable(c), signed_const(rhs)],
                Hlvalue::Variable(ok.clone()),
            ));
            block.borrow_mut().exitswitch = Some(Hlvalue::Variable(ok));
            block.closeblock(vec![
                Link::new(args(block), Some(true_target), Some(bool_true())).into_ref(),
                Link::new(args(block), Some(false_target), Some(bool_false())).into_ref(),
            ]);
        };

    close_digit_test(
        &block_lower_hi,
        "int_le",
        122,
        block_lower_digit.clone(),
        block_upper_lo.clone(),
    );
    close_digit_test(
        &block_upper_hi,
        "int_le",
        90,
        block_upper_digit.clone(),
        block_num_lo.clone(),
    );

    let chars = var(&block_num_hi, 0);
    let strlen = var(&block_num_hi, 1);
    let i = var(&block_num_hi, 3);
    let sign = var(&block_num_hi, 4);
    let val = var(&block_num_hi, 5);
    let oldpos = var(&block_num_hi, 6);
    let c = var(&block_num_hi, 7);
    let le_9 = variable_with_lltype("le_9", LowLevelType::Bool);
    block_num_hi
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_le",
            vec![Hlvalue::Variable(c), signed_const(57)],
            Hlvalue::Variable(le_9.clone()),
        ));
    block_num_hi.borrow_mut().exitswitch = Some(Hlvalue::Variable(le_9));
    block_num_hi.closeblock(vec![
        Link::new(
            args(&block_num_hi),
            Some(block_num_digit.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
                Hlvalue::Variable(oldpos),
            ],
            Some(block_empty_check.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let close_digit_make = |block: &crate::flowspace::model::BlockRef,
                            sub: i64,
                            add: i64,
                            target: crate::flowspace::model::BlockRef| {
        let c = var(block, 7);
        let digit_base = variable_with_lltype("digit_base", LowLevelType::Signed);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(c), signed_const(sub)],
            Hlvalue::Variable(digit_base.clone()),
        ));
        let digit = variable_with_lltype("digit", LowLevelType::Signed);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(digit_base), signed_const(add)],
            Hlvalue::Variable(digit.clone()),
        ));
        let mut target_args = args(block);
        target_args.pop();
        target_args.push(Hlvalue::Variable(digit));
        block.closeblock(vec![Link::new(target_args, Some(target), None).into_ref()]);
    };

    close_digit_make(&block_lower_digit, 97, 10, block_digit_base.clone());
    close_digit_make(&block_upper_digit, 65, 10, block_digit_base.clone());
    close_digit_make(&block_num_digit, 48, 0, block_digit_base.clone());

    let chars = var(&block_upper_lo, 0);
    let strlen = var(&block_upper_lo, 1);
    let _base = var(&block_upper_lo, 2);
    let i = var(&block_upper_lo, 3);
    let sign = var(&block_upper_lo, 4);
    let val = var(&block_upper_lo, 5);
    let oldpos = var(&block_upper_lo, 6);
    let c = var(&block_upper_lo, 7);
    let ge_a = variable_with_lltype("ge_A", LowLevelType::Bool);
    block_upper_lo
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ge",
            vec![Hlvalue::Variable(c.clone()), signed_const(65)],
            Hlvalue::Variable(ge_a.clone()),
        ));
    block_upper_lo.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge_a));
    block_upper_lo.closeblock(vec![
        Link::new(
            args(&block_upper_lo),
            Some(block_upper_hi.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
                Hlvalue::Variable(oldpos),
            ],
            Some(block_empty_check.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_num_lo, 0);
    let strlen = var(&block_num_lo, 1);
    let _base = var(&block_num_lo, 2);
    let i = var(&block_num_lo, 3);
    let sign = var(&block_num_lo, 4);
    let val = var(&block_num_lo, 5);
    let oldpos = var(&block_num_lo, 6);
    let c = var(&block_num_lo, 7);
    let ge_0 = variable_with_lltype("ge_0", LowLevelType::Bool);
    block_num_lo
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ge",
            vec![Hlvalue::Variable(c), signed_const(48)],
            Hlvalue::Variable(ge_0.clone()),
        ));
    block_num_lo.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge_0));
    block_num_lo.closeblock(vec![
        Link::new(
            args(&block_num_lo),
            Some(block_num_hi.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
                Hlvalue::Variable(oldpos),
            ],
            Some(block_empty_check.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let base = var(&block_digit_base, 2);
    let digit = var(&block_digit_base, 7);
    let too_big = variable_with_lltype("too_big", LowLevelType::Bool);
    block_digit_base
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ge",
            vec![Hlvalue::Variable(digit.clone()), Hlvalue::Variable(base)],
            Hlvalue::Variable(too_big.clone()),
        ));
    block_digit_base.borrow_mut().exitswitch = Some(Hlvalue::Variable(too_big));
    block_digit_base.closeblock(vec![
        Link::new(
            vec![
                args(&block_digit_base)[0].clone(),
                args(&block_digit_base)[1].clone(),
                args(&block_digit_base)[3].clone(),
                args(&block_digit_base)[4].clone(),
                args(&block_digit_base)[5].clone(),
                args(&block_digit_base)[6].clone(),
            ],
            Some(block_empty_check.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            args(&block_digit_base),
            Some(block_digit_accum.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_digit_accum, 0);
    let strlen = var(&block_digit_accum, 1);
    let base = var(&block_digit_accum, 2);
    let i = var(&block_digit_accum, 3);
    let sign = var(&block_digit_accum, 4);
    let val = var(&block_digit_accum, 5);
    let oldpos = var(&block_digit_accum, 6);
    let digit = var(&block_digit_accum, 7);
    let val_mul = variable_with_lltype("val_mul", LowLevelType::Signed);
    block_digit_accum
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_mul",
            vec![Hlvalue::Variable(val), Hlvalue::Variable(base.clone())],
            Hlvalue::Variable(val_mul.clone()),
        ));
    let val_next = variable_with_lltype("val_next", LowLevelType::Signed);
    block_digit_accum
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(val_mul), Hlvalue::Variable(digit)],
            Hlvalue::Variable(val_next.clone()),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_digit_accum
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_digit_accum.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i_next),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val_next),
                Hlvalue::Variable(oldpos),
            ],
            Some(block_digit_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let chars = var(&block_empty_check, 0);
    let strlen = var(&block_empty_check, 1);
    let i = var(&block_empty_check, 2);
    let sign = var(&block_empty_check, 3);
    let val = var(&block_empty_check, 4);
    let oldpos = var(&block_empty_check, 5);
    let no_digits = variable_with_lltype("no_digits", LowLevelType::Bool);
    block_empty_check
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(i.clone()), Hlvalue::Variable(oldpos)],
            Hlvalue::Variable(no_digits.clone()),
        ));
    block_empty_check.borrow_mut().exitswitch = Some(Hlvalue::Variable(no_digits));
    block_empty_check.closeblock(vec![
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
            ],
            Some(block_trailing_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let _chars = var(&block_trailing_cond, 0);
    let strlen = var(&block_trailing_cond, 1);
    let i = var(&block_trailing_cond, 2);
    let sign = var(&block_trailing_cond, 3);
    let val = var(&block_trailing_cond, 4);
    let has_trailing = variable_with_lltype("has_trailing", LowLevelType::Bool);
    block_trailing_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(strlen.clone()),
            ],
            Hlvalue::Variable(has_trailing.clone()),
        ));
    block_trailing_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(has_trailing));
    block_trailing_cond.closeblock(vec![
        Link::new(
            args(&block_trailing_cond),
            Some(block_trailing_char.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(sign), Hlvalue::Variable(val)],
            Some(block_return_value.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_trailing_char, 0);
    let strlen = var(&block_trailing_char, 1);
    let i = var(&block_trailing_char, 2);
    let sign = var(&block_trailing_char, 3);
    let val = var(&block_trailing_char, 4);
    let ch = variable_with_lltype("ch", char_lltype);
    block_trailing_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Hlvalue::Variable(ch.clone()),
        ));
    let c = variable_with_lltype("c", LowLevelType::Signed);
    block_trailing_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(ch)],
            Hlvalue::Variable(c.clone()),
        ));
    let is_space = variable_with_lltype("is_space", LowLevelType::Bool);
    block_trailing_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(c), signed_const(32)],
            Hlvalue::Variable(is_space.clone()),
        ));
    block_trailing_char.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_space));
    block_trailing_char.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
            ],
            Some(block_trailing_inc.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_trailing_inc, 0);
    let strlen = var(&block_trailing_inc, 1);
    let i = var(&block_trailing_inc, 2);
    let sign = var(&block_trailing_inc, 3);
    let val = var(&block_trailing_inc, 4);
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_trailing_inc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_trailing_inc.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i_next),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
            ],
            Some(block_trailing_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let sign = var(&block_return_value, 0);
    let val = var(&block_return_value, 1);
    let result = variable_with_lltype("result", LowLevelType::Signed);
    block_return_value
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_mul",
            vec![Hlvalue::Variable(sign), Hlvalue::Variable(val)],
            Hlvalue::Variable(result.clone()),
        ));
    block_return_value.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "base".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_str_is_true`
/// (`rtyper/rstr.py`):
///
/// ```python
/// @classmethod
/// def ll_str_is_true(cls, s):
///     # check if a string is True, allowing for None
///     return bool(s) and cls.ll_strlen(s) != 0
/// ```
///
/// 2-block CFG that mirrors the source-level short-circuit `and`:
/// - **start**: `v_nz = ptr_nonzero(s)`. Branches on `v_nz` —
///   True → `block_check_len` (link arg `[s]`); False → returnblock
///   carrying `Bool(false)`.
/// - **block_check_len**: `getsubstruct(s, 'chars')` +
///   `getarraysize(...)` + `int_ne(len, 0)`; link to returnblock with
///   the comparison result.
///
/// `name` is the helper identity (`"ll_str_is_true"` /
/// `"ll_unicode_is_true"`) and `ptr_lltype` selects between
/// `Ptr(STR)` and `Ptr(UNICODE)`. The chars-array `Ptr` lltype is
/// derived from the resolved STR/UNICODE struct via
/// [`chars_array_ptr_lltype_from_strptr`], so the synthesised
/// `getsubstruct` op carries a valid `SomePtr`-shaped annotation
/// (mirrors `build_ll_strlen_helper_graph`).
pub(crate) fn build_ll_str_is_true_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let arg = variable_with_lltype("s", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_zero = || constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed);

    // block_check_len inputarg: same `s` ptr passed forward through the
    // True branch link.
    let s_for_len = variable_with_lltype("s", ptr_lltype);
    let block_check_len = Block::shared(vec![Hlvalue::Variable(s_for_len.clone())]);

    // start: ptr_nonzero(s); branch on the result.
    let v_nz = variable_with_lltype("v_nz", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_nonzero",
        vec![Hlvalue::Variable(arg.clone())],
        Hlvalue::Variable(v_nz.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_nz));
    let start_true_link = Link::new(
        vec![Hlvalue::Variable(arg)],
        Some(block_check_len.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let start_false_link = Link::new(
        vec![bool_false()],
        Some(graph.returnblock.clone()),
        Some(bool_false()),
    )
    .into_ref();
    startblock.closeblock(vec![start_true_link, start_false_link]);

    // block_check_len: getsubstruct('chars') + getarraysize +
    // int_ne(len, 0); link to returnblock with the comparison result.
    let v_len = emit_chars_length_ops(
        &block_check_len,
        Hlvalue::Variable(s_for_len),
        chars_array_ptr_lltype,
    );
    let v_result = variable_with_lltype("result", LowLevelType::Bool);
    block_check_len
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ne",
            vec![v_len, signed_zero()],
            Hlvalue::Variable(v_result.clone()),
        ));
    block_check_len.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_str(s)` — the identity
/// `str()` of a `Ptr(STR)`.  RPython `AbstractStringRepr.ll_str` returns
/// the receiver unchanged when non-null, else the prebuilt constant
/// `'None'`, so `str(s)` lowers to a branch with no allocation.
///
/// 1-block CFG plus the returnblock:
/// - **start**: `v_nz = ptr_nonzero(s)`. True → returnblock(`s`),
///   False → returnblock(prebuilt `'None'` `Ptr(STR)` constant).
pub(crate) fn build_ll_str_string_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("s", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);

    // False arm: the prebuilt `'None'` STR container (`ll_constant('None')`).
    let none_ptr = const_str_cache_llstr(b"None").map_err(TyperError::message)?;
    let none_const = constant_with_lltype(ConstValue::LLPtr(Box::new(none_ptr)), ptr_lltype);

    // start: ptr_nonzero(s); branch on the result.
    let v_nz = variable_with_lltype("v_nz", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_nonzero",
        vec![Hlvalue::Variable(arg.clone())],
        Hlvalue::Variable(v_nz.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_nz));
    let start_true_link = Link::new(
        vec![Hlvalue::Variable(arg)],
        Some(graph.returnblock.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let start_false_link = Link::new(
        vec![none_const],
        Some(graph.returnblock.clone()),
        Some(bool_false()),
    )
    .into_ref();
    startblock.closeblock(vec![start_true_link, start_false_link]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_streq`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// def ll_streq(s1, s2):
///     if s1 == s2:       # also if both are NULLs
///         return True
///     if not s1 or not s2:
///         return False
///     len1 = len(s1.chars)
///     len2 = len(s2.chars)
///     if len1 != len2:
///         return False
///     j = 0
///     chars1 = s1.chars
///     chars2 = s2.chars
///     while j < len1:
///         if chars1[j] != chars2[j]:
///             return False
///         j += 1
///     return True
/// ```
///
/// 6-block CFG plus the returnblock:
/// - **start**: `eq = ptr_eq(s1, s2)`. True → returnblock(`Bool(true)`),
///   False → `block_null_check_s1`.
/// - **block_null_check_s1**: `nz1 = ptr_nonzero(s1)`. False →
///   returnblock(`Bool(false)`), True → `block_null_check_s2`.
/// - **block_null_check_s2**: `nz2 = ptr_nonzero(s2)`. False →
///   returnblock(`Bool(false)`), True → `block_compare_lens`.
/// - **block_compare_lens**: `chars1 = getsubstruct(s1, 'chars')`,
///   `len1 = getarraysize(chars1)`, `chars2 = getsubstruct(s2,
///   'chars')`, `len2 = getarraysize(chars2)`, `lens_eq = int_eq(len1,
///   len2)`. True → `block_loop_cond` with `j = 0`, False →
///   returnblock(`Bool(false)`).
/// - **block_loop_cond**: `lt = int_lt(j, len1)`. True →
///   `block_loop_body`, False → returnblock(`Bool(true)`)
///   (loop exhausted with all chars matching).
/// - **block_loop_body**: `c1 = getarrayitem(chars1, j)`, `c2 =
///   getarrayitem(chars2, j)`, `chars_eq = char_eq/unichar_eq(c1,
///   c2)`, `j_next = int_add(j, 1)`. True (chars match) →
///   `block_loop_cond` with `j_next`, False → returnblock(`Bool(false)`).
///
/// Polymorphic over `Ptr(STR)` / `Ptr(UNICODE)` via the
/// chars-array Ptr lltype derivation reused from
/// [`chars_array_ptr_lltype_from_strptr`]. The element comparison op
/// (`char_eq` / `unichar_eq`) is selected from the chars-array
/// element type.
pub(crate) fn build_ll_streq_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    // Element comparison op depends on the chars Array element type.
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_streq helper expects Ptr(Array(Char|UniChar)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_streq helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let char_eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_streq helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Inputargs.
    let s1 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2 = variable_with_lltype("s2", ptr_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s1.clone()),
        Hlvalue::Variable(s2.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Pre-create downstream blocks so that `closeblock` calls below
    // can supply already-existing targets. Each block carries its own
    // local copy of the inputargs that flow forward.
    let s1_for_null = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_null = variable_with_lltype("s2", ptr_lltype.clone());
    let block_null_check_s1 = Block::shared(vec![
        Hlvalue::Variable(s1_for_null.clone()),
        Hlvalue::Variable(s2_for_null.clone()),
    ]);

    let s1_for_null2 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_null2 = variable_with_lltype("s2", ptr_lltype.clone());
    let block_null_check_s2 = Block::shared(vec![
        Hlvalue::Variable(s1_for_null2.clone()),
        Hlvalue::Variable(s2_for_null2.clone()),
    ]);

    let s1_for_lens = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_lens = variable_with_lltype("s2", ptr_lltype.clone());
    let block_compare_lens = Block::shared(vec![
        Hlvalue::Variable(s1_for_lens.clone()),
        Hlvalue::Variable(s2_for_lens.clone()),
    ]);

    let chars1_for_cond = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_cond = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len1_for_cond = variable_with_lltype("len1", LowLevelType::Signed);
    let j_for_cond = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars1_for_cond.clone()),
        Hlvalue::Variable(chars2_for_cond.clone()),
        Hlvalue::Variable(len1_for_cond.clone()),
        Hlvalue::Variable(j_for_cond.clone()),
    ]);

    let chars1_for_body = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_body = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len1_for_body = variable_with_lltype("len1", LowLevelType::Signed);
    let j_for_body = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars1_for_body.clone()),
        Hlvalue::Variable(chars2_for_body.clone()),
        Hlvalue::Variable(len1_for_body.clone()),
        Hlvalue::Variable(j_for_body.clone()),
    ]);

    // start: ptr_eq(s1, s2); branch on result.
    let v_eq = variable_with_lltype("eq", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_eq",
        vec![Hlvalue::Variable(s1.clone()), Hlvalue::Variable(s2.clone())],
        Hlvalue::Variable(v_eq.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_eq));
    startblock.closeblock(vec![
        Link::new(
            vec![bool_true()],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(s1), Hlvalue::Variable(s2)],
            Some(block_null_check_s1.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_null_check_s1: ptr_nonzero(s1).
    let nz1 = variable_with_lltype("nz1", LowLevelType::Bool);
    block_null_check_s1
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "ptr_nonzero",
            vec![Hlvalue::Variable(s1_for_null.clone())],
            Hlvalue::Variable(nz1.clone()),
        ));
    block_null_check_s1.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz1));
    block_null_check_s1.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s1_for_null),
                Hlvalue::Variable(s2_for_null),
            ],
            Some(block_null_check_s2.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_null_check_s2: ptr_nonzero(s2).
    let nz2 = variable_with_lltype("nz2", LowLevelType::Bool);
    block_null_check_s2
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "ptr_nonzero",
            vec![Hlvalue::Variable(s2_for_null2.clone())],
            Hlvalue::Variable(nz2.clone()),
        ));
    block_null_check_s2.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz2));
    block_null_check_s2.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s1_for_null2),
                Hlvalue::Variable(s2_for_null2),
            ],
            Some(block_compare_lens.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_compare_lens: extract chars + len for both sides,
    // then int_eq(len1, len2).
    let chars1 = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    block_compare_lens
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s1_for_lens.clone()), chars_field_const()],
            Hlvalue::Variable(chars1.clone()),
        ));
    let len1 = variable_with_lltype("len1", LowLevelType::Signed);
    block_compare_lens
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(chars1.clone())],
            Hlvalue::Variable(len1.clone()),
        ));
    let chars2 = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    block_compare_lens
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s2_for_lens.clone()), chars_field_const()],
            Hlvalue::Variable(chars2.clone()),
        ));
    let len2 = variable_with_lltype("len2", LowLevelType::Signed);
    block_compare_lens
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(chars2.clone())],
            Hlvalue::Variable(len2.clone()),
        ));
    let lens_eq = variable_with_lltype("lens_eq", LowLevelType::Bool);
    block_compare_lens
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(len1.clone()), Hlvalue::Variable(len2)],
            Hlvalue::Variable(lens_eq.clone()),
        ));
    block_compare_lens.borrow_mut().exitswitch = Some(Hlvalue::Variable(lens_eq));
    block_compare_lens.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1),
                Hlvalue::Variable(chars2),
                Hlvalue::Variable(len1),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_cond: int_lt(j, len1); branch.
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(j_for_cond.clone()),
                Hlvalue::Variable(len1_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_cond),
                Hlvalue::Variable(chars2_for_cond),
                Hlvalue::Variable(len1_for_cond),
                Hlvalue::Variable(j_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_true()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_body: getarrayitem on both, char_eq, int_add(j, 1).
    let c1 = variable_with_lltype("c1", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars1_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(c1.clone()),
        ));
    let c2 = variable_with_lltype("c2", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars2_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(c2.clone()),
        ));
    let chars_eq = variable_with_lltype("chars_eq", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            char_eq_op,
            vec![Hlvalue::Variable(c1), Hlvalue::Variable(c2)],
            Hlvalue::Variable(chars_eq.clone()),
        ));
    let j_next = variable_with_lltype("j_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(j_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(j_next.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(chars_eq));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_body),
                Hlvalue::Variable(chars2_for_body),
                Hlvalue::Variable(len1_for_body),
                Hlvalue::Variable(j_next),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s1".to_string(), "s2".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_strcmp`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// def ll_strcmp(s1, s2):
///     if not s1 and not s2:
///         return True
///     if not s1 or not s2:
///         return False
///     chars1 = s1.chars
///     chars2 = s2.chars
///     len1 = len(chars1)
///     len2 = len(chars2)
///
///     if len1 < len2:
///         cmplen = len1
///     else:
///         cmplen = len2
///     i = 0
///     while i < cmplen:
///         diff = ord(chars1[i]) - ord(chars2[i])
///         if diff != 0:
///             return diff
///         i += 1
///     return len1 - len2
/// ```
///
/// Return type is `Signed`: RPython coerces the `True`/`False` literals
/// in the NULL-pair branches to `Signed(1)` / `Signed(0)` so the four
/// returns (1 / 0 / diff / len1-len2) all land in a single `Signed`
/// inputarg on the returnblock.
///
/// 7-block CFG plus the returnblock:
/// - **start**: `nz1 = ptr_nonzero(s1)`. True → `block_s1_nonnull`,
///   False → `block_s1_null`.
/// - **block_s1_null**: `nz2 = ptr_nonzero(s2)`. True (s2 non-NULL,
///   s1 NULL) → returnblock(`Signed(0)`); False (both NULL) →
///   returnblock(`Signed(1)`).
/// - **block_s1_nonnull**: `nz2 = ptr_nonzero(s2)`. True (both
///   non-NULL) → `block_both_nonnull`; False (s1 non-NULL, s2 NULL) →
///   returnblock(`Signed(0)`).
/// - **block_both_nonnull**: `chars1 = getsubstruct(s1, 'chars')`,
///   `len1 = getarraysize(chars1)`, `chars2 = getsubstruct(s2,
///   'chars')`, `len2 = getarraysize(chars2)`, `lens_lt = int_lt(len1,
///   len2)`. True → `block_loop_cond` with `cmplen = len1`, False →
///   `block_loop_cond` with `cmplen = len2`. Both branches initialise
///   `i = 0`.
/// - **block_loop_cond**: `lt = int_lt(i, cmplen)`. True →
///   `block_loop_body`, False → `block_return_len_diff`.
/// - **block_loop_body**: `c1 = getarrayitem(chars1, i)`, `c2 =
///   getarrayitem(chars2, i)`, `c1_int = cast_*_to_int(c1)`, `c2_int
///   = cast_*_to_int(c2)`, `diff = int_sub(c1_int, c2_int)`,
///   `has_diff = int_ne(diff, 0)`, `i_next = int_add(i, 1)`. True
///   (chars differ) → returnblock(`diff`), False (chars match) →
///   `block_loop_cond` with `i_next`.
/// - **block_return_len_diff**: `len_diff = int_sub(len1, len2)`;
///   unconditional link to returnblock.
///
/// Polymorphic over `Ptr(STR)` / `Ptr(UNICODE)` via the chars-array
/// Ptr lltype derivation reused from
/// [`chars_array_ptr_lltype_from_strptr`]. The cast op
/// (`cast_char_to_int` / `cast_unichar_to_int`) is selected from the
/// chars-array element type — mirrors `AbstractStringRepr` /
/// `AbstractUnicodeRepr` `ord(chars[i])` lowering.
pub(crate) fn build_ll_strcmp_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_strcmp helper expects Ptr(Array(Char|UniChar)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_strcmp helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let cast_op = match elem_lltype {
        LowLevelType::Char => "cast_char_to_int",
        LowLevelType::UniChar => "cast_unichar_to_int",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_strcmp helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Inputargs.
    let s1 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2 = variable_with_lltype("s2", ptr_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s1.clone()),
        Hlvalue::Variable(s2.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Pre-create downstream blocks.
    // block_s1_null: s1 IS NULL on entry; only s2 is needed.
    let s2_for_null_pair = variable_with_lltype("s2", ptr_lltype.clone());
    let block_s1_null = Block::shared(vec![Hlvalue::Variable(s2_for_null_pair.clone())]);

    // block_s1_nonnull: s1 IS non-NULL; both pointers carry forward.
    let s1_for_s2check = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_s2check = variable_with_lltype("s2", ptr_lltype.clone());
    let block_s1_nonnull = Block::shared(vec![
        Hlvalue::Variable(s1_for_s2check.clone()),
        Hlvalue::Variable(s2_for_s2check.clone()),
    ]);

    // block_both_nonnull: both pointers are live and non-NULL.
    let s1_for_lens = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_lens = variable_with_lltype("s2", ptr_lltype.clone());
    let block_both_nonnull = Block::shared(vec![
        Hlvalue::Variable(s1_for_lens.clone()),
        Hlvalue::Variable(s2_for_lens.clone()),
    ]);

    // block_loop_cond: carries (chars1, chars2, len1, len2, cmplen, i).
    // len1/len2 propagate through the loop to feed
    // `block_return_len_diff` once the loop exits without finding a
    // mismatch.
    let chars1_for_cond = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_cond = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len1_for_cond = variable_with_lltype("len1", LowLevelType::Signed);
    let len2_for_cond = variable_with_lltype("len2", LowLevelType::Signed);
    let cmplen_for_cond = variable_with_lltype("cmplen", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars1_for_cond.clone()),
        Hlvalue::Variable(chars2_for_cond.clone()),
        Hlvalue::Variable(len1_for_cond.clone()),
        Hlvalue::Variable(len2_for_cond.clone()),
        Hlvalue::Variable(cmplen_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    // block_loop_body: same six-tuple as loop_cond.
    let chars1_for_body = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_body = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len1_for_body = variable_with_lltype("len1", LowLevelType::Signed);
    let len2_for_body = variable_with_lltype("len2", LowLevelType::Signed);
    let cmplen_for_body = variable_with_lltype("cmplen", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars1_for_body.clone()),
        Hlvalue::Variable(chars2_for_body.clone()),
        Hlvalue::Variable(len1_for_body.clone()),
        Hlvalue::Variable(len2_for_body.clone()),
        Hlvalue::Variable(cmplen_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
    ]);

    // block_return_len_diff: takes (len1, len2), computes int_sub.
    let len1_for_lendiff = variable_with_lltype("len1", LowLevelType::Signed);
    let len2_for_lendiff = variable_with_lltype("len2", LowLevelType::Signed);
    let block_return_len_diff = Block::shared(vec![
        Hlvalue::Variable(len1_for_lendiff.clone()),
        Hlvalue::Variable(len2_for_lendiff.clone()),
    ]);

    // start: ptr_nonzero(s1); branch.
    let nz1 = variable_with_lltype("nz1", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_nonzero",
        vec![Hlvalue::Variable(s1.clone())],
        Hlvalue::Variable(nz1.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz1));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s1.clone()), Hlvalue::Variable(s2.clone())],
            Some(block_s1_nonnull.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(s2)],
            Some(block_s1_null.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_s1_null: s1 known NULL; ptr_nonzero(s2) decides
    // both-NULL (False → return 1) vs only-s1-NULL (True → return 0).
    let nz2_in_null = variable_with_lltype("nz2", LowLevelType::Bool);
    block_s1_null
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "ptr_nonzero",
            vec![Hlvalue::Variable(s2_for_null_pair)],
            Hlvalue::Variable(nz2_in_null.clone()),
        ));
    block_s1_null.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz2_in_null));
    block_s1_null.closeblock(vec![
        Link::new(
            vec![signed_const(0)],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![signed_const(1)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_s1_nonnull: ptr_nonzero(s2). True (both non-NULL) →
    // continue; False (only s2 NULL) → return 0.
    let nz2_in_nn = variable_with_lltype("nz2", LowLevelType::Bool);
    block_s1_nonnull
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "ptr_nonzero",
            vec![Hlvalue::Variable(s2_for_s2check.clone())],
            Hlvalue::Variable(nz2_in_nn.clone()),
        ));
    block_s1_nonnull.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz2_in_nn));
    block_s1_nonnull.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s1_for_s2check),
                Hlvalue::Variable(s2_for_s2check),
            ],
            Some(block_both_nonnull.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![signed_const(0)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_both_nonnull: extract chars/len for both, branch on
    // int_lt(len1, len2) to compute cmplen = min(len1, len2).
    let chars1 = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    block_both_nonnull
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s1_for_lens.clone()), chars_field_const()],
            Hlvalue::Variable(chars1.clone()),
        ));
    let len1 = variable_with_lltype("len1", LowLevelType::Signed);
    block_both_nonnull
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(chars1.clone())],
            Hlvalue::Variable(len1.clone()),
        ));
    let chars2 = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    block_both_nonnull
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s2_for_lens), chars_field_const()],
            Hlvalue::Variable(chars2.clone()),
        ));
    let len2 = variable_with_lltype("len2", LowLevelType::Signed);
    block_both_nonnull
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(chars2.clone())],
            Hlvalue::Variable(len2.clone()),
        ));
    let lens_lt = variable_with_lltype("lens_lt", LowLevelType::Bool);
    block_both_nonnull
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(len1.clone()),
                Hlvalue::Variable(len2.clone()),
            ],
            Hlvalue::Variable(lens_lt.clone()),
        ));
    block_both_nonnull.borrow_mut().exitswitch = Some(Hlvalue::Variable(lens_lt));
    block_both_nonnull.closeblock(vec![
        // True (len1 < len2): cmplen = len1.
        Link::new(
            vec![
                Hlvalue::Variable(chars1.clone()),
                Hlvalue::Variable(chars2.clone()),
                Hlvalue::Variable(len1.clone()),
                Hlvalue::Variable(len2.clone()),
                Hlvalue::Variable(len1.clone()),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False (len1 >= len2): cmplen = len2.
        Link::new(
            vec![
                Hlvalue::Variable(chars1),
                Hlvalue::Variable(chars2),
                Hlvalue::Variable(len1),
                Hlvalue::Variable(len2.clone()),
                Hlvalue::Variable(len2),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_cond: int_lt(i, cmplen); branch.
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(cmplen_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_cond),
                Hlvalue::Variable(chars2_for_cond),
                Hlvalue::Variable(len1_for_cond.clone()),
                Hlvalue::Variable(len2_for_cond.clone()),
                Hlvalue::Variable(cmplen_for_cond),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(len1_for_cond),
                Hlvalue::Variable(len2_for_cond),
            ],
            Some(block_return_len_diff.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_body: getarrayitem×2, cast×2, int_sub diff,
    // int_ne, int_add i_next; branch on has_diff.
    let c1 = variable_with_lltype("c1", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars1_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
            ],
            Hlvalue::Variable(c1.clone()),
        ));
    let c2 = variable_with_lltype("c2", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars2_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
            ],
            Hlvalue::Variable(c2.clone()),
        ));
    let c1_int = variable_with_lltype("c1_int", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(c1)],
            Hlvalue::Variable(c1_int.clone()),
        ));
    let c2_int = variable_with_lltype("c2_int", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(c2)],
            Hlvalue::Variable(c2_int.clone()),
        ));
    let diff = variable_with_lltype("diff", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(c1_int), Hlvalue::Variable(c2_int)],
            Hlvalue::Variable(diff.clone()),
        ));
    let has_diff = variable_with_lltype("has_diff", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ne",
            vec![Hlvalue::Variable(diff.clone()), signed_const(0)],
            Hlvalue::Variable(has_diff.clone()),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(has_diff));
    block_loop_body.closeblock(vec![
        // True (diff != 0): return diff.
        Link::new(
            vec![Hlvalue::Variable(diff)],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False (chars match): cycle to loop_cond with i_next.
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_body),
                Hlvalue::Variable(chars2_for_body),
                Hlvalue::Variable(len1_for_body),
                Hlvalue::Variable(len2_for_body),
                Hlvalue::Variable(cmplen_for_body),
                Hlvalue::Variable(i_next),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_return_len_diff: int_sub(len1, len2); link to
    // returnblock.
    let len_diff = variable_with_lltype("len_diff", LowLevelType::Signed);
    block_return_len_diff
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![
                Hlvalue::Variable(len1_for_lendiff),
                Hlvalue::Variable(len2_for_lendiff),
            ],
            Hlvalue::Variable(len_diff.clone()),
        ));
    block_return_len_diff.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(len_diff)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s1".to_string(), "s2".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_startswith`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// @jit.elidable
/// def ll_startswith(s1, s2):
///     len1 = len(s1.chars)
///     len2 = len(s2.chars)
///     if len1 < len2:
///         return False
///     j = 0
///     chars1 = s1.chars
///     chars2 = s2.chars
///     while j < len2:
///         if chars1[j] != chars2[j]:
///             return False
///         j += 1
///
///     return True
/// ```
///
/// 3-block CFG plus the returnblock:
/// - **start**: `chars1 = getsubstruct(s1, 'chars')`, `len1 =
///   getarraysize(chars1)`, `chars2 = getsubstruct(s2, 'chars')`,
///   `len2 = getarraysize(chars2)`, `lens_lt = int_lt(len1, len2)`.
///   True → returnblock(`Bool(false)`) (s1 too short to start with
///   s2); False → `block_loop_cond` with `j = 0`.
/// - **block_loop_cond**: `lt = int_lt(j, len2)`. True →
///   `block_loop_body`; False → returnblock(`Bool(true)`) (loop
///   exhausted, all common chars matched).
/// - **block_loop_body**: `c1 = getarrayitem(chars1, j)`, `c2 =
///   getarrayitem(chars2, j)`, `chars_eq = char_eq/unichar_eq(c1,
///   c2)`, `j_next = int_add(j, 1)`. True (chars match) →
///   `block_loop_cond` with `j_next`; False → returnblock(`Bool(false)`).
///
/// Polymorphic over `Ptr(STR)` / `Ptr(UNICODE)` via the chars-array
/// Ptr lltype derivation reused from
/// [`chars_array_ptr_lltype_from_strptr`]. The element comparison op
/// (`char_eq` / `unichar_eq`) is selected from the chars-array
/// element type (mirrors [`build_ll_streq_helper_graph`]).
pub(crate) fn build_ll_startswith_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_startswith helper expects Ptr(Array(Char|UniChar)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_startswith helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let char_eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_startswith helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Inputargs.
    let s1 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2 = variable_with_lltype("s2", ptr_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s1.clone()),
        Hlvalue::Variable(s2.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Pre-create downstream blocks. loop_cond carries forward (chars1,
    // chars2, len2, j); loop_body shares the same shape so the True
    // branch back-edge from loop_body cycles to loop_cond.
    let chars1_for_cond = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_cond = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len2_for_cond = variable_with_lltype("len2", LowLevelType::Signed);
    let j_for_cond = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars1_for_cond.clone()),
        Hlvalue::Variable(chars2_for_cond.clone()),
        Hlvalue::Variable(len2_for_cond.clone()),
        Hlvalue::Variable(j_for_cond.clone()),
    ]);

    let chars1_for_body = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_body = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len2_for_body = variable_with_lltype("len2", LowLevelType::Signed);
    let j_for_body = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars1_for_body.clone()),
        Hlvalue::Variable(chars2_for_body.clone()),
        Hlvalue::Variable(len2_for_body.clone()),
        Hlvalue::Variable(j_for_body.clone()),
    ]);

    // start: extract chars/len for both, branch on int_lt(len1, len2).
    let chars1 = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s1.clone()), chars_field_const()],
        Hlvalue::Variable(chars1.clone()),
    ));
    let len1 = variable_with_lltype("len1", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars1.clone())],
        Hlvalue::Variable(len1.clone()),
    ));
    let chars2 = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s2.clone()), chars_field_const()],
        Hlvalue::Variable(chars2.clone()),
    ));
    let len2 = variable_with_lltype("len2", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars2.clone())],
        Hlvalue::Variable(len2.clone()),
    ));
    let lens_lt = variable_with_lltype("lens_lt", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(len1), Hlvalue::Variable(len2.clone())],
        Hlvalue::Variable(lens_lt.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(lens_lt));
    startblock.closeblock(vec![
        // True (len1 < len2): s1 too short.
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False (len1 >= len2): enter loop with j = 0.
        Link::new(
            vec![
                Hlvalue::Variable(chars1),
                Hlvalue::Variable(chars2),
                Hlvalue::Variable(len2),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_cond: int_lt(j, len2); branch.
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(j_for_cond.clone()),
                Hlvalue::Variable(len2_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_cond),
                Hlvalue::Variable(chars2_for_cond),
                Hlvalue::Variable(len2_for_cond),
                Hlvalue::Variable(j_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_true()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_body: getarrayitem×2, char_eq/unichar_eq,
    // int_add(j, 1); branch on chars_eq.
    let c1 = variable_with_lltype("c1", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars1_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(c1.clone()),
        ));
    let c2 = variable_with_lltype("c2", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars2_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(c2.clone()),
        ));
    let chars_eq = variable_with_lltype("chars_eq", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            char_eq_op,
            vec![Hlvalue::Variable(c1), Hlvalue::Variable(c2)],
            Hlvalue::Variable(chars_eq.clone()),
        ));
    let j_next = variable_with_lltype("j_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(j_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(j_next.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(chars_eq));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_body),
                Hlvalue::Variable(chars2_for_body),
                Hlvalue::Variable(len2_for_body),
                Hlvalue::Variable(j_next),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s1".to_string(), "s2".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_endswith`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// @jit.elidable
/// def ll_endswith(s1, s2):
///     len1 = len(s1.chars)
///     len2 = len(s2.chars)
///     if len1 < len2:
///         return False
///     j = 0
///     chars1 = s1.chars
///     chars2 = s2.chars
///     offset = len1 - len2
///     while j < len2:
///         if chars1[offset + j] != chars2[j]:
///             return False
///         j += 1
///
///     return True
/// ```
///
/// 3-block CFG plus the returnblock — mirror of
/// [`build_ll_startswith_helper_graph`] with the addition of a
/// constant `offset = len1 - len2` carried through the loop and an
/// `int_add(offset, j) -> idx` indirection in `block_loop_body`:
/// - **start**: `chars1 = getsubstruct(s1, 'chars')`, `len1 =
///   getarraysize(chars1)`, `chars2 = getsubstruct(s2, 'chars')`,
///   `len2 = getarraysize(chars2)`, `offset = int_sub(len1, len2)`,
///   `lens_lt = int_lt(len1, len2)`. True → returnblock(`Bool(false)`);
///   False → `block_loop_cond` with `j = 0`.
/// - **block_loop_cond**: `lt = int_lt(j, len2)`. True →
///   `block_loop_body`; False → returnblock(`Bool(true)`).
/// - **block_loop_body**: `idx = int_add(offset, j)`, `c1 =
///   getarrayitem(chars1, idx)`, `c2 = getarrayitem(chars2, j)`,
///   `chars_eq = char_eq/unichar_eq(c1, c2)`, `j_next = int_add(j,
///   1)`. True (chars match) → `block_loop_cond` with `j_next`,
///   False → returnblock(`Bool(false)`).
///
/// Polymorphic Ptr(STR)/Ptr(UNICODE); element op selected from chars
/// Array element type (mirrors [`build_ll_streq_helper_graph`] /
/// [`build_ll_startswith_helper_graph`]).
pub(crate) fn build_ll_endswith_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_endswith helper expects Ptr(Array(Char|UniChar)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_endswith helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let char_eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_endswith helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s1 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2 = variable_with_lltype("s2", ptr_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s1.clone()),
        Hlvalue::Variable(s2.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Pre-create downstream blocks. loop_cond / loop_body carry
    // (chars1, chars2, len2, offset, j).
    let chars1_for_cond = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_cond = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len2_for_cond = variable_with_lltype("len2", LowLevelType::Signed);
    let offset_for_cond = variable_with_lltype("offset", LowLevelType::Signed);
    let j_for_cond = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars1_for_cond.clone()),
        Hlvalue::Variable(chars2_for_cond.clone()),
        Hlvalue::Variable(len2_for_cond.clone()),
        Hlvalue::Variable(offset_for_cond.clone()),
        Hlvalue::Variable(j_for_cond.clone()),
    ]);

    let chars1_for_body = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_body = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len2_for_body = variable_with_lltype("len2", LowLevelType::Signed);
    let offset_for_body = variable_with_lltype("offset", LowLevelType::Signed);
    let j_for_body = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars1_for_body.clone()),
        Hlvalue::Variable(chars2_for_body.clone()),
        Hlvalue::Variable(len2_for_body.clone()),
        Hlvalue::Variable(offset_for_body.clone()),
        Hlvalue::Variable(j_for_body.clone()),
    ]);

    // start: chars1/len1 + chars2/len2 + offset (int_sub) +
    // length comparison branch.
    let chars1 = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s1.clone()), chars_field_const()],
        Hlvalue::Variable(chars1.clone()),
    ));
    let len1 = variable_with_lltype("len1", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars1.clone())],
        Hlvalue::Variable(len1.clone()),
    ));
    let chars2 = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s2.clone()), chars_field_const()],
        Hlvalue::Variable(chars2.clone()),
    ));
    let len2 = variable_with_lltype("len2", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars2.clone())],
        Hlvalue::Variable(len2.clone()),
    ));
    let offset = variable_with_lltype("offset", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![
            Hlvalue::Variable(len1.clone()),
            Hlvalue::Variable(len2.clone()),
        ],
        Hlvalue::Variable(offset.clone()),
    ));
    let lens_lt = variable_with_lltype("lens_lt", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(len1), Hlvalue::Variable(len2.clone())],
        Hlvalue::Variable(lens_lt.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(lens_lt));
    startblock.closeblock(vec![
        // True (len1 < len2): s1 too short.
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False: enter loop with offset = len1 - len2 and j = 0.
        Link::new(
            vec![
                Hlvalue::Variable(chars1),
                Hlvalue::Variable(chars2),
                Hlvalue::Variable(len2),
                Hlvalue::Variable(offset),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_cond: int_lt(j, len2); branch.
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(j_for_cond.clone()),
                Hlvalue::Variable(len2_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_cond),
                Hlvalue::Variable(chars2_for_cond),
                Hlvalue::Variable(len2_for_cond),
                Hlvalue::Variable(offset_for_cond),
                Hlvalue::Variable(j_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_true()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_body: int_add(offset, j) -> idx; getarrayitem
    // (chars1, idx); getarrayitem(chars2, j); char_eq/unichar_eq;
    // int_add(j, 1); branch on chars_eq.
    let idx = variable_with_lltype("idx", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(offset_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(idx.clone()),
        ));
    let c1 = variable_with_lltype("c1", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars1_for_body.clone()),
                Hlvalue::Variable(idx),
            ],
            Hlvalue::Variable(c1.clone()),
        ));
    let c2 = variable_with_lltype("c2", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars2_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(c2.clone()),
        ));
    let chars_eq = variable_with_lltype("chars_eq", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            char_eq_op,
            vec![Hlvalue::Variable(c1), Hlvalue::Variable(c2)],
            Hlvalue::Variable(chars_eq.clone()),
        ));
    let j_next = variable_with_lltype("j_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(j_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(j_next.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(chars_eq));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_body),
                Hlvalue::Variable(chars2_for_body),
                Hlvalue::Variable(len2_for_body),
                Hlvalue::Variable(offset_for_body),
                Hlvalue::Variable(j_next),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s1".to_string(), "s2".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_startswith_char`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// def ll_startswith_char(s, ch):
///     if not len(s.chars):
///         return False
///     return s.chars[0] == ch
/// ```
///
/// 2-block CFG plus the returnblock:
/// - **start**: `chars = getsubstruct(s, 'chars')`, `length =
///   getarraysize(chars)`, `is_empty = int_eq(length, 0)`. True →
///   returnblock(`Bool(false)`); False → `block_compare` with
///   `chars`.
/// - **block_compare**: `c0 = getarrayitem(chars, 0)`, `eq =
///   char_eq/unichar_eq(c0, ch)`. Unconditional link to returnblock
///   with `eq`.
///
/// `ch` is the helper's second inputarg with lltype derived from the
/// chars Array element type (Char for STR, UniChar for UNICODE).
pub(crate) fn build_ll_startswith_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    build_ll_startsendswith_char_helper_graph(name, ptr_lltype, /* is_endswith */ false)
}

/// Synthesise the helper graph for `LLHelpers.ll_endswith_char`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// def ll_endswith_char(s, ch):
///     if not len(s.chars):
///         return False
///     return s.chars[len(s.chars) - 1] == ch
/// ```
///
/// Mirror of [`build_ll_startswith_char_helper_graph`] differing only
/// in the index used to read the comparison char: instead of
/// `chars[0]`, the body emits `idx = int_sub(length, 1)` then
/// `chars[idx]`. CFG and length-empty fast path are identical.
pub(crate) fn build_ll_endswith_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    build_ll_startsendswith_char_helper_graph(name, ptr_lltype, /* is_endswith */ true)
}

/// Shared body for [`build_ll_startswith_char_helper_graph`] and
/// [`build_ll_endswith_char_helper_graph`]. The two helpers differ
/// only in the index expression for the chars-array read:
/// startswith uses `Constant(0)` directly; endswith inserts an
/// `idx = int_sub(length, 1)` op and reads `chars[idx]`.
fn build_ll_startsendswith_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    is_endswith: bool,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_starts/endswith_char helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_starts/endswith_char helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let char_eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_starts/endswith_char helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let ch = variable_with_lltype("ch", elem_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(ch.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // block_compare carries forward (chars, length, ch); length is
    // only consumed by the endswith variant for `int_sub(length, 1)`.
    let chars_for_compare = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let length_for_compare = variable_with_lltype("length", LowLevelType::Signed);
    let ch_for_compare = variable_with_lltype("ch", elem_lltype.clone());
    let block_compare = Block::shared(vec![
        Hlvalue::Variable(chars_for_compare.clone()),
        Hlvalue::Variable(length_for_compare.clone()),
        Hlvalue::Variable(ch_for_compare.clone()),
    ]);

    // start: getsubstruct + getarraysize + int_eq(length, 0).
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s.clone()), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let is_empty = variable_with_lltype("is_empty", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(length.clone()), signed_const(0)],
        Hlvalue::Variable(is_empty.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_empty));
    startblock.closeblock(vec![
        // True (length == 0): empty string can't start/end with ch.
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False: compare chars[idx] against ch.
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(length),
                Hlvalue::Variable(ch),
            ],
            Some(block_compare.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_compare: read chars[idx], emit char_eq/unichar_eq,
    // unconditional link to returnblock with the result.
    let idx_value = if is_endswith {
        let idx = variable_with_lltype("idx", LowLevelType::Signed);
        block_compare
            .borrow_mut()
            .operations
            .push(SpaceOperation::new(
                "int_sub",
                vec![Hlvalue::Variable(length_for_compare), signed_const(1)],
                Hlvalue::Variable(idx.clone()),
            ));
        Hlvalue::Variable(idx)
    } else {
        signed_const(0)
    };
    let c = variable_with_lltype("c", elem_lltype.clone());
    block_compare
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![Hlvalue::Variable(chars_for_compare), idx_value],
            Hlvalue::Variable(c.clone()),
        ));
    let eq = variable_with_lltype("eq", LowLevelType::Bool);
    block_compare
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            char_eq_op,
            vec![Hlvalue::Variable(c), Hlvalue::Variable(ch_for_compare)],
            Hlvalue::Variable(eq.clone()),
        ));
    block_compare.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(eq)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "ch".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `_hash_string`
/// (`rpython/rlib/objectmodel.py`):
///
/// ```python
/// @specialize.ll()
/// def _hash_string(s):
///     length = len(s)
///     if length == 0:
///         return -1
///     x = ord(s[0]) << 7
///     i = 0
///     while i < length:
///         x = intmask((1000003*x) ^ ord(s[i]))
///         i += 1
///     x ^= length
///     return intmask(x)
/// ```
///
/// `s` is a chars Array (`Ptr(Array(Char|UniChar))`), not a STR/UNICODE
/// pointer — the caller (`ll_hash_string`, objectmodel.py)
/// supplies `ll_s.chars` via `getsubstruct`. `intmask` is a no-op
/// at lltype level (lltype `Signed` arithmetic already wraps), so the
/// synthesizer drops the call.
///
/// 5-block CFG plus the returnblock:
/// - **start**: `length = getarraysize(chars)`, `is_empty =
///   int_eq(length, 0)`. True → returnblock(`Signed(-1)`); False →
///   `block_init` with `(chars, length)`.
/// - **block_init**: `c0 = getarrayitem(chars, 0)`, `c0_int =
///   cast_*_to_int(c0)`, `x = int_lshift(c0_int, 7)`. Unconditional
///   link to `block_loop_cond` with `(chars, length, x, 0)`.
/// - **block_loop_cond**: `lt = int_lt(i, length)`. True →
///   `block_loop_body`; False → `block_finalize` with `(length, x)`.
/// - **block_loop_body**: `x_mul = int_mul(1000003, x)`, `ci =
///   getarrayitem(chars, i)`, `ci_int = cast_*_to_int(ci)`, `x_new =
///   int_xor(x_mul, ci_int)`, `i_next = int_add(i, 1)`. Unconditional
///   link back to `block_loop_cond` with `(chars, length, x_new,
///   i_next)`.
/// - **block_finalize**: `x_final = int_xor(x, length)`. Link to
///   returnblock with `x_final`.
///
/// Polymorphic over `Ptr(Array(Char))` / `Ptr(Array(UniChar))` — the
/// cast op (`cast_char_to_int` / `cast_unichar_to_int`) and the
/// `ord`-emit element type are selected from the array's `OF`. Use
/// [`build_ll_strlen_helper_graph::chars_array_ptr_lltype_from_strptr`]
/// callers wrap this with the STR/UNICODE pointer.
pub(crate) fn build_hash_string_helper_graph(
    name: &str,
    chars_array_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "_hash_string helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "_hash_string helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let cast_op = match elem_lltype {
        LowLevelType::Char => "cast_char_to_int",
        LowLevelType::UniChar => "cast_unichar_to_int",
        ref other => {
            return Err(TyperError::message(format!(
                "_hash_string helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);

    let chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(chars.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Pre-create downstream blocks. block_init / loop_cond / loop_body
    // carry forward (chars, length [, x [, i]]); block_finalize takes
    // only (length, x).
    let chars_for_init = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let length_for_init = variable_with_lltype("length", LowLevelType::Signed);
    let block_init = Block::shared(vec![
        Hlvalue::Variable(chars_for_init.clone()),
        Hlvalue::Variable(length_for_init.clone()),
    ]);

    let chars_for_cond = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let length_for_cond = variable_with_lltype("length", LowLevelType::Signed);
    let x_for_cond = variable_with_lltype("x", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars_for_cond.clone()),
        Hlvalue::Variable(length_for_cond.clone()),
        Hlvalue::Variable(x_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    let chars_for_body = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let length_for_body = variable_with_lltype("length", LowLevelType::Signed);
    let x_for_body = variable_with_lltype("x", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars_for_body.clone()),
        Hlvalue::Variable(length_for_body.clone()),
        Hlvalue::Variable(x_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
    ]);

    let length_for_finalize = variable_with_lltype("length", LowLevelType::Signed);
    let x_for_finalize = variable_with_lltype("x", LowLevelType::Signed);
    let block_finalize = Block::shared(vec![
        Hlvalue::Variable(length_for_finalize.clone()),
        Hlvalue::Variable(x_for_finalize.clone()),
    ]);

    // start: getarraysize + int_eq(length, 0).
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let is_empty = variable_with_lltype("is_empty", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(length.clone()), signed_const(0)],
        Hlvalue::Variable(is_empty.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_empty));
    startblock.closeblock(vec![
        Link::new(
            vec![signed_const(-1)],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(chars), Hlvalue::Variable(length)],
            Some(block_init.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_init: chars[0] cast + lshift(7) -> x.
    let c0 = variable_with_lltype("c0", elem_lltype.clone());
    block_init.borrow_mut().operations.push(SpaceOperation::new(
        "getarrayitem",
        vec![Hlvalue::Variable(chars_for_init.clone()), signed_const(0)],
        Hlvalue::Variable(c0.clone()),
    ));
    let c0_int = variable_with_lltype("c0_int", LowLevelType::Signed);
    block_init.borrow_mut().operations.push(SpaceOperation::new(
        cast_op,
        vec![Hlvalue::Variable(c0)],
        Hlvalue::Variable(c0_int.clone()),
    ));
    let x = variable_with_lltype("x", LowLevelType::Signed);
    block_init.borrow_mut().operations.push(SpaceOperation::new(
        "int_lshift",
        vec![Hlvalue::Variable(c0_int), signed_const(7)],
        Hlvalue::Variable(x.clone()),
    ));
    block_init.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_init),
                Hlvalue::Variable(length_for_init),
                Hlvalue::Variable(x),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_loop_cond: int_lt(i, length); branch.
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(length_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_cond),
                Hlvalue::Variable(length_for_cond.clone()),
                Hlvalue::Variable(x_for_cond.clone()),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(length_for_cond),
                Hlvalue::Variable(x_for_cond),
            ],
            Some(block_finalize.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_body: x_mul = int_mul(1000003, x); ci =
    // getarrayitem(chars, i); ci_int = cast(ci); x_new = int_xor(x_mul,
    // ci_int); i_next = int_add(i, 1).
    let x_mul = variable_with_lltype("x_mul", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_mul",
            vec![signed_const(1000003), Hlvalue::Variable(x_for_body.clone())],
            Hlvalue::Variable(x_mul.clone()),
        ));
    let ci = variable_with_lltype("ci", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
            ],
            Hlvalue::Variable(ci.clone()),
        ));
    let ci_int = variable_with_lltype("ci_int", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(ci)],
            Hlvalue::Variable(ci_int.clone()),
        ));
    let x_new = variable_with_lltype("x_new", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_xor",
            vec![Hlvalue::Variable(x_mul), Hlvalue::Variable(ci_int)],
            Hlvalue::Variable(x_new.clone()),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_body),
                Hlvalue::Variable(length_for_body),
                Hlvalue::Variable(x_new),
                Hlvalue::Variable(i_next),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_finalize: int_xor(x, length); return.
    let x_final = variable_with_lltype("x_final", LowLevelType::Signed);
    block_finalize
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_xor",
            vec![
                Hlvalue::Variable(x_for_finalize),
                Hlvalue::Variable(length_for_finalize),
            ],
            Hlvalue::Variable(x_final.clone()),
        ));
    block_finalize.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(x_final)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["chars".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `ll_hash_string`
/// (`rpython/rlib/objectmodel.py`):
///
/// ```python
/// def ll_hash_string(ll_s):
///     return _hash_string(ll_s.chars)
/// ```
///
/// Single-block wrapper around [`build_hash_string_helper_graph`]:
/// `chars = getsubstruct(s, 'chars')`, `x = direct_call(_hash_string,
/// chars)`, return `x`.
///
/// `inner_helper_name` (e.g. `_hash_string` for STR pair,
/// `_hash_unicode_string` for UNICODE pair) is the cache key for the
/// inner FNV synthesizer; sub-helper graph is registered against that
/// name during this builder. The funcptr for the `direct_call` op is
/// derived via [`crate::translator::rtyper::rtyper::RPythonTyper::getcallable`]
/// once the inner helper materialises.
#[allow(dead_code)]
pub(crate) fn build_ll_hash_string_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Materialise (or retrieve cached) `_hash_string` sub-helper.
    let inner_name_owned = inner_helper_name.to_string();
    let chars_for_inner_builder = chars_array_ptr_lltype.clone();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![chars_array_ptr_lltype.clone()],
        LowLevelType::Signed,
        move |_rtyper, _args, _result| {
            build_hash_string_helper_graph(&inner_name_owned, chars_for_inner_builder)
        },
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s.clone()), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let x = variable_with_lltype("x", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![Hlvalue::Constant(c_inner_funcptr), Hlvalue::Variable(chars)],
        Hlvalue::Variable(x.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(x)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers._ll_strhash`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// @dont_inline
/// @jit.dont_look_inside
/// def _ll_strhash(s):
///     # unlike CPython, there is no reason to avoid to return -1
///     # but our malloc initializes the memory to zero, so we use zero as the
///     # special non-computed-yet value.  Also, jit.conditional_call_elidable
///     # always checks for zero, for now.
///     x = ll_hash_string(s)
///     if x == 0:
///         x = 29872897
///     s.hash = x
///     return x
/// ```
///
/// 2-block CFG plus the returnblock:
/// - **start**: `chars = getsubstruct(s, 'chars')`, `x = direct_call
///   (_hash_string, chars)`, `is_zero = int_eq(x, 0)`. True →
///   `block_set_hash` with `(s, Signed(29872897))`; False →
///   `block_set_hash` with `(s, x)`.
/// - **block_set_hash**: `setfield(s, 'hash', x_use)`. Unconditional
///   link to returnblock with `x_use`.
///
/// Inlines the `getsubstruct + direct_call(_hash_string)` pair instead
/// of routing through a separate `ll_hash_string` helper to avoid a
/// second cross-helper indirection. Result is identical: `_hash_string`
/// is registered once in the helper cache and shared with anyone else
/// who synthesises the same `(name, args, result)` tuple.
pub(crate) fn build_ll_strhash_internal_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);
    let hash_field_const =
        || constant_with_lltype(ConstValue::byte_str("hash"), LowLevelType::Void);

    // Materialise (or retrieve cached) inner `_hash_string` sub-helper
    // and derive its funcptr Constant for the `direct_call` op.
    let inner_name_owned = inner_helper_name.to_string();
    let chars_for_inner_builder = chars_array_ptr_lltype.clone();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![chars_array_ptr_lltype.clone()],
        LowLevelType::Signed,
        move |_rtyper, _args, _result| {
            build_hash_string_helper_graph(&inner_name_owned, chars_for_inner_builder)
        },
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // block_set_hash: takes (s, x_use); writes hash + returns x_use.
    let s_for_set = variable_with_lltype("s", ptr_lltype.clone());
    let x_for_set = variable_with_lltype("x_use", LowLevelType::Signed);
    let block_set_hash = Block::shared(vec![
        Hlvalue::Variable(s_for_set.clone()),
        Hlvalue::Variable(x_for_set.clone()),
    ]);

    // start: chars + direct_call + int_eq(x, 0); branch.
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s.clone()), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let x = variable_with_lltype("x", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![Hlvalue::Constant(c_inner_funcptr), Hlvalue::Variable(chars)],
        Hlvalue::Variable(x.clone()),
    ));
    let is_zero = variable_with_lltype("is_zero", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(x.clone()), signed_const(0)],
        Hlvalue::Variable(is_zero.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_zero));
    startblock.closeblock(vec![
        // True (x == 0): substitute zero-fixup constant.
        Link::new(
            vec![Hlvalue::Variable(s.clone()), signed_const(29872897)],
            Some(block_set_hash.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False: forward the computed x.
        Link::new(
            vec![Hlvalue::Variable(s), Hlvalue::Variable(x)],
            Some(block_set_hash.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_set_hash: setfield(s, 'hash', x_use); return x_use.
    let void_var = variable_with_lltype("__set_hash_void", LowLevelType::Void);
    block_set_hash
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(s_for_set),
                hash_field_const(),
                Hlvalue::Variable(x_for_set.clone()),
            ],
            Hlvalue::Variable(void_var),
        ));
    block_set_hash.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(x_for_set)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_strhash`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// def ll_strhash(s):
///     if s:
///         return jit.conditional_call_elidable(s.hash,
///                                              LLHelpers._ll_strhash, s)
///     else:
///         return 0
/// ```
///
/// 2-block CFG plus the returnblock:
/// - **start**: `nz = ptr_nonzero(s)`. True → `block_lookup`;
///   False → returnblock(`Signed(0)`).
/// - **block_lookup**: `h = getfield(s, 'hash')`, `result =
///   jit_conditional_call_value(h, _ll_strhash_funcptr, s)`. Link to
///   returnblock with `result`. The `jit_conditional_call_value`
///   lloperation returns `h` if `h != 0`, otherwise calls
///   `_ll_strhash(s)` and caches the result back into `s.hash`
///   (rstr.py:396-398 semantics).
///
/// Sub-helper chain: `ll_strhash` -> `_ll_strhash` -> `_hash_string`.
/// All three are registered in the helper-cache and shared across
/// callers via the `(name, args, result)` tuple.
pub(crate) fn build_ll_strhash_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    internal_helper_name: &str,
    inner_helper_name: &str,
) -> Result<PyGraph, TyperError> {
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let hash_field_const =
        || constant_with_lltype(ConstValue::byte_str("hash"), LowLevelType::Void);

    // Materialise (or retrieve cached) `_ll_strhash` sub-helper.
    // `_ll_strhash` itself recursively materialises `_hash_string`.
    let internal_name_owned = internal_helper_name.to_string();
    let inner_name_for_internal = inner_helper_name.to_string();
    let ptr_for_internal = ptr_lltype.clone();
    let internal_helper = rtyper.lowlevel_helper_function_with_builder(
        internal_helper_name.to_string(),
        vec![ptr_lltype.clone()],
        LowLevelType::Signed,
        move |rtyper_inner, _args, _result| {
            build_ll_strhash_internal_helper_graph(
                rtyper_inner,
                &internal_name_owned,
                ptr_for_internal,
                &inner_name_for_internal,
            )
        },
    )?;
    let c_internal_funcptr = sub_helper_funcptr_constant(rtyper, &internal_helper)?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // block_lookup: takes (s); reads cached hash + conditional call.
    let s_for_lookup = variable_with_lltype("s", ptr_lltype.clone());
    let block_lookup = Block::shared(vec![Hlvalue::Variable(s_for_lookup.clone())]);

    // start: ptr_nonzero(s); branch.
    let nz = variable_with_lltype("nz", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_nonzero",
        vec![Hlvalue::Variable(s.clone())],
        Hlvalue::Variable(nz.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s)],
            Some(block_lookup.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![signed_const(0)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_lookup: getfield(s, 'hash') + jit_conditional_call_value.
    let h = variable_with_lltype("h", LowLevelType::Signed);
    block_lookup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(s_for_lookup.clone()), hash_field_const()],
            Hlvalue::Variable(h.clone()),
        ));
    let result = variable_with_lltype("result", LowLevelType::Signed);
    block_lookup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "jit_conditional_call_value",
            vec![
                Hlvalue::Variable(h),
                Hlvalue::Constant(c_internal_funcptr),
                Hlvalue::Variable(s_for_lookup),
            ],
            Hlvalue::Variable(result.clone()),
        ));
    block_lookup.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// lltypesystem/rstr.py `ll_strfasthash(s)`:
/// `ll_assert(s.hash != 0); return s.hash`.
/// Assumes hash is already computed — a single getfield.
pub(crate) fn build_ll_strfasthash_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let hash_field_const =
        || constant_with_lltype(ConstValue::byte_str("hash"), LowLevelType::Void);

    let s = variable_with_lltype("s", ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_hash = variable_with_lltype("hash", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield".to_string(),
        vec![Hlvalue::Variable(s.clone()), hash_field_const()],
        Hlvalue::Variable(v_hash.clone()),
    ));
    // ll_assert(s.hash != 0, "ll_strfasthash: hash==0")
    let v_nz = variable_with_lltype("nz", LowLevelType::Bool);
    let c_zero = constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ne".to_string(),
        vec![Hlvalue::Variable(v_hash.clone()), c_zero],
        Hlvalue::Variable(v_nz.clone()),
    ));
    let c_msg = constant_with_lltype(
        ConstValue::byte_str("ll_strfasthash: hash==0"),
        LowLevelType::Void,
    );
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "debug_assert".to_string(),
        vec![Hlvalue::Variable(v_nz), c_msg],
        constant_with_lltype(ConstValue::None, LowLevelType::Void),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_hash)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_find_char`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// @jit.elidable
/// @signature(types.any(), types.any(), types.int(), types.int(), returns=types.int())
/// def ll_find_char(s, ch, start, end):
///     i = start
///     if end > len(s.chars):
///         end = len(s.chars)
///     while i < end:
///         if s.chars[i] == ch:
///             return i
///         i += 1
///     return -1
/// ```
///
/// 3-block CFG plus the returnblock:
/// - **start**: extract `chars`, `length`. Branch on
///   `int_gt(end, length)`: True → `block_loop_cond` with
///   `(chars, ch, length, start)`; False → `block_loop_cond` with
///   `(chars, ch, end, start)`.
/// - **block_loop_cond**: `int_lt(i, end_clamped)`. True →
///   `block_loop_body`; False → returnblock(`Signed(-1)`).
/// - **block_loop_body**: `c = getarrayitem(chars, i)`,
///   `eq = char_eq/unichar_eq(c, ch)`, `i_next = int_add(i, 1)`.
///   Branch on `eq`: True → returnblock(`i`); False →
///   `block_loop_cond` with `(chars, ch, end_clamped, i_next)`.
///
/// Polymorphic Ptr(STR)/Ptr(UNICODE); element op chosen from chars
/// Array element type. Helper is dead code today; dispatch is
/// `rtype_method_find` follow-up.
pub(crate) fn build_ll_find_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let mut py = build_ll_findlike_char_helper_graph(name, ptr_lltype, FindLikeFlavor::Forward)?;
    // `rlib/jit.py elidable` sets `func._elidable_function_ = True`.
    py.func._elidable_function_ = true;
    if let Some(func) = py.graph.borrow_mut().func.as_mut() {
        func._elidable_function_ = true;
    }
    Ok(py)
}

/// Synthesise the helper graph for `LLHelpers.ll_rfind_char`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// @jit.elidable
/// @signature(types.any(), types.any(), types.int(), types.int(), returns=types.int())
/// def ll_rfind_char(s, ch, start, end):
///     if end > len(s.chars):
///         end = len(s.chars)
///     i = end
///     while i > start:
///         i -= 1
///         if s.chars[i] == ch:
///             return i
///     return -1
/// ```
///
/// Mirror of [`build_ll_find_char_helper_graph`] with the loop
/// running backwards: `i` initialises to `end_clamped` instead of
/// `start`, the loop test is `int_gt(i, start)`, and the body
/// pre-decrements `i` (`i = int_sub(i, 1)`) before reading
/// `chars[i]`.
pub(crate) fn build_ll_rfind_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    build_ll_findlike_char_helper_graph(name, ptr_lltype, FindLikeFlavor::Reverse)
}

/// Discriminator for [`build_ll_findlike_char_helper_graph`]: forward
/// (`ll_find_char`) vs reverse (`ll_rfind_char`).
#[derive(Clone, Copy)]
enum FindLikeFlavor {
    Forward,
    Reverse,
}

/// Shared body for [`build_ll_find_char_helper_graph`] and
/// [`build_ll_rfind_char_helper_graph`]. The two helpers differ in
/// the loop's iteration direction and the condition / index
/// expression; otherwise CFG shape, end-clamp pre-step, and end-of-
/// loop sentinel match.
fn build_ll_findlike_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    flavor: FindLikeFlavor,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_find/rfind_char helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_find/rfind_char helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let char_eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_find/rfind_char helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Inputargs: s, ch, start, end.
    let s = variable_with_lltype("s", ptr_lltype.clone());
    let ch = variable_with_lltype("ch", elem_lltype.clone());
    let start_arg = variable_with_lltype("start", LowLevelType::Signed);
    let end_arg = variable_with_lltype("end", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(ch.clone()),
        Hlvalue::Variable(start_arg.clone()),
        Hlvalue::Variable(end_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Loop carries (chars, ch, end_clamped_or_start_bound, i). For
    // forward the third slot is `end_clamped`; for reverse it is
    // `start_bound` (the i > start_bound test). We name the slot
    // `bound` to capture either flavour.
    let chars_for_cond = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let ch_for_cond = variable_with_lltype("ch", elem_lltype.clone());
    let bound_for_cond = variable_with_lltype("bound", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars_for_cond.clone()),
        Hlvalue::Variable(ch_for_cond.clone()),
        Hlvalue::Variable(bound_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    let chars_for_body = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let ch_for_body = variable_with_lltype("ch", elem_lltype.clone());
    let bound_for_body = variable_with_lltype("bound", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars_for_body.clone()),
        Hlvalue::Variable(ch_for_body.clone()),
        Hlvalue::Variable(bound_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
    ]);

    // start: getsubstruct + getarraysize + int_gt(end, length).
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let end_too_big = variable_with_lltype("end_too_big", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_gt",
        vec![
            Hlvalue::Variable(end_arg.clone()),
            Hlvalue::Variable(length.clone()),
        ],
        Hlvalue::Variable(end_too_big.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(end_too_big));
    // Forward: bound = end_clamped, i = start.
    // Reverse: bound = start (loop while i > bound), i = end_clamped.
    let (initial_i_true, initial_i_false, initial_bound_true, initial_bound_false) = match flavor {
        FindLikeFlavor::Forward => (
            // True: end > length, end := length, i := start, bound := length.
            Hlvalue::Variable(start_arg.clone()),
            Hlvalue::Variable(start_arg),
            Hlvalue::Variable(length.clone()),
            Hlvalue::Variable(end_arg.clone()),
        ),
        FindLikeFlavor::Reverse => (
            // True: end > length, i := length (end_clamped), bound := start.
            // False: i := end, bound := start.
            Hlvalue::Variable(length.clone()),
            Hlvalue::Variable(end_arg.clone()),
            Hlvalue::Variable(start_arg.clone()),
            Hlvalue::Variable(start_arg),
        ),
    };
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(ch.clone()),
                initial_bound_true,
                initial_i_true,
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(ch),
                initial_bound_false,
                initial_i_false,
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_cond: forward int_lt(i, bound), reverse int_gt(i, bound).
    let cond = variable_with_lltype("cond", LowLevelType::Bool);
    let cond_op = match flavor {
        FindLikeFlavor::Forward => "int_lt",
        FindLikeFlavor::Reverse => "int_gt",
    };
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cond_op,
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(bound_for_cond.clone()),
            ],
            Hlvalue::Variable(cond.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_cond),
                Hlvalue::Variable(ch_for_cond),
                Hlvalue::Variable(bound_for_cond),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![signed_const(-1)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_body: shape depends on direction.
    // Forward: c = getarrayitem(chars, i); eq = ...; i_next = int_add(i, 1);
    //   branch on eq: True -> return i; False -> loop_cond(i_next).
    // Reverse: i_next = int_sub(i, 1); c = getarrayitem(chars, i_next);
    //   eq = ...; branch on eq: True -> return i_next; False ->
    //   loop_cond(i_next).
    let (read_idx, idx_var_for_return, idx_for_loop) = match flavor {
        FindLikeFlavor::Forward => {
            // Forward: read at i, return i (current), continue with i_next.
            let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
            let c = variable_with_lltype("c", elem_lltype.clone());
            block_loop_body
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    "getarrayitem",
                    vec![
                        Hlvalue::Variable(chars_for_body.clone()),
                        Hlvalue::Variable(i_for_body.clone()),
                    ],
                    Hlvalue::Variable(c.clone()),
                ));
            let eq = variable_with_lltype("eq", LowLevelType::Bool);
            block_loop_body
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    char_eq_op,
                    vec![Hlvalue::Variable(c), Hlvalue::Variable(ch_for_body.clone())],
                    Hlvalue::Variable(eq.clone()),
                ));
            block_loop_body
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    "int_add",
                    vec![Hlvalue::Variable(i_for_body.clone()), signed_const(1)],
                    Hlvalue::Variable(i_next.clone()),
                ));
            block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(eq));
            (i_for_body.clone(), i_for_body.clone(), i_next)
        }
        FindLikeFlavor::Reverse => {
            // Reverse: pre-decrement, read at i_dec, return i_dec on match.
            let i_dec = variable_with_lltype("i_dec", LowLevelType::Signed);
            block_loop_body
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    "int_sub",
                    vec![Hlvalue::Variable(i_for_body.clone()), signed_const(1)],
                    Hlvalue::Variable(i_dec.clone()),
                ));
            let c = variable_with_lltype("c", elem_lltype.clone());
            block_loop_body
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    "getarrayitem",
                    vec![
                        Hlvalue::Variable(chars_for_body.clone()),
                        Hlvalue::Variable(i_dec.clone()),
                    ],
                    Hlvalue::Variable(c.clone()),
                ));
            let eq = variable_with_lltype("eq", LowLevelType::Bool);
            block_loop_body
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    char_eq_op,
                    vec![Hlvalue::Variable(c), Hlvalue::Variable(ch_for_body.clone())],
                    Hlvalue::Variable(eq.clone()),
                ));
            block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(eq));
            (i_dec.clone(), i_dec.clone(), i_dec)
        }
    };
    let _ = read_idx;
    block_loop_body.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(idx_var_for_return)],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_body),
                Hlvalue::Variable(ch_for_body),
                Hlvalue::Variable(bound_for_body),
                Hlvalue::Variable(idx_for_loop),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "s".to_string(),
            "ch".to_string(),
            "start".to_string(),
            "end".to_string(),
        ],
        func,
    ))
}

/// `HighLevelOp.gendirectcall` / `RPythonTyper.getcallable`: annotate the
/// callee graph (`lowlevel_helper_function_with_builder` is the
/// `MixLevelHelperAnnotator` / `cachedgraph` stand-in) and build the
/// funcptr from that graph so `funcobj.graph` is set.
fn cached_helper_funcptr(
    rtyper: &RPythonTyper,
    name: &str,
    args: Vec<LowLevelType>,
    result: LowLevelType,
    builder: impl FnOnce(&RPythonTyper) -> Result<PyGraph, TyperError>,
) -> Result<Hlvalue, TyperError> {
    let name_owned = name.to_string();
    let helper = rtyper.lowlevel_helper_function_with_builder(
        name_owned,
        args,
        result,
        move |rtyper, _args, _result| builder(rtyper),
    )?;
    Ok(Hlvalue::Constant(sub_helper_funcptr_constant(
        rtyper, &helper,
    )?))
}

/// The `_func` object of a `direct_call` callee constant.
pub(crate) fn funcobj_from_funcptr_constant(hlv: &Hlvalue) -> Option<_func> {
    let Hlvalue::Constant(c) = hlv else {
        return None;
    };
    let ConstValue::LLPtr(ptr) = &c.value else {
        return None;
    };
    match ptr._obj0_value() {
        Ok(Some(_ptr_obj::Func(func))) => Some(func),
        _ => None,
    }
}

/// Every `direct_call` callee funcobj in `graph`.
pub(crate) fn direct_call_funcobjs(graph: &FunctionGraph) -> Vec<_func> {
    graph
        .iterblocks()
        .into_iter()
        .flat_map(|block| {
            block
                .borrow()
                .operations
                .iter()
                .filter(|op| op.opname == "direct_call")
                .filter_map(|op| op.args.first().and_then(funcobj_from_funcptr_constant))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `record_extra_call_by_graph_id`: the live graph `funcobj.graph` names
/// lives on `translator.graphs`.
pub(crate) fn graph_ref_for_funcptr_id(rtyper: &RPythonTyper, graph_id: usize) -> Option<GraphRef> {
    let ann = rtyper.annotator.upgrade()?;
    ann.translator
        .graphs
        .borrow()
        .iter()
        .find(|graph| GraphKey::of(graph).as_usize() == graph_id)
        .cloned()
}

fn require_str_ptr(ptr_lltype: &LowLevelType, helper: &str) -> Result<(), TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "{helper} expects Ptr(Array(Char)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "{helper} Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    match arr.OF {
        LowLevelType::Char | LowLevelType::UniChar => Ok(()),
        ref other => Err(TyperError::message(format!(
            "{helper} expects Ptr(Array(Char|UniChar)), got {other:?}"
        ))),
    }
}

fn emit_record_exact_value_and_return(
    block: &crate::flowspace::model::BlockRef,
    res: Variable,
    end: Variable,
    returnblock: crate::flowspace::model::BlockRef,
) {
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(res.clone()), Hlvalue::Variable(end)],
        Hlvalue::Variable(lt.clone()),
    ));
    let void_res = variable_with_lltype("v", LowLevelType::Void);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "jit_record_exact_value",
        vec![
            Hlvalue::Variable(lt),
            constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool),
        ],
        Hlvalue::Variable(void_res),
    ));
    block.closeblock(vec![
        Link::new(vec![Hlvalue::Variable(res)], Some(returnblock), None).into_ref(),
    ]);
}

/// Synthesise the helper graph for `LLHelpers.ll_find`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// @signature(types.any(), types.any(), types.int(), types.int(), returns=types.int())
/// def ll_find(s1, s2, start, end):
///     from rpython.rlib.rstring import SEARCH_FIND
///     if start < 0:
///         start = 0
///     n = len(s1.chars)
///     if end > n:
///         end = n
///     if end - start < 0:
///         return -1
///     m = len(s2.chars)
///     if m <= 1:
///         if m == 0:
///             return start
///         res = LLHelpers.ll_find_char(s1, s2.chars[0], start, end)
///         jit.record_exact_value(res < end, True)
///         return res
///     res = LLHelpers.ll_search(s1, s2, start, end, SEARCH_FIND)
///     jit.record_exact_value(res < end, True)
///     return res
/// ```
///
/// `Ptr(STR)` and `Ptr(UNICODE)`. UNICODE `ll_search` takes
/// `_search_normal` (look-inside); STR takes `_search_elidable`.
///
/// Callee funcptrs are `getcallable` of the helper graphs
/// (`HighLevelOp.gendirectcall`), so `funcobj.graph` is set. Callers
/// that keep `rtyper` alive (`build_ll_find_helper_graph_on`) can look
/// those graphs up on `translator.graphs`.
///
/// Test-only: a throwaway annotator/rtyper. Production must call
/// [`build_ll_find_helper_graph_on`] so callees stay on the caller's
/// translator (`RPythonTyper.annotate_helper`).
#[cfg(test)]
pub(crate) fn build_ll_find_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let ann = RPythonAnnotator::new(None, None, None, false);
    let rtyper = RPythonTyper::new(&ann);
    build_ll_find_helper_graph_on(&rtyper, name, ptr_lltype)
}

/// [`build_ll_find_helper_graph`] against a live rtyper so callee graphs
/// stay on `translator.graphs` for `funcobj.graph` lookup.
pub(crate) fn build_ll_find_helper_graph_on(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    require_str_ptr(&ptr_lltype, "ll_find")?;
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s1 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2 = variable_with_lltype("s2", ptr_lltype.clone());
    let start_arg = variable_with_lltype("start", LowLevelType::Signed);
    let end_arg = variable_with_lltype("end", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s1.clone()),
        Hlvalue::Variable(s2.clone()),
        Hlvalue::Variable(start_arg.clone()),
        Hlvalue::Variable(end_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let s1_for_n = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_n = variable_with_lltype("s2", ptr_lltype.clone());
    let start_for_n = variable_with_lltype("start", LowLevelType::Signed);
    let end_for_n = variable_with_lltype("end", LowLevelType::Signed);
    let block_n = Block::shared(vec![
        Hlvalue::Variable(s1_for_n.clone()),
        Hlvalue::Variable(s2_for_n.clone()),
        Hlvalue::Variable(start_for_n.clone()),
        Hlvalue::Variable(end_for_n.clone()),
    ]);

    let s1_for_range = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_range = variable_with_lltype("s2", ptr_lltype.clone());
    let start_for_range = variable_with_lltype("start", LowLevelType::Signed);
    let end_for_range = variable_with_lltype("end", LowLevelType::Signed);
    let block_range = Block::shared(vec![
        Hlvalue::Variable(s1_for_range.clone()),
        Hlvalue::Variable(s2_for_range.clone()),
        Hlvalue::Variable(start_for_range.clone()),
        Hlvalue::Variable(end_for_range.clone()),
    ]);

    let s1_for_m = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_m = variable_with_lltype("s2", ptr_lltype.clone());
    let start_for_m = variable_with_lltype("start", LowLevelType::Signed);
    let end_for_m = variable_with_lltype("end", LowLevelType::Signed);
    let block_m = Block::shared(vec![
        Hlvalue::Variable(s1_for_m.clone()),
        Hlvalue::Variable(s2_for_m.clone()),
        Hlvalue::Variable(start_for_m.clone()),
        Hlvalue::Variable(end_for_m.clone()),
    ]);

    let s1_for_le1 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_le1 = variable_with_lltype("s2", ptr_lltype.clone());
    let start_for_le1 = variable_with_lltype("start", LowLevelType::Signed);
    let end_for_le1 = variable_with_lltype("end", LowLevelType::Signed);
    let m_for_le1 = variable_with_lltype("m", LowLevelType::Signed);
    let block_m_le_1 = Block::shared(vec![
        Hlvalue::Variable(s1_for_le1.clone()),
        Hlvalue::Variable(s2_for_le1.clone()),
        Hlvalue::Variable(start_for_le1.clone()),
        Hlvalue::Variable(end_for_le1.clone()),
        Hlvalue::Variable(m_for_le1.clone()),
    ]);

    let s1_for_char = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_char = variable_with_lltype("s2", ptr_lltype.clone());
    let start_for_char = variable_with_lltype("start", LowLevelType::Signed);
    let end_for_char = variable_with_lltype("end", LowLevelType::Signed);
    let block_find_char = Block::shared(vec![
        Hlvalue::Variable(s1_for_char.clone()),
        Hlvalue::Variable(s2_for_char.clone()),
        Hlvalue::Variable(start_for_char.clone()),
        Hlvalue::Variable(end_for_char.clone()),
    ]);

    let s1_for_search = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_search = variable_with_lltype("s2", ptr_lltype.clone());
    let start_for_search = variable_with_lltype("start", LowLevelType::Signed);
    let end_for_search = variable_with_lltype("end", LowLevelType::Signed);
    let block_search = Block::shared(vec![
        Hlvalue::Variable(s1_for_search.clone()),
        Hlvalue::Variable(s2_for_search.clone()),
        Hlvalue::Variable(start_for_search.clone()),
        Hlvalue::Variable(end_for_search.clone()),
    ]);

    let start_neg = variable_with_lltype("start_neg", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(start_arg.clone()), signed_const(0)],
        Hlvalue::Variable(start_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(start_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s1.clone()),
                Hlvalue::Variable(s2.clone()),
                signed_const(0),
                Hlvalue::Variable(end_arg.clone()),
            ],
            Some(block_n.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(s1),
                Hlvalue::Variable(s2),
                Hlvalue::Variable(start_arg),
                Hlvalue::Variable(end_arg),
            ],
            Some(block_n.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars1 = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    block_n.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s1_for_n.clone()), chars_field_const()],
        Hlvalue::Variable(chars1.clone()),
    ));
    let n = variable_with_lltype("n", LowLevelType::Signed);
    block_n.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars1)],
        Hlvalue::Variable(n.clone()),
    ));
    let end_too_big = variable_with_lltype("end_too_big", LowLevelType::Bool);
    block_n.borrow_mut().operations.push(SpaceOperation::new(
        "int_gt",
        vec![
            Hlvalue::Variable(end_for_n.clone()),
            Hlvalue::Variable(n.clone()),
        ],
        Hlvalue::Variable(end_too_big.clone()),
    ));
    block_n.borrow_mut().exitswitch = Some(Hlvalue::Variable(end_too_big));
    block_n.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s1_for_n.clone()),
                Hlvalue::Variable(s2_for_n.clone()),
                Hlvalue::Variable(start_for_n.clone()),
                Hlvalue::Variable(n),
            ],
            Some(block_range.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(s1_for_n),
                Hlvalue::Variable(s2_for_n),
                Hlvalue::Variable(start_for_n),
                Hlvalue::Variable(end_for_n),
            ],
            Some(block_range.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let span = variable_with_lltype("span", LowLevelType::Signed);
    block_range
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![
                Hlvalue::Variable(end_for_range.clone()),
                Hlvalue::Variable(start_for_range.clone()),
            ],
            Hlvalue::Variable(span.clone()),
        ));
    let span_neg = variable_with_lltype("span_neg", LowLevelType::Bool);
    block_range
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![Hlvalue::Variable(span), signed_const(0)],
            Hlvalue::Variable(span_neg.clone()),
        ));
    block_range.borrow_mut().exitswitch = Some(Hlvalue::Variable(span_neg));
    block_range.closeblock(vec![
        Link::new(
            vec![signed_const(-1)],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(s1_for_range),
                Hlvalue::Variable(s2_for_range),
                Hlvalue::Variable(start_for_range),
                Hlvalue::Variable(end_for_range),
            ],
            Some(block_m.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars2 = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    block_m.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s2_for_m.clone()), chars_field_const()],
        Hlvalue::Variable(chars2.clone()),
    ));
    let m = variable_with_lltype("m", LowLevelType::Signed);
    block_m.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars2)],
        Hlvalue::Variable(m.clone()),
    ));
    let m_le_1 = variable_with_lltype("m_le_1", LowLevelType::Bool);
    block_m.borrow_mut().operations.push(SpaceOperation::new(
        "int_le",
        vec![Hlvalue::Variable(m.clone()), signed_const(1)],
        Hlvalue::Variable(m_le_1.clone()),
    ));
    block_m.borrow_mut().exitswitch = Some(Hlvalue::Variable(m_le_1));
    block_m.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s1_for_m.clone()),
                Hlvalue::Variable(s2_for_m.clone()),
                Hlvalue::Variable(start_for_m.clone()),
                Hlvalue::Variable(end_for_m.clone()),
                Hlvalue::Variable(m),
            ],
            Some(block_m_le_1.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(s1_for_m),
                Hlvalue::Variable(s2_for_m),
                Hlvalue::Variable(start_for_m),
                Hlvalue::Variable(end_for_m),
            ],
            Some(block_search.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let m_eq_0 = variable_with_lltype("m_eq_0", LowLevelType::Bool);
    block_m_le_1
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(m_for_le1), signed_const(0)],
            Hlvalue::Variable(m_eq_0.clone()),
        ));
    block_m_le_1.borrow_mut().exitswitch = Some(Hlvalue::Variable(m_eq_0));
    block_m_le_1.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(start_for_le1.clone())],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(s1_for_le1),
                Hlvalue::Variable(s2_for_le1),
                Hlvalue::Variable(start_for_le1),
                Hlvalue::Variable(end_for_le1),
            ],
            Some(block_find_char.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars2_char = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    block_find_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s2_for_char.clone()), chars_field_const()],
            Hlvalue::Variable(chars2_char.clone()),
        ));
    let elem_lltype = match &chars_array_ptr_lltype {
        LowLevelType::Ptr(chars_ptr) => match &chars_ptr.TO {
            PtrTarget::Array(arr) => arr.OF.clone(),
            other => {
                return Err(TyperError::message(format!(
                    "ll_find chars Ptr.TO must be Array, got {other:?}"
                )));
            }
        },
        other => {
            return Err(TyperError::message(format!(
                "ll_find chars must be Ptr(Array), got {other:?}"
            )));
        }
    };
    let ch = variable_with_lltype("ch", elem_lltype.clone());
    block_find_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![Hlvalue::Variable(chars2_char), signed_const(0)],
            Hlvalue::Variable(ch.clone()),
        ));
    let find_char_ptr = cached_helper_funcptr(
        rtyper,
        "ll_find_char",
        vec![
            ptr_lltype.clone(),
            elem_lltype,
            LowLevelType::Signed,
            LowLevelType::Signed,
        ],
        LowLevelType::Signed,
        {
            let ptr = ptr_lltype.clone();
            move |_rtyper| build_ll_find_char_helper_graph("ll_find_char", ptr)
        },
    )?;
    let res_char = variable_with_lltype("res", LowLevelType::Signed);
    block_find_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                find_char_ptr,
                Hlvalue::Variable(s1_for_char),
                Hlvalue::Variable(ch),
                Hlvalue::Variable(start_for_char),
                Hlvalue::Variable(end_for_char.clone()),
            ],
            Hlvalue::Variable(res_char.clone()),
        ));
    emit_record_exact_value_and_return(
        &block_find_char,
        res_char,
        end_for_char,
        graph.returnblock.clone(),
    );

    // `rlib/rstring.py` `SEARCH_FIND`. Named constant: `majit_rlib::rstring::SEARCH_FIND`.
    let search_find = signed_const(1);
    let search_ptr = cached_helper_funcptr(
        rtyper,
        "ll_search",
        vec![
            ptr_lltype.clone(),
            ptr_lltype.clone(),
            LowLevelType::Signed,
            LowLevelType::Signed,
            LowLevelType::Signed,
        ],
        LowLevelType::Signed,
        {
            let ptr = ptr_lltype;
            move |rtyper| build_ll_search_helper_graph_on(rtyper, "ll_search", ptr)
        },
    )?;
    let res_search = variable_with_lltype("res", LowLevelType::Signed);
    block_search
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                search_ptr,
                Hlvalue::Variable(s1_for_search),
                Hlvalue::Variable(s2_for_search),
                Hlvalue::Variable(start_for_search),
                Hlvalue::Variable(end_for_search.clone()),
                search_find,
            ],
            Hlvalue::Variable(res_search.clone()),
        ));
    emit_record_exact_value_and_return(
        &block_search,
        res_search,
        end_for_search,
        graph.returnblock.clone(),
    );

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "s1".to_string(),
            "s2".to_string(),
            "start".to_string(),
            "end".to_string(),
        ],
        func,
    ))
}

/// Synthesise `LLHelpers.ll_search` (`rstr.py`).
///
/// `hlstr` / `hlunicode` are identity on the low-level pointer. `_search`
/// dispatches on `isinstance(value, str)` at translation time: STR is
/// `direct_call _search_elidable`; UNICODE is `direct_call _search_normal`.
///
/// Test-only: a throwaway annotator/rtyper. Production must call
/// [`build_ll_search_helper_graph_on`] so the `_search_*` callee stays
/// on the caller's translator (`RPythonTyper.annotate_helper`).
#[cfg(test)]
pub(crate) fn build_ll_search_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let ann = RPythonAnnotator::new(None, None, None, false);
    let rtyper = RPythonTyper::new(&ann);
    build_ll_search_helper_graph_on(&rtyper, name, ptr_lltype)
}

/// `assert p is not None` on a `Ptr(STR)` / `Ptr(UNICODE)`:
/// `ptr_nonzero` then the False arm raises AssertionError
/// (`LLHelpers.ll_search`; flowspace `assert` lowering).
fn close_assert_ptr_not_none(
    block: &BlockRef,
    ptr: &Variable,
    continue_args: Vec<Hlvalue>,
    continue_block: BlockRef,
    exceptblock: BlockRef,
    exc_args: Vec<Hlvalue>,
) {
    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let nz = variable_with_lltype("nz", LowLevelType::Bool);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_nonzero",
        vec![Hlvalue::Variable(ptr.clone())],
        Hlvalue::Variable(nz.clone()),
    ));
    block.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz));
    block.closeblock(vec![
        Link::new(continue_args, Some(continue_block), Some(bool_true())).into_ref(),
        Link::new(exc_args, Some(exceptblock), Some(bool_false())).into_ref(),
    ]);
}

fn search_arg_vars(
    ptr_lltype: &LowLevelType,
) -> (Variable, Variable, Variable, Variable, Variable) {
    (
        variable_with_lltype("s1", ptr_lltype.clone()),
        variable_with_lltype("s2", ptr_lltype.clone()),
        variable_with_lltype("start", LowLevelType::Signed),
        variable_with_lltype("end", LowLevelType::Signed),
        variable_with_lltype("mode", LowLevelType::Signed),
    )
}

fn search_carry(
    s1: &Variable,
    s2: &Variable,
    start: &Variable,
    end: &Variable,
    mode: &Variable,
) -> Vec<Hlvalue> {
    vec![
        Hlvalue::Variable(s1.clone()),
        Hlvalue::Variable(s2.clone()),
        Hlvalue::Variable(start.clone()),
        Hlvalue::Variable(end.clone()),
        Hlvalue::Variable(mode.clone()),
    ]
}

/// [`build_ll_search_helper_graph`] against a live rtyper so the
/// callee graph stays on `translator.graphs`.
pub(crate) fn build_ll_search_helper_graph_on(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    require_str_ptr(&ptr_lltype, "ll_search")?;
    let kind = crate::rlib::rstring::StringKind::from_ptr(ptr_lltype.clone())?;

    let (s1, s2, start_arg, end_arg, mode_arg) = search_arg_vars(&ptr_lltype);
    let startblock = Block::shared(search_carry(&s1, &s2, &start_arg, &end_arg, &mode_arg));
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let (s1_other, s2_other, start_other, end_other, mode_other) = search_arg_vars(&ptr_lltype);
    let block_check_other = Block::shared(search_carry(
        &s1_other,
        &s2_other,
        &start_other,
        &end_other,
        &mode_other,
    ));
    let (s1_call, s2_call, start_call, end_call, mode_call) = search_arg_vars(&ptr_lltype);
    let block_search = Block::shared(search_carry(
        &s1_call,
        &s2_call,
        &start_call,
        &end_call,
        &mode_call,
    ));

    let search_args = vec![
        ptr_lltype.clone(),
        ptr_lltype.clone(),
        LowLevelType::Signed,
        LowLevelType::Signed,
        LowLevelType::Signed,
    ];
    let (callee_name, callee_ptr) = if kind.is_unicode() {
        (
            "_search_normal",
            cached_helper_funcptr(
                rtyper,
                "_search_normal",
                search_args.clone(),
                LowLevelType::Signed,
                {
                    let ptr = ptr_lltype;
                    move |rtyper| {
                        crate::rlib::rstring::build_search_normal_helper_graph(
                            rtyper,
                            "_search_normal",
                            ptr,
                        )
                    }
                },
            )?,
        )
    } else {
        (
            "_search_elidable",
            cached_helper_funcptr(
                rtyper,
                "_search_elidable",
                search_args,
                LowLevelType::Signed,
                {
                    let ptr = ptr_lltype;
                    move |rtyper| {
                        crate::rlib::rstring::build_search_elidable_helper_graph(
                            rtyper,
                            "_search_elidable",
                            ptr,
                        )
                    }
                },
            )?,
        )
    };
    let _ = callee_name;

    // `LLHelpers.ll_search`: after `hlstr` / `hlunicode`,
    // `assert value is not None` / `assert other is not None`.
    let exc_args = exception_args("AssertionError")?;
    close_assert_ptr_not_none(
        &startblock,
        &s1,
        search_carry(&s1, &s2, &start_arg, &end_arg, &mode_arg),
        block_check_other.clone(),
        graph.exceptblock.clone(),
        exc_args.clone(),
    );
    close_assert_ptr_not_none(
        &block_check_other,
        &s2_other,
        search_carry(&s1_other, &s2_other, &start_other, &end_other, &mode_other),
        block_search.clone(),
        graph.exceptblock.clone(),
        exc_args,
    );

    let res = variable_with_lltype("res", LowLevelType::Signed);
    block_search
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                callee_ptr,
                Hlvalue::Variable(s1_call),
                Hlvalue::Variable(s2_call),
                Hlvalue::Variable(start_call),
                Hlvalue::Variable(end_call),
                Hlvalue::Variable(mode_call),
            ],
            Hlvalue::Variable(res.clone()),
        ));
    block_search.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(res)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "s1".to_string(),
            "s2".to_string(),
            "start".to_string(),
            "end".to_string(),
            "mode".to_string(),
        ],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_count_char`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// @jit.elidable
/// def ll_count_char(s, ch, start, end):
///     count = 0
///     i = start
///     if end > len(s.chars):
///         end = len(s.chars)
///     while i < end:
///         if s.chars[i] == ch:
///             count += 1
///         i += 1
///     return count
/// ```
///
/// 3-block CFG plus the returnblock. Same shape as
/// [`build_ll_find_char_helper_graph`] with a `count` accumulator
/// carried through the loop:
/// - **start**: getsubstruct + getarraysize + int_gt(end, length).
///   True → `block_loop_cond` with `(chars, ch, length, start, 0)`;
///   False → `block_loop_cond` with `(chars, ch, end, start, 0)`.
/// - **block_loop_cond**: int_lt(i, end_clamped). True →
///   `block_loop_body`; False → returnblock(`count`).
/// - **block_loop_body**: getarrayitem + char_eq + int_add(i, 1)
///   `i_next` + int_add(count, 1) `count_inc`. Branch on `eq`:
///   True → loop_cond with `count_inc`; False → loop_cond with
///   `count`.
pub(crate) fn build_ll_count_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_count_char helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_count_char helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let char_eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_count_char helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let ch = variable_with_lltype("ch", elem_lltype.clone());
    let start_arg = variable_with_lltype("start", LowLevelType::Signed);
    let end_arg = variable_with_lltype("end", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(ch.clone()),
        Hlvalue::Variable(start_arg.clone()),
        Hlvalue::Variable(end_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Loop carries (chars, ch, end_clamped, i, count).
    let chars_for_cond = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let ch_for_cond = variable_with_lltype("ch", elem_lltype.clone());
    let end_for_cond = variable_with_lltype("end_clamped", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let count_for_cond = variable_with_lltype("count", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars_for_cond.clone()),
        Hlvalue::Variable(ch_for_cond.clone()),
        Hlvalue::Variable(end_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
        Hlvalue::Variable(count_for_cond.clone()),
    ]);

    let chars_for_body = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let ch_for_body = variable_with_lltype("ch", elem_lltype.clone());
    let end_for_body = variable_with_lltype("end_clamped", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let count_for_body = variable_with_lltype("count", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars_for_body.clone()),
        Hlvalue::Variable(ch_for_body.clone()),
        Hlvalue::Variable(end_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
        Hlvalue::Variable(count_for_body.clone()),
    ]);

    // start: getsubstruct + getarraysize + int_gt(end, length).
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let end_too_big = variable_with_lltype("end_too_big", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_gt",
        vec![
            Hlvalue::Variable(end_arg.clone()),
            Hlvalue::Variable(length.clone()),
        ],
        Hlvalue::Variable(end_too_big.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(end_too_big));
    startblock.closeblock(vec![
        // True (end > length): clamp end := length.
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(ch.clone()),
                Hlvalue::Variable(length),
                Hlvalue::Variable(start_arg.clone()),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False: keep end.
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(ch),
                Hlvalue::Variable(end_arg),
                Hlvalue::Variable(start_arg),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_cond: int_lt(i, end_clamped); branch.
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(end_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_cond),
                Hlvalue::Variable(ch_for_cond),
                Hlvalue::Variable(end_for_cond),
                Hlvalue::Variable(i_for_cond),
                Hlvalue::Variable(count_for_cond.clone()),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(count_for_cond)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_body: getarrayitem + char_eq + int_add(i, 1) +
    // int_add(count, 1); branch on eq.
    let c = variable_with_lltype("c", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    let eq = variable_with_lltype("eq", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            char_eq_op,
            vec![Hlvalue::Variable(c), Hlvalue::Variable(ch_for_body.clone())],
            Hlvalue::Variable(eq.clone()),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    let count_inc = variable_with_lltype("count_inc", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(count_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(count_inc.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(eq));
    block_loop_body.closeblock(vec![
        // True (match): cycle to loop_cond with count_inc.
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_body.clone()),
                Hlvalue::Variable(ch_for_body.clone()),
                Hlvalue::Variable(end_for_body.clone()),
                Hlvalue::Variable(i_next.clone()),
                Hlvalue::Variable(count_inc),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False (no match): cycle to loop_cond with count unchanged.
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_body),
                Hlvalue::Variable(ch_for_body),
                Hlvalue::Variable(end_for_body),
                Hlvalue::Variable(i_next),
                Hlvalue::Variable(count_for_body),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "s".to_string(),
            "ch".to_string(),
            "start".to_string(),
            "end".to_string(),
        ],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_stritem_nonneg`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// @signature(types.any(), types.int(), returns=types.any())
/// def ll_stritem_nonneg(s, i):
///     chars = s.chars
///     ll_assert(i >= 0, "negative str getitem index")
///     ll_assert(i < len(chars), "str getitem index out of bound")
///     return chars[i]
/// ```
///
/// Single-block CFG: `chars = getsubstruct(s, 'chars')`, `c =
/// getarrayitem(chars, i)`, link to returnblock with `c`. The
/// `ll_assert` calls are debug-only annotations in upstream and
/// are not lowered to lloperations (`rffi.py` no-op definition for
/// production translation).
///
/// Polymorphic Ptr(STR)/Ptr(UNICODE); return lltype is the chars
/// Array element type (Char for STR, UniChar for UNICODE).
pub(crate) fn build_ll_stritem_nonneg_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_stritem_nonneg helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_stritem_nonneg helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_stritem_nonneg helper unsupported chars element type {elem_lltype:?}"
        )));
    }

    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let i = variable_with_lltype("i", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", elem_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let c = variable_with_lltype("c", elem_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarrayitem",
        vec![Hlvalue::Variable(chars), Hlvalue::Variable(i)],
        Hlvalue::Variable(c.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "i".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `AbstractLLHelpers.ll_stritem`
/// (`rtyper/rstr.py`):
///
/// ```python
/// @classmethod
/// def ll_stritem(cls, s, i):
///     if i < 0:
///         i += cls.ll_strlen(s)
///     return cls.ll_stritem_nonneg(s, i)
/// ```
///
/// 3-block CFG plus the returnblock:
/// - **start**: `is_neg = int_lt(i, 0)`. True → `block_neg_fix(s,
///   i)`; False → `block_dispatch(s, i)` (direct).
/// - **block_neg_fix**: `chars = getsubstruct(s, 'chars')`,
///   `length = getarraysize(chars)`, `i_fix = int_add(i, length)`.
///   Unconditional link to `block_dispatch(s, i_fix)`.
/// - **block_dispatch**: `c = direct_call(ll_stritem_nonneg, s,
///   i_eff)`. Link to returnblock with `c`.
///
/// `length` is computed inline via getsubstruct/getarraysize rather
/// than a `ll_strlen` sub-helper — inline emit matches the same shape
/// the RPython compiler would produce after inlining the `ll_strlen`
/// 1-line wrapper.
///
/// `inner_helper_name` is the cache key for the `ll_stritem_nonneg`
/// sub-helper (`ll_stritem_nonneg` for STR pair,
/// `ll_unicode_stritem_nonneg` for UNICODE pair).
pub(crate) fn build_ll_stritem_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_stritem helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_stritem helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_stritem helper unsupported chars element type {elem_lltype:?}"
        )));
    }

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Materialise (or retrieve cached) ll_stritem_nonneg sub-helper.
    let inner_name_owned = inner_helper_name.to_string();
    let ptr_for_inner = ptr_lltype.clone();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![ptr_lltype.clone(), LowLevelType::Signed],
        elem_lltype.clone(),
        move |_rtyper, _args, _result| {
            build_ll_stritem_nonneg_helper_graph(&inner_name_owned, ptr_for_inner)
        },
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let i = variable_with_lltype("i", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", elem_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // block_neg_fix: takes (s, i); computes i + length.
    let s_for_fix = variable_with_lltype("s", ptr_lltype.clone());
    let i_for_fix = variable_with_lltype("i", LowLevelType::Signed);
    let block_neg_fix = Block::shared(vec![
        Hlvalue::Variable(s_for_fix.clone()),
        Hlvalue::Variable(i_for_fix.clone()),
    ]);

    // block_dispatch: takes (s, i_eff); direct_calls ll_stritem_nonneg.
    let s_for_dispatch = variable_with_lltype("s", ptr_lltype.clone());
    let i_for_dispatch = variable_with_lltype("i_eff", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(s_for_dispatch.clone()),
        Hlvalue::Variable(i_for_dispatch.clone()),
    ]);

    // start: int_lt(i, 0); branch.
    let is_neg = variable_with_lltype("is_neg", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(i.clone()), signed_const(0)],
        Hlvalue::Variable(is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s.clone()), Hlvalue::Variable(i.clone())],
            Some(block_neg_fix.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(s), Hlvalue::Variable(i)],
            Some(block_dispatch.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_neg_fix: chars + length + i_fix = i + length.
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    block_neg_fix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s_for_fix.clone()), chars_field_const()],
            Hlvalue::Variable(chars.clone()),
        ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    block_neg_fix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(chars)],
            Hlvalue::Variable(length.clone()),
        ));
    let i_fix = variable_with_lltype("i_fix", LowLevelType::Signed);
    block_neg_fix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_fix), Hlvalue::Variable(length)],
            Hlvalue::Variable(i_fix.clone()),
        ));
    block_neg_fix.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s_for_fix), Hlvalue::Variable(i_fix)],
            Some(block_dispatch.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_dispatch: direct_call(ll_stritem_nonneg, s, i_eff).
    let c = variable_with_lltype("c", elem_lltype);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_inner_funcptr),
                Hlvalue::Variable(s_for_dispatch),
                Hlvalue::Variable(i_for_dispatch),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    block_dispatch.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "i".to_string()],
        func,
    ))
}

/// Synthesise a `ll_isdigit/isalpha/isalnum/isspace/isupper/islower`-
/// style `string-isxxx` helper graph that loops over chars and
/// `direct_call`s a per-char predicate sub-helper. RPython
/// `AbstractStringRepr.rtype_method_isdigit` etc. (`rstr.py`)
/// ultimately lower to `ll_isdigit(s)` / `ll_isalpha(s)` etc., which
/// in turn iterate `s.chars` and call the matching `ll_char_isxxx`
/// per-char predicate.
///
/// 3-block CFG plus the returnblock:
/// - **start**: `chars = getsubstruct(s, 'chars')`, `length =
///   getarraysize(chars)`, `is_empty = int_eq(length, 0)`. True →
///   returnblock(`Bool(false)`); False → `block_loop_cond` with
///   `(chars, length, 0)`.
/// - **block_loop_cond**: `lt = int_lt(i, length)`. True →
///   `block_loop_body`; False → returnblock(`Bool(true)`).
/// - **block_loop_body**: `c = getarrayitem(chars, i)`,
///   `is_d = direct_call(per-char-predicate, c)`,
///   `i_next = int_add(i, 1)`. Branch on `is_d`: True →
///   `block_loop_cond` with `i_next`; False → returnblock(`Bool(false)`).
///
/// `inner_helper_name` / `build_inner_helper` are the cache key and
/// builder for the per-char predicate (e.g. `ll_char_isdigit` /
/// `build_ll_char_isdigit_helper_graph`). The helper expects the
/// sub-helper signature to be `(elem) -> Bool`.
pub(crate) fn build_ll_string_isxxx_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
    build_inner_helper: impl FnOnce(&str) -> Result<PyGraph, TyperError> + 'static,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_string_isxxx helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_string_isxxx helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_string_isxxx helper unsupported chars element type {elem_lltype:?}"
        )));
    }

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Materialise (or retrieve cached) per-char predicate sub-helper.
    let inner_name_owned = inner_helper_name.to_string();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![elem_lltype.clone()],
        LowLevelType::Bool,
        move |_rtyper, _args, _result| build_inner_helper(&inner_name_owned),
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let chars_for_cond = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let length_for_cond = variable_with_lltype("length", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars_for_cond.clone()),
        Hlvalue::Variable(length_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    let chars_for_body = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let length_for_body = variable_with_lltype("length", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars_for_body.clone()),
        Hlvalue::Variable(length_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
    ]);

    // start: getsubstruct + getarraysize + int_eq(length, 0).
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let is_empty = variable_with_lltype("is_empty", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(length.clone()), signed_const(0)],
        Hlvalue::Variable(is_empty.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_empty));
    startblock.closeblock(vec![
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(length),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_cond: int_lt(i, length).
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(length_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_cond),
                Hlvalue::Variable(length_for_cond),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_true()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_body: c + direct_call(predicate, c) + int_add.
    let c = variable_with_lltype("c", elem_lltype);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    let is_d = variable_with_lltype("is_d", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(c_inner_funcptr), Hlvalue::Variable(c)],
            Hlvalue::Variable(is_d.clone()),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_d));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_body),
                Hlvalue::Variable(length_for_body),
                Hlvalue::Variable(i_next),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise `LLHelpers.ll_upper` / `LLHelpers.ll_lower`
/// (`lltypesystem/rstr.py`):
///
/// ```python
/// def ll_upper(s):
///     s_chars = s.chars
///     s_len = len(s_chars)
///     if s_len == 0:
///         return s.empty()
///     i = 0
///     result = mallocstr(s_len)
///     while i < s_len:
///         result.chars[i] = LLHelpers.ll_upper_char(s_chars[i])
///         i += 1
///     return result
/// ```
///
/// The helper is intentionally `Ptr(STR)`-only. Upstream uses
/// `mallocstr` specifically so calling it for unicode explodes; the
/// Rust dispatcher mirrors that by never routing UnicodeRepr here, and
/// the builder rejects non-Char arrays.
///
/// The empty branch returns the incoming empty string object. Upstream
/// returns `s.empty()` (the canonical empty string), but avoiding a new
/// allocation preserves the important low-level invariant of the fast
/// path until the full string adtmeth singleton is wired.
pub(crate) fn build_ll_string_casefold_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
    build_inner_helper: impl FnOnce(&str) -> Result<PyGraph, TyperError> + 'static,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_string_casefold helper expects Ptr(Array(Char)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_string_casefold helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    if arr.OF != LowLevelType::Char {
        return Err(TyperError::message(format!(
            "ll_string_casefold helper only supports Char arrays, got {:?}",
            arr.OF
        )));
    }
    let struct_lltype = struct_lltype_from_strptr(&ptr_lltype)?;

    let inner_name_owned = inner_helper_name.to_string();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![LowLevelType::Char],
        LowLevelType::Char,
        move |_rtyper, _args, _result| build_inner_helper(&inner_name_owned),
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let s_for_empty = variable_with_lltype("s", ptr_lltype.clone());
    let block_empty_return = Block::shared(vec![Hlvalue::Variable(s_for_empty.clone())]);

    let chars_for_alloc = variable_with_lltype("s_chars", chars_array_ptr_lltype.clone());
    let len_for_alloc = variable_with_lltype("s_len", LowLevelType::Signed);
    let block_alloc = Block::shared(vec![
        Hlvalue::Variable(chars_for_alloc.clone()),
        Hlvalue::Variable(len_for_alloc.clone()),
    ]);

    let chars_for_cond = variable_with_lltype("s_chars", chars_array_ptr_lltype.clone());
    let newstr_for_cond = variable_with_lltype("result", ptr_lltype.clone());
    let newchars_for_cond = variable_with_lltype("newchars", chars_array_ptr_lltype.clone());
    let len_for_cond = variable_with_lltype("s_len", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars_for_cond.clone()),
        Hlvalue::Variable(newstr_for_cond.clone()),
        Hlvalue::Variable(newchars_for_cond.clone()),
        Hlvalue::Variable(len_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    let chars_for_body = variable_with_lltype("s_chars", chars_array_ptr_lltype.clone());
    let newstr_for_body = variable_with_lltype("result", ptr_lltype.clone());
    let newchars_for_body = variable_with_lltype("newchars", chars_array_ptr_lltype.clone());
    let len_for_body = variable_with_lltype("s_len", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars_for_body.clone()),
        Hlvalue::Variable(newstr_for_body.clone()),
        Hlvalue::Variable(newchars_for_body.clone()),
        Hlvalue::Variable(len_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
    ]);

    // start: s_chars, s_len, empty fast-path branch.
    let chars = variable_with_lltype("s_chars", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s.clone()), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let s_len = variable_with_lltype("s_len", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(s_len.clone()),
    ));
    let is_empty = variable_with_lltype("is_empty", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(s_len.clone()), signed_const(0)],
        Hlvalue::Variable(is_empty.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_empty));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s.clone())],
            Some(block_empty_return.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(chars), Hlvalue::Variable(s_len)],
            Some(block_alloc.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // empty fast path: no allocation, matching `s.empty()`'s
    // allocation-free contract for already-empty input.
    block_empty_return.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s_for_empty)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    // non-empty allocation: result = mallocstr(s_len).
    let newstr = variable_with_lltype("result", ptr_lltype.clone());
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "malloc_varsize",
            vec![
                lowlevel_type_const(struct_lltype),
                gc_flavor_const()?,
                Hlvalue::Variable(len_for_alloc.clone()),
            ],
            Hlvalue::Variable(newstr.clone()),
        ));
    let newchars = variable_with_lltype("newchars", chars_array_ptr_lltype.clone());
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(newstr.clone()), chars_field_const()],
            Hlvalue::Variable(newchars.clone()),
        ));
    block_alloc.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_alloc),
                Hlvalue::Variable(newstr),
                Hlvalue::Variable(newchars),
                Hlvalue::Variable(len_for_alloc),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // loop_cond: i < s_len.
    let keep_going = variable_with_lltype("keep_going", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(len_for_cond.clone()),
            ],
            Hlvalue::Variable(keep_going.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(keep_going));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_cond),
                Hlvalue::Variable(newstr_for_cond.clone()),
                Hlvalue::Variable(newchars_for_cond),
                Hlvalue::Variable(len_for_cond),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(newstr_for_cond)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // loop_body: result.chars[i] = ll_upper/lower_char(s_chars[i]).
    let c = variable_with_lltype("c", LowLevelType::Char);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    let c_folded = variable_with_lltype("c_folded", LowLevelType::Char);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(c_inner_funcptr), Hlvalue::Variable(c)],
            Hlvalue::Variable(c_folded.clone()),
        ));
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(newchars_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
                Hlvalue::Variable(c_folded),
            ],
            Hlvalue::Variable(set_void),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_body), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_body),
                Hlvalue::Variable(newstr_for_body),
                Hlvalue::Variable(newchars_for_body),
                Hlvalue::Variable(len_for_body),
                Hlvalue::Variable(i_next),
            ],
            Some(block_loop_cond),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise `LLHelpers.ll_replace_chr_chr`
/// (`lltypesystem/rstr.py`):
///
/// ```python
/// def ll_replace_chr_chr(s, c1, c2):
///     length = len(s.chars)
///     newstr = s.malloc(length)
///     src = s.chars
///     dst = newstr.chars
///     j = 0
///     while j < length:
///         c = src[j]
///         if c == c1:
///             c = c2
///         dst[j] = c
///         j += 1
///     return newstr
/// ```
pub(crate) fn build_ll_replace_chr_chr_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    elem_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_replace_chr_chr helper expects Ptr(Array(Char|UniChar)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_replace_chr_chr helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    if arr.OF != elem_lltype {
        return Err(TyperError::message(format!(
            "ll_replace_chr_chr helper element mismatch: chars array stores {:?}, argument is {elem_lltype:?}",
            arr.OF
        )));
    }
    let eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_replace_chr_chr helper unsupported char element type {other:?}"
            )));
        }
    };
    let struct_lltype = struct_lltype_from_strptr(&ptr_lltype)?;

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let c1 = variable_with_lltype("c1", elem_lltype.clone());
    let c2 = variable_with_lltype("c2", elem_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(c1.clone()),
        Hlvalue::Variable(c2.clone()),
    ]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let src_for_cond = variable_with_lltype("src", chars_array_ptr_lltype.clone());
    let dst_for_cond = variable_with_lltype("dst", chars_array_ptr_lltype.clone());
    let newstr_for_cond = variable_with_lltype("newstr", ptr_lltype.clone());
    let length_for_cond = variable_with_lltype("length", LowLevelType::Signed);
    let c1_for_cond = variable_with_lltype("c1", elem_lltype.clone());
    let c2_for_cond = variable_with_lltype("c2", elem_lltype.clone());
    let j_for_cond = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(src_for_cond.clone()),
        Hlvalue::Variable(dst_for_cond.clone()),
        Hlvalue::Variable(newstr_for_cond.clone()),
        Hlvalue::Variable(length_for_cond.clone()),
        Hlvalue::Variable(c1_for_cond.clone()),
        Hlvalue::Variable(c2_for_cond.clone()),
        Hlvalue::Variable(j_for_cond.clone()),
    ]);

    let src_for_body = variable_with_lltype("src", chars_array_ptr_lltype.clone());
    let dst_for_body = variable_with_lltype("dst", chars_array_ptr_lltype.clone());
    let newstr_for_body = variable_with_lltype("newstr", ptr_lltype.clone());
    let length_for_body = variable_with_lltype("length", LowLevelType::Signed);
    let c1_for_body = variable_with_lltype("c1", elem_lltype.clone());
    let c2_for_body = variable_with_lltype("c2", elem_lltype.clone());
    let j_for_body = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(src_for_body.clone()),
        Hlvalue::Variable(dst_for_body.clone()),
        Hlvalue::Variable(newstr_for_body.clone()),
        Hlvalue::Variable(length_for_body.clone()),
        Hlvalue::Variable(c1_for_body.clone()),
        Hlvalue::Variable(c2_for_body.clone()),
        Hlvalue::Variable(j_for_body.clone()),
    ]);

    let src_for_store = variable_with_lltype("src", chars_array_ptr_lltype.clone());
    let dst_for_store = variable_with_lltype("dst", chars_array_ptr_lltype.clone());
    let newstr_for_store = variable_with_lltype("newstr", ptr_lltype.clone());
    let length_for_store = variable_with_lltype("length", LowLevelType::Signed);
    let c1_for_store = variable_with_lltype("c1", elem_lltype.clone());
    let c2_for_store = variable_with_lltype("c2", elem_lltype.clone());
    let j_for_store = variable_with_lltype("j", LowLevelType::Signed);
    let c_for_store = variable_with_lltype("c", elem_lltype.clone());
    let block_store = Block::shared(vec![
        Hlvalue::Variable(src_for_store.clone()),
        Hlvalue::Variable(dst_for_store.clone()),
        Hlvalue::Variable(newstr_for_store.clone()),
        Hlvalue::Variable(length_for_store.clone()),
        Hlvalue::Variable(c1_for_store.clone()),
        Hlvalue::Variable(c2_for_store.clone()),
        Hlvalue::Variable(j_for_store.clone()),
        Hlvalue::Variable(c_for_store.clone()),
    ]);

    // start: source chars, length, destination allocation/chars.
    let src = variable_with_lltype("src", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s), chars_field_const()],
        Hlvalue::Variable(src.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(src.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let newstr = variable_with_lltype("newstr", ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(struct_lltype),
            gc_flavor_const()?,
            Hlvalue::Variable(length.clone()),
        ],
        Hlvalue::Variable(newstr.clone()),
    ));
    let dst = variable_with_lltype("dst", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(newstr.clone()), chars_field_const()],
        Hlvalue::Variable(dst.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src),
                Hlvalue::Variable(dst),
                Hlvalue::Variable(newstr),
                Hlvalue::Variable(length),
                Hlvalue::Variable(c1),
                Hlvalue::Variable(c2),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // loop_cond: j < length.
    let keep_going = variable_with_lltype("keep_going", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(j_for_cond.clone()),
                Hlvalue::Variable(length_for_cond.clone()),
            ],
            Hlvalue::Variable(keep_going.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(keep_going));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_for_cond),
                Hlvalue::Variable(dst_for_cond),
                Hlvalue::Variable(newstr_for_cond.clone()),
                Hlvalue::Variable(length_for_cond),
                Hlvalue::Variable(c1_for_cond),
                Hlvalue::Variable(c2_for_cond),
                Hlvalue::Variable(j_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(newstr_for_cond)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // loop_body: c = src[j]; if c == c1: c = c2.
    let c = variable_with_lltype("c", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(src_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    let replace = variable_with_lltype("replace", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            eq_op,
            vec![
                Hlvalue::Variable(c.clone()),
                Hlvalue::Variable(c1_for_body.clone()),
            ],
            Hlvalue::Variable(replace.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(replace));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_for_body.clone()),
                Hlvalue::Variable(dst_for_body.clone()),
                Hlvalue::Variable(newstr_for_body.clone()),
                Hlvalue::Variable(length_for_body.clone()),
                Hlvalue::Variable(c1_for_body.clone()),
                Hlvalue::Variable(c2_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
                Hlvalue::Variable(c2_for_body.clone()),
            ],
            Some(block_store.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(src_for_body),
                Hlvalue::Variable(dst_for_body),
                Hlvalue::Variable(newstr_for_body),
                Hlvalue::Variable(length_for_body),
                Hlvalue::Variable(c1_for_body),
                Hlvalue::Variable(c2_for_body),
                Hlvalue::Variable(j_for_body),
                Hlvalue::Variable(c),
            ],
            Some(block_store.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // store: dst[j] = c; j += 1.
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    block_store
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(dst_for_store.clone()),
                Hlvalue::Variable(j_for_store.clone()),
                Hlvalue::Variable(c_for_store),
            ],
            Hlvalue::Variable(set_void),
        ));
    let j_next = variable_with_lltype("j_next", LowLevelType::Signed);
    block_store
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(j_for_store), signed_const(1)],
            Hlvalue::Variable(j_next.clone()),
        ));
    block_store.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_for_store),
                Hlvalue::Variable(dst_for_store),
                Hlvalue::Variable(newstr_for_store),
                Hlvalue::Variable(length_for_store),
                Hlvalue::Variable(c1_for_store),
                Hlvalue::Variable(c2_for_store),
                Hlvalue::Variable(j_next),
            ],
            Some(block_loop_cond),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "c1".to_string(), "c2".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `AbstractLLHelpers.ll_stritem_nonneg_checked`
/// (`rtyper/rstr.py`):
///
/// ```python
/// @classmethod
/// def ll_stritem_nonneg_checked(cls, s, i):
///     if i >= cls.ll_strlen(s):
///         raise IndexError
///     return cls.ll_stritem_nonneg(s, i)
/// ```
///
/// 2-block CFG plus the returnblock and graph.exceptblock:
/// - **start**: `chars = getsubstruct(s, 'chars')`,
///   `length = getarraysize(chars)`, `oob = int_ge(i, length)`. True →
///   `graph.exceptblock` with `[exc_cls, exc_inst]` for IndexError;
///   False → `block_dispatch(s, i)`.
/// - **block_dispatch**: `c = direct_call(ll_stritem_nonneg, s, i)`.
///   Link to returnblock with `c`.
///
/// `inner_helper_name` is the cache key for the `ll_stritem_nonneg`
/// sub-helper (`ll_stritem_nonneg` for STR pair,
/// `ll_unicode_stritem_nonneg` for UNICODE pair).
pub(crate) fn build_ll_stritem_nonneg_checked_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_stritem_nonneg_checked helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_stritem_nonneg_checked helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_stritem_nonneg_checked helper unsupported chars element type {elem_lltype:?}"
        )));
    }

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Materialise (or retrieve cached) ll_stritem_nonneg sub-helper.
    let inner_name_owned = inner_helper_name.to_string();
    let ptr_for_inner = ptr_lltype.clone();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![ptr_lltype.clone(), LowLevelType::Signed],
        elem_lltype.clone(),
        move |_rtyper, _args, _result| {
            build_ll_stritem_nonneg_helper_graph(&inner_name_owned, ptr_for_inner)
        },
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let exc_args = exception_args("IndexError")?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let i = variable_with_lltype("i", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", elem_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // block_dispatch carries (s, i) and direct_calls ll_stritem_nonneg.
    let s_for_dispatch = variable_with_lltype("s", ptr_lltype.clone());
    let i_for_dispatch = variable_with_lltype("i", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(s_for_dispatch.clone()),
        Hlvalue::Variable(i_for_dispatch.clone()),
    ]);

    // start: getsubstruct + getarraysize + int_ge(i, length); branch.
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s.clone()), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars)],
        Hlvalue::Variable(length.clone()),
    ));
    let oob = variable_with_lltype("oob", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![Hlvalue::Variable(i.clone()), Hlvalue::Variable(length)],
        Hlvalue::Variable(oob.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob));
    startblock.closeblock(vec![
        // True (i >= length): raise IndexError.
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False (in-bounds): forward to block_dispatch.
        Link::new(
            vec![Hlvalue::Variable(s), Hlvalue::Variable(i)],
            Some(block_dispatch.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_dispatch: direct_call(ll_stritem_nonneg, s, i).
    let c = variable_with_lltype("c", elem_lltype);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_inner_funcptr),
                Hlvalue::Variable(s_for_dispatch),
                Hlvalue::Variable(i_for_dispatch),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    block_dispatch.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "i".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `AbstractLLHelpers.ll_stritem_checked`
/// (`rtyper/rstr.py`):
///
/// ```python
/// @classmethod
/// def ll_stritem_checked(cls, s, i):
///     length = cls.ll_strlen(s)
///     if i < 0:
///         i += length
///     if i >= length or i < 0:
///         raise IndexError
///     return cls.ll_stritem_nonneg(s, i)
/// ```
///
/// 5-block CFG plus the returnblock and graph.exceptblock. The
/// `i >= length or i < 0` short-circuit lowers to two sequential
/// conditional blocks (`block_check_high` then `block_check_low`),
/// each with its own raise edge to graph.exceptblock — matching
/// flowspace's `or` lowering.
///
/// - **start**: `chars = getsubstruct`, `length = getarraysize`,
///   `is_neg = int_lt(i, 0)`. True → `block_neg_fix(s, length, i)`;
///   False → `block_check_high(s, length, i)`.
/// - **block_neg_fix**: `i_fix = int_add(i, length)`. Forward
///   `block_check_high(s, length, i_fix)`.
/// - **block_check_high**: `too_high = int_ge(i, length)`. True →
///   raise IndexError; False → `block_check_low(s, i)`.
/// - **block_check_low**: `too_low = int_lt(i, 0)`. True → raise;
///   False → `block_dispatch(s, i)`.
/// - **block_dispatch**: `c = direct_call(ll_stritem_nonneg, s, i)`.
///   Forward returnblock with `c`.
///
/// `inner_helper_name` is the cache key for the `ll_stritem_nonneg`
/// sub-helper.
pub(crate) fn build_ll_stritem_checked_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_stritem_checked helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_stritem_checked helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_stritem_checked helper unsupported chars element type {elem_lltype:?}"
        )));
    }

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Materialise (or retrieve cached) ll_stritem_nonneg sub-helper.
    let inner_name_owned = inner_helper_name.to_string();
    let ptr_for_inner = ptr_lltype.clone();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![ptr_lltype.clone(), LowLevelType::Signed],
        elem_lltype.clone(),
        move |_rtyper, _args, _result| {
            build_ll_stritem_nonneg_helper_graph(&inner_name_owned, ptr_for_inner)
        },
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let exc_args = exception_args("IndexError")?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let i = variable_with_lltype("i", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", elem_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // block_neg_fix: takes (s, length, i); computes i_fix = i + length.
    let s_for_fix = variable_with_lltype("s", ptr_lltype.clone());
    let length_for_fix = variable_with_lltype("length", LowLevelType::Signed);
    let i_for_fix = variable_with_lltype("i", LowLevelType::Signed);
    let block_neg_fix = Block::shared(vec![
        Hlvalue::Variable(s_for_fix.clone()),
        Hlvalue::Variable(length_for_fix.clone()),
        Hlvalue::Variable(i_for_fix.clone()),
    ]);

    // block_check_high: takes (s, length, i); int_ge(i, length).
    let s_for_high = variable_with_lltype("s", ptr_lltype.clone());
    let length_for_high = variable_with_lltype("length", LowLevelType::Signed);
    let i_for_high = variable_with_lltype("i", LowLevelType::Signed);
    let block_check_high = Block::shared(vec![
        Hlvalue::Variable(s_for_high.clone()),
        Hlvalue::Variable(length_for_high.clone()),
        Hlvalue::Variable(i_for_high.clone()),
    ]);

    // block_check_low: takes (s, i); int_lt(i, 0).
    let s_for_low = variable_with_lltype("s", ptr_lltype.clone());
    let i_for_low = variable_with_lltype("i", LowLevelType::Signed);
    let block_check_low = Block::shared(vec![
        Hlvalue::Variable(s_for_low.clone()),
        Hlvalue::Variable(i_for_low.clone()),
    ]);

    // block_dispatch: takes (s, i); direct_calls ll_stritem_nonneg.
    let s_for_dispatch = variable_with_lltype("s", ptr_lltype.clone());
    let i_for_dispatch = variable_with_lltype("i", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(s_for_dispatch.clone()),
        Hlvalue::Variable(i_for_dispatch.clone()),
    ]);

    // start: getsubstruct + getarraysize + int_lt(i, 0); branch.
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s.clone()), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars)],
        Hlvalue::Variable(length.clone()),
    ));
    let is_neg = variable_with_lltype("is_neg", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(i.clone()), signed_const(0)],
        Hlvalue::Variable(is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s.clone()),
                Hlvalue::Variable(length.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Some(block_neg_fix.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(s),
                Hlvalue::Variable(length),
                Hlvalue::Variable(i),
            ],
            Some(block_check_high.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_neg_fix: i_fix = int_add(i, length).
    let i_fix = variable_with_lltype("i_fix", LowLevelType::Signed);
    block_neg_fix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(i_for_fix),
                Hlvalue::Variable(length_for_fix.clone()),
            ],
            Hlvalue::Variable(i_fix.clone()),
        ));
    block_neg_fix.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s_for_fix),
                Hlvalue::Variable(length_for_fix),
                Hlvalue::Variable(i_fix),
            ],
            Some(block_check_high.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_check_high: too_high = int_ge(i, length); branch.
    let too_high = variable_with_lltype("too_high", LowLevelType::Bool);
    block_check_high
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ge",
            vec![
                Hlvalue::Variable(i_for_high.clone()),
                Hlvalue::Variable(length_for_high),
            ],
            Hlvalue::Variable(too_high.clone()),
        ));
    block_check_high.borrow_mut().exitswitch = Some(Hlvalue::Variable(too_high));
    block_check_high.closeblock(vec![
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(s_for_high), Hlvalue::Variable(i_for_high)],
            Some(block_check_low.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_check_low: too_low = int_lt(i, 0); branch.
    let too_low = variable_with_lltype("too_low", LowLevelType::Bool);
    block_check_low
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![Hlvalue::Variable(i_for_low.clone()), signed_const(0)],
            Hlvalue::Variable(too_low.clone()),
        ));
    block_check_low.borrow_mut().exitswitch = Some(Hlvalue::Variable(too_low));
    block_check_low.closeblock(vec![
        Link::new(exc_args, Some(graph.exceptblock.clone()), Some(bool_true())).into_ref(),
        Link::new(
            vec![Hlvalue::Variable(s_for_low), Hlvalue::Variable(i_for_low)],
            Some(block_dispatch.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_dispatch: direct_call(ll_stritem_nonneg, s, i).
    let c = variable_with_lltype("c", elem_lltype);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_inner_funcptr),
                Hlvalue::Variable(s_for_dispatch),
                Hlvalue::Variable(i_for_dispatch),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    block_dispatch.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "i".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_strconcat`
/// (`rtyper/lltypesystem/rstr.py`):
///
/// ```python
/// @staticmethod
/// @jit.elidable
/// @jit.oopspec('stroruni.concat(s1, s2)')
/// def ll_strconcat(s1, s2):
///     len1 = s1.length()
///     len2 = s2.length()
///     newstr = s2.malloc(len1 + len2)
///     newstr.copy_contents_from_str(s1, newstr, 0, 0, len1)
///     newstr.copy_contents_from_str(s2, newstr, 0, len1, len2)
///     return newstr
/// ```
///
/// Pyre lowers `s.malloc(N)` to a `malloc_varsize` SpaceOperation and
/// the two `copy_contents_from_str` calls to two `copystrcontent`
/// (`copyunicodecontent` for UNICODE) ops — the same op the JIT
/// codewriter produces for the `stroruni.copy_contents` oopspec. Both
/// operands are whole string objects, so no interior chars pointer is
/// materialised.
///
/// Straight-line CFG (start → returnblock):
/// - **start**: `chars1 = getsubstruct(s1, 'chars')`,
///   `len1 = getarraysize(chars1)`; likewise `chars2` / `len2`;
///   `total = int_add(len1, len2)`; `newstr = malloc_varsize(struct,
///   gc, total)`; `copystrcontent(s1, newstr, 0, 0, len1)`;
///   `copystrcontent(s2, newstr, 0, len1, len2)`. Forward into
///   `returnblock(newstr)`.
///
/// Polymorphic over `Ptr(STR)` / `Ptr(UNICODE)`: the chars-array
/// element type (Char / UniChar) selects `copystrcontent` vs
/// `copyunicodecontent`.
pub(crate) fn build_ll_strconcat_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_strconcat helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_strconcat helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_strconcat helper unsupported chars element type {elem_lltype:?}"
        )));
    }

    let struct_lltype = struct_lltype_from_strptr(&ptr_lltype)?;

    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s1 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2 = variable_with_lltype("s2", ptr_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s1.clone()),
        Hlvalue::Variable(s2.clone()),
    ]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // start: extract source lengths, malloc the result, then copy each
    //      source's chars into it via copystrcontent.  Mirrors upstream
    //      `ll_strconcat` (rstr.py): malloc + two `copy_contents_from_str`
    //      calls, each rewritten by the JIT codewriter to a single
    //      `copystrcontent` op.
    let chars1 = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s1.clone()), chars_field_const()],
        Hlvalue::Variable(chars1.clone()),
    ));
    let len1 = variable_with_lltype("len1", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars1.clone())],
        Hlvalue::Variable(len1.clone()),
    ));
    let chars2 = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s2.clone()), chars_field_const()],
        Hlvalue::Variable(chars2.clone()),
    ));
    let len2 = variable_with_lltype("len2", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars2.clone())],
        Hlvalue::Variable(len2.clone()),
    ));
    let total = variable_with_lltype("total", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![
            Hlvalue::Variable(len1.clone()),
            Hlvalue::Variable(len2.clone()),
        ],
        Hlvalue::Variable(total.clone()),
    ));
    let newstr = variable_with_lltype("newstr", ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(struct_lltype),
            gc_flavor_const()?,
            Hlvalue::Variable(total),
        ],
        Hlvalue::Variable(newstr.clone()),
    ));
    // Two bulk copies: `copystrcontent(src, dst, srcstart, dststart, length)`
    // for STR / `copyunicodecontent` for UNICODE.  The operands are whole
    // string objects, so there is no interior chars pointer to thread.
    let copy_opname = if matches!(elem_lltype, LowLevelType::Char) {
        "copystrcontent"
    } else {
        "copyunicodecontent"
    };
    let void_copy1 = variable_with_lltype("copy1", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        copy_opname,
        vec![
            Hlvalue::Variable(s1),
            Hlvalue::Variable(newstr.clone()),
            signed_const(0),
            signed_const(0),
            Hlvalue::Variable(len1.clone()),
        ],
        Hlvalue::Variable(void_copy1),
    ));
    let void_copy2 = variable_with_lltype("copy2", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        copy_opname,
        vec![
            Hlvalue::Variable(s2),
            Hlvalue::Variable(newstr.clone()),
            signed_const(0),
            Hlvalue::Variable(len1),
            Hlvalue::Variable(len2),
        ],
        Hlvalue::Variable(void_copy2),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(newstr)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s1".to_string(), "s2".to_string()],
        func,
    ))
}

/// `LLHelpers._ll_stringslice` plus the three wrappers selected by
/// `AbstractStringRepr.rtype_getslice` (`rstr.py`,
/// `lltypesystem/rstr.py`).  The start/stop wrapper preserves the
/// identity-bearing `jit.we_are_jitted()` branch: the JIT arm clamps
/// `stop > len`, while the no-JIT arm returns `s1` for `[0:>=len]`.
pub(crate) fn build_ll_stringslice_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    kind: crate::translator::rtyper::rtyper::SliceKind,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rtyper::SliceKind;

    let core_name = if ptr_lltype == STRPTR.clone() {
        "_ll_stringslice"
    } else {
        "_ll_unicode_slice"
    };
    let core_name_owned = core_name.to_string();
    let ptr_for_core = ptr_lltype.clone();
    let core = rtyper.lowlevel_helper_function_with_builder(
        core_name.to_string(),
        vec![
            ptr_lltype.clone(),
            LowLevelType::Signed,
            LowLevelType::Signed,
        ],
        ptr_lltype.clone(),
        move |_rtyper, _args, _result| {
            build_ll_stringslice_core_helper_graph(&core_name_owned, ptr_for_core)
        },
    )?;
    let core_ptr = sub_helper_funcptr_constant(rtyper, &core)?;

    let s = variable_with_lltype("s1", ptr_lltype.clone());
    let start = variable_with_lltype("start", LowLevelType::Signed);
    let stop = variable_with_lltype("stop", LowLevelType::Signed);
    let inputargs = match kind {
        SliceKind::MinusOne => vec![Hlvalue::Variable(s.clone())],
        SliceKind::StartOnly => vec![
            Hlvalue::Variable(s.clone()),
            Hlvalue::Variable(start.clone()),
        ],
        SliceKind::StartStop => vec![
            Hlvalue::Variable(s.clone()),
            Hlvalue::Variable(start.clone()),
            Hlvalue::Variable(stop.clone()),
        ],
    };
    let startblock = Block::shared(inputargs);
    let result = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(result),
    );
    let chars_ty = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let chars = variable_with_lltype("chars", chars_ty);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![
            Hlvalue::Variable(s.clone()),
            constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
        ],
        Hlvalue::Variable(chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars)],
        Hlvalue::Variable(length.clone()),
    ));

    let signed = |n| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    match kind {
        SliceKind::MinusOne => {
            let end = variable_with_lltype("end", LowLevelType::Signed);
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "int_sub",
                vec![Hlvalue::Variable(length), signed(1)],
                Hlvalue::Variable(end.clone()),
            ));
            emit_stringslice_direct_call(
                &startblock,
                core_ptr,
                s,
                signed(0),
                Hlvalue::Variable(end),
                &graph.returnblock,
                ptr_lltype,
            );
        }
        SliceKind::StartOnly => emit_stringslice_direct_call(
            &startblock,
            core_ptr,
            s,
            Hlvalue::Variable(start),
            Hlvalue::Variable(length),
            &graph.returnblock,
            ptr_lltype,
        ),
        SliceKind::StartStop => {
            // lltypesystem/rstr.py ll_stringslice_startstop preserves both arms of
            // `jit.we_are_jitted()`: the JIT arm clamps only `stop > len`,
            // while the no-JIT arm returns `s1` unchanged for `[0:>=len]`.
            let call_s = variable_with_lltype("s1", ptr_lltype.clone());
            let call_start = variable_with_lltype("start", LowLevelType::Signed);
            let call_stop = variable_with_lltype("stop", LowLevelType::Signed);
            let call_block = Block::shared(vec![
                Hlvalue::Variable(call_s.clone()),
                Hlvalue::Variable(call_start.clone()),
                Hlvalue::Variable(call_stop.clone()),
            ]);

            let jit_s = variable_with_lltype("s1", ptr_lltype.clone());
            let jit_start = variable_with_lltype("start", LowLevelType::Signed);
            let jit_stop = variable_with_lltype("stop", LowLevelType::Signed);
            let jit_len = variable_with_lltype("length", LowLevelType::Signed);
            let jit_block = Block::shared(vec![
                Hlvalue::Variable(jit_s.clone()),
                Hlvalue::Variable(jit_start.clone()),
                Hlvalue::Variable(jit_stop.clone()),
                Hlvalue::Variable(jit_len.clone()),
            ]);
            let jit_too_far = variable_with_lltype("stop_too_far", LowLevelType::Bool);
            jit_block.borrow_mut().operations.push(SpaceOperation::new(
                "int_gt",
                vec![
                    Hlvalue::Variable(jit_stop.clone()),
                    Hlvalue::Variable(jit_len.clone()),
                ],
                Hlvalue::Variable(jit_too_far.clone()),
            ));
            jit_block.borrow_mut().exitswitch = Some(Hlvalue::Variable(jit_too_far));
            jit_block.closeblock(vec![
                Link::new(
                    vec![
                        Hlvalue::Variable(jit_s.clone()),
                        Hlvalue::Variable(jit_start.clone()),
                        Hlvalue::Variable(jit_len),
                    ],
                    Some(call_block.clone()),
                    Some(constant_with_lltype(
                        ConstValue::Bool(true),
                        LowLevelType::Bool,
                    )),
                )
                .into_ref(),
                Link::new(
                    vec![
                        Hlvalue::Variable(jit_s),
                        Hlvalue::Variable(jit_start),
                        Hlvalue::Variable(jit_stop),
                    ],
                    Some(call_block.clone()),
                    Some(constant_with_lltype(
                        ConstValue::Bool(false),
                        LowLevelType::Bool,
                    )),
                )
                .into_ref(),
            ]);

            let nojit_s = variable_with_lltype("s1", ptr_lltype.clone());
            let nojit_start = variable_with_lltype("start", LowLevelType::Signed);
            let nojit_stop = variable_with_lltype("stop", LowLevelType::Signed);
            let nojit_len = variable_with_lltype("length", LowLevelType::Signed);
            let nojit_block = Block::shared(vec![
                Hlvalue::Variable(nojit_s.clone()),
                Hlvalue::Variable(nojit_start.clone()),
                Hlvalue::Variable(nojit_stop.clone()),
                Hlvalue::Variable(nojit_len.clone()),
            ]);
            let reaches_end = variable_with_lltype("stop_reaches_end", LowLevelType::Bool);
            nojit_block
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    "int_ge",
                    vec![
                        Hlvalue::Variable(nojit_stop.clone()),
                        Hlvalue::Variable(nojit_len.clone()),
                    ],
                    Hlvalue::Variable(reaches_end.clone()),
                ));
            let full_s = variable_with_lltype("s1", ptr_lltype.clone());
            let full_start = variable_with_lltype("start", LowLevelType::Signed);
            let full_len = variable_with_lltype("length", LowLevelType::Signed);
            let full_block = Block::shared(vec![
                Hlvalue::Variable(full_s.clone()),
                Hlvalue::Variable(full_start.clone()),
                Hlvalue::Variable(full_len.clone()),
            ]);
            nojit_block.borrow_mut().exitswitch = Some(Hlvalue::Variable(reaches_end));
            nojit_block.closeblock(vec![
                Link::new(
                    vec![
                        Hlvalue::Variable(nojit_s.clone()),
                        Hlvalue::Variable(nojit_start.clone()),
                        Hlvalue::Variable(nojit_len),
                    ],
                    Some(full_block.clone()),
                    Some(constant_with_lltype(
                        ConstValue::Bool(true),
                        LowLevelType::Bool,
                    )),
                )
                .into_ref(),
                Link::new(
                    vec![
                        Hlvalue::Variable(nojit_s),
                        Hlvalue::Variable(nojit_start),
                        Hlvalue::Variable(nojit_stop),
                    ],
                    Some(call_block.clone()),
                    Some(constant_with_lltype(
                        ConstValue::Bool(false),
                        LowLevelType::Bool,
                    )),
                )
                .into_ref(),
            ]);
            let starts_at_zero = variable_with_lltype("starts_at_zero", LowLevelType::Bool);
            full_block.borrow_mut().operations.push(SpaceOperation::new(
                "int_eq",
                vec![Hlvalue::Variable(full_start.clone()), signed(0)],
                Hlvalue::Variable(starts_at_zero.clone()),
            ));
            full_block.borrow_mut().exitswitch = Some(Hlvalue::Variable(starts_at_zero));
            full_block.closeblock(vec![
                Link::new(
                    vec![Hlvalue::Variable(full_s.clone())],
                    Some(graph.returnblock.clone()),
                    Some(constant_with_lltype(
                        ConstValue::Bool(true),
                        LowLevelType::Bool,
                    )),
                )
                .into_ref(),
                Link::new(
                    vec![
                        Hlvalue::Variable(full_s),
                        Hlvalue::Variable(full_start),
                        Hlvalue::Variable(full_len),
                    ],
                    Some(call_block.clone()),
                    Some(constant_with_lltype(
                        ConstValue::Bool(false),
                        LowLevelType::Bool,
                    )),
                )
                .into_ref(),
            ]);

            startblock.borrow_mut().exitswitch =
                Some(Hlvalue::Constant(Constant::with_concretetype(
                    ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID),
                    LowLevelType::Bool,
                )));
            startblock.closeblock(vec![
                Link::new(
                    vec![
                        Hlvalue::Variable(s.clone()),
                        Hlvalue::Variable(start.clone()),
                        Hlvalue::Variable(stop.clone()),
                        Hlvalue::Variable(length.clone()),
                    ],
                    Some(jit_block),
                    Some(constant_with_lltype(
                        ConstValue::Bool(true),
                        LowLevelType::Bool,
                    )),
                )
                .into_ref(),
                Link::new(
                    vec![
                        Hlvalue::Variable(s),
                        Hlvalue::Variable(start),
                        Hlvalue::Variable(stop),
                        Hlvalue::Variable(length),
                    ],
                    Some(nojit_block),
                    Some(constant_with_lltype(
                        ConstValue::Bool(false),
                        LowLevelType::Bool,
                    )),
                )
                .into_ref(),
            ]);
            emit_stringslice_direct_call(
                &call_block,
                core_ptr,
                call_s,
                Hlvalue::Variable(call_start),
                Hlvalue::Variable(call_stop),
                &graph.returnblock,
                ptr_lltype,
            );
        }
    }

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    let args = match kind {
        SliceKind::MinusOne => vec!["s1"],
        SliceKind::StartOnly => vec!["s1", "start"],
        SliceKind::StartStop => vec!["s1", "start", "stop"],
    };
    Ok(helper_pygraph_from_graph(
        graph,
        args.into_iter().map(str::to_string).collect(),
        func,
    ))
}

fn emit_stringslice_direct_call(
    block: &crate::flowspace::model::BlockRef,
    funcptr: Constant,
    s: Variable,
    start: Hlvalue,
    stop: Hlvalue,
    returnblock: &crate::flowspace::model::BlockRef,
    ptr_lltype: LowLevelType,
) {
    let result = variable_with_lltype("result", ptr_lltype);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(funcptr),
            Hlvalue::Variable(s),
            start,
            stop,
        ],
        Hlvalue::Variable(result.clone()),
    ));
    block.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);
}

fn build_ll_stringslice_core_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let chars_ty = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_ty else {
        return Err(TyperError::message(
            "stringslice chars must be an array pointer",
        ));
    };
    let PtrTarget::Array(array) = &chars_ptr.TO else {
        return Err(TyperError::message(
            "stringslice chars target must be an array",
        ));
    };
    let copy_op = if array.OF == LowLevelType::Char {
        "copystrcontent"
    } else {
        "copyunicodecontent"
    };
    let struct_ty = struct_lltype_from_strptr(&ptr_lltype)?;
    let s = variable_with_lltype("s1", ptr_lltype.clone());
    let start = variable_with_lltype("start", LowLevelType::Signed);
    let stop = variable_with_lltype("stop", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(start.clone()),
        Hlvalue::Variable(stop.clone()),
    ]);
    let result = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(result),
    );
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![Hlvalue::Variable(stop), Hlvalue::Variable(start.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let negative = variable_with_lltype("negative", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![
            Hlvalue::Variable(length.clone()),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
        ],
        Hlvalue::Variable(negative.clone()),
    ));
    let alloc_s = variable_with_lltype("s1", ptr_lltype.clone());
    let alloc_start = variable_with_lltype("start", LowLevelType::Signed);
    let alloc_len = variable_with_lltype("length", LowLevelType::Signed);
    let alloc_block = Block::shared(vec![
        Hlvalue::Variable(alloc_s.clone()),
        Hlvalue::Variable(alloc_start.clone()),
        Hlvalue::Variable(alloc_len.clone()),
    ]);
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(negative));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s.clone()),
                Hlvalue::Variable(start.clone()),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            ],
            Some(alloc_block.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(s),
                Hlvalue::Variable(start),
                Hlvalue::Variable(length),
            ],
            Some(alloc_block.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);
    let newstr = variable_with_lltype("newstr", ptr_lltype.clone());
    alloc_block
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "malloc_varsize",
            vec![
                lowlevel_type_const(struct_ty),
                gc_flavor_const()?,
                Hlvalue::Variable(alloc_len.clone()),
            ],
            Hlvalue::Variable(newstr.clone()),
        ));
    alloc_block
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            copy_op,
            vec![
                Hlvalue::Variable(alloc_s),
                Hlvalue::Variable(newstr.clone()),
                Hlvalue::Variable(alloc_start),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
                Hlvalue::Variable(alloc_len),
            ],
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void,
            )),
        ));
    alloc_block.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(newstr)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);
    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s1".to_string(), "start".to_string(), "stop".to_string()],
        func,
    ))
}

/// Helper: derive a funcptr `Constant` (lltype `Ptr(FuncType)`) from
/// a sub-helper `LowLevelFunction` so it can be used as the first arg
/// of a `direct_call` SpaceOperation. Mirrors the funcptr derivation
/// inside
/// [`crate::translator::rtyper::rtyper::LowLevelOpList::gendirectcall`]
/// but exposed for builder closures that emit `direct_call` ops
/// directly into a synthesised `PyGraph`.
pub(crate) fn sub_helper_funcptr_constant(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    sub_helper: &crate::translator::rtyper::rtyper::LowLevelFunction,
) -> Result<Constant, TyperError> {
    let sub_graph = sub_helper.graph.as_ref().ok_or_else(|| {
        TyperError::message(format!(
            "sub-helper {} has no annotated helper graph",
            sub_helper.name
        ))
    })?;
    let func_ptr = rtyper.getcallable(sub_graph)?;
    let func_ptr_type = LowLevelType::Ptr(Box::new(func_ptr._TYPE.clone()));
    crate::translator::rtyper::rmodel::inputconst_from_lltype(
        &func_ptr_type,
        &ConstValue::LLPtr(Box::new(func_ptr)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `STR` resolves to a GcStruct named `rpy_string` with the two
    /// upstream fields `(hash: Signed, chars: Array(Char, ...))` and
    /// the `remove_hash` hint that upstream attaches at
    /// rstr.py:1237.
    #[test]
    fn str_resolves_to_gcstruct_rpy_string_with_hash_and_chars() {
        let resolved = match STR.clone() {
            LowLevelType::ForwardReference(fwd) => {
                fwd.resolved().expect("STR ForwardReference must resolve")
            }
            other => panic!("STR must be ForwardReference, got {other:?}"),
        };
        let LowLevelType::Struct(body) = resolved else {
            panic!("STR resolves to Struct");
        };
        assert_eq!(body._name, "rpy_string");
        assert!(matches!(body._flds.get("hash"), Some(LowLevelType::Signed)));
        assert!(matches!(
            body._flds.get("chars"),
            Some(LowLevelType::Array(_))
        ));
        // upstream `hints={'remove_hash': True}`.
        assert!(
            body._hints.get("remove_hash").is_some(),
            "remove_hash hint must be present on STR"
        );
    }

    /// `STRPTR` is `Ptr(STR)` — the lltype that `OBJECT_VTABLE.name`
    /// uses and that `ClassesPBCRepr.rtype_getattr('__name__')` would
    /// surface as the resulttype of its `getfield` op.
    #[test]
    fn strptr_is_ptr_to_str_forwardreference() {
        let LowLevelType::Ptr(ptr) = STRPTR.clone() else {
            panic!("STRPTR must be Ptr");
        };
        assert!(matches!(&ptr.TO, PtrTarget::ForwardReference(_)));
    }

    #[test]
    fn alloc_array_name_returns_live_str_with_chars() {
        let p = alloc_array_name("Foo").expect("alloc_array_name");
        assert!(p.nonzero());
        // convert_const precomputes the hash (rstr.py:199-200).
        let LowLevelValue::Signed(h) = p.getattr("hash").unwrap() else {
            panic!("hash field must be Signed");
        };
        assert_eq!(h, ll_strhash_value(b"Foo"));
        assert_ne!(h, 0);
        let Some(obj) = p._obj0_value().unwrap() else {
            panic!("STR pointer must be live");
        };
        let _ptr_obj::Struct(s) = obj else {
            panic!("STR pointer must target a Struct");
        };
        let chars_value = s._getattr("chars").expect("chars field must be present");
        let LowLevelValue::Array(chars) = chars_value else {
            panic!("chars field must be an Array");
        };
        assert_eq!(chars.getlength(), 3);
        assert_eq!(chars.getitem(0), Some(LowLevelValue::Char('F')));
        assert_eq!(chars.getitem(1), Some(LowLevelValue::Char('o')));
        assert_eq!(chars.getitem(2), Some(LowLevelValue::Char('o')));
    }

    /// `llstr` fills `chars[i]` with the Latin-1 embedding of each
    /// byte (`mallocstr` + per-index store, annlowlevel.py);
    /// every byte value must round-trip.
    #[test]
    fn llstr_preserves_byte_contents() {
        let bytes: &[u8] = &[b'a', 0x00, 0xFF, b'z'];
        let p = llstr(bytes).expect("llstr");
        assert!(p.nonzero());
        let Some(obj) = p._obj0_value().unwrap() else {
            panic!("llstr pointer must be live");
        };
        let _ptr_obj::Struct(s) = obj else {
            panic!("llstr pointer must target a Struct");
        };
        let LowLevelValue::Array(chars) = s._getattr("chars").expect("chars field") else {
            panic!("chars field must be an Array");
        };
        assert_eq!(chars.getlength(), bytes.len());
        for (i, b) in bytes.iter().enumerate() {
            assert_eq!(
                chars.getitem(i),
                Some(LowLevelValue::Char(*b as char)),
                "chars[{i}] must round-trip byte {b:#x}"
            );
        }
    }

    /// `_hash_string` (objectmodel.py) reproduces CPython
    /// 2.7's non-randomized 64-bit string hash; spot-check known
    /// values plus the empty-string `-1` and the `_ll_strhash`
    /// zero-fixup passthrough (a `-1` hash is not zero, so the
    /// 29872897 substitute must not fire).
    #[test]
    fn hash_string_bytes_matches_cpython2_values() {
        assert_eq!(hash_string_bytes(b""), -1);
        assert_eq!(hash_string_bytes(b"abc"), 1453079729188098211);
        assert_eq!(hash_string_bytes(b"Foo"), -5036909580522842981);
        assert_eq!(ll_strhash_value(b""), -1);
    }

    /// `CONST_STR_CACHE` (rstr.py / `StringRepr.CACHE`): repeated
    /// `convert_const` of the same literal must return the SAME
    /// prebuilt container (`is`-identity upstream), while distinct
    /// literals get distinct containers.
    #[test]
    fn const_str_cache_llstr_returns_same_container_identity() {
        let p1 = const_str_cache_llstr(b"cache-identity-probe").expect("first lookup");
        let p2 = const_str_cache_llstr(b"cache-identity-probe").expect("second lookup");
        // `_struct` PartialEq keys off `_identity`
        // (`fresh_low_level_container_identity`), so `==` here is the
        // upstream `is` check on the container.
        assert_eq!(p1, p2, "same literal must reuse the cached container");
        let p3 = const_str_cache_llstr(b"cache-identity-probe-other").expect("other literal");
        assert_ne!(p1, p3, "distinct literals must not share a container");
        // An uncached `llstr` of the same bytes mints a fresh container.
        let fresh = llstr(b"cache-identity-probe").expect("uncached llstr");
        assert_ne!(p1, fresh, "llstr must not consult the cache");
    }

    /// [`prebuilt_str_bytes_and_hash`] is the inverse of
    /// [`const_str_cache_llstr`]: the bytes round-trip (incl. `0x00`/`0xFF`)
    /// and the precomputed `hash` slot is read back, so the codewriter can
    /// lower a prebuilt `Ptr(STR)` to a deferred runtime descriptor.  Null
    /// pointers yield `None`.
    #[test]
    fn prebuilt_str_bytes_and_hash_round_trips_cache_llstr() {
        let bytes: &[u8] = &[b'h', 0x00, 0xFF, b'i'];
        let p = const_str_cache_llstr(bytes).expect("cache llstr");
        let (got_bytes, got_hash) =
            prebuilt_str_bytes_and_hash(&p).expect("prebuilt STR must decode");
        assert_eq!(got_bytes, bytes, "bytes must round-trip every value");
        assert_eq!(
            got_hash,
            ll_strhash_value(bytes),
            "hash slot must carry the precomputed ll_strhash value"
        );
        // The empty string decodes to empty bytes with the `-1` hash.
        let empty = const_str_cache_llstr(b"").expect("cache llstr empty");
        assert_eq!(
            prebuilt_str_bytes_and_hash(&empty),
            Some((Vec::new(), ll_strhash_value(b"")))
        );
        // A typed null `Ptr(STR)` is not a prebuilt string.
        assert_eq!(prebuilt_str_bytes_and_hash(&null_str_ptr()), None);
    }

    /// `nullptr(self.lowleveltype.TO)` arm of
    /// `BaseLLStringRepr.convert_const` (lltypesystem/rstr.py):
    /// a typed null `Ptr(STR)`.
    #[test]
    fn null_str_ptr_is_typed_null_strptr() {
        let p = null_str_ptr();
        assert!(!p.nonzero(), "null_str_ptr must be null");
        assert!(matches!(p._obj0_value(), Ok(None)));
        let LowLevelType::Ptr(expected) = STRPTR.clone() else {
            panic!("STRPTR must be Ptr");
        };
        assert_eq!(p._TYPE, *expected, "null must carry the STRPTR type");
    }

    /// `UNICODE` resolves to a GcStruct named `rpy_unicode` with
    /// the two upstream fields `(hash: Signed, chars: Array(UniChar,
    /// ...))` and the `remove_hash` hint that upstream attaches at
    /// rstr.py:1246. Mirror of
    /// [`str_resolves_to_gcstruct_rpy_string_with_hash_and_chars`]
    /// — verifies the `Array` element type is `UniChar` (not `Char`)
    /// so that an `Ptr(UNICODE).chars[i]` dereference lowers to a
    /// codepoint, not a byte.
    #[test]
    fn unicode_resolves_to_gcstruct_rpy_unicode_with_hash_and_unichar_chars() {
        let resolved = match UNICODE.clone() {
            LowLevelType::ForwardReference(fwd) => fwd
                .resolved()
                .expect("UNICODE ForwardReference must resolve"),
            other => panic!("UNICODE must be ForwardReference, got {other:?}"),
        };
        let LowLevelType::Struct(body) = resolved else {
            panic!("UNICODE resolves to Struct");
        };
        assert_eq!(body._name, "rpy_unicode");
        assert!(matches!(body._flds.get("hash"), Some(LowLevelType::Signed)));
        let Some(LowLevelType::Array(arr)) = body._flds.get("chars") else {
            panic!("chars field must be Array");
        };
        assert_eq!(arr.OF, LowLevelType::UniChar);
        assert!(
            body._hints.get("remove_hash").is_some(),
            "remove_hash hint must be present on UNICODE"
        );
    }

    /// `UNICODEPTR` is `Ptr(UNICODE)` — the lltype that
    /// `UnicodeRepr.lowleveltype` (rstr.py) carries.
    #[test]
    fn unicodeptr_is_ptr_to_unicode_forwardreference() {
        let LowLevelType::Ptr(ptr) = UNICODEPTR.clone() else {
            panic!("UNICODEPTR must be Ptr");
        };
        assert!(matches!(&ptr.TO, PtrTarget::ForwardReference(_)));
    }

    /// Round-trip a non-ASCII Unicode string through
    /// `alloc_array_unicode` and verify each codepoint lands as a
    /// distinct `UniChar` cell — confirms that the chars Array
    /// element type is wide enough for the full Unicode range, not
    /// truncated to a single byte.
    #[test]
    fn alloc_array_unicode_returns_live_unicode_with_unichar_codepoints() {
        let p = alloc_array_unicode("αβγ").expect("alloc_array_unicode");
        assert!(p.nonzero());
        let LowLevelValue::Signed(0) = p.getattr("hash").unwrap() else {
            panic!("hash field must be Signed(0)");
        };
        let Some(obj) = p._obj0_value().unwrap() else {
            panic!("UNICODE pointer must be live");
        };
        let _ptr_obj::Struct(s) = obj else {
            panic!("UNICODE pointer must target a Struct");
        };
        let chars_value = s._getattr("chars").expect("chars field must be present");
        let LowLevelValue::Array(chars) = chars_value else {
            panic!("chars field must be an Array");
        };
        assert_eq!(chars.getlength(), 3);
        assert_eq!(chars.getitem(0), Some(LowLevelValue::UniChar('α')));
        assert_eq!(chars.getitem(1), Some(LowLevelValue::UniChar('β')));
        assert_eq!(chars.getitem(2), Some(LowLevelValue::UniChar('γ')));
    }

    /// `LLHelpers.ll_chr2str` allocates a length-1 STR and stores the
    /// incoming Char at index 0. The Rust helper graph mirrors that as
    /// `malloc_varsize` + `getsubstruct('chars')` + `setarrayitem`.
    #[test]
    fn build_ll_chr2str_synthesizes_malloc_varsize_and_sets_char_zero() {
        let helper =
            build_ll_chr2str_helper_graph("ll_chr2str", STRPTR.clone(), LowLevelType::Char)
                .expect("build_ll_chr2str_helper_graph");
        let inner = helper.graph.borrow();
        let sb = inner.startblock.borrow();
        let opnames: Vec<&str> = sb.operations.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(
            opnames,
            vec!["malloc_varsize", "getsubstruct", "setarrayitem"]
        );

        let malloc_op = &sb.operations[0];
        assert_eq!(malloc_op.args.len(), 3);
        assert!(matches!(malloc_op.args[0], Hlvalue::Constant(_)));
        assert!(matches!(malloc_op.args[1], Hlvalue::Constant(_)));
        assert!(matches!(
            malloc_op.args[2],
            Hlvalue::Constant(Constant {
                value: ConstValue::Int(1),
                ..
            })
        ));

        let set_op = &sb.operations[2];
        assert_eq!(set_op.opname, "setarrayitem");
        assert!(matches!(
            set_op.args[1],
            Hlvalue::Constant(Constant {
                value: ConstValue::Int(0),
                ..
            })
        ));
    }

    /// The same upstream helper branches on `typeOf(ch)`: UniChar uses
    /// `mallocunicode(1)` and stores a UniChar into `UNICODE.chars[0]`.
    #[test]
    fn build_ll_chr2str_accepts_unichar_to_unicode_shape() {
        let helper = build_ll_chr2str_helper_graph(
            "ll_unichr2unicode",
            UNICODEPTR.clone(),
            LowLevelType::UniChar,
        )
        .expect("build_ll_chr2str_helper_graph for UniChar");
        let inner = helper.graph.borrow();
        let sb = inner.startblock.borrow();
        let opnames: Vec<&str> = sb.operations.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(
            opnames,
            vec!["malloc_varsize", "getsubstruct", "setarrayitem"]
        );
    }

    /// `ll_str2unicode` allocates UNICODE with the source length, then
    /// loops over bytes. The check block raises UnicodeDecodeError on
    /// non-ASCII input before storing through `cast_int_to_unichar`.
    #[test]
    fn build_ll_str2unicode_synthesizes_ascii_checked_copy_loop() {
        let helper =
            build_ll_str2unicode_helper_graph("ll_str2unicode").expect("ll_str2unicode helper");
        let inner = helper.graph.borrow();

        let block_loop_cond = {
            let sb = inner.startblock.borrow();
            let opnames: Vec<&str> = sb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(
                opnames,
                vec![
                    "getsubstruct",
                    "getarraysize",
                    "malloc_varsize",
                    "getsubstruct"
                ]
            );
            let malloc_op = &sb.operations[2];
            assert_eq!(malloc_op.opname, "malloc_varsize");
            assert!(matches!(malloc_op.args[2], Hlvalue::Variable(_)));
            sb.exits[0].borrow().target.as_ref().unwrap().clone()
        };

        let block_check_char = {
            let cb = block_loop_cond.borrow();
            let opnames: Vec<&str> = cb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(opnames, vec!["int_lt"]);
            cb.exits
                .iter()
                .find_map(|link| {
                    let link_borrow = link.borrow();
                    let is_true = matches!(
                        link_borrow.exitcase,
                        Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                    );
                    is_true.then(|| link_borrow.target.as_ref().unwrap().clone())
                })
                .expect("loop true branch")
        };

        let block_store_char = {
            let check = block_check_char.borrow();
            let opnames: Vec<&str> = check
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(opnames, vec!["getarrayitem", "cast_char_to_int", "int_gt"]);
            let mut saw_raise = false;
            let mut store_target = None;
            for link in &check.exits {
                let link_borrow = link.borrow();
                let target = link_borrow.target.as_ref().unwrap();
                let is_true = matches!(
                    link_borrow.exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                );
                if is_true {
                    assert!(std::rc::Rc::ptr_eq(target, &inner.exceptblock));
                    saw_raise = true;
                } else {
                    store_target = Some(target.clone());
                }
            }
            assert!(saw_raise, "non-ASCII branch must raise UnicodeDecodeError");
            store_target.expect("ASCII false branch")
        };

        let store = block_store_char.borrow();
        let opnames: Vec<&str> = store
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            opnames,
            vec!["cast_int_to_unichar", "setarrayitem", "int_add"]
        );
    }

    fn collect_reachable_blocks(
        start: &crate::flowspace::model::BlockRef,
    ) -> Vec<crate::flowspace::model::BlockRef> {
        let mut seen = Vec::new();
        let mut stack = vec![start.clone()];
        while let Some(block) = stack.pop() {
            if seen.iter().any(|known| std::rc::Rc::ptr_eq(known, &block)) {
                continue;
            }
            for link in &block.borrow().exits {
                if let Some(target) = link.borrow().target.as_ref() {
                    stack.push(target.clone());
                }
            }
            seen.push(block);
        }
        seen
    }

    fn unsigned_items_ptr() -> LowLevelType {
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Unsigned)),
        }))
    }

    fn unsigned_list_ptr() -> LowLevelType {
        let items = unsigned_items_ptr();
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(Struct::gc_with_hints(
                "list",
                vec![
                    ("length".into(), LowLevelType::Signed),
                    ("items".into(), items),
                ],
                vec![("list".into(), ConstValue::Bool(true))],
            )),
        }))
    }

    fn dummy_fn(tag: i64) -> Constant {
        Constant::with_concretetype(ConstValue::Int(tag), LowLevelType::Void)
    }

    fn callee_tag(op: &SpaceOperation) -> i64 {
        let Hlvalue::Constant(c) = &op.args[0] else {
            panic!("direct_call callee must be a Constant");
        };
        match c.value {
            ConstValue::Int(n) => n,
            _ => panic!("expected Int dummy callee, got {:?}", c.value),
        }
    }

    fn exit_target(
        block: &crate::flowspace::model::BlockRef,
        want_true: bool,
    ) -> crate::flowspace::model::BlockRef {
        block
            .borrow()
            .exits
            .iter()
            .find_map(|link| {
                let link = link.borrow();
                let is_true = matches!(
                    link.exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                );
                (is_true == want_true).then(|| link.target.as_ref().unwrap().clone())
            })
            .expect("branch target")
    }

    /// `LLHelpers.ll_join_chars` inlines `mallocstr` then copies with
    /// `cast_primitive(Char, Unsigned)` = `cast_uint_to_int` then
    /// `cast_int_to_char`. The original is `unroll_safe`.
    #[test]
    fn build_ll_join_chars_synthesizes_malloc_and_cast_loop() {
        let helper = build_ll_join_chars_helper_graph("ll_join_chars__orig", unsigned_items_ptr())
            .expect("build_ll_join_chars_helper_graph");
        assert!(
            helper.func._jit_unroll_safe_,
            "look_inside_iff marks the original unroll_safe"
        );
        assert_eq!(
            helper.func.annspecialcase.as_deref(),
            Some("specialize:call_location")
        );
        let inner = helper.graph.borrow();
        let blocks = collect_reachable_blocks(&inner.startblock);
        let all_ops: Vec<String> = blocks
            .iter()
            .flat_map(|block| {
                block
                    .borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        for name in [
            "malloc_varsize",
            "getarrayitem",
            "cast_uint_to_int",
            "cast_int_to_char",
            "setarrayitem",
        ] {
            assert!(
                all_ops.iter().any(|op| op == name),
                "expected {name} in {all_ops:?}"
            );
        }
        let loop_cond = inner.startblock.borrow().exits[0]
            .borrow()
            .target
            .as_ref()
            .unwrap()
            .clone();
        let loop_body = exit_target(&loop_cond, true);
        for (label, block) in [("cond", &loop_cond), ("body", &loop_body)] {
            let has_call = block
                .borrow()
                .operations
                .iter()
                .any(|op| op.opname == "direct_call");
            assert!(
                !has_call,
                "ll_join_chars original must not direct_call in the {label}"
            );
        }
    }

    /// `rlib/jit.py look_inside_iff` wrapper: `we_are_jitted` SpecTag
    /// (true link first). Not-jitted calls orig; jitted checks
    /// `isconstant(length) and isvirtual(chars)` and otherwise calls
    /// the trampoline.
    #[test]
    fn build_ll_join_chars_look_inside_iff_wrapper_calls_orig_or_trampoline() {
        let orig_fn = dummy_fn(1);
        let trampoline_fn = dummy_fn(2);
        let helper = build_ll_join_chars_look_inside_iff_helper_graph(
            "ll_join_chars",
            unsigned_items_ptr(),
            orig_fn,
            trampoline_fn,
            dummy_fn(3),
            dummy_fn(4),
        )
        .expect("build_ll_join_chars_look_inside_iff_helper_graph");
        assert_eq!(
            helper.func.annspecialcase, None,
            "look_inside_iff wrapper f does not get call_location"
        );
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert!(startblock.operations.is_empty());
        assert_eq!(startblock.inputargs.len(), 3);
        assert_eq!(startblock.exits.len(), 2);
        let Some(Hlvalue::Constant(c)) = &startblock.exitswitch else {
            panic!("exitswitch must be the we_are_jitted SpecTag constant");
        };
        assert!(matches!(c.value, ConstValue::SpecTag(_)));
        let jit_true = startblock.exits[0].borrow();
        assert!(matches!(
            jit_true.exitcase,
            Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
        ));
        let pred = jit_true.target.as_ref().unwrap().clone();
        let jit_false = startblock.exits[1].borrow();
        assert!(matches!(
            jit_false.exitcase,
            Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)
        ));
        let orig_block = jit_false.target.as_ref().unwrap().clone();
        drop(jit_true);
        drop(jit_false);
        drop(startblock);

        let orig_ops = orig_block.borrow();
        assert_eq!(orig_ops.operations.len(), 1);
        assert_eq!(orig_ops.operations[0].opname, "direct_call");
        assert_eq!(callee_tag(&orig_ops.operations[0]), 1);
        drop(orig_ops);

        let pred_ops = pred.borrow();
        assert_eq!(pred_ops.operations[0].opname, "direct_call");
        assert_eq!(callee_tag(&pred_ops.operations[0]), 3);
        drop(pred_ops);
        let tramp_block = exit_target(&pred, false);
        let tramp_ops = tramp_block.borrow();
        assert_eq!(tramp_ops.operations[0].opname, "direct_call");
        assert_eq!(callee_tag(&tramp_ops.operations[0]), 2);
    }

    /// `LLHelpers.ll_string2list` is `@jit.dont_look_inside`, preallocates
    /// via `RESLIST.ll_newlist`, then copies with Char→Unsigned casts.
    /// `raw_memcopy` is illegal when `DST.OF` is `Unsigned`.
    #[test]
    fn build_ll_string2list_is_dont_look_inside_and_preallocates() {
        let helper = build_ll_string2list_helper_graph(
            "ll_string2list",
            unsigned_list_ptr(),
            LowLevelType::Unsigned,
            dummy_fn(7),
        )
        .expect("build_ll_string2list_helper_graph");
        assert_eq!(helper.func._jit_look_inside_, Some(false));
        let inner = helper.graph.borrow();
        let start = inner.startblock.borrow();
        let start_ops: Vec<&str> = start
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert!(
            start_ops.contains(&"direct_call"),
            "newlist_fn is called before the loop, got {start_ops:?}"
        );
        assert_eq!(
            start_ops.iter().filter(|op| **op == "direct_call").count(),
            1
        );
        let newlist_op = start
            .operations
            .iter()
            .find(|op| op.opname == "direct_call")
            .expect("newlist direct_call");
        assert_eq!(callee_tag(newlist_op), 7);
        let loop_cond = start.exits[0].borrow().target.as_ref().unwrap().clone();
        drop(start);
        let loop_body = exit_target(&loop_cond, true);
        for (label, block) in [("cond", &loop_cond), ("body", &loop_body)] {
            let has_call = block
                .borrow()
                .operations
                .iter()
                .any(|op| op.opname == "direct_call");
            assert!(!has_call, "loop {label} must not call newlist again");
        }
        let blocks = collect_reachable_blocks(&inner.startblock);
        let all_ops: Vec<String> = blocks
            .iter()
            .flat_map(|block| {
                block
                    .borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(
            all_ops.iter().any(|op| op == "cast_char_to_int"),
            "expected cast_char_to_int in {all_ops:?}"
        );
        assert!(
            all_ops.iter().any(|op| op == "cast_int_to_uint"),
            "expected cast_int_to_uint in {all_ops:?}"
        );
        assert!(
            !all_ops.iter().any(|op| op == "raw_memcopy"),
            "Unsigned DST.OF must not raw_memcopy Char"
        );
    }

    /// `LLHelpers.ll_int` keeps PyPy's base parser shape: base range
    /// guard, leading/sign/trailing space loops, digit classification,
    /// base cutoff, ValueError exits, and final `sign * val`.
    #[test]
    fn build_ll_int_synthesizes_base_checked_string_parser_cfg() {
        let helper = build_ll_int_helper_graph("ll_int", STRPTR.clone()).expect("ll_int helper");
        let inner = helper.graph.borrow();

        {
            let startblock = inner.startblock.borrow();
            let start_ops: Vec<&str> = startblock
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(start_ops, vec!["int_ge"]);
        }

        let blocks = collect_reachable_blocks(&inner.startblock);
        for block in &blocks {
            for link in &block.borrow().exits {
                let link = link.borrow();
                if let Some(target) = link.target.as_ref() {
                    assert_eq!(
                        link.args.len(),
                        target.borrow().inputargs.len(),
                        "link arity must match target block arity"
                    );
                }
            }
        }

        let all_ops: Vec<String> = blocks
            .iter()
            .flat_map(|block| {
                block
                    .borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        // Counts, not memberships. `.any()` pinned each opname as PRESENT and
        // discarded multiplicity, so deleting nine of the ten `int_add`s passed.
        //
        // `getarrayitem` is new to this list. It is emitted five times and was
        // asserted nowhere — the only one-directional gap in the set, and the
        // reason the length check below could not have been written before.
        let op_counts: [(&str, usize); 11] = [
            ("int_add", 10),
            ("int_eq", 6),
            ("cast_char_to_int", 5),
            ("getarrayitem", 5),
            ("int_ge", 5),
            ("int_le", 4),
            ("int_lt", 4),
            ("int_sub", 3),
            ("int_mul", 2),
            ("getarraysize", 1),
            ("getsubstruct", 1),
        ];
        for (opname, expected) in op_counts {
            assert_eq!(
                all_ops.iter().filter(|op| op.as_str() == opname).count(),
                expected,
                "ll_int graph must contain {expected} {opname}"
            );
        }
        // Derived from the table rather than restated, so a count that moves
        // cannot leave a stale total beside it. This is what makes the set
        // two-directional: a twelfth opname reds here instead of passing.
        assert_eq!(
            all_ops.len(),
            op_counts.iter().map(|(_, n)| n).sum::<usize>(),
            "the pinned opnames must account for every operation in the graph"
        );

        let raises_value_error = blocks.iter().any(|block| {
            block.borrow().exits.iter().any(|link| {
                let link = link.borrow();
                link.target
                    .as_ref()
                    .is_some_and(|target| std::rc::Rc::ptr_eq(target, &inner.exceptblock))
            })
        });
        assert!(
            raises_value_error,
            "invalid parse paths must raise ValueError"
        );
    }

    /// Unicode inherits the same parser, swapping Char loads for UniChar
    /// loads and `cast_unichar_to_int`.
    #[test]
    fn build_ll_int_uses_unichar_cast_for_unicode() {
        let helper = build_ll_int_helper_graph("ll_unicode_int", UNICODEPTR.clone())
            .expect("unicode ll_int helper");
        let inner = helper.graph.borrow();
        let blocks = collect_reachable_blocks(&inner.startblock);
        let all_ops: Vec<String> = blocks
            .iter()
            .flat_map(|block| {
                block
                    .borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(all_ops.iter().any(|op| op == "cast_unichar_to_int"));
        assert!(!all_ops.iter().any(|op| op == "cast_char_to_int"));
    }

    /// `ll_strlen` synthesised against `Ptr(STR)` produces a single
    /// startblock with `getsubstruct('chars')` then `getarraysize`,
    /// and returns `Signed`. Mirrors upstream
    /// `lltypesystem/rstr.py`:
    /// `def ll_strlen(s): return len(s.chars)`.
    /// `getsubstruct` (not `getfield`) is the correct lltype op for
    /// the inline composite `chars` field — see `rmodel.rs`
    /// `is_container_type` branch.
    #[test]
    fn build_ll_strlen_emits_getsubstruct_chars_then_getarraysize() {
        let helper = build_ll_strlen_helper_graph("ll_strlen", STRPTR.clone())
            .expect("build_ll_strlen_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["getsubstruct", "getarraysize"]);
        // `getsubstruct` first arg is the input `s`, second is the
        // void field-name constant `"chars"`.
        let getsubstruct_args = &startblock.operations[0].args;
        assert_eq!(getsubstruct_args.len(), 2);
        let Hlvalue::Constant(c) = &getsubstruct_args[1] else {
            panic!("getsubstruct's second arg must be a Constant");
        };
        assert!(matches!(c.value, ConstValue::ByteStr(ref b) if b == b"chars"));
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
    }

    /// `ll_unilen` is the `Ptr(UNICODE)` mirror of `ll_strlen` — same
    /// op sequence, just a different input pointer lltype. The chars
    /// Array element lltype derived from the struct body must be
    /// `UniChar`, not `Char`. The `getsubstruct` result is wrapped as
    /// `Ptr(Array(UniChar))` so it carries a valid `SomePtr`-shaped
    /// annotation.
    #[test]
    fn build_ll_unilen_emits_same_op_sequence_with_unichar_chars_lltype() {
        let helper = build_ll_strlen_helper_graph("ll_unilen", UNICODEPTR.clone())
            .expect("build_ll_strlen_helper_graph for UNICODE");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["getsubstruct", "getarraysize"]);
        let getsubstruct_result = &startblock.operations[0].result;
        let Hlvalue::Variable(v) = getsubstruct_result else {
            panic!("getsubstruct result must be a Variable");
        };
        let chars_lltype = v
            .concretetype
            .borrow()
            .clone()
            .expect("chars var must have concretetype");
        let LowLevelType::Ptr(ptr) = chars_lltype else {
            panic!("chars var must be Ptr");
        };
        let PtrTarget::Array(arr) = &ptr.TO else {
            panic!("chars Ptr.TO must be Array");
        };
        assert_eq!(arr.OF, LowLevelType::UniChar);
    }

    /// The synthesised helper graph carries the helper-identity
    /// `name` on both its FunctionGraph and `func` slot, and exposes
    /// the input slot via the startblock's input variable. Required so
    /// `helper_pygraph_from_graph`'s caller cache dedupes by
    /// `(name, args)` rather than by graph identity.
    #[test]
    fn build_ll_strlen_helper_carries_name_and_input_s_variable() {
        let helper = build_ll_strlen_helper_graph("ll_strlen", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_strlen");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_strlen");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 1);
        let Hlvalue::Variable(v) = &startblock.inputargs[0] else {
            panic!("startblock input must be a Variable");
        };
        assert!(
            v.name().starts_with('s'),
            "input variable name = {:?}",
            v.name()
        );
    }

    /// Passing a non-Ptr lltype should be rejected — upstream's
    /// `len(s.chars)` is only meaningful when `s: Ptr(STR/UNICODE)`.
    /// Guards a future caller from accidentally requesting a
    /// `ll_strlen` against e.g. `LowLevelType::Char`.
    #[test]
    fn build_ll_strlen_rejects_non_ptr_input_lltype() {
        let err = build_ll_strlen_helper_graph("ll_strlen", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }

    /// rstr.py — `mallocstr(length)` lowers to
    /// `malloc_varsize(STR, gc, length)` + `setfield(r, 'hash', 0)`,
    /// returning `Ptr(STR)`.
    #[test]
    fn build_ll_mallocstr_emits_malloc_varsize_then_hash_zero() {
        let helper = build_ll_mallocstr_helper_graph("mallocstr", STRPTR.clone())
            .expect("build_ll_mallocstr_helper_graph");
        assert_eq!(helper.func.name, "mallocstr");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["malloc_varsize", "setfield"]);
        // setfield's field-name constant is `"hash"`, value constant 0.
        let setfield_args = &startblock.operations[1].args;
        assert_eq!(setfield_args.len(), 3);
        let Hlvalue::Constant(field) = &setfield_args[1] else {
            panic!("setfield field-name must be a Constant");
        };
        assert!(matches!(field.value, ConstValue::ByteStr(ref b) if b == b"hash"));
        let Hlvalue::Constant(zero) = &setfield_args[2] else {
            panic!("setfield value must be a Constant");
        };
        assert!(matches!(zero.value, ConstValue::Int(0)));
        // The returned value carries `Ptr(STR)`.
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(STRPTR.clone()));
        assert_eq!(startblock.inputargs.len(), 1);
    }

    /// `mallocunicode` is the `Ptr(UNICODE)` mirror — same op sequence,
    /// distinct struct/pointer lltype and helper identity.
    #[test]
    fn build_ll_mallocunicode_mirrors_mallocstr_against_unicode() {
        let helper = build_ll_mallocstr_helper_graph("mallocunicode", UNICODEPTR.clone())
            .expect("build_ll_mallocstr_helper_graph for UNICODE");
        assert_eq!(helper.func.name, "mallocunicode");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["malloc_varsize", "setfield"]);
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(UNICODEPTR.clone()));
    }

    /// `mallocstr` is only meaningful against `Ptr(STR/UNICODE)`.
    #[test]
    fn build_ll_mallocstr_rejects_non_ptr_input_lltype() {
        let err = build_ll_mallocstr_helper_graph("mallocstr", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }

    /// `ll_str_is_true` synthesised against `Ptr(STR)` produces a
    /// 2-block CFG: startblock with `ptr_nonzero(s)` + boolean
    /// exitswitch (True → block_check_len, False → returnblock with
    /// `Bool(false)`), and a check-length block emitting
    /// `getsubstruct('chars')` + `getarraysize` + `int_ne(len, 0)`.
    /// Mirrors upstream `rstr.py:944-947`:
    /// `def ll_str_is_true(cls, s): return bool(s) and cls.ll_strlen(s) != 0`.
    #[test]
    fn build_ll_str_is_true_synthesizes_2_block_short_circuit_cfg() {
        let helper = build_ll_str_is_true_helper_graph("ll_str_is_true", STRPTR.clone())
            .expect("build_ll_str_is_true_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["ptr_nonzero"]);
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);

        // False branch: direct return with Bool(false) link arg.
        let false_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("False exit link present");
        let false_first_arg = false_link
            .borrow()
            .args
            .first()
            .and_then(|opt| opt.as_ref())
            .cloned()
            .expect("False link first arg present");
        assert!(matches!(
            false_first_arg,
            Hlvalue::Constant(c) if c.value == ConstValue::Bool(false)
        ));

        // True branch: forwards `s` ptr to block_check_len.
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        let target = true_link.borrow().target.as_ref().unwrap().clone();
        let target_block = target.borrow();
        let target_ops: Vec<&str> = target_block
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(target_ops, vec!["getsubstruct", "getarraysize", "int_ne"]);
    }

    /// `ll_unicode_is_true` is the `Ptr(UNICODE)` mirror of
    /// `ll_str_is_true` — same op sequence, but the chars-array `Ptr`
    /// derived in `block_check_len` carries `Array(UniChar)` rather
    /// than `Array(Char)` so that downstream length comparison
    /// dispatches against the unicode struct shape.
    #[test]
    fn build_ll_unicode_is_true_uses_unichar_chars_lltype() {
        let helper = build_ll_str_is_true_helper_graph("ll_unicode_is_true", UNICODEPTR.clone())
            .expect("build_ll_str_is_true_helper_graph for UNICODE");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        let target = true_link.borrow().target.as_ref().unwrap().clone();
        let target_block = target.borrow();
        let getsubstruct_result = &target_block.operations[0].result;
        let Hlvalue::Variable(v) = getsubstruct_result else {
            panic!("getsubstruct result must be a Variable");
        };
        let chars_lltype = v
            .concretetype
            .borrow()
            .clone()
            .expect("chars var must have concretetype");
        let LowLevelType::Ptr(ptr) = chars_lltype else {
            panic!("chars var must be Ptr");
        };
        let PtrTarget::Array(arr) = &ptr.TO else {
            panic!("chars Ptr.TO must be Array");
        };
        assert_eq!(arr.OF, LowLevelType::UniChar);
    }

    /// The synthesised helper graph carries the helper-identity `name`
    /// on both its FunctionGraph and `func` slot, returns Bool, and
    /// exposes the input `s` via the startblock's input variable.
    /// Mirror of `build_ll_strlen_helper_carries_name_and_input_s_variable`.
    #[test]
    fn build_ll_str_is_true_carries_name_input_s_and_bool_return() {
        let helper = build_ll_str_is_true_helper_graph("ll_str_is_true", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_str_is_true");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_str_is_true");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 1);
        let Hlvalue::Variable(v) = &startblock.inputargs[0] else {
            panic!("startblock input must be a Variable");
        };
        assert!(
            v.name().starts_with('s'),
            "input variable name = {:?}",
            v.name()
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Bool),
            "ll_str_is_true must return Bool"
        );
    }

    /// Passing a non-Ptr lltype should be rejected — propagated from
    /// the shared `chars_array_ptr_lltype_from_strptr` guard.
    #[test]
    fn build_ll_str_is_true_rejects_non_ptr_input_lltype() {
        let err = build_ll_str_is_true_helper_graph("ll_str_is_true", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }

    /// `ll_str` (`str()` of a `Ptr(STR)`) synthesises a single
    /// `ptr_nonzero` branch: the True arm forwards the receiver `s`
    /// unchanged, the False arm returns the prebuilt `'None'` constant.
    /// Both arms target the returnblock, which carries the `Ptr(STR)`
    /// result type.
    #[test]
    fn build_ll_str_string_synthesizes_identity_or_none_branch() {
        let helper = build_ll_str_string_helper_graph("ll_str", STRPTR.clone())
            .expect("build_ll_str_string_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["ptr_nonzero"]);
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);

        // True arm: forwards the receiver `s` (a Variable) unchanged.
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        let true_first_arg = true_link
            .borrow()
            .args
            .first()
            .and_then(|opt| opt.as_ref())
            .cloned()
            .expect("True link first arg present");
        assert!(matches!(true_first_arg, Hlvalue::Variable(_)));

        // False arm: returns the prebuilt `'None'` STR constant.
        let false_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("False exit link present");
        let false_first_arg = false_link
            .borrow()
            .args
            .first()
            .and_then(|opt| opt.as_ref())
            .cloned()
            .expect("False link first arg present");
        assert!(matches!(
            false_first_arg,
            Hlvalue::Constant(c) if matches!(c.value, ConstValue::LLPtr(_))
        ));

        // Result type is `Ptr(STR)`, and the helper-identity name is carried.
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(STRPTR.clone()));
        assert_eq!(helper.func.name, "ll_str");
    }

    /// `ll_streq` synthesised against `Ptr(STR)` produces the expected
    /// 6-block-plus-returnblock CFG. Each block carries the operations
    /// listed in the synthesizer docstring; the loop edge from
    /// `block_loop_body` back to `block_loop_cond` realises the
    /// upstream `while j < len1` cycle.
    #[test]
    fn build_ll_streq_synthesizes_short_circuit_loop_cfg_for_str() {
        let helper = build_ll_streq_helper_graph("ll_streq", STRPTR.clone())
            .expect("build_ll_streq_helper_graph for STR");
        let inner = helper.graph.borrow();

        // start: ptr_eq.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["ptr_eq"]);
        assert_eq!(startblock.exits.len(), 2);

        // start False branch → block_null_check_s1 (ptr_nonzero).
        let start_false = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("start False exit");
        let null_check_s1 = start_false.borrow().target.as_ref().unwrap().clone();
        let null_check_s1_borrow = null_check_s1.borrow();
        let null_s1_ops: Vec<&str> = null_check_s1_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(null_s1_ops, vec!["ptr_nonzero"]);

        // null_check_s1 True branch → block_null_check_s2 (ptr_nonzero).
        let s1_true = null_check_s1_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("null_check_s1 True exit");
        let null_check_s2 = s1_true.borrow().target.as_ref().unwrap().clone();
        let null_check_s2_borrow = null_check_s2.borrow();
        let null_s2_ops: Vec<&str> = null_check_s2_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(null_s2_ops, vec!["ptr_nonzero"]);

        // null_check_s2 True branch → block_compare_lens.
        let s2_true = null_check_s2_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("null_check_s2 True exit");
        let compare_lens = s2_true.borrow().target.as_ref().unwrap().clone();
        let compare_lens_borrow = compare_lens.borrow();
        let lens_ops: Vec<&str> = compare_lens_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            lens_ops,
            vec![
                "getsubstruct",
                "getarraysize",
                "getsubstruct",
                "getarraysize",
                "int_eq",
            ]
        );

        // compare_lens True branch → block_loop_cond (int_lt).
        let lens_true = compare_lens_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("compare_lens True exit");
        let loop_cond = lens_true.borrow().target.as_ref().unwrap().clone();
        let loop_cond_borrow = loop_cond.borrow();
        let cond_ops: Vec<&str> = loop_cond_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(cond_ops, vec!["int_lt"]);

        // loop_cond True branch → block_loop_body (getarrayitem,
        // getarrayitem, char_eq, int_add).
        let cond_true = loop_cond_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("loop_cond True exit");
        let loop_body = cond_true.borrow().target.as_ref().unwrap().clone();
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec!["getarrayitem", "getarrayitem", "char_eq", "int_add"]
        );

        // loop_body True branch loops back to block_loop_cond.
        let body_true = loop_body_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("loop_body True exit");
        let body_true_target = body_true.borrow().target.as_ref().unwrap().clone();
        // Identity: same Rc as loop_cond.
        assert!(std::rc::Rc::ptr_eq(&body_true_target, &loop_cond));
    }

    /// `ll_streq` against `Ptr(UNICODE)` differs only in the chars
    /// element comparison op — `unichar_eq` instead of `char_eq`. The
    /// body block's third op should be `unichar_eq`.
    #[test]
    fn build_ll_streq_uses_unichar_eq_op_for_unicode() {
        let helper = build_ll_streq_helper_graph("ll_unicode_eq", UNICODEPTR.clone())
            .expect("build_ll_streq_helper_graph for UNICODE");
        let inner = helper.graph.borrow();

        // Walk: start False → null_check_s1 True → null_check_s2 True
        // → compare_lens True → loop_cond True → loop_body.
        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }
        let null_check_s1 = pick_exit_target(&inner.startblock, false);
        let null_check_s2 = pick_exit_target(&null_check_s1, true);
        let compare_lens = pick_exit_target(&null_check_s2, true);
        let loop_cond = pick_exit_target(&compare_lens, true);
        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec!["getarrayitem", "getarrayitem", "unichar_eq", "int_add"]
        );
    }

    /// The synthesised `ll_streq` graph carries the helper-identity
    /// `name` on both `FunctionGraph.name` and `func.name`, exposes
    /// the two `s1` / `s2` input slots via the startblock's
    /// inputargs, and returns Bool. Mirrors the existing
    /// `build_ll_strlen_helper_carries_name_and_input_s_variable`
    /// covenant so callers (`gen_eq_function`) can dedupe by
    /// `(name, args)`.
    #[test]
    fn build_ll_streq_carries_name_two_inputs_and_bool_return() {
        let helper = build_ll_streq_helper_graph("ll_streq", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_streq");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_streq");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 2);
        for (i, expected) in ["s1", "s2"].iter().enumerate() {
            let Hlvalue::Variable(v) = &startblock.inputargs[i] else {
                panic!("startblock input {i} must be a Variable");
            };
            assert!(
                v.name().starts_with(expected),
                "input {i} variable name = {:?}, expected prefix {expected}",
                v.name()
            );
        }
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Bool),
            "ll_streq must return Bool"
        );
    }

    /// Passing a non-Ptr lltype should be rejected — propagated from
    /// the shared `chars_array_ptr_lltype_from_strptr` guard.
    #[test]
    fn build_ll_streq_rejects_non_ptr_input_lltype() {
        let err = build_ll_streq_helper_graph("ll_streq", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }

    /// `ll_strcmp` synthesised against `Ptr(STR)` produces the expected
    /// 7-block-plus-returnblock CFG. Walks: start False → block_s1_null
    /// (ptr_nonzero); start True → block_s1_nonnull (ptr_nonzero);
    /// block_s1_nonnull True → block_both_nonnull
    /// (getsubstruct/getarraysize × 2 + int_lt); block_both_nonnull True
    /// → block_loop_cond (int_lt); block_loop_cond True → block_loop_body
    /// (getarrayitem × 2 + cast_char_to_int × 2 + int_sub + int_ne +
    /// int_add); block_loop_cond False → block_return_len_diff (int_sub);
    /// block_loop_body False (chars match) cycles back to block_loop_cond.
    #[test]
    fn build_ll_strcmp_synthesizes_min_loop_cfg_with_back_edge_for_str() {
        let helper = build_ll_strcmp_helper_graph("ll_strcmp", STRPTR.clone())
            .expect("build_ll_strcmp_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        // start: ptr_nonzero(s1).
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["ptr_nonzero"]);
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        // start False → block_s1_null (ptr_nonzero).
        let s1_null = pick_exit_target(&inner.startblock, false);
        let s1_null_borrow = s1_null.borrow();
        let s1_null_ops: Vec<&str> = s1_null_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(s1_null_ops, vec!["ptr_nonzero"]);
        drop(s1_null_borrow);

        // start True → block_s1_nonnull (ptr_nonzero).
        let s1_nn = pick_exit_target(&inner.startblock, true);
        let s1_nn_borrow = s1_nn.borrow();
        let s1_nn_ops: Vec<&str> = s1_nn_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(s1_nn_ops, vec!["ptr_nonzero"]);
        drop(s1_nn_borrow);

        // block_s1_nonnull True → block_both_nonnull (getsubstruct,
        // getarraysize, getsubstruct, getarraysize, int_lt).
        let both_nn = pick_exit_target(&s1_nn, true);
        let both_nn_borrow = both_nn.borrow();
        let both_nn_ops: Vec<&str> = both_nn_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            both_nn_ops,
            vec![
                "getsubstruct",
                "getarraysize",
                "getsubstruct",
                "getarraysize",
                "int_lt",
            ]
        );
        drop(both_nn_borrow);

        // block_both_nonnull True → block_loop_cond (int_lt).
        let loop_cond = pick_exit_target(&both_nn, true);
        let loop_cond_borrow = loop_cond.borrow();
        let cond_ops: Vec<&str> = loop_cond_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(cond_ops, vec!["int_lt"]);
        drop(loop_cond_borrow);

        // block_loop_cond True → block_loop_body (getarrayitem×2,
        // cast_char_to_int×2, int_sub, int_ne, int_add).
        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec![
                "getarrayitem",
                "getarrayitem",
                "cast_char_to_int",
                "cast_char_to_int",
                "int_sub",
                "int_ne",
                "int_add",
            ]
        );
        drop(loop_body_borrow);

        // block_loop_body False (chars match) cycles to block_loop_cond.
        let body_false_target = pick_exit_target(&loop_body, false);
        assert!(std::rc::Rc::ptr_eq(&body_false_target, &loop_cond));

        // block_loop_cond False → block_return_len_diff (int_sub).
        let len_diff = pick_exit_target(&loop_cond, false);
        let len_diff_borrow = len_diff.borrow();
        let len_diff_ops: Vec<&str> = len_diff_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(len_diff_ops, vec!["int_sub"]);
    }

    /// `ll_strcmp` against `Ptr(UNICODE)` differs only in the chars
    /// element `ord` op — `cast_unichar_to_int` instead of
    /// `cast_char_to_int`. The body block's third + fourth ops should
    /// be `cast_unichar_to_int`.
    #[test]
    fn build_ll_strcmp_uses_cast_unichar_to_int_op_for_unicode() {
        let helper = build_ll_strcmp_helper_graph("ll_unicode_cmp", UNICODEPTR.clone())
            .expect("build_ll_strcmp_helper_graph for UNICODE");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        // start True → s1_nonnull True → both_nonnull True → loop_cond
        // True → loop_body.
        let s1_nn = pick_exit_target(&inner.startblock, true);
        let both_nn = pick_exit_target(&s1_nn, true);
        let loop_cond = pick_exit_target(&both_nn, true);
        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec![
                "getarrayitem",
                "getarrayitem",
                "cast_unichar_to_int",
                "cast_unichar_to_int",
                "int_sub",
                "int_ne",
                "int_add",
            ]
        );
    }

    /// The synthesised `ll_strcmp` graph carries the helper-identity
    /// `name` on both `FunctionGraph.name` and `func.name`, exposes
    /// the two `s1` / `s2` input slots via the startblock's
    /// inputargs, and returns `Signed` (since the diff / len-diff
    /// branches return raw integers and the NULL-pair branches feed
    /// `Signed(1)` / `Signed(0)` into the same slot).
    #[test]
    fn build_ll_strcmp_carries_name_two_inputs_and_signed_return() {
        let helper = build_ll_strcmp_helper_graph("ll_strcmp", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_strcmp");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_strcmp");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 2);
        for (i, expected) in ["s1", "s2"].iter().enumerate() {
            let Hlvalue::Variable(v) = &startblock.inputargs[i] else {
                panic!("startblock input {i} must be a Variable");
            };
            assert!(
                v.name().starts_with(expected),
                "input {i} variable name = {:?}, expected prefix {expected}",
                v.name()
            );
        }
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed),
            "ll_strcmp must return Signed"
        );
    }

    /// Passing a non-Ptr lltype should be rejected — propagated from
    /// the shared `chars_array_ptr_lltype_from_strptr` guard.
    #[test]
    fn build_ll_strcmp_rejects_non_ptr_input_lltype() {
        let err = build_ll_strcmp_helper_graph("ll_strcmp", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }

    /// `ll_startswith` synthesised against `Ptr(STR)` produces the
    /// expected 3-block-plus-returnblock CFG. start (getsubstruct ×2,
    /// getarraysize ×2, int_lt) → block_loop_cond (int_lt) →
    /// block_loop_body (getarrayitem ×2, char_eq, int_add). Loop body
    /// True branch cycles back to loop_cond.
    #[test]
    fn build_ll_startswith_synthesizes_length_check_then_loop_cfg_for_str() {
        let helper = build_ll_startswith_helper_graph("ll_startswith", STRPTR.clone())
            .expect("build_ll_startswith_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        // start: 4 chars/len ops + int_lt.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            start_ops,
            vec![
                "getsubstruct",
                "getarraysize",
                "getsubstruct",
                "getarraysize",
                "int_lt",
            ]
        );
        drop(startblock);

        // start False (len1 >= len2) → block_loop_cond (int_lt).
        let loop_cond = pick_exit_target(&inner.startblock, false);
        let loop_cond_borrow = loop_cond.borrow();
        let cond_ops: Vec<&str> = loop_cond_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(cond_ops, vec!["int_lt"]);
        drop(loop_cond_borrow);

        // block_loop_cond True → block_loop_body.
        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec!["getarrayitem", "getarrayitem", "char_eq", "int_add"]
        );
        drop(loop_body_borrow);

        // block_loop_body True (chars match) cycles back to loop_cond.
        let body_true_target = pick_exit_target(&loop_body, true);
        assert!(std::rc::Rc::ptr_eq(&body_true_target, &loop_cond));
    }

    /// `ll_startswith` against `Ptr(UNICODE)` differs only in the
    /// element comparison op — `unichar_eq` instead of `char_eq`.
    #[test]
    fn build_ll_startswith_uses_unichar_eq_op_for_unicode() {
        let helper = build_ll_startswith_helper_graph("ll_unicode_startswith", UNICODEPTR.clone())
            .expect("build_ll_startswith_helper_graph for UNICODE");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        // start False → loop_cond True → loop_body.
        let loop_cond = pick_exit_target(&inner.startblock, false);
        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec!["getarrayitem", "getarrayitem", "unichar_eq", "int_add"]
        );
    }

    /// The synthesised `ll_startswith` graph carries the helper-identity
    /// `name` on both `FunctionGraph.name` and `func.name`, exposes the
    /// two `s1` / `s2` input slots, and returns Bool. Mirrors the
    /// existing `build_ll_streq_carries_name_two_inputs_and_bool_return`
    /// covenant.
    #[test]
    fn build_ll_startswith_carries_name_two_inputs_and_bool_return() {
        let helper = build_ll_startswith_helper_graph("ll_startswith", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_startswith");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_startswith");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 2);
        for (i, expected) in ["s1", "s2"].iter().enumerate() {
            let Hlvalue::Variable(v) = &startblock.inputargs[i] else {
                panic!("startblock input {i} must be a Variable");
            };
            assert!(
                v.name().starts_with(expected),
                "input {i} variable name = {:?}, expected prefix {expected}",
                v.name()
            );
        }
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Bool),
            "ll_startswith must return Bool"
        );
    }

    /// Passing a non-Ptr lltype should be rejected — propagated from
    /// the shared `chars_array_ptr_lltype_from_strptr` guard.
    #[test]
    fn build_ll_startswith_rejects_non_ptr_input_lltype() {
        let err = build_ll_startswith_helper_graph("ll_startswith", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }

    /// `ll_endswith` synthesised against `Ptr(STR)` produces the
    /// expected 3-block-plus-returnblock CFG. start (getsubstruct ×2,
    /// getarraysize ×2, int_sub offset, int_lt) → block_loop_cond
    /// (int_lt) → block_loop_body (int_add idx, getarrayitem ×2,
    /// char_eq, int_add j_next). Loop body True branch cycles back to
    /// loop_cond.
    #[test]
    fn build_ll_endswith_synthesizes_offset_loop_cfg_for_str() {
        let helper = build_ll_endswith_helper_graph("ll_endswith", STRPTR.clone())
            .expect("build_ll_endswith_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            start_ops,
            vec![
                "getsubstruct",
                "getarraysize",
                "getsubstruct",
                "getarraysize",
                "int_sub",
                "int_lt",
            ]
        );
        drop(startblock);

        let loop_cond = pick_exit_target(&inner.startblock, false);
        let loop_cond_borrow = loop_cond.borrow();
        let cond_ops: Vec<&str> = loop_cond_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(cond_ops, vec!["int_lt"]);
        drop(loop_cond_borrow);

        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec![
                "int_add",
                "getarrayitem",
                "getarrayitem",
                "char_eq",
                "int_add",
            ]
        );
        drop(loop_body_borrow);

        // Loop body True branch (chars match) cycles back to loop_cond.
        let body_true_target = pick_exit_target(&loop_body, true);
        assert!(std::rc::Rc::ptr_eq(&body_true_target, &loop_cond));
    }

    /// `ll_endswith` against `Ptr(UNICODE)` differs only in the
    /// element comparison op — `unichar_eq` instead of `char_eq`.
    #[test]
    fn build_ll_endswith_uses_unichar_eq_op_for_unicode() {
        let helper = build_ll_endswith_helper_graph("ll_unicode_endswith", UNICODEPTR.clone())
            .expect("build_ll_endswith_helper_graph for UNICODE");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let loop_cond = pick_exit_target(&inner.startblock, false);
        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec![
                "int_add",
                "getarrayitem",
                "getarrayitem",
                "unichar_eq",
                "int_add",
            ]
        );
    }

    /// The synthesised `ll_endswith` graph carries the helper-identity
    /// `name`, exposes the two `s1` / `s2` input slots, and returns Bool.
    #[test]
    fn build_ll_endswith_carries_name_two_inputs_and_bool_return() {
        let helper = build_ll_endswith_helper_graph("ll_endswith", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_endswith");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_endswith");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 2);
        for (i, expected) in ["s1", "s2"].iter().enumerate() {
            let Hlvalue::Variable(v) = &startblock.inputargs[i] else {
                panic!("startblock input {i} must be a Variable");
            };
            assert!(
                v.name().starts_with(expected),
                "input {i} variable name = {:?}, expected prefix {expected}",
                v.name()
            );
        }
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Bool),
            "ll_endswith must return Bool"
        );
    }

    /// Passing a non-Ptr lltype should be rejected — propagated from
    /// the shared `chars_array_ptr_lltype_from_strptr` guard.
    #[test]
    fn build_ll_endswith_rejects_non_ptr_input_lltype() {
        let err = build_ll_endswith_helper_graph("ll_endswith", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }

    /// `ll_startswith_char` synthesised against `Ptr(STR)` produces
    /// the expected 2-block-plus-returnblock CFG. start
    /// (getsubstruct, getarraysize, int_eq) → block_compare
    /// (getarrayitem at index 0, char_eq).
    #[test]
    fn build_ll_startswith_char_synthesizes_empty_check_then_compare_cfg_for_str() {
        let helper = build_ll_startswith_char_helper_graph("ll_startswith_char", STRPTR.clone())
            .expect("build_ll_startswith_char_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "getarraysize", "int_eq"]);
        drop(startblock);

        let compare = pick_exit_target(&inner.startblock, false);
        let compare_borrow = compare.borrow();
        let compare_ops: Vec<&str> = compare_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        // No int_sub — startswith reads chars[Constant(0)] directly.
        assert_eq!(compare_ops, vec!["getarrayitem", "char_eq"]);
    }

    /// `ll_endswith_char` against `Ptr(STR)` inserts an additional
    /// `int_sub(length, 1) -> idx` op before the `getarrayitem`.
    #[test]
    fn build_ll_endswith_char_synthesizes_int_sub_idx_for_str() {
        let helper = build_ll_endswith_char_helper_graph("ll_endswith_char", STRPTR.clone())
            .expect("build_ll_endswith_char_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let compare = pick_exit_target(&inner.startblock, false);
        let compare_borrow = compare.borrow();
        let compare_ops: Vec<&str> = compare_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(compare_ops, vec!["int_sub", "getarrayitem", "char_eq"]);
    }

    /// UNICODE variants substitute `unichar_eq` for the comparison op.
    #[test]
    fn build_ll_startswith_endswith_char_use_unichar_eq_op_for_unicode() {
        for (name, builder) in [
            (
                "ll_unicode_startswith_char",
                build_ll_startswith_char_helper_graph
                    as fn(&str, LowLevelType) -> Result<PyGraph, TyperError>,
            ),
            (
                "ll_unicode_endswith_char",
                build_ll_endswith_char_helper_graph
                    as fn(&str, LowLevelType) -> Result<PyGraph, TyperError>,
            ),
        ] {
            let helper = builder(name, UNICODEPTR.clone()).expect(name);
            let inner = helper.graph.borrow();
            let block_borrow = inner.startblock.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
                .expect("compare branch");
            let compare = link.borrow().target.as_ref().unwrap().clone();
            let compare_borrow = compare.borrow();
            let opnames: Vec<&str> = compare_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert!(
                opnames.contains(&"unichar_eq"),
                "{name}: expected 'unichar_eq', got {opnames:?}"
            );
        }
    }

    /// Both helpers carry the helper-identity name + two inputargs
    /// (s + ch) and return Bool.
    #[test]
    fn build_ll_startswith_endswith_char_carry_name_two_inputs_and_bool_return() {
        for (name, builder) in [
            (
                "ll_startswith_char",
                build_ll_startswith_char_helper_graph
                    as fn(&str, LowLevelType) -> Result<PyGraph, TyperError>,
            ),
            (
                "ll_endswith_char",
                build_ll_endswith_char_helper_graph
                    as fn(&str, LowLevelType) -> Result<PyGraph, TyperError>,
            ),
        ] {
            let helper = builder(name, STRPTR.clone()).unwrap();
            assert_eq!(helper.func.name, name);
            let inner = helper.graph.borrow();
            assert_eq!(inner.name, name);
            let startblock = inner.startblock.borrow();
            assert_eq!(startblock.inputargs.len(), 2);
            let Hlvalue::Variable(v_s) = &startblock.inputargs[0] else {
                panic!("startblock input 0 must be Variable");
            };
            assert!(v_s.name().starts_with("s"));
            let Hlvalue::Variable(v_ch) = &startblock.inputargs[1] else {
                panic!("startblock input 1 must be Variable");
            };
            assert!(v_ch.name().starts_with("ch"));
            let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
                panic!("returnblock inputarg must be Variable");
            };
            assert_eq!(
                ret.concretetype.borrow().clone(),
                Some(LowLevelType::Bool),
                "{name}: must return Bool"
            );
        }
    }

    /// `_hash_string` synthesised against `Ptr(Array(Char))` produces
    /// the expected 5-block-plus-returnblock CFG: start (getarraysize +
    /// int_eq) → block_init (getarrayitem 0 + cast + int_lshift) →
    /// block_loop_cond (int_lt) → block_loop_body (int_mul +
    /// getarrayitem + cast + int_xor + int_add) → block_finalize
    /// (int_xor). Loop body always cycles back to loop_cond.
    #[test]
    fn build_hash_string_synthesizes_fnv_loop_cfg_for_char_array() {
        let chars_array_ptr = chars_array_ptr_lltype_from_strptr(&STRPTR.clone())
            .expect("chars_array_ptr_lltype_from_strptr");
        let helper = build_hash_string_helper_graph("_hash_string", chars_array_ptr)
            .expect("build_hash_string_helper_graph for STR chars Array");
        let inner = helper.graph.borrow();

        fn pick_exit_target_bool(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }
        fn first_exit_target(
            block: &crate::flowspace::model::BlockRef,
        ) -> crate::flowspace::model::BlockRef {
            let block_borrow = block.borrow();
            let link = block_borrow.exits.first().expect("at least one exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        // start: getarraysize + int_eq.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getarraysize", "int_eq"]);
        drop(startblock);

        // start False -> block_init.
        let init = pick_exit_target_bool(&inner.startblock, false);
        let init_borrow = init.borrow();
        let init_ops: Vec<&str> = init_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            init_ops,
            vec!["getarrayitem", "cast_char_to_int", "int_lshift"]
        );
        drop(init_borrow);

        // init -> block_loop_cond (single unconditional exit).
        let loop_cond = first_exit_target(&init);
        let loop_cond_borrow = loop_cond.borrow();
        let cond_ops: Vec<&str> = loop_cond_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(cond_ops, vec!["int_lt"]);
        drop(loop_cond_borrow);

        // loop_cond True -> block_loop_body.
        let loop_body = pick_exit_target_bool(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec![
                "int_mul",
                "getarrayitem",
                "cast_char_to_int",
                "int_xor",
                "int_add",
            ]
        );
        drop(loop_body_borrow);

        // loop_body cycles back to loop_cond.
        let body_target = first_exit_target(&loop_body);
        assert!(std::rc::Rc::ptr_eq(&body_target, &loop_cond));

        // loop_cond False -> block_finalize (int_xor).
        let finalize = pick_exit_target_bool(&loop_cond, false);
        let finalize_borrow = finalize.borrow();
        let finalize_ops: Vec<&str> = finalize_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(finalize_ops, vec!["int_xor"]);
    }

    /// UNICODE chars Array routes through `cast_unichar_to_int` for
    /// both `ord` sites (init's chars[0] and the loop body).
    #[test]
    fn build_hash_string_uses_cast_unichar_to_int_for_unicode_array() {
        let chars_array_ptr = chars_array_ptr_lltype_from_strptr(&UNICODEPTR.clone())
            .expect("chars_array_ptr_lltype_from_strptr UNICODE");
        let helper = build_hash_string_helper_graph("_hash_unicode_string", chars_array_ptr)
            .expect("build_hash_string_helper_graph UNICODE");
        let inner = helper.graph.borrow();

        let block_borrow = inner.startblock.borrow();
        let init_link = block_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("init branch");
        let init = init_link.borrow().target.as_ref().unwrap().clone();
        let init_borrow = init.borrow();
        let init_ops: Vec<&str> = init_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(init_ops[1], "cast_unichar_to_int");
    }

    /// Helper-identity covenant: name appears on FunctionGraph.name +
    /// func.name; chars input slot named `chars`; returns Signed.
    #[test]
    fn build_hash_string_carries_name_chars_input_and_signed_return() {
        let chars_array_ptr = chars_array_ptr_lltype_from_strptr(&STRPTR.clone())
            .expect("chars_array_ptr_lltype_from_strptr");
        let helper = build_hash_string_helper_graph("_hash_string", chars_array_ptr).unwrap();
        assert_eq!(helper.func.name, "_hash_string");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "_hash_string");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 1);
        let Hlvalue::Variable(v_chars) = &startblock.inputargs[0] else {
            panic!("startblock input must be Variable");
        };
        assert!(v_chars.name().starts_with("chars"));
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed),
            "_hash_string must return Signed"
        );
    }

    /// Non-Ptr / non-Array input must error at the boundary.
    #[test]
    fn build_hash_string_rejects_non_ptr_array_input_lltype() {
        let err = build_hash_string_helper_graph("_hash_string", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(Array"));
    }

    /// `ll_hash_string` synthesised against `Ptr(STR)` produces a
    /// single-block helper that lowers to `getsubstruct(s, 'chars') +
    /// direct_call(_hash_string, chars)`, then unconditionally links
    /// to returnblock with `x`. The `direct_call` first arg is a
    /// funcptr Constant (LLPtr) whose target is the inner
    /// `_hash_string` helper graph.
    #[test]
    fn build_ll_hash_string_emits_subhelper_direct_call_for_str() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_hash_string_helper_graph(
            &rtyper,
            "ll_hash_string",
            STRPTR.clone(),
            "_hash_string",
        )
        .expect("build_ll_hash_string_helper_graph for STR");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "direct_call"]);

        // direct_call's first arg must be a funcptr Constant (LLPtr).
        let direct_call_op = &startblock.operations[1];
        let Hlvalue::Constant(c_func) = &direct_call_op.args[0] else {
            panic!("direct_call arg[0] must be a Constant funcptr");
        };
        let dbg = format!("{:?}", c_func.value);
        assert!(dbg.contains("LLPtr"), "expected LLPtr funcptr, got {dbg}");
    }

    /// `_ll_strhash` synthesised against `Ptr(STR)` produces a
    /// 2-block-plus-returnblock CFG: start (getsubstruct + direct_call
    /// + int_eq) branches on `is_zero`; both branches converge on
    ///   `block_set_hash` (setfield). The True branch carries the
    ///   zero-fixup constant 29872897.
    #[test]
    fn build_ll_strhash_internal_emits_zero_fixup_then_setfield_for_str() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_strhash_internal_helper_graph(
            &rtyper,
            "_ll_strhash",
            STRPTR.clone(),
            "_hash_string",
        )
        .expect("build_ll_strhash_internal_helper_graph for STR");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "direct_call", "int_eq"]);

        // True branch (is_zero == true): link arg[1] is Signed(29872897).
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit");
        let link_borrow = true_link.borrow();
        let Some(Hlvalue::Constant(c_const)) = &link_borrow.args[1] else {
            panic!(
                "True-branch link arg[1] must be a Constant, got {:?}",
                link_borrow.args[1]
            );
        };
        assert!(
            matches!(c_const.value, ConstValue::Int(29872897)),
            "expected Signed(29872897), got {:?}",
            c_const.value
        );
        drop(link_borrow);

        // Both branches converge on block_set_hash.
        let true_target = true_link.borrow().target.as_ref().unwrap().clone();
        let false_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("False exit");
        let false_target = false_link.borrow().target.as_ref().unwrap().clone();
        assert!(
            std::rc::Rc::ptr_eq(&true_target, &false_target),
            "both branches must converge on the same block_set_hash"
        );
        drop(startblock);

        // block_set_hash: ops = setfield only.
        let set_hash_borrow = true_target.borrow();
        let set_hash_ops: Vec<&str> = set_hash_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(set_hash_ops, vec!["setfield"]);
    }

    /// `ll_stritem_nonneg` synthesised against `Ptr(STR)` produces a
    /// single-block CFG: getsubstruct + getarrayitem; link to
    /// returnblock with the char. Returns `Char` for STR, `UniChar`
    /// for UNICODE.
    #[test]
    fn build_ll_stritem_nonneg_synthesizes_single_block_cfg_for_str() {
        let helper = build_ll_stritem_nonneg_helper_graph("ll_stritem_nonneg", STRPTR.clone())
            .expect("build_ll_stritem_nonneg_helper_graph for STR");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "getarrayitem"]);
        // Single unconditional exit to returnblock.
        assert_eq!(startblock.exits.len(), 1);
    }

    /// UNICODE variant returns `UniChar` (chars Array element type).
    #[test]
    fn build_ll_stritem_nonneg_returns_unichar_for_unicode() {
        let helper =
            build_ll_stritem_nonneg_helper_graph("ll_unicode_stritem_nonneg", UNICODEPTR.clone())
                .expect("build_ll_stritem_nonneg_helper_graph UNICODE");
        let inner = helper.graph.borrow();
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::UniChar),
        );
    }

    /// `ll_stritem` synthesised against `Ptr(STR)` produces a
    /// 3-block-plus-returnblock CFG: start (int_lt(i, 0)) branches on
    /// `is_neg`. True → block_neg_fix (getsubstruct + getarraysize +
    /// int_add(i, length)). False → block_dispatch (direct_call
    /// ll_stritem_nonneg). block_neg_fix unconditionally links to
    /// block_dispatch with i_fix.
    #[test]
    fn build_ll_stritem_synthesizes_neg_fix_then_direct_call_for_str() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_stritem_helper_graph(
            &rtyper,
            "ll_stritem",
            STRPTR.clone(),
            "ll_stritem_nonneg",
        )
        .expect("build_ll_stritem_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["int_lt"]);
        drop(startblock);

        let neg_fix = pick_exit_target(&inner.startblock, true);
        {
            let nf_borrow = neg_fix.borrow();
            let nf_ops: Vec<&str> = nf_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(nf_ops, vec!["getsubstruct", "getarraysize", "int_add"]);
        }

        // Both start branches converge on block_dispatch.
        let dispatch_via_false = pick_exit_target(&inner.startblock, false);
        let dispatch_via_neg = {
            // neg_fix has a single unconditional exit.
            let nf_borrow = neg_fix.borrow();
            let link = nf_borrow.exits.first().expect("neg_fix exit").clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        assert!(std::rc::Rc::ptr_eq(&dispatch_via_false, &dispatch_via_neg));

        // dispatch: direct_call ll_stritem_nonneg.
        let dispatch_borrow = dispatch_via_false.borrow();
        let dispatch_ops: Vec<&str> = dispatch_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(dispatch_ops, vec!["direct_call"]);
        let Hlvalue::Constant(c_func) = &dispatch_borrow.operations[0].args[0] else {
            panic!("direct_call arg[0] must be Constant funcptr");
        };
        assert!(format!("{:?}", c_func.value).contains("LLPtr"));
    }

    /// Helper-identity covenant: name + (s, i) input slots; returns
    /// Char for STR.
    #[test]
    fn build_ll_stritem_nonneg_carries_name_inputs_and_char_return_for_str() {
        let helper =
            build_ll_stritem_nonneg_helper_graph("ll_stritem_nonneg", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_stritem_nonneg");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 2);
        let Hlvalue::Variable(v_s) = &startblock.inputargs[0] else {
            panic!("input 0 must be Variable");
        };
        assert!(v_s.name().starts_with("s"));
        let Hlvalue::Variable(v_i) = &startblock.inputargs[1] else {
            panic!("input 1 must be Variable");
        };
        assert!(v_i.name().starts_with("i"));
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Char));
    }

    /// `ll_stritem_nonneg_checked` synthesises a 2-block-plus-return-
    /// plus-exceptblock CFG: start emits getsubstruct + getarraysize +
    /// int_ge(i, length), with the True branch closing a Link to
    /// graph.exceptblock (raise IndexError) and the False branch
    /// forwarding to block_dispatch which direct_calls
    /// ll_stritem_nonneg.
    #[test]
    fn build_ll_stritem_nonneg_checked_synthesizes_oob_branch_and_dispatch_for_str() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_stritem_nonneg_checked_helper_graph(
            &rtyper,
            "ll_stritem_nonneg_checked",
            STRPTR.clone(),
            "ll_stritem_nonneg",
        )
        .expect("build_ll_stritem_nonneg_checked_helper_graph for STR");
        let inner = helper.graph.borrow();

        // start: getsubstruct + getarraysize + int_ge(i, length).
        {
            let sb = inner.startblock.borrow();
            let ops: Vec<&str> = sb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(
                ops,
                vec!["getsubstruct", "getarraysize", "int_ge"],
                "start emits chars + length + bound check"
            );
        }

        // start.exits[True] should target graph.exceptblock (raise).
        // start.exits[False] should target block_dispatch (direct_call).
        let start_borrow = inner.startblock.borrow();
        let mut saw_raise = false;
        let mut dispatch_block: Option<crate::flowspace::model::BlockRef> = None;
        for link in &start_borrow.exits {
            let lb = link.borrow();
            let target = lb.target.as_ref().expect("link.target");
            let exitcase_is_true = matches!(
                lb.exitcase,
                Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
            );
            if std::rc::Rc::ptr_eq(target, &inner.exceptblock) {
                assert!(
                    exitcase_is_true,
                    "raise edge must be the True branch of int_ge(oob)"
                );
                saw_raise = true;
            } else if !exitcase_is_true {
                dispatch_block = Some(target.clone());
            }
        }
        drop(start_borrow);
        assert!(saw_raise, "missing raise edge to exceptblock");
        let dispatch = dispatch_block.expect("missing False edge to block_dispatch");

        // block_dispatch: direct_call(ll_stritem_nonneg, s, i).
        let db = dispatch.borrow();
        let ops: Vec<&str> = db.operations.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(ops, vec!["direct_call"]);
        let Hlvalue::Constant(c_func) = &db.operations[0].args[0] else {
            panic!("direct_call arg[0] must be Constant funcptr");
        };
        assert!(format!("{:?}", c_func.value).contains("LLPtr"));
    }

    /// `ll_stritem_checked` synthesises a 5-block-plus-return-plus-
    /// exceptblock CFG: start branches on int_lt(i, 0) into block_neg_fix
    /// (which int_adds length and forwards to block_check_high) or
    /// directly into block_check_high. block_check_high (int_ge) raises
    /// or forwards to block_check_low. block_check_low (int_lt) raises
    /// or forwards to block_dispatch (direct_call). Two raise edges
    /// reach graph.exceptblock — mirroring the upstream
    /// `if i >= length or i < 0` short-circuit.
    #[test]
    fn build_ll_stritem_checked_synthesizes_neg_fix_two_bound_checks_for_str() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_stritem_checked_helper_graph(
            &rtyper,
            "ll_stritem_checked",
            STRPTR.clone(),
            "ll_stritem_nonneg",
        )
        .expect("build_ll_stritem_checked_helper_graph for STR");
        let inner = helper.graph.borrow();

        // start: getsubstruct + getarraysize + int_lt(i, 0).
        {
            let sb = inner.startblock.borrow();
            let ops: Vec<&str> = sb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(ops, vec!["getsubstruct", "getarraysize", "int_lt"]);
        }

        // start has 2 exits: True → block_neg_fix; False → block_check_high.
        let (block_neg_fix, block_check_high_via_false) = {
            let sb = inner.startblock.borrow();
            assert_eq!(sb.exits.len(), 2);
            let mut neg_fix: Option<crate::flowspace::model::BlockRef> = None;
            let mut check_high: Option<crate::flowspace::model::BlockRef> = None;
            for link in &sb.exits {
                let lb = link.borrow();
                let exitcase_is_true = matches!(
                    lb.exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                );
                let target = lb.target.as_ref().unwrap().clone();
                if exitcase_is_true {
                    neg_fix = Some(target);
                } else {
                    check_high = Some(target);
                }
            }
            (neg_fix.unwrap(), check_high.unwrap())
        };

        // block_neg_fix: int_add only; unconditional exit to block_check_high.
        {
            let nb = block_neg_fix.borrow();
            let ops: Vec<&str> = nb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(ops, vec!["int_add"]);
            assert_eq!(nb.exits.len(), 1);
        }

        // Both starts converge on the same block_check_high.
        let block_check_high_via_neg = {
            let nb = block_neg_fix.borrow();
            nb.exits[0].borrow().target.as_ref().unwrap().clone()
        };
        assert!(std::rc::Rc::ptr_eq(
            &block_check_high_via_false,
            &block_check_high_via_neg,
        ));

        // block_check_high: int_ge; True → exceptblock, False → block_check_low.
        let block_check_low = {
            let cb = block_check_high_via_false.borrow();
            let ops: Vec<&str> = cb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(ops, vec!["int_ge"]);
            assert_eq!(cb.exits.len(), 2);
            let mut next: Option<crate::flowspace::model::BlockRef> = None;
            let mut saw_raise = false;
            for link in &cb.exits {
                let lb = link.borrow();
                let exitcase_is_true = matches!(
                    lb.exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                );
                let target = lb.target.as_ref().unwrap().clone();
                if exitcase_is_true {
                    assert!(
                        std::rc::Rc::ptr_eq(&target, &inner.exceptblock),
                        "block_check_high True branch must raise"
                    );
                    saw_raise = true;
                } else {
                    next = Some(target);
                }
            }
            assert!(saw_raise);
            next.unwrap()
        };

        // block_check_low: int_lt(i, 0); True → exceptblock, False → block_dispatch.
        let block_dispatch = {
            let cb = block_check_low.borrow();
            let ops: Vec<&str> = cb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(ops, vec!["int_lt"]);
            assert_eq!(cb.exits.len(), 2);
            let mut next: Option<crate::flowspace::model::BlockRef> = None;
            let mut saw_raise = false;
            for link in &cb.exits {
                let lb = link.borrow();
                let exitcase_is_true = matches!(
                    lb.exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                );
                let target = lb.target.as_ref().unwrap().clone();
                if exitcase_is_true {
                    assert!(
                        std::rc::Rc::ptr_eq(&target, &inner.exceptblock),
                        "block_check_low True branch must raise"
                    );
                    saw_raise = true;
                } else {
                    next = Some(target);
                }
            }
            assert!(saw_raise);
            next.unwrap()
        };

        // block_dispatch: direct_call(ll_stritem_nonneg, s, i).
        {
            let db = block_dispatch.borrow();
            let ops: Vec<&str> = db.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(ops, vec!["direct_call"]);
        }
    }

    /// `ll_strconcat` synthesises a straight-line `start → returnblock`
    /// CFG: start emits getsubstruct + getarraysize × 2 + int_add
    /// (total) + **malloc_varsize** + two **copystrcontent** ops, then
    /// forwards `newstr` into the returnblock.  Mirrors upstream
    /// `ll_strconcat` (rstr.py): malloc + two `copy_contents_from_str`
    /// calls (each rewritten to a single `copystrcontent`), no per-char
    /// loop.
    ///
    /// Pins the exact `copystrcontent` operand wiring (src/dst objects,
    /// srcstart/dststart/length) so a transposed argument cannot slip
    /// through structural shape checks alone.
    #[test]
    fn build_ll_strconcat_synthesizes_malloc_varsize_and_two_copystrcontent_for_str() {
        fn as_var(hlv: &Hlvalue) -> crate::flowspace::model::Variable {
            match hlv {
                Hlvalue::Variable(v) => v.clone(),
                other => panic!("expected Variable operand, got {other:?}"),
            }
        }
        fn assert_same_var(
            hlv: &Hlvalue,
            expected: &crate::flowspace::model::Variable,
            what: &str,
        ) {
            assert!(
                matches!(hlv, Hlvalue::Variable(v) if v == expected),
                "copystrcontent {what} operand mismatch: {hlv:?}"
            );
        }
        fn assert_zero(hlv: &Hlvalue, what: &str) {
            assert!(
                matches!(hlv, Hlvalue::Constant(c) if c.value == ConstValue::Int(0)),
                "copystrcontent {what} must be Int(0): {hlv:?}"
            );
        }

        let helper = build_ll_strconcat_helper_graph("ll_strconcat", STRPTR.clone())
            .expect("build_ll_strconcat_helper_graph for STR");
        let inner = helper.graph.borrow();
        let sb = inner.startblock.borrow();

        let start_ops: Vec<String> = sb.operations.iter().map(|op| op.opname.clone()).collect();
        assert_eq!(
            start_ops,
            vec![
                "getsubstruct",
                "getarraysize",
                "getsubstruct",
                "getarraysize",
                "int_add",
                "malloc_varsize",
                "copystrcontent",
                "copystrcontent",
            ],
            "start emits chars1/len1/chars2/len2/total + malloc_varsize + two copystrcontent"
        );

        // malloc_varsize (op[5]) still takes [Constant(struct lltype),
        // Constant(flags dict), Variable(total)].
        let mv_op = &sb.operations[5];
        assert_eq!(mv_op.opname, "malloc_varsize");
        assert_eq!(mv_op.args.len(), 3);
        assert!(
            matches!(&mv_op.args[0], Hlvalue::Constant(c) if matches!(c.value, ConstValue::LowLevelType(_)))
        );
        assert!(
            matches!(&mv_op.args[1], Hlvalue::Constant(c) if matches!(c.value, ConstValue::Dict(_)))
        );
        assert!(matches!(mv_op.args[2], Hlvalue::Variable(_)));

        // Capture the source/length/result Variables to compare against the
        // copystrcontent operands by identity.
        let s1_var = as_var(&sb.operations[0].args[0]);
        let len1_var = as_var(&sb.operations[1].result);
        let s2_var = as_var(&sb.operations[2].args[0]);
        let len2_var = as_var(&sb.operations[3].result);
        let newstr_var = as_var(&sb.operations[5].result);

        // copy1 = copystrcontent(s1, newstr, srcstart=0, dststart=0, length=len1).
        let copy1 = &sb.operations[6];
        assert_eq!(copy1.opname, "copystrcontent");
        assert_eq!(copy1.args.len(), 5);
        assert_same_var(&copy1.args[0], &s1_var, "copy1 src");
        assert_same_var(&copy1.args[1], &newstr_var, "copy1 dst");
        assert_zero(&copy1.args[2], "copy1 srcstart");
        assert_zero(&copy1.args[3], "copy1 dststart");
        assert_same_var(&copy1.args[4], &len1_var, "copy1 length");

        // copy2 = copystrcontent(s2, newstr, srcstart=0, dststart=len1, length=len2).
        let copy2 = &sb.operations[7];
        assert_eq!(copy2.opname, "copystrcontent");
        assert_eq!(copy2.args.len(), 5);
        assert_same_var(&copy2.args[0], &s2_var, "copy2 src");
        assert_same_var(&copy2.args[1], &newstr_var, "copy2 dst");
        assert_zero(&copy2.args[2], "copy2 srcstart");
        assert_same_var(&copy2.args[3], &len1_var, "copy2 dststart");
        assert_same_var(&copy2.args[4], &len2_var, "copy2 length");

        // start has a single unconditional exit to the returnblock carrying [newstr].
        assert_eq!(sb.exits.len(), 1, "start has a single unconditional exit");
        let exit = sb.exits[0].borrow();
        assert!(exit.exitcase.is_none(), "exit is unconditional");
        assert!(
            std::rc::Rc::ptr_eq(exit.target.as_ref().unwrap(), &inner.returnblock),
            "start forwards into the returnblock"
        );
        assert_eq!(exit.args.len(), 1);
        assert_same_var(exit.args[0].as_ref().unwrap(), &newstr_var, "return newstr");
    }

    /// `ll_unicode_concat` synthesised against `Ptr(UNICODE)` produces
    /// the same straight-line CFG as `ll_strconcat` modulo elem type:
    /// the malloc_varsize struct const differs (otherwise the
    /// helper-cache families would alias) and the bulk copies select
    /// `copyunicodecontent` rather than `copystrcontent`.
    #[test]
    fn build_ll_strconcat_unicode_differs_in_struct_const_and_copy_opname() {
        let str_helper =
            build_ll_strconcat_helper_graph("ll_strconcat", STRPTR.clone()).expect("STR strconcat");
        let uni_helper = build_ll_strconcat_helper_graph("ll_unicode_concat", UNICODEPTR.clone())
            .expect("UNICODE strconcat");
        let str_inner = str_helper.graph.borrow();
        let uni_inner = uni_helper.graph.borrow();

        let (str_struct_const, str_copy_opnames) = {
            let sb = str_inner.startblock.borrow();
            let mv = &sb.operations[5];
            assert_eq!(mv.opname, "malloc_varsize");
            let copies: Vec<String> = sb.operations[6..8]
                .iter()
                .map(|op| op.opname.clone())
                .collect();
            (mv.args[0].clone(), copies)
        };
        let (uni_struct_const, uni_copy_opnames) = {
            let sb = uni_inner.startblock.borrow();
            let mv = &sb.operations[5];
            assert_eq!(mv.opname, "malloc_varsize");
            let copies: Vec<String> = sb.operations[6..8]
                .iter()
                .map(|op| op.opname.clone())
                .collect();
            (mv.args[0].clone(), copies)
        };
        assert!(
            format!("{str_struct_const:?}") != format!("{uni_struct_const:?}"),
            "STR and UNICODE malloc_varsize struct constants must differ"
        );
        assert_eq!(str_copy_opnames, vec!["copystrcontent", "copystrcontent"]);
        assert_eq!(
            uni_copy_opnames,
            vec!["copyunicodecontent", "copyunicodecontent"]
        );
    }

    /// `ll_count_char` synthesised against `Ptr(STR)` produces a
    /// 3-block-plus-returnblock CFG: same end-clamp + loop pattern
    /// as `ll_find_char` plus a count accumulator. body emits both
    /// `int_add(i, 1)` and `int_add(count, 1)`; the True branch
    /// carries `count_inc` and the False branch carries the original
    /// `count` back into loop_cond.
    #[test]
    fn build_ll_count_char_synthesizes_count_accumulator_loop_cfg_for_str() {
        let helper = build_ll_count_char_helper_graph("ll_count_char", STRPTR.clone())
            .expect("build_ll_count_char_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "getarraysize", "int_gt"]);
        drop(startblock);

        let loop_cond = pick_exit_target(&inner.startblock, false);
        {
            let lc_borrow = loop_cond.borrow();
            let cond_ops: Vec<&str> = lc_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(cond_ops, vec!["int_lt"]);
        }

        let loop_body = pick_exit_target(&loop_cond, true);
        {
            let lb_borrow = loop_body.borrow();
            let body_ops: Vec<&str> = lb_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(
                body_ops,
                vec!["getarrayitem", "char_eq", "int_add", "int_add"]
            );
        }

        // body True (match) cycles back to loop_cond.
        let body_true = pick_exit_target(&loop_body, true);
        assert!(std::rc::Rc::ptr_eq(&body_true, &loop_cond));
        // body False (no match) also cycles back.
        let body_false = pick_exit_target(&loop_body, false);
        assert!(std::rc::Rc::ptr_eq(&body_false, &loop_cond));
    }

    /// UNICODE variant uses `unichar_eq`.
    #[test]
    fn build_ll_count_char_uses_unichar_eq_for_unicode() {
        let helper = build_ll_count_char_helper_graph("ll_unicode_count_char", UNICODEPTR.clone())
            .expect("build_ll_count_char_helper_graph UNICODE");
        let inner = helper.graph.borrow();
        let loop_cond = {
            let sb_borrow = inner.startblock.borrow();
            let link = sb_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
                .expect("loop_cond branch")
                .clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        let loop_body = {
            let lc_borrow = loop_cond.borrow();
            let link = lc_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
                .expect("loop_body branch")
                .clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        let lb_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = lb_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert!(body_ops.contains(&"unichar_eq"));
    }

    /// `ll_strhash` synthesised against `Ptr(STR)` produces a
    /// 2-block-plus-returnblock CFG. start (ptr_nonzero) branches on
    /// `nz`: True → block_lookup, False → returnblock(0). block_lookup
    /// emits `getfield + jit_conditional_call_value`.
    #[test]
    fn build_ll_strhash_emits_null_guard_then_conditional_call_value_for_str() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_strhash_helper_graph(
            &rtyper,
            "ll_strhash",
            STRPTR.clone(),
            "_ll_strhash",
            "_hash_string",
        )
        .expect("build_ll_strhash_helper_graph for STR");
        let inner = helper.graph.borrow();

        // start: ptr_nonzero.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["ptr_nonzero"]);

        // False branch -> returnblock with Signed(0).
        let false_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("False exit");
        let link_borrow = false_link.borrow();
        let Some(Hlvalue::Constant(c_zero)) = &link_borrow.args[0] else {
            panic!("False-branch link arg[0] must be Constant");
        };
        assert!(matches!(c_zero.value, ConstValue::Int(0)));
        drop(link_borrow);

        // True branch -> block_lookup with getfield + jit_conditional_call_value.
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit");
        let lookup = true_link.borrow().target.as_ref().unwrap().clone();
        drop(startblock);
        let lookup_borrow = lookup.borrow();
        let lookup_ops: Vec<&str> = lookup_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(lookup_ops, vec!["getfield", "jit_conditional_call_value"]);

        // jit_conditional_call_value has 3 args: cond (Variable h),
        // funcptr Constant, s (Variable).
        let cond_op = &lookup_borrow.operations[1];
        assert_eq!(cond_op.args.len(), 3);
        let Hlvalue::Constant(c_func) = &cond_op.args[1] else {
            panic!("jit_conditional_call_value arg[1] must be funcptr Constant");
        };
        assert!(format!("{:?}", c_func.value).contains("LLPtr"));
    }

    /// `ll_find_char` synthesised against `Ptr(STR)` produces the
    /// expected 3-block-plus-returnblock CFG. start (getsubstruct +
    /// getarraysize + int_gt) → block_loop_cond (int_lt) →
    /// block_loop_body (getarrayitem + char_eq + int_add). Body True
    /// branch returns i; False cycles back to loop_cond.
    #[test]
    fn build_ll_find_char_synthesizes_forward_loop_cfg_for_str() {
        let helper = build_ll_find_char_helper_graph("ll_find_char", STRPTR.clone())
            .expect("build_ll_find_char_helper_graph for STR");
        assert!(
            helper.func._elidable_function_,
            "rlib/jit.py elidable stamps GraphFunc._elidable_function_"
        );
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "getarraysize", "int_gt"]);
        drop(startblock);

        // start False branch: end_clamped == end (original).
        let loop_cond_via_false = pick_exit_target(&inner.startblock, false);
        {
            let lc_borrow = loop_cond_via_false.borrow();
            let cond_ops: Vec<&str> = lc_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(cond_ops, vec!["int_lt"]);
        }

        // start True branch also goes to loop_cond (different bound init).
        let loop_cond_via_true = pick_exit_target(&inner.startblock, true);
        assert!(std::rc::Rc::ptr_eq(
            &loop_cond_via_false,
            &loop_cond_via_true
        ));

        let loop_body = pick_exit_target(&loop_cond_via_false, true);
        {
            let lb_borrow = loop_body.borrow();
            let body_ops: Vec<&str> = lb_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(body_ops, vec!["getarrayitem", "char_eq", "int_add"]);
        }

        // body False (no match) cycles back to loop_cond.
        let body_false = pick_exit_target(&loop_body, false);
        assert!(std::rc::Rc::ptr_eq(&body_false, &loop_cond_via_false));
    }

    /// `ll_rfind_char` against `Ptr(STR)` reverses the loop direction:
    /// loop_cond uses `int_gt(i, start_bound)` instead of `int_lt`,
    /// and loop_body pre-decrements i via `int_sub` before reading
    /// `chars[i_dec]`.
    #[test]
    fn build_ll_rfind_char_synthesizes_reverse_loop_cfg_for_str() {
        let helper = build_ll_rfind_char_helper_graph("ll_rfind_char", STRPTR.clone())
            .expect("build_ll_rfind_char_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let loop_cond = pick_exit_target(&inner.startblock, false);
        {
            let lc_borrow = loop_cond.borrow();
            let cond_ops: Vec<&str> = lc_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(cond_ops, vec!["int_gt"]);
        }

        let loop_body = pick_exit_target(&loop_cond, true);
        {
            let lb_borrow = loop_body.borrow();
            let body_ops: Vec<&str> = lb_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(body_ops, vec!["int_sub", "getarrayitem", "char_eq"]);
        }
    }

    fn opnames(block: &crate::flowspace::model::BlockRef) -> Vec<String> {
        block
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect()
    }

    fn pick_bool_exit(
        block: &crate::flowspace::model::BlockRef,
        want_true: bool,
    ) -> crate::flowspace::model::BlockRef {
        let target_value = ConstValue::Bool(want_true);
        let block_borrow = block.borrow();
        let link = block_borrow
            .exits
            .iter()
            .find(|l| {
                matches!(
                    l.borrow().exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == target_value
                )
            })
            .expect("matching bool exit");
        link.borrow().target.as_ref().unwrap().clone()
    }

    /// `LLHelpers.ll_find` (`rstr.py`): clamp start/end, then `m == 0`
    /// return start, `m == 1` `ll_find_char` + `jit.record_exact_value`,
    /// `m > 1` `ll_search(..., SEARCH_FIND)` + the same record.
    #[test]
    fn build_ll_find_synthesizes_find_char_and_search_arms() {
        let helper = build_ll_find_helper_graph("ll_find", STRPTR.clone())
            .expect("build_ll_find_helper_graph for STR");
        let inner = helper.graph.borrow();
        assert_eq!(opnames(&inner.startblock), vec!["int_lt"]);

        let block_n = pick_bool_exit(&inner.startblock, false);
        assert_eq!(
            opnames(&block_n),
            vec!["getsubstruct", "getarraysize", "int_gt"]
        );

        let block_range = pick_bool_exit(&block_n, false);
        assert_eq!(opnames(&block_range), vec!["int_sub", "int_lt"]);
        {
            let range_borrow = block_range.borrow();
            let true_link = range_borrow
                .exits
                .iter()
                .find(|l| {
                    matches!(
                        l.borrow().exitcase,
                        Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                    )
                })
                .expect("span_neg true");
            let args = &true_link.borrow().args;
            assert!(
                matches!(args.first(), Some(Some(Hlvalue::Constant(c))) if c.value == ConstValue::Int(-1)),
                "empty range returns -1: {args:?}"
            );
        }

        let block_m = pick_bool_exit(&block_range, false);
        assert_eq!(
            opnames(&block_m),
            vec!["getsubstruct", "getarraysize", "int_le"]
        );

        let block_m_le_1 = pick_bool_exit(&block_m, true);
        assert_eq!(opnames(&block_m_le_1), vec!["int_eq"]);
        {
            let le1_borrow = block_m_le_1.borrow();
            let true_link = le1_borrow
                .exits
                .iter()
                .find(|l| {
                    matches!(
                        l.borrow().exitcase,
                        Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                    )
                })
                .expect("m == 0");
            assert!(
                matches!(
                    true_link.borrow().args.first(),
                    Some(Some(Hlvalue::Variable(_)))
                ),
                "m == 0 returns start"
            );
        }

        let block_find_char = pick_bool_exit(&block_m_le_1, false);
        assert_eq!(
            opnames(&block_find_char),
            vec![
                "getsubstruct",
                "getarrayitem",
                "direct_call",
                "int_lt",
                "jit_record_exact_value"
            ]
        );
        {
            let borrowed = block_find_char.borrow();
            let call = borrowed
                .operations
                .iter()
                .find(|op| op.opname == "direct_call")
                .expect("ll_find_char call");
            let func = funcobj_from_funcptr_constant(&call.args[0]).expect("ll_find_char funcptr");
            assert_eq!(func._name, "ll_find_char");
            assert!(
                func.graph.is_some(),
                "HighLevelOp.gendirectcall funcptr carries graph"
            );
        }

        let block_search = pick_bool_exit(&block_m, false);
        assert_eq!(
            opnames(&block_search),
            vec!["direct_call", "int_lt", "jit_record_exact_value"]
        );
        {
            let borrowed = block_search.borrow();
            let call = borrowed
                .operations
                .iter()
                .find(|op| op.opname == "direct_call")
                .expect("ll_search call");
            let func = funcobj_from_funcptr_constant(&call.args[0]).expect("ll_search funcptr");
            assert_eq!(func._name, "ll_search");
            assert!(
                func.graph.is_some(),
                "HighLevelOp.gendirectcall funcptr carries graph"
            );
            assert!(
                matches!(
                    call.args.get(5),
                    Some(Hlvalue::Constant(c)) if c.value == ConstValue::Int(1)
                ),
                "SEARCH_FIND is 1: {:?}",
                call.args.get(5)
            );
        }

        drop(inner);
        let uni = build_ll_find_helper_graph("ll_unicode_find", UNICODEPTR.clone())
            .expect("UNICODE ll_find");
        let names: Vec<String> = direct_call_funcobjs(&uni.graph.borrow())
            .into_iter()
            .map(|func| func._name)
            .collect();
        assert!(
            names.iter().any(|n| n == "ll_search"),
            "UNICODE ll_find calls ll_search: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "ll_find_char"),
            "UNICODE ll_find calls ll_find_char: {names:?}"
        );
    }

    fn search_call_block(graph: &FunctionGraph) -> crate::flowspace::model::BlockRef {
        let check_other = pick_bool_exit(&graph.startblock, true);
        pick_bool_exit(&check_other, true)
    }

    fn host_qualname(hlv: &Hlvalue) -> Option<&str> {
        match hlv {
            Hlvalue::Constant(Constant {
                value: ConstValue::HostObject(obj),
                ..
            }) => Some(obj.qualname()),
            _ => None,
        }
    }

    /// `LLHelpers.ll_search` (`rstr.py`) STR arm: `hlstr` is identity,
    /// `_search` picks `_search_elidable` at translation time. The
    /// `direct_call` sits after the two `assert … is not None` branches.
    #[test]
    fn build_ll_search_calls_search_elidable_for_str() {
        let helper = build_ll_search_helper_graph("ll_search", STRPTR.clone())
            .expect("build_ll_search_helper_graph for STR");
        let inner = helper.graph.borrow();
        let call = search_call_block(&inner);
        assert_eq!(opnames(&call), vec!["direct_call"]);
        let call_borrow = call.borrow();
        let func = funcobj_from_funcptr_constant(&call_borrow.operations[0].args[0])
            .expect("_search_elidable funcptr");
        assert_eq!(func._name, "_search_elidable");
        assert!(
            func.graph.is_some(),
            "HighLevelOp.gendirectcall funcptr carries graph"
        );
        drop(call_borrow);
        drop(inner);
        let helper = build_ll_search_helper_graph("ll_unicode_search", UNICODEPTR.clone())
            .expect("build_ll_search_helper_graph for UNICODE");
        let inner = helper.graph.borrow();
        let call = search_call_block(&inner);
        assert_eq!(opnames(&call), vec!["direct_call"]);
        let call_borrow = call.borrow();
        let func = funcobj_from_funcptr_constant(&call_borrow.operations[0].args[0])
            .expect("_search_normal funcptr");
        assert_eq!(func._name, "_search_normal");
        assert!(
            func.graph.is_some(),
            "HighLevelOp.gendirectcall funcptr carries graph"
        );
    }

    /// `LLHelpers.ll_search`: `assert value is not None` /
    /// `assert other is not None` lower as two `ptr_nonzero` branches
    /// whose False arms raise AssertionError.
    #[test]
    fn build_ll_search_asserts_converted_strings_not_none() {
        let helper = build_ll_search_helper_graph("ll_search", STRPTR.clone())
            .expect("build_ll_search_helper_graph for STR");
        let inner = helper.graph.borrow();
        let check_value = inner.startblock.clone();
        let check_other = pick_bool_exit(&check_value, true);
        assert_eq!(opnames(&check_value), vec!["ptr_nonzero"]);
        assert_eq!(opnames(&check_other), vec!["ptr_nonzero"]);
        let mut assertionerror_exits = 0;
        for block in [&check_value, &check_other] {
            let borrowed = block.borrow();
            let false_link = borrowed
                .exits
                .iter()
                .find(|link| {
                    matches!(
                        link.borrow().exitcase,
                        Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)
                    )
                })
                .expect("null arm of assert");
            let link = false_link.borrow();
            let target = link.target.as_ref().expect("exceptblock target");
            assert!(
                std::rc::Rc::ptr_eq(target, &inner.exceptblock),
                "null arm raises via exceptblock"
            );
            let class = link
                .args
                .first()
                .and_then(|arg| arg.as_ref())
                .expect("exception class");
            assert_eq!(host_qualname(class), Some("AssertionError"));
            assertionerror_exits += 1;
        }
        assert_eq!(assertionerror_exits, 2);
    }

    /// `lowlevel_helper_function_with_builder` on one rtyper leaves
    /// every `ll_find` family callee (`ll_search`, `ll_find_char`,
    /// `_search_elidable`, `_search_normal`) on that same rtyper's
    /// translator (`RPythonTyper.annotate_helper`).
    #[test]
    fn ll_find_family_callees_live_on_same_rtyper_translator() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = rtyper
            .lowlevel_helper_function_with_builder(
                "ll_find".to_string(),
                vec![
                    STRPTR.clone(),
                    STRPTR.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                LowLevelType::Signed,
                |rtyper, _args, _result| {
                    build_ll_find_helper_graph_on(rtyper, "ll_find", STRPTR.clone())
                },
            )
            .expect("ll_find helper");
        let py = helper.graph.expect("builder registered the outer graph");
        let mut seen = std::collections::HashSet::new();
        let mut names = Vec::new();
        fn walk(
            rtyper: &RPythonTyper,
            graph: &FunctionGraph,
            seen: &mut std::collections::HashSet<usize>,
            names: &mut Vec<String>,
        ) {
            for func in direct_call_funcobjs(graph) {
                let graph_id = func
                    .graph
                    .expect("HighLevelOp.gendirectcall funcptr carries graph");
                names.push(func._name.clone());
                let callee = graph_ref_for_funcptr_id(rtyper, graph_id).unwrap_or_else(|| {
                    panic!(
                        "{} graph id {graph_id} missing from translator.graphs",
                        func._name
                    )
                });
                if seen.insert(graph_id) {
                    walk(rtyper, &callee.borrow(), seen, names);
                }
            }
        }
        walk(&rtyper, &py.graph.borrow(), &mut seen, &mut names);
        for expected in [
            "ll_search",
            "ll_find_char",
            "_search_elidable",
            "_search_normal",
        ] {
            assert!(
                names.iter().any(|name| name == expected),
                "missing {expected} in {names:?}"
            );
        }
    }

    /// UNICODE variants substitute `unichar_eq` for the comparison op
    /// in both find/rfind body blocks.
    #[test]
    fn build_ll_find_rfind_char_use_unichar_eq_for_unicode() {
        for (name, builder) in [
            (
                "ll_unicode_find_char",
                build_ll_find_char_helper_graph
                    as fn(&str, LowLevelType) -> Result<PyGraph, TyperError>,
            ),
            (
                "ll_unicode_rfind_char",
                build_ll_rfind_char_helper_graph
                    as fn(&str, LowLevelType) -> Result<PyGraph, TyperError>,
            ),
        ] {
            let helper = builder(name, UNICODEPTR.clone()).expect(name);
            let inner = helper.graph.borrow();
            let loop_cond = {
                let sb_borrow = inner.startblock.borrow();
                let link = sb_borrow
                    .exits
                    .iter()
                    .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
                    .expect("loop_cond branch")
                    .clone();

                link.borrow().target.as_ref().unwrap().clone()
            };
            let loop_body = {
                let lc_borrow = loop_cond.borrow();
                let link = lc_borrow
                    .exits
                    .iter()
                    .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
                    .expect("loop_body branch")
                    .clone();

                link.borrow().target.as_ref().unwrap().clone()
            };
            let lb_borrow = loop_body.borrow();
            let body_ops: Vec<&str> = lb_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert!(
                body_ops.contains(&"unichar_eq"),
                "{name}: expected 'unichar_eq', got {body_ops:?}"
            );
        }
    }

    /// UNICODE variants route the inner `_hash_string` sub-helper
    /// against the Unicode chars Array. The inner helper's body uses
    /// `cast_unichar_to_int` (verified indirectly via the inner
    /// helper's helper-cache key — distinct lltype implies distinct
    /// graph).
    #[test]
    fn build_ll_hash_string_routes_through_unicode_inner_helper() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_hash_string_helper_graph(
            &rtyper,
            "ll_unicode_hash_string",
            UNICODEPTR.clone(),
            "_hash_unicode_string",
        )
        .expect("build_ll_hash_string_helper_graph for UNICODE");
        let inner = helper.graph.borrow();

        // direct_call[0] funcptr Constant — debug-format must mention LLPtr.
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.operations[1].opname, "direct_call");
        let Hlvalue::Constant(c_func) = &startblock.operations[1].args[0] else {
            panic!("direct_call arg[0] must be Constant funcptr");
        };
        assert!(format!("{:?}", c_func.value).contains("LLPtr"));
    }

    fn build_identity_char_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
        let ch = variable_with_lltype("ch", LowLevelType::Char);
        let startblock = Block::shared(vec![Hlvalue::Variable(ch.clone())]);
        let return_var = variable_with_lltype("result", LowLevelType::Char);
        let mut graph = FunctionGraph::with_return_var(
            name.to_string(),
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(ch)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let func = GraphFunc::new(
            name.to_string(),
            Constant::new(ConstValue::Dict(Default::default())),
        );
        graph.func = Some(func.clone());
        Ok(helper_pygraph_from_graph(
            graph,
            vec!["ch".to_string()],
            func,
        ))
    }

    /// `ll_upper/ll_lower` graph shape: extract chars/length, keep the
    /// explicit empty branch, allocate result with malloc_varsize, loop
    /// over source chars, call the per-char helper, and store into the
    /// result chars array.
    #[test]
    fn build_ll_string_casefold_synthesizes_malloc_loop_and_char_helper_call() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_string_casefold_helper_graph(
            &rtyper,
            "ll_upper",
            STRPTR.clone(),
            "ll_upper_char",
            build_identity_char_helper_graph,
        )
        .expect("build ll_upper");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "getarraysize", "int_eq"]);

        let alloc = {
            let link = startblock
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
                .expect("non-empty allocation branch")
                .clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        drop(startblock);

        let alloc_borrow = alloc.borrow();
        let alloc_ops: Vec<&str> = alloc_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(alloc_ops, vec!["malloc_varsize", "getsubstruct"]);
        let loop_cond = alloc_borrow.exits[0]
            .borrow()
            .target
            .as_ref()
            .unwrap()
            .clone();
        drop(alloc_borrow);

        let cond_borrow = loop_cond.borrow();
        let body = {
            let link = cond_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
                .expect("loop body branch")
                .clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        drop(cond_borrow);

        let body_borrow = body.borrow();
        let body_ops: Vec<&str> = body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec!["getarrayitem", "direct_call", "setarrayitem", "int_add"]
        );
        let Hlvalue::Constant(c_func) = &body_borrow.operations[1].args[0] else {
            panic!("direct_call arg[0] must be Constant funcptr");
        };
        assert!(format!("{:?}", c_func.value).contains("LLPtr"));
    }

    /// Unicode must not accidentally share the byte-string casefold
    /// helper. PyPy's `ll_upper/lower` use `mallocstr` specifically to
    /// explode for unicode; the Rust builder rejects UniChar arrays.
    #[test]
    fn build_ll_string_casefold_rejects_unicode_ptr() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let err = build_ll_string_casefold_helper_graph(
            &rtyper,
            "ll_unicode_upper",
            UNICODEPTR.clone(),
            "ll_upper_char",
            build_identity_char_helper_graph,
        )
        .expect_err("unicode casefold helper must be rejected");
        assert!(format!("{err:?}").contains("only supports Char arrays"));
    }

    /// `ll_replace_chr_chr` emits the upstream malloc + replacement
    /// loop shape, with an equality branch before storing to the new
    /// string.
    #[test]
    fn build_ll_replace_chr_chr_synthesizes_malloc_replace_loop_for_str() {
        let helper = build_ll_replace_chr_chr_helper_graph(
            "ll_replace_chr_chr",
            STRPTR.clone(),
            LowLevelType::Char,
        )
        .expect("build ll_replace_chr_chr");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            start_ops,
            vec![
                "getsubstruct",
                "getarraysize",
                "malloc_varsize",
                "getsubstruct"
            ]
        );
        let loop_cond = startblock.exits[0]
            .borrow()
            .target
            .as_ref()
            .unwrap()
            .clone();
        drop(startblock);

        let cond_borrow = loop_cond.borrow();
        assert_eq!(cond_borrow.operations[0].opname, "int_lt");
        let loop_body = {
            let link = cond_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
                .expect("loop body branch")
                .clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        drop(cond_borrow);

        let body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(body_ops, vec!["getarrayitem", "char_eq"]);
        let store = body_borrow.exits[0]
            .borrow()
            .target
            .as_ref()
            .unwrap()
            .clone();
        drop(body_borrow);

        let store_borrow = store.borrow();
        let store_ops: Vec<&str> = store_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(store_ops, vec!["setarrayitem", "int_add"]);
    }

    /// Unicode uses the same LLHelpers body but must compare UniChar
    /// elements with `unichar_eq`.
    #[test]
    fn build_ll_replace_chr_chr_uses_unichar_eq_for_unicode() {
        let helper = build_ll_replace_chr_chr_helper_graph(
            "ll_unicode_replace_chr_chr",
            UNICODEPTR.clone(),
            LowLevelType::UniChar,
        )
        .expect("build unicode replace");
        let inner = helper.graph.borrow();
        let loop_cond = inner.startblock.borrow().exits[0]
            .borrow()
            .target
            .as_ref()
            .unwrap()
            .clone();
        let loop_body = {
            let cond_borrow = loop_cond.borrow();
            let link = cond_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
                .expect("loop body branch")
                .clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        let body_borrow = loop_body.borrow();
        assert_eq!(body_borrow.operations[1].opname, "unichar_eq");
    }

    #[test]
    fn build_ll_int2hex_chains_uint_ops_through_hex_chars_table() {
        let helper = build_ll_int2hex_helper_graph("ll_int2hex", true).expect("build ll_int2hex");
        let graph = helper.graph.borrow();

        // BFS over reachable blocks, collecting every opname and the
        // prebuilt hex_chars constant referenced by the digit lookup.
        let mut seen: Vec<*const std::cell::RefCell<crate::flowspace::model::Block>> = Vec::new();
        let mut queue = vec![graph.startblock.clone()];
        let mut opnames: Vec<String> = Vec::new();
        let mut hex_chars: Option<ConstValue> = None;
        while let Some(block) = queue.pop() {
            let key = std::rc::Rc::as_ptr(&block);
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            let b = block.borrow();
            for op in &b.operations {
                opnames.push(op.opname.clone());
                if op.opname == "getarrayitem"
                    && let Hlvalue::Constant(c) = &op.args[0]
                {
                    hex_chars = Some(c.value.clone());
                }
            }
            for link in &b.exits {
                if let Some(target) = link.borrow().target.as_ref() {
                    queue.push(target.clone());
                }
            }
        }

        for expected in [
            "malloc_varsize",
            "int_neg",
            "cast_primitive",
            "uint_is_true",
            "uint_and",
            "uint_rshift",
            "cast_int_to_char",
            "setfield",
            "getsubstruct",
        ] {
            assert!(
                opnames.iter().any(|n| n == expected),
                "ll_int2hex graph missing op {expected}; saw {opnames:?}"
            );
        }

        // hex_chars is an immortal Array(Char) holding "0123456789abcdef".
        let Some(ConstValue::LLPtr(ptr)) = hex_chars else {
            panic!("digit lookup must read from an LLPtr hex_chars table");
        };
        let Ok(Some(_ptr_obj::Array(arr))) = ptr._obj0_value() else {
            panic!("hex_chars LLPtr must target an Array container");
        };
        let items = arr.items.lock();
        assert_eq!(items.len(), 16);
        assert_eq!(items[0], LowLevelValue::Char('0'));
        assert_eq!(items[10], LowLevelValue::Char('a'));
        assert_eq!(items[15], LowLevelValue::Char('f'));
    }

    #[test]
    fn top_level_parity_helpers_expose_original_names() {
        let err = new_malloc().expect_err("new_malloc factory is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("new_malloc"));

        let err = ll_join().expect_err("ll_join is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_join"));

        let err = ll_striter().expect_err("iterator helper is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_striter"));
    }

    #[test]
    fn conststr_and_emptyunicodefun_route_to_existing_allocators() {
        let s1 = conststr("abc").expect("conststr");
        let s2 = conststr("abc").expect("conststr cached");
        assert_eq!(s1, s2);

        let u = emptyunicodefun().expect("emptyunicodefun");
        let Some((bytes, hash)) = prebuilt_str_bytes_and_hash(&s1) else {
            panic!("conststr must produce a prebuilt STR ptr");
        };
        assert_eq!(bytes, b"abc");
        assert_eq!(hash, ll_strhash_value(b"abc"));
        assert!(u.nonzero());
    }

    #[test]
    fn temp_unicode_matches_gcarray_of_unicode_ptr_shape() {
        let LowLevelType::Array(arr) = TEMP_UNICODE.clone() else {
            panic!("TEMP_UNICODE must be an array type");
        };
        assert_eq!(
            arr._gckind,
            crate::translator::rtyper::lltypesystem::lltype::GcKind::Gc
        );
        assert_eq!(arr.OF, UNICODEPTR.clone());
    }

    #[test]
    fn utf8stringiterator_repr_is_gcstruct_utf8iter() {
        let r = Utf8StringIteratorRepr::new(STRPTR.clone()).expect("Utf8StringIteratorRepr::new");
        let LowLevelType::Ptr(ptr) = r.lowleveltype() else {
            panic!("utf8iter lltype must be Ptr");
        };
        let PtrTarget::Struct(st) = &ptr.TO else {
            panic!("utf8iter Ptr.TO must be Struct");
        };
        assert_eq!(st._name, "utf8iter");
        assert_eq!(st._gckind, GcKind::Gc);
        assert!(matches!(st._flds.get("_utf8"), Some(LowLevelType::Ptr(_))));
        assert!(matches!(st._flds.get("_end"), Some(LowLevelType::Signed)));
        assert!(matches!(st._flds.get("_pos"), Some(LowLevelType::Signed)));
    }

    #[test]
    fn build_ll_utf8iter_helper_mallocs_and_sets_fields() {
        let r = Utf8StringIteratorRepr::new(STRPTR.clone()).expect("Utf8StringIteratorRepr::new");
        let helper =
            build_ll_utf8iter_helper_graph("ll_utf8iter", STRPTR.clone(), r.lowleveltype().clone())
                .expect("build_ll_utf8iter_helper_graph");
        crate::flowspace::model::checkgraph(&helper.graph.borrow());
        let inner = helper.graph.borrow();
        let start_ops: Vec<_> = inner
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(
            start_ops,
            vec![
                "malloc".to_string(),
                "setfield".to_string(),
                "getsubstruct".to_string(),
                "getarraysize".to_string(),
                "setfield".to_string(),
                "setfield".to_string(),
            ]
        );
    }

    #[test]
    fn build_ll_utf8next_helper_checks_graph_and_stopiteration() {
        let r = Utf8StringIteratorRepr::new(STRPTR.clone()).expect("Utf8StringIteratorRepr::new");
        let helper =
            build_ll_utf8next_helper_graph("ll_utf8next", r.lowleveltype().clone(), STRPTR.clone())
                .expect("build_ll_utf8next_helper_graph");
        crate::flowspace::model::checkgraph(&helper.graph.borrow());
        let inner = helper.graph.borrow();
        let except_key = crate::flowspace::model::BlockKey::of(&inner.exceptblock);
        {
            let start = inner.startblock.borrow();
            let start_ops: Vec<_> = start
                .operations
                .iter()
                .map(|op| op.opname.clone())
                .collect();
            assert_eq!(
                start_ops,
                vec![
                    "getfield".to_string(),
                    "getfield".to_string(),
                    "getfield".to_string(),
                    "int_eq".to_string()
                ]
            );
            assert!(start.exitswitch.is_some(), "StopIteration guard exitswitch");
            let raises = start.exits.iter().any(|lnk| {
                lnk.borrow()
                    .target
                    .as_ref()
                    .is_some_and(|t| crate::flowspace::model::BlockKey::of(t) == except_key)
            });
            assert!(
                raises,
                "one startblock exit must link to exceptblock (raise StopIteration)"
            );
        }
        let mut opnames = Vec::new();
        for block in inner.iterblocks() {
            for op in &block.borrow().operations {
                opnames.push(op.opname.clone());
            }
        }
        for expected in [
            "getarrayitem",
            "cast_char_to_int",
            "int_le",
            "int_lshift",
            "int_add",
            "int_sub",
            "setfield",
        ] {
            assert!(
                opnames.iter().any(|n| n == expected),
                "ll_utf8next graph missing op {expected}; saw {opnames:?}"
            );
        }
    }

    #[test]
    fn build_ll_utf8next_helper_marks_always_inline() {
        let r = Utf8StringIteratorRepr::new(STRPTR.clone()).expect("Utf8StringIteratorRepr::new");
        let helper =
            build_ll_utf8next_helper_graph("ll_utf8next", r.lowleveltype().clone(), STRPTR.clone())
                .expect("build_ll_utf8next_helper_graph");
        assert_eq!(
            helper.func._always_inline_,
            crate::flowspace::model::AlwaysInline::True,
        );
        assert_eq!(
            helper
                .graph
                .borrow()
                .func
                .as_ref()
                .map(|f| f._always_inline_),
            Some(crate::flowspace::model::AlwaysInline::True),
        );
    }

    /// Annotation intern → one `assign_inheritance_ids` → exceptiondata,
    /// matching `RPythonAnnotator.simplify` / `perform_normalizations`
    /// then `RPythonTyper.specialize` / `exceptiondata.finish`.
    fn rtyper_after_unichr_annotation() -> (std::rc::Rc<RPythonAnnotator>, std::rc::Rc<RPythonTyper>)
    {
        use crate::translator::rtyper::normalizecalls::assign_inheritance_ids;
        let ann = RPythonAnnotator::new(None, None, None, false);
        crate::rlib::rutf8::out_of_range_classdef(&ann.bookkeeper)
            .expect("annotation intern OutOfRange");
        crate::annotator::exception::standard_exception_classdefs(&ann.bookkeeper)
            .expect("annotation intern standard exceptions");
        assign_inheritance_ids(&ann);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        rtyper.finish_exceptiondata().expect("finish_exceptiondata");
        (ann, rtyper)
    }

    fn classdef_vtable(
        rtyper: &std::rc::Rc<RPythonTyper>,
        classdef: &std::rc::Rc<std::cell::RefCell<crate::annotator::classdesc::ClassDef>>,
    ) -> _ptr {
        use crate::translator::rtyper::rclass::{CLASSTYPE, getclassrepr_arc};
        let rclass = getclassrepr_arc(rtyper, Some(classdef)).expect("getclassrepr_arc");
        Repr::setup(&*rclass.as_repr()).expect("ClassRepr.setup");
        rclass.getruntime(&CLASSTYPE).expect("getruntime")
    }

    #[test]
    fn build_ll_unichr_as_utf8_helper_encodes_and_range_checks() {
        use crate::translator::rtyper::rclass::{CLASSTYPE, ll_issubclass};

        let (ann, rtyper) = rtyper_after_unichr_annotation();
        let helper = build_ll_unichr_as_utf8_helper_graph(&rtyper, "ll_unichr_as_utf8")
            .expect("build_ll_unichr_as_utf8_helper_graph");
        crate::flowspace::model::checkgraph(&helper.graph.borrow());
        let inner = helper.graph.borrow();
        let except_key = crate::flowspace::model::BlockKey::of(&inner.exceptblock);
        {
            let start = inner.startblock.borrow();
            let start_ops: Vec<_> = start
                .operations
                .iter()
                .map(|op| op.opname.clone())
                .collect();
            assert_eq!(
                start_ops,
                vec![
                    "int_ge".to_string(),
                    "debug_assert".to_string(),
                    "cast_int_to_uint".to_string(),
                    "uint_le".to_string(),
                ]
            );
            let start_raises = start.exits.iter().any(|lnk| {
                lnk.borrow()
                    .target
                    .as_ref()
                    .is_some_and(|t| crate::flowspace::model::BlockKey::of(t) == except_key)
            });
            assert!(
                !start_raises,
                "startblock holds assert + r_uint; the last uint_le raises"
            );
        }
        let mut too_large_false: Option<crate::flowspace::model::BlockRef> = None;
        for block in inner.iterblocks() {
            let b = block.borrow();
            let le_max = b.operations.iter().any(|op| {
                op.opname == "uint_le"
                    && matches!(
                        op.args.get(1),
                        Some(Hlvalue::Constant(c)) if c.value == ConstValue::UInt(0x10FFFF)
                    )
            });
            if !le_max {
                continue;
            }
            for lnk in &b.exits {
                let link = lnk.borrow();
                let is_false = matches!(
                    &link.exitcase,
                    Some(Hlvalue::Constant(c)) if c.value == ConstValue::Bool(false)
                );
                if is_false {
                    too_large_false = link.target.clone();
                }
            }
        }
        let raise_block = too_large_false.expect("false arm of code <= r_uint(0x10FFFF)");
        assert_ne!(
            crate::flowspace::model::BlockKey::of(&raise_block),
            except_key,
            "code > 0x10FFFF must malloc OutOfRange before the exceptblock link"
        );
        let oor = crate::rlib::rutf8::out_of_range_class();
        let exception = crate::flowspace::model::HOST_ENV
            .lookup_exception_class("Exception")
            .expect("Exception");
        let value_error = crate::flowspace::model::HOST_ENV
            .lookup_exception_class("ValueError")
            .expect("ValueError");
        assert!(oor.is_subclass_of(&exception));
        assert!(!oor.is_subclass_of(&value_error));
        let oor_cd = crate::rlib::rutf8::out_of_range_classdef(&ann.bookkeeper)
            .expect("OutOfRange classdef");
        let exception_cd = ann
            .bookkeeper
            .getuniqueclassdef(&exception)
            .expect("Exception classdef");
        let value_error_cd = ann
            .bookkeeper
            .getuniqueclassdef(&value_error)
            .expect("ValueError classdef");
        let oor_vt = classdef_vtable(&rtyper, &oor_cd);
        let exc_vt = classdef_vtable(&rtyper, &exception_cd);
        let ve_vt = classdef_vtable(&rtyper, &value_error_cd);
        assert!(oor_vt.nonzero());
        assert!(ll_issubclass(&oor_vt, &exc_vt).expect("ll_issubclass Exception"));
        assert!(!ll_issubclass(&oor_vt, &ve_vt).expect("ll_issubclass ValueError"));
        {
            let raise = raise_block.borrow();
            assert!(
                raise.operations.iter().any(|op| op.opname == "malloc"),
                "raise block must malloc OutOfRange"
            );
            let input = match raise.inputargs.first() {
                Some(Hlvalue::Variable(v)) => v.clone(),
                other => panic!("raise block inputarg must be the code Variable, got {other:?}"),
            };
            let typeptr_set = raise.operations.iter().find(|op| {
                op.opname == "setfield"
                    && matches!(
                        op.args.get(1),
                        Some(Hlvalue::Constant(c))
                            if matches!(&c.value, ConstValue::ByteStr(s) if s == b"typeptr")
                    )
            });
            let typeptr_set = typeptr_set.expect("raise block setfield typeptr");
            match typeptr_set.args.get(2) {
                Some(Hlvalue::Constant(c)) => match &c.value {
                    ConstValue::LLPtr(p) => {
                        assert!(p.nonzero());
                        assert_eq!(p.as_ref(), &oor_vt);
                        assert_eq!(c.concretetype.as_ref(), Some(&CLASSTYPE.clone()));
                    }
                    other => panic!("typeptr must be OutOfRange vtable, got {other:?}"),
                },
                other => panic!("typeptr value must be a Constant, got {other:?}"),
            }
            let set_code = raise.operations.iter().find(|op| {
                op.opname == "setfield"
                    && matches!(
                        op.args.get(1),
                        Some(Hlvalue::Constant(c))
                            if matches!(&c.value, ConstValue::ByteStr(s) if s == b"inst_code")
                    )
            });
            let set_code = set_code.expect("raise block setfield inst_code");
            match set_code.args.get(2) {
                Some(Hlvalue::Variable(v)) => assert_eq!(
                    v, &input,
                    "inst_code must be set from the raise-block input"
                ),
                other => panic!("setfield inst_code value must be the input code, got {other:?}"),
            }
            assert_eq!(raise.exits.len(), 1);
            let link = raise.exits[0].borrow();
            assert!(
                link.target
                    .as_ref()
                    .is_some_and(|t| crate::flowspace::model::BlockKey::of(t) == except_key)
            );
            assert_eq!(link.args.len(), 2);
            match &link.args[0] {
                Some(Hlvalue::Constant(c)) => match &c.value {
                    ConstValue::LLPtr(p) => {
                        assert!(p.nonzero());
                        assert_eq!(p.as_ref(), &oor_vt);
                        assert_eq!(c.concretetype.as_ref(), Some(&CLASSTYPE.clone()));
                    }
                    other => panic!("etype must be OutOfRange vtable, got {other:?}"),
                },
                other => panic!("etype must be a Constant, got {other:?}"),
            }
            match &link.args[1] {
                Some(Hlvalue::Variable(_)) => {}
                other => panic!("evalue must be the malloc'd instance Variable, got {other:?}"),
            }
        }
        for block in inner.iterblocks() {
            for op in &block.borrow().operations {
                for arg in &op.args {
                    if let Hlvalue::Constant(c) = arg {
                        let n = match c.value {
                            ConstValue::UInt(n) => Some(n),
                            ConstValue::Int(n) if n >= 0 => Some(n as u64),
                            _ => None,
                        };
                        if let Some(n) = n {
                            assert!(
                                n != 0xD800 && n != 0xDFFF,
                                "allow_surrogates=True folds the D800–DFFF branch"
                            );
                        }
                    }
                }
            }
        }
        let mut opnames = Vec::new();
        for block in inner.iterblocks() {
            for op in &block.borrow().operations {
                opnames.push(op.opname.clone());
            }
        }
        for expected in [
            "malloc_varsize",
            "malloc",
            "getsubstruct",
            "setarrayitem",
            "setfield",
            "uint_le",
            "uint_rshift",
            "uint_and",
            "uint_or",
            "cast_uint_to_int",
            "cast_int_to_char",
        ] {
            assert!(
                opnames.iter().any(|n| n == expected),
                "ll_unichr_as_utf8 graph missing op {expected}; saw {opnames:?}"
            );
        }
    }

    #[test]
    fn build_ll_unichr_as_utf8_helper_after_exception_vtable_keeps_out_of_range_numbered() {
        use crate::translator::rtyper::rclass::{CLASSTYPE, ll_issubclass};

        let (ann, rtyper) = rtyper_after_unichr_annotation();
        let exception = crate::flowspace::model::HOST_ENV
            .lookup_exception_class("Exception")
            .expect("Exception");
        let exception_cd = ann
            .bookkeeper
            .getuniqueclassdef(&exception)
            .expect("Exception classdef");
        let exc_vt = classdef_vtable(&rtyper, &exception_cd);
        assert!(
            exception_cd
                .borrow()
                .repr
                .as_ref()
                .expect("Exception ClassRepr")
                .is_vtable_materialized(),
            "Exception vtable must be materialized before the helper graph"
        );

        let helper = build_ll_unichr_as_utf8_helper_graph(&rtyper, "ll_unichr_as_utf8")
            .expect("build_ll_unichr_as_utf8_helper_graph");
        crate::flowspace::model::checkgraph(&helper.graph.borrow());

        let oor_cd = crate::rlib::rutf8::out_of_range_classdef(&ann.bookkeeper)
            .expect("OutOfRange classdef");
        {
            let cd = oor_cd.borrow();
            assert!(
                cd.minid.is_some() && cd.maxid.is_some(),
                "OutOfRange must stay numbered after Exception's vtable is baked"
            );
        }
        let oor_vt = classdef_vtable(&rtyper, &oor_cd);
        assert!(oor_vt.nonzero());
        assert!(ll_issubclass(&oor_vt, &exc_vt).expect("ll_issubclass Exception"));
        let value_error = crate::flowspace::model::HOST_ENV
            .lookup_exception_class("ValueError")
            .expect("ValueError");
        let value_error_cd = ann
            .bookkeeper
            .getuniqueclassdef(&value_error)
            .expect("ValueError classdef");
        let ve_vt = classdef_vtable(&rtyper, &value_error_cd);
        assert!(!ll_issubclass(&oor_vt, &ve_vt).expect("ll_issubclass ValueError"));

        let inner = helper.graph.borrow();
        let except_key = crate::flowspace::model::BlockKey::of(&inner.exceptblock);
        let mut too_large_false: Option<crate::flowspace::model::BlockRef> = None;
        for block in inner.iterblocks() {
            let b = block.borrow();
            let le_max = b.operations.iter().any(|op| {
                op.opname == "uint_le"
                    && matches!(
                        op.args.get(1),
                        Some(Hlvalue::Constant(c)) if c.value == ConstValue::UInt(0x10FFFF)
                    )
            });
            if !le_max {
                continue;
            }
            for lnk in &b.exits {
                let link = lnk.borrow();
                let is_false = matches!(
                    &link.exitcase,
                    Some(Hlvalue::Constant(c)) if c.value == ConstValue::Bool(false)
                );
                if is_false {
                    too_large_false = link.target.clone();
                }
            }
        }
        let raise_block = too_large_false.expect("false arm of code <= r_uint(0x10FFFF)");
        let raise = raise_block.borrow();
        let typeptr_set = raise
            .operations
            .iter()
            .find(|op| {
                op.opname == "setfield"
                    && matches!(
                        op.args.get(1),
                        Some(Hlvalue::Constant(c))
                            if matches!(&c.value, ConstValue::ByteStr(s) if s == b"typeptr")
                    )
            })
            .expect("raise block setfield typeptr");
        match typeptr_set.args.get(2) {
            Some(Hlvalue::Constant(c)) => match &c.value {
                ConstValue::LLPtr(p) => {
                    assert!(p.nonzero());
                    assert_eq!(p.as_ref(), &oor_vt);
                    assert_eq!(c.concretetype.as_ref(), Some(&CLASSTYPE.clone()));
                }
                other => panic!("typeptr must be OutOfRange vtable, got {other:?}"),
            },
            other => panic!("typeptr value must be a Constant, got {other:?}"),
        }
        let link = raise.exits[0].borrow();
        assert!(
            link.target
                .as_ref()
                .is_some_and(|t| crate::flowspace::model::BlockKey::of(t) == except_key)
        );
        match &link.args[0] {
            Some(Hlvalue::Constant(c)) => match &c.value {
                ConstValue::LLPtr(p) => {
                    assert!(p.nonzero());
                    assert_eq!(p.as_ref(), &oor_vt);
                    assert_eq!(c.concretetype.as_ref(), Some(&CLASSTYPE.clone()));
                }
                other => panic!("etype must be OutOfRange vtable, got {other:?}"),
            },
            other => panic!("etype must be a Constant, got {other:?}"),
        }
    }
}
