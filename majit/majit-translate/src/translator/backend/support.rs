//! Port of `rpython/translator/c/support.py`.
//!
//! The `log = AnsiLogger("c")` handle is represented by [`LOG_NAME`]
//! until `rpython/tool/ansi_print.py` itself is ported; callers can use
//! the same channel name without growing a local logger abstraction.

use std::collections::HashMap;

use crate::flowspace::model::{ConstValue, Constant};
use crate::translator::gensupp::NameManager;
use crate::translator::rtyper::lltypesystem::lltype::{
    Array, GcKind, LowLevelType, PtrTarget, typeOf,
};

/// RPython `support.USESLOTS = True` (`support.py`).
///
/// Rust structs already have a fixed layout, but keeping the flag makes
/// the module-level surface match upstream for callers that branch on
/// it during a line-by-line port.
pub const USESLOTS: bool = true;

/// RPython `support.log = AnsiLogger("c")` (`support.py:169-170`).
pub const LOG_NAME: &str = "c";

/// RPython `support.barebonearray(ARRAY)` (`support.py`).
///
/// True iff `ARRAY` is "simple": carries the `nolength` hint, is not
/// GC-managed, and its element type is not `Void`.
///
/// Upstream `:13 ARRAY._hints.get('nolength', False)` uses Python
/// truthiness, so any truthy hint value (non-zero integer, non-empty
/// string, `True`, …) admits the bareboneness test. Use
/// [`ConstValue::truthy`] which already encodes that semantic.
pub fn barebone_array(array: &Array) -> bool {
    let has_nolength = array
        ._hints
        .get("nolength")
        .and_then(ConstValue::truthy)
        .unwrap_or(false);
    has_nolength && array._gckind != GcKind::Gc && array.OF != LowLevelType::Void
}

/// RPython spelling-preserving wrapper for
/// [`barebone_array`].
pub fn barebonearray(array: &Array) -> bool {
    barebone_array(array)
}

/// RPython `support.cdecl(ctype, cname, is_thread_local=False)`
/// (`support.py:20-31`).
///
/// Replaces the `@` placeholder in `ctype` with `cname`, with the
/// special case `(@)` → `@` so that function declarations don't end
/// up with redundant parentheses around the function name.
pub fn cdecl(ctype: &str, cname: &str, is_thread_local: bool) -> String {
    let prefix = if is_thread_local { "__thread " } else { "" };
    let cleaned = ctype.replace("(@)", "@");
    let with_name = cleaned.replace('@', cname);
    format!("{prefix}{}", with_name.trim())
}

/// RPython `support.forward_cdecl(ctype, cname, standalone,
/// is_thread_local=False, is_exported=False)` (`support.py:33-43`).
///
/// `standalone` is accepted and ignored, matching upstream's signature.
pub fn forward_cdecl(
    ctype: &str,
    cname: &str,
    _standalone: bool,
    is_thread_local: bool,
    is_exported: bool,
) -> String {
    let prefix = if is_exported {
        assert!(
            !is_thread_local,
            "forward_cdecl: is_thread_local incompatible with is_exported",
        );
        "RPY_EXPORTED ".to_string()
    } else if is_thread_local {
        "RPY_EXTERN __thread ".to_string()
    } else {
        "RPY_EXTERN ".to_string()
    };
    format!("{prefix}{}", cdecl(ctype, cname, false))
}

/// RPython `support.somelettersfrom(s)` (`support.py`).
///
/// Upstream iterates Python 2 `str` byte-by-byte (`for c in s`,
/// `s.title()`, `s[:2]`), so the port also walks bytes. The result is
/// returned as `Vec<u8>` for byte-exact parity — `STRUCT._name` (the
/// sole upstream caller, `node.py:73`) is always an ASCII C
/// identifier, but the helper is defined over arbitrary `str` and
/// non-ASCII inputs would split UTF-8 boundaries on the `s[:2]`
/// fallback.
pub fn somelettersfrom(s: &[u8]) -> Vec<u8> {
    let mut upcase: Vec<u8> = s.iter().copied().filter(u8::is_ascii_uppercase).collect();
    if upcase.is_empty() {
        // Python 2 `s.title()` byte-shaped: upper-case the first ASCII
        // letter of each alphanumeric run; non-alnum bytes break runs.
        // Bytes >= 0x80 are not letters under `bytes.title()`, so they
        // pass through unchanged.
        let mut titled: Vec<u8> = Vec::with_capacity(s.len());
        let mut prev_alnum = false;
        for &b in s {
            let is_alnum = b.is_ascii_alphanumeric();
            if is_alnum && !prev_alnum && b.is_ascii_lowercase() {
                titled.push(b.to_ascii_uppercase());
            } else {
                titled.push(b);
            }
            prev_alnum = is_alnum;
        }
        upcase = titled
            .into_iter()
            .filter(|b| b.is_ascii_uppercase())
            .collect();
    }
    let has_locase = s.iter().any(u8::is_ascii_lowercase);
    if has_locase && !upcase.is_empty() {
        for b in &mut upcase {
            *b = b.to_ascii_lowercase();
        }
        upcase
    } else {
        // Upstream `s[:2].lower()` — byte-level slice; ASCII letters
        // lower-case, other bytes pass through.
        s.iter().take(2).map(u8::to_ascii_lowercase).collect()
    }
}

/// RPython `support.is_pointer_to_forward_ref(T)`
/// (`support.py:55-58`).
pub fn is_pointer_to_forward_ref(t: &LowLevelType) -> bool {
    let LowLevelType::Ptr(p) = t else {
        return false;
    };
    matches!(p.TO, PtrTarget::ForwardReference(_))
}

/// RPython `support.llvalue_from_constant(c)` (`support.py`).
///
/// Upstream:
/// ```python
/// T = c.concretetype
/// if T == lltype.Void:
///     return None
/// else:
///     ACTUAL_TYPE = lltype.typeOf(c.value)
///     if not is_pointer_to_forward_ref(ACTUAL_TYPE):
///         assert ACTUAL_TYPE == T
///     return c.value
/// ```
///
/// Adaptation note: upstream `lltype.typeOf(value)` discriminates
/// among the seven integer families (`Signed`, `Unsigned`,
/// `SignedLongLong`, …) by the value's Python class
/// (`int`, `r_uint`, `r_longlong`, …). pyre's [`ConstValue::Int`]
/// flattens every integer Python type into one `i64` carrier, so
/// `typeOf` can no longer return a single type — it returns the
/// *family* the carrier is compatible with. The upstream
/// `ACTUAL_TYPE == T` assertion lowers to
/// "is the value carrier compatible with the declared
/// concretetype": a `Bool` concretetype with an `Int` carrier still
/// surfaces as a panic (real bug — `lltype.typeOf(7) == Signed`,
/// not `Bool`); an `Unsigned` concretetype with an `Int` carrier
/// passes (matches upstream `lltype.typeOf(r_uint(7)) == Unsigned`).
///
/// Pointer-to-forward-reference carriers are exempted exactly as
/// upstream — comment quoted: "If the type is still uncomputed, we
/// can't make this check."
pub fn llvalue_from_constant(c: &Constant) -> Option<ConstValue> {
    let t = c
        .concretetype
        .as_ref()
        .expect("llvalue_from_constant: constant lacks concretetype");
    if *t == LowLevelType::Void {
        return None;
    }
    if let Some(actual_ptr_type) = lltype_when_pointer(&c.value) {
        // Upstream `:69`: forward-ref pointers skip the assertion.
        if is_pointer_to_forward_ref(&actual_ptr_type) {
            return Some(c.value.clone());
        }
    }
    assert!(
        const_value_compatible_with_concretetype(&c.value, t),
        "ll typeOf({:?}) incompatible with {:?} \
         (upstream `support.py:67-69 ACTUAL_TYPE == T`)",
        c.value,
        t,
    );
    Some(c.value.clone())
}

/// Concrete `Ptr` reconstruction for `c.value` — returns the
/// `LowLevelType::Ptr` shape upstream's `lltype.typeOf` produces for
/// a `_ptr`-carrying constant. Used solely to detect the
/// forward-reference exemption upstream applies at `support.py:69`.
fn lltype_when_pointer(value: &ConstValue) -> Option<LowLevelType> {
    match value {
        ConstValue::LLPtr(ptr) => Some(LowLevelType::Ptr(Box::new(typeOf(ptr)))),
        _ => None,
    }
}

/// Compatibility predicate replacing the upstream
/// `lltype.typeOf(value) == T` equality. Each `ConstValue` arm names
/// the lltype family that valid pyre carriers can lower into. The
/// upstream cases are reproduced verbatim where the carrier already
/// distinguishes them (Bool, Float family, Char, UniChar, Address,
/// Ptr). Signed integers are `ConstValue::Int`; `r_uint` /
/// `r_ulonglong` are `ConstValue::UInt` (`rarithmetic.build_int`).
fn const_value_compatible_with_concretetype(value: &ConstValue, t: &LowLevelType) -> bool {
    match (value, t) {
        // Upstream integer families: `int → Signed`, `r_uint →
        // Unsigned`, `r_longlong → SignedLongLong`, `r_ulonglong →
        // UnsignedLongLong`. `SignedLongLong is Signed` on 64-bit
        // (`_numbertypes[r_int] = _numbertypes[int]`); `UnsignedLongLong
        // is Unsigned` (`r_ulonglong is r_uint`).
        (
            ConstValue::Int(_),
            LowLevelType::Signed
            | LowLevelType::SignedLongLong
            | LowLevelType::SignedLongLongLong
            | LowLevelType::UnsignedLongLongLong,
        ) => true,
        (
            ConstValue::UInt(_),
            LowLevelType::Unsigned
            | LowLevelType::UnsignedLongLong
            | LowLevelType::UnsignedLongLongLong,
        ) => true,
        (ConstValue::Bool(_), LowLevelType::Bool) => true,
        // Upstream `lltype.typeOf(float) == Float`, `r_singlefloat →
        // SingleFloat`, `r_longfloat → LongFloat`. pyre flattens the
        // Python-side float subtypes into `Float(u64)` bits — the
        // bit pattern itself does not encode width.
        (
            ConstValue::Float(_),
            LowLevelType::Float | LowLevelType::SingleFloat | LowLevelType::LongFloat,
        ) => true,
        // Upstream `:65 lltype.typeOf(c) == Char` for a 1-char str.
        (ConstValue::ByteStr(s), LowLevelType::Char) => s.len() == 1,
        (ConstValue::UniStr(s), LowLevelType::UniChar) => s.chars().count() == 1,
        (ConstValue::LLAddress(_), LowLevelType::Address) => true,
        (ConstValue::LLPtr(ptr), LowLevelType::Ptr(declared)) => &typeOf(ptr) == declared.as_ref(),
        // Every other pair is a real mismatch (e.g. Bool concretetype
        // with Int carrier — upstream's `lltype.typeOf(7) != Bool`).
        _ => false,
    }
}

/// RPython `support._char_repr(c)` (`support.py`).
///
/// Upstream operates on a Python 2 `str` (byte string), so the input
/// is a raw byte. Mirror by accepting `u8` and matching ASCII space
/// through `~` (0x20..=0x7E) literally; everything else escapes to
/// `\NNN` with three octal digits.
fn byte_repr(b: u8) -> String {
    if matches!(b, b'\\' | b'"' | b'?') {
        format!("\\{}", b as char)
    } else if (0x20..0x7F).contains(&b) {
        (b as char).to_string()
    } else {
        format!("\\{:03o}", b)
    }
}

/// RPython `support._line_repr(s)` (`support.py`).
fn line_repr(s: &[u8]) -> String {
    s.iter().copied().map(byte_repr).collect()
}

/// RPython `support.c_string_constant(s)` (`support.py`).
///
/// Returns a `" "`-delimited string literal for C, broken at every
/// 64th source byte (upstream uses Python 2 byte strings; the
/// chunking happens in byte-space so non-ASCII inputs split at the
/// same offsets as upstream).
pub fn c_string_constant(s: &[u8]) -> String {
    let mut lines = Vec::new();
    for chunk in s.chunks(64) {
        lines.push(format!("\"{}\"", line_repr(chunk)));
    }
    lines.join("\n")
}

/// RPython `support.c_char_array_constant(s)` (`support.py`).
///
/// Returns an initialiser for a `char[N]` array where `N == len(s)`
/// (byte length, matching upstream's Python 2 `str`). When `s` ends
/// with a NUL and is below 1024 bytes, upstream uses a `" "` literal
/// (with the NUL stripped, since C string literals carry an implicit
/// terminator); otherwise the initialiser is a `{ }`-list of
/// `ord(c)` values produced 20 bytes at a time.
pub fn c_char_array_constant(s: &[u8]) -> String {
    if s.last() == Some(&0u8) && (1 < s.len() && s.len() < 1024) {
        return c_string_constant(&s[..s.len() - 1]);
    }
    let lines: Vec<String> = s
        .chunks(20)
        .map(|chunk| {
            chunk
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();
    if lines.len() > 1 {
        format!("{{\n{}}}", lines.join(",\n"))
    } else {
        format!("{{{}}}", lines.first().cloned().unwrap_or_default())
    }
}

/// RPython `support.gen_assignments(assignments)` (`support.py`).
///
/// Reorders `(typename, dest, src)` assignments to avoid clashes —
/// equivalent to a tuple assignment, reading all sources first and
/// writing all targets next, with cycle-breaking via a `tmp` slot for
/// pure disjoint cycles.
pub fn gen_assignments(assignments: &[(String, String, String)]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut srccount: HashMap<String, usize> = HashMap::new();
    let mut dest2src: HashMap<String, (String, String)> = HashMap::new();
    for (typename, dest, src) in assignments {
        if src != dest {
            *srccount.entry(src.clone()).or_insert(0) += 1;
            dest2src.insert(dest.clone(), (src.clone(), typename.clone()));
        }
    }

    while !dest2src.is_empty() {
        let mut progress = false;
        let dsts: Vec<String> = dest2src.keys().cloned().collect();
        for dst in dsts {
            if !srccount.contains_key(&dst) {
                let (src, _typename) = dest2src.remove(&dst).expect("dst present");
                out.push(format!("{dst} = {src};"));
                if let Some(c) = srccount.get_mut(&src) {
                    *c -= 1;
                    if *c == 0 {
                        srccount.remove(&src);
                    }
                }
                progress = true;
            }
        }
        if !progress {
            // Pure disjoint cycles only; break them via a tmp slot.
            while !dest2src.is_empty() {
                let dst0 = dest2src.keys().next().cloned().expect("dest2src non-empty");
                let (src0, typename0) = dest2src.remove(&dst0).expect("dst0 present");
                assert_eq!(srccount.get(&dst0).copied().unwrap_or(0), 1);
                let startingpoint = dst0.clone();
                let tmpdecl = cdecl(&typename0, "tmp", false);
                let mut code = vec![format!("{{ {tmpdecl} = {dst0};")];
                let mut dst = dst0;
                let mut src = src0;
                while src != startingpoint {
                    code.push(format!("{dst} = {src};"));
                    dst = src;
                    let (next_src, _next_typename) =
                        dest2src.remove(&dst).expect("cycle continuation present");
                    assert_eq!(srccount.get(&dst).copied().unwrap_or(0), 1);
                    src = next_src;
                }
                code.push(format!("{dst} = tmp; }}"));
                out.push(code.join(" "));
            }
        }
    }
    out
}

/// RPython `support.CNameManager` (`support.py`).
///
/// Upstream `:74 class CNameManager(NameManager)` is a subclass: every
/// `NameManager` method is callable on a `CNameManager` directly. Rust
/// has no class inheritance; the closest mirror is composition + a
/// transparent `Deref<Target=NameManager>` so
/// `cnm.make_reserved_names(…)` / `cnm.uniquename(…)` resolve the same
/// way they would on a `NameManager`. The `__init__` body preloads the
/// C99 keyword list and the `pypy_` global prefix.
///
/// `local_scope_from` / `localScope` cannot reach through `Deref`
/// because those constructors take `&Rc<RefCell<NameManager>>`. When
/// the LocalScope surface gets its first real consumer (today none),
/// either rewrap `database.namespace` as
/// `Rc<RefCell<CNameManager>>` (then deref through the Rc) or have
/// `CNameManager` expose a thin helper that hands out an
/// `Rc<RefCell<NameManager>>` shared with the embedded inner.
#[derive(Debug, Clone)]
pub struct CNameManager {
    inner: NameManager,
}

impl CNameManager {
    /// RPython `CNameManager.__init__(global_prefix='pypy_')`
    /// (`support.py:75-89`).
    pub fn new() -> Self {
        Self::with_prefix("pypy_")
    }

    /// Constructor variant matching upstream's optional kwarg.
    pub fn with_prefix(global_prefix: impl Into<String>) -> Self {
        let mut inner = NameManager::new(global_prefix, "_");
        // Upstream `:78-89` — the C99 draft keyword list, copied
        // verbatim. The whitespace layout is preserved so a textual
        // diff against upstream highlights any divergence.
        inner.make_reserved_names(
            "auto      enum      restrict  unsigned \
             break     extern    return    void \
             case      float     short     volatile \
             char      for       signed    while \
             const     goto      sizeof    _Bool \
             continue  if        static    _Complex \
             default   inline    struct    _Imaginary \
             do        int       switch \
             double    long      typedef \
             else      register  union",
        );
        Self { inner }
    }
}

impl std::ops::Deref for CNameManager {
    type Target = NameManager;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for CNameManager {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Default for CNameManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdecl_replaces_at_with_name() {
        assert_eq!(cdecl("Signed @", "x", false), "Signed x");
    }

    #[test]
    fn cdecl_collapses_paren_at_for_function_decls() {
        // `Signed (@)(int a)` is the upstream "function-pointer
        // template": the parens around `@` exist only for the
        // pointer-typedef shape and must collapse for the plain
        // function-name case.
        assert_eq!(cdecl("Signed (@)(int a)", "fn", false), "Signed fn(int a)");
    }

    #[test]
    fn cdecl_keeps_explicit_pointer_parens() {
        // For the function-pointer case the template is `Signed
        // (*@)(int a)` — `(*@)` must NOT collapse, because the
        // upstream collapse rule keys on the literal `(@)`. The
        // resulting decl keeps the asterisk and the wrapping parens.
        assert_eq!(
            cdecl("Signed (*@)(int a)", "fn", false),
            "Signed (*fn)(int a)"
        );
    }

    #[test]
    fn cdecl_prepends_thread_local_qualifier() {
        assert_eq!(cdecl("Signed @", "x", true), "__thread Signed x");
    }

    #[test]
    fn cname_manager_reserves_c_keywords() {
        let mut nm = CNameManager::new();
        // Reserved keyword `auto` collides with the preloaded set —
        // first claim must yield a unique name, not the keyword.
        let name = nm.uniquename("auto", Some(false), 50);
        assert_ne!(name, "pypy_auto", "should not return reserved keyword");
    }

    #[test]
    fn cname_manager_uses_pypy_prefix_by_default() {
        let mut nm = CNameManager::new();
        assert_eq!(nm.uniquename("foo", Some(false), 50), "pypy_foo");
    }

    #[test]
    fn cname_manager_with_prefix_overrides_default() {
        let mut nm = CNameManager::with_prefix("custom_");
        assert_eq!(nm.uniquename("foo", Some(false), 50), "custom_foo");
    }

    #[test]
    fn forward_cdecl_uses_rpy_extern_for_default() {
        assert_eq!(
            forward_cdecl("Signed @", "x", false, false, false),
            "RPY_EXTERN Signed x"
        );
    }

    #[test]
    fn forward_cdecl_threadlocal_keeps_rpy_extern_with_thread_qualifier() {
        assert_eq!(
            forward_cdecl("Signed @", "x", false, true, false),
            "RPY_EXTERN __thread Signed x"
        );
    }

    #[test]
    fn forward_cdecl_exported_uses_rpy_exported() {
        assert_eq!(
            forward_cdecl("Signed @", "x", false, false, true),
            "RPY_EXPORTED Signed x"
        );
    }

    #[test]
    #[should_panic(expected = "is_thread_local incompatible with is_exported")]
    fn forward_cdecl_panics_on_exported_threadlocal() {
        let _ = forward_cdecl("Signed @", "x", false, true, true);
    }

    #[test]
    fn somelettersfrom_takes_uppercase_letters_when_present() {
        assert_eq!(somelettersfrom(b"MyType"), b"mt");
    }

    #[test]
    fn somelettersfrom_synthesises_uppercase_via_title() {
        // No uppercase in `ab`, but `.title()` capitalises the first
        // letter → upcase=['A'], locase=['a','b'] → 'a'.
        assert_eq!(somelettersfrom(b"ab"), b"a");
    }

    #[test]
    fn somelettersfrom_falls_back_to_first_two_when_no_locase() {
        // All-uppercase input has no lowercase, so neither branch in
        // upstream's `if locase and upcase:` matches → s[:2].lower().
        assert_eq!(somelettersfrom(b"ABCD"), b"ab");
    }

    #[test]
    fn somelettersfrom_iterates_bytes_not_chars_for_non_ascii() {
        // Python 2 byte string: a single non-ASCII char like "é"
        // (0xC3 0xA9 in UTF-8) is two bytes. Neither byte is an ASCII
        // letter, so both branches collapse to `s[:2].lower()` → the
        // raw UTF-8 prefix unchanged (no byte is ASCII-uppercase).
        assert_eq!(somelettersfrom("é".as_bytes()), &[0xC3, 0xA9]);
    }

    #[test]
    fn is_pointer_to_forward_ref_matches_only_ptr_to_forward_ref() {
        use crate::translator::rtyper::lltypesystem::lltype::{ForwardReference, Ptr, PtrTarget};
        let p = Ptr {
            TO: PtrTarget::ForwardReference(ForwardReference::new()),
        };
        let t: LowLevelType = p.into();
        assert!(is_pointer_to_forward_ref(&t));

        // Non-Ptr → false.
        assert!(!is_pointer_to_forward_ref(&LowLevelType::Signed));
    }

    #[test]
    fn c_string_constant_wraps_in_quotes_with_escapes() {
        assert_eq!(c_string_constant(b"hi\n"), "\"hi\\012\"");
    }

    #[test]
    fn c_string_constant_breaks_at_64_chars() {
        let s = vec![b'a'; 70];
        let out = c_string_constant(&s);
        assert!(out.contains('\n'));
        assert!(out.starts_with('"'));
    }

    #[test]
    fn c_string_constant_chunks_non_ascii_in_byte_space() {
        // Upstream operates on byte strings, so a non-ASCII UTF-8
        // source splits at byte 64, not at character 64.
        let mut s = vec![b'a'; 60];
        s.extend_from_slice("é".as_bytes()); // 2 bytes (0xC3 0xA9)
        s.extend_from_slice(&[b'b'; 10]);
        let out = c_string_constant(&s);
        // First line carries 64 bytes, second line carries the rest;
        // the multi-byte code point is split across the boundary.
        assert!(out.contains('\n'));
    }

    #[test]
    fn c_char_array_constant_uses_string_literal_for_short_nul_terminated() {
        // 4-byte string ending in NUL → `"abc"` (literal, NUL implicit).
        assert_eq!(c_char_array_constant(b"abc\x00"), "\"abc\"");
    }

    #[test]
    fn c_char_array_constant_falls_back_to_brace_init_for_non_nul() {
        // No trailing NUL → `{97,98,99}`.
        assert_eq!(c_char_array_constant(b"abc"), "{97,98,99}");
    }

    #[test]
    fn gen_assignments_runs_simple_chain_in_order() {
        // a = b; b = c; — c is the only "leaf" source.
        let assigns = vec![
            ("Signed @".to_string(), "a".to_string(), "b".to_string()),
            ("Signed @".to_string(), "b".to_string(), "c".to_string()),
        ];
        let out = gen_assignments(&assigns);
        // Read order: write `a` first (b still intact), then `b`.
        assert_eq!(out, vec!["a = b;".to_string(), "b = c;".to_string()]);
    }

    #[test]
    fn gen_assignments_breaks_disjoint_cycle_with_tmp() {
        // a = b; b = a; — pure 2-cycle.
        let assigns = vec![
            ("Signed @".to_string(), "a".to_string(), "b".to_string()),
            ("Signed @".to_string(), "b".to_string(), "a".to_string()),
        ];
        let out = gen_assignments(&assigns);
        assert_eq!(out.len(), 1);
        assert!(out[0].contains("Signed tmp"));
        assert!(out[0].starts_with('{'));
        assert!(out[0].ends_with("}"));
    }

    #[test]
    fn barebone_array_requires_nolength_nongc_nonvoid() {
        use crate::flowspace::model::ConstValue;
        use crate::translator::rtyper::lltypesystem::lltype::Array;
        let plain = Array::with_hints(LowLevelType::Signed, vec![]);
        assert!(!barebone_array(&plain));
        let nolength_nongc = Array::with_hints(
            LowLevelType::Signed,
            vec![("nolength".to_string(), ConstValue::Bool(true))],
        );
        assert!(barebone_array(&nolength_nongc));
        let void = Array::with_hints(
            LowLevelType::Void,
            vec![("nolength".to_string(), ConstValue::Bool(true))],
        );
        assert!(!barebone_array(&void));
    }

    #[test]
    fn llvalue_from_constant_returns_none_for_void() {
        let c = Constant::with_concretetype(ConstValue::Int(7), LowLevelType::Void);
        assert_eq!(llvalue_from_constant(&c), None);
    }

    #[test]
    fn llvalue_from_constant_returns_matching_value() {
        let c = Constant::with_concretetype(ConstValue::Int(7), LowLevelType::Signed);
        assert_eq!(llvalue_from_constant(&c), Some(ConstValue::Int(7)));
    }

    /// Upstream `support.py:65-69` runs
    /// `lltype.typeOf(r_uint(7)) == Unsigned`. The value is
    /// `ConstValue::UInt`.
    #[test]
    fn llvalue_from_constant_accepts_uint_carrier_with_unsigned_concretetype() {
        let c = Constant::with_concretetype(ConstValue::UInt(7), LowLevelType::Unsigned);
        assert_eq!(llvalue_from_constant(&c), Some(ConstValue::UInt(7)));
    }

    /// `SignedLongLong` is `Signed` on 64-bit (`_numbertypes[r_int] =
    /// _numbertypes[int]`), so its constants stay `Int`.
    /// `UnsignedLongLong` is `Unsigned` (`r_ulonglong is r_uint`).
    #[test]
    fn llvalue_from_constant_accepts_int_carrier_with_long_long_family() {
        let c = Constant::with_concretetype(ConstValue::Int(42), LowLevelType::SignedLongLong);
        assert_eq!(llvalue_from_constant(&c), Some(ConstValue::Int(42)));
        let c = Constant::with_concretetype(ConstValue::UInt(42), LowLevelType::UnsignedLongLong);
        assert_eq!(llvalue_from_constant(&c), Some(ConstValue::UInt(42)));
        let c = Constant::with_concretetype(ConstValue::Int(42), LowLevelType::SignedLongLongLong);
        assert_eq!(llvalue_from_constant(&c), Some(ConstValue::Int(42)));
        let c =
            Constant::with_concretetype(ConstValue::Int(42), LowLevelType::UnsignedLongLongLong);
        assert_eq!(llvalue_from_constant(&c), Some(ConstValue::Int(42)));
    }

    /// Upstream invariant still bites real mismatches: `Bool`
    /// concretetype with `Int` value disagrees with
    /// `lltype.typeOf(7) == Signed != Bool` and must panic rather
    /// than silently return the bare `Int(7)`.
    #[test]
    #[should_panic(expected = "ll typeOf")]
    fn llvalue_from_constant_panics_on_bool_concretetype_with_int_value() {
        let c = Constant::with_concretetype(ConstValue::Int(7), LowLevelType::Bool);
        let _ = llvalue_from_constant(&c);
    }
}
