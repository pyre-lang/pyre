//! Port of `rpython/tool/sourcetools.py`.
//!
//! Python code-object builders in the upstream module (`NiceCompile`,
//! `compile2`, `compile_template`, `rpython_wrapper`) rely on CPython's
//! `types.CodeType`, frame locals, and `exec`. The Rust host model does
//! not execute Python source, so this module ports the same observable
//! helper surface onto `GraphFunc` / `HostObject` where the surrounding
//! translator can carry equivalent metadata.

use crate::flowspace::bytecode::{CoFlags, HostCode};
use crate::flowspace::model::{Constant, GraphFunc, HostObject};

/// RPython `sourcetools.render_docstr(func, indent_str='',
/// closing_str='')` (`sourcetools.py`), specialised to an
/// already-extracted doc string.
pub fn render_docstr(
    doc: Option<&str>,
    indent_str: impl AsRef<str>,
    closing_str: impl AsRef<str>,
) -> Option<String> {
    let doc = doc?.replace('\\', r"\\");
    let indent_str = indent_str.as_ref();
    let closing_str = closing_str.as_ref();
    let double = format!(
        "{indent_str}\"\"\"{}\"\"\"{closing_str}",
        doc.replace('"', "\\\"")
    );
    let single = format!(
        "{indent_str}'''{}'''{closing_str}",
        doc.replace('\'', "\\'")
    );
    if single.len() < double.len() {
        Some(single)
    } else {
        Some(double)
    }
}

/// Maximum length of a `valid_identifier` result, mirroring upstream
/// `sourcetools.py PY_IDENTIFIER_MAX = 120`.
pub const PY_IDENTIFIER_MAX: usize = 120;

/// RPython `rpython.tool.sourcetools.valid_identifier`
/// (`sourcetools.py`).
///
/// Translate `stuff` through `PY_IDENTIFIER` (every byte that is not
/// `[0-9A-Za-z]` becomes `_`), prepend `_` if the first byte is
/// otherwise a digit (or empty), then truncate to
/// `PY_IDENTIFIER_MAX`.
pub fn valid_identifier(stuff: impl std::fmt::Display) -> String {
    let mut stuff: String = stuff
        .to_string()
        .bytes()
        .map(|b| if b.is_ascii_alphanumeric() { b } else { b'_' })
        .map(char::from)
        .collect();
    if stuff.is_empty() || stuff.as_bytes()[0].is_ascii_digit() {
        stuff.insert(0, '_');
    }
    stuff.truncate(PY_IDENTIFIER_MAX);
    stuff
}

/// RPython `has_varargs(func)` for callers that already have
/// `co_flags` (`sourcetools.py`).
pub fn has_varargs_flags(co_flags: u32) -> bool {
    (co_flags & CoFlags::VARARGS.bits()) != 0
}

/// RPython `has_varkeywords(func)` for callers that already have
/// `co_flags` (`sourcetools.py`).
pub fn has_varkeywords_flags(co_flags: u32) -> bool {
    (co_flags & CoFlags::VARKEYWORDS.bits()) != 0
}

/// RPython `has_varargs(func)` (`sourcetools.py`). Upstream
/// accepts either a function or a raw code object via
/// `getattr(func, 'func_code', func)`. The Rust port inspects the
/// `code` slot on [`GraphFunc`] (matching `func.func_code` /
/// `func.__code__`) and falls through to `false` when no code is
/// attached. Callers that already hold a [`HostCode`] reach the
/// same check via [`has_varargs_code`].
pub fn has_varargs(func: &GraphFunc) -> bool {
    func.code
        .as_ref()
        .map(|code| has_varargs_flags(code.co_flags))
        .unwrap_or(false)
}

/// RPython `has_varkeywords(func)` (`sourcetools.py`).
/// Upstream `getattr(func, 'func_code', func)` fallback over
/// [`GraphFunc.code`]. See [`has_varargs`].
pub fn has_varkeywords(func: &GraphFunc) -> bool {
    func.code
        .as_ref()
        .map(|code| has_varkeywords_flags(code.co_flags))
        .unwrap_or(false)
}

/// Convenience for callers that already hold a [`HostCode`] (the
/// `func_code` half of upstream's `getattr(func, 'func_code',
/// func)`). Mirrors the same predicate used by
/// `crate::flowspace::flowcontext` whose Frame is built from a raw
/// code object rather than a function carrier.
pub fn has_varargs_code(code: &HostCode) -> bool {
    has_varargs_flags(code.co_flags)
}

/// Sibling of [`has_varargs_code`] for `**kwargs` flag.
pub fn has_varkeywords_code(code: &HostCode) -> bool {
    has_varkeywords_flags(code.co_flags)
}

/// RPython `nice_repr_for_func(fn, name=None)` (`sourcetools.py`)
/// over the Rust `GraphFunc` carrier.
///
/// Upstream uses `fn.__module__` for the module slot and `cls.__name__`
/// for the class qualifier. The Rust port mirrors `cls.__name__` via
/// [`HostObject::simple_name`] (the short name, not the dotted
/// qualname). The module slot reads [`GraphFunc::module`] —
/// `from_host_code` populates it from `globals['__name__']` so any
/// caller that hands real globals through gets upstream-shaped
/// module names. Synthetic callers that hand an empty globals Dict
/// see `"?"` — exactly upstream's `module = '?'` default at
/// `sourcetools.py`.
pub fn nice_repr_for_func(func: &GraphFunc, name: Option<&str>) -> String {
    let display_name = name
        .map(str::to_string)
        .or_else(|| {
            func.class_
                .as_ref()
                .map(|cls| format!("{}.{}", cls.simple_name(), func.name))
        })
        .unwrap_or_else(|| func.name.clone());
    let module = func.module.as_deref().unwrap_or("?");
    let firstlineno = func.firstlineno.map(i64::from).unwrap_or(-1);
    format!("({module}:{firstlineno}){display_name}")
}

/// RPython `func_with_new_name(func, newname, globals=None)`
/// (`sourcetools.py`) over the Rust `GraphFunc` carrier.
///
/// Upstream re-binds the function to a fresh Python `types.FunctionType`
/// reusing `__code__`, `__defaults__`, and `__closure__`, then copies
/// `__dict__` and `__doc__`. pyre's `GraphFunc::clone()` already
/// duplicates every Python-function-attribute slot (`code`,
/// `defaults`, `closure`, `module`, `_jit_look_inside_`, etc.), so
/// `with_new_name` matches the rebind. The `globals=None` arm of
/// upstream's signature defaults to `func.__globals__`; pass
/// `Some(globals)` here to override (mirrors callers like
/// `descrcontainer.subseq_method` that supply a fresh globals dict
/// for the renamed copy).
pub fn func_with_new_name(
    func: &GraphFunc,
    newname: &str,
    globals: Option<&Constant>,
) -> GraphFunc {
    let mut renamed = func.with_new_name(newname);
    if let Some(g) = globals {
        renamed.globals = g.clone();
        // Upstream `types.FunctionType(code, globals, name, ...)`
        // (`sourcetools.py:221`) sets the new function's
        // `__module__` to the supplied globals' `__name__`. Mirror
        // that here so a renamed-into-fresh-globals copy reads its
        // module from the new globals dict — otherwise stale module
        // annotations leak through to the caller (notably
        // `descrcontainer.subseq_method`).
        renamed.module = module_from_globals(g);
    }
    renamed
}

/// Extract `globals.__name__` from a Python globals-dict Constant.
/// Mirrors CPython's `types.FunctionType(..., globals, name, ...)`
/// post-construction `f.__module__ = globals.get('__name__')` rule
/// (`Objects/funcobject.c:func_new_impl`).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn module_from_globals(globals: &Constant) -> Option<String> {
    let dict = match &globals.value {
        crate::flowspace::model::ConstValue::Dict(d) => d,
        _ => return None,
    };
    use crate::flowspace::model::ConstValue;
    let key_uni = ConstValue::UniStr("__name__".to_string());
    let key_byte = ConstValue::ByteStr(b"__name__".to_vec());
    let entry = dict.get(&key_uni).or_else(|| dict.get(&key_byte))?;
    match entry {
        ConstValue::UniStr(s) => Some(s.clone()),
        ConstValue::ByteStr(b) => std::str::from_utf8(b).ok().map(str::to_string),
        _ => None,
    }
}

/// Host-object counterpart for callers that still hold a Python
/// function as `HostObject`.
pub fn host_func_with_new_name(func: &HostObject, newname: &str) -> Option<HostObject> {
    func.renamed_user_function(newname)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{ConstValue, Constant};

    #[test]
    fn alphanumeric_passes_through() {
        assert_eq!(valid_identifier("abc123"), "abc123");
    }

    #[test]
    fn non_alpha_chars_become_underscores() {
        assert_eq!(valid_identifier("a-b.c d"), "a_b_c_d");
    }

    #[test]
    fn leading_digit_is_prefixed_with_underscore() {
        assert_eq!(valid_identifier("9foo"), "_9foo");
    }

    #[test]
    fn empty_string_becomes_underscore() {
        assert_eq!(valid_identifier(""), "_");
    }

    #[test]
    fn long_input_truncates_to_max() {
        let s = "x".repeat(200);
        let out = valid_identifier(&s);
        assert_eq!(out.len(), PY_IDENTIFIER_MAX);
        assert!(out.chars().all(|c| c == 'x'));
    }

    #[test]
    fn valid_identifier_translates_utf8_bytes_like_python2() {
        assert_eq!(valid_identifier("\u{00e9}"), "__");
    }

    #[test]
    fn render_docstr_escapes_and_picks_shorter_quotes() {
        let out = render_docstr(Some("a\"b"), "", "").unwrap();
        assert_eq!(out, "'''a\"b'''");
        assert_eq!(render_docstr(None, "", ""), None);
    }

    #[test]
    fn vararg_flag_helpers_match_upstream_masks() {
        assert!(has_varargs_flags(CoFlags::VARARGS.bits()));
        assert!(has_varkeywords_flags(CoFlags::VARKEYWORDS.bits()));
        assert!(!has_varargs_flags(CoFlags::VARKEYWORDS.bits()));
    }

    #[test]
    fn func_with_new_name_copies_graphfunc_with_fresh_identity() {
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let mut func = GraphFunc::new("old", globals);
        func.filename = Some("/tmp/mod.py".to_string());
        func.firstlineno = Some(17);

        let renamed = func_with_new_name(&func, "new", None);
        assert_eq!(renamed.name, "new");
        assert_ne!(renamed.id, func.id);
        // The default `globals=None` arm reuses the original
        // function's globals (upstream `sourcetools.py:219-220`).
        assert_eq!(renamed.globals, func.globals);
        // `module` is None on a synthetic-globals fixture — upstream
        // `sourcetools.py:265-266` emits `"?"` when `__module__` is
        // missing. The filename slot is intentionally NOT used as a
        // fallback (parity).
        assert_eq!(nice_repr_for_func(&renamed, None), "(?:17)new");
    }

    #[test]
    fn func_with_new_name_overrides_globals() {
        // Upstream `sourcetools.py func_with_new_name(func,
        // newname, globals=None)`: `Some(globals)` replaces the
        // function's `__globals__` on the renamed copy.
        let mut original_globals = indexmap::IndexMap::new();
        original_globals.insert(
            ConstValue::byte_str("__name__"),
            ConstValue::byte_str("orig.mod"),
        );
        let func = GraphFunc::new(
            "old",
            Constant::new(ConstValue::Dict(std::sync::Arc::new(original_globals))),
        );

        let mut new_globals_map = indexmap::IndexMap::new();
        new_globals_map.insert(
            ConstValue::byte_str("__name__"),
            ConstValue::byte_str("override.mod"),
        );
        let new_globals = Constant::new(ConstValue::Dict(std::sync::Arc::new(new_globals_map)));

        let renamed = func_with_new_name(&func, "new", Some(&new_globals));
        assert_eq!(renamed.name, "new");
        assert_eq!(renamed.globals, new_globals);
        // The original function is untouched — upstream creates a new
        // FunctionType, never mutates the input.
        assert_ne!(func.globals, new_globals);
    }

    #[test]
    fn nice_repr_uses_module_when_present() {
        // Upstream `sourcetools.py:264 module = fn.__module__`. When
        // the GraphFunc carrier knows its `__module__`, that value
        // appears in the repr unchanged.
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let mut func = GraphFunc::new("foo", globals);
        func.module = Some("pkg.mod".to_string());
        func.firstlineno = Some(42);
        assert_eq!(nice_repr_for_func(&func, None), "(pkg.mod:42)foo");
    }
}
