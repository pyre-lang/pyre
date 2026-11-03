//! Port of `rpython/translator/gensupp.py`.

use parking_lot::Mutex;
#[cfg(test)]
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
#[cfg(test)]
use std::rc::Rc;
use std::sync::OnceLock;

/// RPython `gensupp.uniquemodulename(name, SEEN=set())`
/// (`gensupp.py`).
///
/// Returns `name_<i>` where `<i>` is the smallest positive integer that
/// has not been handed out before during this process. Upstream stashes
/// the seen set in a default-argument `SEEN=set()` (the well-known
/// Python mutable-default-as-state idiom) so the cache lives across all
/// calls. The Rust port hangs the cache off a process-wide `OnceLock`
/// `Mutex<HashSet<String>>` for the same effect.
pub fn uniquemodulename(name: &str) -> String {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = seen.lock();
    let mut i: usize = 0;
    loop {
        i += 1;
        let result = format!("{name}_{i}");
        if !guard.contains(&result) {
            guard.insert(result.clone());
            return result;
        }
    }
}

/// RPython `gensupp.C_IDENTIFIER` (`gensupp.py`). Maps any byte
/// not in `[0-9A-Za-z]` to `_`.
fn c_identifier_byte(b: u8) -> u8 {
    if b.is_ascii_alphanumeric() { b } else { b'_' }
}

/// Apply the upstream `str.translate(C_IDENTIFIER)` to a UTF-8 input.
/// Upstream Python operates on bytes directly; non-ASCII UTF-8 bytes
/// therefore become `_` byte-by-byte.
#[cfg(test)]
fn translate_c_identifier(s: &str) -> String {
    s.bytes().map(c_identifier_byte).map(char::from).collect()
}

/// RPython `class NameManager(object)` (`gensupp.py`).
///
/// Tracks every global identifier the generated C source uses so that
/// each new name claim returns a string that doesn't collide with a
/// previously claimed one. `localScope()` returns a [`LocalScope`]
/// that mutates `self.scopelist` to give per-nesting-level dedup.
#[derive(Debug, Clone)]
pub struct NameManager {
    /// `seennames`: every name encountered (reserved or claimed),
    /// mapped to the next dedup counter. Upstream uses a Python dict;
    /// the value is the count of times the name has been seen so the
    /// `_ensure_unique` recursion produces `name`, `name_1`, `name_2`,
    /// etc.
    pub seennames: HashMap<String, usize>,
    /// `scope`: top-level scope id. Upstream initialises to 0 and
    /// `localScope()` increments per nesting level; preserved here so
    /// the field surface matches upstream even though nothing reads it
    /// in this slice.
    pub scope: usize,
    /// `scopelist`: per-scope name lists. Empty until `localScope()`
    /// is ported.
    pub scopelist: Vec<HashMap<String, Vec<String>>>,
    pub global_prefix: String,
    pub number_sep: String,
}

impl NameManager {
    /// RPython `NameManager.__init__(global_prefix='', number_sep='_')`.
    pub fn new(global_prefix: impl Into<String>, number_sep: impl Into<String>) -> Self {
        Self {
            seennames: HashMap::new(),
            scope: 0,
            scopelist: Vec::new(),
            global_prefix: global_prefix.into(),
            number_sep: number_sep.into(),
        }
    }

    /// RPython `NameManager.make_reserved_names(txt)` (`gensupp.py`).
    ///
    /// Upstream raises `NameError("%s has already been seen!")` with the
    /// `%s` template literally un-substituted (a pre-existing upstream
    /// bug — `raise NameError("%s has already been seen!")` rather than
    /// `% name`). The Rust port panics with the exact same un-substituted
    /// string so observable behaviour matches. This is a programming
    /// error: all calls must run before any `uniquename`.
    pub fn make_reserved_names(&mut self, txt: &str) {
        for name in txt.split_ascii_whitespace() {
            if self.seennames.contains_key(name) {
                panic!("%s has already been seen!");
            }
            self.seennames.insert(name.to_string(), 1);
        }
    }

    /// RPython `NameManager._ensure_unique(basename)` (`gensupp.py`).
    fn ensure_unique(&mut self, basename: String) -> String {
        let n = *self.seennames.get(&basename).unwrap_or(&0);
        self.seennames.insert(basename.clone(), n + 1);
        if n != 0 {
            self.ensure_unique(format!("{basename}_{n}"))
        } else {
            basename
        }
    }

    /// RPython `NameManager.uniquename(basename, with_number=None,
    /// bare=False, lenmax=50)` (`gensupp.py`).
    ///
    /// Returns the prefixed unique name (upstream `bare=False`).
    /// Callers needing the bare/prefixed pair use
    /// [`Self::uniquename_bare`].
    pub fn uniquename(
        &mut self,
        basename: &str,
        with_number: Option<bool>,
        lenmax: usize,
    ) -> String {
        self.uniquename_bare(basename, with_number, lenmax).1
    }

    /// RPython `NameManager.uniquename(..., bare=True)`. Returns
    /// `(basename, global_prefix + basename)`.
    pub fn uniquename_bare(
        &mut self,
        basename: &str,
        with_number: Option<bool>,
        lenmax: usize,
    ) -> (String, String) {
        // Upstream `basename = basename[:lenmax].translate(C_IDENTIFIER)`.
        // Python 2 strings slice by byte, so keep this byte-shaped
        // instead of taking Rust `char`s.
        let mut basename: String = basename
            .bytes()
            .take(lenmax)
            .map(c_identifier_byte)
            .map(char::from)
            .collect();

        let n = *self.seennames.get(&basename).unwrap_or(&0);
        self.seennames.insert(basename.clone(), n + 1);

        // `with_number=None` defaults to True for the canonical
        // generated-variable prefixes.
        let with_number = with_number.unwrap_or_else(|| basename == "v" || basename == "w_");

        // `fmt = '%%s%s%%d' % self.number_sep` — pick the format based
        // on whether the basename's last character is a digit. Upstream
        // `:59 if with_number and not basename[-1].isdigit()` IndexErrors
        // on empty basename when `with_number` fires; mirror by
        // panicking instead of silently treating the empty case as
        // "not digit".
        let last_is_digit = if with_number {
            let last = basename.chars().last().expect(
                "uniquename: empty basename with with_number=true (upstream basename[-1] IndexErrors)",
            );
            last.is_ascii_digit()
        } else {
            false
        };
        let use_plain_fmt = with_number && !last_is_digit;

        if n != 0 || with_number {
            let candidate = if use_plain_fmt {
                format!("{basename}{n}")
            } else {
                format!("{basename}{}{n}", self.number_sep)
            };
            basename = self.ensure_unique(candidate);
        }

        let prefixed = format!("{}{}", self.global_prefix, basename);
        (basename, prefixed)
    }

    /// RPython `NameManager.localScope(self, parent=None)`
    /// (`gensupp.py`).
    ///
    /// Upstream's `_LocalScope.__init__` stores `self.glob = glob`
    /// (`gensupp.py`), so subsequent `_LocalScope.uniquename`
    /// / `localname` calls implicitly mutate the parent NameManager.
    /// To match that shape, this constructor takes an
    /// `Rc<RefCell<NameManager>>` shared with the caller — the
    /// returned [`LocalScope`] keeps an `Rc::clone` so its methods
    /// can `borrow_mut()` the same allocation without a per-call
    /// parameter. See [`LocalScope.glob`] for the
    /// TODO block on why upstream's bare
    /// `self.glob = glob` lowers to an `Rc<RefCell<>>` here.
    ///
    /// Grows `scopelist` so the new scope's per-basename dedup map
    /// is in place before the scope is used.
    #[cfg(test)]
    pub(crate) fn local_scope_from(
        rc: &Rc<RefCell<Self>>,
        parent: Option<&LocalScope>,
    ) -> LocalScope {
        // Upstream `:69 ret = _LocalScope(self, parent)` then
        // `:70-71 while ret.scope >= len(self.scopelist):
        // self.scopelist.append({})`.
        let parent_scope = parent.map_or_else(|| rc.borrow().scope, |p| p.scope);
        let scope = parent_scope + 1;
        {
            let mut nm = rc.borrow_mut();
            while scope >= nm.scopelist.len() {
                nm.scopelist.push(HashMap::new());
            }
        }
        LocalScope {
            mapping: HashMap::new(),
            usednames: HashMap::new(),
            scope,
            parent: parent.cloned().map(Rc::new),
            glob: Rc::clone(rc),
        }
    }

    /// RPython `NameManager.localScope` CamelCase alias
    /// (`gensupp.py`). Both spellings forward to the same
    /// constructor so callers that grep upstream by method name
    /// land on a hit. Per AGENTS.md §4 ("Removing an RPython method
    /// to 'simplify' things is not allowed"), the upstream-shape
    /// name is preserved as a wrapper even though `local_scope` is
    /// the canonical Rust spelling per CLAUDE.md §2.
    #[allow(non_snake_case)]
    #[cfg(test)]
    pub(crate) fn localScope(rc: &Rc<RefCell<Self>>, parent: Option<&LocalScope>) -> LocalScope {
        Self::local_scope_from(rc, parent)
    }
}

/// RPython `class _LocalScope(object)` (`gensupp.py`).
///
/// Tracks per-scope local-name dedup. Upstream stores `self.glob`
/// (`gensupp.py`) as a reference to the parent `NameManager`,
/// then mutates it through `self.glob.uniquename(...)` /
/// `self.glob.scopelist[...]` inside `uniquename` / `localname`.
///
/// TODO: Rust forbids storing `&mut NameManager`
/// inside the same struct that holds other state without locking
/// the whole `NameManager` for the lifetime of the `LocalScope`,
/// which would break upstream's pattern of keeping a parent
/// `_LocalScope` alive while a child `_LocalScope` is being used
/// (`localScope(parent=self)`). The minimal Rust adaptation is to
/// hold `Rc<RefCell<NameManager>>`: every `LocalScope` clones the
/// `Rc`, so distinct scopes share one `NameManager` allocation and
/// `borrow_mut()` it on demand without colliding. Upstream
/// `self.glob = glob` lowers to `self.glob = Rc::clone(glob_rc)`.
#[derive(Debug, Clone)]
#[cfg(test)]
pub(crate) struct LocalScope {
    /// `mapping`: external-name → mangled-name.
    pub(crate) mapping: HashMap<String, String>,
    /// `usednames`: per-basename use-count, like `seennames` but local.
    pub(crate) usednames: HashMap<String, usize>,
    /// `scope`: parent.scope + 1.
    pub(crate) scope: usize,
    /// RPython `_LocalScope.parent` (`gensupp.py`). Either the
    /// parent `_LocalScope` or the glob `NameManager` when no parent
    /// is supplied (upstream `:78-79 if not parent: parent = glob`).
    /// Held as `Option<Rc<LocalScope>>` — `None` mirrors upstream's
    /// "parent is the NameManager", which is already reachable via
    /// [`Self::glob`]. The field is dead upstream too (no reader
    /// outside `__init__`), but preserved for structural parity per
    /// AGENTS.md: "an RPython object attribute is preserved as a Rust struct field".
    #[allow(dead_code)]
    pub(crate) parent: Option<Rc<LocalScope>>,
    /// RPython `_LocalScope.glob` (`gensupp.py`). Shared
    /// `Rc<RefCell<NameManager>>` so subsequent `uniquename` /
    /// `localname` calls borrow the parent NameManager without a
    /// per-call parameter — see the TODO block
    /// on the struct.
    pub(crate) glob: Rc<RefCell<NameManager>>,
}

#[cfg(test)]
impl LocalScope {
    /// RPython `_LocalScope.uniquename(self, basename)`
    /// (`gensupp.py`).
    #[cfg(test)]
    pub fn uniquename(&mut self, basename: &str) -> String {
        let basename = translate_c_identifier(basename);
        let p = *self.usednames.get(&basename).unwrap_or(&0);
        self.usednames.insert(basename.clone(), p + 1);
        // Upstream `:90-92`:
        //     namesbyscope = glob.scopelist[self.scope]
        //     namelist = namesbyscope.setdefault(basename, [])
        //     if p == len(namelist):
        //         namelist.append(glob.uniquename(basename))
        //
        // We have to release each `borrow*()` before the next so
        // `glob.uniquename` (which also takes `&mut self` on the
        // borrowed NameManager) does not collide.
        let need_new = {
            let nm = self.glob.borrow();
            let namesbyscope = nm
                .scopelist
                .get(self.scope)
                .expect("LocalScope.uniquename: scope index out of range");
            namesbyscope
                .get(&basename)
                .is_none_or(|list| p == list.len())
        };
        if need_new {
            let new_name = self.glob.borrow_mut().uniquename(&basename, None, 50);
            self.glob.borrow_mut().scopelist[self.scope]
                .entry(basename.clone())
                .or_default()
                .push(new_name);
        }
        // Upstream `:94 return namelist[p]`.
        self.glob.borrow().scopelist[self.scope][&basename][p].clone()
    }

    /// RPython `_LocalScope.localname(self, name, wrapped=False)`
    /// (`gensupp.py`). Mangles a local name through
    /// [`Self::uniquename`], with the `v`/`w_`/`l_` prefix selection
    /// upstream uses to pacify a tcc parser bug.
    #[cfg(test)]
    pub fn localname(&mut self, name: &str, wrapped: bool) -> String {
        if let Some(existing) = self.mapping.get(name) {
            return existing.clone();
        }
        let scorepos = name.rfind('_');
        let basename: String = if name.starts_with('v')
            && name[1..].chars().all(|c| c.is_ascii_digit())
            && name.len() > 1
        {
            if wrapped {
                "w_".to_string()
            } else {
                "v".to_string()
            }
        } else if let Some(pos) = scorepos {
            let suffix = &name[pos + 1..];
            if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                let stem = &name[..pos];
                let prefix = if wrapped { "w_" } else { "l_" };
                format!("{prefix}{stem}")
            } else {
                name.to_string()
            }
        } else {
            name.to_string()
        };
        let ret = self.uniquename(&basename);
        self.mapping.insert(name.to_string(), ret.clone());
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_identifier_passes_alnum_and_underscores_others() {
        assert_eq!(translate_c_identifier("abc-9.X"), "abc_9_X");
    }

    #[test]
    fn make_reserved_names_marks_each_token() {
        let mut nm = NameManager::new("", "_");
        nm.make_reserved_names("auto break case");
        assert!(nm.seennames.contains_key("auto"));
        assert!(nm.seennames.contains_key("break"));
        assert!(nm.seennames.contains_key("case"));
    }

    #[test]
    #[should_panic(expected = "already been seen")]
    fn make_reserved_names_panics_on_duplicate() {
        let mut nm = NameManager::new("", "_");
        nm.make_reserved_names("auto");
        nm.make_reserved_names("auto");
    }

    /// First claim of a non-reserved basename returns the prefixed
    /// form unchanged; subsequent claims dedup with `_<n>`.
    #[test]
    fn uniquename_first_claim_returns_prefixed_basename() {
        let mut nm = NameManager::new("pypy_", "_");
        assert_eq!(nm.uniquename("foo", Some(false), 50), "pypy_foo");
        assert_eq!(nm.uniquename("foo", Some(false), 50), "pypy_foo_1");
        assert_eq!(nm.uniquename("foo", Some(false), 50), "pypy_foo_2");
    }

    /// `with_number=None` defaults to True for `v` / `w_`. The
    /// dedup format drops the `number_sep` because the basename
    /// already ends in a non-digit char (`v`).
    #[test]
    fn uniquename_v_basename_appends_digit_without_separator() {
        let mut nm = NameManager::new("", "_");
        assert_eq!(nm.uniquename("v", None, 50), "v0");
        assert_eq!(nm.uniquename("v", None, 50), "v1");
    }

    /// Reserved names do not get returned (they collide with reserved
    /// → ensure_unique walks to `<reserved>_<n>` until clean).
    #[test]
    fn uniquename_avoids_reserved_names() {
        let mut nm = NameManager::new("", "_");
        nm.make_reserved_names("auto");
        // First claim of `auto` sees seennames[auto] == 1; with
        // with_number=Some(false) and last char non-digit, dedup
        // format is `{base}_{n}` → "auto_1".
        assert_eq!(nm.uniquename("auto", Some(false), 50), "auto_1");
    }

    /// `bare=true` returns both the bare and prefixed forms.
    #[test]
    fn uniquename_bare_returns_both_forms() {
        let mut nm = NameManager::new("pypy_", "_");
        let (bare, prefixed) = nm.uniquename_bare("foo", Some(false), 50);
        assert_eq!(bare, "foo");
        assert_eq!(prefixed, "pypy_foo");
    }

    /// `lenmax` truncates the input *before* C-identifier translation.
    #[test]
    fn uniquename_truncates_to_lenmax() {
        let mut nm = NameManager::new("", "_");
        let long = "abcdefghij".repeat(20);
        let out = nm.uniquename(&long, Some(false), 5);
        assert_eq!(out, "abcde");
    }

    /// Non-identifier characters fold to `_` per the C_IDENTIFIER
    /// translation table.
    #[test]
    fn uniquename_translates_non_identifier_chars_to_underscore() {
        let mut nm = NameManager::new("", "_");
        assert_eq!(nm.uniquename("foo-bar.baz", Some(false), 50), "foo_bar_baz");
    }

    /// `uniquemodulename` returns `<name>_<i>` and never reuses the
    /// same suffix within a process.
    #[test]
    fn uniquemodulename_returns_distinct_names_within_process() {
        let a = uniquemodulename("zztest_modname_a");
        let b = uniquemodulename("zztest_modname_a");
        assert_ne!(a, b);
        assert!(a.starts_with("zztest_modname_a_"));
        assert!(b.starts_with("zztest_modname_a_"));
    }

    /// `local_scope_from` extends `scopelist` so the indexed slot is
    /// in place before `LocalScope::uniquename` reads it.
    #[test]
    fn local_scope_extends_scopelist() {
        let nm = Rc::new(RefCell::new(NameManager::new("pypy_", "_")));
        let scope = NameManager::local_scope_from(&nm, None);
        assert_eq!(scope.scope, 1);
        assert!(nm.borrow().scopelist.len() >= 2);
    }

    /// `LocalScope::uniquename` claims a global name and caches it for
    /// successive same-`p` lookups. Distinct `p` values force a new
    /// global claim.
    #[test]
    fn local_scope_uniquename_caches_per_position() {
        let nm = Rc::new(RefCell::new(NameManager::new("pypy_", "_")));
        let mut scope = NameManager::local_scope_from(&nm, None);
        let child = NameManager::local_scope_from(&nm, Some(&scope));
        assert_eq!(child.scope, scope.scope + 1);
        let first = scope.uniquename("foo");
        let second = scope.uniquename("foo");
        // `p` advanced past 0 → distinct name.
        assert_ne!(first, second);
    }

    /// `LocalScope::localname` rewrites `v<digits>` to `v` (or `w_` when
    /// wrapped) before calling `uniquename`.
    #[test]
    fn local_scope_localname_rewrites_v_basename() {
        let nm = Rc::new(RefCell::new(NameManager::new("", "_")));
        let mut scope = NameManager::local_scope_from(&nm, None);
        let out = scope.localname("v42", false);
        assert!(out.starts_with('v'));
        // Stable across repeated lookups for the same external name.
        assert_eq!(scope.localname("v42", false), out);
    }

    /// `LocalScope::localname` of `name_<digits>` collapses to
    /// `l_<stem>` (`w_<stem>` when wrapped).
    #[test]
    fn local_scope_localname_prefixes_named_stem() {
        let nm = Rc::new(RefCell::new(NameManager::new("", "_")));
        let mut scope = NameManager::local_scope_from(&nm, None);
        let out = scope.localname("foo_3", false);
        assert!(out.starts_with("l_foo"));
    }

    /// `localScope` (CamelCase wrapper) and `local_scope_from`
    /// produce structurally equivalent scopes — the alias exists so
    /// callers grepping upstream by name find a hit.
    #[test]
    fn local_scope_camelcase_wrapper_matches_snake_case() {
        let nm = Rc::new(RefCell::new(NameManager::new("", "_")));
        let snake = NameManager::local_scope_from(&nm, None);
        let camel = NameManager::localScope(&nm, None);
        assert_eq!(snake.scope, camel.scope);
    }
}
