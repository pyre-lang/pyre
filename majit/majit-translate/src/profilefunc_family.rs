//! Published `ProfileFunc` family for one translation invocation.
//!
//! Upstream `executioncontext.py` `setllprofile` stores a closed set of
//! RPython functions (`app_profile_call`, `lsprof_call`) into
//! `self.profilefunc`; `_trace` calls that slot and
//! `description.py` `CallFamily` / `rtyper/rpbc.py` `FunctionsPBCRepr`
//! rtype it as a nullable PBC. The translator harvests the same family
//! from LLBC and publishes it here so field annotation and stored
//! `FnDef` constants move together.
//!
//! Thread-local, not a process-global `RwLock`: a translate pipeline
//! runs start-to-finish on one thread, and `cargo test` runs pipelines
//! in parallel — the same TLS adaptation as [`crate::local_crates`].

use std::cell::RefCell;

struct Published {
    graphs: Vec<String>,
    spellings: Vec<String>,
}

thread_local! {
    static PUBLISHED: RefCell<Published> = const {
        RefCell::new(Published {
            graphs: Vec::new(),
            spellings: Vec::new(),
        })
    };
}

/// Replace this thread's published `ProfileFunc` family with one pipeline
/// invocation's harvested paths and type spellings.
pub(crate) fn register_profilefunc_family(graphs: Vec<String>, spellings: Vec<String>) {
    PUBLISHED.with(|published| {
        let mut published = published.borrow_mut();
        published.graphs = graphs;
        published.spellings = spellings;
    });
}

/// Crate-stripped call paths of the published `ProfileFunc` family.
pub(crate) fn profilefunc_graphs() -> Vec<String> {
    PUBLISHED.with(|published| published.borrow().graphs.clone())
}

/// Type spellings of the published `ProfileFunc` slot.
pub(crate) fn profilefunc_spellings() -> Vec<String> {
    PUBLISHED.with(|published| published.borrow().spellings.clone())
}

/// Whether `segments` name a published family member (`__fn_const` tail
/// or a registry key).
pub(crate) fn is_profilefunc_graph_path(segments: &[String]) -> bool {
    if segments.is_empty() {
        return false;
    }
    let key = segments.join("::");
    PUBLISHED.with(|published| {
        published
            .borrow()
            .graphs
            .iter()
            .any(|path| path_keys_match(path, &key))
    })
}

fn path_keys_match(published: &str, key: &str) -> bool {
    published == key
        || published.ends_with(&format!("::{key}"))
        || key.ends_with(&format!("::{published}"))
}

/// Whether `spelling` names the `ProfileFunc` slot or its `Option`.
///
/// Matches the alias leaf (`ProfileFunc` / `Option<ProfileFunc>`) and
/// any published expanded `fn(...)` spelling (`tyref_to_ast_string`
/// peels `TypeDeclKind::Alias`).
pub(crate) fn spelling_is_profilefunc(spelling: &str) -> bool {
    if leaf_is_profilefunc(spelling) {
        return true;
    }
    let inner = peel_option(spelling);
    if inner != spelling && leaf_is_profilefunc(inner) {
        return true;
    }
    PUBLISHED.with(|published| {
        published.borrow().spellings.iter().any(|known| {
            known == spelling
                || known == inner
                || peel_option(known) == inner
                || peel_option(known) == spelling
        })
    })
}

fn peel_option(spelling: &str) -> &str {
    let t = spelling.trim();
    t.strip_prefix("Option<")
        .and_then(|s| s.strip_suffix('>'))
        .map(str::trim)
        .unwrap_or(t)
}

fn leaf_is_profilefunc(spelling: &str) -> bool {
    let t = peel_option(spelling);
    t == "ProfileFunc" || t.ends_with("::ProfileFunc")
}
