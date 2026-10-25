//! Local-crate and harvested foreign-body alias roots for symbolic `CallPath`
//! resolution.
//!
//! RPython needs nothing like this: `Bookkeeper.getdesc`
//! (`bookkeeper.py`) keys `FunctionDesc`s by function-object
//! identity, so a callable has one identity regardless of the import
//! spelling. pyre resolves symbolic paths extracted from LLBC, where a
//! cross-crate callsite spells the callee with its crate name
//! (`somecrate::io::output_flush`) while the graph registers under
//! module-relative spellings — so every *local* (LLBC-extracted) crate
//! name must be an alias root on both the registration side
//! (`free_function_alias_paths`) and the canonical-dedup side
//! (`populate_call_registry_from_call_graphs`).
//!
//! A foreign FunDecl whose body is present is the same callable:
//! `FunctionDesc.getuniquegraph` / `cachedgraph` (`description.py`) →
//! `TranslationContext.buildflowgraph` (`translator.py`). Harvest that
//! FunDecl so a `core::num::<Impl>::div_ceil` callsite hits the graph
//! registered under `num::<Impl>::div_ceil`. An Opaque FunDecl is
//! `extfunc.py` / `lloperation.py` and keeps its crate-qualified spelling
//! (`CallControl.guess_call_kind` / `Bookkeeper.getdesc`). Admission is
//! per callable: harvesting one `core` body does not strip every
//! `core::m::f`.
//!
//! Seeded from the loaded LLBC set's `crate_name()`s by
//! `build_semantic_program_via_active_frontend`. Consumers that construct
//! programs without the active frontend must register their own crate roots.

use std::cell::RefCell;

struct CrateAliasRoots {
    local: Vec<String>,
}

impl CrateAliasRoots {
    const fn new() -> Self {
        Self { local: Vec::new() }
    }
}

thread_local! {
    /// Per-pipeline-invocation crate alias roots, seeded once at the
    /// top of `build_semantic_program_via_active_frontend` and read back at
    /// the alias / dedup / tie-break sites during the SAME invocation.
    ///
    /// Thread-local, not a process-global `RwLock`: a translate pipeline
    /// runs start-to-finish on one thread (there is no `par_iter` inside it),
    /// and `generated::all_jitcodes` already scopes the whole per-thread
    /// pipeline registry with a `thread_local!` `OnceCell` to preserve
    /// RPython's single-thread annotator invariant. A shared `RwLock` let a
    /// second pipeline on another thread (parallel `cargo test`, or any
    /// future parallel translate) overwrite this run's roots between its own
    /// seed and read, flaking alias resolution. The roots belong to one
    /// invocation, so scoping them to the invocation's thread is exact — the
    /// same TLS adaptation of a PyPy GIL-singleton as
    /// `generated::all_jitcodes`.
    static REGISTERED: RefCell<CrateAliasRoots> = const { RefCell::new(CrateAliasRoots::new()) };
}

/// Replace this thread's registered local-crate set with one pipeline
/// invocation's LLBC crate names. A later invocation on the same thread
/// overwrites (per-invocation semantics, like the
/// `STRUCT_ORIGIN_REGISTRY` re-seed). Harvested foreign-body tables live
/// on the translation's `StructFieldRegistry`.
pub(crate) fn register_local_crate_roots(names: impl IntoIterator<Item = String>) {
    REGISTERED.with(|registered| {
        registered.borrow_mut().local = names.into_iter().collect();
    });
}

fn set_local_crate_roots(names: Vec<String>) {
    REGISTERED.with(|registered| {
        registered.borrow_mut().local = names;
    });
}

/// Registered local crate names for the current pipeline invocation.
pub(crate) fn local_crate_roots() -> Vec<String> {
    REGISTERED.with(|registered| registered.borrow().local.clone())
}

pub(crate) fn is_local_crate_root(seg: &str) -> bool {
    REGISTERED.with(|registered| registered.borrow().local.iter().any(|r| r == seg))
}

/// True when `seg` is a crate prefix that every path under it may strip:
/// `crate` or a local crate. A harvested foreign crate is not an alias
/// root; only a FunDecl that lowered a body strips (`Bookkeeper.getdesc`).
pub(crate) fn is_crate_alias_root(seg: &str) -> bool {
    seg == "crate" || is_local_crate_root(seg)
}

/// Strip a leading `crate` / local crate. `None` when the path must keep
/// its spelling. Harvested foreign callables use
/// [`strip_alias_crate_prefix_with`].
pub(crate) fn strip_alias_crate_prefix(segments: &[String]) -> Option<Vec<String>> {
    strip_alias_crate_prefix_with(segments, &[])
}

/// Strip a leading `crate` / local crate, or a harvested foreign callable's
/// crate root from `callables`. `None` when the path must keep its spelling.
pub(crate) fn strip_alias_crate_prefix_with(
    segments: &[String],
    callables: &[(String, String)],
) -> Option<Vec<String>> {
    let first = segments.first()?;
    if is_crate_alias_root(first) {
        return Some(segments[1..].to_vec());
    }
    if !callables.is_empty() && segments.len() > 1 {
        let remainder = segments[1..].join("::");
        if callables
            .iter()
            .any(|(root, path)| root == first && path == &remainder)
        {
            return Some(segments[1..].to_vec());
        }
    }
    None
}

/// Run one stand-alone, single-LLBC lowering with that artefact's crate root
/// present in the same invocation-local alias set used by the whole-program
/// frontend.
///
/// `build_semantic_program_via_active_frontend` seeds all loaded crates once.
/// The public `lower_fun_decl*` entry points bypass that driver, so without a
/// scoped seed their transparent constructors retain the defining crate while
/// the layout metadata derived from the same LLBC uses crate-relative names.
/// Preserve any surrounding multi-crate invocation and restore it even when
/// lowering unwinds. Harvested foreign-body tables live on the registry.
pub(crate) fn with_local_crate_root<R>(root: &str, f: impl FnOnce() -> R) -> R {
    struct Restore(Vec<String>);

    impl Drop for Restore {
        fn drop(&mut self) {
            set_local_crate_roots(std::mem::take(&mut self.0));
        }
    }

    let previous = local_crate_roots();
    if previous.iter().any(|registered| registered == root) {
        return f();
    }
    let mut scoped = previous.clone();
    scoped.push(root.to_string());
    set_local_crate_roots(scoped);
    let _restore = Restore(previous);
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harvested_foreign_callable_strips_only_that_path() {
        assert!(!is_crate_alias_root("core"));
        assert!(!is_local_crate_root("core"));
        assert!(is_crate_alias_root("crate"));
        let callables = vec![("core".to_string(), "num::<Impl>::div_ceil".to_string())];
        let stripped = strip_alias_crate_prefix_with(
            &[
                "core".to_string(),
                "num".to_string(),
                "<Impl>".to_string(),
                "div_ceil".to_string(),
            ],
            &callables,
        );
        assert_eq!(
            stripped,
            Some(vec![
                "num".to_string(),
                "<Impl>".to_string(),
                "div_ceil".to_string()
            ])
        );
        assert_eq!(
            strip_alias_crate_prefix_with(
                &[
                    "core".to_string(),
                    "mem".to_string(),
                    "transmute".to_string()
                ],
                &callables
            ),
            None
        );
        assert_eq!(
            strip_alias_crate_prefix(&[
                "core".to_string(),
                "num".to_string(),
                "<Impl>".to_string(),
                "div_ceil".to_string(),
            ]),
            None
        );
    }

    #[test]
    fn with_local_crate_root_does_not_make_foreign_crate_an_alias() {
        with_local_crate_root("pyre_interpreter", || {
            assert!(is_local_crate_root("pyre_interpreter"));
            assert!(!is_crate_alias_root("core"));
        });
    }
}
